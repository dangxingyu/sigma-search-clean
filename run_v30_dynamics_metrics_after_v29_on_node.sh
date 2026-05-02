#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_REPO="$(cd "$REPO/.." && pwd)"
cd "$REPO"

export PYTHONPATH="$REPO:$REPO/nanochat"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

if [[ -f "$PARENT_REPO/nanochat/.venv/bin/activate" ]]; then
  # Prefer the live project venv on this cluster; the standalone nanochat tree
  # is source-only in the handoff bundle.
  # shellcheck disable=SC1091
  source "$PARENT_REPO/nanochat/.venv/bin/activate"
elif [[ -d "$REPO/nanochat/.venv" ]]; then
  # shellcheck disable=SC1091
  source "$REPO/nanochat/.venv/bin/activate"
fi

STAMP="${STAMP:-20260501}"
OUT_ROOT="${OUT_ROOT:-search_evals/v30_dynamics_metrics_${STAMP}}"
LOG_ROOT="${LOG_ROOT:-logs/v30_dynamics_metrics_${STAMP}}"
TOKENS="${TOKENS:-1073741824}"
NPROC="${NPROC:-8}"
SEQ="${SEQ:-1024}"
DEPTH="${DEPTH:-8}"
MAX_DEVICE_BATCH_SIZE="${MAX_DEVICE_BATCH_SIZE:-16}"
SEED="${SEED:-42}"
LR="${LR:-0.01}"
TOP_ALPHA="${TOP_ALPHA:-0.75}"
TOP_K="${TOP_K:-1}"

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

echo "v30 dynamics metrics queue"
echo "repo=$REPO"
echo "out_root=$OUT_ROOT"
echo "tokens=$TOKENS lr=$LR top_k=$TOP_K alpha=$TOP_ALPHA"

while pgrep -af "run_v28_v29_topaware_k1_alpha_sweeps_on_node.sh|v29_topaware_k1_256k_alpha_sweep_20260430" >/dev/null; do
  echo "$(date) waiting for v29 top-aware sweep to finish..."
  sleep 120
done

while read -r pid cmd; do
  [[ -z "${pid:-}" ]] && continue
  if [[ "$pid" != "$$" ]]; then
    echo "keeping v26b supervisor stopped pid=$pid: $cmd"
    kill -STOP "$pid" || true
  fi
done < <(pgrep -af "run_v26b_large_then_maybe_v27_on_node.sh" || true)

device_batch_for() {
  local batch="$1"
  local max_no_accum=$(( batch / (SEQ * NPROC) ))
  if (( max_no_accum < 1 )); then
    echo "batch too small: $batch" >&2
    exit 1
  fi
  if (( max_no_accum < MAX_DEVICE_BATCH_SIZE )); then
    echo "$max_no_accum"
  else
    echo "$MAX_DEVICE_BATCH_SIZE"
  fi
}

run_case() {
  local method="$1"
  local batch="$2"
  local case="$method"_bsz"$batch"_lr"${LR/./p}"_s"$SEED"
  if [[ "$method" == "top_aware_muon" ]]; then
    case="top_aware_k${TOP_K}_a${TOP_ALPHA/./p}_bsz${batch}_lr${LR/./p}_s${SEED}"
  fi

  local out="$OUT_ROOT/$case/result.json"
  local log="$LOG_ROOT/$case.log"
  local device_batch
  device_batch="$(device_batch_for "$batch")"
  local steps=$(( TOKENS / batch ))
  local warmup=$(( (steps + 19) / 20 ))
  local eval_every=$(( steps / 8 ))
  if (( eval_every < 1 )); then eval_every=1; fi

  if [[ -f "$out" ]] && python - "$out" <<'PY'
import json, sys
data=json.load(open(sys.argv[1]))
raise SystemExit(0 if data.get("score") is not None and data.get("error") is None else 1)
PY
  then
    echo "[skip] $case already complete"
    return
  fi

  mkdir -p "$(dirname "$out")"
  echo
  echo "=== $case ==="
  echo "steps=$steps device_batch=$device_batch eval_every=$eval_every"

  local common=(
    --nanochat-dir nanochat
    --output-file "$out"
    --depth "$DEPTH"
    --max-seq-len "$SEQ"
    --device-batch-size "$device_batch"
    --total-batch-size "$batch"
    --max-steps "$steps"
    --matrix-lr "$LR"
    --warmup-steps "$warmup"
    --warmdown-ratio 0.65
    --final-lr-frac 0.05
    --eval-every "$eval_every"
    --eval-tokens 524288
    --seed "$SEED"
    --metrics-every 1
    --metrics-top-k 4
    --metrics-module-regex 'transformer\.h'
    --metrics-max-modules 8
    --metrics-split-momentum
    --metrics-alignment-side lite
    --metrics-hessian-every 100
    --metrics-hessian-top-k 1
    --metrics-hessian-iters 2
    --metrics-hessian-max-modules 8
  )

  if [[ "$method" == "native_muon" ]]; then
    torchrun --standalone --nproc_per_node="$NPROC" run_native_muon_v9.py "${common[@]}" --ns-steps 5 2>&1 | tee "$log"
  elif [[ "$method" == "streaming_identity" ]]; then
    torchrun --standalone --nproc_per_node="$NPROC" run_eval.py \
      --candidate-file candidates/identity.py \
      "${common[@]}" \
      --k -1 --num-iters 2 --pure-qr --fallback-ortho-tol 0.01 2>&1 | tee "$log"
  elif [[ "$method" == "top_aware_muon" ]]; then
    torchrun --standalone --nproc_per_node="$NPROC" run_eval.py \
      --candidate-file candidates/top_aware_muon.py \
      --candidate-param "top_k=$TOP_K" \
      --candidate-param "alpha=$TOP_ALPHA" \
      "${common[@]}" \
      --k -1 --num-iters 2 --pure-qr --fallback-ortho-tol 0.01 2>&1 | tee "$log"
  else
    echo "unknown method: $method" >&2
    exit 1
  fi
}

for batch in 131072 262144 524288 1048576 2097152; do
  run_case native_muon "$batch"
  run_case streaming_identity "$batch"
  run_case top_aware_muon "$batch"
done

MPLCONFIGDIR=/tmp/matplotlib-sigma python make_v30_dynamics_dashboard.py --root "$OUT_ROOT"
