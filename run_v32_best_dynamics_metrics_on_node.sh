#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_REPO="$(cd "$REPO/.." && pwd)"
cd "$REPO"

if [[ -f "$PARENT_REPO/nanochat/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$PARENT_REPO/nanochat/.venv/bin/activate"
elif [[ -f "$REPO/nanochat/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$REPO/nanochat/.venv/bin/activate"
elif [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "[error] no nanochat venv found; run scripts/setup_env.sh first" >&2
  exit 1
fi

export PYTHONPATH="$REPO:$REPO/nanochat"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NANOCHAT_FORCE_MATH_SDPA="${NANOCHAT_FORCE_MATH_SDPA:-1}"

STAMP="${STAMP:-20260501}"
OUT_ROOT="${OUT_ROOT:-$PARENT_REPO/search_evals/v32_best_dynamics_metrics_${STAMP}}"
LOG_ROOT="${LOG_ROOT:-$PARENT_REPO/logs/v32_best_dynamics_metrics_${STAMP}}"
CASE_FILE="${CASE_FILE:-$OUT_ROOT/selected_cases.tsv}"
TOKENS="${TOKENS:-1073741824}"
NPROC="${NPROC:-8}"
SEQ="${SEQ:-1024}"
DEPTH="${DEPTH:-8}"
MAX_DEVICE_BATCH_SIZE="${MAX_DEVICE_BATCH_SIZE:-16}"
SEED="${SEED:-42}"

METRICS_EVERY="${METRICS_EVERY:-1}"
METRICS_TOP_K="${METRICS_TOP_K:-4}"
METRICS_MAX_MODULES="${METRICS_MAX_MODULES:-16}"
METRICS_HESSIAN_EVERY="${METRICS_HESSIAN_EVERY:-100}"
METRICS_HESSIAN_ITERS="${METRICS_HESSIAN_ITERS:-2}"
METRICS_HESSIAN_MAX_MODULES="${METRICS_HESSIAN_MAX_MODULES:-4}"

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

echo "v32 best-row dynamics metrics"
echo "repo=$REPO"
echo "out_root=$OUT_ROOT"
echo "tokens=$TOKENS seed=$SEED"

python - <<'PY' > "$CASE_FILE"
import json
import re
from pathlib import Path

parent = Path("..").resolve()
roots = {
    262144: parent / "search_evals/v29_topaware_k1_256k_alpha_sweep_20260430",
    524288: parent / "search_evals/v31_topaware_k1_512k_alpha_sweep_20260501",
}
patterns = {
    "native_muon": re.compile(r"^native_muon_bsz(?P<batch>\d+)_lr(?P<lr>[0-9p]+)_s42$"),
    "streaming_identity": re.compile(r"^streaming_identity_bsz(?P<batch>\d+)_lr(?P<lr>[0-9p]+)_s42$"),
    "top_aware_muon": re.compile(r"^top_aware_k1_a(?P<alpha>[0-9p]+)_bsz(?P<batch>\d+)_lr(?P<lr>[0-9p]+)_s42$"),
}

def unslug(x: str) -> float:
    return float(x.replace("p", "."))

rows = []
for batch, root in roots.items():
    if not root.exists():
        raise SystemExit(f"missing result root: {root}")
    for method, pat in patterns.items():
        best = None
        for path in root.glob("*/result.json"):
            m = pat.match(path.parent.name)
            if not m or int(m.group("batch")) != batch:
                continue
            data = json.loads(path.read_text())
            score = data.get("score")
            if score is None or data.get("error") is not None:
                continue
            alpha = unslug(m.group("alpha")) if "alpha" in m.groupdict() and m.group("alpha") else "-"
            lr = unslug(m.group("lr"))
            item = (float(score), method, batch, lr, alpha, str(path))
            if best is None or item[0] < best[0]:
                best = item
        if best is None:
            raise SystemExit(f"no valid best row for batch={batch} method={method}")
        rows.append(best)

print("score\tmethod\tbatch\tlr\talpha\tsource_result")
for score, method, batch, lr, alpha, source in rows:
    print(f"{score:.12g}\t{method}\t{batch}\t{lr:g}\t{alpha}\t{source}")
PY

cat "$CASE_FILE"

slug_float() {
  local value="$1"
  value="${value/-/m}"
  value="${value/./p}"
  echo "$value"
}

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

valid_result() {
  local out="$1"
  [[ -f "$out" ]] || return 1
  python - "$out" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
raise SystemExit(0 if data.get("score") is not None and data.get("error") is None else 1)
PY
}

run_case() {
  local method="$1"
  local batch="$2"
  local lr="$3"
  local alpha="$4"
  local score="$5"
  local lr_slug
  lr_slug="$(slug_float "$lr")"
  local case="metrics_${method}_bsz${batch}_lr${lr_slug}_s${SEED}"
  if [[ "$method" == "top_aware_muon" ]]; then
    local alpha_slug
    alpha_slug="$(slug_float "$alpha")"
    case="metrics_top_aware_k1_a${alpha_slug}_bsz${batch}_lr${lr_slug}_s${SEED}"
  fi

  local out="$OUT_ROOT/$case/result.json"
  local log="$LOG_ROOT/$case.log"
  local device_batch
  device_batch="$(device_batch_for "$batch")"
  local steps=$(( TOKENS / batch ))
  local warmup=$(( (steps + 19) / 20 ))
  local eval_every=$(( steps / 8 ))
  if (( eval_every < 1 )); then eval_every=1; fi

  if valid_result "$out"; then
    echo "[skip] $case already complete"
    return
  fi

  mkdir -p "$(dirname "$out")" "$LOG_ROOT"
  echo
  echo "=== $case ==="
  echo "selected_source_score=$score steps=$steps device_batch=$device_batch eval_every=$eval_every"

  local common=(
    --nanochat-dir nanochat
    --output-file "$out"
    --depth "$DEPTH"
    --max-seq-len "$SEQ"
    --device-batch-size "$device_batch"
    --total-batch-size "$batch"
    --max-steps "$steps"
    --matrix-lr "$lr"
    --warmup-steps "$warmup"
    --warmdown-ratio 0.65
    --final-lr-frac 0.05
    --eval-every "$eval_every"
    --eval-tokens 524288
    --seed "$SEED"
    --metrics-every "$METRICS_EVERY"
    --metrics-top-k "$METRICS_TOP_K"
    --metrics-module-regex 'transformer\.h'
    --metrics-max-modules "$METRICS_MAX_MODULES"
    --metrics-split-momentum
    --metrics-alignment-side lite
    --metrics-hessian-every "$METRICS_HESSIAN_EVERY"
    --metrics-hessian-top-k 1
    --metrics-hessian-iters "$METRICS_HESSIAN_ITERS"
    --metrics-hessian-max-modules "$METRICS_HESSIAN_MAX_MODULES"
  )

  set +e
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
      --candidate-param top_k=1 \
      --candidate-param "alpha=$alpha" \
      "${common[@]}" \
      --k -1 --num-iters 2 --pure-qr --fallback-ortho-tol 0.01 2>&1 | tee "$log"
  else
    echo "unknown method: $method" >&2
    return 2
  fi
  local rc=$?
  set -e
  if (( rc != 0 )); then
    echo "[warn] $case exited with rc=$rc; continuing to next metrics case"
  fi
}

tail -n +2 "$CASE_FILE" | while IFS=$'\t' read -r score method batch lr alpha source; do
  run_case "$method" "$batch" "$lr" "$alpha" "$score"
done

echo "v32 dynamics metrics queue complete"
