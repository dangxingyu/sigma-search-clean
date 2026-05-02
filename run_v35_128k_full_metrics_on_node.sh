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
OUT_ROOT="${OUT_ROOT:-$PARENT_REPO/search_evals/v35_full_metrics_128k_${STAMP}}"
LOG_ROOT="${LOG_ROOT:-$PARENT_REPO/logs/v35_full_metrics_128k_${STAMP}}"
NPROC="${NPROC:-8}"
SEED="${SEED:-42}"
DEPTH="${DEPTH:-8}"
SEQ="${SEQ:-1024}"
BATCH="${BATCH:-131072}"
DEVICE_BATCH="${DEVICE_BATCH:-8}"
STEPS="${STEPS:-8192}"
LR="${LR:-0.01}"
WARMUP="${WARMUP:-410}"
EVAL_EVERY="${EVAL_EVERY:-1024}"

METRICS_EVERY="${METRICS_EVERY:-1}"
METRICS_TOP_K="${METRICS_TOP_K:-4}"
METRICS_MAX_MODULES="${METRICS_MAX_MODULES:-16}"
METRICS_HESSIAN_EVERY="${METRICS_HESSIAN_EVERY:-100}"
METRICS_HESSIAN_ITERS="${METRICS_HESSIAN_ITERS:-2}"
METRICS_HESSIAN_MAX_MODULES="${METRICS_HESSIAN_MAX_MODULES:-4}"

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

valid_result() {
  local out="$1"
  [[ -f "$out" ]] || return 1
  python - "$out" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
raise SystemExit(0 if data.get("score") is not None and data.get("error") is None else 1)
PY
}

common_args=(
  --nanochat-dir nanochat
  --depth "$DEPTH"
  --max-seq-len "$SEQ"
  --device-batch-size "$DEVICE_BATCH"
  --total-batch-size "$BATCH"
  --max-steps "$STEPS"
  --matrix-lr "$LR"
  --warmup-steps "$WARMUP"
  --warmdown-ratio 0.65
  --final-lr-frac 0.05
  --eval-every "$EVAL_EVERY"
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

run_streaming_identity() {
  local case="streaming_identity_bsz${BATCH}_lr0p01_s${SEED}"
  local out="$OUT_ROOT/$case/result.json"
  local log="$LOG_ROOT/$case.log"
  if valid_result "$out"; then
    echo "[skip] $case already complete"
    return
  fi
  mkdir -p "$(dirname "$out")"
  echo "=== $case ==="
  torchrun --standalone --nproc_per_node="$NPROC" run_eval.py \
    --candidate-file candidates/identity.py \
    --output-file "$out" \
    "${common_args[@]}" \
    --k -1 --num-iters 2 --pure-qr --fallback-ortho-tol 0.01 2>&1 | tee "$log"
}

run_topaware_alpha125() {
  local case="topaware_k1_a1p25_bsz${BATCH}_lr0p01_s${SEED}"
  local out="$OUT_ROOT/$case/result.json"
  local log="$LOG_ROOT/$case.log"
  if valid_result "$out"; then
    echo "[skip] $case already complete"
    return
  fi
  mkdir -p "$(dirname "$out")"
  echo "=== $case ==="
  torchrun --standalone --nproc_per_node="$NPROC" run_eval.py \
    --candidate-file candidates/top_aware_muon.py \
    --candidate-param top_k=1 \
    --candidate-param alpha=1.25 \
    --output-file "$out" \
    "${common_args[@]}" \
    --k -1 --num-iters 2 --pure-qr --fallback-ortho-tol 0.01 2>&1 | tee "$log"
}

run_native_muon() {
  local case="native_muon_bsz${BATCH}_lr0p01_s${SEED}"
  local out="$OUT_ROOT/$case/result.json"
  local log="$LOG_ROOT/$case.log"
  if valid_result "$out"; then
    echo "[skip] $case already complete"
    return
  fi
  mkdir -p "$(dirname "$out")"
  echo "=== $case ==="
  torchrun --standalone --nproc_per_node="$NPROC" run_native_muon_v9.py \
    --output-file "$out" \
    "${common_args[@]}" \
    --ns-steps 5 2>&1 | tee "$log"
}

echo "v35 full 128K metrics queue"
echo "out_root=$OUT_ROOT"
echo "steps=$STEPS batch=$BATCH device_batch=$DEVICE_BATCH lr=$LR"
echo "metrics_every=$METRICS_EVERY hessian_every=$METRICS_HESSIAN_EVERY force_math_sdpa=$NANOCHAT_FORCE_MATH_SDPA"

run_streaming_identity
run_topaware_alpha125
run_native_muon

echo "v35 full 128K metrics queue complete"
