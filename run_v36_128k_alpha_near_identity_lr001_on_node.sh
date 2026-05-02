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

STAMP="${STAMP:-20260501}"
OUT_ROOT="${OUT_ROOT:-$PARENT_REPO/search_evals/v36_topaware_128k_near_identity_lr001_${STAMP}}"
LOG_ROOT="${LOG_ROOT:-$PARENT_REPO/logs/v36_topaware_128k_near_identity_lr001_${STAMP}}"
NPROC="${NPROC:-8}"
SEED="${SEED:-42}"
DEPTH="${DEPTH:-8}"
SEQ="${SEQ:-1024}"
BATCH="${BATCH:-131072}"
DEVICE_BATCH="${DEVICE_BATCH:-16}"
STEPS="${STEPS:-8192}"
LR="${LR:-0.01}"
WARMUP="${WARMUP:-410}"
EVAL_EVERY="${EVAL_EVERY:-1024}"
ALPHAS="${ALPHAS:-0.85 1.15}"

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

slug_float() {
  local value="$1"
  value="${value/-/m}"
  value="${value/./p}"
  echo "$value"
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

for alpha in $ALPHAS; do
  alpha_slug="$(slug_float "$alpha")"
  lr_slug="$(slug_float "$LR")"
  case_name="top_aware_k1_a${alpha_slug}_bsz${BATCH}_lr${lr_slug}_s${SEED}"
  out="$OUT_ROOT/$case_name/result.json"
  log="$LOG_ROOT/$case_name.log"
  if valid_result "$out"; then
    echo "[skip] $case_name already complete"
    continue
  fi
  mkdir -p "$(dirname "$out")"
  echo
  echo "=== $case_name ==="
  torchrun --standalone --nproc_per_node="$NPROC" run_eval.py \
    --candidate-file candidates/top_aware_muon.py \
    --candidate-param top_k=1 \
    --candidate-param "alpha=$alpha" \
    --nanochat-dir nanochat \
    --output-file "$out" \
    --depth "$DEPTH" \
    --max-seq-len "$SEQ" \
    --device-batch-size "$DEVICE_BATCH" \
    --total-batch-size "$BATCH" \
    --max-steps "$STEPS" \
    --matrix-lr "$LR" \
    --warmup-steps "$WARMUP" \
    --warmdown-ratio 0.65 \
    --final-lr-frac 0.05 \
    --eval-every "$EVAL_EVERY" \
    --eval-tokens 524288 \
    --seed "$SEED" \
    --k -1 \
    --num-iters 2 \
    --pure-qr \
    --fallback-ortho-tol 0.01 2>&1 | tee "$log"
done

echo "v36 near-identity alpha queue complete"
