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
OUT_ROOT="${OUT_ROOT:-$PARENT_REPO/search_evals/v33_topaware_k1_128k_alpha125_${STAMP}}"
LOG_ROOT="${LOG_ROOT:-$PARENT_REPO/logs/v33_topaware_k1_128k_alpha125_${STAMP}}"
TOKENS="${TOKENS:-1073741824}"
DEPTH="${DEPTH:-8}"
NPROC="${NPROC:-8}"
LOCK_FILE="${LOCK_FILE:-$PARENT_REPO/logs/v33_topaware_k1_128k_alpha125_${STAMP}.lock}"

if [[ -e "$LOCK_FILE" ]]; then
  echo "[skip] lock exists: $LOCK_FILE" >&2
  exit 0
fi

python run_top_aware_muon_sweep.py \
  --out-root "$OUT_ROOT" \
  --log-root "$LOG_ROOT" \
  --nanochat-dir nanochat \
  --methods "top_aware_muon" \
  --batches "131072" \
  --alphas "1.25" \
  --lrs "0.005 0.01 0.02 0.04" \
  --seeds "42" \
  --tokens "$TOKENS" \
  --depth "$DEPTH" \
  --nproc-per-node "$NPROC" \
  --max-device-batch-size 16 \
  --pure-qr \
  --streaming-num-iters 2 \
  --fallback-ortho-tol 0.01
