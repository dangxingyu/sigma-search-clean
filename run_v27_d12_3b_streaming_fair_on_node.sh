#!/usr/bin/env bash
# Prepared scale-up launcher. Do not run until v26 satisfies the scale-up gate.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$SCRIPT_DIR/candidates" ]; then
    REPO="${REPO:-$SCRIPT_DIR}"
else
    REPO="${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
fi
cd "$REPO"
if [ -f nanochat/.venv/bin/activate ]; then
    source nanochat/.venv/bin/activate
elif [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "[error] activate nanochat/.venv or pass REPO to a checkout with nanochat/" >&2
    exit 1
fi

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-search_evals/v27_d12_3b_streaming_fair_${STAMP}}"
LOG_ROOT="${LOG_ROOT:-logs/v27_d12_3b_streaming_fair_${STAMP}}"
BATCHES="${BATCHES:-1048576 8388608}"
SEEDS="${SEEDS:-42}"
LRS="${LRS:-0.005 0.01 0.02 0.04}"
TOKENS="${TOKENS:-3221225472}"

export OMP_NUM_THREADS=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-sigma}"

python run_v26_small_bsz_streaming_fair.py \
    --out-root "$OUT_ROOT" \
    --log-root "$LOG_ROOT" \
    --depth 12 \
    --tokens "$TOKENS" \
    --batches "$BATCHES" \
    --seeds "$SEEDS" \
    --lrs "$LRS" \
    "$@"
