#!/usr/bin/env bash
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

export OMP_NUM_THREADS=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-sigma}"

python run_v26_small_bsz_streaming_fair.py "$@"
