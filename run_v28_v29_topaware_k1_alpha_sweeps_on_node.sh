#!/usr/bin/env bash
# v28/v29: clean Top-Aware Muon k=1 alpha sweeps.
#
# v28: bsz=128K, alpha=0.75, LR sweep only for Top-Aware.
# v29: bsz=256K, alpha grid against baselines under the same clean infra.
#
# This script can be launched while v26 is still running. It pauses the waiting
# v26b supervisor, waits for the active v26 training process to finish, runs
# these sweeps, then resumes v26b. It never cancels allocation 29702470.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f nanochat/.venv/bin/activate ]; then
    source nanochat/.venv/bin/activate
elif [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "[error] activate nanochat/.venv first" >&2
    exit 1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-sigma}"

STAMP="${STAMP:-20260430}"
V26_ROOT="${V26_ROOT:-search_evals/v26_small_bsz_streaming_fair_20260430}"
V28_ROOT="${V28_ROOT:-search_evals/v28_topaware_k1_128k_alpha075_${STAMP}}"
V28_LOG_ROOT="${V28_LOG_ROOT:-logs/v28_topaware_k1_128k_alpha075_${STAMP}}"
V29_ROOT="${V29_ROOT:-search_evals/v29_topaware_k1_256k_alpha_sweep_${STAMP}}"
V29_LOG_ROOT="${V29_LOG_ROOT:-logs/v29_topaware_k1_256k_alpha_sweep_${STAMP}}"

V26B_PID="${V26B_PID:-}"
if [ -z "$V26B_PID" ]; then
    V26B_PID="$(pgrep -f "^/usr/bin/bash run_v26b_large_then_maybe_v27_on_node.sh$" | head -n 1 || true)"
fi

resume_v26b() {
    if [ -n "${V26B_PID:-}" ] && kill -0 "$V26B_PID" 2>/dev/null; then
        echo "[resume] v26b supervisor pid=$V26B_PID"
        kill -CONT "$V26B_PID" 2>/dev/null || true
    fi
}
trap resume_v26b EXIT

if [ -n "$V26B_PID" ] && kill -0 "$V26B_PID" 2>/dev/null; then
    echo "[pause] v26b supervisor pid=$V26B_PID so v28/v29 can run next"
    kill -STOP "$V26B_PID"
else
    echo "[info] no v26b supervisor found to pause"
fi

echo "[wait] waiting for active v26 small-batch controller to finish"
while pgrep -af "[p]ython run_v26_small_bsz_streaming_fair.py --out-root ${V26_ROOT}" >/dev/null 2>&1; do
    date
    sleep 120
done

echo "[run] v28: 128K Top-Aware k=1 alpha=0.75 LR sweep"
python run_top_aware_muon_sweep.py \
    --out-root "$V28_ROOT" \
    --log-root "$V28_LOG_ROOT" \
    --methods "top_aware_muon" \
    --batches "131072" \
    --alphas "0.75" \
    --lrs "0.005 0.01 0.02 0.04" \
    --seeds "42" \
    --tokens 1073741824 \
    --depth 8 \
    --nproc-per-node 8

echo "[run] v29: 256K Top-Aware k=1 alpha grid with baselines"
python run_top_aware_muon_sweep.py \
    --out-root "$V29_ROOT" \
    --log-root "$V29_LOG_ROOT" \
    --methods "streaming_identity streaming_lite native_muon native_lite top_aware_muon" \
    --batches "262144" \
    --alphas "0.5 0.75 0.875" \
    --lrs "0.005 0.01 0.02 0.04" \
    --seeds "42" \
    --tokens 1073741824 \
    --depth 8 \
    --nproc-per-node 8

echo "[done] v28/v29 complete"
echo "  v28: $V28_ROOT"
echo "  v29: $V29_ROOT"
