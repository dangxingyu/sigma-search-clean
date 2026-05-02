#!/usr/bin/env bash
# Wait for v26 small-batch sweep, run large-batch same-driver sweep, then
# launch v27 d12/3B only if the scale-up gate passes.
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

SMALL_ROOT="${SMALL_ROOT:-search_evals/v26_small_bsz_streaming_fair_20260430}"
LARGE_ROOT="${LARGE_ROOT:-search_evals/v26b_large_bsz_streaming_fair_20260430}"
LARGE_LOG_ROOT="${LARGE_LOG_ROOT:-logs/v26b_large_bsz_streaming_fair_20260430}"
DECISION_JSON="${DECISION_JSON:-scaleup_gate_decision_20260430.json}"

export OMP_NUM_THREADS=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-sigma}"

echo "[wait] waiting for active v26 small-batch controller to finish"
while pgrep -af "[p]ython run_v26_small_bsz_streaming_fair.py --out-root ${SMALL_ROOT}" >/dev/null 2>&1; do
    sleep 300
done

echo "[run] v26b large-batch same-driver sweep"
python run_v26_small_bsz_streaming_fair.py \
    --out-root "$LARGE_ROOT" \
    --log-root "$LARGE_LOG_ROOT" \
    --batches "1048576 8388608" \
    --seeds "42" \
    --lrs "0.005 0.01 0.02 0.04"

echo "[gate] deciding whether to launch v27"
set +e
python decide_scaleup_gate.py "$SMALL_ROOT" "$LARGE_ROOT" --decision-json "$DECISION_JSON"
gate_rc=$?
set -e

if [ "$gate_rc" -eq 0 ]; then
    echo "[launch] scale-up gate passed; starting v27 d12/3B"
    STAMP=20260430_gate_passed bash run_v27_d12_3b_streaming_fair_on_node.sh
else
    echo "[stop] scale-up gate did not pass; not launching v27 (rc=$gate_rc)"
fi
