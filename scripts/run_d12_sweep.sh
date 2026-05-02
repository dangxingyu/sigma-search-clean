#!/usr/bin/env bash
set -euo pipefail

# Blessed d12 optimizer-quality sweep for handoff use.
#
# Run this from a node/session that already has the intended GPUs visible.
# This script is standalone: it directly invokes run_top_aware_muon_sweep.py
# and does not depend on any other sweep wrapper.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PARENT_REPO="$(cd "$REPO/.." && pwd)"
if [[ -f "$PARENT_REPO/nanochat/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$PARENT_REPO/nanochat/.venv/bin/activate"
elif [[ -f "$REPO/nanochat/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$REPO/nanochat/.venv/bin/activate"
fi

export PYTHONPATH="$REPO:$REPO/nanochat"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

DEPTH="${DEPTH:-12}"
# 1x Chinchilla-style d12 budget: 20 tokens per non-embedding parameter.
# For nanochat d12 this is 20 * 84,935,570 = 1,698,711,400 tokens,
# rounded down by 18,280 tokens so it is divisible by the 4M batch grid.
TOKENS="${TOKENS:-1698693120}"
METHODS="${METHODS:-streaming_identity top_aware_muon}"
BATCHES="${BATCHES:-262144 1048576 4194304}"
ALPHAS="${ALPHAS:-0.5}"
TOP_KS="${TOP_KS:-1}"
LRS="${LRS:-0.005 0.01 0.02 0.04}"
SEEDS="${SEEDS:-42}"

NPROC="${NPROC:-8}"
MAX_DEVICE_BATCH_SIZE="${MAX_DEVICE_BATCH_SIZE:-16}"
SAVE_EVERY="${SAVE_EVERY:-100}"
KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-2}"
RESUME="${RESUME:-1}"

ADAPTIVE_LR="${ADAPTIVE_LR:-1}"
LR_EXTEND_FACTOR="${LR_EXTEND_FACTOR:-2.0}"
LR_MIN="${LR_MIN:-0.0005}"
LR_MAX="${LR_MAX:-0.16}"
MAX_LR_EXTENSION_ROUNDS="${MAX_LR_EXTENSION_ROUNDS:-2}"
ADAPTIVE_MIN_EDGE_IMPROVEMENT="${ADAPTIVE_MIN_EDGE_IMPROVEMENT:-0.0}"

# Optimizer-quality sweep: keep dense dynamics metrics off by default.
METRICS_EVERY="${METRICS_EVERY:-0}"
METRICS_HESSIAN_EVERY="${METRICS_HESSIAN_EVERY:-0}"
if [[ "$METRICS_HESSIAN_EVERY" != "0" ]]; then
  export NANOCHAT_FORCE_MATH_SDPA="${NANOCHAT_FORCE_MATH_SDPA:-1}"
fi

STAMP="${STAMP:-d12_sweep_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-search_evals/${STAMP}}"
LOG_ROOT="${LOG_ROOT:-logs/${STAMP}}"

cmd=(
  python run_top_aware_muon_sweep.py
  --out-root "$OUT_ROOT"
  --log-root "$LOG_ROOT"
  --nanochat-dir nanochat
  --methods "$METHODS"
  --batches "$BATCHES"
  --alphas "$ALPHAS"
  --top-ks "$TOP_KS"
  --lrs "$LRS"
  --seeds "$SEEDS"
  --tokens "$TOKENS"
  --depth "$DEPTH"
  --nproc-per-node "$NPROC"
  --max-device-batch-size "$MAX_DEVICE_BATCH_SIZE"
  --save-every "$SAVE_EVERY"
  --keep-last-checkpoints "$KEEP_LAST_CHECKPOINTS"
  --pure-qr
  --streaming-num-iters "${STREAMING_NUM_ITERS:-2}"
  --fallback-ortho-tol "${FALLBACK_ORTHO_TOL:-0.01}"
  --metrics-every "$METRICS_EVERY"
  --metrics-top-k "${METRICS_TOP_K:-4}"
  --metrics-module-regex "${METRICS_MODULE_REGEX:-transformer\\.h\\.(?:[0-9]+)\\.(?:attn\\.(?:c_q|c_k|c_v|c_proj)|mlp\\.(?:c_fc|c_proj))\\.weight$}"
  --metrics-max-modules "${METRICS_MAX_MODULES:-0}"
  --metrics-hessian-every "$METRICS_HESSIAN_EVERY"
  --metrics-hessian-top-k "${METRICS_HESSIAN_TOP_K:-4}"
  --metrics-hessian-iters "${METRICS_HESSIAN_ITERS:-6}"
  --metrics-hessian-max-modules "${METRICS_HESSIAN_MAX_MODULES:-0}"
  --metrics-projection-correlation-window "${METRICS_PROJECTION_CORRELATION_WINDOW:-16}"
  --lr-extend-factor "$LR_EXTEND_FACTOR"
  --lr-min "$LR_MIN"
  --lr-max "$LR_MAX"
  --max-lr-extension-rounds "$MAX_LR_EXTENSION_ROUNDS"
  --adaptive-min-edge-improvement "$ADAPTIVE_MIN_EDGE_IMPROVEMENT"
)

if [[ "$ADAPTIVE_LR" == "1" ]]; then
  cmd+=(--adaptive-lr)
fi
if [[ "${ALLOW_TOP_K_SWEEP:-0}" == "1" ]]; then
  cmd+=(--allow-top-k-sweep)
fi
if [[ "$RESUME" == "1" ]]; then
  cmd+=(--resume)
else
  cmd+=(--no-resume)
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  cmd+=(--dry-run)
fi
if [[ "${RERUN_EXISTING:-0}" == "1" ]]; then
  cmd+=(--rerun-existing)
fi
cmd+=("$@")

printf 'Running optimizer-quality sweep (d12 defaults; DEPTH/TOKENS may override)\n'
printf 'OUT_ROOT=%s\nLOG_ROOT=%s\n' "$OUT_ROOT" "$LOG_ROOT"
printf 'DEPTH=%s TOKENS=%s NPROC=%s\n' "$DEPTH" "$TOKENS" "$NPROC"
printf 'METHODS=%s\nBATCHES=%s\nALPHAS=%s\nTOP_KS=%s\nLRS=%s\nSEEDS=%s\n' \
  "$METHODS" "$BATCHES" "$ALPHAS" "$TOP_KS" "$LRS" "$SEEDS"
printf 'CHECKPOINTING=save_every:%s keep_last:%s resume:%s\n' "$SAVE_EVERY" "$KEEP_LAST_CHECKPOINTS" "$RESUME"
printf 'ADAPTIVE_LR=%s LR_EXTEND_FACTOR=%s LR_MIN=%s LR_MAX=%s MAX_ROUNDS=%s\n' \
  "$ADAPTIVE_LR" "$LR_EXTEND_FACTOR" "$LR_MIN" "$LR_MAX" "$MAX_LR_EXTENSION_ROUNDS"
printf 'Command:\n'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
