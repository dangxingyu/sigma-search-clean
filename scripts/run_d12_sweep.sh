#!/usr/bin/env bash
set -euo pipefail

# Blessed d12 optimizer-quality sweep for handoff use.
#
# Run this from a node/session that already has the intended GPUs visible.
# This script is standalone: it directly invokes run_optimizer_sweep.py
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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

DEPTH="${DEPTH:-12}"
CHINCHILLA_MULT="${CHINCHILLA_MULT:-2}"
# Exact TOKENS remains available for smoke tests or custom truncated runs.
TOKENS="${TOKENS:-}"
METHODS="${METHODS:-top_aware_muon}"
BATCHES="${BATCHES:-262144 1048576 4194304}"
ALPHAS="${ALPHAS:-1.0 0.5}"
TOP_KS="${TOP_KS:-1}"
LRS="${LRS:-0.005 0.0075 0.01 0.015 0.02 0.03 0.04}"
SEEDS="${SEEDS:-42}"
ARCHITECTURE="${ARCHITECTURE:-gpt2}"

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
  python run_optimizer_sweep.py
  --out-root "$OUT_ROOT"
  --log-root "$LOG_ROOT"
  --nanochat-dir nanochat
  --methods "$METHODS"
  --batches "$BATCHES"
  --alphas "$ALPHAS"
  --top-ks "$TOP_KS"
  --lrs "$LRS"
  --seeds "$SEEDS"
  --depth "$DEPTH"
  --architecture "$ARCHITECTURE"
  --chinchilla-mult "$CHINCHILLA_MULT"
  --nproc-per-node "$NPROC"
  --max-device-batch-size "$MAX_DEVICE_BATCH_SIZE"
  --weight-decay "${WEIGHT_DECAY:-0.28}"
  --muon-momentum "${MUON_MOMENTUM:-0.95}"
  --muon-momentum-schedule "${MUON_MOMENTUM_SCHEDULE:-nanochat}"
  --save-every "$SAVE_EVERY"
  --keep-last-checkpoints "$KEEP_LAST_CHECKPOINTS"
  --pure-qr
  --streaming-num-iters "${STREAMING_NUM_ITERS:-2}"
  --fallback-ortho-tol "${FALLBACK_ORTHO_TOL:-0.01}"
  --matrix-lr-adjust "${MATRIX_LR_ADJUST:-moonlight}"
  --adam-lr-mode "${ADAM_LR_MODE:-relative_to_matrix}"
  --structured-config "${STRUCTURED_CONFIG:-auto}"
  --precondition-frequency "${PRECONDITION_FREQUENCY:-5}"
  --shampoo-beta "${SHAMPOO_BETA:-0.95}"
  --optimizer-beta1 "${OPTIMIZER_BETA1:-0.9}"
  --optimizer-beta2 "${OPTIMIZER_BETA2:-0.95}"
  --structured-init-factor "${STRUCTURED_INIT_FACTOR:-1.0}"
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

if [[ -n "$TOKENS" ]]; then
  cmd+=(--tokens "$TOKENS")
fi
if [[ "$ADAPTIVE_LR" == "1" ]]; then
  cmd+=(--adaptive-lr)
fi
if [[ "${ALLOW_TOP_K_SWEEP:-0}" == "1" ]]; then
  cmd+=(--allow-top-k-sweep)
fi
if [[ "${STRUCTURED_USE_QR:-1}" == "1" ]]; then
  cmd+=(--structured-use-qr)
else
  cmd+=(--no-structured-use-qr)
fi
if [[ "${BATCH_BETA_ALIGN:-1}" == "1" ]]; then
  cmd+=(--batch-beta-align)
else
  cmd+=(--no-batch-beta-align)
fi
cmd+=(--batch-beta-align-mode "${BATCH_BETA_ALIGN_MODE:-all}")
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
if [[ "${SUMMARY_ONLY:-0}" == "1" ]]; then
  cmd+=(--summary-only)
fi
if [[ "${ALLOW_CONFIG_MISMATCH:-0}" == "1" ]]; then
  cmd+=(--allow-config-mismatch)
fi
cmd+=("$@")

printf 'Running optimizer-quality sweep (d12 defaults; DEPTH/CHINCHILLA_MULT may override)\n'
printf 'OUT_ROOT=%s\nLOG_ROOT=%s\n' "$OUT_ROOT" "$LOG_ROOT"
if [[ -n "$TOKENS" ]]; then
  printf 'DEPTH=%s TOKENS=%s NPROC=%s ARCHITECTURE=%s\n' "$DEPTH" "$TOKENS" "$NPROC" "$ARCHITECTURE"
else
  printf 'DEPTH=%s CHINCHILLA_MULT=%s TOKENS=auto NPROC=%s ARCHITECTURE=%s\n' "$DEPTH" "$CHINCHILLA_MULT" "$NPROC" "$ARCHITECTURE"
fi
printf 'METHODS=%s\nBATCHES=%s\nALPHAS=%s\nTOP_KS=%s\nLRS=%s\nSEEDS=%s\n' \
  "$METHODS" "$BATCHES" "$ALPHAS" "$TOP_KS" "$LRS" "$SEEDS"
printf 'CHECKPOINTING=save_every:%s keep_last:%s resume:%s\n' "$SAVE_EVERY" "$KEEP_LAST_CHECKPOINTS" "$RESUME"
printf 'ADAPTIVE_LR=%s LR_EXTEND_FACTOR=%s LR_MIN=%s LR_MAX=%s MAX_ROUNDS=%s\n' \
  "$ADAPTIVE_LR" "$LR_EXTEND_FACTOR" "$LR_MIN" "$LR_MAX" "$MAX_LR_EXTENSION_ROUNDS"
printf 'Command:\n'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
