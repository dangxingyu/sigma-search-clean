#!/usr/bin/env bash
set -euo pipefail

# Dense d12 dynamics/statistics run.
#
# Use this after an optimizer-quality sweep has identified the LR/batch points
# worth inspecting. This script is standalone and directly invokes
# run_top_aware_muon_sweep.py with metrics enabled.

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
export NANOCHAT_FORCE_MATH_SDPA="${NANOCHAT_FORCE_MATH_SDPA:-1}"

DEPTH="${DEPTH:-12}"
CHINCHILLA_MULT="${CHINCHILLA_MULT:-2}"
# Exact TOKENS remains available for smoke tests or custom truncated runs.
TOKENS="${TOKENS:-}"
METHODS="${METHODS:-top_aware_muon}"
BATCHES="${BATCHES:-262144 1048576 4194304}"
ALPHAS="${ALPHAS:-1.0 0.5}"
TOP_KS="${TOP_KS:-1}"
LRS="${LRS:-0.02}"
SEEDS="${SEEDS:-42}"

NPROC="${NPROC:-8}"
MAX_DEVICE_BATCH_SIZE="${MAX_DEVICE_BATCH_SIZE:-16}"
SAVE_EVERY="${SAVE_EVERY:-100}"
KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-2}"
RESUME="${RESUME:-1}"

METRICS_EVERY="${METRICS_EVERY:-1}"
METRICS_TOP_K="${METRICS_TOP_K:-4}"
METRICS_MAX_MODULES="${METRICS_MAX_MODULES:-0}"
METRICS_HESSIAN_EVERY="${METRICS_HESSIAN_EVERY:-50}"
METRICS_HESSIAN_TOP_K="${METRICS_HESSIAN_TOP_K:-4}"
METRICS_HESSIAN_ITERS="${METRICS_HESSIAN_ITERS:-6}"
METRICS_HESSIAN_MAX_MODULES="${METRICS_HESSIAN_MAX_MODULES:-0}"
METRICS_PROJECTION_CORRELATION_WINDOW="${METRICS_PROJECTION_CORRELATION_WINDOW:-16}"

STAMP="${STAMP:-d12_statistics_$(date +%Y%m%d_%H%M%S)}"
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
  --depth "$DEPTH"
  --chinchilla-mult "$CHINCHILLA_MULT"
  --nproc-per-node "$NPROC"
  --max-device-batch-size "$MAX_DEVICE_BATCH_SIZE"
  --save-every "$SAVE_EVERY"
  --keep-last-checkpoints "$KEEP_LAST_CHECKPOINTS"
  --pure-qr
  --streaming-num-iters "${STREAMING_NUM_ITERS:-2}"
  --fallback-ortho-tol "${FALLBACK_ORTHO_TOL:-0.01}"
  --metrics-every "$METRICS_EVERY"
  --metrics-top-k "$METRICS_TOP_K"
  --metrics-module-regex "${METRICS_MODULE_REGEX:-transformer\\.h\\.(?:[0-9]+)\\.(?:attn\\.(?:c_q|c_k|c_v|c_proj)|mlp\\.(?:c_fc|c_proj))\\.weight$}"
  --metrics-max-modules "$METRICS_MAX_MODULES"
  --metrics-hessian-every "$METRICS_HESSIAN_EVERY"
  --metrics-hessian-top-k "$METRICS_HESSIAN_TOP_K"
  --metrics-hessian-iters "$METRICS_HESSIAN_ITERS"
  --metrics-hessian-max-modules "$METRICS_HESSIAN_MAX_MODULES"
  --metrics-projection-correlation-window "$METRICS_PROJECTION_CORRELATION_WINDOW"
)

if [[ -n "$TOKENS" ]]; then
  cmd+=(--tokens "$TOKENS")
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

printf 'Running statistics/dynamics run (d12 defaults; DEPTH/CHINCHILLA_MULT may override)\n'
printf 'OUT_ROOT=%s\nLOG_ROOT=%s\n' "$OUT_ROOT" "$LOG_ROOT"
if [[ -n "$TOKENS" ]]; then
  printf 'DEPTH=%s TOKENS=%s NPROC=%s\n' "$DEPTH" "$TOKENS" "$NPROC"
else
  printf 'DEPTH=%s CHINCHILLA_MULT=%s TOKENS=auto NPROC=%s\n' "$DEPTH" "$CHINCHILLA_MULT" "$NPROC"
fi
printf 'METHODS=%s\nBATCHES=%s\nALPHAS=%s\nTOP_KS=%s\nLRS=%s\nSEEDS=%s\n' \
  "$METHODS" "$BATCHES" "$ALPHAS" "$TOP_KS" "$LRS" "$SEEDS"
printf 'METRICS_EVERY=%s METRICS_HESSIAN_EVERY=%s METRICS_HESSIAN_TOP_K=%s\n' \
  "$METRICS_EVERY" "$METRICS_HESSIAN_EVERY" "$METRICS_HESSIAN_TOP_K"
printf 'Command:\n'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
