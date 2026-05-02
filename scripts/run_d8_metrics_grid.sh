#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PARENT_REPO="$(cd "$REPO/.." && pwd)"
if [[ -f "$PARENT_REPO/nanochat/.venv/bin/activate" ]]; then
  # Prefer the live cluster venv when this repo is used in-place.
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
export NANOCHAT_FORCE_MATH_SDPA="${NANOCHAT_FORCE_MATH_SDPA:-1}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-search_evals/d8_metrics_alpha_grid_${STAMP}}"
LOG_ROOT="${LOG_ROOT:-logs/d8_metrics_alpha_grid_${STAMP}}"

python run_top_aware_muon_sweep.py \
  --out-root "$OUT_ROOT" \
  --log-root "$LOG_ROOT" \
  --nanochat-dir nanochat \
  --methods "${METHODS:-top_aware_muon}" \
  --batches "${BATCHES:-262144 1048576 4194304}" \
  --alphas "${ALPHAS:-0.5 1.0}" \
  --lrs "${LRS:-0.02}" \
  --seeds "${SEEDS:-42}" \
  --tokens "${TOKENS:-402653184}" \
  --depth "${DEPTH:-8}" \
  --nproc-per-node "${NPROC:-8}" \
  --max-device-batch-size "${MAX_DEVICE_BATCH_SIZE:-16}" \
  --pure-qr \
  --streaming-num-iters "${STREAMING_NUM_ITERS:-2}" \
  --fallback-ortho-tol "${FALLBACK_ORTHO_TOL:-0.01}" \
  --metrics-every "${METRICS_EVERY:-1}" \
  --metrics-top-k "${METRICS_TOP_K:-4}" \
  --metrics-module-regex "${METRICS_MODULE_REGEX:-transformer\\.h}" \
  --metrics-max-modules "${METRICS_MAX_MODULES:-8}" \
  --metrics-split-momentum \
  --metrics-alignment-side "${METRICS_ALIGNMENT_SIDE:-lite}" \
  --metrics-hessian-every "${METRICS_HESSIAN_EVERY:-50}" \
  --metrics-hessian-top-k "${METRICS_HESSIAN_TOP_K:-1}" \
  --metrics-hessian-iters "${METRICS_HESSIAN_ITERS:-2}" \
  --metrics-hessian-max-modules "${METRICS_HESSIAN_MAX_MODULES:-8}"
