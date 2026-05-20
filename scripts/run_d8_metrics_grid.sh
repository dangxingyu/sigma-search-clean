#!/usr/bin/env bash
set -euo pipefail

# Canonical d8 no-tuning dynamics/metrics grid.
#
# This restores the documented entrypoint from results/recipes/d8_metrics_grid_recipe.json.
# It delegates to run_d12_statistics.sh after pinning the d8 recipe knobs.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

export DEPTH="${DEPTH:-8}"
export TOKENS="${TOKENS:-402653184}"
export METHODS="${METHODS:-top_aware_muon}"
export BATCHES="${BATCHES:-262144 1048576 4194304}"
export ALPHAS="${ALPHAS:-0.5 1.0}"
export TOP_KS="${TOP_KS:-1}"
export LRS="${LRS:-0.02}"
export SEEDS="${SEEDS:-42}"
export ARCHITECTURE="${ARCHITECTURE:-gpt2}"

export NPROC="${NPROC:-8}"
export MAX_DEVICE_BATCH_SIZE="${MAX_DEVICE_BATCH_SIZE:-16}"
export SAVE_EVERY="${SAVE_EVERY:-100}"
export KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-2}"
export RESUME="${RESUME:-1}"

export STREAMING_NUM_ITERS="${STREAMING_NUM_ITERS:-2}"
export FALLBACK_ORTHO_TOL="${FALLBACK_ORTHO_TOL:-0.01}"

export METRICS_EVERY="${METRICS_EVERY:-1}"
export METRICS_TOP_K="${METRICS_TOP_K:-4}"
export METRICS_MAX_MODULES="${METRICS_MAX_MODULES:-8}"
export METRICS_HESSIAN_EVERY="${METRICS_HESSIAN_EVERY:-50}"
export METRICS_HESSIAN_TOP_K="${METRICS_HESSIAN_TOP_K:-1}"
export METRICS_HESSIAN_ITERS="${METRICS_HESSIAN_ITERS:-2}"
export METRICS_HESSIAN_MAX_MODULES="${METRICS_HESSIAN_MAX_MODULES:-8}"
export METRICS_PROJECTION_CORRELATION_WINDOW="${METRICS_PROJECTION_CORRELATION_WINDOW:-16}"
export NANOCHAT_FORCE_MATH_SDPA="${NANOCHAT_FORCE_MATH_SDPA:-1}"

export STAMP="${STAMP:-d8_metrics_grid_$(date +%Y%m%d_%H%M%S)}"

echo "Running canonical d8 metrics grid via scripts/run_d12_statistics.sh"
echo "DEPTH=$DEPTH TOKENS=$TOKENS BATCHES=$BATCHES ALPHAS=$ALPHAS LRS=$LRS"
echo "METRICS_EVERY=$METRICS_EVERY METRICS_HESSIAN_EVERY=$METRICS_HESSIAN_EVERY"

bash scripts/run_d12_statistics.sh "$@"
