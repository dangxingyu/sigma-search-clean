#!/usr/bin/env bash
set -euo pipefail

# Canonical d8 dense-metrics grid. This is a thin wrapper around the main
# handoff sweep so optimizer, checkpoint, and metrics flags stay in one place.

export DEPTH="${DEPTH:-8}"
export TOKENS="${TOKENS:-402653184}"
export METHODS="${METHODS:-streaming_identity top_aware_muon}"
export BATCHES="${BATCHES:-262144 1048576 4194304}"
export ALPHAS="${ALPHAS:-0.5}"
export TOP_KS="${TOP_KS:-1}"
export LRS="${LRS:-0.02}"
export SEEDS="${SEEDS:-42}"
export ADAPTIVE_LR="${ADAPTIVE_LR:-0}"

export METRICS_EVERY="${METRICS_EVERY:-1}"
export METRICS_TOP_K="${METRICS_TOP_K:-4}"
export METRICS_MAX_MODULES="${METRICS_MAX_MODULES:-0}"
export METRICS_HESSIAN_EVERY="${METRICS_HESSIAN_EVERY:-50}"
export METRICS_HESSIAN_TOP_K="${METRICS_HESSIAN_TOP_K:-4}"
export METRICS_HESSIAN_ITERS="${METRICS_HESSIAN_ITERS:-6}"
export METRICS_HESSIAN_MAX_MODULES="${METRICS_HESSIAN_MAX_MODULES:-0}"
export NANOCHAT_FORCE_MATH_SDPA="${NANOCHAT_FORCE_MATH_SDPA:-1}"

export STAMP="${STAMP:-d8_metrics_grid_$(date +%Y%m%d_%H%M%S)}"
export OUT_ROOT="${OUT_ROOT:-search_evals/${STAMP}}"
export LOG_ROOT="${LOG_ROOT:-logs/${STAMP}}"

bash "$(dirname "${BASH_SOURCE[0]}")/run_handoff_sweep.sh" "$@"
