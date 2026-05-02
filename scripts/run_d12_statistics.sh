#!/usr/bin/env bash
set -euo pipefail

# Dense d12 dynamics/statistics run.
#
# Use this after an optimizer-quality sweep has identified the LR/batch points
# worth inspecting. Defaults are conservative example points; override BATCHES,
# LRS, METHODS, and ALPHAS to match the selected sweep winners.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

export DEPTH="${DEPTH:-12}"
# Same 1x Chinchilla-style d12 budget as run_d12_sweep.sh.
export TOKENS="${TOKENS:-1698693120}"
export METHODS="${METHODS:-streaming_identity top_aware_muon}"
export BATCHES="${BATCHES:-262144 1048576 4194304}"
export ALPHAS="${ALPHAS:-0.5}"
export TOP_KS="${TOP_KS:-1}"
export LRS="${LRS:-0.02}"
export SEEDS="${SEEDS:-42}"
export ADAPTIVE_LR="${ADAPTIVE_LR:-0}"
export NPROC="${NPROC:-8}"
export MAX_DEVICE_BATCH_SIZE="${MAX_DEVICE_BATCH_SIZE:-16}"

export METRICS_EVERY="${METRICS_EVERY:-1}"
export METRICS_TOP_K="${METRICS_TOP_K:-4}"
export METRICS_MAX_MODULES="${METRICS_MAX_MODULES:-0}"
export METRICS_HESSIAN_EVERY="${METRICS_HESSIAN_EVERY:-50}"
export METRICS_HESSIAN_TOP_K="${METRICS_HESSIAN_TOP_K:-4}"
export METRICS_HESSIAN_ITERS="${METRICS_HESSIAN_ITERS:-6}"
export METRICS_HESSIAN_MAX_MODULES="${METRICS_HESSIAN_MAX_MODULES:-0}"
export NANOCHAT_FORCE_MATH_SDPA="${NANOCHAT_FORCE_MATH_SDPA:-1}"

export STAMP="${STAMP:-d12_statistics_$(date +%Y%m%d_%H%M%S)}"
export OUT_ROOT="${OUT_ROOT:-search_evals/${STAMP}}"
export LOG_ROOT="${LOG_ROOT:-logs/${STAMP}}"

echo "Running d12 statistics/dynamics run"
echo "  DEPTH=$DEPTH TOKENS=$TOKENS NPROC=$NPROC"
echo "  METHODS=$METHODS"
echo "  BATCHES=$BATCHES"
echo "  ALPHAS=$ALPHAS TOP_KS=$TOP_KS"
echo "  LRS=$LRS"
echo "  SEEDS=$SEEDS"
echo "  METRICS_EVERY=$METRICS_EVERY METRICS_HESSIAN_EVERY=$METRICS_HESSIAN_EVERY"
echo "  OUT_ROOT=$OUT_ROOT"
echo

bash scripts/run_handoff_sweep.sh "$@"
