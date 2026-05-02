#!/usr/bin/env bash
set -euo pipefail

# Blessed d12 optimizer-quality sweep for handoff use.
#
# Run this from a node/session that already has the intended GPUs visible.
# Do not wrap scheduler-specific logic in this script; use your cluster's
# allocation mechanism outside this command.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

export DEPTH="${DEPTH:-12}"
# 1x Chinchilla-style d12 budget. This scales the d8 0.4B recipe by
# d12/d8 scaling-parameter ratio, then rounds to be divisible by 4M.
export TOKENS="${TOKENS:-977272832}"
export METHODS="${METHODS:-streaming_identity top_aware_muon}"
export BATCHES="${BATCHES:-262144 1048576 4194304}"
export ALPHAS="${ALPHAS:-0.5}"
export LRS="${LRS:-0.005 0.01 0.02 0.04}"
export SEEDS="${SEEDS:-42}"
export ADAPTIVE_LR="${ADAPTIVE_LR:-1}"
export LR_MAX="${LR_MAX:-0.08}"
export NPROC="${NPROC:-8}"
export MAX_DEVICE_BATCH_SIZE="${MAX_DEVICE_BATCH_SIZE:-16}"

# Optimizer-quality sweep: keep dense dynamics metrics off by default.
export METRICS_EVERY="${METRICS_EVERY:-0}"
export METRICS_HESSIAN_EVERY="${METRICS_HESSIAN_EVERY:-0}"

export STAMP="${STAMP:-d12_sweep_$(date +%Y%m%d_%H%M%S)}"
export OUT_ROOT="${OUT_ROOT:-search_evals/${STAMP}}"
export LOG_ROOT="${LOG_ROOT:-logs/${STAMP}}"

echo "Running blessed d12 sweep"
echo "  DEPTH=$DEPTH TOKENS=$TOKENS NPROC=$NPROC"
echo "  METHODS=$METHODS"
echo "  BATCHES=$BATCHES"
echo "  ALPHAS=$ALPHAS"
echo "  LRS=$LRS"
echo "  SEEDS=$SEEDS"
echo "  OUT_ROOT=$OUT_ROOT"
echo

bash scripts/run_handoff_sweep.sh "$@"
