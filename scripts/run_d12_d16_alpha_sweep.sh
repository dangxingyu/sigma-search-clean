#!/usr/bin/env bash
set -euo pipefail

# Sequential d12/d16 alpha sweep for Top-Aware Muon.
#
# This is the copy-paste-safe handoff command when someone wants to sweep the
# Top-Aware coefficient c/alpha rather than only compare c=1.0 vs c=0.5.
# Run inside an existing GPU allocation, or use submit_slurm_grid.sh for
# per-case SLURM arrays as described in docs/sadhika_alpha_sweep_handoff.md.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

DEPTHS="${DEPTHS:-12 16}"
CHINCHILLA_MULT="${CHINCHILLA_MULT:-2}"
BATCHES="${BATCHES:-524288 2097152 8388608}"
ALPHAS="${ALPHAS:-0.25 0.5 0.75 0.85 1.0 1.15}"
TOP_KS="${TOP_KS:-1}"
LRS="${LRS:-0.005 0.0075 0.01 0.015 0.02 0.03 0.04}"
SEEDS="${SEEDS:-42}"
STAMP_PREFIX="${STAMP_PREFIX:-alpha_sweep_$(date +%Y%m%d_%H%M%S)}"

# Forward the usual optimizer-quality defaults unless the caller overrides.
export METHODS="${METHODS:-top_aware_muon}"
export NPROC="${NPROC:-8}"
export MAX_DEVICE_BATCH_SIZE="${MAX_DEVICE_BATCH_SIZE:-16}"
export SAVE_EVERY="${SAVE_EVERY:-100}"
export KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-2}"
export RESUME="${RESUME:-1}"
export ADAPTIVE_LR="${ADAPTIVE_LR:-1}"
export LR_MIN="${LR_MIN:-0.0005}"
export LR_MAX="${LR_MAX:-0.16}"
export MAX_LR_EXTENSION_ROUNDS="${MAX_LR_EXTENSION_ROUNDS:-2}"
export METRICS_EVERY="${METRICS_EVERY:-0}"
export METRICS_HESSIAN_EVERY="${METRICS_HESSIAN_EVERY:-0}"

echo "Running d12/d16 Top-Aware alpha sweep"
echo "DEPTHS=${DEPTHS}"
echo "CHINCHILLA_MULT=${CHINCHILLA_MULT}"
echo "BATCHES=${BATCHES}"
echo "ALPHAS=${ALPHAS}"
echo "TOP_KS=${TOP_KS}"
echo "LRS=${LRS}"
echo "SEEDS=${SEEDS}"
echo "STAMP_PREFIX=${STAMP_PREFIX}"
echo "ADAPTIVE_LR=${ADAPTIVE_LR} LR_MIN=${LR_MIN} LR_MAX=${LR_MAX}"

for depth in $DEPTHS; do
  stamp="${STAMP_PREFIX}_d${depth}_${CHINCHILLA_MULT}x_alpha"
  stamp="${stamp//./p}"
  echo
  echo "=== depth ${depth}: STAMP=${stamp} ==="
  DEPTH="$depth" \
  CHINCHILLA_MULT="$CHINCHILLA_MULT" \
  BATCHES="$BATCHES" \
  ALPHAS="$ALPHAS" \
  TOP_KS="$TOP_KS" \
  LRS="$LRS" \
  SEEDS="$SEEDS" \
  STAMP="$stamp" \
  bash scripts/run_d12_sweep.sh "$@"
done
