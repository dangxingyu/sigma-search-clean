#!/usr/bin/env bash
set -euo pipefail

# Sequential multi-optimizer baseline sweep.
# Run from a GPU allocation/session; this script does not request SLURM itself.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

DEPTHS="${DEPTHS:-12 16}"
CHINCHILLA_MULT="${CHINCHILLA_MULT:-2}"
BATCHES="${BATCHES:-524288 2097152 8388608}"
METHODS="${METHODS:-plain_muon adamw soap shampoo kl_shampoo kl_soap}"
ALPHAS="${ALPHAS:-1.0}"
TOP_KS="${TOP_KS:-1}"
LRS="${LRS:-0.005 0.0075 0.01 0.015 0.02 0.03 0.04}"
SEEDS="${SEEDS:-42}"
ARCHITECTURE="${ARCHITECTURE:-gpt2}"
STAMP_PREFIX="${STAMP_PREFIX:-optimizer_baselines_$(date +%Y%m%d_%H%M%S)}"

echo "Running multi-optimizer baseline sweep"
echo "DEPTHS=${DEPTHS}"
echo "CHINCHILLA_MULT=${CHINCHILLA_MULT}"
echo "BATCHES=${BATCHES}"
echo "METHODS=${METHODS}"
echo "ALPHAS=${ALPHAS} (ignored unless METHODS includes top_aware_muon)"
echo "LRS=${LRS}"
echo "ARCHITECTURE=${ARCHITECTURE}"
echo "STAMP_PREFIX=${STAMP_PREFIX}"

for depth in $DEPTHS; do
  stamp="${STAMP_PREFIX}_d${depth}_${CHINCHILLA_MULT}x"
  stamp="${stamp//./p}"
  echo
  echo "=== depth ${depth}: STAMP=${stamp} ==="
  DEPTH="$depth" \
  CHINCHILLA_MULT="$CHINCHILLA_MULT" \
  BATCHES="$BATCHES" \
  METHODS="$METHODS" \
  ALPHAS="$ALPHAS" \
  TOP_KS="$TOP_KS" \
  LRS="$LRS" \
  SEEDS="$SEEDS" \
  ARCHITECTURE="$ARCHITECTURE" \
  STAMP="$stamp" \
  bash scripts/run_d12_sweep.sh "$@"
done
