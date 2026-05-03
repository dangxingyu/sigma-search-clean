#!/usr/bin/env bash
set -euo pipefail

# Canonical sequential handoff grid.
#
# This is the safest command to give someone who should not need to remember
# optimizer flags. It runs d12 and d16 separately because each depth has a
# different token budget and manifest signature.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

DEPTHS="${DEPTHS:-12 16}"
CHINCHILLA_MULT="${CHINCHILLA_MULT:-2}"
BATCHES="${BATCHES:-524288 2097152 8388608}"
ALPHAS="${ALPHAS:-1.0 0.5}"
TOP_KS="${TOP_KS:-1}"
LRS="${LRS:-0.005 0.01 0.02 0.04}"
SEEDS="${SEEDS:-42}"
STAMP_PREFIX="${STAMP_PREFIX:-handoff_grid_$(date +%Y%m%d_%H%M%S)}"

echo "Running canonical d12/d16 2x-Chinchilla grid"
echo "DEPTHS=${DEPTHS}"
echo "CHINCHILLA_MULT=${CHINCHILLA_MULT}"
echo "BATCHES=${BATCHES}"
echo "ALPHAS=${ALPHAS}"
echo "LRS=${LRS}"
echo "SEEDS=${SEEDS}"
echo "STAMP_PREFIX=${STAMP_PREFIX}"

for depth in $DEPTHS; do
  stamp="${STAMP_PREFIX}_d${depth}_${CHINCHILLA_MULT}x"
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
