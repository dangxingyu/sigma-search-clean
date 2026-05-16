#!/usr/bin/env bash
set -euo pipefail

# Sequential multi-optimizer baseline sweep.
# Run from a GPU allocation/session; this script does not request SLURM itself.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

DEPTHS="${DEPTHS:-12 16}"
CHINCHILLA_MULT="${CHINCHILLA_MULT:-2}"
BATCHES="${BATCHES:-524288 2097152 8388608}"
SEEDS="${SEEDS:-42}"
ARCHITECTURE="${ARCHITECTURE:-gpt2}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
STAMP_PREFIX="${STAMP_PREFIX:-optimizer_baselines_$(date +%Y%m%d_%H%M%S)}"

echo "Running multi-optimizer baseline sweep"
echo "DEPTHS=${DEPTHS}"
echo "CHINCHILLA_MULT=${CHINCHILLA_MULT}"
echo "BATCHES=${BATCHES}"
echo "OPTIMIZER_GROUPS=${OPTIMIZER_GROUPS:-plain_muon adamw soap kl_soap kl_shampoo}"
echo "WEIGHT_DECAY=${WEIGHT_DECAY}"
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
  SEEDS="$SEEDS" \
  ARCHITECTURE="$ARCHITECTURE" \
  WEIGHT_DECAY="$WEIGHT_DECAY" \
  STAMP="$stamp" \
  bash scripts/run_optimizer_baseline_groups.sh "$@"
done
