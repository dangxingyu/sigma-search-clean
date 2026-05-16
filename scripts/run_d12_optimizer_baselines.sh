#!/usr/bin/env bash
set -euo pipefail

# Apple-to-apple d12 non-streaming optimizer baseline sweep.
#
# Defaults use method-grouped fixed LR grids because AdamW, Muon, Shampoo, and
# KL-Shampoo do not share the same LR scale. Adaptive LR is off by default for
# handoff clusters; set ADAPTIVE_LR=1 only if the cluster can run boundary
# closure jobs after the base grid.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

export DEPTH="${DEPTH:-12}"
export CHINCHILLA_MULT="${CHINCHILLA_MULT:-2}"
export BATCHES="${BATCHES:-524288 2097152 8388608}"
export SEEDS="${SEEDS:-42}"
export ARCHITECTURE="${ARCHITECTURE:-gpt2}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
export STAMP="${STAMP:-d12_optimizer_baselines_$(date +%Y%m%d_%H%M%S)}"

bash scripts/run_optimizer_baseline_groups.sh "$@"
