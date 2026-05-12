#!/usr/bin/env bash
set -euo pipefail

# Apple-to-apple d16 non-streaming optimizer baseline sweep.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

export DEPTH="${DEPTH:-16}"
export CHINCHILLA_MULT="${CHINCHILLA_MULT:-2}"
export BATCHES="${BATCHES:-524288 2097152 8388608}"
export METHODS="${METHODS:-plain_muon adamw soap shampoo kl_shampoo kl_soap}"
export ALPHAS="${ALPHAS:-1.0}"
export TOP_KS="${TOP_KS:-1}"
export LRS="${LRS:-0.005 0.0075 0.01 0.015 0.02 0.03 0.04}"
export SEEDS="${SEEDS:-42}"
export ARCHITECTURE="${ARCHITECTURE:-gpt2}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
export STAMP="${STAMP:-d16_optimizer_baselines_$(date +%Y%m%d_%H%M%S)}"

bash scripts/run_d12_sweep.sh "$@"
