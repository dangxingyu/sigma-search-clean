#!/usr/bin/env bash
set -euo pipefail

# Method-grouped non-streaming optimizer baseline sweep.
#
# This is the safer handoff path for AdamW/Muon/SOAP-family comparisons because
# these optimizers do not share a useful LR scale. Each group writes to its own
# STAMP suffix, so manifests stay clean and resume remains straightforward.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

DEPTH="${DEPTH:-12}"
CHINCHILLA_MULT="${CHINCHILLA_MULT:-2}"
BATCHES="${BATCHES:-524288 2097152 8388608}"
SEEDS="${SEEDS:-42}"
ARCHITECTURE="${ARCHITECTURE:-gpt2}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
ADAPTIVE_LR="${ADAPTIVE_LR:-0}"
BASE_STAMP="${STAMP:-d${DEPTH}_optimizer_baselines_$(date +%Y%m%d_%H%M%S)}"

# Space-separated group suffixes to run. Useful subsets:
#   OPTIMIZER_GROUPS="plain_muon soap_klsoap"  # cheap competitive check
#   OPTIMIZER_GROUPS="adamw shampoo"           # low-LR ablations
#
# Do not use the shell variable name GROUPS here; Bash reserves it for the
# current user's Unix group IDs.
OPTIMIZER_GROUPS="${OPTIMIZER_GROUPS:-plain_muon adamw soap_klsoap kl_shampoo shampoo}"

run_group() {
  local suffix="$1"
  local methods="$2"
  local lrs="$3"
  local lr_min="$4"
  local lr_max="$5"
  shift 5

  if [[ " ${OPTIMIZER_GROUPS} " != *" ${suffix} "* ]]; then
    echo "Skipping group ${suffix}; OPTIMIZER_GROUPS=${OPTIMIZER_GROUPS}"
    return
  fi

  local stamp="${BASE_STAMP}_${suffix}"
  stamp="${stamp//./p}"
  echo
  echo "=== optimizer baseline group: ${suffix} ==="
  echo "METHODS=${methods}"
  echo "LRS=${lrs}"
  echo "STAMP=${stamp}"

  DEPTH="$DEPTH" \
  CHINCHILLA_MULT="$CHINCHILLA_MULT" \
  BATCHES="$BATCHES" \
  METHODS="$methods" \
  ALPHAS="1.0" \
  TOP_KS="1" \
  LRS="$lrs" \
  SEEDS="$SEEDS" \
  ARCHITECTURE="$ARCHITECTURE" \
  WEIGHT_DECAY="$WEIGHT_DECAY" \
  ADAPTIVE_LR="$ADAPTIVE_LR" \
  STAMP="$stamp" \
  LR_MIN="$lr_min" \
  LR_MAX="$lr_max" \
  bash scripts/run_d12_sweep.sh "$@"
}

echo "Running method-grouped optimizer baseline sweep"
echo "DEPTH=${DEPTH}"
echo "CHINCHILLA_MULT=${CHINCHILLA_MULT}"
echo "BATCHES=${BATCHES}"
echo "SEEDS=${SEEDS}"
echo "WEIGHT_DECAY=${WEIGHT_DECAY}"
echo "ADAPTIVE_LR=${ADAPTIVE_LR}"
echo "BASE_STAMP=${BASE_STAMP}"
echo "OPTIMIZER_GROUPS=${OPTIMIZER_GROUPS}"

if [[ "${GROUPED:-1}" == "0" ]]; then
  echo
  echo "GROUPED=0: running legacy single-grid baseline wrapper"
  export DEPTH CHINCHILLA_MULT BATCHES SEEDS ARCHITECTURE WEIGHT_DECAY ADAPTIVE_LR
  export METHODS="${METHODS:-plain_muon adamw soap shampoo kl_shampoo kl_soap}"
  export ALPHAS="${ALPHAS:-1.0}"
  export TOP_KS="${TOP_KS:-1}"
  export LRS="${LRS:-0.0005 0.001 0.002 0.004 0.008 0.015 0.02 0.04}"
  export STAMP="$BASE_STAMP"
  bash scripts/run_d12_sweep.sh "$@"
  exit 0
fi

run_group \
  "plain_muon" \
  "plain_muon" \
  "${PLAIN_MUON_LRS:-0.0025 0.005 0.0075 0.01 0.015 0.02 0.03 0.04 0.06 0.08}" \
  "${PLAIN_MUON_LR_MIN:-0.0025}" \
  "${PLAIN_MUON_LR_MAX:-0.16}" \
  "$@"

run_group \
  "adamw" \
  "adamw" \
  "${ADAMW_LRS:-0.00025 0.0005 0.001 0.002 0.003 0.004 0.005 0.006 0.0075}" \
  "${ADAMW_LR_MIN:-0.00003125}" \
  "${ADAMW_LR_MAX:-0.016}" \
  "$@"

run_group \
  "soap_klsoap" \
  "soap kl_soap" \
  "${SOAP_LRS:-0.001 0.002 0.003 0.004 0.005 0.006 0.008 0.012 0.016 0.024}" \
  "${SOAP_LR_MIN:-0.0005}" \
  "${SOAP_LR_MAX:-0.064}" \
  "$@"

run_group \
  "kl_shampoo" \
  "kl_shampoo" \
  "${KL_SHAMPOO_LRS:-0.001 0.002 0.004 0.005 0.006 0.008 0.012 0.016 0.024}" \
  "${KL_SHAMPOO_LR_MIN:-0.0005}" \
  "${KL_SHAMPOO_LR_MAX:-0.08}" \
  "$@"

run_group \
  "shampoo" \
  "shampoo" \
  "${SHAMPOO_LRS:-0.0005 0.001 0.002 0.004 0.005 0.006 0.008 0.012 0.016 0.02 0.024}" \
  "${SHAMPOO_LR_MIN:-0.000125}" \
  "${SHAMPOO_LR_MAX:-0.032}" \
  "$@"
