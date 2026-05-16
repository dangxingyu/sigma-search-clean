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

# Space-separated group suffixes to run.
#
# Default handoff keeps the competitive/core comparisons:
#   plain_muon: ordinary Muon, using alpha=1 best LR by depth/batch
#   adamw: dense AdamW baseline
#   soap: SOAP candidate
#   kl_soap: KL-SOAP candidate
#   kl_shampoo: KL-Shampoo candidate
#
# Optional group remains implemented but is not a default handoff job:
#   OPTIMIZER_GROUPS="plain_muon adamw soap kl_soap kl_shampoo shampoo"
#
# Do not use the shell variable name GROUPS here; Bash reserves it for the
# current user's Unix group IDs.
OPTIMIZER_GROUPS="${OPTIMIZER_GROUPS:-plain_muon adamw soap kl_soap kl_shampoo}"

contains_word() {
  local needle="$1"
  local haystack="$2"
  [[ " ${haystack} " == *" ${needle} "* ]]
}

sanitize() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  printf '%s' "$value"
}

run_group_body() {
  local suffix="$1"
  local methods="$2"
  local batches="$3"
  local lrs="$4"
  local lr_min="$5"
  local lr_max="$6"
  shift 6

  local stamp="${BASE_STAMP}_${suffix}"
  stamp="${stamp//./p}"
  echo
  echo "=== optimizer baseline group: ${suffix} ==="
  echo "METHODS=${methods}"
  echo "BATCHES=${batches}"
  echo "LRS=${lrs}"
  echo "STAMP=${stamp}"

  DEPTH="$DEPTH" \
  CHINCHILLA_MULT="$CHINCHILLA_MULT" \
  BATCHES="$batches" \
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

run_group() {
  local suffix="$1"
  local methods="$2"
  local lrs="$3"
  local lr_min="$4"
  local lr_max="$5"
  shift 5

  if ! contains_word "$suffix" "$OPTIMIZER_GROUPS"; then
    echo "Skipping group ${suffix}; OPTIMIZER_GROUPS=${OPTIMIZER_GROUPS}"
    return
  fi

  run_group_body "$suffix" "$methods" "$BATCHES" "$lrs" "$lr_min" "$lr_max" "$@"
}

plain_muon_lr_for_batch() {
  local batch="$1"
  if [[ -n "${PLAIN_MUON_LR:-}" ]]; then
    printf '%s\n' "$PLAIN_MUON_LR"
    return
  fi

  # Source: results/alpha-sweep/best_by_depth_batch_alpha.csv, alpha=1.
  case "${DEPTH}:${batch}" in
    12:524288) printf '%s\n' "0.0075" ;;
    12:2097152) printf '%s\n' "0.015" ;;
    12:8388608) printf '%s\n' "0.01" ;;
    16:524288) printf '%s\n' "0.005" ;;
    16:2097152) printf '%s\n' "0.005" ;;
    16:8388608) printf '%s\n' "0.015" ;;
    *)
      echo "No default plain_muon LR for DEPTH=${DEPTH}, batch=${batch}." >&2
      echo "Set PLAIN_MUON_LR=<lr> to run a custom one-LR Muon baseline." >&2
      exit 2
      ;;
  esac
}

run_plain_muon() {
  if ! contains_word "plain_muon" "$OPTIMIZER_GROUPS"; then
    echo "Skipping group plain_muon; OPTIMIZER_GROUPS=${OPTIMIZER_GROUPS}"
    return
  fi

  local batch lr
  for batch in $BATCHES; do
    lr="$(plain_muon_lr_for_batch "$batch")"
    run_group_body \
      "plain_muon_bsz${batch}_lr$(sanitize "$lr")" \
      "plain_muon" \
      "$batch" \
      "$lr" \
      "${PLAIN_MUON_LR_MIN:-0.0005}" \
      "${PLAIN_MUON_LR_MAX:-0.16}" \
      "$@"
  done
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
  export METHODS="${METHODS:-plain_muon adamw soap kl_soap kl_shampoo}"
  export ALPHAS="${ALPHAS:-1.0}"
  export TOP_KS="${TOP_KS:-1}"
  export LRS="${LRS:-0.0005 0.001 0.002 0.004 0.008 0.015 0.02 0.04}"
  export STAMP="$BASE_STAMP"
  bash scripts/run_d12_sweep.sh "$@"
  exit 0
fi

run_plain_muon "$@"

run_group \
  "adamw" \
  "adamw" \
  "${ADAMW_LRS:-0.0005 0.001 0.002 0.003 0.004 0.005 0.0075}" \
  "${ADAMW_LR_MIN:-0.00003125}" \
  "${ADAMW_LR_MAX:-0.016}" \
  "$@"

run_group \
  "soap" \
  "soap" \
  "${SOAP_LRS:-0.001 0.002 0.003 0.004 0.006 0.008 0.012}" \
  "${SOAP_LR_MIN:-0.0005}" \
  "${SOAP_LR_MAX:-0.064}" \
  "$@"

run_group \
  "kl_soap" \
  "kl_soap" \
  "${KL_SOAP_LRS:-0.001 0.002 0.003 0.004 0.006 0.008 0.012}" \
  "${KL_SOAP_LR_MIN:-0.0005}" \
  "${KL_SOAP_LR_MAX:-0.064}" \
  "$@"

run_group \
  "kl_shampoo" \
  "kl_shampoo" \
  "${KL_SHAMPOO_LRS:-0.002 0.004 0.006 0.008 0.012 0.016}" \
  "${KL_SHAMPOO_LR_MIN:-0.0005}" \
  "${KL_SHAMPOO_LR_MAX:-0.08}" \
  "$@"

run_group \
  "shampoo" \
  "shampoo" \
  "${SHAMPOO_LRS:-0.002 0.004 0.005 0.0075 0.01 0.015 0.02}" \
  "${SHAMPOO_LR_MIN:-0.000125}" \
  "${SHAMPOO_LR_MAX:-0.032}" \
  "$@"
