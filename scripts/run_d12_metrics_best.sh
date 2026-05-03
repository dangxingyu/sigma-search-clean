#!/usr/bin/env bash
set -euo pipefail

# Recommended d12 2x-Chinchilla metrics jobs from the completed sweep.
#
# These are fixed-recipe diagnostics, not an LR sweep. Each row is the best LR
# for one (batch, alpha) pair from sweep_prefix_002_d12_2x.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

DEPTH="${DEPTH:-12}"
CHINCHILLA_MULT="${CHINCHILLA_MULT:-2}"
SEEDS="${SEEDS:-42}"

NPROC="${NPROC:-8}"
MAX_DEVICE_BATCH_SIZE="${MAX_DEVICE_BATCH_SIZE:-16}"
SAVE_EVERY="${SAVE_EVERY:-100}"
KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-2}"
RESUME="${RESUME:-1}"

METRICS_EVERY="${METRICS_EVERY:-1}"
METRICS_HESSIAN_EVERY="${METRICS_HESSIAN_EVERY:-50}"
METRICS_HESSIAN_TOP_K="${METRICS_HESSIAN_TOP_K:-4}"
METRICS_HESSIAN_ITERS="${METRICS_HESSIAN_ITERS:-6}"
METRICS_MAX_MODULES="${METRICS_MAX_MODULES:-0}"
METRICS_HESSIAN_MAX_MODULES="${METRICS_HESSIAN_MAX_MODULES:-0}"
METRICS_PROJECTION_CORRELATION_WINDOW="${METRICS_PROJECTION_CORRELATION_WINDOW:-16}"

STAMP_PREFIX="${STAMP_PREFIX:-d12_2x_best_metrics_$(date +%Y%m%d_%H%M%S)}"

# Format: batch alpha lr
CASES=(
  "524288 1.0 0.0075"
  "524288 0.5 0.0075"
  "2097152 1.0 0.015"
  "2097152 0.5 0.02"
  "8388608 1.0 0.01"
  "8388608 0.5 0.015"
)

sanitize() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  printf '%s' "$value"
}

print_cases() {
  printf 'Recommended d12 metrics cases, from sweep_prefix_002_d12_2x:\n'
  printf '  %5s  %10s  %8s  %10s\n' index batch alpha lr
  local i
  for i in "${!CASES[@]}"; do
    read -r batch alpha lr <<<"${CASES[$i]}"
    printf '  %5s  %10s  %8s  %10s\n' "$i" "$batch" "$alpha" "$lr"
  done
}

run_case() {
  local index="$1"
  shift
  if (( index < 0 || index >= ${#CASES[@]} )); then
    printf 'Invalid CASE_INDEX=%s; valid range is 0..%s\n' "$index" "$((${#CASES[@]} - 1))" >&2
    exit 2
  fi

  local batch alpha lr label stamp
  read -r batch alpha lr <<<"${CASES[$index]}"
  label="bsz${batch}_a$(sanitize "$alpha")_lr$(sanitize "$lr")"
  stamp="${STAMP_PREFIX}_${label}"

  printf '\n=== metrics case %s/%s: batch=%s alpha=%s lr=%s ===\n' \
    "$index" "$((${#CASES[@]} - 1))" "$batch" "$alpha" "$lr"
  printf 'STAMP=%s\n' "$stamp"

  STAMP="$stamp" \
  DEPTH="$DEPTH" \
  CHINCHILLA_MULT="$CHINCHILLA_MULT" \
  METHODS="top_aware_muon" \
  BATCHES="$batch" \
  ALPHAS="$alpha" \
  LRS="$lr" \
  SEEDS="$SEEDS" \
  NPROC="$NPROC" \
  MAX_DEVICE_BATCH_SIZE="$MAX_DEVICE_BATCH_SIZE" \
  SAVE_EVERY="$SAVE_EVERY" \
  KEEP_LAST_CHECKPOINTS="$KEEP_LAST_CHECKPOINTS" \
  RESUME="$RESUME" \
  METRICS_EVERY="$METRICS_EVERY" \
  METRICS_HESSIAN_EVERY="$METRICS_HESSIAN_EVERY" \
  METRICS_HESSIAN_TOP_K="$METRICS_HESSIAN_TOP_K" \
  METRICS_HESSIAN_ITERS="$METRICS_HESSIAN_ITERS" \
  METRICS_MAX_MODULES="$METRICS_MAX_MODULES" \
  METRICS_HESSIAN_MAX_MODULES="$METRICS_HESSIAN_MAX_MODULES" \
  METRICS_PROJECTION_CORRELATION_WINDOW="$METRICS_PROJECTION_CORRELATION_WINDOW" \
  bash scripts/run_d12_statistics.sh "$@"
}

if [[ "${PRINT_CASES:-0}" == "1" ]]; then
  print_cases
  exit 0
fi

if [[ -n "${CASE_INDEX:-}" ]]; then
  run_case "$CASE_INDEX" "$@"
else
  print_cases
  printf '\nRunning all cases sequentially. For SLURM arrays, set CASE_INDEX=$SLURM_ARRAY_TASK_ID.\n'
  for i in "${!CASES[@]}"; do
    run_case "$i" "$@"
  done
fi
