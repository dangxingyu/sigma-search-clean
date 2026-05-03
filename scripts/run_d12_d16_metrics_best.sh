#!/usr/bin/env bash
set -euo pipefail

# Curated d12/d16 2x-Chinchilla metrics jobs from the imported sweeps.
#
# These are fixed-recipe diagnostics, not an LR sweep. Each row is the best LR
# for one (depth, batch, alpha) pair from results/d12-d16-sweep/.
#
# Caveat: d16 512K/2M are not fully LR-closed because some best rows hit
# lr=0.005, the lower boundary of the imported sweep. Update those cases after
# the d16 lower-LR extension if using them for final claims.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

CHINCHILLA_MULT="${CHINCHILLA_MULT:-2}"
SEEDS="${SEEDS:-42}"
NPROC="${NPROC:-8}"
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

STAMP_PREFIX="${STAMP_PREFIX:-d12_d16_2x_best_metrics_$(date +%Y%m%d_%H%M%S)}"

# Format: depth batch alpha lr max_device_batch_size note
CASES=(
  "12 524288 1.0 0.0075 32 d12_closed"
  "12 524288 0.5 0.0075 32 d12_closed"
  "12 2097152 1.0 0.015 32 d12_closed"
  "12 2097152 0.5 0.02 32 d12_closed"
  "12 8388608 1.0 0.01 32 d12_closed"
  "12 8388608 0.5 0.015 32 d12_closed"
  "16 524288 1.0 0.005 16 d16_low_lr_boundary"
  "16 524288 0.5 0.005 16 d16_low_lr_boundary"
  "16 2097152 1.0 0.005 16 d16_low_lr_boundary"
  "16 2097152 0.5 0.0075 16 d16_partial"
  "16 8388608 1.0 0.0075 16 d16_closed_enough"
  "16 8388608 0.5 0.015 16 d16_closed_enough"
)

sanitize() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  printf '%s' "$value"
}

print_cases() {
  printf 'Curated d12/d16 metrics cases from results/d12-d16-sweep:\n'
  printf '  %5s  %5s  %10s  %8s  %10s  %8s  %s\n' \
    index depth batch alpha lr maxdbs note
  local i
  for i in "${!CASES[@]}"; do
    read -r depth batch alpha lr maxdbs note <<<"${CASES[$i]}"
    printf '  %5s  d%-4s  %10s  %8s  %10s  %8s  %s\n' \
      "$i" "$depth" "$batch" "$alpha" "$lr" "$maxdbs" "$note"
  done
}

run_case() {
  local index="$1"
  shift
  if (( index < 0 || index >= ${#CASES[@]} )); then
    printf 'Invalid CASE_INDEX=%s; valid range is 0..%s\n' "$index" "$((${#CASES[@]} - 1))" >&2
    exit 2
  fi

  local depth batch alpha lr maxdbs note label stamp
  read -r depth batch alpha lr maxdbs note <<<"${CASES[$index]}"
  label="d${depth}_bsz${batch}_a$(sanitize "$alpha")_lr$(sanitize "$lr")"
  stamp="${STAMP_PREFIX}_${label}"

  printf '\n=== metrics case %s/%s: d%s batch=%s alpha=%s lr=%s max_device_batch=%s note=%s ===\n' \
    "$index" "$((${#CASES[@]} - 1))" "$depth" "$batch" "$alpha" "$lr" "$maxdbs" "$note"
  printf 'STAMP=%s\n' "$stamp"

  STAMP="$stamp" \
  DEPTH="$depth" \
  CHINCHILLA_MULT="$CHINCHILLA_MULT" \
  METHODS="top_aware_muon" \
  BATCHES="$batch" \
  ALPHAS="$alpha" \
  LRS="$lr" \
  SEEDS="$SEEDS" \
  NPROC="$NPROC" \
  MAX_DEVICE_BATCH_SIZE="$maxdbs" \
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
