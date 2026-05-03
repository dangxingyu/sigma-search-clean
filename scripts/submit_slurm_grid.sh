#!/usr/bin/env bash
set -euo pipefail

# Submit or run one SLURM-array task for the canonical optimizer-quality grid.
#
# Submit mode:
#   STAMP=d12_main_001 MAX_PARALLEL=8 bash scripts/submit_slurm_grid.sh
#
# Task mode is used internally by sbatch. To debug one task inside an
# allocation:
#   GRID_TASK_MODE=1 SLURM_ARRAY_TASK_ID=0 STAMP=d12_main_001 bash scripts/submit_slurm_grid.sh

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

DEPTH="${DEPTH:-12}"
CHINCHILLA_MULT="${CHINCHILLA_MULT:-2}"
TOKENS="${TOKENS:-}"
METHODS="${METHODS:-top_aware_muon}"
BATCHES="${BATCHES:-262144 1048576 4194304}"
ALPHAS="${ALPHAS:-1.0 0.5}"
TOP_KS="${TOP_KS:-1}"
LRS="${LRS:-0.005 0.01 0.02 0.04}"
SEEDS="${SEEDS:-42}"

NPROC="${NPROC:-8}"
MAX_DEVICE_BATCH_SIZE="${MAX_DEVICE_BATCH_SIZE:-16}"
SAVE_EVERY="${SAVE_EVERY:-100}"
KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-2}"
RESUME="${RESUME:-1}"
ADAPTIVE_LR="${ADAPTIVE_LR:-1}"

STAMP="${STAMP:-d12_grid_${SLURM_ARRAY_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
OUT_ROOT="${OUT_ROOT:-search_evals/${STAMP}}"
LOG_ROOT="${LOG_ROOT:-logs/${STAMP}}"

# Cluster-specific knobs. Override these at submit time if her cluster uses
# different names, account, partition, or GPU syntax.
MAX_PARALLEL="${MAX_PARALLEL:-4}"
SBATCH_JOB_NAME="${SBATCH_JOB_NAME:-sigma-grid}"
SBATCH_TIME="${SBATCH_TIME:-24:00:00}"
SBATCH_PARTITION="${SBATCH_PARTITION:-}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-}"
SBATCH_QOS="${SBATCH_QOS:-}"
SBATCH_CPUS_PER_TASK="${SBATCH_CPUS_PER_TASK:-16}"
SBATCH_MEM="${SBATCH_MEM:-}"
SBATCH_GPU_ARG="${SBATCH_GPU_ARG:---gres=gpu:${NPROC}}"
SBATCH_EXTRA_ARGS="${SBATCH_EXTRA_ARGS:-}"

if [[ "${GRID_TASK_MODE:-0}" == "1" || -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "SLURM_ARRAY_TASK_ID is required in GRID_TASK_MODE=1" >&2
    exit 2
  fi
  echo "Running grid case index ${SLURM_ARRAY_TASK_ID} under STAMP=${STAMP}"
  bash scripts/run_d12_sweep.sh --case-index "$SLURM_ARRAY_TASK_ID"
  exit
fi

count_cmd=(
  python run_top_aware_muon_sweep.py
  --out-root "$OUT_ROOT"
  --log-root "$LOG_ROOT"
  --nanochat-dir nanochat
  --methods "$METHODS"
  --batches "$BATCHES"
  --alphas "$ALPHAS"
  --top-ks "$TOP_KS"
  --lrs "$LRS"
  --seeds "$SEEDS"
  --depth "$DEPTH"
  --chinchilla-mult "$CHINCHILLA_MULT"
  --nproc-per-node "$NPROC"
  --max-device-batch-size "$MAX_DEVICE_BATCH_SIZE"
  --save-every "$SAVE_EVERY"
  --keep-last-checkpoints "$KEEP_LAST_CHECKPOINTS"
  --pure-qr
  --streaming-num-iters "${STREAMING_NUM_ITERS:-2}"
  --fallback-ortho-tol "${FALLBACK_ORTHO_TOL:-0.01}"
  --metrics-every "${METRICS_EVERY:-0}"
  --metrics-hessian-every "${METRICS_HESSIAN_EVERY:-0}"
  --print-case-count
)
if [[ -n "$TOKENS" ]]; then
  count_cmd+=(--tokens "$TOKENS")
fi
if [[ "$ADAPTIVE_LR" == "1" ]]; then
  count_cmd+=(--adaptive-lr)
fi
if [[ "${ALLOW_TOP_K_SWEEP:-0}" == "1" ]]; then
  count_cmd+=(--allow-top-k-sweep)
fi
case_count="$("${count_cmd[@]}")"
if [[ "$case_count" -le 0 ]]; then
  echo "No grid cases to submit." >&2
  exit 2
fi
array_max=$((case_count - 1))

mkdir -p "$LOG_ROOT"

export DEPTH CHINCHILLA_MULT TOKENS METHODS BATCHES ALPHAS TOP_KS LRS SEEDS
export NPROC MAX_DEVICE_BATCH_SIZE SAVE_EVERY KEEP_LAST_CHECKPOINTS RESUME ADAPTIVE_LR
export STAMP OUT_ROOT LOG_ROOT GRID_TASK_MODE=1
export STREAMING_NUM_ITERS="${STREAMING_NUM_ITERS:-2}"
export FALLBACK_ORTHO_TOL="${FALLBACK_ORTHO_TOL:-0.01}"
export METRICS_EVERY="${METRICS_EVERY:-0}"
export METRICS_HESSIAN_EVERY="${METRICS_HESSIAN_EVERY:-0}"

sbatch_args=(
  --job-name "$SBATCH_JOB_NAME"
  --nodes 1
  --ntasks 1
  --cpus-per-task "$SBATCH_CPUS_PER_TASK"
  --time "$SBATCH_TIME"
  --array "0-${array_max}%${MAX_PARALLEL}"
  --output "${LOG_ROOT}/slurm_%A_%a.out"
  --export ALL
)
if [[ -n "$SBATCH_PARTITION" ]]; then
  sbatch_args+=(--partition "$SBATCH_PARTITION")
fi
if [[ -n "$SBATCH_ACCOUNT" ]]; then
  sbatch_args+=(--account "$SBATCH_ACCOUNT")
fi
if [[ -n "$SBATCH_QOS" ]]; then
  sbatch_args+=(--qos "$SBATCH_QOS")
fi
if [[ -n "$SBATCH_MEM" ]]; then
  sbatch_args+=(--mem "$SBATCH_MEM")
fi
if [[ -n "$SBATCH_GPU_ARG" ]]; then
  read -r -a gpu_args <<< "$SBATCH_GPU_ARG"
  sbatch_args+=("${gpu_args[@]}")
fi
if [[ -n "$SBATCH_EXTRA_ARGS" ]]; then
  read -r -a extra_args <<< "$SBATCH_EXTRA_ARGS"
  sbatch_args+=("${extra_args[@]}")
fi

echo "Submitting ${case_count} grid cases as SLURM array 0-${array_max}%${MAX_PARALLEL}"
echo "STAMP=${STAMP}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "LOG_ROOT=${LOG_ROOT}"
echo "After the array finishes, run:"
echo "  STAMP=${STAMP} bash scripts/run_d12_sweep.sh"
echo "This collates the CSV, skips completed base-grid cases, and runs adaptive LR boundary closure if enabled."
printf 'sbatch command:'
printf ' %q' sbatch "${sbatch_args[@]}" "$0"
printf '\n'

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi
sbatch "${sbatch_args[@]}" "$0"
