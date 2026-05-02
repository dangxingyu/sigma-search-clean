#!/usr/bin/env bash
set -euo pipefail

# Optional SLURM submitter for the d12 handoff sweep.
#
# This script creates one sbatch job per sweep shard. Each shard gets a stable
# STAMP so preempted jobs can be requeued or resubmitted safely.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

CAMPAIGN="${CAMPAIGN:-d12_c001}"
SHARD_BY="${SHARD_BY:-batch}"  # batch | batch_alpha | batch_lr | batch_alpha_lr

METHODS="${METHODS:-streaming_identity top_aware_muon}"
BATCHES="${BATCHES:-262144 1048576 4194304}"
ALPHAS="${ALPHAS:-0.5}"
LRS="${LRS:-0.005 0.01 0.02 0.04}"
SEEDS="${SEEDS:-42}"

TOKENS="${TOKENS:-977272832}"
DEPTH="${DEPTH:-12}"
NPROC="${NPROC:-8}"
MAX_DEVICE_BATCH_SIZE="${MAX_DEVICE_BATCH_SIZE:-16}"
SAVE_EVERY="${SAVE_EVERY:-100}"
KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-2}"
RESUME="${RESUME:-1}"
ADAPTIVE_LR="${ADAPTIVE_LR:-1}"

SBATCH_JOB_PREFIX="${SBATCH_JOB_PREFIX:-sigma-d12}"
SBATCH_TIME="${SBATCH_TIME:-24:00:00}"
SBATCH_NODES="${SBATCH_NODES:-1}"
SBATCH_NTASKS="${SBATCH_NTASKS:-1}"
SBATCH_CPUS_PER_TASK="${SBATCH_CPUS_PER_TASK:-32}"
SBATCH_MEM="${SBATCH_MEM:-}"
SBATCH_GPU_DIRECTIVE="${SBATCH_GPU_DIRECTIVE:---gpus=8}"
SBATCH_PARTITION="${SBATCH_PARTITION:-}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-}"
SBATCH_QOS="${SBATCH_QOS:-}"
SBATCH_EXTRA_DIRECTIVES="${SBATCH_EXTRA_DIRECTIVES:-}"

SUBMIT_DIR="${SUBMIT_DIR:-logs/slurm_submit/${CAMPAIGN}}"
SUBMIT_DRY_RUN="${SUBMIT_DRY_RUN:-0}"

split_words() {
  local spec="$1"
  spec="${spec//,/ }"
  # shellcheck disable=SC2206
  WORDS_OUT=($spec)
}

slug() {
  local x="$1"
  x="${x//./p}"
  x="${x//-/m}"
  x="${x// /_}"
  printf '%s' "$x"
}

batch_slug() {
  local b="$1"
  if [[ "$b" == "262144" ]]; then
    printf '262k'
  elif (( b % 1048576 == 0 )); then
    printf '%dm' "$((b / 1048576))"
  elif (( b % 1024 == 0 )); then
    printf '%dk' "$((b / 1024))"
  else
    printf '%s' "$b"
  fi
}

write_job_script() {
  local stamp="$1"
  local batch_spec="$2"
  local alpha_spec="$3"
  local lr_spec="$4"
  local job_script="$SUBMIT_DIR/${stamp}.sbatch"
  mkdir -p "$SUBMIT_DIR"

  {
    printf '#!/usr/bin/env bash\n'
    printf '#SBATCH --job-name=%s_%s\n' "$SBATCH_JOB_PREFIX" "$stamp"
    printf '#SBATCH --output=logs/slurm_%%x_%%j.out\n'
    printf '#SBATCH --error=logs/slurm_%%x_%%j.err\n'
    printf '#SBATCH --time=%s\n' "$SBATCH_TIME"
    printf '#SBATCH --nodes=%s\n' "$SBATCH_NODES"
    printf '#SBATCH --ntasks=%s\n' "$SBATCH_NTASKS"
    printf '#SBATCH --cpus-per-task=%s\n' "$SBATCH_CPUS_PER_TASK"
    if [[ -n "$SBATCH_MEM" ]]; then
      printf '#SBATCH --mem=%s\n' "$SBATCH_MEM"
    fi
    if [[ -n "$SBATCH_GPU_DIRECTIVE" ]]; then
      printf '#SBATCH %s\n' "$SBATCH_GPU_DIRECTIVE"
    fi
    if [[ -n "$SBATCH_PARTITION" ]]; then
      printf '#SBATCH --partition=%s\n' "$SBATCH_PARTITION"
    fi
    if [[ -n "$SBATCH_ACCOUNT" ]]; then
      printf '#SBATCH --account=%s\n' "$SBATCH_ACCOUNT"
    fi
    if [[ -n "$SBATCH_QOS" ]]; then
      printf '#SBATCH --qos=%s\n' "$SBATCH_QOS"
    fi
    if [[ "${SBATCH_REQUEUE:-1}" == "1" ]]; then
      printf '#SBATCH --requeue\n'
    fi
    if [[ -n "$SBATCH_EXTRA_DIRECTIVES" ]]; then
      while IFS= read -r directive; do
        [[ -n "$directive" ]] && printf '#SBATCH %s\n' "$directive"
      done <<< "$SBATCH_EXTRA_DIRECTIVES"
    fi
    printf '\nset -euo pipefail\n'
    printf 'cd %q\n' "$REPO"
    printf 'export STAMP=%q\n' "$stamp"
    printf 'export METHODS=%q\n' "$METHODS"
    printf 'export BATCHES=%q\n' "$batch_spec"
    printf 'export ALPHAS=%q\n' "$alpha_spec"
    printf 'export LRS=%q\n' "$lr_spec"
    printf 'export SEEDS=%q\n' "$SEEDS"
    printf 'export TOKENS=%q\n' "$TOKENS"
    printf 'export DEPTH=%q\n' "$DEPTH"
    printf 'export NPROC=%q\n' "$NPROC"
    printf 'export MAX_DEVICE_BATCH_SIZE=%q\n' "$MAX_DEVICE_BATCH_SIZE"
    printf 'export SAVE_EVERY=%q\n' "$SAVE_EVERY"
    printf 'export KEEP_LAST_CHECKPOINTS=%q\n' "$KEEP_LAST_CHECKPOINTS"
    printf 'export RESUME=%q\n' "$RESUME"
    printf 'export ADAPTIVE_LR=%q\n' "$ADAPTIVE_LR"
    printf '\nbash scripts/run_d12_sweep.sh\n'
  } > "$job_script"

  printf '%s\n' "$job_script"
}

submit_job() {
  local stamp="$1"
  local batch_spec="$2"
  local alpha_spec="$3"
  local lr_spec="$4"
  local job_script
  job_script="$(write_job_script "$stamp" "$batch_spec" "$alpha_spec" "$lr_spec")"

  if [[ "$SUBMIT_DRY_RUN" == "1" ]]; then
    printf '[dry-run] sbatch %s\n' "$job_script"
  else
    sbatch "$job_script"
  fi
}

split_words "$BATCHES"; batches=("${WORDS_OUT[@]}")
split_words "$ALPHAS"; alphas=("${WORDS_OUT[@]}")
split_words "$LRS"; lrs=("${WORDS_OUT[@]}")

if [[ "$SHARD_BY" != "batch" && "$METHODS" == *"streaming_identity"* && "${#alphas[@]}" -gt 1 ]]; then
  printf 'WARNING: SHARD_BY=%s with multiple ALPHAS will duplicate streaming_identity baselines. Consider METHODS=top_aware_muon plus a separate identity batch shard.\n' "$SHARD_BY" >&2
fi

case "$SHARD_BY" in
  batch)
    for batch in "${batches[@]}"; do
      submit_job "${CAMPAIGN}_b$(batch_slug "$batch")" "$batch" "$ALPHAS" "$LRS"
    done
    ;;
  batch_alpha)
    for batch in "${batches[@]}"; do
      for alpha in "${alphas[@]}"; do
        submit_job "${CAMPAIGN}_b$(batch_slug "$batch")_a$(slug "$alpha")" "$batch" "$alpha" "$LRS"
      done
    done
    ;;
  batch_lr)
    for batch in "${batches[@]}"; do
      for lr in "${lrs[@]}"; do
        submit_job "${CAMPAIGN}_b$(batch_slug "$batch")_lr$(slug "$lr")" "$batch" "$ALPHAS" "$lr"
      done
    done
    ;;
  batch_alpha_lr)
    for batch in "${batches[@]}"; do
      for alpha in "${alphas[@]}"; do
        for lr in "${lrs[@]}"; do
          submit_job "${CAMPAIGN}_b$(batch_slug "$batch")_a$(slug "$alpha")_lr$(slug "$lr")" "$batch" "$alpha" "$lr"
        done
      done
    done
    ;;
  *)
    printf 'Unknown SHARD_BY=%s. Use batch, batch_alpha, batch_lr, or batch_alpha_lr.\n' "$SHARD_BY" >&2
    exit 2
    ;;
esac
