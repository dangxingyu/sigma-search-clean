#!/usr/bin/env bash
set -euo pipefail

# Merlin container entrypoint for sigma-search sweeps.
#
# This script intentionally delegates all experiment logic to the normal repo
# wrappers. It only normalizes the runtime environment that Merlin jobs need.

if [[ -n "${SIGMA_SEARCH_REPO:-}" ]]; then
  REPO="$SIGMA_SEARCH_REPO"
else
  REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$REPO"

if [[ -f "$REPO/nanochat/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$REPO/nanochat/.venv/bin/activate"
fi

if [[ "${MERLIN_SETUP_ENV:-1}" == "1" ]]; then
  if ! python - <<'PY' >/dev/null 2>&1
import pyarrow
import rustbpe
import tiktoken
import tokenizers
import torch
PY
  then
    if ! command -v uv >/dev/null 2>&1; then
      python3 -m pip install --user uv
      export PATH="$HOME/.local/bin:$PATH"
    fi
    (cd "$REPO/nanochat" && uv sync --extra gpu)
    # shellcheck disable=SC1091
    source "$REPO/nanochat/.venv/bin/activate"
  fi
fi

export PYTHONPATH="${PYTHONPATH:-$REPO:$REPO/nanochat}"
case ":$PYTHONPATH:" in
  *":$REPO:"*) ;;
  *) export PYTHONPATH="$REPO:$PYTHONPATH" ;;
esac
case ":$PYTHONPATH:" in
  *":$REPO/nanochat:"*) ;;
  *) export PYTHONPATH="$REPO/nanochat:$PYTHONPATH" ;;
esac

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TORCH_NCCL_AVOID_RECORD_STREAMS="${TORCH_NCCL_AVOID_RECORD_STREAMS:-True}"

STAMP="${STAMP:-merlin_sweep}"
if [[ -n "${MERLIN_OUTPUT_BASE:-}" ]]; then
  export OUT_ROOT="${OUT_ROOT:-$MERLIN_OUTPUT_BASE/search_evals/$STAMP}"
  export LOG_ROOT="${LOG_ROOT:-$MERLIN_OUTPUT_BASE/logs/$STAMP}"
else
  export OUT_ROOT="${OUT_ROOT:-$REPO/search_evals/$STAMP}"
  export LOG_ROOT="${LOG_ROOT:-$REPO/logs/$STAMP}"
fi

SUMMARY_ONLY="${SUMMARY_ONLY:-0}"
MERLIN_SKIP_DATA_CHECK="${MERLIN_SKIP_DATA_CHECK:-0}"
if [[ "$SUMMARY_ONLY" != "1" && "$MERLIN_SKIP_DATA_CHECK" != "1" ]]; then
  if [[ -z "${NANOCHAT_BASE_DIR:-}" ]]; then
    echo "NANOCHAT_BASE_DIR must point to prepared nanochat data." >&2
    exit 2
  fi
  if [[ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]]; then
    echo "Missing tokenizer: $NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" >&2
    exit 2
  fi
  if [[ ! -f "$NANOCHAT_BASE_DIR/tokenizer/token_bytes.pt" ]]; then
    echo "Missing token bytes: $NANOCHAT_BASE_DIR/tokenizer/token_bytes.pt" >&2
    exit 2
  fi
  if ! compgen -G "$NANOCHAT_BASE_DIR/base_data_climbmix/*.parquet" >/dev/null; then
    echo "Missing ClimbMix parquet shards under $NANOCHAT_BASE_DIR/base_data_climbmix" >&2
    exit 2
  fi
fi

MERLIN_SWEEP_SCRIPT="${MERLIN_SWEEP_SCRIPT:-scripts/run_d12_sweep.sh}"
case_index="${MERLIN_CASE_INDEX:-${CASE_INDEX:-}}"

cmd=(bash "$MERLIN_SWEEP_SCRIPT")
if [[ -n "$case_index" ]]; then
  cmd+=(--case-index "$case_index")
fi
if [[ -n "${MERLIN_SWEEP_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_args=($MERLIN_SWEEP_ARGS)
  cmd+=("${extra_args[@]}")
fi

echo "Merlin sigma-search entrypoint"
echo "REPO=$REPO"
echo "NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-}"
echo "OUT_ROOT=$OUT_ROOT"
echo "LOG_ROOT=$LOG_ROOT"
echo "STAMP=$STAMP"
echo "MERLIN_SWEEP_SCRIPT=$MERLIN_SWEEP_SCRIPT"
if [[ -n "$case_index" ]]; then
  echo "CASE_INDEX=$case_index"
fi
printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
