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

export UV_PYTHON_DOWNLOADS="${UV_PYTHON_DOWNLOADS:-never}"
export UV_PYTHON="${UV_PYTHON:-python3}"
export UV_PYTHON_PREFERENCE="${UV_PYTHON_PREFERENCE:-only-system}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export UV_INDEX_URL="${UV_INDEX_URL:-https://bytedpypi.byted.org/simple}"
export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://bytedpypi.byted.org/simple}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://bytedpypi.byted.org/simple}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-bytedpypi.byted.org}"

if [[ -n "${HDFS_RUNTIME_TGZ:-}" && ! -x "$REPO/nanochat/.venv/bin/python" ]]; then
  echo "Restoring runtime from HDFS_RUNTIME_TGZ=$HDFS_RUNTIME_TGZ"
  rm -rf "$REPO/nanochat/.venv"
  hdfs dfs -get "$HDFS_RUNTIME_TGZ" /tmp/sigma_runtime.tgz
  mkdir -p "$REPO/nanochat"
  tar -xzf /tmp/sigma_runtime.tgz -C "$REPO/nanochat"
fi

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

    NANOCHAT_DIR="$REPO/nanochat"
    if [[ -f "$NANOCHAT_DIR/pyproject.toml" ]]; then
      python3 - <<'PY'
from pathlib import Path
import re
p = Path("$NANOCHAT_DIR/pyproject.toml")
text = p.read_text()
text = text.replace('    "torch==2.9.1",\n', '')
text = re.sub(r'\n# target torch to cuda 12\.8 or CPU\n\[tool\.uv\.sources\]\n(?:.*\n)*?\n\[\[tool\.uv\.index\]\]\nname = "pytorch-cpu"\nurl = "https://download\.pytorch\.org/whl/cpu"\nexplicit = true\n\n\[\[tool\.uv\.index\]\]\nname = "pytorch-cu128"\nurl = "https://download\.pytorch\.org/whl/cu128"\nexplicit = true\n', '\n', text, count=1)
p.write_text(text)
PY
      rm -f "$NANOCHAT_DIR/uv.lock"
    fi

    rm -rf "$REPO/nanochat/.venv"
    (cd "$REPO/nanochat" && uv venv --system-site-packages && uv sync)
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
case_indices="${MERLIN_CASE_INDICES:-}"

base_cmd=(bash "$MERLIN_SWEEP_SCRIPT")
if [[ -n "${MERLIN_SWEEP_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_args=($MERLIN_SWEEP_ARGS)
  base_cmd+=("${extra_args[@]}")
fi

echo "Merlin sigma-search entrypoint"
echo "REPO=$REPO"
echo "NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-}"
echo "OUT_ROOT=$OUT_ROOT"
echo "LOG_ROOT=$LOG_ROOT"
echo "STAMP=$STAMP"
echo "MERLIN_SWEEP_SCRIPT=$MERLIN_SWEEP_SCRIPT"
if [[ -n "$case_indices" ]]; then
  echo "CASE_INDICES=$case_indices"
fi
if [[ -n "$case_index" ]]; then
  echo "CASE_INDEX=$case_index"
fi

parallel_cases="${MERLIN_PARALLEL_CASES:-1}"
gpus_per_case="${MERLIN_GPUS_PER_CASE:-${NPROC:-1}}"
if [[ -n "$case_indices" && "$parallel_cases" -gt 1 ]]; then
  running=0
  slot=0
  pids=()
  for idx in $case_indices; do
    gpu_start=$(((slot % parallel_cases) * gpus_per_case))
    gpu_ids=""
    for ((i = 0; i < gpus_per_case; i++)); do
      gpu_id=$((gpu_start + i))
      if [[ -z "$gpu_ids" ]]; then
        gpu_ids="$gpu_id"
      else
        gpu_ids="$gpu_ids,$gpu_id"
      fi
    done
    (
      export CUDA_VISIBLE_DEVICES="$gpu_ids"
      cmd=("${base_cmd[@]}" --case-index "$idx")
      printf 'Parallel command on GPUs %s:' "$gpu_ids"
      printf ' %q' "${cmd[@]}"
      printf '\n'
      "${cmd[@]}"
    ) &
    pids+=("$!")
    running=$((running + 1))
    slot=$((slot + 1))
    if [[ "$running" -ge "$parallel_cases" ]]; then
      wait "${pids[0]}"
      pids=("${pids[@]:1}")
      running=$((running - 1))
    fi
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
elif [[ -n "$case_indices" ]]; then
  for idx in $case_indices; do
    cmd=("${base_cmd[@]}" --case-index "$idx")
    printf 'Command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    "${cmd[@]}"
  done
elif [[ -n "$case_index" ]]; then
  cmd=("${base_cmd[@]}" --case-index "$case_index")
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"
else
  printf 'Command:'
  printf ' %q' "${base_cmd[@]}"
  printf '\n'
  "${base_cmd[@]}"
fi
