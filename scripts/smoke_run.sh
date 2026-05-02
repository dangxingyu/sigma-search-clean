#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/nanochat/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="python"
fi

export PYTHONPATH="$REPO:$REPO/nanochat:${PYTHONPATH:-}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"

cd "$REPO"
"$PY" run_eval.py \
  --candidate-file candidates/identity.py \
  --nanochat-dir "$REPO/nanochat" \
  --output-file smoke_results/identity_smoke.json \
  --depth 1 \
  --max-seq-len 128 \
  --device-batch-size 1 \
  --total-batch-size 128 \
  --max-steps 2 \
  --eval-every 0 \
  --metrics-every 1 \
  --metrics-top-k 1 \
  --metrics-max-modules 1
