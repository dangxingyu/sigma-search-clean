#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHARDS="${1:-170}"
WORKERS="${2:-8}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"

PY="$REPO/nanochat/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="python"
fi

cd "$REPO/nanochat"
"$PY" -m nanochat.dataset -n "$SHARDS" -w "$WORKERS"

cat <<EOF

Dataset download attempted.
Data directory:
  $NANOCHAT_BASE_DIR/base_data_climbmix

Usage:
  scripts/download_climbmix.sh 170 8

The validation shard is always downloaded by nanochat.dataset.
EOF
