#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO/nanochat"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it first, then rerun this script." >&2
  echo "See: https://docs.astral.sh/uv/" >&2
  exit 1
fi

uv sync --extra gpu --group dev

cat <<EOF

Environment ready.

Activate with:
  source "$REPO/nanochat/.venv/bin/activate"

Recommended runtime exports:
  export PYTHONPATH="$REPO:$REPO/nanochat"
  export NANOCHAT_BASE_DIR="\${NANOCHAT_BASE_DIR:-\$HOME/.cache/nanochat}"
EOF
