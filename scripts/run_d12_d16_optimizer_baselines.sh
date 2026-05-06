#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper that runs d12 then d16 non-streaming optimizer baselines.
#
# Prefer run_d12_optimizer_baselines.sh and run_d16_optimizer_baselines.sh when
# launching jobs separately on a cluster.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

base_stamp="${STAMP_PREFIX:-optimizer_baselines_$(date +%Y%m%d_%H%M%S)}"

STAMP="${base_stamp}_d12_2x" bash scripts/run_d12_optimizer_baselines.sh "$@"
STAMP="${base_stamp}_d16_2x" bash scripts/run_d16_optimizer_baselines.sh "$@"
