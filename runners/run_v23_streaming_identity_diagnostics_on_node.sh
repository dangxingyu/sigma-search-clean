#!/bin/bash
# v23: StreamingMuon identity diagnostics if exact native DDP controls disagree.
#
# Runs only the 1M/lr=0.01 recipe, because this is a diagnostic for the
# StreamingMuon identity implementation rather than a new sweep.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$SCRIPT_DIR}"
OUT_ROOT="$REPO/search_evals/v23_streaming_identity_diagnostics"
LOG_ROOT="$REPO/logs/v23_streaming_identity_diagnostics"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

DEPTH=8
SEQ=1024
DEV_BSZ=16
BSZ=1048576
STEPS=1024
WARMUP=51
EVAL_EVERY=128
SEED="${SEED:-42}"
LR=0.01

run_case() {
    local tag="$1"
    shift
    local out="$OUT_ROOT/${tag}/result.json"
    local log="$LOG_ROOT/${tag}.log"
    if [ -f "$out" ]; then
        echo "[skip] $tag"
        return
    fi
    mkdir -p "$OUT_ROOT/${tag}"
    echo "[run $(date +%H:%M:%S)] $tag $*"
    cd "$REPO"
    source "$REPO/nanochat/.venv/bin/activate"
    torchrun --standalone --nproc_per_node=8 run_eval.py \
        --candidate-file candidates/identity.py \
        --output-file "$out" \
        --depth "$DEPTH" --max-seq-len "$SEQ" \
        --device-batch-size "$DEV_BSZ" \
        --total-batch-size "$BSZ" \
        --max-steps "$STEPS" \
        --eval-every "$EVAL_EVERY" \
        --eval-tokens 524288 \
        --matrix-lr "$LR" \
        --warmup-steps "$WARMUP" \
        --fallback-ortho-tol 0.01 \
        --seed "$SEED" \
        "$@" > "$log" 2>&1
}

run_case identity_pureqr --pure-qr
run_case identity_pureqr_2iter --pure-qr --num-iters 2
run_case identity_scqr_2iter --num-iters 2
run_case identity_inputnorm --input-normalize

python - <<'PY'
import json
from pathlib import Path
root = Path("search_evals/v23_streaming_identity_diagnostics")
for p in sorted(root.glob("*/result.json")):
    d = json.loads(p.read_text())
    print(f"{p.parent.name}: score={d.get('score')} error={d.get('error')}")
PY
