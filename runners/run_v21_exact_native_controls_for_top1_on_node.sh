#!/bin/bash
# v21: exact native controls for the top1-damp StreamingMuon sweep.
#
# This isolates whether the StreamingMuon identity gap is a tol/optimizer issue
# or a recipe/eval/DDP issue by running native Muon under the exact top1 recipe.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$SCRIPT_DIR}"
OUT_ROOT="$REPO/sweep_results_v21_exact_native_controls_top1"
mkdir -p "$OUT_ROOT/logs"

WAIT_FOR_IDLE="${WAIT_FOR_IDLE:-1}"
DEPTH=8
SEQ=1024
DEV_BSZ=16
TOKENS=1073741824
SEED="${SEED:-42}"

if [ "$WAIT_FOR_IDLE" = "1" ]; then
    echo "[wait] waiting for all GPUs to be idle enough for DDP native controls"
    while true; do
        busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2000 {n += 1} END {print n+0}')
        if [ "$busy" -eq 0 ]; then
            break
        fi
        sleep 30
    done
fi

run_native() {
    local bsz="$1" lr="$2"
    local steps=$((TOKENS / bsz))
    local warmup
    warmup=$(python - <<PY
print(round(0.05 * ${steps}))
PY
)
    if [ "$warmup" -lt 1 ]; then warmup=1; fi
    local eval_every
    eval_every=$(python - <<PY
print(max(1, ${steps} // 8))
PY
)
    local tag="native_muon_bsz${bsz}_lr${lr}_s${SEED}"
    local out_json="$OUT_ROOT/${tag}.json"
    local out_log="$OUT_ROOT/logs/${tag}.log"
    if [ -f "$out_json" ]; then
        echo "[skip] $tag"
        return
    fi
    echo "[run $(date +%H:%M:%S)] $tag steps=$steps warmup=$warmup eval_every=$eval_every"
    cd "$REPO"
    source "$REPO/nanochat/.venv/bin/activate"
    torchrun --standalone --nproc_per_node=8 run_native_muon_match.py \
        --depth "$DEPTH" --max-seq-len "$SEQ" \
        --device-batch-size "$DEV_BSZ" \
        --total-batch-size "$bsz" \
        --max-steps "$steps" \
        --matrix-lr "$lr" \
        --warmup-steps "$warmup" \
        --warmdown-ratio 0.65 \
        --eval-every "$eval_every" \
        --eval-tokens 524288 \
        --output-file "$out_json" \
        --seed "$SEED" > "$out_log" 2>&1
}

# Match top1 identity best-LR points.
run_native 131072 0.01
run_native 1048576 0.01
run_native 8388608 0.02

python - <<'PY'
import json
from pathlib import Path
root = Path("sweep_results_v21_exact_native_controls_top1")
for p in sorted(root.glob("*.json")):
    d = json.loads(p.read_text())
    print(f"{p.name}: score={d.get('score')} error={d.get('error')}")
PY
