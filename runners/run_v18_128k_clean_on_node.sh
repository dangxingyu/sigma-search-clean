#!/bin/bash
# v18: clean 128K small-batch baseline, Muon vs fixed LITE.
#
# Purpose: test whether fixed chi=2 LITE starts losing below the clean 256K
# tie point. This is normal-scale d=8 / 1B-token training with a small LR grid.
#
# Run inside the active allocation:
#   srun --jobid=29702470 --overlap --ntasks=1 bash run_v18_128k_clean_on_node.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$SCRIPT_DIR}"
NGPUS="${NGPUS:-8}"
TOKENS=1073741824
DEPTH=8
DEV_BSZ=16
SEQ=1024
BSZ=131072
OUT_ROOT="$REPO/sweep_results_v18_128k_clean"
mkdir -p "$OUT_ROOT/logs"

declare -A slot_pid
for ((g=0; g<NGPUS; g++)); do slot_pid[$g]=""; done

launch_on() {
    local gpu="$1" opt="$2" lr="$3" seed="$4"
    local steps=$((TOKENS / BSZ))
    local warmup=$(( steps / 20 < 10 ? 10 : steps / 20 ))
    local eval_every=$(( steps / 10 > 0 ? steps / 10 : 1 ))
    local tag="${opt}_bsz${BSZ}_lr${lr}_s${seed}"
    local out_json="$OUT_ROOT/${tag}.json"
    local out_log="$OUT_ROOT/logs/${tag}.log"
    if [ -f "$out_json" ]; then echo "[skip] $tag"; return 2; fi

    local tpm=$(( DEV_BSZ * SEQ ))
    local ga=$(( BSZ / tpm ))
    if [ "$ga" -lt 2 ]; then
        echo "[error] $tag: grad_accum=$ga < 2 (bsz=$BSZ, tpm=$tpm)"
        return 1
    fi

    local runner
    if [ "$opt" = "muon" ]; then
        runner="run_native_muon_v9.py"
    else
        runner="run_lite_v9.py"
    fi

    echo "[launch $(date +%H:%M:%S)] host=$(hostname) gpu=$gpu $tag steps=$steps warmup=$warmup grad_accum=$ga"
    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        export PYTHONUNBUFFERED=1
        export PYTORCH_ALLOC_CONF=expandable_segments:True
        cd "$REPO"
        source "$REPO/nanochat/.venv/bin/activate"
        exec python -u "$runner" \
            --depth "$DEPTH" --max-seq-len "$SEQ" \
            --device-batch-size "$DEV_BSZ" \
            --total-batch-size "$BSZ" \
            --max-steps "$steps" \
            --matrix-lr "$lr" \
            --warmup-steps "$warmup" \
            --warmdown-ratio 0.65 \
            --eval-every "$eval_every" \
            --output-file "$out_json" \
            --seed "$seed" \
            $([ "$opt" = "lite" ] && echo "--lite-chi 2.0 --lite-rs 0.1 --lite-chi-warmup 0.5 --lite-chi-schedule warmup_hold")
    ) > "$out_log" 2>&1 &
    slot_pid[$gpu]=$!
    return 0
}

find_free_slot() {
    for ((g=0; g<NGPUS; g++)); do
        local pid="${slot_pid[$g]}"
        if [ -z "$pid" ]; then echo "$g"; return; fi
        if ! kill -0 "$pid" 2>/dev/null; then
            slot_pid[$g]=""; echo "$g"; return
        fi
    done
    echo ""
}

queue=()
for opt in muon lite; do
    for lr in 0.01 0.02 0.04; do
        for seed in 42 43 44; do
            queue+=("${opt}|${lr}|${seed}")
        done
    done
done

echo "Queue size: ${#queue[@]} runs"
echo "Plan:"
for entry in "${queue[@]}"; do echo "  128K ${entry//|/ }"; done

pos=0; launched=0
while [ $pos -lt ${#queue[@]} ]; do
    g=$(find_free_slot)
    if [ -z "$g" ]; then sleep 10; continue; fi
    IFS='|' read -r opt lr seed <<< "${queue[$pos]}"
    if launch_on "$g" "$opt" "$lr" "$seed"; then
        launched=$((launched + 1))
    elif [ $? -eq 2 ]; then
        slot_pid[$g]=""
    fi
    pos=$((pos + 1))
    sleep 3
done

echo "All $pos queue entries dispatched ($launched launched). Waiting for active jobs..."
wait
echo "=== v18 128K clean sweep complete ==="
ls -la "$OUT_ROOT/" | head -60

cd "$REPO"
MPLCONFIGDIR=/tmp/matplotlib-sigma python analyze_v18_128k_clean.py
