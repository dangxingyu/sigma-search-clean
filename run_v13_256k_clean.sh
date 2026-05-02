#!/bin/bash
# v13: clean local 256K crossover-lower-edge baseline with v9-style telemetry.
#
# Purpose: calibrate the D1 gate below the 512K v9 point using the same local
# single-GPU drivers that produced v9/v10/v11/v12. This avoids relying on older
# Modal/legacy sweep telemetry for the lower edge of the crossover.
#
# Setup: d=8, dev_bsz=16, seq=1024, 1B tokens, single-GPU slots.
# Sweep: opt {Muon, LITE chi=2} x seeds {42,43,44} at bsz=256K, lr=0.01.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$SCRIPT_DIR}"
JOB="${JOB:-29702470}"
NGPUS="${NGPUS:-8}"
TOKENS=1073741824
DEPTH=8
DEV_BSZ=16
SEQ=1024
BSZ=262144
LR=0.01
OUT_ROOT="$REPO/sweep_results_v13_256k_clean"
mkdir -p "$OUT_ROOT/logs"

declare -A slot_pid
for ((g=0; g<NGPUS; g++)); do slot_pid[$g]=""; done

launch_on() {
    local gpu="$1" opt="$2" seed="$3"
    local steps=$((TOKENS / BSZ))
    local warmup=$(( steps / 20 < 10 ? 10 : steps / 20 ))
    local eval_every=$(( steps / 10 > 0 ? steps / 10 : 1 ))
    local tag="${opt}_bsz${BSZ}_lr${LR}_s${seed}"
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

    echo "[launch $(date +%H:%M:%S)] gpu=$gpu $tag steps=$steps warmup=$warmup grad_accum=$ga"
    srun --jobid="$JOB" --overlap --ntasks=1 \
        bash -c "
        export CUDA_VISIBLE_DEVICES=$gpu
        export PYTHONUNBUFFERED=1
        export PYTORCH_ALLOC_CONF=expandable_segments:True
        cd $REPO
        source $REPO/nanochat/.venv/bin/activate
        exec python -u $runner \
            --depth $DEPTH --max-seq-len $SEQ \
            --device-batch-size $DEV_BSZ \
            --total-batch-size $BSZ \
            --max-steps $steps \
            --matrix-lr $LR \
            --warmup-steps $warmup \
            --warmdown-ratio 0.65 \
            --eval-every $eval_every \
            --output-file $out_json \
            --seed $seed \
            $([ "$opt" = "lite" ] && echo "--lite-chi 2.0 --lite-rs 0.1 --lite-chi-warmup 0.5 --lite-chi-schedule warmup_hold")
        " > "$out_log" 2>&1 &
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
    for seed in 42 43 44; do
        queue+=("${opt}|${seed}")
    done
done

echo "Queue size: ${#queue[@]} runs"
echo "Plan:"
for entry in "${queue[@]}"; do echo "  256K ${entry//|/ }"; done

pos=0; launched=0
while [ $pos -lt ${#queue[@]} ]; do
    g=$(find_free_slot)
    if [ -z "$g" ]; then sleep 5; continue; fi
    IFS='|' read -r opt seed <<< "${queue[$pos]}"
    if launch_on "$g" "$opt" "$seed"; then
        launched=$((launched + 1))
    elif [ $? -eq 2 ]; then
        slot_pid[$g]=""
    fi
    pos=$((pos + 1))
    sleep 3
done

echo "All $pos queue entries dispatched ($launched launched). Waiting for active jobs..."
wait
echo "=== v13 256K clean sweep complete ==="
ls -la "$OUT_ROOT/" | head -40

cd "$REPO"
MPLCONFIGDIR=/tmp/matplotlib-sigma python analyze_v13_256k_clean.py
