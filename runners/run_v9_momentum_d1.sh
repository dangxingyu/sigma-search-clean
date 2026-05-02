#!/bin/bash
# v9: D1(M_tilde) sweep.
#
# Runs each training job as a single-GPU world=1 process, with 8 jobs in
# parallel on the allocated B200 node. This is intentional: native Muon has a
# distributed optimizer, but the current LITE implementation does not, so
# torchrun/DDP would make Muon and LITE incomparable.
#
# Each run computes:
#   D1(g): Sadhika's original fresh-gradient left-U split-batch probe.
#   D1(M_tilde): split-batch scheduled-momentum Nesterov direction, using the
#                singular-vector side LITE actually projects on (V for tall,
#                U for wide). This is stored in telemetry.d1_probes_momentum.
#
# Setup: d=8, dev_bsz=16, seq=1024, 1B tokens, 8 single-GPU slots.
# Sweep: 4 bsz × 2 opt × 3 seeds = 24 runs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$SCRIPT_DIR}"
JOB="${JOB:-29702470}"
NGPUS="${NGPUS:-8}"
TOKENS=1073741824
DEPTH=8
DEV_BSZ=16
SEQ=1024
OUT_ROOT="$REPO/sweep_results_v9"
mkdir -p "$OUT_ROOT/logs"

# Best-LR per (opt, bsz) from v2/v6/v7 analysis (mode across seeds).
declare -A LR
LR[muon_524288]=0.01;    LR[lite_524288]=0.01
LR[muon_1048576]=0.01;   LR[lite_1048576]=0.02
LR[muon_4194304]=0.04;   LR[lite_4194304]=0.04
LR[muon_16777216]=0.04;  LR[lite_16777216]=0.04

declare -A slot_pid
for ((g=0; g<NGPUS; g++)); do slot_pid[$g]=""; done

launch_on() {
    local gpu="$1" opt="$2" bsz="$3" lr="$4" seed="$5"
    local steps=$((TOKENS / bsz))
    local warmup=$(( steps / 20 < 10 ? 10 : steps / 20 ))
    local eval_every=$(( steps / 10 > 0 ? steps / 10 : 1 ))
    local tag="${opt}_bsz${bsz}_lr${lr}_s${seed}"
    local out_json="$OUT_ROOT/${tag}.json"
    local out_log="$OUT_ROOT/logs/${tag}.log"
    if [ -f "$out_json" ]; then echo "[skip] $tag"; return 2; fi

    local driver="run_native_muon_v9.py"
    local extra=""
    if [ "$opt" = "lite" ]; then
        driver="run_lite_v9.py"
        extra="--lite-chi 2.0 --lite-rs 0.1"
    fi

    # With world=1: grad_accum = bsz / (DEV_BSZ * SEQ).
    local tpm=$(( DEV_BSZ * SEQ ))
    local ga=$(( bsz / tpm ))
    if [ "$ga" -lt 2 ]; then
        echo "[error] $tag: grad_accum=$ga < 2 (bsz=$bsz, tpm=$tpm); cannot do split-batch snapshot"
        return 1
    fi

    echo "[launch $(date +%H:%M:%S)] gpu=$gpu $tag steps=$steps warmup=$warmup grad_accum=$ga"
    srun --jobid="$JOB" --overlap --ntasks=1 \
        bash -c "
        export CUDA_VISIBLE_DEVICES=$gpu
        export PYTHONUNBUFFERED=1
        export PYTORCH_ALLOC_CONF=expandable_segments:True
        cd $REPO
        source $REPO/nanochat/.venv/bin/activate
        exec python -u $driver \
            --depth $DEPTH --max-seq-len $SEQ \
            --device-batch-size $DEV_BSZ \
            --total-batch-size $bsz \
            --max-steps $steps \
            --matrix-lr $lr \
            --warmup-steps $warmup \
            --warmdown-ratio 0.65 \
            --eval-every $eval_every \
            --output-file $out_json \
            --seed $seed \
            $extra
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
for bsz in 16777216 4194304 1048576 524288; do
    for opt in muon lite; do
        for seed in 42 43 44; do
            queue+=("${opt}|${bsz}|${LR[${opt}_${bsz}]}|${seed}")
        done
    done
done

echo "Queue size: ${#queue[@]} runs"
echo "Plan:"
for entry in "${queue[@]}"; do echo "  ${entry//|/ }"; done

pos=0; launched=0
while [ $pos -lt ${#queue[@]} ]; do
    g=$(find_free_slot)
    if [ -z "$g" ]; then sleep 5; continue; fi
    IFS='|' read -r opt bsz lr seed <<< "${queue[$pos]}"
    if launch_on "$g" "$opt" "$bsz" "$lr" "$seed"; then
        launched=$((launched + 1))
    elif [ $? -eq 2 ]; then
        slot_pid[$g]=""
    fi
    pos=$((pos + 1))
    sleep 3
done

echo "All $pos queue entries dispatched ($launched launched). Waiting for active jobs..."
wait
echo "=== v9 D1(M_tilde) sweep complete ==="
ls -la "$OUT_ROOT/" | head -40
