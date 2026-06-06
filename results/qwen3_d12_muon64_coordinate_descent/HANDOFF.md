# Qwen3 d12 64K Muon Coordinate Descent Handoff

This handoff is for continuing the coordinate-descent hyperparameter search for Qwen3 d12 Muon at `batch=64K`.

## Goal

Tune five coordinates:

- `matrix_lr`
- Muon matrix optimizer `beta1` / `muon_momentum`
- auxiliary AdamW LR via `adam_lr_multiplier`
- auxiliary AdamW `adam_beta1`
- `weight_decay`

Coordinate-descent rule:

1. In each round, sweep 5 candidate values for each active coordinate.
2. Evaluate all candidates independently.
3. Pick the single coordinate change with the largest loss improvement.
4. Update the center recipe to that candidate.
5. In the next round, sweep the other 4 coordinates and skip the coordinate just changed.
6. Stop after 6 rounds max, or when no coordinate improves.

Metric: lower `score` / final validation BPB is better.

## Current Center Recipe

This is the best known recipe before coordinate descent:

```text
architecture=qwen3
depth=12
batch=65536
chinchilla_mult=1.0
optimizer=plain_muon
matrix_lr=0.008
muon_momentum=0.95
adam_lr_multiplier=1.0
adam_beta1=0.8
weight_decay=0.20
score=0.8623740088803121
```

This came from:

```text
qwen3_d12_1x_a100r34rw_muon64k_b0p95_wd0p2_20260605/plain_muon_bsz65536_lr0p008_s42/result.json
```

## State Files

Use this directory as the experiment ledger:

```text
results/qwen3_d12_muon64_coordinate_descent/
```

Already created:

```text
README.md
state.json
round_00_candidates.csv
HANDOFF.md
```

Update `state.json` and add one `round_XX_candidates.csv` per round. Keep these files small and human-readable.

## Required Code Version

Coordinate descent needs two knobs that were added after the last pushed commit:

- `--adam-lr-multiplier`
- `--adam-beta1`

These are threaded through:

```text
run_eval.py
run_top_aware_muon_sweep.py
scripts/run_d12_sweep.sh
scripts/submit_merlin_sweep.py
```

The code tarball currently uploaded for round 0 is:

```text
hdfs://haruna/home/byte_data_seed/hdd_hldy/user/xingyu.dang/sigma-search/sigma-search-clean-qwen3-d12-muon64-cd-20260605.tgz
```

If you change the code, package and upload a new tarball before launching later rounds.

Packaging pattern:

```bash
STAMP=qwen3-d12-muon64-cd-YYYYMMDD-HHMM
TGZ=/tmp/sigma-search-clean-${STAMP}.tgz
HDFS_TGZ=hdfs://haruna/home/byte_data_seed/hdd_hldy/user/xingyu.dang/sigma-search/sigma-search-clean-${STAMP}.tgz
tar --exclude='.git' --exclude='nanochat/.venv' --exclude='figures' --exclude='logs' --exclude='search_evals' --exclude='results' \
  -czf "$TGZ" -C /mlx_devbox/users/xingyu.dang/playground sigma-search-clean
hdfs dfs -mkdir -p hdfs://haruna/home/byte_data_seed/hdd_hldy/user/xingyu.dang/sigma-search
hdfs dfs -put -f "$TGZ" "$HDFS_TGZ"
```

## Round 0

Round 0 has already been submitted: 25 independent 8-GPU jobs.

Candidate grid:

```text
matrix_lr:          0.004, 0.0057, 0.008, 0.0113, 0.016
muon_momentum:      0.90, 0.93, 0.95, 0.97, 0.985
adam_lr_multiplier: 0.25, 0.5, 1.0, 1.5, 2.0
adam_beta1:         0.6, 0.7, 0.8, 0.9, 0.95
weight_decay:       0.08, 0.12, 0.16, 0.20, 0.28
```

All round-0 stamps are listed in:

```text
results/qwen3_d12_muon64_coordinate_descent/round_00_candidates.csv
```

Initial status after submission was:

```text
6 RUNNING / 19 STARTED / 0 result.json
```

Round 0 was submitted to `seed_eval_x`, cluster 34 / soil-hl:

```text
group_ids=914
cluster_id=34
queue_name=a100-sxm-80gb.i110617261685691719770.ai
gpuv=A100_SXM_80GB
gpu=8
cpu=120
memory=1941504
```

## Resource Notes

Do not rely only on soil-hl. There are often many free A100s in soil-lq:

```text
seed_eval_x / cluster_id=46 / soil-lq
queue_name=a100-sxm-80gb.hpccluster-ycrh6ea3w5lvlmmax48x.ai
gpuv=A100_SXM_80GB
gpu=8
cpu=112
memory=1859584
```

Important caveats:

- Always use HDFS volume key `access_mode`, not `accessMode`, in `--hdfs-volume-json`.
- Verify `get-run` shows `hdfsVolumes[].accessMode == "RW"` before trusting a new batch.
- Verify `list-trial-exit-info` or logs show actual `gpuv=A100-SXM-80GB`. An earlier bad submission to soil-lq ended up on `NVIDIA-L20-16` and was evicted. If using soil-lq, run one probe first or watch the first few jobs closely.

Known-good HDFS volume JSON:

```json
{"path":"hdfs://haruna/home/byte_data_seed/hdd_hldy/user/xingyu.dang/","mnt":"/mnt/hdfs/user/xingyu.dang","roles":["worker"],"access_mode":"RW"}
```

If H20s are free and you just need throughput, `seed_us_research` also works:

```text
group_ids=1538
cluster_id=48
queue_name=nvidia-h20.hpccluster-ydu7v0g5r0m5d250o7b8.ai
gpuv=NVIDIA_H20
gpu=8
cpu=176
memory=1859584
```

## Monitoring Round 0

Use this to summarize status and result count:

```bash
python3 - <<'PY'
import csv, json, subprocess, collections

csv_path = "results/qwen3_d12_muon64_coordinate_descent/round_00_candidates.csv"
base = "hdfs://haruna/home/byte_data_seed/hdd_hldy/user/xingyu.dang/sigma-search-runs/search_evals"
stamps = [row["stamp"] for row in csv.DictReader(open(csv_path))]
counts = collections.Counter()
result_paths = []

for stamp in stamps:
    runs = json.loads(subprocess.check_output([
        "merlin-cli", "--control-plane", "cn-seed", "job", "list-run",
        "--json", json.dumps({"job_name": stamp, "pageSize": 50, "current": 1}),
    ], text=True)).get("list") or []
    counts.update(run.get("status") or "UNKNOWN" for run in runs)
    try:
        listing = subprocess.check_output(["hdfs", "dfs", "-ls", "-R", f"{base}/{stamp}"], text=True)
    except subprocess.CalledProcessError:
        continue
    result_paths.extend(line.split()[-1] for line in listing.splitlines() if line.endswith("/result.json"))

print("counts", dict(counts))
print("results", len(result_paths), "expected", len(stamps))
PY
```

Use this to collect scores:

```bash
python3 - <<'PY'
import csv, json, subprocess

csv_path = "results/qwen3_d12_muon64_coordinate_descent/round_00_candidates.csv"
base = "hdfs://haruna/home/byte_data_seed/hdd_hldy/user/xingyu.dang/sigma-search-runs/search_evals"
rows = list(csv.DictReader(open(csv_path)))

scored = []
for row in rows:
    stamp = row["stamp"]
    try:
        listing = subprocess.check_output(["hdfs", "dfs", "-ls", "-R", f"{base}/{stamp}"], text=True)
    except subprocess.CalledProcessError:
        continue
    paths = [line.split()[-1] for line in listing.splitlines() if line.endswith("/result.json")]
    for path in paths:
        data = json.loads(subprocess.check_output(["hdfs", "dfs", "-cat", path], text=True))
        if data.get("error") or data.get("score") is None:
            continue
        scored.append({**row, "score": float(data["score"]), "path": path})

for row in sorted(scored, key=lambda r: r["score"]):
    print(
        f"{row['score']:.6f}",
        row["coordinate"],
        row["candidate"],
        "matrix_lr", row["matrix_lr"],
        "muon_momentum", row["muon_momentum"],
        "adam_lr_multiplier", row["adam_lr_multiplier"],
        "adam_beta1", row["adam_beta1"],
        "weight_decay", row["weight_decay"],
        row["stamp"],
    )
PY
```

## Choosing Next Round

Round 0 baseline score:

```text
0.8623740088803121
```

After all candidates finish:

1. For each coordinate, find its best candidate.
2. Compute improvement: `baseline_score - best_candidate_score`.
3. Pick the coordinate with largest positive improvement.
4. Update center recipe to that candidate.
5. If no improvement, declare convergence.
6. For round 1, sweep the four other coordinates only.

Example next-round naming:

```text
qwen3_d12_muon64_cd_r01_<coordinate>_<value>_20260605
```

## Submission Template

For one candidate:

```bash
python scripts/submit_merlin_sweep.py --submit \
  --preset d12_2x \
  --stamp "<STAMP>" \
  --methods plain_muon \
  --batches "65536" \
  --lrs "<MATRIX_LR>" \
  --alphas "1.0" \
  --top-ks "1" \
  --depth 12 \
  --chinchilla-mult 1.0 \
  --architecture qwen3 \
  --nproc 8 \
  --max-device-batch-size 16 \
  --weight-decay "<WEIGHT_DECAY>" \
  --muon-momentum "<MUON_MOMENTUM>" \
  --muon-momentum-schedule static \
  --adam-lr-multiplier "<ADAM_LR_MULTIPLIER>" \
  --adam-beta1 "<ADAM_BETA1>" \
  --batch-beta-align \
  --batch-beta-align-mode beta2_only \
  --structured-config global \
  --precondition-frequency 1 \
  --shampoo-beta 0.95 \
  --optimizer-beta1 "<MUON_MOMENTUM>" \
  --optimizer-beta2 0.95 \
  --structured-init-factor 0.1 \
  --save-every 100 \
  --keep-last-checkpoints 2 \
  --streaming-num-iters 2 \
  --fallback-ortho-tol 0.01 \
  --metrics-every 0 \
  --metrics-hessian-every 0 \
  --image-url hub.byted.org/reckon/data.reckon.mlx.image_11319:50545093b9947634c8247599da45c75e \
  --hdfs-code-tgz hdfs://haruna/home/byte_data_seed/hdd_hldy/user/xingyu.dang/sigma-search/sigma-search-clean-qwen3-d12-muon64-cd-20260605.tgz \
  --hdfs-runtime-tgz hdfs://haruna/home/byte_data_seed/hdd_hldy/user/xingyu.dang/sigma-search/runtime/nanochat_venv_torch291_cu128.tgz \
  --nanochat-base-dir /mnt/hdfs/user/xingyu.dang/nanochat \
  --merlin-output-base /mnt/hdfs/user/xingyu.dang/sigma-search-runs \
  --group-ids 914 \
  --cluster-id 34 \
  --queue-name a100-sxm-80gb.i110617261685691719770.ai \
  --gpuv A100_SXM_80GB \
  --gpu 8 \
  --cpu 120 \
  --memory 1941504 \
  --hdfs-volume-json '{"path":"hdfs://haruna/home/byte_data_seed/hdd_hldy/user/xingyu.dang/","mnt":"/mnt/hdfs/user/xingyu.dang","roles":["worker"],"access_mode":"RW"}'
```

To use soil-lq, change:

```text
--cluster-id 46
--queue-name a100-sxm-80gb.hpccluster-ycrh6ea3w5lvlmmax48x.ai
--cpu 112
--memory 1859584
```

Then inspect one started job before launching a large batch.
