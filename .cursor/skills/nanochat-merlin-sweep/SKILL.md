---
name: nanochat-merlin-sweep
description: Submit and babysit nanochat/sigma-search optimizer sweeps on Merlin H800/A100 jobs. Use when launching run_optimizer_sweep.py, scripts/run_d12_sweep.sh, scripts/submit_merlin_sweep.py, or when the user mentions nanochat sweep, sigma-search sweep, Merlin job, H800, seed_eval_x, HDFS_CODE_TGZ, HDFS_RUNTIME_TGZ, or babysitting sweep jobs.
---

# Nanochat Merlin Sweep

Use this for sigma-search/nanochat sweeps launched through Merlin.

## Core Rule

Prefer the repo submitter:

```bash
python scripts/submit_merlin_sweep.py ...
```

For the standard d8 plain-Muon sweep, prefer the dedicated wrapper:

```bash
python scripts/submit_d8_muon_merlin_sweep.py --submit
```

Do not hand-roll `create-run` payloads unless the submitter cannot express the job.

## Preflight

1. Check schema before write operations:

```bash
merlin-cli --control-plane cn-seed job create-run --schema
```

2. Check available resources:

```bash
merlin-cli --control-plane cn-seed resource list-my-resource \
  --json '{"filters":{"type":"gpu"}}'
```

Known good H800 resource:

```text
group_id: 914
group_name: seed_eval_x
cluster_id: 34
cluster_name: soil-hl
queue_name: nvidia-h800.hpccluster-ydkee1a2cjm0adk3uzf4.ai
gpuv: NVIDIA-H800 CLI / NVIDIA_H800 payload
8 GPU package: cpu=160, memory=2850816
HDFS mount: hdfs://haruna/home/byte_data_seed/hdd_hldy/user/xingyu.dang/ -> /mnt/hdfs/user/xingyu.dang
```

## Code And Runtime

Successful jobs use an HDFS code tarball plus offline runtime.

Package code:

```bash
tar --exclude='.git' --exclude='nanochat/.venv' \
  --exclude='results' --exclude='figures' --exclude='logs' --exclude='search_evals' \
  -czf /tmp/sigma-search-clean-<stamp>.tgz \
  -C /mlx_devbox/users/xingyu.dang/playground sigma-search-clean

hdfs dfs -put -f /tmp/sigma-search-clean-<stamp>.tgz \
  hdfs://haruna/home/byte_data_seed/hdd_hldy/user/xingyu.dang/sigma-search/sigma-search-clean-<stamp>.tgz
```

Known good runtime:

```text
hdfs://haruna/home/byte_data_seed/hdd_hldy/user/xingyu.dang/sigma-search/runtime/nanochat_venv_torch291_cu128.tgz
```

Always pass `--hdfs-runtime-tgz` unless intentionally testing a new image/runtime.

## Entrypoint Requirements

`scripts/merlin_entrypoint.sh` must avoid GitHub/PyTorch downloads on H800:

```bash
export UV_PYTHON_DOWNLOADS="${UV_PYTHON_DOWNLOADS:-never}"
export UV_PYTHON="${UV_PYTHON:-python3}"
export UV_PYTHON_PREFERENCE="${UV_PYTHON_PREFERENCE:-only-system}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export UV_INDEX_URL="${UV_INDEX_URL:-https://bytedpypi.byted.org/simple}"
export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://bytedpypi.byted.org/simple}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://bytedpypi.byted.org/simple}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-bytedpypi.byted.org}"
```

If dependencies are missing, patch `nanochat/pyproject.toml` to remove direct `torch==...`, pytorch uv sources/indexes, and `uv.lock`, then run:

```bash
cd "$REPO/nanochat"
uv venv --system-site-packages --python "${UV_PYTHON:-python3}" .venv || true
uv sync
```

This prevents failures like:

```text
Failed to download https://github.com/astral-sh/python-build-standalone/...
```

## Submit Pattern

Base command skeleton:

```bash
python scripts/submit_merlin_sweep.py \
  --preset d8_quality \
  --stamp <unique_stamp> \
  --methods plain_muon \
  --batches "<batches>" \
  --lrs "<lrs>" \
  --alphas "1.0" \
  --depth 8 \
  --chinchilla-mult 2 \
  --nproc <nproc> \
  --max-device-batch-size <mbs> \
  --weight-decay <wd> \
  --shard-count <nodes> \
  --image-url hub.byted.org/reckon/data.reckon.mlx.image_11319:50545093b9947634c8247599da45c75e \
  --hdfs-code-tgz hdfs://haruna/home/byte_data_seed/hdd_hldy/user/xingyu.dang/sigma-search/<code>.tgz \
  --hdfs-runtime-tgz hdfs://haruna/home/byte_data_seed/hdd_hldy/user/xingyu.dang/sigma-search/runtime/nanochat_venv_torch291_cu128.tgz \
  --nanochat-base-dir /mnt/hdfs/user/xingyu.dang/nanochat \
  --merlin-output-base /mnt/hdfs/user/xingyu.dang/sigma-search-runs \
  --group-ids 914 \
  --cluster-id 34 \
  --queue-name nvidia-h800.hpccluster-ydkee1a2cjm0adk3uzf4.ai \
  --gpuv NVIDIA-H800 \
  --gpu 8 \
  --cpu 160 \
  --memory 2850816 \
  --hdfs-volume-json '{"path":"hdfs://haruna/home/byte_data_seed/hdd_hldy/user/xingyu.dang/","mnt":"/mnt/hdfs/user/xingyu.dang","access_mode":"RW","roles":["worker"]}' \
  --submit
```

First dry-run without `--submit`; inspect payloads under `/tmp/sigma_merlin_payloads_<stamp>/`.

## Standard d8 Muon Sweep

Use:

```bash
python scripts/submit_d8_muon_merlin_sweep.py \
  --stamp plainmuon_d8_a100_<stamp> \
  --submit
```

Defaults:

```text
method: plain_muon only
depth: 8
CHINCHILLA_MULT: 2
4K: one 8-GPU node running two 4-GPU cases concurrently
64K/128K/512K/1M: 8-GPU DDP jobs, main-shard-count=3 by default
resource: seed_eval_x A100 queue
runtime: nanochat_venv_torch291_cu128.tgz
```

Muon momentum is not batch-scaled. Batch half-life beta alignment remains for Adam/structured optimizer betas.

## Manifest Discipline

Use one unique `STAMP` / `OUT_ROOT` per sweep signature.

Do not mix different `BATCHES`, `NPROC`, `MAX_DEVICE_BATCH_SIZE`, token budget, methods, LR grid, or metrics under one output root. If mixed accidentally, jobs fail with:

```text
manifest.json already exists with a different sweep configuration
```

For example, 4K with `NPROC=4` needs a separate stamp from 64K-1M with `NPROC=8`.

## Batch And Accumulation Check

For `SEQ=1024`, `world_tokens = NPROC * max_device_batch_size * 1024`.

Examples:

```text
4K:   NPROC=4, max_device_batch_size=1  -> accum=1
64K:  NPROC=8, max_device_batch_size=64 -> actual device_batch=8,  accum=1
128K: NPROC=8, max_device_batch_size=64 -> actual device_batch=16, accum=1
512K: NPROC=8, max_device_batch_size=64 -> actual device_batch=64, accum=1
1M:   NPROC=8, max_device_batch_size=64 -> actual device_batch=64, accum=2
```

`device_batch_for()` in `run_top_aware_muon_sweep.py` caps device batch at `max_device_batch_size`.

## Parallel Cases Inside One Node

For tiny batches that cannot use all 8 GPUs in one DDP job, use:

```bash
--merlin-parallel-cases 2
--merlin-gpus-per-case 4
--nproc 4
```

The entrypoint assigns disjoint `CUDA_VISIBLE_DEVICES` groups per case.

## Status And Logs

Check runs:

```bash
merlin-cli --control-plane cn-seed job get-run \
  --json '{"job_run_id":"<job_run_id>"}'
```

List recent user runs:

```bash
merlin-cli --control-plane cn-seed job list-run \
  --json '{"username":"xingyu.dang","pageSize":20,"current":1}'
```

For failures:

```bash
merlin-cli --control-plane cn-seed job list-trial-exit-info \
  --json '{"job_run_id":"<job_run_id>","trial_id":"<trial_id>"}'

merlin-cli --control-plane cn-seed job list-trial-logs \
  --json '{"job_run_id":"<job_run_id>","trial_id":"<trial_id>"}'
```

Fetch stdout/stderr URLs and inspect the last 100-200 lines.

## Common Failure Modes

- GitHub timeout during `uv sync`: entrypoint did not force system Python / byted PyPI or did not use runtime tarball.
- `manifest.json already exists`: reused `OUT_ROOT` for a different sweep signature; use a new stamp.
- Missing data: check `NANOCHAT_BASE_DIR` has `tokenizer/` and `base_data_climbmix/*.parquet`.
- Small batch invalid: ensure `batch >= SEQ * NPROC`; for 4K use `NPROC=4`, not `8`.

## Babysitting

When babysitting, keep a list of active job IDs, poll status, and only debug failures. Report deltas:

```text
DONE: ...
RUNNING: ...
FAILED: ... root cause ...
NEXT: ...
```

Stop babysitting when all selected cases are `DONE` or when a failure requires user choice.

