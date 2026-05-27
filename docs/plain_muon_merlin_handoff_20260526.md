# Plain-Muon Merlin Sweep Handoff - 2026-05-26

## Current Goal

Rerun d8 / 40TPP plain-Muon sweeps for batches:

- `4096`
- `65536`
- `131072`
- `524288`
- `1048576`

The motivation is to rerun previous Muon LR points after fixing the local optimizer recipe:

- `plain_muon` now uses Moonlight update scaling: `0.2 * sqrt(max(rows, cols))`.
- Weight decay is not Moonlight-scaled.
- Adam auxiliary parameter groups use `relative_to_matrix`.
- Batch beta/momentum half-life alignment is enabled by default.
- KL-SOAP/SOAP do not use Moonlight scaling.

## Important Local Issue

The agent shell became unusable during babysitting:

```text
spawn /bin/bash ENOENT
```

After that, shell/subagent attempts no longer had access to the original `/mlx_devbox` workspace or `merlin-cli`. This appears to be an execution-environment issue, not a repo issue. The handoff below records all state needed to continue from a working devbox shell.

## Code State

The working tree contains uncommitted optimizer changes and Merlin entrypoint changes. Key files changed:

- `optimizer_recipe.py`
- `baseline_optim.py`
- `run_eval.py`
- `run_top_aware_muon_sweep.py`
- `scripts/run_d12_sweep.sh`
- `scripts/submit_merlin_sweep.py`
- `scripts/merlin_entrypoint.sh`
- `tests/test_optimizer_recipe.py`
- `tests/test_baseline_optim.py`

Validation before shell broke:

```text
python -m py_compile optimizer_recipe.py baseline_optim.py run_eval.py run_top_aware_muon_sweep.py
python -m pytest tests -q
25 passed
```

## First Submission

I packaged the current code to:

```text
hdfs://haruna/home/byte_data_seed/hdd_hldy/user/xingyu.dang/sigma-search/sigma-search-clean-d8-plainmuon-20260526.tgz
```

Then submitted 8 H800 jobs on `seed_eval_x`, cluster `soil-hl`, queue:

```text
nvidia-h800.hpccluster-ydkee1a2cjm0adk3uzf4.ai
```

Initial submitted jobs:

```text
85601890ea834eac  4K shard
c01cfccca97ce75f  64K-1M shard 0
7ef2d56c68a954de  64K-1M shard 1
3955f6086d13d207  64K-1M shard 2
1f3d0b3ea75a41d2  64K-1M shard 3
9a65021029aed035  64K-1M shard 4
e74e7c2df71bdcc8  64K-1M shard 5
3b0e4f4c74b7a0f5  64K-1M shard 6
```

All failed quickly.

Root cause from logs:

```text
uv sync
Failed to download https://github.com/astral-sh/python-build-standalone/...
operation timed out
```

So the H800 node could not reach GitHub to download Python standalone.

## Second Submission

I found an existing offline runtime:

```text
hdfs://haruna/home/byte_data_seed/hdd_hldy/user/xingyu.dang/sigma-search/runtime/nanochat_venv_torch291_cu128.tgz
```

Then resubmitted with `--hdfs-runtime-tgz`:

```text
0365883fd3cc7d4d  4K shard
88f5ed7e733c6120  64K-1M shard 0
d3d3db28fe4266b9  64K-1M shard 1
9ec336e83fea330c  64K-1M shard 2
0ec18a3773436f51  64K-1M shard 3
cd71fd13132f18f7  64K-1M shard 4
0a44cd0ae0934ca9  64K-1M shard 5
1cecfccb0230f8f6  64K-1M shard 6
```

Last known statuses before shell broke:

```text
0365883fd3cc7d4d  STARTED
88f5ed7e733c6120  STARTED
d3d3db28fe4266b9  STARTED
9ec336e83fea330c  STARTED
0ec18a3773436f51  STARTED
cd71fd13132f18f7  STARTED
0a44cd0ae0934ca9  RUNNING
1cecfccb0230f8f6  STARTED
```

These need to be checked first from a working shell.

## Successful Historical Run To Imitate

User provided successful run:

```text
https://seed.bytedance.net/development/instance/jobs/fd47f14595e27971?tabState=task_config&trialId=352259010
```

Important: that successful run apparently used a fully packaged HDFS code tarball, but patched the uv environment before running `scripts/merlin_entrypoint.sh`.

Successful entrypoint prefix:

```bash
set -euo pipefail
export SIGMA_SEARCH_REPO="${SIGMA_SEARCH_REPO:-/opt/tiger/sigma-search-clean}"
export UV_PYTHON_DOWNLOADS="${UV_PYTHON_DOWNLOADS:-never}"
export UV_PYTHON="${UV_PYTHON:-python3}"
export UV_PYTHON_PREFERENCE="${UV_PYTHON_PREFERENCE:-only-system}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export UV_INDEX_URL="${UV_INDEX_URL:-https://bytedpypi.byted.org/simple}"
export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://bytedpypi.byted.org/simple}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://bytedpypi.byted.org/simple}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-bytedpypi.byted.org}"
```

It then patched `nanochat/pyproject.toml`:

- Remove `"torch==2.9.1"` direct dependency.
- Remove `[tool.uv.sources]` block if it only defines torch/pytorch sources.
- Remove pytorch `[[tool.uv.index]]` blocks.
- Remove `nanochat/uv.lock`.
- Replace `uv sync --extra gpu` or `uv sync --extra gpu --frozen` inside `scripts/merlin_entrypoint.sh` with:

```bash
uv venv --system-site-packages --python "${UV_PYTHON:-python3}" .venv || true
uv sync
```

Then run:

```bash
bash scripts/merlin_entrypoint.sh
```

This avoids GitHub/PyTorch downloads and uses the system Python / system-site torch in the image.

## Observed Successful Rerun2 Config

Later successful jobs were submitted under:

```text
plainmuon_d8_h800_20260527_0831_rerun2
```

Representative DONE job:

```text
job_run_id: 7fafa9c3612e7857
trial_id: 353307209
status: DONE
caption: sigma-sweep-plainmuon_d8_h800_20260527_0831_rerun2-shard-006-006-013
```

What made this submission work:

- It still used an HDFS code tarball, not Codebase/Git:

```text
HDFS_CODE_TGZ=hdfs://haruna/home/byte_data_seed/hdd_hldy/user/xingyu.dang/sigma-search/sigma-search-clean-d8-plainmuon-20260527_0829_fix2-671fd00.tgz
```

- It also provided the offline runtime tarball:

```text
HDFS_RUNTIME_TGZ=hdfs://haruna/home/byte_data_seed/hdd_hldy/user/xingyu.dang/sigma-search/runtime/nanochat_venv_torch291_cu128.tgz
```

- It ran through the normal wrapper:

```text
MERLIN_SWEEP_SCRIPT=scripts/run_d12_sweep.sh
METHODS=plain_muon
DEPTH=8
CHINCHILLA_MULT=2.0
NPROC=8
MAX_DEVICE_BATCH_SIZE=64
MATRIX_LR_ADJUST=moonlight        # default via wrapper
ADAM_LR_MODE=relative_to_matrix   # default via wrapper
BATCH_BETA_ALIGN=1                # default via wrapper
```

- The H800 image still installed `uv`, but the patched entrypoint prevented the earlier GitHub Python-standalone failure. In logs this showed:

```text
Restoring runtime from HDFS_RUNTIME_TGZ=...
Using CPython 3.11.2 interpreter at: /usr/bin/python3
Creating virtual environment at: .venv
Resolved 114 packages ...
Installed 64 packages ...
```

Current rerun2 status when checked:

```text
7fafa9c3612e7857  DONE
a0e533b549a101af  DONE
ea436b09b308983a  RUNNING
4ebf363e20705643  RUNNING
5897e744c8cdc63f  RUNNING
cddd27272e14c3fa  RUNNING
64f5fde42f1b1bef  RUNNING
e1019b9361c2b231  FAILED
```

The failed `e1019b9361c2b231` is the 4K shard, and its failure is not the old `uv`/GitHub issue. Its stderr ended with:

```text
ValueError: .../search_evals/plainmuon_d8_h800_20260527_0831_rerun2/manifest.json already exists with a different sweep configuration.
Use a new STAMP/OUT_ROOT for a changed recipe, or pass --allow-config-mismatch if you intentionally want to mix configs.
```

Root cause: the 4K sweep used the same `STAMP` / `OUT_ROOT` as the 64K-1M sweep but has a different sweep signature (`BATCHES=4096`, `NPROC=4`, `MAX_DEVICE_BATCH_SIZE=1`, etc.). The sweep manifest guard correctly rejected mixing these configurations.

Fix for 4K: submit it under a separate stamp/output root, for example:

```text
STAMP=plainmuon_d8_4k_h800_20260527_0831_rerun2
OUT_ROOT=/mnt/hdfs/user/xingyu.dang/sigma-search-runs/search_evals/plainmuon_d8_4k_h800_20260527_0831_rerun2
LOG_ROOT=/mnt/hdfs/user/xingyu.dang/sigma-search-runs/logs/plainmuon_d8_4k_h800_20260527_0831_rerun2
```

Do not reuse the 64K-1M `plainmuon_d8_h800_20260527_0831_rerun2` output root for 4K unless intentionally passing `--allow-config-mismatch`.

## Recommended Fix

Patch `scripts/merlin_entrypoint.sh` directly so future H800 jobs do not need an inline entrypoint patch:

1. Add the `UV_*` and `PIP_*` env defaults near the top.
2. Before dependency setup, patch `nanochat/pyproject.toml` to remove the torch dependency and pytorch uv source/index blocks.
3. Remove `nanochat/uv.lock`.
4. Change dependency setup from:

```bash
(cd "$REPO/nanochat" && uv sync --extra gpu)
```

to:

```bash
(
  cd "$REPO/nanochat"
  uv venv --system-site-packages --python "${UV_PYTHON:-python3}" .venv || true
  uv sync
)
```

Then rebuild and upload:

```bash
tar --exclude='.git' --exclude='nanochat/.venv' --exclude='results' --exclude='figures' --exclude='logs' --exclude='search_evals' \
  -czf /tmp/sigma-search-clean-d8-plainmuon-20260526.tgz \
  -C /mlx_devbox/users/xingyu.dang/playground sigma-search-clean

hdfs dfs -put -f /tmp/sigma-search-clean-d8-plainmuon-20260526.tgz \
  hdfs://haruna/home/byte_data_seed/hdd_hldy/user/xingyu.dang/sigma-search/sigma-search-clean-d8-plainmuon-20260526.tgz
```

Then resubmit only missing/failed cases.

## Sweep Plan

4K sweep:

```text
batch: 4096
methods: plain_muon
LRs: 0.0005 0.00075 0.001 0.0015 0.002
NPROC=4
MAX_DEVICE_BATCH_SIZE=1
MERLIN_PARALLEL_CASES=2
MERLIN_GPUS_PER_CASE=4
```

64K-1M sweep:

```text
batches: 65536 131072 524288 1048576
methods: plain_muon
LRs: 0.001 0.0015 0.002 0.003 0.004
NPROC=8
MAX_DEVICE_BATCH_SIZE=64
shard-count: 7
```

Grad accumulation check:

```text
4K:    NPROC=4, device_batch=1  -> world tokens = 4096, accum=1
64K:   NPROC=8, device_batch=8  -> world tokens = 65536, accum=1
128K:  NPROC=8, device_batch=16 -> world tokens = 131072, accum=1
512K:  NPROC=8, device_batch=64 -> world tokens = 524288, accum=1
1M:    NPROC=8, device_batch=64 -> world tokens = 524288, accum=2
```

## Commands To Check Status

```bash
python - <<'PY'
import json, subprocess
ids = '''
0365883fd3cc7d4d
88f5ed7e733c6120
d3d3db28fe4266b9
9ec336e83fea330c
0ec18a3773436f51
cd71fd13132f18f7
0a44cd0ae0934ca9
1cecfccb0230f8f6
'''.split()
for jid in ids:
    out = subprocess.check_output([
        'merlin-cli', '--control-plane', 'cn-seed', 'job', 'get-run',
        '--json', json.dumps({'job_run_id': jid})
    ], text=True)
    d = json.loads(out)
    jr = d.get('job_run', {})
    print(jid, jr.get('status'), jr.get('meta', {}).get('arnold_trial_id'), jr.get('job_def_name'))
PY
```

For a failed trial:

```bash
merlin-cli --control-plane cn-seed job list-trial-exit-info \
  --json '{"job_run_id":"<job_run_id>","trial_id":"<trial_id>"}'

merlin-cli --control-plane cn-seed job list-trial-logs \
  --json '{"job_run_id":"<job_run_id>","trial_id":"<trial_id>"}'
```

## Resource Config

Use `seed_eval_x` H800:

```text
group_id: 914
cluster_id: 34
cluster_name: soil-hl
queue_name: nvidia-h800.hpccluster-ydkee1a2cjm0adk3uzf4.ai
gpuv: NVIDIA-H800 / NVIDIA_H800 in payload
8-GPU package: cpu=160, memory=2850816
HDFS mount:
  hdfs://haruna/home/byte_data_seed/hdd_hldy/user/xingyu.dang/
  -> /mnt/hdfs/user/xingyu.dang
```

