# Merlin Parallel Sweeps

This repo can run sweep cases in parallel on Merlin by submitting one Merlin job
per `run_optimizer_sweep.py --case-index`. All jobs share one `STAMP`,
`OUT_ROOT`, and `LOG_ROOT`.

## Runtime Shape

```text
scripts/submit_merlin_sweep.py
  -> merlin-cli --control-plane cn-seed job create-run --from-file payload.json
    -> scripts/merlin_entrypoint.sh
      -> scripts/run_d12_sweep.sh --case-index N
        -> run_optimizer_sweep.py
          -> torch.distributed.run --standalone --nproc_per_node=8 run_eval.py
```

`scripts/run_d8_metrics_grid.sh` follows the same shape, but delegates to
`scripts/run_d12_statistics.sh` after pinning the d8 metrics recipe.

## Required Merlin Inputs

Use `merlin-cli --control-plane cn-seed job create-run --schema` before changing
payload fields. The submitter expects:

- `--image-url` or `--image-vid`
- prepared `--nanochat-base-dir` with `tokenizer/` and `base_data_climbmix/`
- persistent `--merlin-output-base`
- either `--resource-config-file` or resource flags: group, cluster, queue,
  GPU type, CPU, and memory

Resource discovery:

```bash
merlin-cli --control-plane cn-seed resource list-my-resource \
  --json '{"filters":{"type":"gpu"}}'
```

Merlin resource discovery may return GPU names with hyphens, but `create-run`
roles use underscore names, for example `A100-SXM-80GB -> A100_SXM_80GB`.

## Code Source

Two launch styles are supported.

Use Codebase/Git when clone is reliable:

```bash
--repo-name dangxingyu/sigma-search-clean \
--branch-name main \
--repo-mnt /opt/tiger/sigma-search-clean
```

Use an HDFS tarball when Git/SCM clone is fragile:

```bash
tar -czf /tmp/sigma-search-clean.tgz -C /path/to sigma-search-clean
hdfs dfs -put -f /tmp/sigma-search-clean.tgz \
  hdfs://haruna/home/byte_data_seed/hdd_hldy/user/xingyu.dang/sigma-search/sigma-search-clean.tgz
```

Then submit with:

```bash
--hdfs-code-tgz hdfs://haruna/home/byte_data_seed/hdd_hldy/user/xingyu.dang/sigma-search/sigma-search-clean.tgz
```

The generated Merlin entrypoint downloads and unpacks the tarball before calling
`scripts/merlin_entrypoint.sh`.

## Example: d12 Parallel Base Grid

First generate payloads only:

```bash
python scripts/submit_merlin_sweep.py \
  --preset d12_2x \
  --stamp d12_merlin_2x_001 \
  --image-url hub.byted.org/reckon/data.reckon.mlx.image_11319:50545093b9947634c8247599da45c75e \
  --hdfs-code-tgz hdfs://haruna/home/byte_data_seed/hdd_hldy/user/xingyu.dang/sigma-search/sigma-search-clean.tgz \
  --nanochat-base-dir /mnt/hdfs/user/xingyu.dang/nanochat \
  --merlin-output-base /mnt/hdfs/user/xingyu.dang/sigma-search-runs \
  --group-ids 914 \
  --cluster-id 34 \
  --queue-name a100-sxm-80gb.i110617261685691719770.ai \
  --gpuv A100_SXM_80GB \
  --cpu 120 \
  --memory 1941504 \
  --hdfs-volume-json '{"path":"hdfs://haruna/home/byte_data_seed/hdd_hldy/user/xingyu.dang/","mnt":"/mnt/hdfs/user/xingyu.dang","access_mode":"RW","roles":["worker"]}'
```

Add `--submit` to create the Merlin jobs:

```bash
python scripts/submit_merlin_sweep.py ... --submit
```

Limit the launch during smoke tests:

```bash
python scripts/submit_merlin_sweep.py ... --case-indices 0,1 --submit
python scripts/submit_merlin_sweep.py ... --case-start 0 --case-end 8 --submit
```

## d16 And d8 Presets

Use the same command with a different preset:

```bash
python scripts/submit_merlin_sweep.py --preset d16_2x ...
python scripts/submit_merlin_sweep.py --preset d8_quality ...
python scripts/submit_merlin_sweep.py --preset d8_metrics ...
```

`d8_metrics` pins the documented no-tuning recipe:

- depth `8`
- tokens `402653184`
- batches `262144 1048576 4194304`
- alpha `0.5 1.0`
- LR `0.02`
- cheap metrics every step and Hessian every 50 logged steps

## Optimizer Baseline LR Sweeps

On a Merlin-only cluster, use optimizer-specific presets instead of local
baseline wrappers. Each preset expands to one Merlin job per `(batch, lr, seed)`
case and uses `WEIGHT_DECAY=0.1`.

Available baseline presets:

```text
d12_adamw      d16_adamw
d12_soap       d16_soap
d12_kl_soap    d16_kl_soap
d12_kl_shampoo d16_kl_shampoo
```

Example AdamW dry-run:

```bash
python scripts/submit_merlin_sweep.py \
  --preset d12_adamw \
  --stamp d12_adamw_lr_sweep_001 \
  --image-url hub.byted.org/reckon/data.reckon.mlx.image_11319:50545093b9947634c8247599da45c75e \
  --hdfs-code-tgz hdfs://haruna/home/byte_data_seed/hdd_hldy/user/xingyu.dang/sigma-search/sigma-search-clean.tgz \
  --nanochat-base-dir /mnt/hdfs/user/xingyu.dang/nanochat \
  --merlin-output-base /mnt/hdfs/user/xingyu.dang/sigma-search-runs \
  --group-ids 914 \
  --cluster-id 34 \
  --queue-name a100-sxm-80gb.i110617261685691719770.ai \
  --gpuv A100_SXM_80GB \
  --cpu 120 \
  --memory 1941504
```

The dry-run prints `case_count=21` for `d12_adamw`: three batches times seven
LRs. Add `--submit` to launch all cases, or use `--case-indices 0,1` for a smoke
subset.

Override presets when needed:

```bash
python scripts/submit_merlin_sweep.py \
  --preset d12_adamw \
  --methods adamw \
  --batches "524288" \
  --lrs "0.0005 0.001 0.002" \
  ...
```

## Collation And Adaptive Closure

After base-grid jobs finish, submit one summary job:

```bash
python scripts/submit_merlin_sweep.py \
  --preset d12_2x \
  --stamp d12_merlin_2x_001 \
  --job-kind summary \
  ... \
  --submit
```

If boundary LR closure is desired, submit one adaptive job after summary/base
results are present:

```bash
python scripts/submit_merlin_sweep.py \
  --preset d12_2x \
  --stamp d12_merlin_2x_001 \
  --job-kind adaptive \
  ... \
  --submit
```

The adaptive job reuses `scripts/run_d12_sweep.sh`; completed base cases are
skipped and only new boundary LR cases are launched sequentially by that job.

## Notes

- Keep one depth per `STAMP`; d12 and d16 have different token budgets and
  manifest signatures.
- Use `--resource-config-file` if the resource JSON becomes more complex than
  the basic single-role flags.
- Case jobs set `ADAPTIVE_LR=0`; adaptive closure is a separate step because it
  depends on all base LR results.
- `scripts/merlin_entrypoint.sh` validates data unless `SUMMARY_ONLY=1` or
  `MERLIN_SKIP_DATA_CHECK=1`.
