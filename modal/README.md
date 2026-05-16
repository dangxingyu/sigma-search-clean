# Modal Launchers

This directory contains Modal wrappers around the same local sweep scripts used
on SLURM. The Modal code does not reimplement training; it copies this repo into
the image and runs `scripts/run_d12_sweep.sh` with environment overrides.

## Authentication

On the current machine, `modal` is installed and a profile is configured. Check
without printing secrets:

```bash
modal profile current
```

If a new machine is missing auth, run `modal token new` or provide
`MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` in the environment.

## Prepare Data Once

Dataset shards and tokenizer files are stored in the Modal volume
`sigma-search-nanochat-data`.

```bash
modal run modal/run_sweep.py::prepare_data --dataset-shards 170 --dataset-workers 16
```

For a quick smoke-only tokenizer, lower `--tokenizer-max-chars`, but use the
default for comparable runs.

## Run A Sweep

Set `MODAL_GPU` before launch because GPU type/count is fixed when Modal builds
the function definition.

```bash
MODAL_GPU=B200:8 modal run modal/run_sweep.py::sweep \
  --depth 12 \
  --chinchilla-mult 2 \
  --methods top_aware_muon \
  --batches "524288 2097152 8388608" \
  --alphas "1.0 0.5" \
  --lrs "0.005 0.0075 0.01 0.015 0.02 0.03 0.04" \
  --nproc 8
```

Outputs go to the Modal volume `sigma-search-runs` under:

```text
/outputs/search_evals/<stamp>
/outputs/logs/<stamp>
```

## Baseline Optimizer Sweep

```bash
MODAL_GPU=B200:8 modal run modal/run_sweep.py::optimizer_baselines \
  --depth 12 \
  --chinchilla-mult 2 \
  --nproc 8
```

This uses `plain_muon adamw soap shampoo kl_shampoo kl_soap` with the same
batch/token defaults as the handoff scripts, but the default run is grouped by
optimizer family because the LR scales differ:

| group | methods | default LR grid |
|---|---|---|
| `plain_muon` | `plain_muon` | `0.01 0.02 0.03 0.04 0.06 0.08` |
| `adamw` | `adamw` | `0.00025 0.0005 0.001 0.002 0.004` |
| `soap_klsoap` | `soap kl_soap` | `0.002 0.004 0.006 0.008 0.012 0.016 0.024` |
| `kl_shampoo` | `kl_shampoo` | `0.002 0.004 0.008 0.015 0.03` |
| `shampoo` | `shampoo` | `0.0005 0.001 0.002 0.004 0.008` |

Use `--groups "plain_muon soap_klsoap"` to run a subset. Use `--no-grouped`
with `--methods` and `--lrs` only for a deliberate single-grid ablation. It
defaults to `--weight-decay 0.1` for non-streaming optimizer baselines; pass
`--weight-decay 0.28` only for strict Top-Aware recipe matching.

## Generic Command

For debugging, run an arbitrary command inside the Modal image:

```bash
MODAL_GPU=B200:1 modal run modal/run_sweep.py::command \
  --cmd "TOKENS=1048576 METHODS=top_aware_muon BATCHES=262144 ALPHAS=1.0 LRS=0.005 NPROC=1 DRY_RUN=1 bash scripts/run_d12_sweep.sh"
```
