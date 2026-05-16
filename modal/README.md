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

This uses the same batch/token defaults as the handoff scripts. The default
grouped run is the compact core handoff set `plain_muon adamw kl_shampoo`;
optional structured groups remain available by overriding `--groups`.

| group | default | methods | LR setting |
|---|---:|---|---|
| `plain_muon` | yes | `plain_muon` | one alpha=`1.0` best LR per depth/batch |
| `adamw` | yes | `adamw` | `0.0005 0.001 0.002 0.003 0.004 0.005 0.0075` |
| `kl_shampoo` | yes | `kl_shampoo` | `0.002 0.004 0.006 0.008 0.012 0.016` |
| `soap_klsoap` | opt-in | `soap kl_soap` | `0.001 0.002 0.003 0.004 0.006 0.008 0.012` |
| `shampoo` | opt-in | `shampoo` | `0.002 0.004 0.005 0.0075 0.01 0.015 0.02` |

For `plain_muon`, d12 uses `{512K: 0.0075, 2M: 0.015, 8M: 0.01}` and d16
uses `{512K: 0.005, 2M: 0.005, 8M: 0.015}`. Set `PLAIN_MUON_LR=<lr>` only for
a deliberate custom one-LR Muon run.

Use `--groups "plain_muon adamw kl_shampoo shampoo"` to include optional
ordinary Shampoo. Use `--no-grouped`
with `--methods` and `--lrs` only for a deliberate single-grid ablation. It
defaults to fixed-grid mode with adaptive LR disabled, and uses
`--weight-decay 0.1` for non-streaming optimizer baselines. Pass
`--adaptive-lr` only if boundary-extension jobs are supported; pass
`--weight-decay 0.28` only for strict Top-Aware recipe matching.

## Generic Command

For debugging, run an arbitrary command inside the Modal image:

```bash
MODAL_GPU=B200:1 modal run modal/run_sweep.py::command \
  --cmd "TOKENS=1048576 METHODS=top_aware_muon BATCHES=262144 ALPHAS=1.0 LRS=0.005 NPROC=1 DRY_RUN=1 bash scripts/run_d12_sweep.sh"
```
