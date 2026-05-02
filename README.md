# Sigma-Search Clean

Standalone handoff repo for Muon-family optimizer experiments on nanochat-style LLM pretraining. It contains the vendored `nanochat/` source, StreamingMuon, native Muon/LITE controls, Top-Aware Muon, metric logging, curated result summaries, and reusable sweep scripts.

## Layout

```text
nanochat/                    vendored nanochat source
candidates/                  sigma transforms: identity, LITE-like, Top-Aware
run_eval.py                  StreamingMuon candidate train/eval entrypoint
run_top_aware_muon_sweep.py  reusable batch x alpha x LR runner
run_native_muon_v9.py        native Muon baseline
run_lite_v9.py               native LITE baseline, single-process only
metric_logging.py            opt-in dynamics diagnostics
analysis/                    plotting and historical result parsers
scripts/                     setup, data download, catalog, canonical launchers
results/                     curated JSON/CSV summaries for pass-by
figures/                     copied summary plots/tables
docs/                        research notes and experiment logs
```

Generated outputs such as `search_evals/`, `logs/`, `wandb/`, checkpoints, and data shards are intentionally ignored.

## Setup

```bash
bash scripts/setup_env.sh
source nanochat/.venv/bin/activate
export PYTHONPATH="$PWD:$PWD/nanochat"
bash scripts/download_climbmix.sh 170 8
```

Run a tiny sanity check:

```bash
bash scripts/smoke_run.sh
```

## Optimizers

For a matrix parameter, Muon acts on Nesterov-corrected momentum:

```text
M_t  = beta M_{t-1} + (1 - beta) G_t
M'_t = beta M_t + (1 - beta) G_t
M'_t = U Sigma V^T
Muon update = U V^T
```

StreamingMuon approximates the same SVD with a warm-started basis and applies a spectral transform:

```text
V_t     = streaming_power_iteration(M'_t, V_{t-1})
R_t     = M'_t V_t
sigma_i = ||R_t[:, i]||_2
U_i     = R_t[:, i] / sigma_i
update  = U diag(f(sigma)) V^T
```

Current methods:

| Method | Runner | Definition |
|---|---|---|
| `streaming_identity` | `run_eval.py + candidates/identity.py` | `f(sigma)=1`; StreamingMuon identity baseline |
| `streaming_lite` | `run_eval.py + candidates/lite_chi2_rs01.py` | same-driver LITE-like sanity baseline |
| `top_aware_muon` | `run_eval.py + candidates/top_aware_muon.py` | scale top-`k` singular directions by `alpha`, others by `1` |
| `native_muon` | `run_native_muon_v9.py` | nanochat/native Muon control |
| `native_lite` | `run_lite_v9.py` | native LITE control; use `nproc_per_node=1` only |

Stable StreamingMuon DDP settings:

```text
--pure-qr --streaming-num-iters 2 --fallback-ortho-tol 0.01
```

## Top-Aware Muon

Top-Aware Muon is the current algorithm under study.

```python
def f(sigma, top_k=1, alpha=0.5):
    scale = ones_like(sigma)
    scale[topk(sigma, top_k)] = alpha
    return scale
```

Clean recipe: keep `top_k=1`; sweep `alpha`, batch size, and LR only when doing optimizer-quality comparisons. The runner rejects `top_k != 1` unless `--allow-top-k-sweep` is passed.

## D8 Metrics Recipe

For dynamics/logging studies, use the d8 Chinchilla-style budget of `402,653,184` tokens, about `0.4B`, not the older `1.07B` budget. This value is divisible by the three current batch sizes.

| batch | role | steps | device batch | grad accum | base matrix LR | effective matrix LR |
|---:|---|---:|---:|---:|---:|---:|
| `262144` | d8 critical batch | `1536` | `16` | `2` | `0.02` | `0.01414` |
| `1048576` | medium batch | `384` | `16` | `8` | `0.02` | `0.02828` |
| `4194304` | large batch | `96` | `16` | `32` | `0.02` | `0.05657` |

The effective LR follows nanochat's batch scaling: `lr_eff = matrix_lr * sqrt(batch / 524288)`.

Canonical no-tuning metrics grid:

```bash
bash scripts/run_d8_metrics_grid.sh
```

This runs `top_aware_muon` with `alpha in {0.5, 1.0}`, `top_k=1`, `lr=0.02`, batches `{262K, 1M, 4M}`, d8, seq1024, 8 GPUs, full per-step cheap metrics, and Hessian probes every 50 logged steps. `alpha=1.0` is the identity/Muon-like control through the same candidate path.

## Metric Logging

`run_eval.py` and `run_native_muon_v9.py` support opt-in logging. Normal runs do not cache tensors unless `--metrics-every` is set.

Recommended dynamics flags:

```bash
--metrics-every 1 \
--metrics-top-k 4 \
--metrics-max-modules 8 \
--metrics-split-momentum \
--metrics-alignment-side lite \
--metrics-hessian-every 50 \
--metrics-hessian-top-k 1 \
--metrics-hessian-iters 2 \
--metrics-hessian-max-modules 8
```

Set `NANOCHAT_FORCE_MATH_SDPA=1` when Hessian probes are enabled; flash/mem-efficient attention kernels may not support double backward.

Logged quantities include:

| Metric family | Contents |
|---|---|
| loss | train loss, validation BPB/loss |
| norms | per-module weight, gradient, momentum, Nesterov momentum RMS |
| spectrum | momentum and Nesterov spectral norm, top singular values, StreamingMuon sigma |
| split alignment | side-aware split-half momentum SVD alignment, using `V` for tall and `U` for wide matrices |
| Hessian probe | approximate sharpness and Hessian-gradient/momentum/component alignments |

## Sweeps And Results

Curated summaries live in `results/`. Rebuild the consolidated sweep catalog with:

```bash
python scripts/build_sweep_catalog.py
```

Main files:

```text
results/sweep_catalog/all_sweeps.csv
results/sweep_catalog/best_by_batch_method.csv
results/sweep_catalog/summary.json
results/sweep_catalog/README.md
```

Comparison rule: first compare rows within the same `family`. Historical single-GPU native rows and newer same-driver StreamingMuon DDP rows can have different absolute BPB.

Currently organized result sets:

```text
results/topaware_128k_near_identity_lr001/summary.json
results/topaware_128k_alpha125_lrsweep/summary.json
results/dynamics_128k_200step/summary.json
results/recipes/d8_metrics_grid_recipe.json
```

## Reusable Commands

Top-Aware alpha/LR sweep:

```bash
python run_top_aware_muon_sweep.py \
  --methods "streaming_identity native_muon top_aware_muon" \
  --batches "262144 1048576" \
  --alphas "0.5 0.75 1.0" \
  --lrs "0.005 0.01 0.02 0.04" \
  --seeds "42" \
  --tokens 402653184 \
  --depth 8 \
  --nproc-per-node 8
```

Single StreamingMuon identity run:

```bash
torchrun --standalone --nproc_per_node=8 run_eval.py \
  --candidate-file candidates/identity.py \
  --total-batch-size 262144 \
  --device-batch-size 16 \
  --max-seq-len 1024 \
  --max-steps 1536 \
  --matrix-lr 0.02 \
  --num-iters 2 --pure-qr --fallback-ortho-tol 0.01 \
  --output-file results/raw/identity_262k_lr002.json
```

Native Muon control:

```bash
torchrun --standalone --nproc_per_node=8 run_native_muon_v9.py \
  --total-batch-size 262144 \
  --device-batch-size 16 \
  --max-steps 1536 \
  --matrix-lr 0.02 \
  --output-file results/raw/native_muon_262k_lr002.json
```

Native LITE control, single-process only:

```bash
python run_lite_v9.py \
  --total-batch-size 262144 \
  --device-batch-size 16 \
  --max-steps 1536 \
  --matrix-lr 0.02 \
  --lite-chi 2 --lite-rs 0.1 --lite-chi-warmup 0.5 \
  --lite-chi-schedule warmup_hold \
  --output-file results/raw/native_lite_262k_lr002.json
```

## Caveats

- `native_lite` is not DDP-safe in this repo. Do not include it in 8-GPU DDP sweeps until a distributed LITE optimizer is implemented.
- Historical 1B-token results remain useful for context, but new d8 dynamics runs should use the `0.4B` recipe unless the question specifically requires longer training.
- Dense metric runs can be much slower than no-metrics runs. Use result JSON elapsed time to report measured overhead instead of assuming exactly `2x`.
