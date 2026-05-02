# Sigma-Search Clean

Standalone handoff repo for Muon-family optimizer experiments on nanochat-style LLM pretraining. The main question this repo supports is:

> At fixed model/data recipe, how do native Muon, StreamingMuon identity, native LITE, and Top-Aware Muon compare across batch size, LR, and Top-Aware `alpha`?

The code is self-contained: it includes `nanochat/`, optimizer implementations, sweep launchers, metric logging, and curated result folders.

## Quick Start

```bash
bash scripts/setup_env.sh
source nanochat/.venv/bin/activate
export PYTHONPATH="$PWD:$PWD/nanochat"
bash scripts/download_climbmix.sh 170 8
```

Run a cheap sanity check:

```bash
bash scripts/smoke_run.sh
```

Run the recommended handoff sweep inside an allocation:

```bash
srun --jobid=<JOBID> --overlap --ntasks=1 bash scripts/run_handoff_sweep.sh
```

Outputs go to:

```text
search_evals/handoff_sweep_<stamp>/      result JSONs, manifest, CSV, adaptive trace
logs/handoff_sweep_<stamp>/              stdout/stderr logs per run
```

## Repository Layout

```text
nanochat/                    vendored nanochat source
candidates/                  sigma transforms: identity, LITE-like, Top-Aware
run_eval.py                  StreamingMuon candidate train/eval runner
run_top_aware_muon_sweep.py  main reusable sweep engine
run_native_muon_v9.py        native Muon baseline runner
run_lite_v9.py               native LITE baseline runner; single-process only
metric_logging.py            opt-in optimizer dynamics metrics
scripts/run_handoff_sweep.sh user-facing sweep wrapper
scripts/run_d8_metrics_grid.sh canonical dense-metrics dynamics run
results/                     curated JSON/CSV summaries for pass-by
docs/                        experiment plan, log, and current conclusions
```

Generated folders such as `search_evals/`, `logs/`, `wandb/`, checkpoints, and data shards are ignored by git.

## Optimizers

Muon-like methods act on Nesterov-corrected momentum:

```text
M_t  = beta M_{t-1} + (1 - beta) G_t
M'_t = beta M_t + (1 - beta) G_t
M'_t = U Sigma V^T
Muon update = U V^T
```

StreamingMuon approximates the same decomposition with a warm-started basis and then applies a spectral transform:

```text
V_t     = streaming_power_iteration(M'_t, V_{t-1})
R_t     = M'_t V_t
sigma_i = ||R_t[:, i]||_2
U_i     = R_t[:, i] / sigma_i
update  = U diag(f(sigma)) V^T
```

Supported sweep method names:

| Method | Runner | Meaning |
|---|---|---|
| `streaming_identity` | `run_eval.py + candidates/identity.py` | StreamingMuon with `f(sigma)=1` |
| `top_aware_muon` | `run_eval.py + candidates/top_aware_muon.py` | scale top-`k` singular directions by `alpha` |
| `native_muon` | `run_native_muon_v9.py` | nanochat native Muon control |
| `native_lite` | `run_lite_v9.py` | native LITE control; use single process only |
| `streaming_lite` | `run_eval.py + candidates/lite_chi2_rs01.py` | implementation sanity check, not a main baseline |

Stable StreamingMuon DDP settings are:

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

Current clean recipe: keep `top_k=1`; sweep `alpha`, batch size, and LR. The sweep runner rejects `top_k != 1` unless `--allow-top-k-sweep` is explicitly passed.

## Recommended Sweep

For optimizer-quality comparisons, keep dense metrics off and sweep LR carefully. The handoff wrapper exposes common knobs as environment variables:

```bash
METHODS="streaming_identity native_muon top_aware_muon" \
BATCHES="262144 524288 1048576 2097152" \
ALPHAS="0.5 0.75 1.0 1.25" \
LRS="0.005 0.01 0.02 0.04" \
SEEDS="42" \
TOKENS=402653184 \
ADAPTIVE_LR=1 \
srun --jobid=<JOBID> --overlap --ntasks=1 bash scripts/run_handoff_sweep.sh
```

Defaults use d8, seq1024, 8 GPUs, max device batch size 16, and `402,653,184` tokens. That is the d8 Chinchilla-style `~0.4B` token recipe.

### Adaptive LR Behavior

`scripts/run_handoff_sweep.sh` calls `run_top_aware_muon_sweep.py --adaptive-lr` by default.

For each independent group `(method, batch, seed, top_k, alpha)`:

1. Run the initial LR grid.
2. Pick the best finite validation BPB; lower is better.
3. If the best LR is the lowest grid point, run `lr / LR_EXTEND_FACTOR`.
4. If the best LR is the highest grid point, run `lr * LR_EXTEND_FACTOR`.
5. Repeat up to `MAX_LR_EXTENSION_ROUNDS`, stopping at `LR_MIN`, `LR_MAX`, or the first failed outward run.

Useful knobs:

```bash
LR_EXTEND_FACTOR=2.0
LR_MIN=0.0005
LR_MAX=0.08
MAX_LR_EXTENSION_ROUNDS=2
ADAPTIVE_MIN_EDGE_IMPROVEMENT=0.0
```

Dry-run without launching training:

```bash
DRY_RUN=1 bash scripts/run_handoff_sweep.sh
```

## Dynamics Metrics

For dynamics studies, enable logging on a small number of modules. Metrics are off by default because every-step SVD diagnostics are expensive.

Recommended dense-metrics smoke:

```bash
METRICS_EVERY=1 \
METRICS_SPLIT_MOMENTUM=1 \
METRICS_HESSIAN_EVERY=100 \
BATCHES="262144" ALPHAS="1.0" LRS="0.02" TOKENS=52428800 \
srun --jobid=<JOBID> --overlap --ntasks=1 bash scripts/run_handoff_sweep.sh
```

Logged metric families:

| Family | What is logged |
|---|---|
| loss | raw train loss per optimizer step, EMA train loss, validation BPB |
| norms | per-module weight, gradient, momentum, and `M'` RMS |
| spectrum | momentum spectral norm, `M'` spectral norm, top singular values, StreamingMuon sigma |
| split alignment | split-half SVD alignment of global DDP-averaged `M'` |
| Hessian | approximate rank0-local block sharpness and Hessian-gradient/momentum/component alignments |

Hessian probes use post-update weights and a representative rank0 microbatch from the same optimizer step. Split alignment is global DDP-averaged; Hessian is intentionally local because global HVP would be much more expensive. Set `NANOCHAT_FORCE_MATH_SDPA=1` when Hessian probes are enabled.

## D8 Recipe Table

| batch | role | steps at 0.4B tokens | device batch | grad accum | base matrix LR | effective matrix LR |
|---:|---|---:|---:|---:|---:|---:|
| `262144` | d8 critical batch | `1536` | `16` | `2` | `0.02` | `0.01414` |
| `524288` | small/medium | `768` | `16` | `4` | `0.02` | `0.02000` |
| `1048576` | medium | `384` | `16` | `8` | `0.02` | `0.02828` |
| `2097152` | medium/large | `192` | `16` | `16` | `0.02` | `0.04000` |
| `4194304` | large | `96` | `16` | `32` | `0.02` | `0.05657` |

Effective matrix LR follows nanochat batch scaling:

```text
lr_eff = matrix_lr * sqrt(batch / 524288)
```

## Results

Each run writes `result.json`. The sweep root also writes:

```text
manifest.json                exact sweep configuration
top_aware_sweep_rows.csv     one row per attempted run
adaptive_lr_trace.json       LR boundary-extension decisions
```

Rebuild the historical consolidated catalog:

```bash
python scripts/build_sweep_catalog.py
```

Curated catalog files:

```text
results/sweep_catalog/all_sweeps.csv
results/sweep_catalog/best_by_batch_method.csv
results/sweep_catalog/summary.json
```

When comparing results, compare within the same recipe family first. Historical single-GPU native rows and newer same-driver StreamingMuon DDP rows can have different absolute BPB.

## Caveats

- `native_lite` is not DDP-safe here. Do not include it in 8-GPU DDP sweeps until a distributed LITE optimizer is implemented.
- `streaming_lite` is not a primary claim baseline; it is mainly an implementation sanity check.
- Dense metrics can be much slower than no-metrics training. Report measured `eval_time_seconds` from `result.json`.
- New d8 dynamics runs should use the `0.4B` recipe unless the question explicitly needs longer training.
