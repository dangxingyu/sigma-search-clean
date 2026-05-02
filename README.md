# Sigma-Search Clean

Standalone handoff repo for StreamingMuon-family optimizer experiments on top
of an in-tree `nanochat/` checkout. The current research question is:

> Under the same nanochat training recipe, when does Top-Aware Muon with
> `alpha=0.5` beat its same-candidate identity setting `alpha=1.0` across
> batch size and LR?

The active runtime surface is intentionally small:

```text
run_eval.py                  single train/eval entrypoint
run_top_aware_muon_sweep.py  reusable sweep engine
streaming_muon_torch.py      StreamingMuon optimizer implementation
metric_logging.py            opt-in dynamics and Hessian metrics
candidates/top_aware_muon.py Top-Aware Muon transform
scripts/run_d12_sweep.sh     optimizer-quality sweep wrapper
scripts/run_d12_statistics.sh metrics/statistics wrapper
```

Historical native Muon/LITE results may still exist under `docs/`, `figures/`,
and `results/`, but the clean handoff scripts run the current alpha sweep via
`top_aware_muon`.

## Basic Setup

Run setup once from the repo root:

```bash
bash scripts/setup_env.sh
source nanochat/.venv/bin/activate
export PYTHONPATH="$PWD:$PWD/nanochat"
bash scripts/download_climbmix.sh 170 8
```

The tokenizer and tokenized CLIMB-mix shards come from nanochat data prep, not
from a static file committed in this repo. If you already ran nanochat data
prep elsewhere, that is fine; make sure `NANOCHAT_BASE_DIR` points to the same
data directory before launching training.

Run a cheap sanity check:

```bash
bash scripts/smoke_run.sh
```

Run training commands inside whatever GPU allocation your cluster provides.
The repo deliberately does not ship SLURM/Ray/Kubernetes submitters; wrap the
same scripts with your local scheduler.

## Sweep

Use sweeps for optimizer-quality comparisons. Dense metrics and Hessian logging
are off by default.

### Main d12 Sweep

```bash
STAMP=d12_main_001 bash scripts/run_d12_sweep.sh
```

Default recipe:

| knob | default |
|---|---|
| methods | `top_aware_muon` |
| depth | `12` |
| token budget | `CHINCHILLA_MULT=1`, auto-resolved from `DEPTH` |
| batches | `262144 1048576 4194304` |
| LRs | `0.005 0.01 0.02 0.04` |
| Top-Aware | `top_k=1`, `alpha=1.0 0.5` |
| distributed | `NPROC=8`, max device batch size `16` |
| StreamingMuon | `--pure-qr --streaming-num-iters 2 --fallback-ortho-tol 0.01` |
| checkpointing | `SAVE_EVERY=100`, `KEEP_LAST_CHECKPOINTS=2`, `RESUME=1` |
| adaptive LR | on by default |

The wrapper hard-codes the token table below, so normal runs should specify
`DEPTH` and `CHINCHILLA_MULT` rather than a large raw token count.

| depth | 1x Chinchilla tokens |
|---:|---:|
| `8` | `402653184` |
| `12` | `1698693120` |
| `16` | `4026531840` |

These are the repo's 20-token-per-non-embedding-parameter budgets, rounded
where needed to stay compatible with the main batch grid. `TOKENS=...` is still
available for smoke tests or intentionally truncated custom runs.

Inspect commands without launching training:

```bash
DRY_RUN=1 bash scripts/run_d12_sweep.sh
```

### Common Overrides

```bash
STAMP=d12_c001_b262k \
BATCHES="262144" \
ALPHAS="1.0 0.5" \
LRS="0.005 0.01 0.02 0.04" \
SEEDS="42 43" \
bash scripts/run_d12_sweep.sh
```

`alpha=1.0` is the `c=1` identity baseline under the same
`top_aware_muon.py` implementation. `alpha=0.5` is the main Top-Aware setting.

For d8 / 0.4B-token runs, reuse the same script with overrides:

```bash
DEPTH=8 \
CHINCHILLA_MULT=1 \
BATCHES="262144 1048576 4194304" \
ALPHAS="1.0 0.5" \
LRS="0.005 0.01 0.02 0.04" \
ADAPTIVE_LR=1 \
bash scripts/run_d12_sweep.sh
```

For d16 / 4.0B-token runs, use the same wrapper:

```bash
DEPTH=16 \
CHINCHILLA_MULT=1 \
BATCHES="262144 1048576 4194304" \
ALPHAS="1.0 0.5" \
LRS="0.005 0.01 0.02 0.04" \
ADAPTIVE_LR=1 \
STAMP=d16_main_001 \
bash scripts/run_d12_sweep.sh
```

For a smaller multiple, change only the multiplier, for example
`CHINCHILLA_MULT=0.5`. For a quick command/path smoke, override exact tokens:

```bash
DEPTH=16 TOKENS=16777216 STAMP=d16_smoke bash scripts/run_d12_sweep.sh
```

### Adaptive LR

`ADAPTIVE_LR=1` is on by default for `run_d12_sweep.sh`. This is not an
optimizer LR schedule; it is a sweep-grid boundary extension:

1. Run the initial LR grid.
2. For each `(method, batch, seed, top_k, alpha)`, choose the lowest validation BPB.
3. If the best LR is the lowest grid point, run one lower LR.
4. If the best LR is the highest grid point, run one higher LR.
5. Repeat up to `MAX_LR_EXTENSION_ROUNDS`.

Defaults:

```bash
LR_EXTEND_FACTOR=2.0
LR_MIN=0.0005
LR_MAX=0.16
MAX_LR_EXTENSION_ROUNDS=2
```

Disable it when you want an exact fixed grid:

```bash
ADAPTIVE_LR=0 bash scripts/run_d12_sweep.sh
```

### Resume And Outputs

Use a stable `STAMP` for preemption-safe jobs. Re-submit the exact same command
after preemption:

```bash
STAMP=d12_main_001 bash scripts/run_d12_sweep.sh
STAMP=d12_main_001 bash scripts/run_d12_sweep.sh
```

Completed cases with valid `result.json` are skipped. Incomplete cases resume
from the latest complete checkpoint if checkpointing is enabled. The sweep
manifest stores a config signature; reusing an `OUT_ROOT` with changed recipe
knobs fails fast instead of silently mixing old and new results. Use a new
`STAMP` for changed recipes.

Outputs:

```text
search_evals/<stamp>/manifest.json
search_evals/<stamp>/top_aware_sweep_rows.csv
search_evals/<stamp>/<case>/result.json
search_evals/<stamp>/<case>/checkpoints/      if SAVE_EVERY > 0
logs/<stamp>/<case>.log
```

## Metrics

Use metrics runs after a sweep has identified the batch/LR recipes worth
inspecting. Metrics runs should be fixed-recipe diagnostics, not LR searches.

```bash
METHODS="top_aware_muon" \
BATCHES="262144 1048576 4194304" \
ALPHAS="1.0 0.5" \
LRS="0.02" \
SEEDS="42" \
bash scripts/run_d12_statistics.sh
```

Default metrics recipe:

| knob | default |
|---|---|
| metrics frequency | `METRICS_EVERY=1` |
| Hessian frequency | `METRICS_HESSIAN_EVERY=50` |
| Hessian directions | `METRICS_HESSIAN_TOP_K=4` |
| Hessian Lanczos steps | `METRICS_HESSIAN_ITERS=6` |
| module regex | normal transformer attention/MLP matrix weights |
| module limits | `METRICS_MAX_MODULES=0`, `METRICS_HESSIAN_MAX_MODULES=0` |

If the best LR differs by method or batch, run separate fixed-recipe metrics
jobs:

```bash
METHODS="top_aware_muon" BATCHES="262144" ALPHAS="1.0" LRS="0.04" \
bash scripts/run_d12_statistics.sh

METHODS="top_aware_muon" BATCHES="262144" ALPHAS="0.5" LRS="0.02" \
bash scripts/run_d12_statistics.sh
```

For a short d8 metrics smoke:

```bash
DEPTH=8 \
TOKENS=52428800 \
BATCHES="262144" \
ALPHAS="1.0 0.5" \
LRS="0.02" \
METRICS_EVERY=1 \
METRICS_HESSIAN_EVERY=100 \
bash scripts/run_d12_statistics.sh
```

### What Gets Logged

Cheap per-step metrics:

| metric | meaning |
|---|---|
| `train/loss`, `train/loss_ema` | optimizer-step CE and debiased EMA |
| `train/lr_multiplier`, `train/muon_momentum` | scheduler state |
| `weight_norm/<module>` | RMS weight norm |
| `grad_norm/<module>` | RMS gradient norm |
| `momentum_after_nesterov_norm/<module>` | RMS of optimizer input `M'` |
| `momentum_after_nesterov_spectral_norm/<module>` | top cached StreamingMuon sigma |
| `muon_singular_values/<module>` | top cached sigma values |
| `streaming_sigma_values/<module>` | same sigma values, explicit for sigma analysis |

Hessian/projection metrics:

| metric | meaning |
|---|---|
| `sharpness/selected_subspace` | top Hessian eigenvalue in selected matrix subspace |
| `gradient_hessian_projection/selected_subspace` | signed `dot(e_H, G)` |
| `momentum_after_nesterov_hessian_projection/selected_subspace` | signed `dot(e_H, M')` |
| `gradient_hessian_alignment/selected_subspace` | cosine between Hessian directions and gradient |
| `momentum_after_nesterov_hessian_alignment/selected_subspace` | cosine between Hessian directions and `M'` |
| `gradient_projection_on_last_hessian_space_coefficients/selected_subspace` | coefficients `E^T g_t` in the last Hessian eigenspace |
| `gradient_projection_on_last_hessian_space_consecutive_pearson/selected_subspace` | Pearson correlation between consecutive coefficient vectors |
| `gradient_projection_on_last_hessian_top1_lag1_pearson/selected_subspace` | rolling lag-1 Pearson correlation for the top Hessian coefficient |
| `hessian_eigenvector_block_norm/<module>` | module contribution to global Hessian eigenvector |
| `alignment_between_covariance_hessian_at_k_th_component/<module>` | Hessian block alignment with cached StreamingMuon components |

Metric scope:

- Per-module optimizer metrics are gathered from the rank that owns each
  StreamingMuon parameter chunk.
- `train/loss` is averaged across DDP ranks.
- Hessian probes use post-update weights and one local microbatch per rank from
  the same optimizer step. Each rank computes local HVP blocks and averages
  them with `all_reduce`, so the HVP is distributed across ranks.
- The Hessian is still a representative-microbatch Hessian, not the full
  grad-accum optimizer-batch Hessian.
- The selected Hessian subspace defaults to all normal transformer attention/MLP
  matrix weights matched by `METRICS_MODULE_REGEX`; embeddings, LM head, and
  scalar parameters are not included.

Analyze metrics outputs:

```bash
python analysis/analyze_metrics_dynamics.py
```

This writes `results/metrics_dynamics_analysis/` with per-step CSVs, summary
tables, and plots for sharpness, Hessian alignment, projection correlation,
optimizer-state norms, and validation BPB deltas.

## Results Catalog

Each sweep root contains machine-readable JSON/CSV outputs. To rebuild the
historical consolidated catalog:

```bash
python analysis/build_sweep_catalog.py
```

Curated summaries live under:

```text
results/sweep_catalog/
results/recipes/
figures/
```

Compare results within the same recipe family first. Historical single-GPU
native rows and newer same-driver StreamingMuon DDP rows can have different
absolute BPB.
