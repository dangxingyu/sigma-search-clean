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
scripts/run_d12_d16_2x_grid.sh canonical sequential d12/d16 handoff grid
scripts/run_d12_d16_alpha_sweep.sh multi-alpha d12/d16 handoff grid
scripts/submit_slurm_grid.sh SLURM array submitter for fixed-grid cases
scripts/run_d12_statistics.sh metrics/statistics wrapper
scripts/run_d12_d16_metrics_best.sh curated d12/d16 best-point metrics jobs
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

Run training commands inside whatever GPU allocation your cluster provides, or
use the included SLURM array submitter if the cluster supports `sbatch`.

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
| token budget | `CHINCHILLA_MULT=2`, auto-resolved from `DEPTH` |
| batches | `262144 1048576 4194304` |
| LRs | `0.005 0.0075 0.01 0.015 0.02 0.03 0.04` |
| Top-Aware | `top_k=1`, `alpha=1.0 0.5` |
| distributed | `NPROC=8`, max device batch size `16` |
| StreamingMuon | `--pure-qr --streaming-num-iters 2 --fallback-ortho-tol 0.01` |
| checkpointing | `SAVE_EVERY=100`, `KEEP_LAST_CHECKPOINTS=2`, `RESUME=1` |
| adaptive LR | on by default |

The wrapper hard-codes the 1x token table below. Normal runs should specify
`DEPTH` and `CHINCHILLA_MULT` rather than a large raw token count. The default
handoff recipe uses `CHINCHILLA_MULT=2` because 1x can leave large-batch runs
with too few optimizer steps.

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
LRS="0.005 0.0075 0.01 0.015 0.02 0.03 0.04" \
SEEDS="42 43" \
bash scripts/run_d12_sweep.sh
```

`alpha=1.0` is the `c=1` identity baseline under the same
`top_aware_muon.py` implementation. `alpha=0.5` is the main Top-Aware setting.

For d8 runs, reuse the same script with overrides:

```bash
DEPTH=8 \
CHINCHILLA_MULT=2 \
BATCHES="262144 1048576 4194304" \
ALPHAS="1.0 0.5" \
LRS="0.005 0.0075 0.01 0.015 0.02 0.03 0.04" \
ADAPTIVE_LR=1 \
bash scripts/run_d12_sweep.sh
```

For d16 runs, use the same wrapper:

```bash
DEPTH=16 \
CHINCHILLA_MULT=2 \
BATCHES="262144 1048576 4194304" \
ALPHAS="1.0 0.5" \
LRS="0.005 0.0075 0.01 0.015 0.02 0.03 0.04" \
ADAPTIVE_LR=1 \
STAMP=d16_main_001 \
bash scripts/run_d12_sweep.sh
```

For a different budget, change only the multiplier, for example
`CHINCHILLA_MULT=1` or `CHINCHILLA_MULT=8`. For a quick command/path smoke,
override exact tokens:

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

### Canonical Handoff Grid

If you want one safe bash command for the current main study, run:

```bash
STAMP_PREFIX=handoff_main_001 bash scripts/run_d12_d16_2x_grid.sh
```

This sequentially runs:

| knob | value |
|---|---|
| depths | `12 16` |
| token budget | `CHINCHILLA_MULT=2` |
| batches | `524288 2097152 8388608` |
| alphas | `1.0 0.5` |
| LRs | `0.005 0.0075 0.01 0.015 0.02 0.03 0.04` plus adaptive boundary closure |
| seeds | `42` |

It writes separate sweep roots, for example
`search_evals/handoff_main_001_d12_2x/` and
`search_evals/handoff_main_001_d16_2x/`, because d12 and d16 have different
token budgets.

### d12/d16 Alpha Sweep

For Sadhika's broader alpha sweep, use:

```bash
STAMP_PREFIX=sadhika_alpha_2x_001 bash scripts/run_d12_d16_alpha_sweep.sh
```

Default grid:

| knob | value |
|---|---|
| depths | `12 16` |
| token budget | `CHINCHILLA_MULT=2` |
| batches | `524288 2097152 8388608` |
| alphas | `0.25 0.5 0.75 0.85 1.0 1.15` |
| LRs | `0.005 0.0075 0.01 0.015 0.02 0.03 0.04` plus adaptive boundary closure |
| seeds | `42` |

Detailed copy-paste instructions, including the SLURM-array version and files
to send back, are in `docs/alpha_sweep_handoff.md`.

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

### SLURM Array Grid

For preemptible SLURM clusters, submit the fixed base grid as separate jobs:

```bash
STAMP=d12_main_001 \
BATCHES="524288 2097152 8388608" \
ALPHAS="1.0 0.5" \
CHINCHILLA_MULT=2 \
MAX_PARALLEL=8 \
SBATCH_PARTITION=<partition> \
SBATCH_ACCOUNT=<account> \
SBATCH_GPU_ARG="--gres=gpu:8" \
bash scripts/submit_slurm_grid.sh
```

Set `SBATCH_GPU_ARG="--gpus-per-node=8"` instead if that is the local GPU
syntax. Leave `STAMP` unchanged when re-submitting after preemption. Completed
cases with valid `result.json` are skipped; incomplete cases resume from
`<case>/checkpoints/`.

After the array finishes, run one sequential cleanup command:

```bash
STAMP=d12_main_001 \
BATCHES="524288 2097152 8388608" \
ALPHAS="1.0 0.5" \
CHINCHILLA_MULT=2 \
bash scripts/run_d12_sweep.sh
```

This rebuilds `top_aware_sweep_rows.csv`, skips completed base-grid cases, and
runs adaptive LR boundary closure if `ADAPTIVE_LR=1`. The array tasks
intentionally do not run adaptive closure independently because boundary
extension depends on the full LR grid being complete.

Inspect the array size and exact `sbatch` command without submitting:

```bash
DRY_RUN=1 \
STAMP=d12_main_001 \
BATCHES="524288 2097152 8388608" \
ALPHAS="1.0 0.5" \
CHINCHILLA_MULT=2 \
bash scripts/submit_slurm_grid.sh
```

Submit d12 and d16 as two separate arrays by changing only `DEPTH` and `STAMP`,
for example `DEPTH=12 STAMP=d12_main_001 ...` and
`DEPTH=16 STAMP=d16_main_001 ...`.

## Metrics

Use metrics runs after a sweep has identified the batch/LR recipes worth
inspecting. Metrics runs should be fixed-recipe diagnostics, not LR searches.

For the completed d12 2x Chinchilla sweep, use the curated best-point wrapper:

```bash
PRINT_CASES=1 bash scripts/run_d12_metrics_best.sh

STAMP_PREFIX=sadhika_d12_metrics_001 \
bash scripts/run_d12_metrics_best.sh
```

This runs the six best sweep points sequentially:

| batch | alpha / c | LR |
|---:|---:|---:|
| `524288` | `1.0` | `0.0075` |
| `524288` | `0.5` | `0.0075` |
| `2097152` | `1.0` | `0.015` |
| `2097152` | `0.5` | `0.02` |
| `8388608` | `1.0` | `0.01` |
| `8388608` | `0.5` | `0.015` |

For the combined imported d12/d16 sweeps, use the curated combined wrapper:

```bash
PRINT_CASES=1 bash scripts/run_d12_d16_metrics_best.sh

STAMP_PREFIX=sadhika_d12_d16_metrics_001 \
bash scripts/run_d12_d16_metrics_best.sh
```

This exposes 12 cases with `CASE_INDEX=0..11`, automatically using
`MAX_DEVICE_BATCH_SIZE=32` for d12 and `MAX_DEVICE_BATCH_SIZE=16` for d16.
The d16 512K/2M cases are included for convenience but are marked in the case
table as lower-LR-boundary cases until the d16 lower-LR extension is complete.

For one job per case, submit an array over `CASE_INDEX=0..5` with a stable
`STAMP_PREFIX`, for example:

```bash
STAMP_PREFIX=sadhika_d12_metrics_001 \
CASE_INDEX="${SLURM_ARRAY_TASK_ID}" \
bash scripts/run_d12_metrics_best.sh
```

For the combined d12/d16 wrapper, submit an array over `CASE_INDEX=0..11` and
replace the script name with `scripts/run_d12_d16_metrics_best.sh`.

Checkpointing/resume is enabled by default. If preempted, rerun the same
`CASE_INDEX` with the same `STAMP_PREFIX`.

This curated wrapper defaults to `MAX_DEVICE_BATCH_SIZE=32`, based on d12
8xB200 smoke tests with full Hessian settings
`METRICS_HESSIAN_TOP_K=4, METRICS_HESSIAN_ITERS=6`. With sequence length 1024,
each grad-accum microbatch contains 262K distributed tokens:

```text
8 GPUs * 32 sequences/GPU * 1024 tokens = 262144 tokens
```

The Hessian probe now accumulates all grad-accum microbatches from the current
optimizer step. For example, a 2M global batch with the default d12 metrics
microbatch uses 8 local HVP passes per Lanczos vector.

`MAX_DEVICE_BATCH_SIZE=64` and `128` fit cheap training/optimizer metrics on
8xB200, but OOM in the full Hessian top4/iters6 smoke. For conservative
hardware, set `MAX_DEVICE_BATCH_SIZE=16`.

For d16 full-Hessian metrics on 8xB200, use `MAX_DEVICE_BATCH_SIZE=16`.
The d16 smoke with `MAX_DEVICE_BATCH_SIZE=32` completed the training step but
OOMed inside the Hessian top4/iters6 probe; the same smoke with
`MAX_DEVICE_BATCH_SIZE=16` completed and logged sharpness/alignment metrics.

For d16 fixed-recipe metrics, use the generic statistics wrapper with explicit
d16 overrides after the d16 sweep identifies best LRs:

```bash
DEPTH=16 \
MAX_DEVICE_BATCH_SIZE=16 \
METHODS="top_aware_muon" \
BATCHES="524288" \
ALPHAS="1.0 0.5" \
LRS="0.0075" \
SEEDS="42" \
STAMP=d16_metrics_example \
bash scripts/run_d12_statistics.sh
```

Do not use `run_d12_metrics_best.sh` for d16 unless its hard-coded cases have
been updated from a completed d16 sweep; that wrapper currently contains d12
best points.

For custom fixed recipes, call the lower-level wrapper directly:

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

Notation used below:

```text
W_i      selected matrix parameter
G_i      gradient for W_i at the logged optimizer step
M'_i     optimizer input after Nesterov momentum, cached by StreamingMuon
sigma_i cached StreamingMuon singular-value estimates for M'_i
S        selected matrix subspace matched by METRICS_MODULE_REGEX
E        [e_1, ..., e_k], top Hessian directions in S from Lanczos HVP
rms(X)   sqrt(mean(X^2))
cos(a,b) <a,b> / (||a|| ||b||)
```

Cheap per-step metrics:

| metric | meaning | definition / pseudocode |
|---|---|---|
| `train/loss` | optimizer-step CE | `mean(CE)` over grad-accum microbatches, DDP-averaged |
| `train/loss_ema` | smoothed train CE | debiased EMA of `train/loss` |
| `train/lr_multiplier`, `train/muon_momentum` | scheduler state | current LR multiplier and Muon momentum coefficient |
| `weight_norm/<module>` | RMS weight norm | `rms(W_i)` |
| `grad_norm/<module>` | RMS gradient norm | `rms(G_i)` |
| `momentum_after_nesterov_norm/<module>` | RMS optimizer input | `rms(M'_i)` |
| `momentum_after_nesterov_spectral_norm/<module>` | top cached sigma | `max_j sigma_i[j]` |
| `muon_singular_values/<module>` | top cached sigma values | `sort_desc(sigma_i)[:METRICS_TOP_K]` |
| `streaming_sigma_values/<module>` | same sigma values, explicit for sigma analysis | alias of cached `sort_desc(sigma_i)[:METRICS_TOP_K]` |

Hessian/projection metrics:

| metric | meaning | definition / pseudocode |
|---|---|---|
| `sharpness/selected_subspace` | top Hessian eigenvalue in selected matrix subspace | `lambda_1` from Lanczos on `H_S v = grad_S(<grad_S L, v>)` |
| `gradient_hessian_projection/selected_subspace` | signed gradient projection on Hessian directions | `[<e_j, G_S> for e_j in E]` |
| `momentum_after_nesterov_hessian_projection/selected_subspace` | signed optimizer-input projection on Hessian directions | `[<e_j, M'_S> for e_j in E]` |
| `gradient_hessian_alignment/selected_subspace` | gradient cosine with Hessian directions | `[cos(e_j, G_S) for e_j in E]` |
| `momentum_after_nesterov_hessian_alignment/selected_subspace` | optimizer-input cosine with Hessian directions | `[cos(e_j, M'_S) for e_j in E]` |
| `gradient_projection_on_last_hessian_space_coefficients/selected_subspace` | current gradient coordinates in last Hessian space | `c_t = E_last^T G_t` |
| `gradient_projection_on_last_hessian_space_norm/selected_subspace` | norm of projected gradient | `norm2(E_last E_last^T G_t) = norm2(c_t)` |
| `gradient_projection_on_last_hessian_space_top1_abs_fraction/selected_subspace` | top-1 coordinate dominance | `abs(c_t[0]) / norm2(c_t)` |
| `gradient_projection_on_last_hessian_space_consecutive_pearson/selected_subspace` | consecutive coefficient-vector correlation | `pearson(c_{t-1}, c_t)` for the same fixed `E_last`; undefined for `top_k=1` |
| `gradient_projection_on_last_hessian_top1_lag1_pearson/selected_subspace` | signed top-1 projection oscillation statistic | rolling `pearson([c_{s,1}], [c_{s+1,1}])` over `METRICS_PROJECTION_CORRELATION_WINDOW` |
| `hessian_eigenvector_block_norm/<module>` | module's share of global Hessian direction | `norm2(e_1[module])` |
| `alignment_between_covariance_hessian_at_k_th_component/<module>` | Hessian block alignment with cached StreamingMuon component | `abs(cos(e_j[module], q_j[module]))`, where `q_j` is reconstructed from cached StreamingMuon basis/sigma |

Metric scope:

- Per-module optimizer metrics are gathered from the rank that owns each
  StreamingMuon parameter chunk.
- `train/loss` is averaged across DDP ranks.
- Hessian probes use post-update weights and all grad-accum microbatches from
  the same optimizer step. For a fixed Lanczos vector `v`, each rank computes
  `H_b v` on each local microbatch, accumulates by token count, then
  `all_reduce(SUM)` averages by total distributed tokens.
- With `MAX_DEVICE_BATCH_SIZE=16`, 8 GPUs, and sequence length 1024, each
  microbatch contributes 128K tokens; a 512K global batch therefore uses four
  local microbatches per rank in the Hessian estimate.
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
