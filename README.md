# Sigma-Search Clean

Standalone handoff repo for StreamingMuon-family optimizer experiments on nanochat-style LLM pretraining. The main question this repo supports is:

> At fixed model/data recipe, how does Top-Aware Muon compare to the same-driver StreamingMuon identity baseline across batch size, LR, and Top-Aware `alpha`?

The code is self-contained: it includes `nanochat/`, StreamingMuon, Top-Aware Muon, sweep launchers, metric logging, and curated result folders. Current runnable sweeps intentionally expose only `streaming_identity` and `top_aware_muon`.

## Quick Start

If you only want one command for the main d12 optimizer-quality sweep, use a fixed `STAMP`:

```bash
STAMP=d12_main_001 bash scripts/run_d12_sweep.sh
```

This runs the current mainline comparison, `streaming_identity` vs Top-Aware Muon `alpha=0.5`, at d12 with the default batch/LR grid and a 1x Chinchilla-style token budget (`1698693120` tokens). Here 1x means `20 * non_embedding_params`; for nanochat d12, non-embedding params are `84,935,570`, then the token count is rounded to the nearest 4M-batch-compatible value. Run it from a node/session that already has the intended GPUs visible; wrap it with your local scheduler outside the repo if needed.

`STAMP` controls the output directory. If you omit it, the wrapper creates a timestamped directory, which is useful for one-off runs but not for preemption resume.

To inspect the exact expanded `torchrun` commands without launching training:

```bash
DRY_RUN=1 bash scripts/run_d12_sweep.sh
```

After the sweep identifies the LR/batch points to inspect, run dense statistics separately:

```bash
BATCHES="262144 1048576 4194304" LRS="0.02" bash scripts/run_d12_statistics.sh
```

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

Cluster launchers are intentionally not part of the command. If you use SLURM, Kubernetes, Ray, or another scheduler, wrap the same standalone script with your local allocation/launcher convention.

Tokenizer/data note: the tokenizer and tokenized CLIMB-mix shards come from nanochat's data prep, not from a static file committed in this repo. `scripts/download_climbmix.sh` calls `python -m nanochat.dataset`, which is the intended setup path. If you already ran nanochat's tokenizer/data preparation successfully, that is correct; make sure `NANOCHAT_BASE_DIR` points to the same data directory when launching training.

Outputs go to:

```text
search_evals/<stamp>/                    result JSONs, manifest, CSV, adaptive trace
logs/<stamp>/                            stdout/stderr logs per run
```

Checkpointing is enabled by default in the sweep wrappers. Each case writes resumable checkpoints under:

```text
search_evals/<stamp>/<case_name>/checkpoints/
```

The default is `SAVE_EVERY=100`, `KEEP_LAST_CHECKPOINTS=2`, and `RESUME=1`. Re-submit the same command with the same `STAMP`/`OUT_ROOT` after preemption: completed cases with valid `result.json` are skipped, and incomplete cases resume from their latest complete checkpoint. A complete DDP checkpoint requires `model_<step>.pt`, `meta_<step>.json`, and every `optim_<step>_rank*.pt`; half-written checkpoints are ignored.

Example preemption-safe pattern:

```bash
STAMP=d12_main_001 bash scripts/run_d12_sweep.sh
# if preempted, submit exactly the same command again
STAMP=d12_main_001 bash scripts/run_d12_sweep.sh
```

For many scheduler jobs, shard the grid explicitly and give each shard its own stable `STAMP`. Do not launch multiple jobs that write the same `OUT_ROOT` at the same time.

```bash
STAMP=d12_c001_b262k BATCHES="262144" bash scripts/run_d12_sweep.sh
STAMP=d12_c001_b1m   BATCHES="1048576" bash scripts/run_d12_sweep.sh
STAMP=d12_c001_b4m   BATCHES="4194304" bash scripts/run_d12_sweep.sh
```

If a shard is preempted, resubmit only that shard with the same `STAMP`. If you change core recipe knobs such as `DEPTH`, `TOKENS`, `BATCHES`, `LRS`, `ALPHAS`, or `SEEDS`, use a new `STAMP` rather than reusing an old checkpoint directory.

To submit all default d12 batch shards to SLURM at once, use the optional submitter:

```bash
SUBMIT_DRY_RUN=1 \
CAMPAIGN=d12_c001 \
SBATCH_PARTITION=<partition> \
SBATCH_ACCOUNT=<account> \
bash scripts/submit_d12_sweep_shards_slurm.sh
```

Remove `SUBMIT_DRY_RUN=1` after inspecting the generated sbatch files under `logs/slurm_submit/<campaign>/`. By default this submits one job per batch:

```text
d12_c001_b262k
d12_c001_b1m
d12_c001_b4m
```

Each job runs `scripts/run_d12_sweep.sh` with its own stable `STAMP`, so preemption resume is safe. If the scheduler does not automatically requeue jobs, rerun the same submitter command with the same `CAMPAIGN`; completed cases skip and incomplete cases resume. Do not rerun the submitter while the same campaign's jobs are still active.

Useful submitter overrides:

```bash
CAMPAIGN=d12_alpha_scan \
SHARD_BY=batch_alpha \
METHODS=top_aware_muon \
ALPHAS="0.25 0.5 0.75 1.0" \
TOP_KS="1" \
SBATCH_GPU_DIRECTIVE="--gres=gpu:8" \
bash scripts/submit_d12_sweep_shards_slurm.sh
```

`SHARD_BY=batch` is the default and avoids duplicating the `streaming_identity` baseline. If you shard by `batch_alpha` with `METHODS="streaming_identity top_aware_muon"` and multiple alphas, the identity baseline will be duplicated in each alpha shard.

## Repository Layout

```text
nanochat/                    vendored nanochat source
candidates/                  sigma transforms: identity and Top-Aware
run_eval.py                  StreamingMuon candidate train/eval runner
run_top_aware_muon_sweep.py  main reusable sweep engine
metric_logging.py            opt-in optimizer dynamics metrics
scripts/run_d12_sweep.sh     blessed d12 optimizer-quality sweep
scripts/run_d12_statistics.sh dense d12 dynamics/statistics runner
scripts/submit_d12_sweep_shards_slurm.sh optional SLURM shard submitter
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

For the standard d12 handoff sweep, use the standalone consolidated script. This is an optimizer-quality sweep: dense metrics and Hessian logging are off.

```bash
STAMP=d12_main_001 bash scripts/run_d12_sweep.sh
```

Useful overrides:

```bash
STAMP=d12_c001_b262k \
BATCHES="262144 1048576 4194304" \
ALPHAS="0.5 1.0" \
TOP_KS="1" \
LRS="0.005 0.01 0.02 0.04" \
SEEDS="42 43" \
TOKENS=1698693120 \
bash scripts/run_d12_sweep.sh
```

## Statistics Runs

Run statistics only after the sweep identifies the LR points worth inspecting. This keeps expensive Hessian/projection logging out of the main LR search.

```bash
METHODS="streaming_identity top_aware_muon" \
BATCHES="262144 1048576 4194304" \
ALPHAS="0.5" \
TOP_KS="1" \
LRS="0.02" \
SEEDS="42" \
bash scripts/run_d12_statistics.sh
```

The statistics wrapper uses the same d12 1x token budget as `run_d12_sweep.sh`, but enables `METRICS_EVERY=1` and `METRICS_HESSIAN_EVERY=50` by default. Override `LRS` and `BATCHES` to match the selected sweep winners. It intentionally sets `ADAPTIVE_LR=0`.

If the best LR differs by method or batch, run statistics in separate groups rather than forcing one shared LR, for example:

```bash
METHODS="streaming_identity" BATCHES="262144" LRS="0.04" bash scripts/run_d12_statistics.sh
METHODS="top_aware_muon" BATCHES="262144" ALPHAS="0.5" LRS="0.02" bash scripts/run_d12_statistics.sh
```

For d8 optimizer-quality comparisons, keep dense metrics off and sweep LR carefully. Reuse the standalone sweep script with d8 overrides:

```bash
METHODS="streaming_identity top_aware_muon" \
BATCHES="262144 1048576 4194304" \
ALPHAS="0.5" \
TOP_KS="1" \
LRS="0.005 0.01 0.02 0.04" \
SEEDS="42" \
DEPTH=8 \
TOKENS=402653184 \
ADAPTIVE_LR=1 \
bash scripts/run_d12_sweep.sh
```

Defaults use `METHODS="streaming_identity top_aware_muon"`, where `streaming_identity` is `c=1` and Top-Aware uses `alpha/c=0.5`. The default batches are `{262144,1048576,4194304}`, with d8, seq1024, 8 GPUs, max device batch size 16, and `402,653,184` tokens. That is the d8 Chinchilla-style `~0.4B` token recipe.

### Adaptive LR Behavior

`scripts/run_d12_sweep.sh` calls `run_top_aware_muon_sweep.py --adaptive-lr` by default. The statistics scripts keep adaptive LR off.

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
DRY_RUN=1 bash scripts/run_d12_sweep.sh
```

## Dynamics Metrics

For dynamics studies, enable logging on StreamingMuon runs. Metrics are off by default. The metrics path reuses StreamingMuon's cached `sigma` and basis from the optimizer step.

Recommended dense-metrics smoke:

```bash
METRICS_EVERY=1 \
METRICS_HESSIAN_EVERY=100 \
BATCHES="262144" ALPHAS="1.0" LRS="0.02" TOKENS=52428800 \
bash scripts/run_d8_metrics_grid.sh
```

Canonical logged metrics:

| Metric key | Meaning |
|---|---|
| `train/loss` | raw mean cross-entropy over the optimizer step's grad-accum microbatches |
| `train/loss_ema` | debiased EMA of `train/loss` |
| `train/lr_multiplier` | current LR schedule multiplier |
| `train/muon_momentum` | current Muon momentum schedule value |
| `weight_norm/{module}` | RMS of the module weight |
| `grad_norm/{module}` | RMS of the full-batch gradient |
| `momentum_after_nesterov_norm/{module}` | RMS of `M' = beta M_t + (1 - beta) G_t` |
| `momentum_after_nesterov_spectral_norm/{module}` | top cached StreamingMuon `sigma` for `M'` |
| `muon_singular_values/{module}` | top-k cached StreamingMuon sigma values, sorted descending |
| `streaming_sigma_values/{module}` | same cached StreamingMuon sigma values, kept explicit for sigma-transform analysis |
| `sharpness/selected_subspace` | top Hessian eigenvalue from one global Lanczos HVP over selected matrix weights |
| `gradient_hessian_projection/selected_subspace` | global signed projection `dot(e_H, G)` |
| `momentum_after_nesterov_hessian_projection/selected_subspace` | global signed projection `dot(e_H, M')` |
| `gradient_hessian_alignment/selected_subspace` | global cosine alignment between Hessian eigenvector and gradient |
| `momentum_after_nesterov_hessian_alignment/selected_subspace` | global cosine alignment between Hessian eigenvector and `M'` |
| `gradient_projection_on_last_hessian_space_coefficients/selected_subspace` | coefficients of current gradient projected onto the most recent Hessian eigenspace |
| `gradient_projection_on_last_hessian_space_norm/selected_subspace` | norm of the current gradient projection onto the most recent Hessian eigenspace |
| `gradient_projection_on_last_hessian_space_top1_abs_fraction/selected_subspace` | fraction of projected-gradient norm explained by the top Hessian direction |
| `gradient_projection_on_last_hessian_space_consecutive_pearson/selected_subspace` | Pearson correlation between consecutive coefficient vectors `E^T g_t` and `E^T g_{t-1}` in the same last Hessian eigenspace; undefined for `METRICS_HESSIAN_TOP_K=1` |
| `gradient_projection_on_last_hessian_top1_lag1_pearson/selected_subspace` | rolling lag-1 Pearson correlation of scalar top-Hessian-direction gradient projection coefficients |
| `hessian_eigenvector_block_norm/{module}` | norm of the global Hessian eigenvector restricted to this module |
| `gradient_hessian_alignment/{module}` | per-module cosine between Hessian block and gradient block |
| `momentum_after_nesterov_hessian_alignment/{module}` | per-module cosine between Hessian block and `M'` block |
| `gradient_hessian_projection/{module}` | per-module signed projection of gradient onto the unnormalized Hessian block |
| `momentum_after_nesterov_hessian_projection/{module}` | per-module signed projection of `M'` onto the unnormalized Hessian block |
| `alignment_between_covariance_hessian_at_k_th_component/{module}` | alignment between Hessian block and cached StreamingMuon component `u_k v_k^T` |
| `hessian_muon_component_alignment_matrix/{module}` | Hessian-vs-cached-StreamingMuon component alignment matrix |
| `hessian_muon_component_signed_projection_matrix/{module}` | signed Hessian-block projection onto cached StreamingMuon components |

In DDP, per-module optimizer metrics are gathered from the rank that owns each StreamingMuon parameter chunk, and `train/loss` is all-reduced across ranks. Hessian probes currently use post-update weights and a representative rank0 microbatch from the same optimizer step; they are rank0-local loss HVPs, not full distributed-batch Hessians. The Hessian routine runs one Lanczos probe over all normal transformer matrix weights selected by `METRICS_MODULE_REGEX`, including cross-module Hessian blocks within that selected parameter subspace. Per-module Muon-component alignment uses cached StreamingMuon basis/sigma; if that cache is unavailable, component alignment is reported as unavailable. Set `NANOCHAT_FORCE_MATH_SDPA=1` when Hessian probes are enabled.

Use `METRICS_HESSIAN_TOP_K=4` for subspace projection-correlation studies. With `top_k=1`, component-wise correlation is undefined, so use `gradient_projection_on_last_hessian_top1_lag1_pearson/selected_subspace` for largest-eigen-direction temporal correlation instead. That metric treats the signed coefficient sequence `c_t = <g_t, e_1>` as a scalar time series and reports a rolling lag-1 Pearson correlation over `METRICS_PROJECTION_CORRELATION_WINDOW` steps.

Analyze existing dense-metrics runs with:

```bash
python analysis/analyze_metrics_dynamics.py
```

This writes `results/metrics_dynamics_analysis/` with per-step CSVs, a summary table, and plots for sharpness, Hessian alignment, projection correlation, optimizer-state norms, and validation BPB deltas.

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

- The current handoff runner does not launch native Muon/LITE or LITE-like streaming variants. Historical results may still appear in `results/` for context, but new sweeps should compare only `streaming_identity` and `top_aware_muon` unless a separate validation script is added deliberately.
- Dense metrics can be much slower than no-metrics training. Report measured `eval_time_seconds` from `result.json`.
- New d8 dynamics runs should use the `0.4B` recipe unless the question explicitly needs longer training.
