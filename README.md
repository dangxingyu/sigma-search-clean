# Standalone Sigma-Search Optimizer Infra

This repository is a standalone handoff bundle for Muon-family optimizer experiments on nanochat-style LLM pretraining. It vendors the required `nanochat/` source tree, optimizer implementations, sweep runners, candidate spectral transforms, metric logging, and analysis scripts. It should run without the outer `sigma-search` repository.

The current research focus is **Top-Aware Muon**: damp the largest current singular direction(s) of the Muon update while leaving the remaining directions Muon-like. The clean recipe currently fixes `top_k=1` and sweeps `alpha`, batch size, and LR.

## Repository Layout

```text
.
├── nanochat/                         # vendored nanochat source, pyproject, uv.lock
├── candidates/                       # f(sigma) candidate transforms
├── run_eval.py                       # StreamingMuon candidate training/eval entrypoint
├── streaming_muon_torch.py           # StreamingMuon + DDP optimizer implementation
├── metric_logging.py                 # opt-in diagnostic logger
├── run_top_aware_muon_sweep.py       # batch x alpha x lr sweep runner
├── run_v31_256k_512k_alpha_sweeps_after_v29_on_node.sh # overnight alpha closure
├── run_v30_dynamics_metrics_after_v29_on_node.sh # queued dynamics logging launcher
├── run_native_muon_v9.py             # native Muon baseline runner
├── run_lite_v9.py                    # native LITE baseline runner
├── muon_lite.py                      # native-style Muon-LITE optimizer
├── scripts/setup_env.sh              # create nanochat uv env
├── scripts/download_climbmix.sh      # download ClimbMix parquet shards
├── scripts/smoke_run.sh              # tiny sanity run
├── analysis/                         # result parsers / plotting scripts
├── results/                          # curated machine-readable result summaries
├── figures/                          # copied summary figures and tables
└── docs/                             # research notes copied at handoff time
```

The root files are the canonical runnable copies. `core/`, `runners/`, and `analysis/` are categorized mirrors for review and pass-by convenience.

## Result Organization

Use `results/` for curated, machine-readable experiment logs that should travel with the clean repo. Each experiment family gets its own directory with a compact `summary.json` and optional notes. Large raw `result.json` files from dense metric runs should stay in runtime output directories such as `search_evals/` unless they are intentionally copied into `results/raw/`; `results/raw/` is ignored by default.

Current tracked summaries:

```text
results/index.json
results/topaware_128k_near_identity_lr001/summary.json
results/topaware_128k_alpha125_lrsweep/summary.json
results/dynamics_128k_200step/summary.json
```

For future sweeps, prefer writing raw outputs under a run-specific directory, then add a compact summary entry with: method, batch size, LR, seed, token/step budget, score, source path, git commit if available, and status.

## Setup

Install dependencies from the vendored nanochat project:

```bash
cd /path/to/refactored-repo
bash scripts/setup_env.sh
source nanochat/.venv/bin/activate
export PYTHONPATH="$PWD:$PWD/nanochat"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
```

The setup script runs:

```bash
cd nanochat && uv sync --extra gpu --group dev
```

The data is not vendored. Download ClimbMix shards with:

```bash
bash scripts/download_climbmix.sh 170 8
```

This downloads `170` train shards plus the required validation shard to:

```text
$NANOCHAT_BASE_DIR/base_data_climbmix
```

For serious 1B-token sweeps, use enough shards that the training loader does not loop too aggressively. On the original cluster, the data path was `/home/xd7812.princeton/.cache/nanochat/base_data_climbmix`.

Sanity check:

```bash
bash scripts/smoke_run.sh
```

## Optimizer Formulation

All StreamingMuon-family methods operate on the same Muon-like matrix object.

For a weight matrix `W` and gradient `G_t`, the optimizer maintains:

```text
M_t = beta * M_{t-1} + (1 - beta) * G_t
M'_t = (1 - beta) * G_t + beta * M_t
```

`M'_t` is the Nesterov-corrected momentum object passed to Muon/StreamingMuon. Vanilla Muon computes:

```text
M'_t = U Sigma V^T
update = U V^T
```

StreamingMuon approximates this through a warm-started basis:

```text
V_t          = streaming_power_iteration(M'_t, V_{t-1})
R_t          = M'_t V_t
sigma_i      = ||R_t[:, i]||_2
U_t[:, i]    = R_t[:, i] / sigma_i
update       = U_t diag(f(sigma)) V_t^T
```

The candidate transform `f(sigma)` is the experimental hook. `f(sigma)=1` recovers the StreamingMuon identity baseline.

Current stable DDP settings for rigorous sweeps:

```text
pure_qr=True
num_iters=2
fallback_ortho_tol=0.01
```

These are slower than SCQR fast paths but avoid the earlier numerical-drift confound.

## Implemented Methods

| Method | Runner | Definition | Use |
|---|---|---|---|
| `streaming_identity` | `run_eval.py + candidates/identity.py` | `f_i(sigma)=1` | Muon-like StreamingMuon baseline |
| `streaming_lite` | `run_eval.py + candidates/lite_chi2_rs01.py` | top 10% directions scale `1`, tail directions scale `2` | LITE-like same-driver baseline |
| `top_aware_muon` | `run_eval.py + candidates/top_aware_muon.py` | largest `top_k` singular directions scale `alpha`, all others scale `1` | current algorithm of interest |
| `native_muon` | `run_native_muon_v9.py` | nanochat/native Polar Express Muon | exact Muon control |
| `native_lite` | `run_lite_v9.py` | native-style LITE projection/amplification | exact LITE control for `world_size=1` only |

Important: `native_lite` is not currently a valid DDP baseline. `run_lite_v9.py`
initializes a process group, but `MuonLiteAdamW` does not implement distributed
gradient synchronization and the model is not wrapped in DDP. Use `native_lite`
only with `python run_lite_v9.py` or `torchrun --nproc_per_node=1` until a real
distributed LITE optimizer is implemented.

Top-Aware Muon:

```python
def f(sigma, top_k=1, alpha=0.5):
    scale = ones_like(sigma)
    idx = topk(sigma, k=top_k)
    scale[idx] = alpha
    return scale
```

Current clean recipe: keep `top_k=1`; sweep `alpha`. `run_top_aware_muon_sweep.py` rejects `top_k != 1` unless `--allow-top-k-sweep` is explicitly passed.

## Metric Logging

`run_eval.py` and `run_native_muon_v9.py` have opt-in metric logging. It does not change normal runs unless `--metrics-every N` is set. On logging steps, the optimizer caches the exact post-allreduce tensors used by the step:

```text
G_t, M_t, M'_t, sigma_t
```

These are written into `result.json` under `metric_logs`.

Example:

```bash
torchrun --standalone --nproc_per_node=8 run_eval.py \
  --candidate-file candidates/top_aware_muon.py \
  --candidate-param top_k=1 --candidate-param alpha=0.5 \
  --total-batch-size 1048576 \
  --device-batch-size 16 \
  --max-seq-len 1024 \
  --max-steps 1024 \
  --matrix-lr 0.02 \
  --num-iters 2 --pure-qr --fallback-ortho-tol 0.01 \
  --metrics-every 1 \
  --metrics-top-k 4 \
  --metrics-hessian-every 100 \
  --metrics-hessian-max-modules 8 \
  --metrics-split-momentum
```

### Logged Scalars

| Metric | Meaning | Pseudocode |
|---|---|---|
| `train/loss` | EMA-smoothed train cross-entropy | `loss_ema` |
| `val/loss` / `val_bpb` | validation loss/BPB from nanochat eval | `evaluate_bpb(model, val_loader)` |
| `weight_norm/<module>` | RMS norm of matrix weight | `sqrt(mean(W^2))` |
| `grad_norm/<module>` | RMS norm of post-allreduce gradient | `sqrt(mean(G_t^2))` |
| `momentum_norm/<module>` | RMS norm of Muon momentum | `sqrt(mean(M_t^2))` |
| `momentum_after_nesterov_norm/<module>` | RMS norm of optimizer input | `sqrt(mean((M'_t)^2))` |
| `momentum_spectral_norm/<module>` | top singular value of momentum | `svdvals(M_t)[0]` |
| `momentum_after_nesterov_spectral_norm/<module>` | top singular value of Muon input | `svdvals(M'_t)[0]` |
| `sharpness/<module>` | approximate max Hessian eigenvalue | `lambda_1` from HVP power iteration |

### Logged Vectors

| Metric | Meaning | Pseudocode |
|---|---|---|
| `muon_singular_values/<module>` | top singular values of `M'_t` | `svd(M'_t).S[:k]` |
| `streaming_sigma_values/<module>` | current StreamingMuon column-norm sigmas | `sigma_t[:k]` |
| `variance_of_u_i/<module>` | split-half momentum singular-direction alignment | `abs(dot(side_vec_A[i], side_vec_B[i]))` |
| `gradient_hessian_alignment/<module>` | gradient vs Hessian eigenvector | `cos(vec(G_t), e_i(H))` |
| `momentum_hessian_alignment/<module>` | momentum vs Hessian eigenvector | `cos(vec(M_t), e_i(H))` |
| `momentum_after_nesterov_hessian_alignment/<module>` | Muon input vs Hessian eigenvector | `cos(vec(M'_t), e_i(H))` |
| `alignment_between_covariance_hessian_at_k_th_component/<module>` | Hessian eigenvector vs Muon rank component | `cos(e_i(H), u_i v_i^T)` |

### Split-Momentum Alignment

When `--metrics-split-momentum` is enabled and gradient accumulation has at least two microsteps, the logger keeps two independent diagnostic momentum buffers:

```text
G_A = mean gradient from first half of accumulation
G_B = mean gradient from second half of accumulation
M_A = beta * M_A + (1 - beta) * G_A
M_B = beta * M_B + (1 - beta) * G_B
```

Then for each matrix:

```text
M_A = U_A Sigma_A V_A^T
M_B = U_B Sigma_B V_B^T
c_i = |<v_A_i, v_B_i>| for tall matrices
c_i = |<u_A_i, u_B_i>| for wide matrices
```

The default `--metrics-alignment-side lite` uses the same side that LITE/Top-Aware projects on: right singular vectors for tall matrices, left singular vectors for wide matrices.

### Hessian Probe Caveat

Hessian metrics are best-effort and expensive. For optimization-dynamics studies, use per-step cheap metrics and a much lower Hessian frequency:

```bash
--metrics-every 1 \
--metrics-hessian-every 100 \
--metrics-hessian-top-k 1 \
--metrics-hessian-iters 2 \
--metrics-hessian-max-modules 8
```

Some attention kernels do not support double backward. Failures are recorded in `metric_logs[*].hessian.error` and do not abort training. Treat Hessian logging as a targeted diagnostic, not a default 1B-token sweep setting.

When Hessian probes are enabled, force math SDPA so the attention path supports double backward:

```bash
export NANOCHAT_FORCE_MATH_SDPA=1
```

The default flash/mem-efficient SDPA kernels can train normally but may not support the higher-order autograd path used by HVP-based sharpness probes.

## Sweep Protocol

Primary normal-scale recipe:

```text
model depth: d8
sequence length: 1024
tokens: 1,073,741,824
GPUs: 8
device batch: 16 where possible
schedule: warmup_steps = round(0.05 * max_steps)
warmdown_ratio: 0.65
final_lr_frac: 0.05
StreamingMuon: pure_qr=True, num_iters=2, fallback_ortho_tol=0.01
seed: start with 42; confirm important claims with 42/43/44
```

LR policy:

1. Start with `lr in {0.005, 0.01, 0.02, 0.04}`.
2. If the best point is interior, treat the LR as closed.
3. If the best point is on a boundary and the neighboring trend is strong, extend outward.
4. Do not extend upward after a clearly bad high-LR point.
5. Compare optimizers only after each method has a reasonably tuned LR under the same batch and schedule.

For Top-Aware Muon, sweep:

```text
batch_size x alpha x lr
top_k = 1 fixed
alpha candidates = {0.5, 0.75, 0.875} initially
identity baseline = alpha 1.0, implemented as streaming_identity
```

Alpha policy:

1. Treat `alpha` as batch-size dependent, with the working hypothesis that larger batches need stronger top-direction damping, so the best `alpha` should move monotonically downward as batch size increases.
2. Use a narrow first pass: `128K: {0.75, 0.875, 1.0}`, `256K/512K: {0.5, 0.75, 0.875, 1.0}`, `1M/2M: {0.25, 0.5, 0.75, 1.0}`.
3. If the best alpha is on the lower boundary and the curve is still improving, extend lower by one point; if the best alpha is near `1.0`, stop damping for that batch.
4. Do not spend compute on 8M unless the 1M/2M trend is unclear or the larger cluster has slack.
5. For each `(batch, alpha)` pair, use the same LR boundary rule below rather than blindly running every LR.

Run a full sweep:

```bash
python run_top_aware_muon_sweep.py \
  --out-root search_evals/topaware_full \
  --log-root logs/topaware_full \
  --methods "streaming_identity streaming_lite native_muon top_aware_muon" \
  --batches "131072 262144 1048576 8388608" \
  --alphas "0.5 0.75 0.875" \
  --lrs "0.005 0.01 0.02 0.04" \
  --seeds "42" \
  --tokens 1073741824 \
  --depth 8 \
  --nproc-per-node 8 \
  --metrics-every 512 \
  --metrics-top-k 4 \
  --metrics-split-momentum
```

On SLURM allocation:

```bash
srun --jobid=<ALLOC> --overlap --ntasks=1 \
  bash run_top_aware_muon_sweep_on_node.sh \
  --batches "131072 262144 1048576 8388608" \
  --alphas "0.5 0.75 0.875" \
  --lrs "0.005 0.01 0.02 0.04"
```

Run the queued dynamics-logging study:

```bash
bash run_v30_dynamics_metrics_after_v29_on_node.sh
```

This waits for the v29 alpha sweep to finish, then runs `native_muon`, `streaming_identity`, and `top_aware_muon(top_k=1, alpha=0.75)` at `128K`, `256K`, `512K`, `1M`, and `2M` with `--metrics-every 1` and `--metrics-hessian-every 100`. The dashboard is written to `<out-root>/v30_dashboard/`.

Run the overnight 256K/512K alpha-closure queue:

```bash
bash run_v31_256k_512k_alpha_sweeps_after_v29_on_node.sh
```

This waits for the active v29 run, fills/verifies the `256K` alpha grid, then runs `512K` with `streaming_identity`, `native_muon`, and `top_aware_muon`. The default 512K grid is `alpha in {0.25, 0.5, 0.75, 0.875}` and `lr in {0.005, 0.01, 0.02, 0.04}`.

The runner writes:

```text
manifest.json
top_aware_sweep_rows.csv
<case>/result.json
```

## Current Batch Grids

Already used or planned in this codebase:

| Purpose | Batch sizes |
|---|---|
| small-batch same-driver v26 | `32K`, `64K` |
| Top-Aware alpha closure | `128K`, `256K` |
| medium/large Top-Aware | `1M`, `8M` |
| historical native Muon/LITE crossover | `128K`, `256K`, `512K`, `1M`, `4M`, `16M` |
| prepared d12 scale-up | `1M`, `8M`, `3B` token budget |

## Current Results Snapshot

Lower BPB is better. Same-driver StreamingMuon results are the most relevant for Top-Aware comparisons.

### Small Batch Same-Driver v26

| batch | method | best BPB | best LR | note |
|---:|---|---:|---:|---|
| `32K` | streaming identity | `0.935064` | `0.005` | Muon-like |
| `32K` | streaming LITE-like | `0.933131` | `0.005` | better than identity by `0.001933` |
| `32K` | Top-Aware `alpha=0.5` | `0.936364` | `0.005` | worse than identity |
| `64K` | streaming identity | `0.922145` | `0.01` | Muon-like |
| `64K` | streaming LITE-like | `0.922291` | `0.01` | essentially tied, slightly worse |
| `64K` | Top-Aware `alpha=0.5` | `0.923802` | `0.01` | worse than identity |

Interpretation: the very small-batch evidence does not show Top-Aware helping. LITE-vs-Muon is mixed/tiny at 32K/64K and should not be overclaimed from one seed.

### 128K Top-Aware Closure

| batch | method | best BPB | best LR |
|---:|---|---:|---:|
| `128K` | streaming identity | `0.916968` | `0.01` |
| `128K` | Top-Aware `alpha=0.5` | `0.917040` | `0.01` |
| `128K` | Top-Aware `alpha=0.75` | `0.917183` | `0.01` |

Interpretation: 128K wants little or no top-direction damping. Identity (`alpha=1`) is still best among these StreamingMuon variants.

### 256K In-Progress Grid

Completed rows at handoff time:

| batch | method | best BPB | best LR |
|---:|---|---:|---:|
| `256K` | streaming identity | `0.915119` | `0.01` |
| `256K` | streaming LITE-like | `0.914020` | `0.01` |
| `256K` | native Muon | `0.913946` | `0.01` |

Pending or incomplete at the last snapshot: native LITE and Top-Aware alpha grid `{0.5, 0.75, 0.875}`.

### Medium / Large Historical Top-Aware

| batch | streaming identity | Top-Aware `alpha=0.5` | delta top-aware minus identity |
|---:|---:|---:|---:|
| `1M` | `0.934979 @ 0.01` | `0.933161 @ 0.02` | `-0.001818` |
| `8M` | `1.098501 @ 0.02` | `1.083120 @ 0.02` | `-0.015381` |

Interpretation: Top-Aware helps at medium/large batch in same-driver StreamingMuon, especially 8M. The 8M native-vs-streaming identity gap was not fully explained, so report same-driver deltas unless native controls are refreshed under the same recipe.

### Native Muon vs Native LITE Historical Trend

Clean single-GPU/native-style evidence says LITE advantage grows with batch:

```text
128K: tie / tiny LITE edge
256K: tie / tiny Muon edge
512K: LITE wins by ~0.0024 BPB
1M:   LITE wins by ~0.0048 BPB
4M:   LITE wins by ~0.0124 BPB
16M:  LITE wins by ~0.0253 BPB
```

This supports the high-batch LITE story, but Top-Aware should be judged against same-driver StreamingMuon identity/LITE-like baselines unless exact native DDP controls are included.

## Recommended Next Sweep for Sadhika

Run the clean same-driver grid first:

```text
batches = {128K, 256K, 1M, 8M}
methods = streaming_identity, streaming_lite, top_aware_muon
top_k = 1
alpha = {0.5, 0.75, 0.875}
lr = {0.005, 0.01, 0.02, 0.04}, with boundary extension
seed = 42 first, then 43/44 for regimes where gaps are small
```

Add native controls selectively:

```text
native_muon: all key batch/LR points if compute allows
native_lite: run separately with world_size=1; do not include in DDP sweeps yet
```

Metric logging recommendation:

```text
--metrics-every eval_every
--metrics-top-k 4 or 8
--metrics-split-momentum
--metrics-alignment-side lite
```

Only run Hessian probes on a small subset:

```text
--metrics-hessian-every 100
--metrics-hessian-max-modules 4
--metrics-hessian-iters 4
```

## Known Caveats

- `native_lite` is not DDP-safe yet. The current exact LITE optimizer lacks the distributed reduce/gather path used by `DistMuonAdamW`; DDP `native_lite` rows should be excluded from reports.
- `native_lite` is slower than native Muon because it computes a Gram eigensolve (`torch.linalg.eigh`) for the LITE subspace every matrix group and step.
- Historical native single-GPU BPB and newer StreamingMuon DDP BPB can differ in absolute scale. Prefer same-driver deltas.
- Hessian metrics are not guaranteed on every PyTorch attention kernel.
- Current Top-Aware conclusions are mostly one-seed except where historical native sweeps used seeds `42/43/44`.
- Generated outputs (`search_evals/`, `logs/`, `wandb/`, checkpoints, data shards) should remain untracked.

## Minimal Command Reference

Single StreamingMuon identity run:

```bash
torchrun --standalone --nproc_per_node=8 run_eval.py \
  --candidate-file candidates/identity.py \
  --total-batch-size 262144 \
  --device-batch-size 16 \
  --max-seq-len 1024 \
  --max-steps 4096 \
  --matrix-lr 0.01 \
  --num-iters 2 --pure-qr --fallback-ortho-tol 0.01 \
  --output-file results/identity_256k_lr001.json
```

Single Top-Aware run:

```bash
torchrun --standalone --nproc_per_node=8 run_eval.py \
  --candidate-file candidates/top_aware_muon.py \
  --candidate-param top_k=1 \
  --candidate-param alpha=0.75 \
  --total-batch-size 262144 \
  --device-batch-size 16 \
  --max-seq-len 1024 \
  --max-steps 4096 \
  --matrix-lr 0.01 \
  --num-iters 2 --pure-qr --fallback-ortho-tol 0.01 \
  --metrics-every 512 --metrics-split-momentum \
  --output-file results/topaware_a075_256k_lr001.json
```

Native Muon:

```bash
torchrun --standalone --nproc_per_node=8 run_native_muon_v9.py \
  --total-batch-size 262144 \
  --device-batch-size 16 \
  --max-steps 4096 \
  --matrix-lr 0.01 \
  --output-file results/native_muon_256k_lr001.json
```

Native LITE:

```bash
python run_lite_v9.py \
  --total-batch-size 262144 \
  --device-batch-size 16 \
  --max-steps 4096 \
  --matrix-lr 0.01 \
  --lite-chi 2 \
  --lite-rs 0.1 \
  --lite-chi-warmup 0.5 \
  --lite-chi-schedule warmup_hold \
  --output-file results/native_lite_256k_lr001.json
```
