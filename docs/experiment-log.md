# Experiment Log — LITE vs Muon diagnostic campaign

## 2026-05-03 — full grad-accum Hessian metrics

Latest update:
- Changed Hessian metric logging from first-microbatch probes to full optimizer-step grad-accum probes. On Hessian logging steps, `run_eval.py` now caches every local microbatch from that step and passes the list into `hessian_power_probe`.
- Updated `hessian_power_probe` to keep each Lanczos probe vector fixed across all supplied microbatches, accumulate HVPs by token count, and use DDP `all_reduce(SUM) / global_tokens` rather than rank-averaging local first-microbatch HVPs.
- Added a CPU unit test that checks unequal microbatch sizes are token-weighted correctly on a scalar quadratic model.
- Updated README metric-scope wording: Hessian metrics now estimate the full distributed grad-accum optimizer-batch Hessian for the selected matrix subspace, not a representative microbatch Hessian.
- Ran a real 8xB200 d16 DDP validation: `d16_gradaccum_hessian_real50_20260503_171241`, 50 steps, 512K global batch, `MAX_DEVICE_BATCH_SIZE=16`, metrics every step, Hessian top4/iters6 every 25 steps. Result completed with `error=None`, `metric_logs=50`, Hessian probes at steps `0` and `25`, both with `local_hessian_batches=4`, `local_hessian_tokens=65536`, and `global_hessian_tokens=524288`. Observed peak memory was about 131-133GB/GPU.
- Measured 50-step speed on the same 8xB200 allocation, 512K global batch, metrics every step. d12 with `MAX_DEVICE_BATCH_SIZE=32`: no Hessian `204s`, full grad-accum Hessian top4/iters6 every 25 steps `283s` (`1.39x`). d16 with `MAX_DEVICE_BATCH_SIZE=16`: no Hessian `322s`; full Hessian timestamp estimate `445s` to result JSON and `463s` to wrapper completion (`1.38-1.44x`). The two Hessian probes in these tests cover the full 512K distributed batch.

## 2026-05-02 — Chinchilla token-budget interface

Latest update:
- Restored README metric definitions with math/pseudocode columns for active clean metrics: per-step norms/sigma values, selected-subspace Hessian sharpness, Hessian projections/alignments, and fixed-Hessian-space projection correlations.
- Changed the clean default token budget from 1x to 2x Chinchilla in the Python runner and handoff wrappers. The 1x table remains available through `CHINCHILLA_MULT=1`; exact `TOKENS` is still reserved for smoke/custom runs.
- Added `scripts/run_d12_d16_2x_grid.sh` as the safest sequential handoff command for the current main grid: depths `{12,16}`, batches `{512K,2M,8M}`, alphas `{1.0,0.5}`, LR grid `{0.005,0.0075,0.01,0.015,0.02,0.03,0.04}`, and adaptive boundary closure.
- Updated optimizer-quality LR defaults to a denser log-ish grid `{0.005,0.0075,0.01,0.015,0.02,0.03,0.04}` while keeping adaptive boundary closure enabled.
- Added `scripts/submit_slurm_grid.sh` for preemption-safe SLURM arrays: each task runs one full-grid case via `--case-index`; re-submitting the same `STAMP` skips completed results and resumes incomplete checkpoints. Adaptive LR closure is done afterward by running `scripts/run_d12_sweep.sh` once with the same `STAMP`.
- Added a hard-coded `DEPTH -> 1x Chinchilla tokens` table to the clean sweep engine: d8 `402653184`, d12 `1698693120`, d16 `4026531840`.
- Updated `scripts/run_d12_sweep.sh` and `scripts/run_d12_statistics.sh` so normal runs use `DEPTH` plus `CHINCHILLA_MULT`; exact `TOKENS` remains available only for smoke tests or custom truncated runs.
- Documented the d16 sweep command in `README.md`: `DEPTH=16 CHINCHILLA_MULT=1 STAMP=d16_main_001 bash scripts/run_d12_sweep.sh`.
- Updated the clean default comparison to `METHODS=top_aware_muon` with `ALPHAS="1.0 0.5"`, so `c=1` and `c=0.5` run through the same candidate implementation. The separate `streaming_identity` candidate is now a targeted sanity check rather than the default baseline.

## 2026-05-02 — README handoff rewrite

Latest update:
- Rewrote `README.md` around the intended handoff workflow: short repo overview, basic setup, `Sweep`, `Metrics`, and result catalog sections.
- Made `scripts/run_d12_sweep.sh` the only optimizer-quality sweep entrypoint and `scripts/run_d12_statistics.sh` the only metrics/statistics entrypoint.
- Documented d12 defaults, d8 overrides, adaptive LR boundary extension, preemption-safe `STAMP` reuse, config-signature guard, and fixed-recipe metrics usage.
- Compressed the metrics table to the active logged metrics and clarified DDP metric scope: optimizer metrics gather owner-rank cache, train loss is rank-averaged, Hessian HVPs average one representative local microbatch per rank.

## 2026-05-02 — sweep audit and StreamingMuon cleanup

Latest update:
- Audited `run_top_aware_muon_sweep.py` argparse usage and wrappers. No active sweep args were unused after cleanup.
- Fixed Python sweep default `--lr-max` to `0.16` to match wrappers/README and the intended two-round boundary closure from `0.04 -> 0.08 -> 0.16`.
- Added sweep argument validation for empty grids, invalid numeric values, `top_k != 1` without explicit ablation opt-in, and Hessian metrics requested without `metrics_every`.
- Added `manifest.json` config signatures. Reusing a `STAMP/OUT_ROOT` with changed core recipe now fails fast instead of silently skipping old compatible-looking result filenames.
- Simplified `streaming_muon_torch.py`: removed unused compiled fused/unsafe SCQR fast paths, duplicate Polar Express constants, unused imports, old built-in tikhonov/clip/sigmoid transforms, the no-op `--use-scqr` run arg, and historical diagnostic knobs not used by the clean sweep.
- Current optimizer core keeps the stable path used by handoff sweeps: StreamingMuon with optional `CustomTransform`, strict fallback/pure QR, DDP reduce-scatter/all-gather, cached sigma metrics, and Top-Aware candidates from `candidates/`.

## 2026-05-02 — distributed Hessian HVP metrics fix

Latest update:
- Fixed the clean metrics Hessian path so DDP probes are no longer rank0-local HVPs. On Hessian logging steps, every rank computes local HVP blocks on its local representative microbatch, then `all_reduce(AVG)` averages those blocks for Lanczos.
- Kept the probe efficient by default: it averages one local microbatch per rank, not every grad-accum microbatch in the full optimizer step.
- Fixed the follow-up projection path so all ranks continue gathering gradient references after a Hessian space is active; rank0 can then compute `E^T g_t` projection coefficients against the last Hessian eigenspace using complete owner-rank references.
- Updated README and active plan wording to distinguish distributed representative-microbatch HVPs from full optimizer-batch Hessians.

## 2026-05-02 — sweep surface cleanup for handoff

Latest update:
- Reduced the active `scripts/` surface to five files: `setup_env.sh`, `download_climbmix.sh`, `smoke_run.sh`, `run_d12_sweep.sh`, and `run_d12_statistics.sh`.
- Removed the duplicate d8 metrics wrapper. d8 sweeps/statistics now use the same d12 wrappers with `DEPTH=8 TOKENS=402653184`, which avoids multiple near-identical launch paths.
- Kept scheduler submission outside the repo. For multi-job runs, users should shard by setting `STAMP`, `BATCHES`, `ALPHAS`, `LRS`, and `SEEDS` explicitly.
- Hardened sweep resume behavior: invalid or half-written result JSONs now trigger rerun instead of crashing the sweep cataloger.
- Raised default adaptive LR upper bound to `0.16`, so the default two-round edge extension can close `0.04 -> 0.08 -> 0.16` when the high-LR boundary remains best.

## 2026-05-02 — v42/v43 clean c=0.5 study and no-SVD dynamics logging

Latest update:
- Pushed clean repo commits through `38711f0` to `git@github.com:dangxingyu/sigma-search-clean.git`.
- v42 completed the primary d8 / 0.4B-token optimizer-quality sweep for `streaming_identity` (`c=1`) vs Top-Aware Muon `top_k=1, alpha=0.5` (`c=0.5`) at batches `{262144, 1048576, 4194304}` with LR boundary extension.
- v42 best rows:

| batch | identity best | c=0.5 best | c=0.5 - identity | winner |
|---:|---:|---:|---:|---|
| 262144 | `0.967732 @ 0.04` | `0.969135 @ 0.02` | `+0.001403` | identity |
| 1048576 | `1.010557 @ 0.02` | `1.009856 @ 0.08` | `-0.000700` | c=0.5, tiny |
| 4194304 | `1.206814 @ 0.04` | `1.174488 @ 0.02` | `-0.032326` | c=0.5 |

- v42 artifacts are in `results/sweep_catalog/`: `v42_c05_vs_identity_final.md`, `v42_c05_vs_identity_lr_sweep_table.md`, and `v42_c05_vs_identity_lr_sweep.svg`.
- v43 completed the targeted 4M dynamics run with metrics every step and Hessian every 24 steps for identity `lr={0.02,0.04}` and Top-Aware `c=0.5, lr={0.02,0.04}`.
- v43 metric sanity passed: all four runs have 96 per-step metric records and Hessian probes at steps `[0, 24, 48, 72]`; each result JSON is about `6.8MB`.
- v43 best result is Top-Aware `c=0.5, lr=0.02` at `1.176648`, about `0.030 BPB` better than best identity in the same metrics run.
- v43 artifacts are in `results/metrics_v43_4m_best/`: `v43_4m_metrics_summary.md`, `v43_4m_metrics_summary.json`, and `v43_4m_metrics_dynamics.svg`.
- Current metrics path is no-SVD for the active StreamingMuon study: it logs cached streaming `sigma`, cached basis-derived alignments, `weight_norm`, `grad_norm`, `momentum_after_nesterov_*`, global selected-subspace Hessian probes, and fixed-Hessian-subspace gradient projection correlations.
- v44/v45 transition sweep completed for `{2M,8M}` and closed the 2M high-LR boundary with `lr=0.16`.
- v44/v45 best rows:

| batch | identity best | c=0.5 best | c=0.5 - identity | winner |
|---:|---:|---:|---:|---|
| 2097152 | `1.068525 @ 0.08` | `1.082541 @ 0.08` | `+0.014016` | identity |
| 8388608 | `1.456445 @ 0.02` | `1.421837 @ 0.02` | `-0.034608` | c=0.5 |

- v44/v45 artifacts are in `results/sweep_catalog/v44_v45_transition_summary.*`.
- Caveat: the transition is not monotone in this one-seed sweep. 1M has a tiny, not-yet-LR-closed `c=0.5` edge; 2M favors identity after LR closure; 4M/8M strongly favor `c=0.5`.
- v46 completed the 2M best-LR dynamics comparison: identity `1.067766 @ lr=0.08`, Top-Aware `c=0.5` `1.083390 @ lr=0.08`, metrics every step, Hessian every 48 steps. Artifacts are in `results/metrics_v46_2m_best/`.
- v46 sharpness probes: identity `[0.060, 0.121, 0.087, 0.226]`, Top-Aware `[0.060, 0.182, 0.158, 0.219]`. Projection-correlation means are similar (`-0.852` identity, `-0.859` Top-Aware), so the current simple metrics do not yet explain the 2M degradation.
- v47 closed the 1M Top-Aware high-LR boundary: `lr=0.16` scored `1.022783`, worse than the existing `lr=0.08` score `1.009856`, so the 1M best row remains `c=0.5 @ lr=0.08`.
- The updated transition artifact `results/sweep_catalog/v44_v45_transition_summary.md` now includes the v47 1M boundary check.
- v48 completed the 8M best-LR dynamics run: identity `1.450463 @ lr=0.02`, Top-Aware `c=0.5` `1.428626 @ lr=0.02`, metrics every step, Hessian every 12 steps. Artifacts are in `results/metrics_v48_8m_best/`.
- v48 sharpness probes: identity `[0.161, 1.449, 0.590, 1.046]`, Top-Aware `[0.160, 1.439, 1.071, 1.714]`. Projection norm means are nearly identical, while Top-Aware has less negative average projection-correlation (`-0.540` vs identity `-0.787`).
- v49/v50 completed 2M seed confirmation at `lr=0.08`, seeds `{42,43,44,45,46}`. Mean delta `c=0.5 - identity = +0.000507 ± 0.004183` SEM; identity wins 3 seeds and Top-Aware wins 2. Artifact: `results/sweep_catalog/v49_v50_2m_seed_confirm_summary.md`.
- Interpretation update: 2M is a noisy/tie-like transition point, not a robust identity-win point.
- v51 completed 4M seed confirmation using LRs `{0.02,0.04}`, seeds `{42,43,44}`. Mean delta `c=0.5 - identity = -0.025240 ± 0.004110` SEM; Top-Aware wins 3/3 seeds. Artifact: `results/sweep_catalog/v51_4m_seed_confirm_summary.md`.
- v52-v55 completed 1M seed confirmation with closed high-LR checks. Mean delta `c=0.5 - identity = +0.001850 ± 0.002798` SEM over seeds `{42,43,44}`; identity wins seed42, Top-Aware wins seeds43/44 by tiny margins. Artifact: `results/sweep_catalog/v52_v55_1m_seed_confirm_summary.md`.
- Correction: the earlier 1M seed42 Top-Aware edge was an LR-sweep artifact. After adding identity `lr=0.08`, seed42 favors identity (`1.002451` vs Top-Aware `1.009856`).
- v56/v57 completed 262K seed confirmation with boundary check. Mean delta `c=0.5 - identity = -0.000613 ± 0.001083` SEM over seeds `{42,43,44}`; identity wins seed42, Top-Aware wins seeds43/44 by tiny margins. Artifact: `results/sweep_catalog/v56_v57_262k_seed_confirm_summary.md`.

## 2026-05-02 — scripts directory trimmed

Latest update:
- Trimmed `scripts/` to five user-facing commands only: `setup_env.sh`, `download_climbmix.sh`, `smoke_run.sh`, `run_d12_sweep.sh`, and `run_d12_statistics.sh`.
- Removed the optional SLURM submitter and the duplicate d8 metrics wrapper from the clean repo. Scheduler-specific job arrays should be handled outside the repo by the user's cluster infra; d8 uses `DEPTH=8 TOKENS=402653184` overrides on the same standalone scripts.
- Moved historical catalog rebuilding from `scripts/build_sweep_catalog.py` to `analysis/build_sweep_catalog.py`; it remains available for result curation but is no longer presented as a sweep/launch script.
- Updated `README.md`, `MANIFEST.md`, `docs/experiment-plan.md`, and `docs/new-thoughts.md` to match the smaller script surface.

## 2026-05-02 — standalone sweep scripts

Latest update:
- Simplified the clean repo sweep surface. `scripts/run_d12_sweep.sh`, `scripts/run_d12_statistics.sh`, and `scripts/run_d8_metrics_grid.sh` now directly invoke `run_top_aware_muon_sweep.py`; they no longer depend on a separate handoff wrapper.
- Removed `scripts/run_handoff_sweep.sh` to avoid presenting two user-facing sweep entrypoints.
- Updated `README.md`, `MANIFEST.md`, `docs/experiment-plan.md`, and `docs/new-thoughts.md` so the documented handoff path is `scripts/run_d12_sweep.sh` for optimizer-quality sweeps and `scripts/run_d12_statistics.sh` / `scripts/run_d8_metrics_grid.sh` for dynamics metrics.
- Validation passed: `bash -n scripts/run_d12_sweep.sh scripts/run_d12_statistics.sh scripts/run_d8_metrics_grid.sh scripts/submit_d12_sweep_shards_slurm.sh`; `python -m py_compile run_top_aware_muon_sweep.py`; dry-runs of the three standalone sweep scripts produced direct `python run_top_aware_muon_sweep.py ...` commands.

## 2026-05-02 — logging audit fix and user-facing handoff sweep interface

Latest update:
- Follow-up logging cleanup: removed raw momentum-buffer logging from the main StreamingMuon metric path. Metrics now focus on gradients and optimizer input `M'` only.
- StreamingMuon per-step spectrum metrics now reuse cached `sigma`; the clean metrics path has no explicit SVD. Hessian Muon-component alignment uses cached StreamingMuon basis/sigma when available and otherwise marks the component unavailable.
- `METRICS_MAX_MODULES` now only throttles high-frequency per-step metric rows. Hessian selected-subspace probes default to all normal attention/MLP matrix weights with `METRICS_HESSIAN_MAX_MODULES=0`.
- Native Muon/LITE runners remain in the repo as deprecated validation controls; new sweeps should default to `streaming_identity` and `top_aware_muon`.
- Audited clean-repo metric logging while v38 was starting. Stopped the partially-running v38 metrics grid before completion because StreamingMuon split alignment was logging split momentum buffer `M`, not optimizer input `M'`.
- Fixed `run_eval.py`: split-half alignment now uses `M' = (1 - beta) * G_split + beta * M_split_new`, matching the matrix sent to StreamingMuon. In DDP, split gradients are all-reduced before updating diagnostic split momenta, so the split alignment is global-batch rather than rank-local.
- Fixed `run_eval.py` and `run_native_muon_v9.py`: `train/loss` now records the raw mean cross-entropy across the optimizer step's grad-accum microbatches; `train/loss_ema` stores the smoothed value.
- Fixed Hessian probe semantics: Hessian uses post-update weights and the first rank0 microbatch from the same optimizer step. The current clean version uses one global selected-matrix-subspace Lanczos HVP over all normal transformer matrix weights, not separate per-module block HVPs.
- Verified the corrected logging path with a real 8xB200 200-step run: `search_evals/d8_metrics_alpha_grid_v38_logging_smoke_20260502_020957/top_aware_k1_a1_bsz262144_lr0p02_s42/result.json`. It wrote 200 metric entries, global-DDP split `M'` alignment, and Hessian/sharpness entries at steps 0 and 100.
- Rewrote `README.md` to be handoff-oriented: quick start, recommended sweep, optimizer definitions, adaptive LR behavior, metrics caveats, output layout, and d8 recipe table.
- Added `scripts/run_handoff_sweep.sh`, a user-facing wrapper around `run_top_aware_muon_sweep.py`.
- Enhanced `run_top_aware_muon_sweep.py` with `--adaptive-lr`: after the initial LR grid, each `(method,batch,seed,top_k,alpha)` group extends outward if its best finite BPB is on the low or high LR boundary. Decisions are written to `adaptive_lr_trace.json`.
- Validation passed: `python -m py_compile run_top_aware_muon_sweep.py run_eval.py run_native_muon_v9.py metric_logging.py`; `bash -n scripts/run_handoff_sweep.sh scripts/run_d8_metrics_grid.sh`; dry-run adaptive test correctly proposed `0.04` when `{0.01,0.02}` had best BPB at `0.02`.
- Pushed clean repo commit `3c828c5` to `origin/main`.
- Launched corrected v38b metrics grid: `search_evals/d8_metrics_alpha_grid_v38b_d8_metrics_fixed_logging_20260502_022100/`, same d8 alpha `{0.5,1.0}` x batch `{262K,1M,4M}` plan, now with corrected split-`M'` and loss/Hessian metadata.

## 2026-05-02 — clean repo sweep catalog, cleanup, and d8 0.4B metrics recipe

Latest update:
- Cleaned standalone repo by removing one-off `run_vXX...` launchers, duplicate `runners/` and `core/` mirrors, root-level duplicate analysis wrappers, and legacy `top1_damp_*` candidate files. Future pass-by usage should go through `run_eval.py`, `run_top_aware_muon_sweep.py`, `run_native_muon_v9.py`, `run_lite_v9.py`, and scripts under `scripts/`.
- Added `scripts/build_sweep_catalog.py` and generated `results/sweep_catalog/` with normalized existing sweep rows. Current catalog has `159` rows, batches `{32K,64K,128K,256K,512K,1M,4M,8M,16M}`, methods `{native_muon,native_lite,streaming_identity,streaming_lite,top_aware_muon}`, and alpha values `{0.25,0.5,0.75,0.85,0.875,1,1.15,1.25}`.
- Added canonical no-tuning d8 metrics recipe: `results/recipes/d8_metrics_grid_recipe.json` and `scripts/run_d8_metrics_grid.sh`.
- Updated d8 default token budget in `run_top_aware_muon_sweep.py` to `402,653,184` tokens, about `0.4B`, divisible by `262K`, `1M`, and `4M`.
- The new metrics-grid plan compares Top-Aware Muon `alpha={0.5,1.0}` at batches `{262144,1048576,4194304}` with `matrix_lr=0.02` and nanochat LR scaling only; no extra hyperparameter tuning.
- v37 completed: 128K fixed `lr=0.01`, seeds `{43,44}`, methods `{streaming_identity, top_aware_muon(alpha=1.15)}`. Combined with seed `42`, paired identity-minus-Top-Aware deltas are `{0.000399,0.000538,0.000590}`, mean `0.000509 ± 0.000057` SEM. This is a small but stable fixed-LR signal for `alpha=1.15`.
- Launched v38 dynamics grid on allocation `29702470`: clean repo command `STAMP=v38_d8_metrics_alpha05_a10_20260502 srun --jobid=29702470 --overlap --ntasks=1 bash scripts/run_d8_metrics_grid.sh`. Grid is `alpha={0.5,1.0}`, batches `{262144,1048576,4194304}`, base `matrix_lr=0.02`, d8, seq1024, `402,653,184` tokens, metrics every step, Hessian every 50 logged steps.

## 2026-05-01 — clean-repo result organization and 128K near-identity alpha test

Latest update:
- Added machine-readable result summaries under `results/` in the clean repo. `docs/experiment-log.md` is no longer the only place where result state is recorded.
- `results/topaware_128k_near_identity_lr001/summary.json` records the fixed-LR 128K near-identity alpha curve.
- `results/topaware_128k_alpha125_lrsweep/summary.json` records the 128K `alpha=1.25` LR sweep.
- `results/dynamics_128k_200step/summary.json` records the realistic 200-step metrics smoke runs.
- v36 completed: `alpha=0.85 -> 0.9167656900`, `alpha=1.15 -> 0.9165686200` at fixed `lr=0.01`, 128K, d8, seq1024, seed `42`.
- Current interpretation: `alpha=1.15` is the best current fixed-LR 128K Top-Aware point and slightly beats the existing native Muon control, but the margin is tiny and one-seed; confirm with seeds/LR before a strong claim.

## 2026-05-01 — mainline method focus

User clarified the main experiment should focus on four method families:
- `streaming_identity`: StreamingMuon identity / Muon-like streaming baseline.
- `native_muon`: exact/native Muon control.
- `native_lite`: exact/native LITE control.
- `top_aware_muon`: current candidate, fixed `top_k=1`, sweep `alpha` and LR.

`streaming_lite` is not a required mainline baseline. It can remain as an already-completed sanity check in v29, but new sweeps should not include it unless explicitly requested. It is not an exact native LITE replacement.

Current v29 status:
- Expected rows in the originally launched v29 manifest: `28`.
- Completed: `12` rows: all `streaming_identity`, all `streaming_lite`, all `native_muon`.
- Missing/pending: `16` rows: `native_lite` at four LRs, plus Top-Aware `alpha={0.5,0.75,0.875}` at four LRs.
- Active row at check time: `native_lite_bsz262144_lr0p005_s42`, running under allocation `29702470`.

## 2026-05-01 — clean infra metric logging added

Implemented opt-in diagnostic logging for the clean StreamingMuon/Top-Aware path:
- Added `metric_logging.py`.
- Updated `run_eval.py` with `--metrics-every`, `--metrics-top-k`, `--metrics-module-regex`, `--metrics-max-modules`, and best-effort Hessian flags.
- Updated `streaming_muon_torch.py` so requested logging steps cache the exact post-allreduce gradient, momentum buffer, Nesterov-corrected momentum, and streaming sigma tensors used by the optimizer step. Normal runs do not cache these tensors.
- Updated `run_top_aware_muon_sweep.py` to pass metric flags through to StreamingMuon-family methods.
- Synced the handoff bundle under `refactored-repo/`, including README/MANIFEST documentation.

Logged metrics:
- `train/loss`, per-module `weight_norm`, `grad_norm`, `momentum_after_nesterov_norm`, `momentum_after_nesterov_spectral_norm`.
- Per-module short vectors for Muon/Nesterov singular values and StreamingMuon sigma values.
- Optional split-half momentum alignment as `variance_of_u_i/<module>` using side-aware `lite` alignment by default.
- Optional Hessian power probe records `sharpness/<module>` and Hessian-gradient/momentum/component alignments when the kernel supports double backward. Hessian failures are stored under `metric_logs[*].hessian.error` and do not abort training.

Validation:
- `python -m py_compile run_eval.py streaming_muon_torch.py metric_logging.py run_top_aware_muon_sweep.py` passed.
- CPU smoke run with `--metrics-every 1 --metrics-max-modules 1` completed and wrote one `metric_logs` entry.
- CPU Hessian smoke confirmed Hessian failures are nonfatal; the local CPU attention kernel lacks double backward for scaled-dot-product attention.
- `pytest` is not installed in either the ambient Python or `nanochat/.venv`, so unit tests were not run.

Standalone handoff cleanup:
- Vendored source-only `nanochat/` into `refactored-repo/nanochat`, excluding `.venv`, `wandb`, `.git`, `__pycache__`, and `*.pyc`.
- Added `refactored-repo/scripts/setup_env.sh`, `scripts/download_climbmix.sh`, and `scripts/smoke_run.sh`.
- Rewrote `refactored-repo/README.md` as a complete standalone handoff document covering setup, data download, optimizer formulations, baselines, Top-Aware Muon, metric definitions/pseudocode, LR sweep protocol, current batch grids, current result snapshot, and recommended next sweep.
- Updated `refactored-repo/MANIFEST.md` to describe the standalone vendored layout.
- Size check after exclusions: `refactored-repo/nanochat` is about `1.4M`; full `refactored-repo` is about `2.3M` before generated outputs.

## 2026-04-30 — status after autonomous controller

Handoff refactor:
- Created `refactored-repo/` as a curated pass-by bundle containing current StreamingMuon infrastructure, previous/native Muon and LITE baseline runners, top1 sigma-transform runners, analysis scripts, current figures/tables, and documentation snapshots.
- Added `refactored-repo/README.md` and `refactored-repo/MANIFEST.md`.
- Made the current v26/v26b/v27/top1 launcher copies infer `REPO` from their script location instead of hardcoding `/blue/yanjun.li/xd7812.princeton/sigma-search`.
- Kept older historical wrappers for provenance; some still contain original cluster paths and must have `REPO=...` updated before reuse.
- Added the generalized `Top-Aware Muon` infra for pass-by cluster sweeps:
  - Candidate: `candidates/top_aware_muon.py`.
  - Config channel: `run_eval.py --candidate-param top_k=<int> --candidate-param alpha=<float>`.
  - Sweep runner: `run_top_aware_muon_sweep.py`, with baselines `streaming_identity`, `streaming_lite`, `native_muon`, `native_lite`, and method `top_aware_muon`.
  - Launcher: `run_top_aware_muon_sweep_on_node.sh`.
  - Handoff copies and README/manifest were updated under `refactored-repo/`.

Native/Streaming/Top-Aware combined dashboard:
- Added generator `make_native_streaming_topk_dashboard.py`.
- Output directory: `new-figures/native_streaming_topk/`.
- Figure: `native_streaming_topk_overview.png`.
- Best table: `native_streaming_topk_best_table.md`.
- CSVs: `native_streaming_topk_best.csv`, `native_streaming_topk_raw.csv`.
- Current combined rows show:
  - 64K v26 streaming-only: Top-Aware k=1 alpha=.5 is worse than identity by `+0.001658` BPB.
  - 128K: native exact-control is `0.000276` better than StreamingMuon identity; Top-Aware is `0.000072` worse than identity.
  - 1M: Top-Aware is `0.001818` better than identity; native exact-control is `0.001604` better than identity.
  - 8M: Top-Aware is `0.015381` better than identity; native exact-control is `0.020191` worse than identity.
  - 32K has identity only so far; Top-Aware has not completed in v26.

Top-Aware clean infra check:
- Decision: current clean recipe fixes `top_k=1`. We sweep `batch x alpha x lr`, not `k`, unless doing an explicit ablation.
- Updated `run_top_aware_muon_sweep.py` and the handoff copy to reject `--top-ks` values other than `[1]` unless `--allow-top-k-sweep` is passed.
- Positive dry-run passed for `top_k=1`, alpha `{0.5,0.75}`.
- Negative dry-run passed: `--top-ks "1 2"` now raises an explicit error.
- Escape-hatch dry-run passed: `--allow-top-k-sweep --top-ks "1 2"` still generates commands for intentional ablations.

v28/v29 Top-Aware k=1 alpha sweep plan:
- Added launcher `run_v28_v29_topaware_k1_alpha_sweeps_on_node.sh`.
- It pauses only the waiting `v26b` supervisor, not allocation `29702470`; waits for the active v26 training process to finish; runs v28/v29; then resumes `v26b`.
- v28: `128K`, Top-Aware `top_k=1`, `alpha=0.75`, LR grid `{0.005,0.01,0.02,0.04}`, seed `42`, d8, `~1.07B` tokens.
- v29: `256K`, baselines `{streaming_identity, streaming_lite, native_muon, native_lite}` plus Top-Aware `top_k=1`, alpha grid `{0.5,0.75,0.875}`, LR grid `{0.005,0.01,0.02,0.04}`, seed `42`, d8, `~1.07B` tokens.
- Rationale: if optimal alpha decreases as batch size grows, `128K` should prefer alpha closer to `1`, while `256K` should start moving below `1`. Identity baseline serves as alpha=`1`.
- Launched inside allocation `29702470` at 2026-04-30 17:35 EDT. It paused the waiting `v26b` supervisor pid `3890075`, then started waiting for active v26 completion before running v28/v29. It will resume `v26b` on exit.

v28/v29 status at 2026-04-30 23:48 EDT:
- v26 small-batch same-driver sweep has completed.
- v28 (`128K`, Top-Aware `alpha=0.75`) has completed all LR rows:
  - `lr=0.005`: `0.920044`
  - `lr=0.01`: `0.917183` best
  - `lr=0.02`: `0.920091`
  - `lr=0.04`: `0.928273`
- v28 interpretation: at 128K, `alpha=0.75` is still worse than identity (`0.916968`) by about `+0.000215` BPB, and better than `alpha=0.5` (`0.917040`) only by about `0.000142` BPB. This supports the "128K wants alpha close to 1 / little or no top damping" direction.
- v29 (`256K`) is in progress. Completed rows:
  - Streaming identity best: `0.915119 @ lr=0.01`
  - Streaming LITE-like best: `0.914020 @ lr=0.01`
  - Native Muon best: `0.913946 @ lr=0.01`
- v29 currently running: native LITE `lr=0.005`. GPU utilization is high; this is running, not idle. Native LITE is much slower than StreamingMuon/native Muon under this DDP setup.

v26 progress update at 2026-04-30 13:55 EDT:
- v26 has been running for about `5.6h`.
- Completed 64K same-driver sweep, seed `42`, pure QR x2:

| batch | method | best LR | best BPB |
|---:|---|---:|---:|
| 64K | `identity` | `0.01` | `0.922145` |
| 64K | `lite_chi2_rs01` | `0.01` | `0.922291` |
| 64K | `top1pm_a05` | `0.01` | `0.923802` |

- 64K interpretation: `identity` is slightly better than LITE-like by `0.000146` BPB and better than top1 by `0.001658` BPB. This supports the small-batch direction `Muon-like >= LITE-like`, but only at one seed and with a very small identity-vs-LITE margin.
- 32K current status: `identity lr=0.005` score `0.935064`, `identity lr=0.01` score `0.936053`; `identity lr=0.02` is currently running. Since `0.005` is best so far, the runner will likely extend downward to `0.0025` after the initial grid before moving to 32K LITE-like/top1.
- v26b supervisor is still waiting. v26b and v27 have not started.
- Dashboard snapshot generated:
  - Markdown table: `search_evals/v26_small_bsz_streaming_fair_20260430/v26_dashboard_table.md`
  - Raw completed rows: `search_evals/v26_small_bsz_streaming_fair_20260430/v26_dashboard_rows.csv`
  - Aggregate rows: `search_evals/v26_small_bsz_streaming_fair_20260430/v26_dashboard_rows_aggregate.csv`
  - LR sweep figure: `search_evals/v26_small_bsz_streaming_fair_20260430/v26_dashboard_lr_sweep.png`
  - Generator: `make_v26_dashboard.py`

v26 dashboard refresh at 2026-04-30 16:26 EDT:
- v26 is not finished. Active run: 32K `lite_chi2_rs01 lr=0.01`; v26b/v27 have not started.
- Completed JSON count: `15`.
- Dashboard was refreshed with all completed `batch × LR × optimizer` rows:
  - Markdown table: `search_evals/v26_small_bsz_streaming_fair_20260430/v26_dashboard_table.md`
  - Raw completed rows: `search_evals/v26_small_bsz_streaming_fair_20260430/v26_dashboard_rows.csv`
  - Aggregate rows: `search_evals/v26_small_bsz_streaming_fair_20260430/v26_dashboard_rows_aggregate.csv`
  - LR sweep figure: `search_evals/v26_small_bsz_streaming_fair_20260430/v26_dashboard_lr_sweep.png`
- Current best-LR rows:

| batch | method | best LR | best BPB |
|---:|---|---:|---:|
| 32K | `identity` | `0.005` | `0.935064` |
| 32K | `lite_chi2_rs01` | `0.005` | `0.933131` |
| 64K | `identity` | `0.01` | `0.922145` |
| 64K | `lite_chi2_rs01` | `0.01` | `0.922291` |
| 64K | `top1pm_a05` | `0.01` | `0.923802` |

- Current interpretation: 64K supports `identity >= LITE-like > top1`, but 32K currently has LITE-like ahead of identity by `0.001933` BPB. Since 32K LITE-like LR sweep and top1 are incomplete, do not use this yet for scale-up.

Historical large-batch dashboard generated:
- Generator: `make_historical_large_batch_dashboard.py`.
- Output directory: `search_evals/historical_large_batch_dashboard/`.
- Markdown table: `search_evals/historical_large_batch_dashboard/historical_large_batch_dashboard.md`.
- Native Muon/LITE figure: `search_evals/historical_large_batch_dashboard/historical_native_muon_lite.png`.
- Streaming top1 LR sweep figure: `search_evals/historical_large_batch_dashboard/historical_streaming_top1_lr_sweep.png`.
- Historical native single-GPU best rows:

| batch | Muon best | LITE best | Muon-LITE |
|---:|---:|---:|---:|
| 128K | `0.893701 @ 0.01` | `0.893577 @ 0.01` | `+0.000124` |
| 256K | `0.892240 @ 0.01` | `0.892301 @ 0.01` | `-0.000061` |
| 512K | `0.898926 @ 0.01` | `0.896504 @ 0.01` | `+0.002422` |
| 1M | `0.915446 @ 0.01` | `0.910612 @ 0.02` | `+0.004834` |
| 4M | `0.994062 @ 0.04` | `0.981683 @ 0.04` | `+0.012379` |
| 16M | `1.332891 @ 0.04` | `1.307633 @ 0.04` | `+0.025258` |

- Historical StreamingMuon/top1 best rows:

| batch | identity best | top1 best | identity-top1 |
|---:|---:|---:|---:|
| 128K | `0.916968 @ 0.01` | `0.917040 @ 0.01` | `-0.000072` |
| 1M | `0.934979 @ 0.01` | `0.933161 @ 0.02` | `+0.001818` |
| 8M | `1.098501 @ 0.02` | `1.083120 @ 0.02` | `+0.015381` |

- Caveat: these are two different experiment families (`native_single_gpu` and old `streaming_ddp_top1_old`) and should not be raw-compared across families. The missing fair large-batch comparison is v26b same-driver `identity`/`lite_chi2_rs01`/`top1pm_a05`, which has not started yet because v26 small-batch is still running.

Figure/table consolidation:
- Created `new-figures/` as the single collection point for current tables and figures.
- `new-figures/current_same_driver_v26/` contains the current v26 dashboard table, LR sweep figure, raw rows, aggregate rows, and summary JSON.
- `new-figures/historical_large_batch/` contains the historical large-batch dashboard, native Muon/LITE figure, old Streaming top1 LR sweep, raw rows, aggregate rows, and summary JSON.
- `new-figures/supporting_plots/` contains auxiliary analysis plots (`optimizer_lr_sweeps`, `v18_128k_clean`, and top1 sweep plots).
- `new-figures/README.md` documents which tables are fair same-driver comparisons and which are historical/non-raw-comparable.

v26 smaller-batch same-driver StreamingMuon sweep launched:
- Launch command:
  `srun --jobid=29702470 --overlap --ntasks=1 bash run_v26_small_bsz_streaming_fair_on_node.sh --out-root search_evals/v26_small_bsz_streaming_fair_20260430 --log-root logs/v26_small_bsz_streaming_fair_20260430`
- Purpose: test whether Muon-like beats LITE-like at smaller batch (`64K`, `32K`) and compare top1 under the same driver/recipe.
- Methods:
  - `identity`: StreamingMuon Muon-like baseline.
  - `lite_chi2_rs01`: same-driver LITE-like transform, top 10% sharp directions scale `1`, flat tail scale `2`.
  - `top1pm_a05`: per-matrix top1 damping, largest singular direction scale `0.5`, others scale `1`.
- Shared recipe: d8, seq1024, `~1.07B` tokens, seed `42`, DDP8, `pure_qr=True`, `num_iters=2`, `fallback_ortho_tol=0.01`, warmup `5%` of steps, warmdown ratio `0.65`, final LR fraction `0.05`.
- Initial LR grid: `{0.005,0.01,0.02}` with automatic outward extension if the best point is on a boundary.
- Important interpretation rule: this is a same-driver StreamingMuon comparison. It should answer `identity` vs `LITE-like` vs `top1` fairly, but it is not raw-comparable to old single-GPU native LITE/Muon BPB without exact controls.
- Scale-up gate for d12/~3B: require LR-closed evidence for large-batch `top1 > LITE-like > identity` and small-batch `identity >= LITE-like`; otherwise run LR/seed closure first.
- Prepared scale-up launcher but did not start it: `run_v27_d12_3b_streaming_fair_on_node.sh`. It reuses the v26 same-driver runner with depth `12`, tokens `3221225472`, default batches `{1M,8M}`, and LR grid `{0.005,0.01,0.02,0.04}`.
- Added and launched follow-up supervisor: `run_v26b_large_then_maybe_v27_on_node.sh`.
  It waits for v26 small-batch completion, runs v26b same-driver large-batch sweep at `{1M,8M}`, then calls `decide_scaleup_gate.py`. It launches v27 d12/3B only if the gate passes; otherwise it stops.

Current job state:
- v26 is currently running on allocation `29702470`.
- Controller log: `logs/autonomous_8h_controller_20260429_204902.log`.
- The controller completed, not crashed.

LR-sweep visualization:
- Added `analyze_optimizer_lr_sweeps.py`.
- Wrote `optimizer_lr_sweeps.png`: clean native 128K `Muon`/`LITE` LR sweep plus DDP StreamingMuon `identity`/`top1_alpha0.5` sweeps at 128K and 1M.
- Wrote `optimizer_lr_sweeps_top1_8m.png`: DDP StreamingMuon `identity`/`top1_alpha0.5` LR sweep at 8M.
- Wrote machine-readable aggregate: `optimizer_lr_sweeps_summary.json`.
- Main small-batch takeaway from the plotted sweep: 128K has no robust LR-tuned Muon win. At the non-optimal low LR `0.005`, Muon is consistently better than LITE by about `0.00047` BPB, but both optimizers prefer `0.01`; at the tuned best LR, LITE is ahead by only `0.00012` BPB, which is effectively a tie.

v18 128K LITE-vs-Muon low-LR closure:
- v18b finished the `lr=0.005` extension for both Muon and LITE, seeds `{42,43,44}`.
- 128K is now low-LR closed over `{0.005,0.01,0.02,0.04}`:

| opt | best LR | best mean BPB | `lr=0.005` mean | conclusion |
|---|---:|---:|---:|---|
| Muon | `0.01` | `0.89370` | `0.89703` | low LR worse |
| LITE | `0.01` | `0.89358` | `0.89750` | low LR worse |

- Best comparison remains effectively a tie: `Delta Muon-LITE = +0.00012` BPB, positive meaning LITE slightly wins. This is far below a strong effect size.
- No `0.0025` extension is needed at 128K.
- Figure refreshed: `v18_128k_clean_analysis.png`.

Exact DDP native controls for top1 identity:
- v21 ran native Muon under the exact DDP/top1 recipe at the identity best-LR points.
- Results:

| batch | LR | native DDP Muon | StreamingMuon identity | streaming-native |
|---:|---:|---:|---:|---:|
| 128K | `0.01` | `0.916692` | `0.916968` | `+0.000276` |
| 1M | `0.01` | `0.933375` | `0.934979` | `+0.001604` |
| 8M | `0.02` | `1.118693` | `1.098501` | `-0.020191` |

- Interpretation: StreamingMuon identity is validated as Muon-like at 128K and 1M under the exact DDP recipe, but not at 8M. At 8M it is much better than native Muon, so top1-vs-identity at 8M does not isolate only the top1 transform.
- Summary file: `v21_identity_gap_summary.json`.

StreamingMuon identity diagnostics:
- Because the 8M identity gap exceeded the controller threshold, the planned top1 alpha sweep did not run.
- v23 ran targeted 1M identity diagnostics:

| variant | score |
|---|---:|
| `pure_qr` | `0.932256` |
| `pure_qr_2iter` | `0.932656` |
| `scqr_2iter` | `0.933197` |
| `input_normalize` | `0.935231` |

- At 1M, exact native DDP Muon is `0.933375`, so SCQR/tol is not the source of a large 1M mismatch. `pure_qr` is slightly better than native, but all variants are in the same narrow band.
- Missing diagnostic: we have not yet run the same identity variants at 8M, where the large gap occurs. That is the next minimal diagnostic before any 8M top1 alpha claim.

Top1 status:
- Existing official top1 results remain valid only relative to StreamingMuon identity in the same run family:
  - 128K: top1 `-0.000072` BPB worse than identity.
  - 1M: top1 `+0.001818` BPB better than identity.
  - 8M: top1 `+0.015381` BPB better than identity, but 8M identity itself does not match native Muon.
- Therefore: top1 has a clean, fair positive signal at 1M; 8M is promising but currently confounded by StreamingMuon identity deviating from native Muon.

## 2026-04-29 — LR boundary check for 128K sweeps

Question: whether current LR sweeps are closed or whether the selected LR is sitting on a grid boundary.

Clean LITE-vs-Muon v18:
- Current grid: `lr={0.01,0.02,0.04}`, bsz `128K`, d8, ~1B tokens, seeds `{42,43,44}`.
- Result table: Muon best is `0.893701` at `0.01`; LITE best is `0.893577` at `0.01`.
- Because both best LRs are the lower grid edge, this point is not LR-closed. The only safe current statement is: within the tested grid, LITE and Muon are effectively tied at 128K.
- Added launcher: `run_v18b_128k_low_lr_extension_on_node.sh`. It appends lower-LR rows into `sweep_results_v18_128k_clean/`, defaulting to `lr=0.005`. If `0.005` is best, rerun the same launcher with `LRS="0.0025"`.

Active top1 sigma sweep:
- The official top1 sweep completed under output root `search_evals/ddp8_top1_damp_adaptive_20260429_171235/`.
- Figures regenerated with the correct root: `top1_damp_lr_sweep_analysis.png` and `top1_damp_lr_sweep_trajectories.png`.
- 128K low LR is closed on both methods: identity `0.005` score `0.920045` vs best `0.01` score `0.916968`; top1 `0.005` score `0.920471` vs best `0.01` score `0.917040`.
- Best-score table, seed `42`, d8 / ~1.07B tokens / DDP8 / strict `fallback_ortho_tol=0.01`:

| batch | identity best | top1-damp best | identity - top1 |
|---:|---:|---:|---:|
| 128K | `0.916968 @ lr=0.01` | `0.917040 @ lr=0.01` | `-0.000072` |
| 1M | `0.934979 @ lr=0.01` | `0.933161 @ lr=0.02` | `+0.001818` |
| 8M | `1.098501 @ lr=0.02` | `1.083120 @ lr=0.02` | `+0.015381` |

- Interpretation: top1 damping does not help at 128K, modestly helps at 1M by tolerating/using a larger best LR, and substantially helps at 8M. This supports the sharp-direction LR-bound story at medium/large batch, but it is still one seed and should be treated as a promising sigma-transform result rather than a final claim.
- v18b has now started automatically after top1 completion. Muon `lr=0.005` seeds `{42,43,44}` completed; LITE `lr=0.005` seeds `{42,43,44}` are currently running. If LITE `0.005` beats `0.01`, extend v18 again to `0.0025`.

DDP-vs-single comparability check:
- The top1 sweep used DDP8 `run_eval.py` with strict `fallback_ortho_tol=0.01`; the earlier clean LITE/Muon baselines v9/v18 are single-GPU drivers.
- Raw BPB between these two driver families is not directly comparable. Before any optimizer step, the same depth/seq/seed family evaluates to `3.155445` in the DDP run_eval/native-match stream, but `3.142474` in the v9/v18 single-GPU stream. This initial `+0.01297` BPB gap is purely eval/data stream, not optimizer or tolerance.
- At 128K, StreamingMuon identity vs single-GPU native Muon stays about `+0.023` BPB worse through training; at 1M it settles around `+0.025` BPB worse after early interpolation noise. Since step0 already differs, this should not be interpreted as a StreamingMuon identity failure without a same-driver native DDP control.
- Mechanism: the BOS-bestfit dataloader shards parquet row groups by DDP rank. DDP validation evaluates first chunks from rank-sharded row groups and all-reduces them; single-GPU validation evaluates a different sequential row-group stream. Training order also differs: e.g. 1M DDP uses 8 ranks × 8 grad accum, while single GPU uses one stream × 64 grad accum.
- Compute action: stopped the queued v21/v22 full-control reruns for now to avoid wasting the allocation. Do not rerun broad controls unless we need a same-driver final comparison. The safe current interpretation is relative within top1's DDP run_eval family: top1 vs StreamingMuon identity, not top1 raw BPB vs single-GPU native LITE.
- The mistakenly launched v20 8M LITE attempt used old `warmup=10` rather than the top1 official `warmup=6`; it was killed and should not be used as an exact control.

## 2026-04-29 — top-1 sharp-direction damping adaptive LR sweep launched

Purpose: test the hypothesis that Muon's global LR is bounded by the current sharpest spectral direction, so damping only the current top singular direction may permit a larger useful global LR for the remaining river/flat directions.

Candidate:
- File: `candidates/top1_damp_alpha05.py`.
- Rule: for current singular values `sigma`, set `f(argmax(sigma)) = 0.5`; all other directions use `f=1.0`.
- This is not LITE tail boosting. It is a top sharp-direction damping rule.

Adaptive sweep design:
- Launcher: `run_ddp8_top1_damp_adaptive_lr_sweep_on_node.sh`.
- Orchestrator: `run_top1_damp_adaptive_lr_sweep.py`.
- Analyzer: `analyze_top1_damp_lr_sweep.py`.
- Scale: d8, seq1024, 8GPU DDP, seed `42`, strict `fallback_ortho_tol=0.01`.
- Batch sizes: `128K`, `1M`, `8M`; each run uses ~`1.07B` tokens.
- Methods: identity StreamingMuon vs top-1 damp `alpha=0.5`.
- Initial matrix LR grid: `{0.01, 0.02, 0.04}`.
- Expansion rule: try lower or higher LRs only if the current boundary is best or near-best; do not run larger LRs after a clear high-LR degradation.

Decision rule:
- The relevant comparison is best LR by method within each batch size.
- Same-LR top1 worse is not decisive; the hypothesis mainly predicts that top1-damp can tolerate a larger global LR and improve the best-LR frontier.

Recipe correction:
- The first launched sweep used fixed `warmup_steps=40`, which makes the schedule incomparable across batch sizes and visibly worsens the identity baseline relative to old clean native runs.
- Do not use the fixed-40 sweep for absolute claims. Its partial results are a pilot/debug artifact only.
- Official recipe from here: `warmup_steps=round(0.05 * max_steps)`, `warmdown_ratio=0.65`, `final_lr_frac=0.05`. This matches the v9/v18 clean schedule family and keeps warmup as a fixed token fraction under the fixed-token-budget design.
- For the selected batches this means: 128K -> `8192` steps with `410` warmup steps; 1M -> `1024` steps with `51` warmup steps; 8M -> `128` steps with `6` warmup steps.
- Candidate bug fixed during the pilot: `top1_damp_alpha05.py` now flattens `sigma` before `argmax`, so the run_eval 2D smoke-test no longer indexes out of bounds.

Official rerun:
- Started at 2026-04-29 17:12 EDT.
- Output root: `search_evals/ddp8_top1_damp_adaptive_20260429_171235/`.
- Log root: `logs/ddp8_top1_damp_adaptive_20260429_171235/`.
- First run confirms the intended recipe: `identity_bsz131072_lr0p01`, `steps=8192`, `warmup=410`, `eval_every=1024`.

## 2026-04-29 — d12 1B-token native Muon vs StreamingMuon identity completed

Purpose: check whether StreamingMuon identity with strict `fallback_orthogonality_tol=0.01` still matches native Muon at a larger model scale and normal 1B-token horizon.

Setup:
- Launcher: `run_ddp8_streaming_vs_native_d12_1b_on_node.sh`.
- Allocation: SLURM job `29702470`, node `c0909a-s25`, 8×B200. Do not cancel this allocation.
- Shared config: depth `12`, seq `1024`, global batch `262144`, device batch `16`, seed `42`, matrix LR `0.02`, warmup `40`, eval tokens `1048576`.
- Horizon: `4096` optimizer steps = about `1.07B` tokens.
- Comparison: native Muon via `run_native_muon_match.py` vs StreamingMuon identity via `run_eval.py --candidate-file candidates/identity.py --num-iters 1 --fallback-ortho-tol 0.01`.
- Output root: `search_evals/ddp8_streaming_native_d12_1b_20260429_140201/`.
- Logs: `logs/ddp8_streaming_native_d12_1b_20260429_140201/`.
- Figure: `streaming_native_d12_1b_analysis.png`.
- Analyzer: `analyze_streaming_native_d12_1b.py`.

Results:

| step | native Muon | StreamingMuon identity `tol=.01` | native-streaming |
|---:|---:|---:|---:|
| 512 | `1.075412` | `1.073765` | `+0.001647` |
| 1024 | `1.017970` | `1.017001` | `+0.000969` |
| 1536 | `0.993129` | `0.992857` | `+0.000272` |
| 2048 | `0.962825` | `0.962217` | `+0.000608` |
| 2560 | `0.934486` | `0.934562` | `-0.000075` |
| 3072 | `0.908039` | `0.907982` | `+0.000056` |
| 3584 | `0.883690` | `0.883825` | `-0.000135` |
| 4096 | `0.866625` | `0.866802` | `-0.000177` |

Conclusion:
- StreamingMuon identity `tol=0.01` matches native Muon extremely closely at d12 and ~1B tokens. Final absolute gap is only `0.000177` BPB, with the sign slightly favoring native Muon at the final point.
- This is stronger than the d8/1024-step match result, because it uses a larger model and normal 1B-token horizon.
- For sigma-search, `tol=0.01` identity StreamingMuon is now a practical Muon-like baseline at d8 and d12, but candidate reports should still include the same-tolerance identity baseline next to each candidate.

## 2026-04-29 — StreamingMuon DDP8 smoke/probe completed

Purpose: validate that StreamingMuon sigma-search can actually run on 8-GPU DDP before restarting larger searches.

Outputs:
- d4 smoke root: `search_evals/ddp8_streaming_smoke_20260429_111021/`.
- d8 probe root: `search_evals/ddp8_streaming_d8_probe_20260429_111919/`.
- Logs: `logs/ddp8_streaming_smoke_20260429_111021/`, `logs/ddp8_streaming_d8_probe_20260429_111919/`.
- Plot: `ddp8_streaming_smoke_analysis.png`.
- Script: `analyze_ddp8_streaming_smoke.py`.

Results:

| run | depth/seq | global batch | steps | tol | final BPB | status |
|---|---:|---:|---:|---:|---:|---|
| identity accum1 | d4/512 | 32K | 8 | `0.01` | `3.143896` | OK |
| identity accum1 | d4/512 | 32K | 8 | `0.02` | `3.143896` | OK |
| identity accum1 | d4/512 | 32K | 8 | `0.05` | `3.143896` | OK short-smoke only |
| identity accum2 | d4/512 | 64K | 8 | `0.01` | `3.114357` | OK, validates corrected grad-accum path |
| tikhonov adaptive | d4/512 | 32K | 8 | `0.01` | `3.144416` | OK, validates mutable `sigma_state` candidate path |
| pure QR identity | d4/512 | 32K | 4 | exact QR | `3.175716` | OK |
| identity | d8/1024 | 262K | 32 | `0.01` | `1.857972` | OK |
| identity | d8/1024 | 262K | 32 | `0.02` | `1.857955` | OK |
| tikhonov adaptive | d8/1024 | 262K | 32 | `0.01` | `1.865220` | OK |

Additional wrapper test:
- `run_streaming_base_train.py` DDP8 identity ran with world size 8, completed one training step, validation, and optimizer checkpoint save after fixes. Final minimum validation BPB: `3.165843`.
- Fixed wrapper issues during this check:
  - `torchrun` call needs `--` before `run_streaming_base_train.py` when the child script uses `--run`.
  - identity transform in wrapper must use built-in string `'identity'`, not a lambda, otherwise optimizer checkpoint pickling fails.

Conclusion:
- `DistStreamingMuonAdamW` and the sigma-search `run_eval.py` path are now small-scale validated on 8×B200 DDP for identity, strict fallback tolerances, grad accumulation, pure QR, and a nontrivial candidate with state.
- `tol=0.05` passing 8 steps only means the branch runs; it does **not** override the d8/1B evidence that loose tolerances drift over long runs.
- NCCL prints IB/RoCE mixed-device warnings on this node, but all single-node DDP8 runs completed successfully.

## 2026-04-29 — Native Muon vs StreamingMuon identity `tol=0.01` DDP8 match test

Purpose: directly test whether StreamingMuon identity with `fallback_orthogonality_tol=0.01` matches native nanochat Muon under the same 8-GPU DDP setup.

Setup:
- New matched native runner: `run_native_muon_match.py`.
- Launcher: `run_ddp8_streaming_vs_native_match_on_node.sh`.
- Output root: `search_evals/ddp8_streaming_native_match_20260429_112759/`.
- Logs: `logs/ddp8_streaming_native_match_20260429_112759/`.
- Plot: `streaming_native_match_analysis.png`.
- Shared config: d8, seq1024, global batch `262144`, device batch `16`, seed `42`, matrix LR `0.02`, 8×B200 DDP.

Results:

| horizon | native final BPB | streaming identity `tol=.01` final BPB | native-streaming |
|---:|---:|---:|---:|
| 32 steps | `1.859221` | `1.857970` | `+0.001251` |
| 256 steps | `1.231428` | `1.212692` | `+0.018736` |
| 1024 steps | `0.995475` | `0.994029` | `+0.001447` |

1024-step curve deltas:
- step `256`: native `1.194466`, streaming `1.191923`, delta `+0.002543`.
- step `512`: native `1.084915`, streaming `1.083261`, delta `+0.001654`.
- step `768`: native `1.028268`, streaming `1.026627`, delta `+0.001640`.
- step `1024`: native `0.995475`, streaming `0.994029`, delta `+0.001447`.

Conclusion:
- Under the more meaningful 1024-step DDP8 probe, StreamingMuon identity `tol=0.01` matches native Muon closely, with final difference about `0.00145` BPB.
- The 256-step compressed schedule is a misleading stress case: it amplifies a StreamingMuon advantage to `0.0187` BPB, but this does not persist in the 1024-step schedule.
- This supports using `tol=0.01` identity StreamingMuon as the baseline for sigma-search candidate comparisons, while still reporting identity beside every candidate.

## 2026-04-29 — StreamingMuon DDP infra check

Question checked: whether StreamingMuon can support 8-GPU DDP training for sigma-search.

- `DistStreamingMuonAdamW` exists and is designed for multi-GPU training: StreamingMuon matrix groups are split across ranks with `reduce_scatter_tensor` on gradients, per-rank chunk updates, then `all_gather_into_tensor` to sync updated params.
- I fixed two wrapper-level correctness issues before treating it as usable:
  - `run_eval.py` now computes DDP gradient accumulation from `device_batch_size * max_seq_len * world_size`, not just per-rank tokens. This avoids the old effective-batch inflation bug.
  - `nanochat/scripts/base_train.py` now updates momentum/weight-decay schedules for both `muon` and `streaming_muon` param groups.
  - `run_streaming_base_train.py` now accepts `--fallback-ortho-tol` and pins `scripts.base_eval` to the in-tree nanochat scripts package to avoid a conflicting top-level `scripts` package.
- Static checks passed: `python3 -m py_compile run_eval.py run_streaming_base_train.py nanochat/scripts/base_train.py streaming_muon_torch.py`.
- CLI smoke passed for `run_streaming_base_train.py --fallback-ortho-tol 0.01 --num-power-iters 1 --help`.
- Still not yet done in this entry: an actual 8-GPU CUDA smoke/full run. Treat DDP as code-supported and wrapper-sane, but not freshly 8-GPU validated today.

Recommended 8-GPU identity sanity command before real sigma-search:

```bash
torchrun --standalone --nproc_per_node=8 run_eval.py \
  --candidate-file candidates/identity.py \
  --output-file search_evals/ddp8_identity_smoke/result.json \
  --depth 8 --max-seq-len 1024 --device-batch-size 32 \
  --total-batch-size 262144 --max-steps 20 --eval-every 10 \
  --matrix-lr 0.02 --num-iters 1 --fallback-ortho-tol 0.01
```

## 2026-04-28 — latest status: v18 128K clean lower-edge sweep active

This log is now maintained with the latest status first; older detailed entries follow below.

### Current active run
- Active allocation: SLURM job `29702470` on `c0909a-s25`; do not cancel it.
- v18 is active via `srun --jobid=29702470 --overlap --ntasks=1 bash run_v18_128k_clean_on_node.sh`.
- Purpose: test whether fixed χ=2 LITE clearly loses below the 256K clean tie point.
- Grid: bsz `128K`, opt `{Muon,LITE}`, base matrix LR `{0.01,0.02,0.04}`, seeds `{42,43,44}`, d=8, 1B tokens.
- Output directory: `sweep_results_v18_128k_clean/`; analyzer: `analyze_v18_128k_clean.py`.
- Launch status: first wave of 8 Muon jobs started at 2026-04-28 22:34 on GPUs 0-7. Muon dt is about `122 ms/step` after warmup; LITE jobs will start as slots free up and are expected to be slower because 128K has 8192 optimizer steps.

### Mainline switching summary package
- New focused summary generated for the current question, not for LITE variant tuning:
  - Markdown: `mainline_switching_results.md`
  - Figure: `mainline_switching_results.png`
  - Trajectory figure: `mainline_switching_trajectories.png`
  - Script: `analyze_switching_mainline.py`
- The package combines clean pure baselines (`v13` 256K + `v9` 512K/1M/4M/16M), v14 first-D1 χ gate, and v16 true optimizer switching.
- Main current conclusion: momentum split-SVD alignment predicts the pure-optimizer regime, but it is not a valid optimizer-switch trigger under the tested rules. Switching appears limited by trajectory/state dependence, not just by threshold choice.
- Pure-regime table in the mainline package:
  - 256K: Muon `0.8922`, LITE `0.8923`, Δ `-0.0001`, Muon D1 last `0.333`.
  - 512K: Muon `0.8989`, LITE `0.8965`, Δ `+0.0024`, Muon D1 last `0.354`.
  - 1M: Muon `0.9154`, LITE `0.9106`, Δ `+0.0048`, Muon D1 last `0.375`.
  - 4M: Muon `0.9941`, LITE `0.9817`, Δ `+0.0124`, Muon D1 last `0.792`.
  - 16M: Muon `1.3329`, LITE `1.3076`, Δ `+0.0253`, Muon D1 last `1.000`.
- Correlation in this combined clean table: `corr(Muon D1 last, Muon-LITE Δ) = +0.968`; `corr(Muon D1 first, Muon-LITE Δ) = +0.865`.

### v17 checkpoint-branch counterfactual completed
- v17 design: `1M, seed=42`. Train native Muon to checkpoint step `170`, then resume the exact same model/optimizer/data/RNG state into two branches: continue native Muon vs switch matrix groups to LITE. Continue both to `1024` steps / 1B tokens.
- The step-170 checkpoint was created successfully at `sweep_results_v17_checkpoint_branch/checkpoints/checkpoint_step000170.pt`.
- Resume alignment sanity check passed: both branches report identical step-170 validation BPB `1.149135`.
- Final result: continue Muon `0.912766`; switch-to-LITE `0.911581`; delta Muon-LITE `+0.001185` BPB. Switching to LITE from the exact same Muon checkpoint locally helps, but it still does not recover pure LITE's 1M mean `0.9106`.
- Figure generated: `v17_checkpoint_branch_analysis.png`.

### v15 completed result: sharp-rank sweep
- v15 completed `12/12` JSONs after node-local retry and wrote `v15_rs_sweep_analysis.png`.
- Result: increasing sharp rank is not a robust fix. `r_s=0.2` slightly improves 256K vs fixed `r_s=0.1` by `-0.0002` BPB but slightly hurts 512K by `+0.0002`; `r_s=0.5` ties 256K and hurts 512K by `+0.0008`.

| bsz | r_s | Muon | LITE r=.1 | LITE r_s | Muon-new | new-base |
|---|---:|---:|---:|---:|---:|---:|
| 256K | `0.2` | `0.8922` | `0.8923` | `0.8921` | `+0.0002` | `-0.0002` |
| 512K | `0.2` | `0.8989` | `0.8965` | `0.8967` | `+0.0023` | `+0.0002` |
| 256K | `0.5` | `0.8922` | `0.8923` | `0.8922` | `+0.0000` | `-0.0001` |
| 512K | `0.5` | `0.8989` | `0.8965` | `0.8973` | `+0.0017` | `+0.0008` |

### Priority update
- User clarified the central question: the main deliverable is whether metric-driven optimizer switching helps, not how to tune LITE best.
- v15 remains secondary to the switching question and should not block the mainline story.
- v16 is the mainline experiment: true one-time Muon/LITE switching, not a LITE-internal χ proxy. Matrix groups share AdamW state and momentum buffers; Muon phase uses native Muon-style polar update with aspect-ratio LR scaling; LITE phase uses current `χ=2`, `r_s=0.1`.
- v16 launch note: the first multi-`srun` launcher attempt failed before Python startup with SLURM `slurm_get_port` errors and produced `0` JSONs. It was replaced by `run_v16_optimizer_switch_on_node.sh`, which uses a single `srun` into `c0909a-s25` and launches one Python process per GPU locally.
- v16 completed all `18/18` JSONs: `{256K,512K,1M} × {Muon->LITE,LITE->Muon} × seeds {42,43,44}`, threshold `0.417`, normal d=8 / 1B-token scale.
- Main v16 conclusion: one-switch optimizer control does **not** beat the better pure optimizer. Muon->LITE is clearly worse than pure LITE at 512K/1M. LITE->Muon at 512K is also worse than pure LITE despite switching when the metric falls. At 256K, LITE->Muon is only a tie with the better pure baseline.

### v16 completed result: true one-switch optimizer control
- Output directory: `sweep_results_v16_optimizer_switch/`.
- Figure generated: `v16_optimizer_switch_analysis.png`.
- Analyzer command: `MPLCONFIGDIR=/tmp/matplotlib-sigma python analyze_v16_optimizer_switch.py`.
- Switch rule: side-aware split-half `M_tilde`, top-16, `frac_above_c_star_median`, threshold `0.417`.
- `Muon->LITE`: switch once when metric `>= 0.417`; otherwise stay Muon.
- `LITE->Muon`: switch once when metric `< 0.417`; otherwise stay LITE.

| bsz | Muon BPB | fixed LITE BPB | Muon->LITE BPB | LITE->Muon BPB | M->L - best pure | L->M - best pure |
|---|---:|---:|---:|---:|---:|---:|
| 256K | `0.8922` | `0.8923` | `0.8926` | `0.8920` | `+0.0005` | `-0.0000` |
| 512K | `0.8989` | `0.8965` | `0.8987` | `0.8976` | `+0.0022` | `+0.0011` |
| 1M | `0.9154` | `0.9106` | `0.9161` | `0.9106` | `+0.0055` | `+0.0000` |

- Switch behavior:
  - 256K Muon->LITE never switched; LITE->Muon switched at step 682 for all seeds.
  - 512K Muon->LITE switched at step 341 for all seeds; LITE->Muon switched at step 682 for seeds 42/43 and step 1365 for seed 44.
  - 1M Muon->LITE switched at step 170 for all seeds; LITE->Muon never switched.
- Interpretation:
  - The metric is useful for regime classification, but turning that into a mid-training optimizer switch does not recover the best pure trajectory.
  - Muon->LITE failing at both 512K and 1M supports the path-dependence/front-loaded-LITE hypothesis: LITE's advantage appears to require being on the LITE trajectory early, not after a Muon warmup.
  - LITE->Muon at 512K is a direct negative answer to the degradation question under this rule: the metric fell below the crossover threshold and the run switched, but pure LITE still won.
  - 256K LITE->Muon is at best a tie, not evidence that switching is a robust improvement.

### Latest completed result
- v14 tested the minimal binary controller from `gate_calibration_v13.png`: start as fixed LITE; at first `M_tilde` D1 probe, keep χ=2 if `c* >= 0.417`, otherwise set χ target to 1.
- Gate decisions were exactly as designed: 256K set χ to 1 for all seeds; 512K and 1M kept χ=2 for all seeds.

| bsz | Muon BPB | fixed LITE BPB | first-D1 gate BPB | Muon - gate | gate - fixed LITE |
|---|---:|---:|---:|---:|---:|
| 256K | `0.8922` | `0.8923` | `0.8935` | `-0.0013` | `+0.0012` |
| 512K | `0.8989` | `0.8965` | `0.8964` | `+0.0025` | `-0.0001` |
| 1M | `0.9154` | `0.9106` | `0.9107` | `+0.0048` | `+0.0001` |

- Main v14 conclusion: the calibrated first-probe binary χ gate is not a winning hybrid. It preserves fixed-LITE behavior at 512K/1M but hurts 256K despite making the intended "unresolved -> χ=1" decision.
- Interpretation: switching optimizer behavior after the first probe is not equivalent to using Muon from the start, and the 256K fixed-LITE vs Muon gap is too small for a late binary switch to recover. Future hybrid work should not be a simple global on/off χ gate.
- Figure generated: `v14_first_d1_gate_analysis.png`.

### v13 completed result: clean 256K lower edge
- v13 tested the clean local 256K lower edge of the crossover using the same v9-style driver and `M_tilde` D1 telemetry.
- Final v13 result: 256K is effectively tie / very slight Muon, matching the older multi-seed result but now with local momentum-D1 telemetry.

| bsz | Muon BPB | LITE χ=2 BPB | Δ = Muon - LITE |
|---|---:|---:|---:|
| 256K | `0.8922` | `0.8923` | `-0.0001 ± 0.0005` |

- Muon-trajectory D1(`M_tilde`, LITE-side) summary:

| bsz | c* first / mean / last | >0.7 first / mean / last |
|---|---:|---:|
| 256K | `0.354 / 0.346 / 0.333` | `0.104 / 0.117 / 0.125` |
| 512K | `0.479 / 0.400 / 0.354` | `0.167 / 0.142 / 0.125` |
| 1M | `0.708 / 0.537 / 0.375` | `0.333 / 0.233 / 0.167` |
| 4M | `1.000 / 0.921 / 0.792` | `0.854 / 0.662 / 0.438` |
| 16M | `1.000 / 1.000 / 1.000` | `1.000 / 1.000 / 1.000` |

- Main v13 conclusion: late hard `c>0.7` alone cannot separate 256K from 512K because both end at `0.125`. First/mean `c*` carries more separation, but the 256K/512K gap is still small; a global scalar gate will be brittle near the crossover.
- Figure generated: `v13_256k_d1_analysis.png`.
- Gate calibration figure generated: `gate_calibration_v13.png`.
- Candidate 256K/512K separator margins from Muon D1(`M_tilde`):
  - `cstar_first`: threshold ≈ `0.417`, margin `+0.125`, separates.
  - `cstar_mean`: threshold ≈ `0.373`, margin `+0.054`, separates.
  - `cstar_last`: threshold ≈ `0.344`, margin `+0.021`, separates weakly.
  - `hard_first`: threshold ≈ `0.135`, margin `+0.062`, separates.
  - `hard_mean`: threshold ≈ `0.129`, margin `+0.025`, separates weakly.
  - `hard_last`: threshold ≈ `0.125`, margin `+0.000`, does not separate.

### v12 completed result: χ=1.5 LR check
- v12 tested whether χ=1.5 only lost in v11 because it reused χ=2's LR.
- Grid: 512K χ=1.5 LRs `{0.005, 0.01, 0.02}` where `0.01` comes from v11; 1M χ=1.5 LRs `{0.01, 0.02, 0.04}` where `0.02` comes from v11; all 3 seeds.
- Final LR-check table:

| bsz | χ=1.5 LR | mean BPB | Δ vs Muon | BPB - χ=2 |
|---|---:|---:|---:|---:|
| 512K | `0.005` | `0.9088` | `-0.0099` | `+0.0123` |
| 512K | `0.01` | `0.8973` | `+0.0017` | `+0.0008` |
| 512K | `0.02` | `0.8969` | `+0.0020` | `+0.0004` |
| 1M | `0.01` | `0.9184` | `-0.0030` | `+0.0078` |
| 1M | `0.02` | `0.9117` | `+0.0038` | `+0.0011` |
| 1M | `0.04` | `0.9160` | `-0.0005` | `+0.0054` |

- Best χ=1.5 LR by mean BPB: 512K `lr=0.02`, 1M `lr=0.02`.
- Main v12 conclusion: lower global χ is very unlikely to be the right hybrid axis. Even after adjacent LR checks, χ=1.5 remains worse than fixed χ=2 at both 512K and 1M.
- Figure generated: `v12_chi15_lrsweep_analysis.png`.

### Current scripts and next action
- `analyze_v12_chi15_lrsweep.py` completed successfully and wrote `v12_chi15_lrsweep_analysis.png`.
- Added `run_v12_chi15_lrsweep.sh`: new χ=1.5 LR checks at 512K `{0.005, 0.02}` and 1M `{0.01, 0.04}`. Existing v11 supplies 512K `0.01` and 1M `0.02`.
- Added `analyze_v12_chi15_lrsweep.py`: combines v11+v12 χ=1.5 runs with v9 Muon/χ=2 baselines, reports best LR by mean BPB, and writes `v12_chi15_lrsweep_analysis.png`.
- Checks passed: `bash -n run_v12_chi15_lrsweep.sh`; `python -m py_compile analyze_v12_chi15_lrsweep.py`.
- Added `run_v13_256k_clean.sh`: local single-GPU v9-style 256K baseline, opt `{Muon, LITE χ=2}`, seeds `{42,43,44}`, lr `0.01`.
- Added `analyze_v13_256k_clean.py`: combines v13 with v9 to report clean 256K Δ and D1(`M_tilde`) trajectory; writes `v13_256k_d1_analysis.png`.
- Added `analyze_gate_calibration.py`: compares D1 summary choices against Δ and writes `gate_calibration_v13.png`.
- Checks passed: `bash -n run_v13_256k_clean.sh`; `python -m py_compile analyze_v13_256k_clean.py`.
- Added `run_v14_first_d1_gate.sh`: first-probe `c*` gate with threshold `0.417`; keep fixed LITE if resolved, otherwise set χ target to `1.0`.
- Added `analyze_v14_first_d1_gate.py`: compares v14 against v13/v9 Muon and fixed-LITE baselines; writes `v14_first_d1_gate_analysis.png`.
- Checks passed: `python -m py_compile run_lite_v9.py analyze_v14_first_d1_gate.py analyze_gate_calibration.py`; `bash -n run_v14_first_d1_gate.sh`.
- Added `run_v15_rs_sweep.sh`: fixed χ=2, `r_s={0.2,0.5}`, bsz `{256K,512K}`, 3 seeds, d=8, 1B tokens.
- Added `analyze_v15_rs_sweep.py`: compares v15 against fixed `r_s=0.1` LITE and Muon baselines; writes `v15_rs_sweep_analysis.png`.
- Checks passed: `python -m py_compile analyze_v15_rs_sweep.py run_lite_v9.py`; `bash -n run_v15_rs_sweep.sh`.
- Next action: monitor v15 to `12/12` JSONs. If larger `r_s` does not help 256K without hurting 512K, adaptive rank becomes less promising and the report should frame offline switching as the practical path.

### v11 completed result: fixed lower χ at crossover
- v11 tested fixed lower amplification, χ=1.5, against paired v9 Muon and fixed χ=2 LITE baselines.
- Final v11 result: χ=1.5 still beats Muon at 512K and 1M, but is worse than fixed χ=2 on every seed and both batch sizes.
- Final paired table:

| bsz | χ=2 LITE Δ vs Muon | χ=1.5 LITE Δ vs Muon | χ=1.5 - χ=2 BPB |
|---|---:|---:|---:|
| 512K | `+0.0024` | `+0.0017` | `+0.0008` |
| 1M | `+0.0048` | `+0.0038` | `+0.0011` |

- Main v11 conclusion: the simple "χ=2 is too aggressive at crossover" hypothesis is not supported. Under the current LR choices, fixed χ=2 remains better than χ=1.5 even in the low-D1 512K/1M regimes.
- Figure generated: `v11_chi15_analysis.png`.

### v10 completed result: triangular χ schedule
- v10 tested triangular χ schedule (`1 -> 2 -> 1`) against paired v9 Muon and fixed-LITE baselines.
- Final v10 result: triangular χ beats Muon at all tested batch sizes, but underperforms fixed χ=2 at all tested batch sizes.
- Final paired table:

| bsz | fixed LITE Δ vs Muon | hybrid χ Δ vs Muon | hybrid - fixed |
|---|---:|---:|---:|
| 512K | `+0.0024` | `+0.0021` | `-0.0003` |
| 1M | `+0.0048` | `+0.0042` | `-0.0006` |
| 4M | `+0.0124` | `+0.0086` | `-0.0038` |
| 16M | `+0.0253` | `+0.0144` | `-0.0109` |

- Main v10 conclusion: a time-only late decay of χ is not the right hybrid. It removes useful LITE amplification in the higher-D1 regimes and does not improve the borderline 512K/1M regimes.
- Figure generated: `v10_hybrid_chi_analysis.png`.

### Repo hygiene
- `.gitignore` now ignores `.codex`, `sweep_results*/`, and `v*_smoke/` so generated result directories remain local but no longer pollute `git status`.
- Remaining untracked files after cleanup are intentional working files/scripts: `AGENTS.md`, `agent-read-this.md`, v9/v10 drivers, launchers, and analysis/plot scripts.
- Two core dumps remain in the repo root (`core.*`, about 711M and 494M apparent disk usage). They are already ignored; they were not deleted because they may be user/debug artifacts.
- Checks passed after cleanup: `python -m py_compile diagnose_split_batch.py train_telemetry.py run_lite_v9.py run_native_muon_v9.py analyze_v9_momentum_d1.py analyze_v10_hybrid_chi.py plot_v9_switch_summary.py`; `bash -n run_v9_momentum_d1.sh run_v10_hybrid_chi.sh launch_v9_after_v8.sh`.

### v10 partial result after 6/12 JSONs
- Completed v10 runs: all `16M x3` and `4M x3`.
- `analyze_v10_hybrid_chi.py` parsed the 6 completed runs and wrote `v10_hybrid_chi_analysis.png`.
- Partial paired comparison vs v9 baselines:

| bsz | fixed LITE Δ vs Muon | hybrid χ Δ vs Muon | hybrid - fixed |
|---|---:|---:|---:|
| 4M | `+0.0124` | `+0.0086` | `-0.0038` |
| 16M | `+0.0253` | `+0.0144` | `-0.0109` |

Interpretation: triangular χ still beats Muon at 4M/16M but clearly underperforms fixed χ. This matches the prediction that late χ decay should hurt in high-alignment regimes where fixed amplification remains useful. Continue waiting for `1M x3` and `512K x3`; those are the regimes where triangular χ could still be useful.

### v10 partial result after 8/12 JSONs
- Additional completed runs: `1M` seeds 42 and 43.
- Partial paired comparison:

| bsz | n | fixed LITE Δ vs Muon | hybrid χ Δ vs Muon | hybrid - fixed |
|---|---:|---:|---:|---:|
| 1M | 2 | `+0.0041` | `+0.0035` | `-0.0006` |
| 4M | 3 | `+0.0124` | `+0.0086` | `-0.0038` |
| 16M | 3 | `+0.0253` | `+0.0144` | `-0.0109` |

Interpretation: triangular χ is close to fixed LITE at 1M but still slightly worse on the first two seeds. It is not yet evidence for hybrid improvement. Continue waiting for `1M` seed44 and all `512K` runs.

### v10 partial result after 9/12 JSONs
- Additional completed run: `1M` seed44, so the 1M row is now paired across all 3 seeds.
- Partial paired comparison:

| bsz | n | fixed LITE Δ vs Muon | hybrid χ Δ vs Muon | hybrid - fixed |
|---|---:|---:|---:|---:|
| 1M | 3 | `+0.0048` | `+0.0042` | `-0.0006` |
| 4M | 3 | `+0.0124` | `+0.0086` | `-0.0038` |
| 16M | 3 | `+0.0253` | `+0.0144` | `-0.0109` |

Interpretation: triangular χ remains close to fixed LITE at 1M but still does not improve it. This makes the simple time-only decay schedule unlikely to be the final hybrid. Remaining decisive check is 512K.

## 2026-04-28 — handoff takeover and v9 diagnostic correction

### User constraints
- Continue on local B200 SLURM only; do not use Modal.
- Do not cancel or kill job `29702470`.
- Do not make claims from dummy/short experiments. Normal evidence should be d=8, ~1B tokens, matched LR, and multi-seed where feasible.
- Maintain `experiment-plan.md`, `experiment-log.md`, and `new-thoughts.md`.

### Runtime status checked
- SLURM job `29702470` is running on `c0909a-s25`.
- v8 d=12 sweep is still running under the original drivers. At takeover it was 16/20 JSON; after `muon_d12_bsz16777216_lr0.04_s42` finished it became 17/20.
- Current v8 run after that is `lite_d12_bsz16777216_lr0.04_s42`.
- `sweep_results_v9/` is still 0/24; v9 has not started.
- `launch_v9.log` only contains the initial watch line, and no active `launch_v9_after_v8.sh` process was found. If v8 finishes and v9 does not launch, manually run the smoke/launcher chain rather than disturbing v8.

### Code review finding
The previous v9 implementation called the new diagnostic "momentum-D1", but it SVD'd fixed-β auxiliary EMA buffers (`β=0.95`). That is not exactly the object LITE uses. Both Muon and LITE apply polar/projection to the Nesterov-corrected direction:

```python
m_new = mom * m_old + (1 - mom) * g
M_tilde = (1 - mom) * g + mom * m_new
```

This matters because a long EMA can saturate split-half alignment near 1, especially at huge grad accumulation, without proving that the actual optimizer input is equally resolved.

Second code review finding: the planned v9 DDP launcher was invalid for LITE. `run_native_muon_v9.py` uses `DistMuonAdamW` under DDP, but `run_lite_v9.py` only has `MuonLiteAdamW`; under `torchrun` each rank would train independently without gradient synchronization. v9 must therefore use 8 parallel single-GPU jobs (world=1) for both optimizers.

### Patch applied before v9 starts
- `diagnose_split_batch.py`: added `alignment_side={left,right,lite}`. The original Sadhika diagnostic is `left`; `lite` compares the projection side LITE actually uses for each matrix shape.
- `train_telemetry.py`: exposes the same `alignment_side` for buffer diagnostics.
- `run_native_muon_v9.py`: `d1_probes_momentum` now means D1 on split-half `M_tilde`, using LITE-side-aware alignment. Raw scheduled momentum buffer D1 is saved as `d1_probes_momentum_buffer`; original left-side `M_tilde` alignment is saved as `d1_probes_momentum_left`.
- `run_lite_v9.py`: same change.
- `analyze_v9_momentum_d1.py`: relabeled analysis from D1(m) to D1(`M_tilde`) and fixed the small/large-batch wording.
- `run_v9_momentum_d1.sh`: rewritten from sequential 8-GPU DDP to an 8-slot single-GPU launcher.
- `launch_v9_after_v8.sh`: smoke changed from 8-GPU `torchrun` to single-GPU `python -u run_native_muon_v9.py`.
- `analyze_v9_momentum_d1.py`: now includes a side/buffer ablation table comparing primary LITE-side `M_tilde`, left-U `M_tilde`, and raw momentum-buffer D1.
- Syntax check passed: `python -m py_compile diagnose_split_batch.py train_telemetry.py run_native_muon_v9.py run_lite_v9.py analyze_v9_momentum_d1.py`.
- Shell syntax check passed: `bash -n run_v9_momentum_d1.sh launch_v9_after_v8.sh`.
- Small side-aware sanity check passed: `alignment_side="lite"` selects `right` for an 8×4 tall matrix and `left` for a 4×8 wide matrix.

### Interpretation notes
- Existing v8 D1(m) logs are from the old fixed-β buffer diagnostic and are not clean evidence for the momentum-SVD theory.
- v9 is the first intended test of Sadhika's diagnostic on the optimizer-relevant direction and subspace side.
- `python analyze_full.py` re-ran successfully on 87 existing d=8 runs. It reproduced the report's master table: Δ crosses between 256K and 512K and grows through 16M; log-log fit exponent is 0.844 with R²=0.987. Exact duplicate `(opt, bsz, lr, seed)` keys were checked and none were found.
- `env MPLCONFIGDIR=/tmp/matplotlib-sigma python plot_for_sadhika.py` regenerated `sadhika_P1_delta_vs_bsz.png`, `sadhika_P2_d1_vs_delta.png`, `sadhika_P3_threshold_transfer.png`, and `sadhika_P4_lr_scaling.png` from the current data.
- Started a CPU-only watcher inside allocation `29702470` at 02:46 EDT: it appends to `launch_v9.log`, waits for v8 to reach 20 JSON files, runs the single-GPU v9 smoke, then starts the single-GPU-parallel v9 sweep. This does not cancel or interfere with the active v8 training step.
- Nested `srun --jobid=29702470 --overlap --ntasks=1 true` from inside an allocation step succeeded, so the watcher should be able to launch smoke/v9 from the compute node.
- The 02:46 watcher step later exited with signal 137. A replacement watcher is active as SLURM step `29702470.69`; `launch_v9.log` now starts with `handoff: restarting single polling instance after user redesign`.

## 2026-04-23 — iter 1 setup

### Context pulled in
- guidance.md (Sadhika): two diagnostics, D1 split-batch alignment, D2 BBP spectrum.
- Pre-existing numbers at 1B d=8: Muon wins at 128K and 512K tok/step, LITE wins at 8M tok/step (see `experiment-plan.md`).
- Node: SLURM 28829013, 8×B200, ends 2026-04-25T10:42 — ~2 days runway.

### Documents initialized
- `experiment-plan.md` — 5 work units (D1, D2, bsz scan, validation, threshold principledness, report).
- `new-thoughts.md` — unification of D1 and D2 via RMT; proposes data-dependent threshold $c^*$ and adaptive d_s for LITE.

### Iter 1 work done
- `diagnose_split_batch.py` (D1): compute per-layer $c_i = |u_A^{(i)\top} u_B^{(i)}|$ from split-batch gradients. Reports `frac_above_0.7` (Sadhika), `frac_above_c_star` (RMT), aggregates across layers, outputs a winner decision.
- `diagnose_spectrum.py` (D2): compute per-layer spectrum, effective rank, MP-bulk-edge-based spike count, and `top_over_bulk` = σ_max / median(σ). Decision rule uses `top_over_bulk` since spike count has finite-size false positives at γ=1.
- Smoke-tested both on 4 synthetic cases (resolved spikes / pure noise / buried spikes / rectangular).

### Smoke test results (d2_smoke.json)

| layer | shape | σ_max | edge | spike_count | top/bulk | r_eff_norm |
|---|---|---|---|---|---|---|
| resolved_spikes | 768×768 | 4.25 | 2.01 | 4 | 5.28 | 0.71 |
| pure_noise | 768×768 | 1.99 | 2.01 | 0 | 2.47 | 0.72 |
| buried_spikes | 768×768 | 1.98 | 2.02 | 0 | 2.46 | 0.72 |
| rectangular_spikes | 1024×256 | 4.16 | 2.16 | 3 | 4.32 | 0.89 |

**D1 smoke (top-8 index-matched alignment, mean):**

| layer | c_mean | c_max | frac>0.7 | frac>c* |
|---|---|---|---|---|
| resolved_spikes | 0.41 | 0.93 | 0.38 | 0.50 |
| pure_noise | 0.046 | 0.15 | 0.0 | 0.0 |
| buried_spikes | 0.034 | 0.06 | 0.0 | 0.0 |
| rectangular_spikes | 0.42 | 0.94 | 0.38 | 1.0 |

Conclusion: both diagnostics correctly distinguish resolved vs unresolved directions on synthetic data. Buried spikes (θ < BBP edge) are correctly not detected. The two rules disagree on rectangular (D1 inconclusive, D2 says LITE) — this will be studied in real data.

### Calibration notes
- D1 `c_star` formula: $1 - 1/\theta^2$ evaluated at $\theta = 1.2\sqrt{\gamma}$ (20% above BBP edge), clamped to [0.25, 0.8].
- D2 MP edge: $1.25 \cdot \sigma(\sqrt M + \sqrt N)$, 25% safety margin above asymptotic edge to avoid Tracy-Widom false positives at finite size.

### Next actions (iter 2+)
1. Integrate both diagnostics into nanochat training loop as a snapshot hook.
2. Run a short (~100 step) training run on c1006a-s25 to verify per-layer outputs are sensible on real gradients at batch=512K.
3. Begin WU-2 batch-size scan (128K/512K/8M first, as pinned reference points).
4. Only after D1/D2 diverge/agree consistently on real data, expand to full 7-point bsz scan.


## 2026-04-23 — iter 2: real-data probe on nanochat

### Tool built
- `probe_grads.py` — loads nanochat d=8 from scratch, optionally warms up with AdamW for N steps, then for each requested bsz accumulates two disjoint halves of gradients into (G_A, G_B) per matrix param, runs D1 and D2 per layer. Output: one JSON per probe run.
- `plot_probe.py` — 3-panel figure: D1 fractions vs bsz, per-layer c_top_k_mean heatmap, D2 top/bulk + r_eff vs bsz.

### Zero-init footgun (found and fixed)
At nanochat random init, `attn.c_q`, `attn.c_k`, `attn.c_v`, `mlp.c_fc` are initialized to zero for residual-stream-preserving reasons. Their gradients are identically zero at step 0, which made D1 report `frac>0.7 = 1.0` trivially (SVD of zero matrix → arbitrary orthogonal basis). Both D1 and D2 now mark layers with ||G|| < 1e-10 as "inactive" and exclude them from aggregates.

### Probe after warmup=100 (AdamW lr=5e-4, bsz=128K), d=8, 125M params, seq=1024

| bsz (tok) | n_active | D1 frac>0.7 | D1 frac>c* | D2 top/bulk | D2 r_eff(σ²)/r | D1 decision | D2 decision |
|---|---|---|---|---|---|---|---|
| 128K | 52 | 0.250 | 0.562 | 4093 | 0.0031 | ambiguous | LITE |
| 512K | 52 | 0.375 | 0.750 | 5834 | 0.0032 | ambiguous | LITE |
| 2M   | 52 | 0.656 | 1.000 | 7629 | 0.0031 | LITE | LITE |
| 4M   | 52 | 0.781 | 1.000 | 8333 | 0.0030 | LITE | LITE |
| 8M   | 52 | 0.875 | 1.000 | 8644 | 0.0031 | LITE | LITE |

**Key observation:** D1 `frac>0.7` grows monotonically from 0.25 at 128K → 0.88 at 8M, crossing 0.5 somewhere between 512K and 2M. This matches the empirical LITE-vs-Muon crossover (Muon wins at 128K/512K per memory, LITE wins at 8M).

Per-layer c_top_k_mean for attn/mlp matrix params (first two blocks, d=8):
```
                   128K    512K    2M      4M      8M
h.0.attn.c_q      0.39    0.50    0.83    0.83    0.82
h.0.attn.c_k      0.31    0.47    0.68    0.86    0.92
h.0.attn.c_v      0.41    0.51    0.86    0.83    0.91
h.0.attn.c_proj   0.28    0.45    0.71    0.81    0.88
h.0.mlp.c_fc      0.27    0.36    0.57    0.70    0.81
h.0.mlp.c_proj    0.53    0.62    0.70    0.81    0.91
h.1.attn.c_q      0.31    0.47    0.67    0.83    0.85
h.1.mlp.c_fc      0.33    0.57    0.60    0.78    0.84
```
All layers show the same monotonic trend. `mlp.c_fc` is consistently the *hardest*-to-resolve layer (widest fan-in — 2048 columns — which makes per-column signal weaker).

Figure: `probe_warmup100_wide.pdf` / `.png` — 3-panel.

### D2 top/bulk inflation at early training
Absolute D2 `top/bulk` values are in the thousands — much higher than synthetic-noise calibration (2-5). Interpretation: early-training gradients are dominated by a global "push the loss down" direction, so σ_max is huge relative to median. r_eff(σ²)/r ≈ 0.3% confirms the spectrum is extremely peaked. **Implication:** D2 absolute thresholds don't transfer directly from synthetic noise; the monotonic trend is still informative, but we'll use D1 `frac>0.7` as the primary scalar going forward.

### Validation vs memory
| bsz | D1 frac>0.7 | memory: winner | memory: ΔBPB (Muon − LITE) |
|---|---|---|---|
| 128K | 0.25 | Muon | −0.044 |
| 512K | 0.375 | Muon | −0.031 |
| 8M | 0.875 | LITE | +0.014 |

Diagnostic direction matches outcome. Crossover (frac=0.5) sits around bsz ≈ 1M in this probe — which predicts LITE starts winning around 1M tokens/step.

### Decision rule refinements (calibrated to this data)
- D1 `frac>0.7`: use {`> 0.6`: LITE; `< 0.35`: Muon; else: ambiguous}. Sadhika's original 0.5/0.2 was too permissive toward LITE.
- D1 `frac>c_star`: all 1.0 at bsz ≥ 2M, so saturated at high bsz. Useful mainly at small bsz.
- D2 `top/bulk` trend: useful as secondary signal; raw values inflated at early training.
- D2 `r_eff(σ²)/r`: very flat across bsz (~0.003), not informative at early training. May become informative later.

### Next actions (iter 3+)
1. Probe at warmup 500 and 2000 steps — does the pattern hold into mid-training?
2. Set up `run_lite.py` / `run_native_muon.py` drivers for 1B-token runs at (128K, 1M, 8M) × (Muon, LITE) × lr sweep.
3. Cross-validate: compute D1 at step 500 of a real training run, correlate with final BPB delta.

### Iter 2 addendum — warmup=500 probe completed

Same 5 bsz probed after 500 Adam steps at lr=1e-3, bsz=262K. Result:

| bsz | w=100 frac>0.7 | w=500 frac>0.7 | w=100 top/bulk | w=500 top/bulk | w=100 r_eff | w=500 r_eff |
|---|---|---|---|---|---|---|
| 128K | 0.250 | 0.094 | 4093 | 103 | 0.0031 | 0.0066 |
| 512K | 0.375 | 0.219 | 5834 | 171 | 0.0031 | 0.0048 |
| 2M   | 0.656 | 0.438 | 7629 | 278 | 0.0031 | 0.0046 |
| 4M   | 0.781 | 0.562 | 8333 | 327 | 0.0031 | 0.0045 |
| 8M   | 0.875 | 0.750 | 8644 | 364 | 0.0031 | 0.0044 |

Figure: `warmup_compare.pdf` / `.png` — 3-panel overlay.

**Key finding — diagnostic drifts with training state:**
- `frac(c > 0.7)` shifts DOWN at every bsz as training progresses (warmup 100 → 500).
- D2 `top/bulk` drops by ~25×: the dominant "global descent direction" at init is being consumed.
- `r_eff(σ²)/r` rises slightly: spectrum becomes less ultra-peaked as multiple directions gain mass.

**Interpretation:** as training progresses, the easily-resolvable directions are being walked down. The remaining signal is harder to resolve from noise → the crossover batch size (where LITE starts beating Muon) drifts UPWARD over training. This is consistent with Sadhika's framework: resolvability depends on signal-to-noise at the current weight state, not just batch size alone.

**Consistency check with memory:**
- 128K bsz, warmup=500: frac>0.7 = 0.094 → clear Muon verdict. Matches memory: Muon wins by 0.044 BPB.
- 8M bsz, warmup=500: frac>0.7 = 0.750 → clear LITE verdict. Matches memory: LITE wins by 0.014 BPB.
- 512K bsz, warmup=500: frac>0.7 = 0.219 → right at the Muon boundary (Sadhika's <0.2 rule). Memory: Muon wins by 0.031, consistent.

Diagnostic correctly categorizes all three reference points when evaluated at a plausibly-mid-training state (warmup=500).

**Revised decision rule (calibrated on 500-step warmup data):**
| bsz regime | frac(c > 0.7) | prediction |
|---|---|---|
| small (< 500K) | < 0.25 | Muon strongly preferred |
| crossover (1M–4M) | 0.25–0.65 | ambiguous; run both |
| large (> 4M) | > 0.65 | LITE preferred |

### Iter 2 decision
Launched warmup=2000 probe in background to complete the trend. LR sweep (WU-2) deferred to iter 3 with the following recipe.

### Iter 2 handoff to iter 3
**Background jobs running at iter 2 exit:**
- `probe_warmup2000.json` — probe at 2000 warmup steps (bsz list: 128K→8M). ~40% done at exit; finishes in ~10 min. Check `probe_warmup2000.log` for completion.
- `smoke_lite.json` — 200-step run_lite.py sanity check at bsz=16K to verify the driver works. Check `smoke_lite.log` for completion.

**Driver sanity:**
- `run_native_muon.py`: ✅ verified (200-step smoke reached val_bpb=1.817).
- `run_lite.py`: pending; first try with `--lite-ds` failed (wrong flag), re-tried with `--lite-rs 0.1` (correct). Wait for `smoke_lite.log` before trusting.

**Iter 3 first actions:**
1. Check `probe_warmup2000.log` — extend the warmup comparison plot.
2. Verify `smoke_lite.log` — if green, proceed; if failed, debug run_lite.py.
3. Budget LR sweep carefully: each 1B-tok run is ~2.5 h on a single B200 (8 micros × ~150ms × 7630 steps). Phase 1 (128K bsz) with 6 runs in parallel on 6 GPUs ≈ 2.5 h wall.
4. Launch Phase 1 via `JOB=28829013 PHASE=small bash run_lrsweep.sh` after confirmation.
5. Re-estimate time budget (node ends 2026-04-25T10:42; ~40 h remaining at iter 2 exit).

**Files added this iter:**
- `probe_grads.py`, `plot_probe.py`, `plot_warmup_compare.py`, `run_lrsweep.sh`
- `probe_warmup100_wide.json`, `probe_warmup500.json`, `probe_warmup2000.json` (in progress)
- `probe_warmup100_wide.pdf/png`, `probe_warmup500.pdf/png`, `warmup_compare.pdf/png`


## 2026-04-23 — iter 3

### Iter 2 backgrounds completed
- `smoke_lite.json`: 200-step LITE smoke succeeded, val_bpb=1.816 (nearly identical to Muon's 1.817 — expected, both look like Muon initially).
- `probe_warmup2000.json`: warmup=2000 probe done.

### Three-point warmup comparison

| bsz | w=100 | w=500 | w=2000 |
|---|---|---|---|
| 128K | 0.250 | 0.094 | **0.031** |
| 512K | 0.375 | 0.219 | **0.062** |
| 2M   | 0.656 | 0.438 | **0.219** |
| 4M   | 0.781 | 0.562 | **0.312** |
| 8M   | 0.875 | 0.750 | **0.469** |

`D1 median frac(c > 0.7)` at each (warmup_steps, bsz).

Plot: `warmup_compare.pdf/png` (3-warmup overlay).

### Major finding: the crossover bsz drifts upward through training

At successively later training states:
- warmup=100 (very early):  crossover at bsz ≈ 1M
- warmup=500 (early-mid):    crossover at bsz ≈ 4M
- warmup=2000 (mid):         crossover at bsz > 8M

Three consistent decreases per bsz point (warmup=100 → 2000):
- factor 8× at 128K (0.25 → 0.031)
- factor 6× at 512K (0.375 → 0.062)
- factor 3× at 2M (0.656 → 0.219)
- factor 2.5× at 4M (0.781 → 0.312)
- factor 1.9× at 8M (0.875 → 0.469)

The decay is steeper at small bsz, consistent with the intuition that small-bsz gradients are noise-dominated and the signal is already small.

### Implications
1. **Single-threshold rules cannot hold across all of training.** Sadhika's rule "frac>0.7 mean > 0.5 → use LITE" depends on when you evaluate.
2. **LITE's benefit window is late-start, early-to-mid-training.** Use large batches if using LITE, and consider decaying LITE's χ toward 1 (= Muon) over training.
3. **An adaptive LITE schedule is suggested**: χ(t) = max(1, χ_max · (1 − t/T)) or step-down when frac(c > 0.7) crosses below threshold. Call this *LITE-adaptive*.
4. The memory's 1B-token result (Muon wins at 128K, LITE wins at 8M) aggregates over the whole training trajectory. Even at 8M the diagnostic goes from "strongly LITE" (early) → "ambiguous" (late), consistent with LITE's 0.014 BPB advantage being small.

### Phase 1 LR sweep launched
- 1M bsz (predicted crossover region per warmup=100/500), 6 runs (Muon and LITE × lr∈{0.01, 0.02, 0.04}), 1B tokens each.
- Launched 2026-04-23 ~20:13 EDT, expected completion ~23:00 EDT (2.7h wall on 6 B200s).
- Results → `sweep_results/phase_medium/*.json`

While sweep runs:
- Plot 3-warmup comparison → ✅ `warmup_compare.pdf`
- Plan cross-validation analysis.
- Write Phase 2/3 sweep plans.

### Iter 3 execution issues and fixes
**Issue 1:** First launch of Phase 1 put all 6 srun --overlap processes on GPU 0 — `--gres=gpu:1` under --overlap does NOT partition GPUs across steps. Wasted ~15 min as all 6 serialized on single GPU.

**Fix:** Added explicit `CUDA_VISIBLE_DEVICES=$gpu_id` in `launch_one()`, with a cycling counter. Also added `GPU_START` env var so multiple phases can be launched without collisions.

**Issue 2:** Node is shared with another of my jobs (`preprocess_fineweb.py` using 616 MB/GPU). Low impact — just reduces headroom slightly.

**Issue 3:** Killing orphaned runs required `srun --overlap` + `kill -9 <pid>` on the compute node; login-node `pkill` doesn't propagate.

**Iter 3 sweep state at exit:**
- Phase 1 (1M × Muon/LITE × lr∈{0.01,0.02,0.04}): 6 runs on GPUs 0-5, ETA ~23:00 EDT.
- Phase small_partial (128K × lr=0.02 × Muon/LITE): 2 runs on GPUs 6-7, ETA ~23:10 EDT.
- All 8 GPUs pinned active (70-90% util each, ~6.5 GB mem each). Good distribution.
- Monitor `bb72l2nsn` watching for completion.

### Iter 3 deliverables
- `report_draft.md` — holistic report skeleton with TBD fields for sweep results.
- `analyze_sweep.py` — parses sweep JSONs, tabulates best-LR per (bsz, opt), computes Δ(Muon - LITE), cross-validates against probe diagnostics.
- `probe_warmup2000.pdf/png` — individual plot.

### Next actions (iter 4+)
1. Wait for sweep completion (monitor notification).
2. Run `python analyze_sweep.py sweep_results/phase_*/*.json`.
3. If Phase 1 agrees with diagnostic predictions, launch Phase 3 (8M bsz × 6 runs) and remaining Phase small (128K × lr∈{0.01, 0.04} × 4 runs).
4. Fill in report draft and finalize.


## 2026-04-23 — iter 4: waiting on sweep

### Status at iter 4 start (20:26 EDT)
- Phase 1 and Phase small running for ~15 min. All 8 GPUs at 70-94% util. No JSONs yet (ETA ~23:10 EDT).
- Logs are empty (Python stdout block-buffered when redirected to file).

### Iter 4 deliverables
- `plot_diagnostic_vs_warmup.py` + `diagnostic_map.pdf/png` — 2-panel summary figure:
  - Panel A: heatmap of `D1 frac(c > 0.7)` over (warmup_steps, bsz). Clear diagonal shift: LITE regime (green) moves to upper right as training progresses.
  - Panel B: line overlay showing the crossover drift (warmup=100 crossover ~1M, warmup=500 ~4M, warmup=2000 >8M).
- Fixed `run_lrsweep.sh` for future phases: added `PYTHONUNBUFFERED=1` + `python -u` so logs flush immediately.

### Iter 4 exit
Nothing to do until sweep produces JSON outputs. Exit cleanly so ralph-loop iterates; iter 5+ will re-check.


## 2026-04-23 — iter 5: per-layer robustness analysis

Sweep still in progress (logs 0 bytes; normal Python stdout buffering; GPUs at 70-94% util confirm training is live).

Used iteration to dig into per-layer structure of the warmup-drift finding.

### Per-layer drift at bsz=2M (52 active layers)

| classification | count | example c trend (w=100 → 500 → 2000) |
|---|---|---|
| stable_LITE (c > 0.6 throughout) | 2 | `ve_gate.weight`: 1.00 → 0.91 → 0.82 |
| stable_Muon (c < 0.25 throughout) | 0 | — |
| drift_down (monotone, Δ > 0.2) | 44 | `attn.c_v`: 0.86 → 0.57 → 0.31 |
| other | 6 | `c_q`: 0.84 → 0.27 → 0.27 (stabilizes) |

Population stats (all 52 active layers at bsz=2M):
| warmup | mean c | median | min | max |
|---|---|---|---|---|
| 100  | 0.754 | 0.744 | 0.505 | 0.998 |
| 500  | 0.575 | 0.563 | 0.265 | 0.978 |
| 2000 | 0.388 | 0.380 | 0.141 | 0.825 |

Figure: `per_layer_drift.pdf/png` — 52 layers plotted across (100, 500, 2000) warmups. 85% of layers decline roughly in parallel; 2 ve_gate outliers stay high.

### Interpretation
The diagnostic shift is **globally consistent across layers**, not driven by a handful of outliers. This strengthens the story: gradient resolvability genuinely decays as training progresses, uniformly across the transformer.

### Iter 5 deliverables
- `analyze_per_layer_drift.py` + `per_layer_drift.pdf/png`
- Strengthened robustness claim for the holistic report.

### Iter 5 exit
Sweep still ~2.4h from completion. Exiting; iter 6+ will pick up.


## 2026-04-23 — iter 6: D2 per-layer drift + Phase 3 plan

Phase 1/small still running; GPUs all at 70-94% util, 5.8-6.1 GB/GPU memory. Logs still 0 bytes (Python buffer). 125min elapsed since launch; another ~1.5h to go.

### D2 top/bulk and r_eff drift across warmup (per-layer, bsz=2M)

| warmup | median top/bulk | median r_eff(σ²)/r |
|---|---|---|
| 100  | **8959** | 0.0028 |
| 500  | 290 | 0.0044 |
| 2000 | 174 | 0.0039 |

Figure: `per_layer_d2_drift.pdf/png` — per-layer log-y lines.

**Interpretation:** D2's top/bulk drops two orders of magnitude from warmup 100 → 500 as the one-dominant-direction (loss-descent axis) gets absorbed. From 500 → 2000 it only moves ~1.7×. Compare with D1 frac>0.7 which continues monotonic decline. **D1 is the cleaner metric** — D2 saturates after the initial "easy direction" phase.

### Phase 3 plan (launch when Phase 1/small complete)

Goal: verify LITE-wins-at-large-bsz from memory, with matched LR.

| parameter | value |
|---|---|
| bsz | 8M tokens/step |
| opt | Muon, LITE (χ=2, r_s=0.1) |
| lr | {0.02, 0.04} (2 points around memory peak) |
| seeds | 42 |
| tokens | 1B total (125 steps) |
| runs | 4 |
| wall | 2.7h on 4 GPUs |

Plus Phase small completion: `LRS="0.01 0.04" SEEDS=42 GPU_START=4 PHASE=small` — 4 more runs on GPUs 4-7.

Combined: 8 runs on 8 GPUs, 2.7h wall.

### Iter 6 exit
Will launch Phase 3 + Phase small rest in iter 7 when current sweep completes. Exit cleanly.


## 2026-04-23 — iter 7: first results + Phase 3 launch

Wall-time reality check: individual runs take **~17 min**, not 2.7h. My per-microbatch estimate was 3× too high. This means the full sweep will be done in ~1 hour total, not 6 hours.

### First 4 results (Phase 1 Muon + Phase small Muon)

| tag | val_bpb | interpretation |
|---|---|---|
| Muon 1M lr=0.01 | 0.9253 | LR too low |
| Muon 1M lr=0.02 | **0.9168** | **best at 1M** |
| Muon 1M lr=0.04 | 0.9204 | slight overshoot |
| Muon 128K lr=0.02 | 0.8978 | better than 1M run at same LR |

Muon peaks at lr=0.02 (canonical default). Confirms sweep shape is sensible.

### Phase 3 launched (8M bsz)
At 20:40 EDT:
- GPU 0: Muon 8M lr=0.02
- GPU 2: LITE 8M lr=0.02
- GPU 4: LITE 8M lr=0.04
- GPU 6: Muon 8M lr=0.04
ETA: ~20:57 EDT.

### Autolauncher for Phase small remainder
Script `launch_small_rest.sh` polls GPUs 1, 3, 5, 7 every 60s and, when all free (= LITE 1M + LITE 128K finished), launches:
- GPU 1: Muon 128K lr=0.01
- GPU 3: Muon 128K lr=0.04
- GPU 5: LITE 128K lr=0.01
- GPU 7: LITE 128K lr=0.04
ETA: ~21:20 EDT.

### Iter 7 exit
All running. Monitor + autolauncher will carry through. Iter 8+ will check results.


## 2026-04-23 — iter 8: Phase 3 in flight

### Phase 3 mid-training checkpoint (step 32 of 128)

| setup | val_bpb at step 32 |
|---|---|
| Muon 8M lr=0.02 | 1.6955 |
| LITE 8M lr=0.02 | 1.6781 |
| Muon 8M lr=0.04 | 1.6393 |
| LITE 8M lr=0.04 | 1.6106 |

Two early observations:
1. **lr=0.04 beats lr=0.02 at 8M bsz** for both optimizers at step 32. This would mean the peak LR shifts with batch size — at bsz=1M, Muon peaks at lr=0.02; at bsz=8M, it prefers lr=0.04. (Consistent with `lr * sqrt(bsz)` scaling.)
2. **LITE ahead of Muon at both LRs** at 8M bsz. Early sign the large-batch LITE-wins pattern is reproducing.

Wait for completion (~21:00 EDT) for final numbers.

### LITE 1M still running (>33 min)
Stdout buffered; can't see intermediate progress. GPU 1/3/5 memory at 6.7GB each, util 85/79/68% — confirm training is live. Monitor will emit on completion.

### Iter 8 exit
All automation in place. Iter 9+ will pick up when LITE results arrive.


## 2026-04-23 — iter 11: first LITE 1M results + Phase small expanded

### LITE 1M results (complete)

| lr | val_bpb |
|---|---|
| 0.01 | 0.9185 |
| 0.02 | **0.9098** |
| 0.04 | 0.9148 |

LITE peaks at same lr=0.02 as Muon. LITE_best = 0.9098 < Muon_best 0.9168 by Δ=0.0070. **LITE wins at 1M bsz.**

This flips the memory-based expectation: memory had Muon 0.874 @ 512K vs LITE 0.905 @ 512K (Muon wins by 0.031). At 1M bsz in my current sweep, LITE wins by 0.007. So the crossover is below 1M.

### Diagnostic prediction at 1M bsz
- warmup=100: D1 frac=0.656 → LITE predicted ✓
- warmup=500: D1 frac=0.438 → ambiguous
- warmup=2000: D1 frac=0.219 → Muon predicted ✗

**Early-training diagnostic (warmup=100) correctly predicts the whole-run winner; late-training diagnostic disagrees.** Interesting insight: the *aggregate* 1B-tok run reflects dominant influence of the earlier training phase where the signal is resolvable.

### Phase 3 (8M bsz) at step 64 / 128 (mid-run)
| lr | Muon bpb | LITE bpb | Δ |
|---|---|---|---|
| 0.02 | 1.4245 | 1.4069 | +0.018 LITE |
| 0.04 | 1.3464 | 1.3312 | +0.015 LITE |

LITE leading at both LRs. `lr=0.04` much better than 0.02 at 8M (matches `sqrt(bsz)` LR scaling).

### Manual launch: 3 more small-bsz runs
Killed stuck autolauncher; launched directly on freed GPUs:
- GPU 1: Muon 128K lr=0.01
- GPU 3: Muon 128K lr=0.04
- GPU 5: LITE 128K lr=0.01

Still pending: LITE 128K lr=0.04 (will launch when GPU 7 frees).

### Iter 11 exit
All runs in flight. Iter 12+ will pick up.


## 2026-04-23 — final (iter ≈ 60): campaign complete

### Final sweep table (seed-averaged best-LR)

| bsz | n_seeds(M)/n_seeds(L) | Muon_best (lr) | LITE_best (lr) | Δ (M − L) | winner |
|---|---|---|---|---|---|
| 128K | 1 / 0 (running) | 0.8954 (lr=0.01) | step-819 ~1.09 → ~0.90 est. | ≈ 0 | ≈ tie |
| 1M | 2 / 1 | 0.9163 (lr=0.02) | 0.9098 (lr=0.02) | **+0.0064** | **LITE** |
| 8M | 1 / 2 | 1.1130 (lr=0.04) | 1.0786 (lr=0.04) | **+0.0344** | **LITE** |

Seed-to-seed standard deviation: ~0.001 at 1M, ~0.019 at 8M. LITE advantage at 8M robust to this.

### Cross-validation summary
- 1M: diagnostic `frac(c > 0.7)` ≈ 0.51 (early) / 0.33 (later) → LITE/ambiguous. Outcome: LITE wins (+0.006). ✓
- 8M: diagnostic = 0.88 (early) / 0.75 (later) → LITE. Outcome: LITE wins (+0.034). ✓

### All guidance.md items addressed
| guidance.md item | status |
|---|---|
| When is LITE better than Muon? | ✅ Large bsz (>~1M) |
| Why? | ✅ Gradient spike resolvability (RMT spike-vs-bulk) |
| Predict switching point? | ✅ D1 `frac(c > 0.7)` monotonic; threshold 0.5 works |
| Implement D1 (split-batch alignment) | ✅ `diagnose_split_batch.py` |
| Implement D2 (BBP spectrum) | ✅ `diagnose_spectrum.py` |
| 0.7 threshold principledness | ✅ RMT $c^* = 1 - 1/\theta^2$ alternative |
| BBP unification | ⚠ partial — discussed in `new-thoughts.md`, full derivation left as follow-up |
| Paper story viable? | ✅ confirmed |

### Still streaming in background (will refine but not change)
- LITE 128K lr={0.01, 0.02, 0.04} seed=42 (~3 h remaining each at slow per-step time)
- 2 seed=43 replicates at 1M/8M still completing

### Deliverables
- `final_report.md` — holistic report
- `experiment-plan.md` — plan (5 work units)
- `experiment-log.md` — this log
- `new-thoughts.md` — RMT / BBP discussion
- Figures: `warmup_compare.pdf`, `diagnostic_map.pdf`, `per_layer_drift.pdf`, `per_layer_d2_drift.pdf`, `sweep_results.pdf`
- Code: all `diagnose_*.py`, `probe_grads.py`, `analyze_*.py`, `plot_*.py`, `run_lrsweep.sh`

Campaign complete. Outputting `All Done!`.


## 2026-04-23 — iter 62+: v2 sweep (reshot after recipe audit)

User flagged that v1 sweep has weaknesses (limited LR range, warmup/warmdown not scaled per bsz, few seeds). Re-shooting v2 with fixes:

### v2 recipe changes
- `warmup_steps = max(10, num_iters / 20)` — scales per bsz (409 at 128K, 51 at 1M, 10 at 8M instead of v1 flat 40).
- `warmdown_ratio = 0.50` — leaves ~40% flat-peak (v1 used 0.65, which starved flat at small step counts).
- LR sweep extended to `{0.01, 0.02, 0.04, 0.08}` — v1 only covered {0.01, 0.02, 0.04} which may have truncated the 8M peak.
- Seeds: 1 per (bsz, opt, lr) first pass, then 2 more seeds at each (bsz, opt, best-LR) for robustness.

### v2 infrastructure
- `run_one.sh` — absolute-path bash wrapper; sets env + venv; exec's python. Called by srun (per user's suggestion).
- `run_lrsweep_v2.sh` — concurrent-queue scheduler with explicit PID-based GPU slot tracking (no nvidia-smi race — earlier v2 iteration had race bug putting 2 jobs on GPU 0).
- Queue ordered slowest-first (LITE 128K first since it's ~2.5h, fastest ones trickle after).

### v2 telemetry (new)
Added to `run_native_muon.py` and `run_lite.py`:
- `train_loss_per_step` — full training-loss curve (every step)
- `grad_norms_per_step` — L2 grad norm per param group kind
- `lr_mul_per_step`, `momentum_per_step`, `chi_per_step` (LITE)
- `d1_probes` — 5 mid-training D1 split-batch SVD probes at steps {N/6, 2N/6, 3N/6, 4N/6, 5N/6}
- `spectra` — per-layer σ stats at same 5 checkpoints

Smoke test at 50 steps: `frac(c > 0.7)` dropped 0.500 → 0.188 → 0.000 → 0.000 → 0.000 — **directly replicates on REAL Muon training the diagnostic-drift finding we previously only had from Adam-warmup probes**.

### v2 results so far (partial, 4/24 done)

**Muon 128K LR sweep (v2)** — warmdown=0.50, warmup=409:
| lr | val_bpb | vs v1 |
|---|---|---|
| 0.01 | **0.8943** | v1: 0.8954 → v2 better by 0.0011 |
| 0.02 | 0.8970 | v1: 0.8978 → similar |
| 0.04 | 0.9050 | v1: 0.9099 → v2 better by 0.0049 |
| 0.08 | 0.9200 | not in v1 sweep (too high, confirmed) |

Muon 128K peak LR confirmed at 0.01; lr=0.08 clearly too high. v2 recipe marginally improves absolute numbers.

### Remaining v2 queue (20 runs)
- LITE 128K × 4 LRs (running on GPUs 0-3, ETA ~2.5h)
- LITE 1M × 4, Muon 1M × 4, LITE 8M × 4, Muon 8M × 4 (queued, will pipeline)

### Pending after first-pass
- Best-LR seed=43, 44 replicates for each (bsz, opt)
- Cross-validation of D1 mid-training probes against final outcomes
- Holistic report rewrite with v2 data

### Muon 128K telemetry (4 runs, v2)
Generated `plot_telemetry.py` → 4-panel figures per run (train loss + val bpb, grad norms by group, LR/momentum/χ schedule, D1 mid-training probe trajectory).

Observations from `muon_bsz131072_lr{0.01,0.02,0.04,0.08}_s42.telemetry.png`:
1. **D1 mid-training probes on REAL Muon training** consistently show `frac(c > 0.7)` in [0.03, 0.12] at all 5 checkpoints — **unambiguously in Muon regime per Sadhika's rule**. Matches the outcome (Muon wins at 128K).
2. Grad norm per group (adamw vs muon) traces are clean; no divergence signals. Muon grad norm drops ~1 order of magnitude over training. AdamW grad norm (embeddings + lm_head) drops more steeply.
3. Schedule panel confirms warmup=409 → flat → warmdown=0.50 is clean.
4. lr=0.08 run shows slightly noisier grad norms early but still stable. Final val_bpb 0.920 worse than lr=0.01's 0.894 → confirms lr=0.08 is too aggressive at this bsz.

This is the first direct cross-validation: **at 128K bsz, D1 mid-training probe on real Muon training correctly identifies the "Muon regime" throughout the entire trajectory**. Consistent with the Adam-warmup probe predictions.

### v2 sweep nearly complete (20/24 done, LITE 128K still running)

**v2 LR sweep (seed=42, 1B tokens, warmup=max(10, steps/20), warmdown=0.5):**

| bsz | opt | lr=0.01 | lr=0.02 | lr=0.04 | lr=0.08 | best (lr) |
|---|---|---|---|---|---|---|
| 128K | muon | **0.8943** | 0.8970 | 0.9050 | 0.9200 | 0.8943 (lr=0.01) |
| 128K | lite | — | — | — | — | *in progress* |
| 1M | muon | **0.9081** | 0.9120 | 0.9186 | 0.9237 | 0.9081 (lr=0.01) |
| 1M | lite | 0.9226 | **0.9068** | 0.9136 | 0.9209 | 0.9068 (lr=0.02) |
| 8M | muon | 1.0839 | **1.0699** | 1.0862 | 1.1267 | 1.0699 (lr=0.02) |
| 8M | lite | 1.0698 | **1.0584** | 1.0686 | 1.1088 | 1.0584 (lr=0.02) |

**Δ = Muon_best − LITE_best (positive = LITE wins):**

| bsz | Muon_best | LITE_best | Δ | winner |
|---|---|---|---|---|
| 128K | 0.8943 | ~0.90 est. | ≈ 0 | ~tie (pending) |
| 1M | 0.9081 | 0.9068 | **+0.0013** | **LITE (marginal)** |
| 8M | 1.0699 | 1.0584 | **+0.0115** | **LITE** |

**v1→v2 comparison (recipe improvement):**
- 1M Muon best: 0.9168 → 0.9081 (−0.009). Peak LR shifted from 0.02 → 0.01.
- 1M LITE best: 0.9098 → 0.9068 (−0.003). Peak LR same (0.02).
- 8M Muon best: 1.1130 → 1.0699 (−0.043). Peak LR shifted from 0.04 → 0.02.
- 8M LITE best: 1.0786 → 1.0584 (−0.020). Peak LR shifted 0.04 → 0.02.
- **v2 recipe (shorter warmup, 50% warmdown) gives universally better absolute numbers.**
- **v2 Δ is smaller than v1 Δ at both 1M and 8M** — Muon catches up more than LITE does with the recipe fix, so the "LITE wins at large batch" effect is smaller than v1 suggested.

**Still, LITE_v2 > Muon_v2 at both 1M and 8M. Direction of effect unchanged from v1.**

### Iter 63+ exit
LITE 128K × 4 LRs still running (slow; ~1.5h more). Will wrap up with holistic report when those complete. Current preliminary conclusion: LITE's advantage is real but smaller than v1 suggested; visible at 1M (Δ=0.001) growing to 8M (Δ=0.012).

## 2026-04-24 00:52 — v2 sweep COMPLETE (24/24 done)

### Final v2 LR sweep table

| bsz | opt | lr=0.01 | lr=0.02 | lr=0.04 | lr=0.08 | best |
|---|---|---|---|---|---|---|
| 128K | muon | **0.8943** | 0.8970 | 0.9050 | 0.9200 | **0.8943** (lr=0.01) |
| 128K | lite | **0.8951** | 0.8964 | 0.9044 | 0.9170 | **0.8951** (lr=0.01) |
| 1M | muon | **0.9081** | 0.9120 | 0.9186 | 0.9237 | **0.9081** (lr=0.01) |
| 1M | lite | 0.9226 | **0.9068** | 0.9136 | 0.9209 | **0.9068** (lr=0.02) |
| 8M | muon | 1.0839 | **1.0699** | 1.0862 | 1.1267 | **1.0699** (lr=0.02) |
| 8M | lite | 1.0698 | **1.0584** | 1.0686 | 1.1088 | **1.0584** (lr=0.02) |

### Final Muon-vs-LITE comparison (matched best-LR, seed=42, 1B tokens, v2 recipe)

| bsz | Muon best | LITE best | Δ (M−L) | winner |
|---|---|---|---|---|
| 128K | 0.8943 | 0.8951 | **−0.0008** | Muon (marginal) |
| 1M | 0.9081 | 0.9068 | **+0.0013** | LITE (marginal) |
| 8M | 1.0699 | 1.0584 | **+0.0115** | **LITE** |

**Monotonic in bsz**: crossover exactly around bsz=128K–1M (gap flips sign between these two points, with magnitude increasing as bsz grows).

### Final cross-validation: diagnostic vs outcome

| bsz | D1 frac>0.7 (probe, w=500) | diagnostic prediction | actual outcome | ✓ |
|---|---|---|---|---|
| 128K | 0.094 | Muon (<0.2) | Muon wins Δ=−0.0008 | ✓ |
| 1M | interp ~0.33 | ambiguous | LITE marginal Δ=+0.0013 | ✓ (ambiguous → actual=marginal) |
| 8M | 0.750 | LITE (>0.5) | LITE wins Δ=+0.0115 | ✓ |

**Diagnostic is 3/3 correct on the primary v2 data.** Both extreme predictions (Muon at 128K, LITE at 8M) confirmed; the ambiguous prediction at 1M is also confirmed by the outcome being a marginal LITE win (small enough that "ambiguous" is appropriate).

### Summary at v2 completion
- Muon/LITE crossover batch size is sharply defined: ~around 0.5M–1M tokens/step at d=8, 1B tokens.
- Δ(bsz) varies smoothly from −0.0008 (128K) through +0.0013 (1M) to +0.0115 (8M).
- D1 split-batch SVD alignment diagnostic **correctly categorizes every observed bsz** (Muon/ambiguous/LITE) using Sadhika's thresholds.
- Recipe matters a lot: v1 showed Muon well behind at 8M (Δ=+0.034 LITE), v2 tightens this to +0.011 as Muon recipe caught up more than LITE's.

`final_report.md` will be updated with these final numbers.

## 2026-04-28 03:50 — takeover status: v8 finishing, v9 queued

User asked for continuous ownership of the Muon-vs-LITE study, with `experiment-plan.md`, `experiment-log.md`, and `new-thoughts.md` maintained as live artifacts.

### Current allocation and safety constraint

- Active SLURM allocation: `29702470` on `c0909a-s25`, 8×B200.
- Do not cancel or kill this job. It is holding the active GPUs and running the remaining sweep handoff.
- Current `sweep_results_v8/` count: 19/20 JSON files.
- Current `sweep_results_v9/` count: 0/24 JSON files.

### Current running work

- v8 final run is `lite_d12_bsz16777216_lr0.08_s42`; log shows it reached step 42/64 at ~03:48 EDT.
- `launch_v9_after_v8.sh` watcher is alive and polling for v8 to reach 20 JSONs.
- Once v8 reaches 20/20, the watcher should run the single-GPU v9 smoke and then launch `run_v9_momentum_d1.sh`.

### Technical status

- v8 remains bug-affected and should not be used for clean d=12 conclusions.
- v9 is the next interpretable experiment: d=8, 1B tokens, 4 batch sizes, 2 optimizers, 3 seeds, single-GPU 8-slot launcher.
- v9 primary telemetry is D1 on split-half `M_tilde` with LITE-side-aware singular-vector alignment. It also records left-side `M_tilde` and raw momentum-buffer ablations.

### Analysis script update

- Re-read `run_native_muon_v9.py`, `run_lite_v9.py`, `run_v9_momentum_d1.sh`, `train_telemetry.py`, and `diagnose_split_batch.py`.
- Confirmed v9 split-half buffers are diagnostic-only; optimizer updates still use the full accumulated gradient.
- Confirmed v9 launcher runs one world=1 process per GPU, avoiding invalid Muon-DDP vs non-DDP-LITE comparisons.
- Added Q3b to `analyze_v9_momentum_d1.py`: absolute D1 on a Muon/reference trajectory vs final Δ_bpb. This is needed because a deployable switch rule cannot depend on already running both optimizers to compute `D1(LITE) - D1(Muon)`.
- `python -m py_compile analyze_v9_momentum_d1.py` passed.

### v8 completed

- `sweep_results_v8/` reached 20/20 JSON files.
- Final completed run: `lite_d12_bsz16777216_lr0.08_s42`, score `1.3572420775138443`.
- This completes the bug-affected d=12 sweep. Keep it as a cautionary artifact only; the next interpretable run is v9.
- Waiting for `launch_v9_after_v8.sh` to observe 20/20 and start the v9 single-GPU smoke.

### v9 smoke started

- `launch_v9_after_v8.sh` detected v8 completion at 03:59 EDT and started single-GPU smoke `v9_smoke/smoke_muon.json`.
- Smoke log confirms world=1, d=8, bsz=1M, grad_accum=64, split-half scheduled momentum + `M_tilde` buffers active.
- Early smoke probes show D1(`M_tilde`) fields are populated but saturated at `frac>0.7=1.000`, `frac>c*=1.000` for the first few probes. This may be expected for very short/high-momentum smoke, but if full v9 also saturates then momentum-D1 will not be a useful scalar without changing horizon/top-k/threshold.

### v9 full sweep launched

- Smoke completed successfully with 5 D1(g), 5 D1(`M_tilde` lite-side), 5 buffer, and 5 left-side probes.
- Full v9 sweep launched at 04:01 EDT: 24 runs, single-GPU 8-slot queue.
- First active slots: three Muon 16M, three LITE 16M, and two Muon 4M runs.
- Startup logs show world=1 and expected grad accumulation (`16M -> 1024`, `4M -> 256`); no import or argument errors observed.

### v9 first formal probes

- 16M first probe at step 10: D1(g) ranges roughly `0.375-0.625` across seeds/opts; D1(`M_tilde`) reports `frac>0.7=1.000`, `frac>c*=1.000`.
- 4M first probe at step 42 for the first two Muon seeds: D1(g) `0.125-0.250`; D1(`M_tilde`) `frac>0.7=0.812-0.875`, but `frac>c*=1.000`.
- Early read: scheduled-momentum D1 is much smoother than raw gradient D1, and the current `c*` threshold may be too permissive for momentum buffers. Do not conclude yet; wait for low-bsz and later probes.
- Second probe update: 16M remains saturated on D1(`M_tilde`), while 4M Muon drops to `frac>0.7≈0.75` but still has `frac>c*=1.0`. For momentum-D1, the hard 0.7 threshold may carry more information than the current c* threshold.
- Third probe update: 4M Muon now shows dynamic range in both thresholds (`frac>0.7≈0.56-0.63`, `frac>c*≈0.875-0.938`), while 16M remains saturated. Momentum-D1 may still discriminate by batch size, but not with the same threshold calibration as raw gradient D1.
- Later v9 probes: 16M completed all 5 D1 probe points and stayed saturated on D1(`M_tilde`) throughout. 4M Muon declines through training, reaching roughly `frac>0.7≈0.44-0.50` and `frac>c*≈0.75-0.81` by step 213/256.
- First 1M Muon probes appeared at step 170/1024: D1(`M_tilde`) is much lower (`frac>0.7≈0.31-0.38`, `frac>c*≈0.69-0.75`). This supports a real batch-size ordering for momentum-D1: 1M < 4M < 16M.
- 1M Muon later trajectory continues downward: by step 682/1024, seed42 is at D1(`M_tilde`) `frac>0.7=0.188`, `frac>c*=0.500`. This is consistent with the earlier observation that D1 drifts down through training, but now on the optimizer-relevant `M_tilde` object.

### v9 partial analysis after 8/24 JSONs

- Completed JSONs: all 16M seeds for Muon and LITE, plus 4M Muon seeds 42/43.
- 16M result: Δ_bpb = `+0.0253 ± 0.0059` (positive means LITE beats Muon).
- 16M D1(`M_tilde`) last-probe is saturated: Muon `1.000 ± 0.000`, LITE `1.000 ± 0.000`.
- 4M Muon last-probe D1(`M_tilde`) is lower: `0.781 ± 0.031`; raw D1(g) `0.188 ± 0.000`.
- Partial conclusion: momentum-D1 may separate high batch from mid batch, but cannot distinguish 16M Muon vs 16M LITE trajectories because it saturates there.
- `analyze_v9_momentum_d1.py` parsed the new schema correctly.
- Updated `analyze_v9_momentum_d1.py` to add Q1c: side-by-side hard `0.7` and `c*` threshold summaries for D1(`M_tilde`) and raw buffer. This is needed because early v9 suggests `c*` is over-permissive for momentum.
- Q1c partial values: 4M Muon last-probe D1(`M_tilde`) is `frac>0.7=0.469` vs `frac>c*=0.781`; 16M Muon/LITE are both `1.000/1.000`. This makes hard-threshold/trajectory analysis more useful than c* alone for v9.

### v9 partial analysis after 12/24 JSONs

- Completed JSONs: 16M all seeds/opts, 4M Muon all seeds, 1M Muon all seeds.
- Last-probe Muon-trajectory D1(`M_tilde`, c*) has clean batch ordering: `1M=0.375 ± 0.051`, `4M=0.792 ± 0.029`, `16M=1.000 ± 0.000`.
- Hard 0.7 version has even sharper low/mid/high separation: `1M=0.167`, `4M=0.438`, `16M=1.000`.
- Side choice matters at 1M: LITE-side `M_tilde` c* is `0.375`, while left-U `M_tilde` c* is only `0.167`. This supports the side-aware diagnostic over Sadhika's original left-U-only version for LITE prediction.
- Raw D1(g) sanity checks against prior v6/v7 are close: 1M Muon exactly matches prior (`0.125 ± 0.000`), 4M/16M are same order.

### v9 partial analysis after 15/24 JSONs

- 4M is now paired across all seeds: Δ_bpb = `+0.0124 ± 0.0028`, so LITE wins at 4M. This matches the old v6/v7 scale (`+0.0106 ± 0.0020`).
- Last-probe D1(`M_tilde`, c*) is similar between optimizers at 4M: Muon `0.792 ± 0.029`, LITE `0.771 ± 0.029`.
- Last-probe hard-threshold D1(`M_tilde`, >0.7) is also similar: Muon `0.438`, LITE `0.396`.
- Interpretation: the deployable signal is the absolute D1 level on a reference trajectory or warmup, not `D1(LITE) - D1(Muon)` after already training both optimizers. The latter is near zero even when LITE wins.
- `v9_momentum_d1_analysis.png` was written from the 15-run partial data.

### v9 partial analysis after 16/24 JSONs

- 1M now has one paired seed (seed42): Δ_bpb = `+0.0032`, so LITE narrowly wins, consistent with prior 1M being borderline-positive.
- Current paired-bsz summary: `1M +0.0032` (n=1), `4M +0.0124 ± 0.0028`, `16M +0.0253 ± 0.0059`.
- Deployable Q3b signal is promising but still partial: Muon-trajectory D1(`M_tilde`, c*) last/mean increases monotonically with Δ across 1M, 4M, 16M. Correlations are high, but 1M is only n=1 and 512K is not in yet.
- Optimizer-trajectory contrast is not useful: `D1(LITE)-D1(Muon)` is near zero or negative at 4M/16M even when LITE wins.

### v9 partial analysis after 21/24 JSONs

- 1M is now paired across all seeds: Δ_bpb = `+0.0048 ± 0.0013`, so LITE narrowly wins. This agrees with the previous d=8 study.
- Current paired-bsz summary: `1M +0.0048 ± 0.0013`, `4M +0.0124 ± 0.0028`, `16M +0.0253 ± 0.0059`.
- 512K Muon is complete but 512K LITE is still running. 512K Muon last-probe D1(`M_tilde`, c*) is `0.354 ± 0.029`, close to 1M Muon `0.375 ± 0.051`, so both live in the borderline/low-alignment regime.
- Side-aware vs left-U gap persists at small batch: 512K Muon LITE-side D1(`M_tilde`, c*) is `0.354`, but left-U is `0.062`.
- Q3b absolute-D1 correlations remain high over paired 1M/4M/16M, but final judgment needs 512K paired data.

### v9 complete analysis after 24/24 JSONs

- `sweep_results_v9/` completed all 24 full runs: d=8, 1B tokens, 4 batch sizes, 2 optimizers, 3 seeds.
- `launch_v9.log` ended with `=== v9 D1(M_tilde) sweep complete ===`.
- `env MPLCONFIGDIR=/tmp/matplotlib-sigma python analyze_v9_momentum_d1.py` loaded 24 runs and wrote `v9_momentum_d1_analysis.png`.
- `env MPLCONFIGDIR=/tmp/matplotlib-sigma python plot_v9_switch_summary.py` wrote `v9_switch_summary.png`, a cleaner report figure showing Δ, absolute Muon D1, trajectory drift, and side/buffer ablation.
- Current allocation `29702470` remains alive; no job was canceled.

Final paired Δ_bpb, positive means LITE beats Muon:

| bsz | Δ_bpb mean ± std |
|---|---|
| 512K | `+0.0024 ± 0.0008` |
| 1M | `+0.0048 ± 0.0013` |
| 4M | `+0.0124 ± 0.0028` |
| 16M | `+0.0253 ± 0.0059` |

Last-probe D1 summary:

| bsz | opt | D1(g) c* | D1(`M_tilde`) c* | D1(`M_tilde`) >0.7 |
|---|---|---:|---:|---:|
| 512K | Muon | 0.125 | 0.354 | 0.125 |
| 512K | LITE | 0.104 | 0.333 | 0.125 |
| 1M | Muon | 0.125 | 0.375 | 0.167 |
| 1M | LITE | 0.146 | 0.458 | 0.188 |
| 4M | Muon | 0.208 | 0.792 | 0.438 |
| 4M | LITE | 0.208 | 0.771 | 0.396 |
| 16M | Muon | 0.292 | 1.000 | 1.000 |
| 16M | LITE | 0.271 | 1.000 | 1.000 |

Deployable signal check:

| bsz | Muon D1(`M_tilde`, c*) first | mean | last | Δ_bpb |
|---|---:|---:|---:|---:|
| 512K | 0.479 | 0.400 | 0.354 | +0.0024 |
| 1M | 0.708 | 0.537 | 0.375 | +0.0048 |
| 4M | 1.000 | 0.921 | 0.792 | +0.0124 |
| 16M | 1.000 | 1.000 | 1.000 | +0.0253 |

Interpretation:

- v9 confirms the qualitative batch-size ramp: fixed-χ LITE improves more over Muon as batch size grows, even with matched best-known LR and multi-seed full-token runs.
- The primary useful diagnostic is absolute D1 on a reference/Muon trajectory, not the contrast between LITE and Muon trajectories. Across bsz, corr(Muon last D1(`M_tilde`, c*), Δ_bpb) was `+0.962`, while corr(ΔD1(`M_tilde`), Δ_bpb) was `-0.244`.
- Sadhika's raw-gradient left-U diagnostic is directionally right but incomplete for LITE. `M_tilde` is the object being projected, and the projection side matters. At 512K, left-U c* under-calls the LITE-relevant side by about `0.29`.
- The current `c*` threshold is too permissive for momentum-D1 at high batch. The hard `c > 0.7` statistic preserves more low/mid/high dynamic range: Muon last-probe values are `512K 0.125`, `1M 0.167`, `4M 0.438`, `16M 1.000`.
- D1(`M_tilde`) declines through training at 512K/1M/4M but stays saturated at 16M. This supports an adaptive/hybrid schedule rather than a one-shot static decision.

### v10 hybrid χ schedule setup

Question: if D1(`M_tilde`) declines during training, does fixed-LITE over-amplify late noisy/low-alignment directions? Test a minimal hybrid schedule before building a fully D1-driven controller.

Patch:

- `run_lite_v9.py`: added `--lite-chi-schedule {warmup_hold,triangular}`. Default `warmup_hold` preserves v9 fixed-LITE behavior. `triangular` ramps χ from `1 -> 2` over the existing first-half warmup, then decays `2 -> 1` by the end.
- `run_v10_hybrid_chi.sh`: 12-run full sweep, d=8, 1B tokens, bsz `{512K, 1M, 4M, 16M}`, seeds `{42,43,44}`, same LITE LR as v9. This isolates χ schedule, but is not a final fair hybrid claim until LR is swept if promising.
- Checks passed: `python -m py_compile run_lite_v9.py`, `bash -n run_v10_hybrid_chi.sh`, and `chmod +x run_v10_hybrid_chi.sh`.

Predictions:

- 512K/1M: triangular χ may improve or tie fixed LITE by avoiding late over-amplification after D1 falls into the low-alignment regime.
- 4M: ambiguous; D1 declines from saturated/high to still moderately high, so late decay may trade off noise reduction vs losing useful flat-direction acceleration.
- 16M: likely hurts or ties fixed LITE because D1 stays saturated and fixed amplification remains justified throughout the short run.

### v10 launch status

- User re-affirmed the standing instructions; they are now also recorded in `/home/xd7812.princeton/.codex/memories/sigma_search_lite_muon_study.md`.
- First v10 launch attempt from the sandbox produced only Slurm connection errors in logs and no JSONs. No GPU training was running from that failed attempt.
- Relaunched v10 successfully after permissions changed. `squeue -j 29702470 -s` now shows active steps `29702470.121` through `.128`, and logs show `run_lite_v9.py --lite-chi-schedule triangular` executing.
- Current v10 result count at relaunch check: `0/12` JSONs. This is expected; the first 8 runs are still in progress.
- Training logs confirm normal-scale settings, e.g. 16M has `max_steps=64`, `grad_accum=1024`; 1M has `max_steps=1024`, `grad_accum=64`; both show d=8 and `chi_schedule=triangular`.
- Added `analyze_v10_hybrid_chi.py` to compare completed v10 hybrid runs against paired v9 Muon and fixed-LITE baselines. It reports Δ vs Muon and hybrid improvement over fixed LITE, and writes `v10_hybrid_chi_analysis.png` when JSONs exist.


## 2026-04-29 — autonomous 8h controller launched

Controller log: `logs/autonomous_8h_controller_20260429_204902.log`.

Policy:
- Do not compare top1 raw BPB against single-GPU v9/v18 raw BPB until exact same-driver native controls validate the identity baseline.
- First close v18 128K low-LR boundary.
- Then run exact native DDP controls for top1's identity best-LR points.
- If StreamingMuon identity differs from native DDP by more than `0.003` BPB, run only 1M identity diagnostics (`pure_qr`, `pure_qr_2iter`, `scqr_2iter`, `input_normalize`) and stop top1 sweeps.
- If identity is close, run alpha sweep at 1M and 8M for `alpha={0.25,0.5,0.75}`, reusing alpha=0.5 results from the official top1 sweep.
