# Handoff Manifest

Standalone curated optimizer-research repo. It vendors source-only `nanochat/`
and omits virtualenvs, raw sweep outputs, logs, datasets, checkpoints, and
`wandb/`.

## Current Runtime

- `run_eval.py`: train/eval entrypoint for StreamingMuon, plain Muon, nanochat Muon/NormMuon, AdamW, SOAP, Shampoo, KL-Shampoo, and KL-SOAP.
- `run_optimizer_sweep.py`: batch x optimizer x alpha x LR sweep runner for the current method set.
- `run_top_aware_muon_sweep.py`: historical filename kept as the implementation module for compatibility.
- `baseline_optim.py`: replicated AdamW/plain-Muon/SOAP/Shampoo/KL-Shampoo/KL-SOAP optimizer baselines.
- `streaming_muon_torch.py`: StreamingMuon and DDP optimizer implementation.
- `streaming_muon.py`: lightweight/legacy StreamingMuon utilities.
- `metric_logging.py`: opt-in dynamics metrics and Hessian probes.
- `train_telemetry.py`: telemetry helpers used by historical analyses.
- `diagnose_split_batch.py`: split-batch diagnostic utilities retained for analysis context.

## Candidates

- `candidates/top_aware_muon.py`: Top-Aware Muon transform with `top_k` and `alpha`; `alpha=1.0` is the same-candidate identity baseline.
- `candidates/identity.py`: retained identity sanity candidate, `f(sigma)=1`.

LITE launchers and LITE-like streaming transforms were removed from the clean
runtime surface. Plain Muon is available through
`run_eval.py --optimizer plain_muon`. Nanochat's Muon/NormMuon remains available
through `run_eval.py --optimizer muon` (`native_muon` is retained as an alias).
Historical results are still preserved
under `docs/`, `figures/`, and `results/`.

## Scripts

- `scripts/setup_env.sh`: create the vendored nanochat `uv` environment.
- `scripts/download_climbmix.sh`: download ClimbMix shards.
- `scripts/smoke_run.sh`: tiny sanity run.
- `scripts/run_d12_sweep.sh`: standalone optimizer-quality sweep wrapper; defaults to d12 but supports d8/d16 via `DEPTH` and `CHINCHILLA_MULT`.
- `scripts/run_optimizer_baselines.sh`: sequential multi-optimizer baseline sweep wrapper.
- `scripts/run_d12_optimizer_baselines.sh`: d12 apple-to-apple non-streaming baseline grid.
- `scripts/run_d16_optimizer_baselines.sh`: d16 apple-to-apple non-streaming baseline grid.
- `scripts/run_d12_d16_optimizer_baselines.sh`: convenience wrapper that runs the d12 and d16 baseline scripts sequentially.
- `scripts/run_d12_d16_2x_grid.sh`: canonical sequential handoff grid for d12/d16, 2x Chinchilla, batches `{512K,2M,8M}`, alphas `{1.0,0.5}`.
- `scripts/run_d12_d16_alpha_sweep.sh`: sequential d12/d16 alpha sweep for Top-Aware Muon, default alphas `{0.25,0.5,0.75,0.85,1.0,1.15}`.
- `scripts/submit_slurm_grid.sh`: submit fixed-grid cases as a preemption-safe SLURM array.
- `scripts/run_d12_statistics.sh`: standalone dense metrics/statistics wrapper; defaults to d12 but supports d8/d16 via `DEPTH` and `CHINCHILLA_MULT`.
- `scripts/run_d12_metrics_best.sh`: curated d12 2x Chinchilla metrics wrapper for the six best `(batch, alpha, LR)` points from the completed sweep.

## Analysis And Results

- `analysis/`: reusable plot/table scripts retained for dashboards.
- `analysis/build_sweep_catalog.py`: rebuild historical `results/sweep_catalog/`.
- `figures/`: copied summary plots and raw CSVs used by the catalog builder.
- `results/`: curated machine-readable summaries and recipes.
- `results/sweep_catalog/`: generated consolidated sweep index.
- `results/recipes/`: canonical experiment recipes.

## Documentation

- `README.md`: standalone usage and recipe guide.
- `docs/AGENTS.md`: contributor guide.
- `docs/experiment-plan.md`: active research plan.
- `docs/experiment-log.md`: latest-first research log.
- `docs/new-thoughts.md`: conclusion ledger.
- `docs/alpha_sweep_handoff.md`: copy-paste d12/d16 alpha-sweep instructions for external runs or agents.

## Tests

- `tests/test_streaming_muon.py`
- `tests/test_metric_logging.py`
- `tests/test_baseline_optim.py`
Run with:

```bash
PYTHONPATH=.:nanochat pytest tests
```
