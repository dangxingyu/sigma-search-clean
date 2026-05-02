# Handoff Manifest

Standalone curated optimizer-research repo. It vendors source-only `nanochat/` and omits virtualenvs, raw sweep outputs, logs, datasets, checkpoints, and `wandb/`.

## Core Runtime

- `run_eval.py`: StreamingMuon candidate train/eval entrypoint.
- `run_top_aware_muon_sweep.py`: reusable batch x alpha x LR sweep runner.
- `run_native_muon_v9.py`: native Muon baseline.
- `run_lite_v9.py`: native LITE baseline; single-process only.
- `streaming_muon_torch.py`: StreamingMuon and DDP optimizer implementation.
- `streaming_muon.py`: non-torch/legacy StreamingMuon utilities.
- `muon_lite.py`: native-style LITE optimizer.
- `spectra_optim.py`: additional optimizer utilities.
- `metric_logging.py`: opt-in dynamics metrics and Hessian probes.
- `diagnose_split_batch.py`: split-batch SVD diagnostic utilities.
- `train_telemetry.py`: telemetry helpers used by historical analyses.

## Candidates

- `candidates/identity.py`: StreamingMuon identity baseline.
- `candidates/lite_chi2_rs01.py`: same-driver LITE-like transform.
- `candidates/top_aware_muon.py`: current Top-Aware Muon transform with `top_k` and `alpha`.

Legacy `top1_damp_*` candidate files were removed from the clean repo; use `top_aware_muon.py` for all future alpha/top-k sweeps.

## Scripts

- `scripts/setup_env.sh`: create the vendored nanochat `uv` environment.
- `scripts/download_climbmix.sh`: download ClimbMix shards.
- `scripts/smoke_run.sh`: tiny sanity run.
- `scripts/run_d8_metrics_grid.sh`: canonical no-tuning d8 dynamics grid, alpha `{0.5, 1.0}` at batches `{262K, 1M, 4M}`.
- `scripts/build_sweep_catalog.py`: rebuild `results/sweep_catalog/` from curated CSV/JSON summaries.

One-off historical `run_vXX...` launchers were intentionally removed. Their outcomes are preserved in `docs/`, `figures/`, and `results/`.

## Analysis And Results

- `analysis/`: reusable plot/table scripts retained for historical dashboards.
- `figures/`: copied summary plots and raw CSVs used by the catalog builder.
- `results/`: curated machine-readable summaries and recipes.
- `results/sweep_catalog/`: generated consolidated sweep index.
- `results/recipes/`: canonical experiment recipes.

## Documentation

- `README.md`: concise standalone usage and recipe guide.
- `docs/AGENTS.md`: contributor guide.
- `docs/experiment-plan.md`: active research plan.
- `docs/experiment-log.md`: latest-first research log.
- `docs/new-thoughts.md`: conclusion ledger.

## Tests

- `tests/test_streaming_muon.py`
- `tests/test_muon_lite.py`
- `tests/test_controlled_comparison.py`

Run with:

```bash
PYTHONPATH=.:nanochat pytest tests
```
