# Handoff Manifest

Standalone curated optimizer-research repo. It vendors source-only `nanochat/`
and omits virtualenvs, raw sweep outputs, logs, datasets, checkpoints, and
`wandb/`.

## Current Runtime

- `run_eval.py`: StreamingMuon candidate train/eval entrypoint.
- `run_top_aware_muon_sweep.py`: batch x alpha x LR sweep runner for the current method set.
- `streaming_muon_torch.py`: StreamingMuon and DDP optimizer implementation.
- `streaming_muon.py`: lightweight/legacy StreamingMuon utilities.
- `metric_logging.py`: opt-in dynamics metrics and Hessian probes.
- `train_telemetry.py`: telemetry helpers used by historical analyses.
- `diagnose_split_batch.py`: split-batch diagnostic utilities retained for analysis context.

## Candidates

- `candidates/top_aware_muon.py`: Top-Aware Muon transform with `top_k` and `alpha`; `alpha=1.0` is the same-candidate identity baseline.
- `candidates/identity.py`: retained identity sanity candidate, `f(sigma)=1`.

Native Muon/LITE launchers and LITE-like streaming transforms were removed from
the clean runtime surface. Historical results are still preserved under
`docs/`, `figures/`, and `results/`.

## Scripts

- `scripts/setup_env.sh`: create the vendored nanochat `uv` environment.
- `scripts/download_climbmix.sh`: download ClimbMix shards.
- `scripts/smoke_run.sh`: tiny sanity run.
- `scripts/run_d12_sweep.sh`: standalone optimizer-quality sweep wrapper; defaults to d12 but supports d8/d16 via `DEPTH` and `CHINCHILLA_MULT`.
- `scripts/run_d12_statistics.sh`: standalone dense metrics/statistics wrapper; defaults to d12 but supports d8/d16 via `DEPTH` and `CHINCHILLA_MULT`.

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

## Tests

- `tests/test_streaming_muon.py`
Run with:

```bash
PYTHONPATH=.:nanochat pytest tests
```
