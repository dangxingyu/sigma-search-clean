# Recipes

Canonical recipes for experiments that should be reproducible from the clean repo.

## D8 Dynamics Metrics Grid

Use `d8_metrics_grid_recipe.json` and `scripts/run_d8_metrics_grid.sh`.

This is a no-hyperparameter-tuning dynamics study:

- Model: d8, seq1024.
- Token budget: `402,653,184` tokens, about `0.4B`.
- Batches: `262144` critical batch, `1048576`, `4194304`.
- Optimizers: Top-Aware Muon with `top_k=1`, `alpha in {0.5, 1.0}`.
- LR: base `matrix_lr=0.02`, scaled by nanochat as `sqrt(batch / 524288)`.
- Metrics: cheap metrics every step; Hessian probes every 50 logged steps.
