# d12/d16 Base Optimizer Sweep Import

This directory contains the imported d12/d16 baseline optimizer sweep and plots.
It is useful for bookkeeping and visualization, but **not** for optimizer-quality
claims about SOAP/Shampoo/KL-SOAP.

Reason: after importing these rows, the structured optimizer implementation was
audited against reference SOAP and KellerJordan/modded-nanogpt PR 290. The old
run used stale implementation/configuration details, including incorrect SOAP
momentum coordinates and non-reference beta/frequency settings.

Curated files:

- `all_rows.csv`: all parsed result JSON rows.
- `best_by_depth_batch_optimizer.csv`: best LR per `(depth, batch, optimizer)`.
- `best_by_depth_batch.csv`: best optimizer per `(depth, batch)` in the stale import.
- `validation_curves.csv`: validation BPB time series.
- `missing_grid.csv`: one missing point, `d16 / kl_shampoo / 2097152 / lr=0.03`.
- `figures/`: LR sweeps, heatmaps, validation curves, and deltas.

Status: rerun SOAP/Shampoo/KL variants with the corrected code before
interpreting winners.
