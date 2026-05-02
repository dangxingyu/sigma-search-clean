# New Figures And Tables

This directory collects the current optimizer-study tables and figures in one place.

## Current Same-Driver v26

Directory: `current_same_driver_v26/`

These are the fair same-driver StreamingMuon comparisons for `identity`, `lite_chi2_rs01`, and `top1pm_a05`. All runs use the same DDP `run_eval.py` driver with pure QR and two streaming iterations per step.

Files:
- `dashboard_table.md`: best rows, deltas, and all completed `batch x LR x optimizer` rows.
- `lr_sweep.png`: LR sweep plot for completed v26 rows.
- `raw_rows.csv`: one row per completed run.
- `aggregate_rows.csv`: aggregated rows by batch, optimizer, and LR.
- `summary.json`: machine-readable dashboard summary.

Status: this is a snapshot while v26 is still running. Rebuild the source dashboard with:

```bash
MPLCONFIGDIR=/tmp/matplotlib-sigma python make_v26_dashboard.py
```

## Historical Large Batch

Directory: `historical_large_batch/`

These are previous large-batch results split by experiment family. Do not raw-compare BPB across families because the native single-GPU and old StreamingMuon DDP runs use different data/eval streams.

Files:
- `dashboard_table.md`: native Muon/LITE best rows and old StreamingMuon identity/top1 best rows.
- `native_muon_lite.png`: native single-GPU Muon vs LITE best-LR comparison.
- `streaming_top1_lr_sweep.png`: old StreamingMuon identity vs top1 LR sweep.
- `raw_rows.csv`: one row per historical run.
- `aggregate_rows.csv`: aggregated rows by family, batch, optimizer, and LR.
- `summary.json`: machine-readable dashboard summary.

Source rebuild command:

```bash
MPLCONFIGDIR=/tmp/matplotlib-sigma python make_historical_large_batch_dashboard.py
```

## Native / Streaming / Top-Aware

Directory: `native_streaming_topk/`

This combines the available Native Muon exact-control rows, StreamingMuon identity LR sweeps, and Top-Aware `top_k=1, alpha=0.5` rows. It is the clearest current view of the native-vs-streaming-vs-top-k question, but native Muon is only available as exact-control single-LR DDP points at 128K, 1M, and 8M.

Files:
- `native_streaming_topk_overview.png`: absolute BPB plus delta-to-StreamingMuon-identity plot.
- `native_streaming_topk_best_table.md`: compact comparison table.
- `native_streaming_topk_best.csv`: best available row per method and batch.
- `native_streaming_topk_raw.csv`: all source rows used by the dashboard.

Source rebuild command:

```bash
MPLCONFIGDIR=/tmp/matplotlib-sigma python make_native_streaming_topk_dashboard.py
```

## Top-Aware Alpha Dashboard

Directory: `topaware_alpha_dashboard/`

This combines the completed 128K `top_k=1, alpha=0.75` sweep with the currently completed 256K alpha-grid rows.

Files:
- `lr_sweeps.png`: LR sweep curves by batch, method, and alpha.
- `best_delta_vs_identity.png`: best-row delta relative to StreamingMuon identity when paired data exists.
- `best_table.md`: compact best-row table.
- `raw_rows.csv`: one row per parsed source run.
- `best_rows.csv`: best available row per batch/method/alpha.

Source rebuild command:

```bash
MPLCONFIGDIR=/tmp/matplotlib-sigma python make_topaware_alpha_dashboard.py
```

## Supporting Plots

Directory: `supporting_plots/`

These are auxiliary plots used during the analysis:
- `optimizer_lr_sweeps.png`
- `optimizer_lr_sweeps_top1_8m.png`
- `v18_128k_clean_analysis.png`
- `top1_damp_lr_sweep_analysis.png`
- `top1_damp_lr_sweep_trajectories.png`
