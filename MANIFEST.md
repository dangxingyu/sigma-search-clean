# Handoff Manifest

This is a standalone curated subset of `sigma-search` for sharing optimizer experiment code. It vendors source-only `nanochat/` so the code does not need the parent repository. Large raw sweeps, checkpoints, logs, datasets, virtualenvs, and `wandb/` are intentionally omitted.

## Current Runnable Core

- `streaming_muon_torch.py`: current StreamingMuon implementation with pure QR, SCQR fallback, candidate transforms, and DDP optimizer wrapper.
- `run_eval.py`: main same-driver StreamingMuon training/eval entrypoint.
- `metric_logging.py`: opt-in Muon-style diagnostic logger for `run_eval.py` and `run_native_muon_v9.py`, including norms, momentum spectra, split-momentum SVD alignment, optional component dumps, and best-effort Hessian probes.
- `nanochat/`: vendored nanochat source, including `pyproject.toml`, `uv.lock`, model, dataloader, tokenizer, and dataset downloader. Excludes `.venv`, `wandb`, and `__pycache__`.
- `candidates/identity.py`: Muon-like identity transform.
- `candidates/lite_chi2_rs01.py`: LITE-like transform used in the fair v26 comparison.
- `candidates/top1_damp_alpha05_per_matrix.py`: current per-matrix top1 damp transform.
- `candidates/top_aware_muon.py`: Top-Aware Muon transform. Current clean recipe fixes `top_k=1` and sweeps `alpha`.
- `run_v26_small_bsz_streaming_fair.py`: adaptive same-driver sweep orchestrator.
- `run_v26_small_bsz_streaming_fair_on_node.sh`: portable v26 launcher.
- `run_v26b_large_then_maybe_v27_on_node.sh`: waits for v26, runs large-batch v26b, then gates scale-up.
- `run_v27_d12_3b_streaming_fair_on_node.sh`: prepared d12 / 3B scale-up launcher.
- `run_top_aware_muon_sweep.py`: current pass-by sweep runner for `batch x alpha x lr`, with `top_k=1`.
- `run_top_aware_muon_sweep_on_node.sh`: portable launcher for the Top-Aware sweep runner.
- `run_v28_v29_topaware_k1_alpha_sweeps_on_node.sh`: staged clean alpha-sweep launcher for 128K alpha=.75 and 256K alpha grid.
- `run_v31_256k_512k_alpha_sweeps_after_v29_on_node.sh`: queued overnight launcher that fills/verifies 256K and then runs 512K Top-Aware alpha grid with native Muon baseline.
- `run_v30_dynamics_metrics_after_v29_on_node.sh`: waits for v29, then runs dynamics metric logging for native Muon, StreamingMuon identity, and Top-Aware Muon at 128K/256K/512K/1M/2M.
- `scripts/setup_env.sh`: creates the vendored nanochat `uv` environment.
- `scripts/download_climbmix.sh`: downloads ClimbMix-400B parquet shards into `$NANOCHAT_BASE_DIR/base_data_climbmix`.
- `scripts/smoke_run.sh`: tiny local sanity run that also verifies metric logging output.

## Native Muon / LITE Baselines

- `muon_lite.py`: native-style Muon-LITE implementation.
- `run_native_muon_v9.py`: native Muon baseline runner.
- `run_lite_v9.py`: native LITE baseline runner.
- `run_v18_128k_clean_on_node.sh` and `run_v18b_128k_low_lr_extension_on_node.sh`: historical 128K native Muon/LITE sweeps.
- `run_v13_256k_clean.sh` and `run_v9_momentum_d1.sh`: older baseline/diagnostic launchers.

## Top1 / Sigma-Transform Infrastructure

- `run_top1_damp_adaptive_lr_sweep.py`: adaptive LR sweep for identity vs top1 damping.
- `run_ddp8_top1_damp_adaptive_lr_sweep_on_node.sh`: portable top1 sweep launcher.
- `run_v21_exact_native_controls_for_top1_on_node.sh`: historical exact native DDP controls.
- `run_v23_streaming_identity_diagnostics_on_node.sh`: historical StreamingMuon identity diagnostics.
- `candidates/top1_damp_alpha*.py`: older alpha variants kept for follow-up sweeps.

Top-Aware Muon is the current name for the top1 direction rule. The old `top1_damp_*` files are historical variants. The candidate keeps a `top_k` parameter for explicit ablations, but the clean handoff recipe fixes `top_k=1`.

## Analysis And Figures

- `make_v26_dashboard.py`: current same-driver dashboard generator.
- `make_historical_large_batch_dashboard.py`: historical large-batch dashboard generator.
- `make_topaware_alpha_dashboard.py`: alpha=.75 / alpha-grid dashboard generator.
- `make_v30_dynamics_dashboard.py`: metric-log dynamics dashboard generator.
- `decide_scaleup_gate.py`: scale-up gate based on v26/v26b ordering.
- `analyze_*.py`: supporting analysis scripts for previous runs.
- `figures/current_same_driver_v26/`: current same-driver partial table and LR sweep plot.
- `figures/historical_large_batch/`: historical native Muon/LITE and old Streaming top1 summaries.
- `figures/supporting_plots/`: auxiliary LR-sweep and diagnostic plots.
- `figures/topaware_alpha_dashboard/`: 128K alpha=.75 and partial 256K alpha-grid Top-Aware dashboard.
- `results/`: curated machine-readable result summaries. Runtime raw outputs remain in `search_evals/` and are intentionally ignored unless explicitly copied into `results/raw/`.

## Documentation

- `README.md`: usage guide for this handoff bundle.
- `docs/AGENTS.md`: contributor guide copied from the working repo.
- `docs/experiment-plan.md`, `docs/experiment-log.md`, `docs/new-thoughts.md`: current research notes at copy time.

## Tests

- `tests/test_streaming_muon.py`
- `tests/test_muon_lite.py`
- `tests/test_controlled_comparison.py`

Run with `PYTHONPATH=.:nanochat pytest tests` after providing or symlinking `nanochat/`.
