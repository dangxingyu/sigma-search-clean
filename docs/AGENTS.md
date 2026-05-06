# Repository Guidelines

## Project Structure & Module Organization

This clean repo studies optimizer baselines and StreamingMuon spectral transforms on top of an in-tree `nanochat/` checkout. Current runtime code is `run_eval.py`, `run_optimizer_sweep.py`, `baseline_optim.py`, `streaming_muon_torch.py`, `streaming_muon.py`, and `metric_logging.py`. Candidate transforms live in `candidates/` and must define `def f(sigma, state)`. The supported candidates are `identity.py` and `top_aware_muon.py`. User-facing launchers are in `scripts/`; generated outputs belong under `search_evals/`, `logs/`, `wandb/`, and checkpoint directories, which should stay untracked. Curated summaries and plots live under `results/` and `figures/`.

## Build, Test, And Development Commands

Set up the vendored nanochat environment:

```bash
bash scripts/setup_env.sh
source nanochat/.venv/bin/activate
export PYTHONPATH="$PWD:$PWD/nanochat"
bash scripts/download_climbmix.sh 170 8
```

Common checks:

```bash
python -m py_compile run_eval.py run_optimizer_sweep.py run_top_aware_muon_sweep.py baseline_optim.py metric_logging.py
PYTHONPATH=.:nanochat pytest tests
DRY_RUN=1 bash scripts/run_d12_sweep.sh
bash scripts/smoke_run.sh
```

Main experiment entrypoints:

```bash
STAMP=d12_main_001 bash scripts/run_d12_sweep.sh
BATCHES="262144" LRS="0.02" bash scripts/run_d12_statistics.sh
```

## Coding Style & Naming Conventions

Use Python 3.10+ and PyTorch-first implementations. Keep CLI scripts executable from repo root and prefer explicit environment variables in shell wrappers. Candidate files should be lowercase snake_case and side-effect light. Use descriptive result names that encode method, batch, LR, alpha, top-k, and seed.

## Testing Guidelines

Keep default tests CPU-compatible and fast. Do not add heavy training or SLURM jobs to pytest. For optimizer math changes, test both direct tensor behavior and candidate integration. For training changes, record `depth`, `tokens`, `batch`, `lr`, `seed`, hardware, and final val BPB in `result.json` or the relevant report.

## Commit & Pull Request Guidelines

Use short imperative subjects such as `Simplify StreamingMuon sweep runner` or experiment prefixes when appropriate. PRs should include the hypothesis, commands run, validation results, and any generated artifacts intentionally added. Do not commit secrets, local venvs, raw data, checkpoints, `wandb/`, or one-off scheduler scripts with site-specific absolute paths.
