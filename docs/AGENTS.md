# Repository Guidelines

## Project Structure & Module Organization

This repository searches for spectral transforms for StreamingMuon on top of an in-tree `nanochat/` checkout. Core optimizer code lives in `streaming_muon_torch.py`, `streaming_muon.py`, `muon_lite.py`, and `spectra_optim.py`. Experiment entrypoints are root-level scripts such as `run_eval.py`, `run_streaming_base_train.py`, `run_lite.py`, and `run_native_muon.py`. Candidate transforms belong in `candidates/` and should define `def f(sigma, state)`. Root tests are `test_*.py`; embedded nanochat tests are in `nanochat/tests/`. Generated sweep outputs live under `sweep_results*/`; search runtime artifacts such as `search_pool/`, `search_evals/`, `eval_logs/`, `wandb/`, and `*.log` should stay untracked.

## Build, Test, and Development Commands

Set up dependencies from the embedded nanochat project:

```bash
cd nanochat && uv sync --extra gpu --group dev
source nanochat/.venv/bin/activate
```

Common workflows:

```bash
PYTHONPATH=.:nanochat pytest test_streaming_muon.py test_muon_lite.py nanochat/tests
python search_harness.py seed          # create baseline candidate pool
python search_harness.py summary       # show candidate rankings
python run_eval.py --nanochat-dir ./nanochat --candidate-file candidates/identity.py --output-file result.json --max-steps 500 --depth 4
./submit_eval.sh --candidate-file candidates/identity.py --output-file result.json --max-steps 500 --depth 4 --wait
```

Use `torchrun --standalone --nproc_per_node=N run_streaming_base_train.py ...` for full distributed training runs.

## Coding Style & Naming Conventions

Use Python 3.10+ and PyTorch-first implementations for nanochat integration. Keep root scripts executable from the CLI; `run_eval.py` parses arguments at module import time, so do not import it from tests or libraries. Prefer explicit names for experiment files and outputs, e.g. `run_lrsweep_v2.sh`, `probe_warmup500.json`, and `phase_medium/lite_bsz1048576_lr0.02_s42.json`. Candidate files should be lowercase snake_case and side-effect light.

## Testing Guidelines

Add or update `test_*.py` when changing optimizer math, candidate mechanics, or training wrappers. Run fast CPU-compatible tests before committing; mark or isolate GPU/SLURM-only checks. For training changes, record `max_steps`, `depth`, seed, hardware, and val BPB in the result JSON or report note.

## Commit & Pull Request Guidelines

Recent commits use short, descriptive subjects such as `Fix StreamingMuon SCQR drift; add Muon-LITE and SPECTRA optimizers` or `v3: Modal H100 follow-up sweep...`. Use an imperative fix/add/update subject or an experiment version prefix. PRs should state the hypothesis, changed files or scripts, exact commands run, validation metrics, and any generated artifacts intentionally added. Do not commit secrets, hardcoded WANDB keys, local `.claude/` state, or one-off launcher scripts with absolute paths.
