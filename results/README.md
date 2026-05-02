# Experiment Results

This directory stores curated, machine-readable result summaries that should travel with the clean codebase. Runtime output directories such as `search_evals/`, `logs/`, and `wandb/` remain ignored by default.

Conventions:

- Put each experiment family in its own directory.
- Use `summary.json` for compact records suitable for tables and plots.
- Keep large dense metric `result.json` files out of Git unless explicitly needed.
- Include enough provenance to recover the raw result: source path, runner, batch size, LR, seed, token/step budget, and status.
- Rebuild the consolidated catalog with `python scripts/build_sweep_catalog.py`.

Important subdirectories:

- `sweep_catalog/`: normalized CSV/JSON catalog of existing sweeps.
- `recipes/`: canonical recipes, including the d8 0.4B-token metrics grid.
