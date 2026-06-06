# The Art of Tuning

This handbook collects practical hyperparameter-tuning moves for optimizer and training-recipe work.

The intended format is a field guide: each chapter names a move, explains when to use it, gives the operating protocol, and records concrete case studies.

## Chapters

1. [Coordinate Descent: The First Move](01-coordinate-descent.md)

## Working Principles

- Tune one clear objective at a time.
- Keep a machine-readable ledger for every sweep.
- Move only one accepted coordinate per round.
- Prefer small, decision-oriented grids over large undifferentiated sweeps.
- Treat dashboarding and result collection as part of the experiment, not afterthoughts.
- Preserve the exact job recipe: code tarball, resource queue, data/eval recipe, and all optimizer hyperparameters.

## Ledger Template

Each tuning campaign should have a directory like:

```text
results/<experiment_name>/
  README.md
  state.json
  round_00_candidates.csv
  round_01_candidates.csv
  visualizations/
```

`state.json` should include:

- current center recipe
- round status
- grid for each coordinate
- best candidate per coordinate
- accepted coordinate
- stop reason
