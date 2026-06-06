---
name: the-art-of-tuning
description: Plan, run, monitor, and summarize hyperparameter tuning campaigns using The Art of Tuning handbook. Use when tuning optimizers, learning rates, betas, weight decay, batch-size recipes, coordinate descent, sweep dashboards, or iterative experiment ledgers.
---

# The Art of Tuning

Use this skill for optimizer and training-recipe tuning work.

Primary handbook:

```text
docs/the-art-of-tuning/
```

Start with:

```text
docs/the-art-of-tuning/01-coordinate-descent.md
```

## Default Move

When there are several coupled hyperparameters and a reasonable baseline exists, use coordinate descent.

Protocol:

1. Establish the center recipe and baseline score.
2. Choose coordinates.
3. Generate 5 candidates per active coordinate.
4. Launch independent jobs for every candidate.
5. Record every candidate in `results/<experiment>/round_XX_candidates.csv`.
6. Monitor until every candidate has a result or terminal failure.
7. Pick the coordinate with the largest positive improvement.
8. Update only that coordinate.
9. Skip that coordinate in the next round.
10. Stop after no improvement or the round budget.

## Ledger

Every campaign should create:

```text
results/<experiment>/
  README.md
  state.json
  round_00_candidates.csv
  visualizations/
```

Keep `state.json` current with:

- current center recipe
- active coordinates
- grids
- best candidate per coordinate
- accepted coordinate
- stop reason

## Grid Rules

- Positive knobs: use multiplicative/log grids around the center.
- Beta-like knobs: grid in `1 - beta` complement space.
- Keep center in the grid for calibration.
- Extend only the winning/boundary coordinate unless there is a strong interaction hypothesis.

## Reporting

Report:

- starting score and final score
- accepted coordinate path
- final recipe
- best candidate per coordinate
- known caveats, failures, and possible next interaction sweeps
