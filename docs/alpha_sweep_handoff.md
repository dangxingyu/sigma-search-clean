# Alpha Sweep Handoff: d12/d16

This repo's active optimizer is `top_aware_muon`. Its coefficient is called
`alpha` in code and is the same as `c` in the experiment notes:

- `alpha=1.0`: identity StreamingMuon baseline under the same candidate path.
- `alpha<1.0`: damp top-k spectral directions; current main setting is `0.5`.
- `alpha>1.0`: amplify the top direction; use only as a near-identity ablation.

The sweep keeps `top_k=1` fixed unless explicitly studying top-k.

## Recommended Full Alpha Grid

Run inside an existing 8-GPU allocation:

```bash
STAMP_PREFIX=alpha_2x_001 \
bash scripts/run_d12_d16_alpha_sweep.sh
```

Defaults:

| knob | value |
|---|---|
| depths | `12 16` |
| token budget | `CHINCHILLA_MULT=2` |
| batches | `524288 2097152 8388608` |
| alphas | `0.25 0.5 0.75 0.85 1.0 1.15` |
| LRs | `0.005 0.0075 0.01 0.015 0.02 0.03 0.04` |
| seeds | `42` |
| adaptive LR | enabled, with closure in `[0.0005, 0.16]` |
| metrics | off |

Outputs are separated by depth:

```text
search_evals/alpha_2x_001_d12_2x_alpha/
search_evals/alpha_2x_001_d16_2x_alpha/
logs/alpha_2x_001_d12_2x_alpha/
logs/alpha_2x_001_d16_2x_alpha/
```

## First Smoke / Dry Run

Before launching real training:

```bash
DRY_RUN=1 STAMP_PREFIX=alpha_dry bash scripts/run_d12_d16_alpha_sweep.sh
```

For a tiny real path check:

```bash
DEPTHS="12" \
TOKENS=16777216 \
BATCHES="524288" \
ALPHAS="1.0 0.5" \
LRS="0.005" \
ADAPTIVE_LR=0 \
STAMP_PREFIX=alpha_smoke \
bash scripts/run_d12_d16_alpha_sweep.sh
```

## SLURM Array Version

If the cluster supports preemptible arrays, submit d12 and d16 separately.
This is safer than one long sequential job because every case has its own
checkpoint directory and can resume.

```bash
DEPTH=12 \
CHINCHILLA_MULT=2 \
BATCHES="524288 2097152 8388608" \
ALPHAS="0.25 0.5 0.75 0.85 1.0 1.15" \
LRS="0.005 0.0075 0.01 0.015 0.02 0.03 0.04" \
SEEDS="42" \
STAMP=alpha_d12_2x_001 \
MAX_PARALLEL=8 \
SBATCH_TIME=24:00:00 \
bash scripts/submit_slurm_grid.sh
```

Then submit d16:

```bash
DEPTH=16 \
CHINCHILLA_MULT=2 \
BATCHES="524288 2097152 8388608" \
ALPHAS="0.25 0.5 0.75 0.85 1.0 1.15" \
LRS="0.005 0.0075 0.01 0.015 0.02 0.03 0.04" \
SEEDS="42" \
STAMP=alpha_d16_2x_001 \
MAX_PARALLEL=8 \
SBATCH_TIME=24:00:00 \
bash scripts/submit_slurm_grid.sh
```

After each array finishes, first run a pure collation command. This only reads
completed `result.json` files and writes `sweep_rows.csv`; it does not need
checkpoints and does not launch training:

```bash
DEPTH=12 CHINCHILLA_MULT=2 \
BATCHES="524288 2097152 8388608" \
ALPHAS="0.25 0.5 0.75 0.85 1.0 1.15" \
LRS="0.005 0.0075 0.01 0.015 0.02 0.03 0.04" \
SEEDS="42" \
STAMP=alpha_d12_2x_001 \
SUMMARY_ONLY=1 \
bash scripts/run_d12_sweep.sh

DEPTH=16 CHINCHILLA_MULT=2 \
BATCHES="524288 2097152 8388608" \
ALPHAS="0.25 0.5 0.75 0.85 1.0 1.15" \
LRS="0.005 0.0075 0.01 0.015 0.02 0.03 0.04" \
SEEDS="42" \
STAMP=alpha_d16_2x_001 \
SUMMARY_ONLY=1 \
bash scripts/run_d12_sweep.sh
```

If you also want adaptive LR boundary closure, run the same command without
`SUMMARY_ONLY=1` and with `ADAPTIVE_LR=1`. That step may launch new training
cases at boundary LRs; skip it if you want full control over job submission.

Re-running the exact same command is preemption-safe: completed `result.json`
files are skipped, and incomplete cases resume from checkpoints.

## Smaller Pilot Grid

If compute is tight, start with:

```bash
DEPTHS="12 16" \
BATCHES="2097152 8388608" \
ALPHAS="0.5 0.75 1.0 1.15" \
LRS="0.0075 0.01 0.015 0.02 0.03" \
STAMP_PREFIX=alpha_pilot_001 \
bash scripts/run_d12_d16_alpha_sweep.sh
```

Interpretation rule: choose the best LR per `(depth, batch, alpha)` before
comparing alphas. Do not compare a non-best LR row for one alpha to the best LR
row of another alpha.

## Files To Send Back

For each `STAMP`, send:

```text
search_evals/<STAMP>/manifest.json
search_evals/<STAMP>/sweep_rows.csv
search_evals/<STAMP>/adaptive_lr_trace.json       if present
search_evals/<STAMP>/*/result.json
logs/<STAMP>/*.log                                optional, useful for failures
```

If transferring as one archive:

```bash
tar -czf alpha_d12_2x_001_results.tar.gz \
  search_evals/alpha_d12_2x_001 \
  logs/alpha_d12_2x_001
```
