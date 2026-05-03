# d12/d16 Sweep Summary

Negative delta means `c=0.5` beats `c=1` after LR selection.

Completeness check: both d12 and d16 sweeps have 42/42 result JSONs and 0
recorded errors. The d16 CSV had empty `alpha/top_k` columns, so the summary
parses those fields from case names.

| depth | batch | c=1 best BPB @ LR | c=0.5 best BPB @ LR | delta c=0.5-c=1 | winner |
|---|---:|---:|---:|---:|---|
| d12 | 512K | 0.814978 @ 0.0075 | 0.816066 @ 0.0075 | +0.001088 | c=1 |
| d12 | 2M | 0.834533 @ 0.015 | 0.833789 @ 0.02 | -0.000745 | c=0.5 |
| d12 | 8M | 0.890333 @ 0.01 | 0.887032 @ 0.015 | -0.003300 | c=0.5 |
| d16 | 512K | 0.750007 @ 0.005 | 0.750464 @ 0.005 | +0.000456 | c=1 |
| d16 | 2M | 0.752986 @ 0.005 | 0.753916 @ 0.0075 | +0.000929 | c=1 |
| d16 | 8M | 0.793761 @ 0.0075 | 0.796430 @ 0.015 | +0.002670 | c=1 |

Interpretation:

- d12 shows the expected transition: `c=1` wins at 512K, `c=0.5` wins mildly at
  2M, and `c=0.5` wins more clearly at 8M.
- d16 currently favors `c=1` at all three batch sizes.
- d16 is not fully closed at 512K/2M because the `c=1` best LR is the lowest
  swept LR (`0.005`), and 512K `c=0.5` is also at the lower boundary. Run a
  lower-LR extension before treating the 512K/2M d16 conclusion as final.
- d16 8M is more reliable because neither method's best LR is at the lower
  boundary.

Recommended d16 lower-LR extension:

```bash
DEPTH=16 \
CHINCHILLA_MULT=2 \
BATCHES="524288 2097152" \
ALPHAS="1.0 0.5" \
LRS="0.00125 0.0025 0.00375" \
ADAPTIVE_LR=1 \
STAMP=d16_lower_lr_extension \
bash scripts/run_d12_sweep.sh
```

Figures:

- `figures/d12_d16_lr_sweeps.png`
- `figures/d12_d16_best_bpb_by_batch.png`
- `figures/d12_d16_delta_c05_minus_c1.png`
