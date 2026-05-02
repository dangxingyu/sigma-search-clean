# Metrics Cross-Batch Summary

Best available dense-metrics comparison for each batch. Lower BPB is better; negative delta means Top-Aware c=0.5 wins.

| batch | identity | c=0.5 | delta | winner | identity sharp last | c=0.5 sharp last | identity corr mean | c=0.5 corr mean |
|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 2097152 | 1.067766 @ 0.08 | 1.083390 @ 0.08 | +0.015624 | identity | 0.226 | 0.219 | -0.852 | -0.859 |
| 4194304 | 1.206642 @ 0.02 | 1.176648 @ 0.02 | -0.029995 | c=0.5 | 0.833 | 1.510 | -0.784 | -0.779 |
| 8388608 | 1.450463 @ 0.02 | 1.428626 @ 0.02 | -0.021837 | c=0.5 | 1.046 | 1.714 | -0.787 | -0.540 |

Initial read:
- Dense metrics reproduce the optimizer-quality sign at 2M, 4M, and 8M.
- Sharpness alone is not a winner predictor: Top-Aware wins at 4M/8M while often having comparable or higher last-probe sharpness.
- Projection-correlation may be more informative at 8M, but it does not cleanly explain 2M vs 4M yet.
