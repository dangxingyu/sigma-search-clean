# Metrics Cross-Batch Summary

Best available dense-metrics comparison for each batch. Lower BPB is better; negative delta means Top-Aware c=0.5 wins.

Dynamics plot: [hessian_dynamics_overview.md](hessian_dynamics_overview.md) embeds the cross-batch sharpness, gradient-Hessian alignment, and fixed-Hessian projection-correlation figure.

| batch | identity | c=0.5 | delta | winner | identity sharp last | c=0.5 sharp last | identity corr mean | c=0.5 corr mean |
|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 262144 | 0.967446 @ 0.04 | 0.969429 @ 0.02 | +0.001983 | identity | 3.945 | 24.454 | -0.835 | -0.817 |
| 1048576 | 1.002178 @ 0.08 | 1.010222 @ 0.08 | +0.008044 | identity | 0.312 | 0.871 | -0.789 | -0.849 |
| 2097152 | 1.067766 @ 0.08 | 1.083390 @ 0.08 | +0.015624 | identity | 0.226 | 0.219 | -0.852 | -0.859 |
| 4194304 | 1.206642 @ 0.02 | 1.176648 @ 0.02 | -0.029995 | c=0.5 | 0.833 | 1.510 | -0.784 | -0.779 |
| 8388608 | 1.450463 @ 0.02 | 1.428626 @ 0.02 | -0.021837 | c=0.5 | 1.046 | 1.714 | -0.787 | -0.540 |

Initial read:
- Dense metrics now cover 262K, 1M, 2M, 4M, and 8M at the selected best-LR settings.
- At 262K/1M, Top-Aware c=0.5 has higher measured sharpness and worse BPB than identity.
- Sharpness alone is not a winner predictor: Top-Aware wins at 4M/8M while often having comparable or higher last-probe sharpness.
- Projection-correlation may be informative at 8M, but it does not cleanly explain the full transition.
