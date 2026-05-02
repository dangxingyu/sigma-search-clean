# v48 8M Metrics Summary

Run: d8, 0.4B tokens, batch size 8,388,608, StreamingMuon slow QR, metrics every step, Hessian every 12 steps, Hessian top-k=4. Lower BPB is better.

| case | lr | val BPB | metric logs | Hessian steps | JSON size |
|---|---:|---:|---:|---|---:|
| identity c=1 | 0.02 | 1.450463 | 48 | [0, 12, 24, 36] | 4.2 MB |
| top-aware c=0.5 | 0.02 | 1.428626 | 48 | [0, 12, 24, 36] | 4.2 MB |

## Dynamics Aggregates

| case | sharpness probes | proj-norm mean | proj-norm last10 | proj-corr mean | proj-corr last10 |
|---|---|---:|---:|---:|---:|
| identity c=1 | 0.161, 1.449, 0.590, 1.046 | 0.08623 | 0.05823 | -0.787 | -0.981 |
| top-aware c=0.5 | 0.160, 1.439, 1.071, 1.714 | 0.08624 | 0.05755 | -0.540 | -0.901 |

## Initial Read

- The 8M best-LR dynamics run confirms the optimizer-quality result: Top-Aware c=0.5 beats identity at the shared best LR `0.02`.
- This gives dynamics coverage on both sides of the 2M/4M transition: v46 covers a 2M identity-win point; v43/v48 cover 4M/8M Top-Aware-win points.
