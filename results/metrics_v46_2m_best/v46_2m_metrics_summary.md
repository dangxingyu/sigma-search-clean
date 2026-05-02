# v46 2M Metrics Summary

Run: d8, 0.4B tokens, batch size 2,097,152, StreamingMuon slow QR, metrics every step, Hessian every 48 steps, Hessian top-k=4. Lower BPB is better.

| case | lr | val BPB | metric logs | Hessian steps | JSON size |
|---|---:|---:|---:|---|---:|
| identity c=1 | 0.08 | 1.067766 | 192 | [0, 48, 96, 144] | 12.1 MB |
| top-aware c=0.5 | 0.08 | 1.083390 | 192 | [0, 48, 96, 144] | 12.1 MB |

## Dynamics Aggregates

| case | sharpness probes | proj-norm mean | proj-norm last10 | proj-corr mean | proj-corr last10 |
|---|---|---:|---:|---:|---:|
| identity c=1 | 0.060, 0.121, 0.087, 0.226 | 0.01082 | 0.00346 | -0.852 | -0.973 |
| top-aware c=0.5 | 0.060, 0.182, 0.158, 0.219 | 0.01725 | 0.00162 | -0.859 | -0.686 |

## Initial Read

- The 2M best-LR dynamics run confirms the optimizer-quality result: identity beats Top-Aware c=0.5 at the shared best LR `0.08`.
- Compared with v43 4M, 2M has weaker evidence for top-direction damping; this supports treating the transition region as unsettled rather than monotone.
