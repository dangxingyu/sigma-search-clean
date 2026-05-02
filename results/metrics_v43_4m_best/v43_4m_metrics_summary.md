# v43 4M Metrics Summary

Run: d8, 0.4B tokens, batch size 4,194,304, StreamingMuon slow QR, metrics every step, Hessian every 24 steps, Hessian top-k=4 over all selected transformer matrices. Lower val BPB is better.

## Scores

| case | lr | val BPB | metric logs | Hessian steps | JSON size |
|---|---:|---:|---:|---|---:|
| identity c=1 | 0.02 | 1.206642 | 96 | [0, 24, 48, 72] | 6.8 MB |
| identity c=1 | 0.04 | 1.206870 | 96 | [0, 24, 48, 72] | 6.8 MB |
| top-aware c=0.5 | 0.02 | 1.176648 | 96 | [0, 24, 48, 72] | 6.8 MB |
| top-aware c=0.5 | 0.04 | 1.177314 | 96 | [0, 24, 48, 72] | 6.8 MB |

## Dynamics Aggregates

| case | lr | sharpness probes | proj-norm mean | proj-norm last10 | proj-corr mean | proj-corr last10 |
|---|---:|---|---:|---:|---:|---:|
| identity c=1 | 0.02 | 0.101, 0.335, 0.574, 0.833 | 0.03731 | 0.01982 | -0.784 | -0.978 |
| identity c=1 | 0.04 | 0.100, 0.194, 0.171, 0.220 | 0.01989 | 0.00983 | -0.825 | -0.997 |
| top-aware c=0.5 | 0.02 | 0.101, 0.686, 0.962, 1.510 | 0.04271 | 0.01897 | -0.779 | -0.966 |
| top-aware c=0.5 | 0.04 | 0.100, 0.336, 0.329, 0.510 | 0.02226 | 0.00633 | -0.784 | -0.967 |

## Initial Read

- The metrics path is active: all four runs have 96 per-step metric records and Hessian probes at steps 0/24/48/72.
- The 4M result remains robust under metrics logging: best top-aware c=0.5 is about 0.030 BPB better than best identity in this run.
- Top-aware c=0.5 reaches higher measured sharpness than identity while improving BPB, so the current evidence is not simply "lower sharpness is always better". The useful question is how the top-direction transform changes stability and useful progress in the flat subspace.
