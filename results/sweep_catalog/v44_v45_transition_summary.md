# v42/v44/v45/v47 Transition Sweep Summary

Clean d8 / 0.4B-token StreamingMuon sweep, same driver, seed 42. Lower BPB is better. `c=0.5` is Top-Aware Muon with `top_k=1, alpha=0.5`.

| batch | identity best | c=0.5 best | c=0.5 - identity | winner | LR closed? |
|---:|---:|---:|---:|---|---|
| 262144 | 0.967732 @ 0.04 | 0.969135 @ 0.02 | +0.001403 | identity | yes |
| 1048576 | 1.002451 @ 0.08 | 1.009856 @ 0.08 | +0.007405 | identity | yes |
| 2097152 | 1.068525 @ 0.08 | 1.082541 @ 0.08 | +0.014016 | identity | yes |
| 4194304 | 1.206814 @ 0.04 | 1.174488 @ 0.02 | -0.032326 | c=0.5 | yes |
| 8388608 | 1.456445 @ 0.02 | 1.421837 @ 0.02 | -0.034608 | c=0.5 | yes |

Notes:
- 1M Top-Aware was explicitly extended to `lr=0.16`; it got worse than `0.08`, so the 1M high-LR side is closed.
- 2M was explicitly extended to `lr=0.16`; both methods got worse, so the `0.08` optima are closed on the high-LR side.
- The one-seed pattern is not perfectly monotone: after identity high-LR closure, 1M seed42 favors identity, 2M seed42 favors identity, while 4M/8M strongly favor c=0.5.
- Follow-up 1M seeds `{42,43,44}` at closed LR grids show a near-tie: mean `c=0.5 - identity = +0.001850 ± 0.002798` SEM, with identity winning 1 seed and c=0.5 winning 2 seeds.
- Follow-up 2M seeds `{42,43,44,45,46}` at `lr=0.08` show a tie/noisy transition: mean `c=0.5 - identity = +0.000507 ± 0.004183` SEM, with identity winning 3 seeds and c=0.5 winning 2 seeds.
- Follow-up 4M seeds `{42,43,44}` show robust c=0.5 wins: mean `c=0.5 - identity = -0.025240 ± 0.004110` SEM.
