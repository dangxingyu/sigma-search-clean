# v52-v55 1M Seed Confirmation

Batch `1,048,576`, d8 / 0.4B tokens. For each seed, identity and Top-Aware choose best available LR from the closed grid. Delta is `c=0.5 - identity`; lower is better.

| seed | identity best | c=0.5 best | delta | winner |
|---:|---:|---:|---:|---|
| 42 | 1.002451 @ 0.08 | 1.009856 @ 0.08 | +0.007405 | identity |
| 43 | 1.004451 @ 0.08 | 1.002947 @ 0.08 | -0.001504 | c=0.5 |
| 44 | 1.007175 @ 0.08 | 1.006822 @ 0.08 | -0.000353 | c=0.5 |

Mean delta: `+0.001850 ± 0.002798` SEM over `3` seeds.
Win count: identity `1`, c=0.5 `2`.

Interpretation: 1M is a near-tie transition point. The initial seed42 c=0.5 edge disappeared after identity high-LR closure; current mean leans identity but is not robust.
