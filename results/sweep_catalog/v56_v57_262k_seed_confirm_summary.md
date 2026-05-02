# v56-v57 262K Seed Confirmation

Batch `262,144`, d8 / 0.4B tokens. For each seed, identity and Top-Aware choose best available LR from the closed/checked grid. Delta is `c=0.5 - identity`; lower is better.

| seed | identity best | c=0.5 best | delta | winner |
|---:|---:|---:|---:|---|
| 42 | 0.967732 @ 0.04 | 0.969135 @ 0.02 | +0.001403 | identity |
| 43 | 0.967086 @ 0.02 | 0.964778 @ 0.02 | -0.002308 | c=0.5 |
| 44 | 0.968867 @ 0.02 | 0.967932 @ 0.04 | -0.000935 | c=0.5 |

Mean delta: `-0.000613 ± 0.001083` SEM over `3` seeds.
Win count: identity `1`, c=0.5 `2`.

Interpretation: 262K is also a near-tie/noisy region under this clean StreamingMuon recipe; it does not support a robust identity-only claim.
