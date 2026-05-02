# v51 4M Seed Confirmation

Batch `4,194,304`, d8 / 0.4B tokens. For each seed, identity and Top-Aware choose best available LR from `{0.02,0.04}`. Delta is `c=0.5 - identity`; lower is better.

| seed | identity best | c=0.5 best | delta | winner |
|---:|---:|---:|---:|---|
| 42 | 1.206814 @ 0.04 | 1.174488 @ 0.02 | -0.032326 | c=0.5 |
| 43 | 1.228649 @ 0.04 | 1.203347 @ 0.04 | -0.025302 | c=0.5 |
| 44 | 1.209536 @ 0.04 | 1.191446 @ 0.02 | -0.018091 | c=0.5 |

Mean delta: `-0.025240 ± 0.004110` SEM over `3` seeds.
Win count: identity `0`, c=0.5 `3`.

Interpretation: 4M Top-Aware win is robust across the three seeds tested.
