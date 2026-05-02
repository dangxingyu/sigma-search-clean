# v49/v50 2M Seed Confirmation

Batch `2,097,152`, d8 / 0.4B tokens, same `lr=0.08` for identity and Top-Aware `c=0.5`. Delta is `c=0.5 - identity`; lower is better.

| seed | identity | c=0.5 | delta | winner |
|---:|---:|---:|---:|---|
| 42 | 1.068525 | 1.082541 | +0.014016 | identity |
| 43 | 1.062976 | 1.066248 | +0.003272 | identity |
| 44 | 1.081495 | 1.072403 | -0.009092 | c=0.5 |
| 45 | 1.081773 | 1.074180 | -0.007593 | c=0.5 |
| 46 | 1.068684 | 1.070616 | +0.001932 | identity |

Mean delta: `+0.000507 ± 0.004183` SEM over `5` seeds.
Win count: identity `3`, c=0.5 `2`.

Interpretation: 2M is a noisy/tie-like transition point, not a stable identity-win claim. The mean currently leans identity, but the sign flips across seeds.
