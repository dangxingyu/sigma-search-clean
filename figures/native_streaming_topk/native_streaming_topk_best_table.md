# Native / Streaming Identity / Top-Aware Dashboard

Top-Aware here means the old top1 rule: `top_k=1`, `alpha=0.5`.
Caveat: native Muon rows are exact-control single-LR DDP runs, not full LR sweeps. Streaming identity and Top-Aware rows are LR-swept within their run family.

| batch | native Muon | StreamingMuon identity | Top-Aware k=1 a=.5 | native - identity | Top-Aware - identity | notes |
|---:|---:|---:|---:|---:|---:|---|
| 32K | `` | `0.935064 @ 0.005` | `` | `` | `` | native missing, top-aware missing, v26 streaming-only |
| 64K | `` | `0.922145 @ 0.01` | `0.923802 @ 0.01` | `` | `+0.001658` | native missing, v26 streaming-only |
| 128K | `0.916692 @ 0.01` | `0.916968 @ 0.01` | `0.917040 @ 0.01` | `-0.000276` | `+0.000072` |  |
| 1M | `0.933375 @ 0.01` | `0.934979 @ 0.01` | `0.933161 @ 0.02` | `-0.001604` | `-0.001818` |  |
| 8M | `1.118693 @ 0.02` | `1.098501 @ 0.02` | `1.083120 @ 0.02` | `+0.020191` | `-0.015381` |  |

Interpretation:
- 128K and 1M: native Muon exact-control is slightly better than StreamingMuon identity, but this is not a full native LR sweep.
- 1M and 8M: Top-Aware k=1 alpha=.5 beats StreamingMuon identity after LR tuning.
- 64K: v26 streaming-only says Top-Aware is worse than identity.
- 32K: identity exists, but Top-Aware has not completed in v26, so no comparison row yet.
