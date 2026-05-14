# Modal d8 Structured Optimizer Tune

Run: `modal_d8_structured_tune_lrmin_20260512_194654`

Recipe: depth 8 GPT-2, batch `262144`, `0.1x` Chinchilla (`40,265,318` tokens, `153` steps), seed `42`, `8xB200`, `WEIGHT_DECAY=0.1`, `STRUCTURED_CONFIG=auto`, adaptive LR enabled with `LR_MIN=0.00003125`.

Best rows:

| method | best lr | val BPB |
|---|---:|---:|
| `kl_soap` | `0.008` | `1.523088` |
| `adamw` | `0.0005` | `1.677638` |
| `soap` | `0.004` | `1.726418` |
| `shampoo` | `0.000125` | `1.741547` |
| `kl_shampoo` | `0.000125` | `2.353109` |

Interpretation: KL-SOAP beats AdamW in this short d8/262K sanity run. SOAP, Shampoo, and KL-Shampoo do not beat AdamW; KL-Shampoo is nearly flat and likely needs implementation/formulation audit before using it as a credible baseline.
