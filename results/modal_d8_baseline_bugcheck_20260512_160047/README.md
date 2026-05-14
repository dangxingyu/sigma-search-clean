# Modal d8 Structured Optimizer Bugcheck

Run: `modal_d8_baseline_bugcheck_20260512_160047`

Recipe: depth 8 GPT-2, batch `262144`, `0.1x` Chinchilla (`153` steps), seed `42`, `8xB200`. This run used the older structured config for SOAP/Shampoo (`beta2=0.95`, `shampoo_beta=0.95`) and `WEIGHT_DECAY=0.28`, so it is a bugcheck rather than the reference recipe.

Best rows:

| method | best lr | val BPB |
|---|---:|---:|
| `kl_soap` | `0.015` | `1.509367` |
| `adamw` | `0.0005` | `1.677759` |
| `soap` | `0.002` | `1.704571` |
| `shampoo` | `0.00025` | `1.746945` |
| `kl_shampoo` | `0.0005` | `2.353443` |

Interpretation: KL-SOAP is not fundamentally broken and beats AdamW in this short sanity run. SOAP, Shampoo, and KL-Shampoo do not beat AdamW under this old recipe.
