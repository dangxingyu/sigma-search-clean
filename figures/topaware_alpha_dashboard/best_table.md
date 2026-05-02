# Top-Aware Alpha Dashboard

Lower BPB is better. This table auto-scans the completed v28/v29/v31 Top-Aware sweep result JSONs plus the historical 128K alpha=.5 rows.

| batch | method | best BPB | best LR | delta vs identity | status |
|---:|---|---:|---:|---:|---|
| 128K | `streaming_identity` | `0.916968` | `0.01` | `` | complete |
| 128K | `native_muon` | `0.916692` | `0.01` | `-0.000276` | complete |
| 128K | `top_aware_k1_a0.5` | `0.917040` | `0.01` | `+0.000072` | complete |
| 128K | `top_aware_k1_a0.75` | `0.917183` | `0.01` | `+0.000215` | complete |
| 256K | `streaming_identity` | `0.915119` | `0.01` | `` | complete |
| 256K | `streaming_lite` | `0.914020` | `0.01` | `-0.001099` | complete |
| 256K | `native_muon` | `0.913946` | `0.01` | `-0.001173` | complete |
| 256K | `top_aware_k1_a0.25` | `0.920442` | `0.02` | `+0.005323` | complete |
| 256K | `top_aware_k1_a0.5` | `0.916748` | `0.01` | `+0.001629` | complete |
| 256K | `top_aware_k1_a0.75` | `0.916292` | `0.01` | `+0.001173` | complete |
| 256K | `top_aware_k1_a0.875` | `0.916023` | `0.01` | `+0.000904` | complete |
| 512K | `streaming_identity` | `0.920964` | `0.01` | `` | complete |
| 512K | `native_muon` | `0.921940` | `0.01` | `+0.000975` | complete |
| 512K | `top_aware_k1_a0.25` | `0.923897` | `0.01` | `+0.002933` | complete |
| 512K | `top_aware_k1_a0.5` | `0.922203` | `0.01` | `+0.001239` | complete |
| 512K | `top_aware_k1_a0.75` | `0.924523` | `0.02` | `+0.003559` | complete |
| 512K | `top_aware_k1_a0.875` | `0.924970` | `0.01` | `+0.004006` | complete |

Current reading should be regenerated after each overnight sweep; use the raw CSV when a batch is still partially complete.
DDP `native_lite` rows from v29/v31 are excluded because that runner is not a valid distributed exact-LITE implementation.
