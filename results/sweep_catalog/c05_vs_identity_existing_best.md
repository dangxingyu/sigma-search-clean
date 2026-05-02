# Existing Catalog: c=0.5 vs identity

| batch | method | c/alpha | best lr | best BPB | source |
|---:|---|---:|---:|---:|---|
| 32768 | streaming_identity | 1 | 0.005 | 0.935064 | v26_streaming_small |
| 65536 | streaming_identity | 1 | 0.01 | 0.922145 | v26_streaming_small |
| 65536 | top_aware_muon | 0.5 | 0.01 | 0.923802 | v26_streaming_small |
| 131072 | streaming_identity | 1 | 0.01 | 0.916501 | topaware_128k_alpha115_seed_confirm |
| 131072 | top_aware_muon | 0.5 | 0.01 | 0.917040 | ddp_top1_official |
| 262144 | streaming_identity | 1 | 0.01 | 0.915119 | v29_topaware_k1_256k_alpha_sweep_20260430 |
| 262144 | top_aware_muon | 0.5 | 0.01 | 0.916748 | v29_topaware_k1_256k_alpha_sweep_20260430 |
| 524288 | streaming_identity | 1 | 0.01 | 0.920964 | v31_topaware_k1_512k_alpha_sweep_20260501 |
| 524288 | top_aware_muon | 0.5 | 0.01 | 0.922203 | v31_topaware_k1_512k_alpha_sweep_20260501 |
| 1048576 | streaming_identity | 1 | 0.01 | 0.934979 | streaming_ddp_top1_old |
| 1048576 | top_aware_muon | 0.5 | 0.02 | 0.933161 | streaming_ddp_top1_old |
| 8388608 | streaming_identity | 1 | 0.02 | 1.098501 | streaming_ddp_top1_old |
| 8388608 | top_aware_muon | 0.5 | 0.02 | 1.083120 | streaming_ddp_top1_old |
