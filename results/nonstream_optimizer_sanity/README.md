# Non-streaming optimizer sanity sweep

Scope: depth=1 GPT-2-style nanochat smoke, seq=64, 100 optimizer steps, single B200, eager mode (`NANOCHAT_DISABLE_COMPILE=1`). This excludes StreamingMuon and Top-Aware Muon; it checks the separate baseline optimizer implementations only.

Note: the d1 `plain_muon` BPB rows below were generated before the NS5 correction and used exact SVD. Treat those old `plain_muon` BPB rows as stale; the d12 speed table uses the corrected implementation.

Files:
- `all_rows.csv`: every successful LR row.
- `best_by_batch_method.csv`: best LR/BPB per batch and optimizer.
- `lr_sweep_val_bpb.png`: LR sweep plot for the d1 sanity run.
- `d12_speed_ns5.csv`: d12 8xB200 speed probe using the corrected NS5 plain Muon implementation.

Best rows:

| batch | method | best lr | best val BPB |
|---:|---|---:|---:|
| 512 | adamw | 0.004 | 2.522123 |
| 512 | kl_shampoo | 0.002 | 2.525024 |
| 512 | kl_soap | 0.004 | 2.522290 |
| 512 | plain_muon | 0.002 | 2.525050 |
| 512 | shampoo | 0.002 | 2.525075 |
| 512 | soap | 0.004 | 2.522288 |
| 2048 | adamw | 0.000125 | 2.370991 |
| 2048 | kl_shampoo | 0.0005 | 2.370675 |
| 2048 | kl_soap | 0.000125 | 2.371001 |
| 2048 | plain_muon | 0.00025 | 2.370737 |
| 2048 | shampoo | 0.0005 | 2.370667 |
| 2048 | soap | 0.000125 | 2.371003 |

## d12 speed probe

Recipe: depth `12`, dim `768`, seq `1024`, global batch `524288`, device batch `32`, 8xB200 DDP, no eval, 60 optimizer steps.

The first `plain_muon` probe accidentally used exact SVD and is not a valid
ordinary-Muon speed row. `plain_muon` now uses NS5 matrix-sign orthogonalization
without nanochat's dimension LR normalization.

| method | total wall time | step-50 dt | approximate steady-state tokens/s |
|---|---:|---:|---:|
| adamw | 37.1s | 0.090s | 5.83M |
| plain_muon | 33.3s | 0.105s | 4.99M |
| soap | 44.9s | 0.128s | 4.10M |
| shampoo | 45.9s | 0.135s | 3.88M |
| kl_shampoo | 41.9s | 0.160s | 3.28M |
| kl_soap | 43.4s | 0.157s | 3.34M |

Interpretation: with NS5, ordinary Muon is not the speed bottleneck. The structured optimizer baselines are all in the same broad speed range for this 60-step d12 probe.
