# Sweep Catalog

Curated index of existing optimizer sweeps. Lower validation BPB is better.

Generated files:
- `all_sweeps.csv`: normalized row-level results.
- `best_by_batch_method.csv`: best observed row per `(family, batch, method, alpha)`.
- `summary.json`: counts and coverage.

Important comparison rule: compare rows within the same `family` first. Historical native single-GPU rows and newer same-driver StreamingMuon DDP rows can differ in absolute BPB.

## Coverage

- Batch sizes present: 32K, 64K, 128K, 256K, 512K, 1M, 4M, 8M, 16M.
- Alpha values present: 0.25, 0.5, 0.75, 0.85, 0.875, 1, 1.15, 1.25.
- Methods present: native_lite=27, native_muon=39, streaming_identity=27, streaming_lite=9, top_aware_muon=57.

## Sources

- `figures/current_same_driver_v26/raw_rows.csv`: 15 rows.
- `figures/historical_large_batch/raw_rows.csv`: 67 rows.
- `figures/native_streaming_topk/native_streaming_topk_raw.csv`: 2 rows.
- `figures/topaware_alpha_dashboard/raw_rows.csv`: 65 rows.
- `results/dynamics_128k_200step/summary.json`: 3 rows.
- `results/topaware_128k_alpha125_lrsweep/summary.json`: 4 rows.
- `results/topaware_128k_near_identity_lr001/summary.json`: 3 rows.

## Best Rows Snapshot

| family | batch | method | alpha | lr | seed | BPB |
|---|---:|---|---:|---:|---:|---:|
| same_driver_streaming_ddp | 32K | streaming_identity | 1 | 0.005 | 42 | 0.935064 |
| same_driver_streaming_ddp | 32K | streaming_lite | - | 0.005 | 42 | 0.933131 |
| same_driver_streaming_ddp | 64K | streaming_identity | 1 | 0.01 | 42 | 0.922145 |
| same_driver_streaming_ddp | 64K | streaming_lite | - | 0.01 | 42 | 0.922291 |
| same_driver_streaming_ddp | 64K | top_aware_muon | 0.5 | 0.01 | 42 | 0.923802 |
| dynamics_smoke | 128K | streaming_identity | 1 | 0.01 | 42 | 1.586512 |
| dynamics_smoke | 128K | top_aware_muon | 1.25 | 0.01 | 42 | 1.600977 |
| native_ddp_or_control | 128K | native_muon | - | 0.01 | 42 | 0.916692 |
| native_single_gpu | 128K | native_lite | - | 0.01 | 44 | 0.892904 |
| native_single_gpu | 128K | native_muon | - | 0.01 | 44 | 0.893413 |
| same_driver_streaming_ddp | 128K | streaming_identity | 1 | 0.01 | 42 | 0.916968 |
| same_driver_streaming_ddp | 128K | top_aware_muon | 0.5 | 0.01 | 42 | 0.917040 |
| same_driver_streaming_ddp | 128K | top_aware_muon | 0.75 | 0.01 | 42 | 0.917183 |
| same_driver_streaming_ddp | 128K | top_aware_muon | 0.85 | 0.01 | 42 | 0.916766 |
| same_driver_streaming_ddp | 128K | top_aware_muon | 1.15 | 0.01 | 42 | 0.916569 |
| same_driver_streaming_ddp | 128K | top_aware_muon | 1.25 | 0.01 | 42 | 0.917164 |
| native_ddp_or_control | 256K | native_muon | - | 0.01 | 42 | 0.913946 |
| native_single_gpu | 256K | native_lite | - | 0.01 | 42 | 0.891713 |
| native_single_gpu | 256K | native_muon | - | 0.01 | 44 | 0.891630 |
| same_driver_streaming_ddp | 256K | streaming_identity | 1 | 0.01 | 42 | 0.915119 |
| same_driver_streaming_ddp | 256K | streaming_lite | - | 0.01 | 42 | 0.914020 |
| same_driver_streaming_ddp | 256K | top_aware_muon | 0.25 | 0.02 | 42 | 0.920442 |
| same_driver_streaming_ddp | 256K | top_aware_muon | 0.5 | 0.01 | 42 | 0.916748 |
| same_driver_streaming_ddp | 256K | top_aware_muon | 0.75 | 0.01 | 42 | 0.916292 |
| same_driver_streaming_ddp | 256K | top_aware_muon | 0.875 | 0.01 | 42 | 0.916023 |
| native_ddp_or_control | 512K | native_muon | - | 0.01 | 42 | 0.921940 |
| native_single_gpu | 512K | native_lite | - | 0.01 | 43 | 0.896285 |
| native_single_gpu | 512K | native_muon | - | 0.01 | 44 | 0.897899 |
| same_driver_streaming_ddp | 512K | streaming_identity | 1 | 0.01 | 42 | 0.920964 |
| same_driver_streaming_ddp | 512K | top_aware_muon | 0.25 | 0.01 | 42 | 0.923897 |
| same_driver_streaming_ddp | 512K | top_aware_muon | 0.5 | 0.01 | 42 | 0.922203 |
| same_driver_streaming_ddp | 512K | top_aware_muon | 0.75 | 0.02 | 42 | 0.924523 |
| same_driver_streaming_ddp | 512K | top_aware_muon | 0.875 | 0.01 | 42 | 0.924970 |
| native_ddp_or_control | 1M | native_muon | - | 0.01 | 42 | 0.933375 |
| native_single_gpu | 1M | native_lite | - | 0.02 | 42 | 0.906798 |
| native_single_gpu | 1M | native_muon | - | 0.01 | 42 | 0.910022 |
| streaming_ddp_top1_old | 1M | streaming_identity | 1 | 0.01 | 42 | 0.934979 |
| streaming_ddp_top1_old | 1M | top_aware_muon | 0.5 | 0.02 | 42 | 0.933161 |
| native_single_gpu | 4M | native_lite | - | 0.04 | 42 | 0.969289 |
| native_single_gpu | 4M | native_muon | - | 0.04 | 42 | 0.979475 |
| native_ddp_or_control | 8M | native_muon | - | 0.02 | 42 | 1.118693 |
| streaming_ddp_top1_old | 8M | streaming_identity | 1 | 0.02 | 42 | 1.098501 |
| streaming_ddp_top1_old | 8M | top_aware_muon | 0.5 | 0.02 | 42 | 1.083120 |
| native_single_gpu | 16M | native_lite | - | 0.04 | 44 | 1.295139 |
| native_single_gpu | 16M | native_muon | - | 0.04 | 42 | 1.325998 |
