# Historical Large-Batch Dashboard

This dashboard separates experiment families. Do not raw-compare BPB between `native_single_gpu` and `streaming_ddp_top1_old`; data/eval streams differ.

## Native Single-GPU Muon vs LITE Best Rows

| batch | Muon best | LITE best | Muon - LITE | winner |
|---:|---:|---:|---:|---|
| 128K | `0.893701 @ lr=0.01` | `0.893577 @ lr=0.01` | `+0.000124` | LITE |
| 256K | `0.892240 @ lr=0.01` | `0.892301 @ lr=0.01` | `-0.000061` | Muon/tie |
| 512K | `0.898926 @ lr=0.01` | `0.896504 @ lr=0.01` | `+0.002422` | LITE |
| 1M | `0.915446 @ lr=0.01` | `0.910612 @ lr=0.02` | `+0.004834` | LITE |
| 4M | `0.994062 @ lr=0.04` | `0.981683 @ lr=0.04` | `+0.012379` | LITE |
| 16M | `1.332891 @ lr=0.04` | `1.307633 @ lr=0.04` | `+0.025258` | LITE |

## Streaming DDP Identity vs Top1 Best Rows

| batch | identity best | top1 best | identity - top1 | winner |
|---:|---:|---:|---:|---|
| 128K | `0.916968 @ lr=0.01` | `0.917040 @ lr=0.01` | `-0.000072` | identity/tie |
| 1M | `0.934979 @ lr=0.01` | `0.933161 @ lr=0.02` | `+0.001818` | top1 |
| 8M | `1.098501 @ lr=0.02` | `1.083120 @ lr=0.02` | `+0.015381` | top1 |

## All Aggregate Rows

| family | batch | optimizer | LR | n | mean BPB | SEM |
|---|---:|---|---:|---:|---:|---:|
| `native_single_gpu` | 128K | `lite` | `0.005` | 3 | `0.897499` | `0.000316` |
| `native_single_gpu` | 128K | `lite` | `0.01` | 3 | `0.893577` | `0.000417` |
| `native_single_gpu` | 128K | `lite` | `0.02` | 3 | `0.895528` | `0.000164` |
| `native_single_gpu` | 128K | `lite` | `0.04` | 3 | `0.903576` | `0.000248` |
| `native_single_gpu` | 128K | `muon` | `0.005` | 3 | `0.897033` | `0.000325` |
| `native_single_gpu` | 128K | `muon` | `0.01` | 3 | `0.893701` | `0.000222` |
| `native_single_gpu` | 128K | `muon` | `0.02` | 3 | `0.895984` | `0.000208` |
| `native_single_gpu` | 128K | `muon` | `0.04` | 3 | `0.904335` | `0.000243` |
| `native_single_gpu` | 256K | `lite` | `0.01` | 3 | `0.892301` | `0.000445` |
| `native_single_gpu` | 256K | `muon` | `0.01` | 3 | `0.892240` | `0.000330` |
| `native_single_gpu` | 512K | `lite` | `0.01` | 3 | `0.896504` | `0.000217` |
| `native_single_gpu` | 512K | `muon` | `0.01` | 3 | `0.898926` | `0.000745` |
| `native_single_gpu` | 1M | `lite` | `0.02` | 3 | `0.910612` | `0.002621` |
| `native_single_gpu` | 1M | `muon` | `0.01` | 3 | `0.915446` | `0.003487` |
| `native_single_gpu` | 4M | `lite` | `0.04` | 3 | `0.981683` | `0.008167` |
| `native_single_gpu` | 4M | `muon` | `0.04` | 3 | `0.994062` | `0.010102` |
| `native_single_gpu` | 16M | `lite` | `0.04` | 3 | `1.307633` | `0.007694` |
| `native_single_gpu` | 16M | `muon` | `0.04` | 3 | `1.332891` | `0.005610` |
| `streaming_ddp_top1_old` | 128K | `identity` | `0.005` | 1 | `0.920045` | `0.000000` |
| `streaming_ddp_top1_old` | 128K | `identity` | `0.01` | 1 | `0.916968` | `0.000000` |
| `streaming_ddp_top1_old` | 128K | `identity` | `0.02` | 1 | `0.919606` | `0.000000` |
| `streaming_ddp_top1_old` | 128K | `identity` | `0.04` | 1 | `0.927473` | `0.000000` |
| `streaming_ddp_top1_old` | 128K | `top1_damp_a05` | `0.005` | 1 | `0.920471` | `0.000000` |
| `streaming_ddp_top1_old` | 128K | `top1_damp_a05` | `0.01` | 1 | `0.917040` | `0.000000` |
| `streaming_ddp_top1_old` | 128K | `top1_damp_a05` | `0.02` | 1 | `0.919900` | `0.000000` |
| `streaming_ddp_top1_old` | 128K | `top1_damp_a05` | `0.04` | 1 | `0.927863` | `0.000000` |
| `streaming_ddp_top1_old` | 1M | `identity` | `0.005` | 1 | `0.974548` | `0.000000` |
| `streaming_ddp_top1_old` | 1M | `identity` | `0.01` | 1 | `0.934979` | `0.000000` |
| `streaming_ddp_top1_old` | 1M | `identity` | `0.02` | 1 | `0.937271` | `0.000000` |
| `streaming_ddp_top1_old` | 1M | `identity` | `0.04` | 1 | `0.942769` | `0.000000` |
| `streaming_ddp_top1_old` | 1M | `top1_damp_a05` | `0.01` | 1 | `0.942311` | `0.000000` |
| `streaming_ddp_top1_old` | 1M | `top1_damp_a05` | `0.02` | 1 | `0.933161` | `0.000000` |
| `streaming_ddp_top1_old` | 1M | `top1_damp_a05` | `0.04` | 1 | `0.937538` | `0.000000` |
| `streaming_ddp_top1_old` | 8M | `identity` | `0.01` | 1 | `1.115494` | `0.000000` |
| `streaming_ddp_top1_old` | 8M | `identity` | `0.02` | 1 | `1.098501` | `0.000000` |
| `streaming_ddp_top1_old` | 8M | `identity` | `0.04` | 1 | `1.103663` | `0.000000` |
| `streaming_ddp_top1_old` | 8M | `top1_damp_a05` | `0.01` | 1 | `1.099640` | `0.000000` |
| `streaming_ddp_top1_old` | 8M | `top1_damp_a05` | `0.02` | 1 | `1.083120` | `0.000000` |
| `streaming_ddp_top1_old` | 8M | `top1_damp_a05` | `0.04` | 1 | `1.087870` | `0.000000` |
