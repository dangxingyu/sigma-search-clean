# Qwen3 d12 64K Muon Coordinate Descent Visualization

Lower score / BPB is better.

## Final recipe

```text
matrix_lr=0.008
muon_momentum=0.95
adam_lr_multiplier=4.0
adam_beta1=0.683772
weight_decay=0.2
score=0.856601
source=qwen3_d12_muon64_cd_r02_adam_lr_multiplier_4_20260605
```

## Files

- [qwen3_d12_muon64_cd_path.png](qwen3_d12_muon64_cd_path.png)
- [qwen3_d12_muon64_cd_improvement_heatmap.png](qwen3_d12_muon64_cd_improvement_heatmap.png)
- [qwen3_d12_muon64_cd_sweeps.png](qwen3_d12_muon64_cd_sweeps.png)
- [qwen3_d12_muon64_cd_coordinate_overlays.png](qwen3_d12_muon64_cd_coordinate_overlays.png)
- [qwen3_d12_muon64_cd_round_summary.csv](qwen3_d12_muon64_cd_round_summary.csv)
- [qwen3_d12_muon64_cd_round_00_lrsweep_style.png](qwen3_d12_muon64_cd_round_00_lrsweep_style.png)
- [qwen3_d12_muon64_cd_round_01_lrsweep_style.png](qwen3_d12_muon64_cd_round_01_lrsweep_style.png)
- [qwen3_d12_muon64_cd_round_02_lrsweep_style.png](qwen3_d12_muon64_cd_round_02_lrsweep_style.png)
- [qwen3_d12_muon64_cd_round_03_lrsweep_style.png](qwen3_d12_muon64_cd_round_03_lrsweep_style.png)
- [qwen3_d12_muon64_cd_full_dashboard.png](qwen3_d12_muon64_cd_full_dashboard.png)

## Notes

- Each sweep panel varies only the coordinate named in the column; all other hyperparameters are fixed at that round's center.
- Positive-valued coordinates use a log-spaced x-axis. Beta-like coordinates use a log(1-beta) transformed x-axis, matching the search grid near 1.
- The horizontal center score is the accepted center for that round; candidate rows at the same hyperparameter value are independent re-evaluations.
