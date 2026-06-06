# Qwen3 d12 64K Muon Coordinate Descent

Goal: tune Muon at `depth=12`, `architecture=qwen3`, `batch=64K`, `1x Chinchilla`.

Coordinate dimensions:

- `matrix_lr`
- `muon_momentum` / matrix optimizer beta1
- auxiliary `adam_lr_multiplier`
- auxiliary `adam_beta1`
- `weight_decay`

Initial point is the best known d12 64K Muon recipe before coordinate descent:

```text
matrix_lr=0.008
muon_momentum=0.95
adam_lr_multiplier=1.0
adam_beta1=0.8
weight_decay=0.20
score=0.8623740088803121
```

Each round sweeps five candidates per active coordinate, evaluates completed runs by final/best validation BPB, then accepts the single coordinate change with the largest gain. The run stops after at most six rounds or when no coordinate improves.
