"""Shared optimizer recipe helpers for LR and EMA scaling.

Moonlight-style matrix LR scaling and token-constant EMA half-life batch
adjustments are centralized here so ``run_eval.py`` and ``baseline_optim.py``
stay aligned.
"""

from __future__ import annotations

import math

# nanochat d12 reference batch in tokens (see nanochat/scripts/base_train.py).
B_REF = 2**19  # 524_288
REF_MATRIX_LR = 0.02

# nanochat ``setup_optimizer`` peak LRs at ``matrix_lr=REF_MATRIX_LR`` and dim=768.
ADAM_LR_RATIO_TO_MATRIX = {
    "lm_head": 0.008 / REF_MATRIX_LR,
    "embedding": 0.3 / REF_MATRIX_LR,
    "value_embed": 0.15 / REF_MATRIX_LR,  # embedding * 0.5
    "scalar": 0.005 / REF_MATRIX_LR,  # scalar_lr(0.5) * 0.01
}


def batch_lr_scale(total_batch_size: int, ref_batch: int = B_REF) -> float:
    if total_batch_size <= 0 or ref_batch <= 0:
        raise ValueError(f"batch sizes must be positive, got {total_batch_size=} {ref_batch=}")
    if total_batch_size == ref_batch:
        return 1.0
    return (total_batch_size / ref_batch) ** 0.5


def ema_beta_for_batch(beta_ref: float, batch_size: int, ref_batch: int = B_REF) -> float:
    """Match EMA half-life in tokens when the global batch size changes.

    If ``m_t = beta m_{t-1} + (1-beta) g_t`` has half-life ``T`` tokens at
    ``ref_batch``, then at ``batch_size`` we use ``beta_ref ** (batch_size /
    ref_batch)`` so the token half-life stays ``T``.
    """
    if not (0.0 < beta_ref < 1.0):
        return beta_ref
    if batch_size <= 0 or ref_batch <= 0 or batch_size == ref_batch:
        return beta_ref
    return float(beta_ref ** (batch_size / ref_batch))


def moonlight_matrix_lr_factor(rows: int, cols: int) -> float:
    """Moonlight RMS-matching scale: ``0.2 * sqrt(max(A, B))``."""
    return 0.2 * math.sqrt(float(max(rows, cols)))


def matrix_lr_step_scale(group: dict, rows: int, cols: int) -> float:
    mode = str(group.get("matrix_lr_adjust", "none")).lower()
    if mode == "moonlight":
        return moonlight_matrix_lr_factor(rows, cols)
    if mode in {"none", ""}:
        return 1.0
    raise ValueError(f"unknown matrix_lr_adjust={group.get('matrix_lr_adjust')!r}")


def adam_lr_from_matrix_lr(
    matrix_lr: float,
    role: str,
    *,
    dmodel_lr_scale: float,
    batch_lr_scale_value: float,
    ref_matrix_lr: float = REF_MATRIX_LR,
) -> float:
    if role not in ADAM_LR_RATIO_TO_MATRIX:
        raise KeyError(f"unknown Adam LR role {role!r}")
    lr = matrix_lr * ADAM_LR_RATIO_TO_MATRIX[role] * batch_lr_scale_value
    if role != "scalar":
        lr *= dmodel_lr_scale
    return lr
