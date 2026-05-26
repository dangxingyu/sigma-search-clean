from __future__ import annotations

import math

from optimizer_recipe import (
    B_REF,
    adam_lr_from_matrix_lr,
    batch_lr_scale,
    ema_beta_for_batch,
    moonlight_matrix_lr_factor,
)


def test_batch_lr_scale_reference_is_one() -> None:
    assert batch_lr_scale(B_REF) == 1.0


def test_batch_lr_scale_grows_with_sqrt_batch() -> None:
    assert batch_lr_scale(4 * B_REF) == 2.0


def test_ema_beta_for_batch_keeps_token_half_life() -> None:
    beta_ref = 0.95
    batch_small = B_REF
    batch_large = 4 * B_REF
    beta_large = ema_beta_for_batch(beta_ref, batch_large, B_REF)

    half_life_tokens = batch_small * math.log(0.5) / math.log(beta_ref)
    half_life_large = batch_large * math.log(0.5) / math.log(beta_large)
    assert math.isclose(half_life_tokens, half_life_large, rel_tol=1e-6, abs_tol=1e-6)


def test_moonlight_matrix_lr_factor() -> None:
    assert moonlight_matrix_lr_factor(768, 768) == 0.2 * math.sqrt(768)


def test_adam_lr_relative_to_matrix_lr() -> None:
    lr = adam_lr_from_matrix_lr(
        0.04,
        "embedding",
        dmodel_lr_scale=1.0,
        batch_lr_scale_value=1.0,
    )
    assert lr == 0.04 * (0.3 / 0.02)


def test_scalar_adam_lr_does_not_use_dmodel_scale() -> None:
    lr = adam_lr_from_matrix_lr(
        0.04,
        "scalar",
        dmodel_lr_scale=0.5,
        batch_lr_scale_value=2.0,
    )
    assert lr == 0.04 * (0.005 / 0.02) * 2.0


def test_embedding_adam_lr_uses_dmodel_scale() -> None:
    lr = adam_lr_from_matrix_lr(
        0.04,
        "embedding",
        dmodel_lr_scale=0.5,
        batch_lr_scale_value=2.0,
    )
    assert lr == 0.04 * (0.3 / 0.02) * 0.5 * 2.0
