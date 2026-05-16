from __future__ import annotations

import torch

from baseline_optim import (
    StructuredAdamW,
    _kl_inv_sqrt_clamp,
    _project_2d,
    _project_back_2d,
    matrix_sign_exact,
    matrix_sign_ns5,
)


def test_structured_optimizer_matrix_kinds_step() -> None:
    torch.manual_seed(0)
    kinds = ["soap", "shampoo", "kl_shampoo", "kl_soap"]
    for kind in kinds:
        weight = torch.nn.Parameter(torch.randn(8, 4))
        weight.grad = torch.randn_like(weight)
        before = weight.detach().clone()
        opt = StructuredAdamW([
            {
                "kind": kind,
                "params": [weight],
                "lr": 1e-3,
                "betas": (0.9, 0.95),
                "shampoo_beta": 0.95,
                "eps": 1e-8,
                "weight_decay": 0.01,
                "precondition_frequency": 2,
                "init_factor": 1.0,
                "use_qr": True,
            }
        ])
        opt.step()
        assert torch.isfinite(weight).all()
        assert torch.allclose(weight, before)
        weight.grad = torch.randn_like(weight)
        opt.step()
        assert torch.isfinite(weight).all()
        assert not torch.allclose(weight, before)


def test_kl_basis_refresh_preserves_eigenvalue_ema() -> None:
    torch.manual_seed(0)
    weight = torch.nn.Parameter(torch.randn(8, 4))
    opt = StructuredAdamW([
        {
            "kind": "kl_shampoo",
            "params": [weight],
            "lr": 1e-3,
            "betas": (0.95, 0.9),
            "shampoo_beta": 0.9,
            "eps": 1e-8,
            "weight_decay": 0.0,
            "precondition_frequency": 1,
            "init_factor": 0.1,
            "use_qr": True,
        }
    ])
    weight.grad = torch.randn_like(weight)
    opt.step()
    state = opt.state[weight]

    weight.grad = torch.randn_like(weight)
    opt.step()
    expected = [x.clone() for x in state["eigen_sqrt_inv"]]
    q_before = [x.clone() for x in state["Q"]]
    opt._refresh_kl_basis(state, opt.param_groups[0])

    assert all(torch.allclose(a, b) for a, b in zip(state["eigen_sqrt_inv"], expected))
    assert any(not torch.allclose(a, b) for a, b in zip(state["Q"], q_before))


def test_kl_shampoo_update_has_no_adam_bias_correction() -> None:
    torch.manual_seed(0)
    weight = torch.nn.Parameter(torch.randn(4, 3))
    grad1 = torch.randn_like(weight)
    grad2 = torch.randn_like(weight)
    lr = 1e-3
    beta1 = 0.95
    eps = 1e-8
    opt = StructuredAdamW([
        {
            "kind": "kl_shampoo",
            "params": [weight],
            "lr": lr,
            "betas": (beta1, 0.9),
            "shampoo_beta": 0.9,
            "eps": eps,
            "weight_decay": 0.0,
            "precondition_frequency": 100,
            "init_factor": 0.1,
            "use_qr": True,
        }
    ])

    weight.grad = grad1
    opt.step()  # bootstrap only
    before = weight.detach().clone()
    state = opt.state[weight]
    q_left, q_right = [x.clone() for x in state["Q"]]
    inv_left, inv_right = [x.clone() for x in state["eigen_sqrt_inv"]]

    momentum = (1.0 - beta1) * grad2.float()
    projected = _project_2d(momentum, q_left, q_right)
    scale = inv_left.view(-1, 1) * inv_right.view(1, -1)
    scale = scale / (1.0 + scale * eps)
    expected_update = _project_back_2d(projected * scale, q_left, q_right)

    weight.grad = grad2
    opt.step()

    assert torch.allclose(weight, before - lr * expected_update.to(weight.dtype), atol=1e-6, rtol=1e-6)


def test_kl_inv_sqrt_clamp_matches_reference_dimension_rule() -> None:
    assert _kl_inv_sqrt_clamp(4, {}) == 10.0
    assert _kl_inv_sqrt_clamp(512, {}) == 512.0
    assert _kl_inv_sqrt_clamp(8192, {}) == 4000.0


def test_soap_bootstrap_uses_ema_scaled_preconditioner() -> None:
    torch.manual_seed(0)
    weight = torch.nn.Parameter(torch.randn(4, 3))
    grad = torch.randn_like(weight)
    weight.grad = grad.clone()
    beta = 0.9
    opt = StructuredAdamW([
        {
            "kind": "soap",
            "params": [weight],
            "lr": 1e-3,
            "betas": (0.95, 0.99),
            "shampoo_beta": beta,
            "eps": 1e-8,
            "weight_decay": 0.0,
            "precondition_frequency": 10,
            "init_factor": 1.0,
            "use_qr": True,
        }
    ])

    opt.step()
    state = opt.state[weight]

    assert torch.allclose(state["GG"][0], (1.0 - beta) * (grad.float() @ grad.float().T))
    assert torch.allclose(state["GG"][1], (1.0 - beta) * (grad.float().T @ grad.float()))


def test_shampoo_update_matches_two_sided_inverse_quarter_power() -> None:
    torch.manual_seed(0)
    weight = torch.nn.Parameter(torch.randn(4, 3))
    grad1 = torch.randn_like(weight)
    grad2 = torch.randn_like(weight)
    lr = 1e-3
    beta1 = 0.95
    eps = 1e-8
    opt = StructuredAdamW([
        {
            "kind": "shampoo",
            "params": [weight],
            "lr": lr,
            "betas": (beta1, 0.99),
            "shampoo_beta": 0.95,
            "eps": eps,
            "weight_decay": 0.0,
            "precondition_frequency": 100,
            "init_factor": 1.0,
            "use_qr": True,
            "correct_bias": False,
        }
    ])

    weight.grad = grad1
    opt.step()  # bootstrap only
    before = weight.detach().clone()
    state = opt.state[weight]
    q_left, q_right = [x.clone() for x in state["Q"]]
    left_scale = state["eigenvalues"][0].clamp_min(eps).pow(-0.25)
    right_scale = state["eigenvalues"][1].clamp_min(eps).pow(-0.25)

    momentum = (1.0 - beta1) * grad2.float()
    projected = _project_2d(momentum, q_left, q_right)
    expected_update = _project_back_2d(
        projected * left_scale.view(-1, 1) * right_scale.view(1, -1),
        q_left,
        q_right,
    )

    weight.grad = grad2
    opt.step()

    assert torch.allclose(weight, before - lr * expected_update.to(weight.dtype), atol=1e-6, rtol=1e-6)


def test_plain_muon_uses_ns5_without_dim_lr_scaling() -> None:
    grad = torch.tensor([[3.0, 0.0], [0.0, 1.0]])
    weight = torch.nn.Parameter(torch.zeros_like(grad))
    weight.grad = grad.clone()
    opt = StructuredAdamW([
        {
            "kind": "plain_muon",
            "params": [weight],
            "lr": 0.1,
            "momentum": 0.0,
            "ns_steps": 5,
            "weight_decay": 0.0,
        }
    ])
    opt.step()
    expected = -0.1 * matrix_sign_ns5(grad, steps=5)
    assert torch.allclose(weight, expected, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(matrix_sign_ns5(grad, steps=5), matrix_sign_exact(grad), atol=1e-3, rtol=1e-3)


def test_plain_muon_weight_decay_is_decoupled_from_update() -> None:
    grad = torch.zeros(2, 2)
    weight = torch.nn.Parameter(torch.ones_like(grad))
    weight.grad = grad.clone()
    opt = StructuredAdamW([
        {
            "kind": "plain_muon",
            "params": [weight],
            "lr": 0.1,
            "momentum": 0.0,
            "ns_steps": 5,
            "weight_decay": 0.2,
        }
    ])
    opt.step()
    assert torch.allclose(weight, torch.full_like(weight, 0.98))
