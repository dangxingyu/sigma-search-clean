from __future__ import annotations

import torch

from baseline_optim import StructuredAdamW, matrix_sign_exact, matrix_sign_ns5


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
