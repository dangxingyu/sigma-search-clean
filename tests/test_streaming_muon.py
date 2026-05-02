"""Fast CPU checks for the clean StreamingMuon runtime."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from streaming_muon_torch import CustomTransform, IdentityTransform, streaming_msign


def _true_msign(matrix: torch.Tensor) -> torch.Tensor:
    u, _, vt = torch.linalg.svd(matrix, full_matrices=False)
    return u @ vt


def test_streaming_msign_matches_svd_tall_matrix() -> None:
    torch.manual_seed(42)
    matrix = torch.randn(64, 32)
    expected = _true_msign(matrix)

    basis = None
    for _ in range(40):
        update, basis, sigma = streaming_msign(matrix, basis, steps=1)

    rel_err = (update - expected).norm() / expected.norm()
    assert rel_err < 0.02
    assert sigma.shape == (32,)


def test_streaming_msign_matches_svd_wide_matrix() -> None:
    torch.manual_seed(123)
    matrix = torch.randn(32, 64)
    expected = _true_msign(matrix)

    basis = None
    for _ in range(40):
        update, basis, _ = streaming_msign(matrix, basis, steps=1)

    rel_err = (update - expected).norm() / expected.norm()
    assert rel_err < 0.02
    assert update.shape == matrix.shape


def test_identity_transform_matches_plain_update() -> None:
    torch.manual_seed(7)
    matrix = torch.randn(48, 24)
    basis = None
    for _ in range(25):
        _, basis, _ = streaming_msign(matrix, basis, steps=1)

    plain, _, _ = streaming_msign(matrix, basis, steps=1)
    transformed, _, _ = streaming_msign(
        matrix,
        basis,
        steps=1,
        sigma_transform=IdentityTransform(),
        sigma_state={},
    )

    assert torch.allclose(transformed, plain, atol=1e-6, rtol=1e-6)


def test_top_aware_candidate_scales_largest_sigma() -> None:
    candidate_path = Path(__file__).resolve().parents[1] / "candidates" / "top_aware_muon.py"
    namespace = {"torch": torch, "math": math}
    exec(candidate_path.read_text(), namespace)
    transform = CustomTransform(namespace["f"])

    sigma = torch.tensor([[4.0, 1.0, 3.0]])
    scale = transform(sigma, {"candidate_params": {"top_k": 2, "alpha": 0.5}})

    assert torch.equal(scale, torch.tensor([[0.5, 1.0, 0.5]]))
