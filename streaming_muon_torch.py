"""
StreamingMuon (PyTorch): Port of the JAX/Levanter reference implementation for nanochat.

Core algorithm from streaming_muon.py (JAX): streaming power iteration with
double-pass Shifted Cholesky QR orthogonalization, warm-started bases.

Adds f(Σ) transform support for sigma-search compatibility.

Drop-in replacement for nanochat's MuonAdamW / DistMuonAdamW.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Protocol, Optional
import math


# Polar Express msign coefficients (from Polar Express paper, same as nanochat).
# 5-step quintic iteration that maps singular values toward 1.
# Robust to close/equal singular values (unlike power iteration on A^T A).
_POLAR_EXPRESS_COEFFS = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)


# =============================================================================
# Sigma Transform Interface (pluggable f(Σ) for sigma-search)
# =============================================================================

class SigmaTransform(Protocol):
    """Protocol for f(Σ) transforms. Operates coordinate-wise on singular values."""

    def __call__(self, sigma: Tensor, state: dict) -> Tensor:
        """
        Args:
            sigma: (..., k) current singular values
            state: mutable dict for EMA / auxiliary state (persists across steps)
        Returns:
            scaling: (..., k) multiplicative factors for each singular direction
        """
        ...


class IdentityTransform:
    """f(σ) = 1. Recovers standard Muon (msign)."""
    def __call__(self, sigma: Tensor, state: dict) -> Tensor:
        return torch.ones_like(sigma)


class TopAwareTransform:
    """Top-Aware Muon: damp the top-k sharp singular directions by alpha."""
    def __init__(self, top_k: int = 1, alpha: float = 0.5):
        self.top_k = top_k
        self.alpha = alpha

    def __call__(self, sigma: Tensor, state: dict) -> Tensor:
        if sigma.numel() == 0:
            return torch.ones_like(sigma)
        k = min(max(int(self.top_k), 0), sigma.shape[-1])
        scale = torch.ones_like(sigma)
        if k == 0:
            return scale
        top_idx = torch.topk(sigma.detach(), k=k, dim=-1).indices
        scale.scatter_(-1, top_idx, float(self.alpha))
        return scale


class TikhonovTransform:
    """f(σ) = λ / (σ + λ). Soft suppression of sharp directions."""
    def __init__(self, lam: float = 1.0):
        self.lam = lam
    def __call__(self, sigma: Tensor, state: dict) -> Tensor:
        return self.lam / (sigma + self.lam)


class ClipTransform:
    """f(σ) = min(1, c / σ). Hard clipping of sharp directions."""
    def __init__(self, c: float = 1.0, adaptive: bool = False, gamma: float = 1.0,
                 ema_beta: float = 0.99):
        self.c = c
        self.adaptive = adaptive
        self.gamma = gamma
        self.ema_beta = ema_beta

    def __call__(self, sigma: Tensor, state: dict) -> Tensor:
        if self.adaptive:
            if 'sigma_ema' not in state:
                state['sigma_ema'] = sigma.clone()
            else:
                state['sigma_ema'].lerp_(sigma, 1 - self.ema_beta)
            c = self.gamma * state['sigma_ema'].median(dim=-1, keepdim=True).values
        else:
            c = self.c
        return torch.clamp(c / (sigma + 1e-8), max=1.0)


class SigmoidTransform:
    """f(σ) = 1 / (1 + (σ/c)^p). Smooth sigmoid-like suppression."""
    def __init__(self, c: float = 1.0, p: float = 2.0, adaptive: bool = False,
                 gamma: float = 1.0, ema_beta: float = 0.99):
        self.c = c
        self.p = p
        self.adaptive = adaptive
        self.gamma = gamma
        self.ema_beta = ema_beta

    def __call__(self, sigma: Tensor, state: dict) -> Tensor:
        if self.adaptive:
            if 'sigma_ema' not in state:
                state['sigma_ema'] = sigma.clone()
            else:
                state['sigma_ema'].lerp_(sigma, 1 - self.ema_beta)
            c = self.gamma * state['sigma_ema'].median(dim=-1, keepdim=True).values
        else:
            c = self.c
        return 1.0 / (1.0 + (sigma / (c + 1e-8)) ** self.p)


class CustomTransform:
    """Wraps an arbitrary callable for sigma-search candidates."""
    def __init__(self, fn):
        self.fn = fn
    def __call__(self, sigma: Tensor, state: dict) -> Tensor:
        return self.fn(sigma, state)


SIGMA_TRANSFORMS = {
    'identity': IdentityTransform,
    'top_aware': TopAwareTransform,
    'tikhonov': TikhonovTransform,
    'clip': ClipTransform,
    'sigmoid': SigmoidTransform,
}


# =============================================================================
# Core: Shifted Cholesky QR orthogonalization (ported from JAX)
# =============================================================================

def _symmetrize(matrix: Tensor) -> Tensor:
    return 0.5 * (matrix + matrix.mT)


def _diagonal_regularizer(gram: Tensor, ridge_epsilon: float) -> Tensor:
    diagonal = torch.clamp(torch.diagonal(gram, dim1=-2, dim2=-1), min=0.0)
    ridge_floor = torch.finfo(gram.dtype).eps
    ridge_diag = torch.clamp(diagonal * ridge_epsilon, min=ridge_floor)
    return torch.diag_embed(ridge_diag)


def _scalar_regularizer(gram: Tensor, ridge_epsilon: float) -> Tensor:
    scale = torch.linalg.norm(gram.flatten(-2, -1), dim=-1, keepdim=True).unsqueeze(-1)
    ridge = torch.clamp(scale * ridge_epsilon, min=torch.finfo(gram.dtype).eps)
    eye = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
    return ridge * eye


def _orthogonality_error(matrix: Tensor) -> Tensor:
    gram = torch.matmul(matrix.mT.float(), matrix.float())
    eye = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
    return torch.linalg.norm((gram - eye).flatten(-2, -1), dim=-1) / gram.shape[-1]


def _qr_positive_diagonal(matrix: Tensor) -> Tensor:
    """QR decomposition with positive diagonal on R (for sign consistency)."""
    q, r = torch.linalg.qr(matrix, mode='reduced')
    diag_sign = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
    diag_sign = torch.where(diag_sign == 0, torch.ones_like(diag_sign), diag_sign)
    return q * diag_sign.unsqueeze(-2)


def _scqr_once(matrix: Tensor, gram: Tensor, ridge_epsilon: float,
               diagonal_regularization: bool) -> Tensor:
    """Single SCQR pass. Returns the approximately-orthogonal Q."""
    gram = _symmetrize(gram)
    if diagonal_regularization:
        regularized = gram + _diagonal_regularizer(gram, ridge_epsilon)
    else:
        regularized = gram + _scalar_regularizer(gram, ridge_epsilon)
    chol_lower = torch.linalg.cholesky(regularized)
    # Q = A R^{-1}: solve L X = A^T => X^T = Q
    scqr_q = torch.linalg.solve_triangular(chol_lower, matrix.mT.float(), upper=False).mT
    return scqr_q


@torch.compile(dynamic=False, fullgraph=True)
def _scqr_fused(matrix: Tensor, gram: Tensor, ridge_epsilon: float) -> Tensor:
    """
    Fused SCQR path (no fallback, no orthogonality check).
    Compiled for speed: all ops in one CUDA graph. All fp32 for stability.
    """
    gram = 0.5 * (gram + gram.mT)
    diagonal = torch.diagonal(gram, dim1=-2, dim2=-1).clamp(min=0.0)
    ridge_floor = torch.finfo(gram.dtype).eps
    ridge_diag = torch.clamp(diagonal * ridge_epsilon, min=ridge_floor)
    regularized = gram + torch.diag_embed(ridge_diag)
    chol_lower = torch.linalg.cholesky(regularized)
    scqr_q = torch.linalg.solve_triangular(chol_lower, matrix.mT, upper=False).mT
    return scqr_q


@torch.compile(dynamic=False, fullgraph=True)
def _streaming_power_step_fused(
    working: Tensor,
    basis: Tensor,
    gram_matrix: Tensor,
    ridge_epsilon: float,
) -> Tensor:
    """Single step of streaming power iteration with dual SCQR."""
    first_pass = torch.matmul(gram_matrix, basis)
    first_gram = torch.matmul(basis.mT, first_pass)
    second_pass = _scqr_fused(first_pass, first_gram, ridge_epsilon)
    second_gram = torch.matmul(second_pass.mT, second_pass)
    new_basis = _scqr_fused(second_pass, second_gram, ridge_epsilon)
    return new_basis


# NOTE: Matmul-only orthogonalization attempts (ColNorm+NS, Polar Express
# on gram @ V, etc.) were tested and all failed. The fundamental issue is
# that power iteration requires proper QR decomposition, not msign (polar
# factor). For arbitrary/SPD intermediate matrices, msign-based methods
# collapse to wrong answers (e.g., msign of SPD → I).
# Cholesky-based SCQR is the fastest stable QR for this use case on GPU,
# despite being slower than pure matmul (tensor cores).


@torch.compile(dynamic=False, fullgraph=True)
def _streaming_muon_fused_1iter(
    stacked_grads: Tensor,
    stacked_params: Tensor,
    momentum_buffer: Tensor,
    basis_state: Tensor,
    momentum_t: Tensor,
    lr_t: Tensor,
    wd_t: Tensor,
    ridge_epsilon: float,
    muon_eps: float,
) -> Tensor:
    """Fully fused StreamingMuon step with num_iters=1."""
    mt = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - mt)
    g = stacked_grads.lerp_(momentum_buffer, mt)

    working = g.float()
    basis = basis_state.float()

    gram_matrix = torch.matmul(working.mT, working)

    # 1 iteration of dual SCQR
    first_pass = torch.matmul(gram_matrix, basis)
    first_gram = torch.matmul(basis.mT, first_pass)
    second_pass = _scqr_fused(first_pass, first_gram, ridge_epsilon)
    second_gram = torch.matmul(second_pass.mT, second_pass)
    basis = _scqr_fused(second_pass, second_gram, ridge_epsilon)

    rotated = torch.matmul(working, basis)
    sigma = torch.linalg.norm(rotated, dim=-2).clamp(min=muon_eps)
    left_vectors = rotated / sigma.unsqueeze(-2)
    update = torch.matmul(left_vectors, basis.mT)

    lr = lr_t.to(update.dtype)
    wd = wd_t.to(update.dtype)
    stacked_params.sub_(lr * update + lr * wd * stacked_params)

    return basis


# Polar Express coefficients (from nanochat/optim.py, ns_steps=5)
_POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


@torch.compile(dynamic=False, fullgraph=True)
def _streaming_muon_fused_1iter_pe(
    stacked_grads: Tensor,
    stacked_params: Tensor,
    momentum_buffer: Tensor,
    basis_state: Tensor,
    momentum_t: Tensor,
    lr_t: Tensor,
    wd_t: Tensor,
    ridge_epsilon: float,
    muon_eps: float,
) -> Tensor:
    """Streaming muon + polar express cleanup. Tests whether close-singular-value
    drift in the basis is the root cause of divergence by snapping the output
    back toward the polar factor with native muon's NS5 coefficients."""
    mt = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - mt)
    g = stacked_grads.lerp_(momentum_buffer, mt)

    working = g.float()
    basis = basis_state.float()

    gram_matrix = torch.matmul(working.mT, working)

    first_pass = torch.matmul(gram_matrix, basis)
    first_gram = torch.matmul(basis.mT, first_pass)
    second_pass = _scqr_fused(first_pass, first_gram, ridge_epsilon)
    second_gram = torch.matmul(second_pass.mT, second_pass)
    basis = _scqr_fused(second_pass, second_gram, ridge_epsilon)

    rotated = torch.matmul(working, basis)
    sigma = torch.linalg.norm(rotated, dim=-2).clamp(min=muon_eps)
    left_vectors = rotated / sigma.unsqueeze(-2)
    update = torch.matmul(left_vectors, basis.mT)

    # Polar express cleanup — ensure spectral norm ≈ 1 even if basis is wrong
    # Normalize input (polar express requires spec norm ≤ 1)
    X = update / (update.flatten(-2, -1).norm(dim=-1, keepdim=True).unsqueeze(-1) * 1.01 + 1e-6)
    # Fast path always sees n >= m (transposed), so matrix is tall
    for a, b, c in _POLAR_EXPRESS_COEFFS:
        A = X.mT @ X
        B = b * A + c * (A @ A)
        X = a * X + X @ B
    update = X

    lr = lr_t.to(update.dtype)
    wd = wd_t.to(update.dtype)
    stacked_params.sub_(lr * update + lr * wd * stacked_params)

    return basis


@torch.compile(dynamic=False, fullgraph=True)
def _streaming_muon_fused_1iter_fnorm(
    stacked_grads: Tensor,
    stacked_params: Tensor,
    momentum_buffer: Tensor,
    basis_state: Tensor,
    momentum_t: Tensor,
    lr_t: Tensor,
    wd_t: Tensor,
    ridge_epsilon: float,
    muon_eps: float,
    expected_fnorm: float,
) -> Tensor:
    """Same as _streaming_muon_fused_1iter but enforces Frobenius norm of update.
    Fixes divergence at high lr by bounding the step magnitude to UV^T scale."""
    mt = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - mt)
    g = stacked_grads.lerp_(momentum_buffer, mt)

    working = g.float()
    basis = basis_state.float()

    gram_matrix = torch.matmul(working.mT, working)

    first_pass = torch.matmul(gram_matrix, basis)
    first_gram = torch.matmul(basis.mT, first_pass)
    second_pass = _scqr_fused(first_pass, first_gram, ridge_epsilon)
    second_gram = torch.matmul(second_pass.mT, second_pass)
    basis = _scqr_fused(second_pass, second_gram, ridge_epsilon)

    rotated = torch.matmul(working, basis)
    sigma = torch.linalg.norm(rotated, dim=-2).clamp(min=muon_eps)
    left_vectors = rotated / sigma.unsqueeze(-2)
    update = torch.matmul(left_vectors, basis.mT)

    # Enforce expected Frobenius norm of UV^T (= sqrt(min(m,n))) per matrix in batch.
    fnorm = update.flatten(-2, -1).norm(dim=-1, keepdim=True).unsqueeze(-1).clamp(min=muon_eps)
    update = update * (expected_fnorm / fnorm)

    lr = lr_t.to(update.dtype)
    wd = wd_t.to(update.dtype)
    stacked_params.sub_(lr * update + lr * wd * stacked_params)

    return basis


@torch.compile(dynamic=False, fullgraph=True)
def _streaming_muon_fused_1iter_qr(
    stacked_grads: Tensor,
    stacked_params: Tensor,
    momentum_buffer: Tensor,
    basis_state: Tensor,
    momentum_t: Tensor,
    lr_t: Tensor,
    wd_t: Tensor,
    ridge_epsilon: float,
    muon_eps: float,
) -> Tensor:
    """Streaming muon (num_iters=1) with left_vectors obtained via SCQR instead
    of column-normalization. Guarantees update = left_vectors @ basis^T has
    orthonormal factors (spec norm = 1) even when basis has not converged
    (close singular values, per 苏剑林 Part 4 warning about power iteration
    failure at σ_i ≈ σ_{i+1}).
    """
    mt = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - mt)
    g = stacked_grads.lerp_(momentum_buffer, mt)

    working = g.float()
    basis = basis_state.float()

    gram_matrix = torch.matmul(working.mT, working)

    first_pass = torch.matmul(gram_matrix, basis)
    first_gram = torch.matmul(basis.mT, first_pass)
    second_pass = _scqr_fused(first_pass, first_gram, ridge_epsilon)
    second_gram = torch.matmul(second_pass.mT, second_pass)
    basis = _scqr_fused(second_pass, second_gram, ridge_epsilon)

    # rotated = M V ≈ U Σ (orthogonal columns only if basis = true V)
    rotated = torch.matmul(working, basis)
    # True orthogonalization via SCQR — not column-norm — to guarantee
    # orthonormal left_vectors even when basis is imperfect.
    rotated_gram = torch.matmul(rotated.mT, rotated)
    left_vectors = _scqr_fused(rotated, rotated_gram, ridge_epsilon)

    update = torch.matmul(left_vectors, basis.mT)

    lr = lr_t.to(update.dtype)
    wd = wd_t.to(update.dtype)
    stacked_params.sub_(lr * update + lr * wd * stacked_params)

    return basis


@torch.compile(dynamic=False, fullgraph=True)
def _streaming_muon_fused_2iter(
    stacked_grads: Tensor,
    stacked_params: Tensor,
    momentum_buffer: Tensor,
    basis_state: Tensor,
    momentum_t: Tensor,
    lr_t: Tensor,
    wd_t: Tensor,
    ridge_epsilon: float,
    muon_eps: float,
) -> Tensor:
    """
    Fully fused StreamingMuon step (num_iters=2) with SCQR orthogonalization.

    Algorithm (苏剑林 Part 3 reordered dual orthogonalization):
        gram = M^T M                            # O(nm²), ONCE (Part 3 trick)
        for i in range(2):
            first_pass = gram @ basis
            first_gram = basis^T @ first_pass
            second_pass = SCQR(first_pass, first_gram)   # inner (Part 2 dual)
            basis = SCQR(second_pass)                    # outer
        rotated = M @ basis
        update = ColNorm(rotated) @ basis^T

    All fp32 for numerical stability (SCQR requires fp32 Cholesky).
    Fused into a single compiled graph to minimize Python/kernel-launch overhead.
    """
    mt = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - mt)
    g = stacked_grads.lerp_(momentum_buffer, mt)

    working = g.float()
    basis = basis_state.float()

    # Gram: M^T M (ONCE, reused across iterations)
    gram_matrix = torch.matmul(working.mT, working)

    # Iteration 1 (dual SCQR)
    first_pass = torch.matmul(gram_matrix, basis)
    first_gram = torch.matmul(basis.mT, first_pass)
    second_pass = _scqr_fused(first_pass, first_gram, ridge_epsilon)
    second_gram = torch.matmul(second_pass.mT, second_pass)
    basis = _scqr_fused(second_pass, second_gram, ridge_epsilon)

    # Iteration 2 (gram reused)
    first_pass = torch.matmul(gram_matrix, basis)
    first_gram = torch.matmul(basis.mT, first_pass)
    second_pass = _scqr_fused(first_pass, first_gram, ridge_epsilon)
    second_gram = torch.matmul(second_pass.mT, second_pass)
    basis = _scqr_fused(second_pass, second_gram, ridge_epsilon)

    # msign: U @ V^T where U = ColNorm(M V)
    rotated = torch.matmul(working, basis)
    sigma = torch.linalg.norm(rotated, dim=-2).clamp(min=muon_eps)
    left_vectors = rotated / sigma.unsqueeze(-2)
    update = torch.matmul(left_vectors, basis.mT)

    lr = lr_t.to(update.dtype)
    wd = wd_t.to(update.dtype)
    stacked_params.sub_(lr * update + lr * wd * stacked_params)

    return basis


def _orthogonalize_columns(
    matrix: Tensor,
    *,
    gram: Optional[Tensor] = None,
    ridge_epsilon: float = 1e-9,
    fallback_to_qr: bool = True,
    diagonal_regularization: bool = True,
    fallback_orthogonality_tol: Optional[float] = None,
    pure_qr: bool = False,
) -> Tensor:
    """
    Orthogonalize columns via Shifted Cholesky QR with fallback to Householder QR.
    Handles batched input: (..., n, m).

    If pure_qr=True, skips SCQR entirely and uses Householder QR. Used to
    diagnose whether SCQR numerical drift is the cause of divergence.

    If fallback_orthogonality_tol is set and SCQR's orthogonality error exceeds
    it, falls back to Householder QR for accuracy.
    """
    if pure_qr:
        return _qr_positive_diagonal(matrix.float()).to(matrix.dtype)

    if gram is None:
        gram = torch.matmul(matrix.mT.float(), matrix.float())

    try:
        scqr_q = _scqr_once(matrix, gram, ridge_epsilon, diagonal_regularization)

        if not torch.all(torch.isfinite(scqr_q)):
            raise RuntimeError("SCQR non-finite")

        if fallback_orthogonality_tol is not None:
            max_err = _orthogonality_error(scqr_q).max()
            if max_err > fallback_orthogonality_tol:
                raise RuntimeError(f"SCQR ortho err {max_err:.4f} > {fallback_orthogonality_tol}")

        return scqr_q.to(matrix.dtype)

    except (RuntimeError, torch.linalg.LinAlgError):
        if not fallback_to_qr:
            return _qr_positive_diagonal(matrix.float()).to(matrix.dtype)
        return _qr_positive_diagonal(matrix.float()).to(matrix.dtype)


# =============================================================================
# Core: Streaming matrix sign with f(Σ) support
# =============================================================================

def streaming_msign(
    matrix: Tensor,
    previous_basis: Optional[Tensor],
    *,
    steps: int = 1,
    muon_eps: float = 1e-8,
    ridge_epsilon: float = 1e-9,
    fallback_to_qr: bool = True,
    diagonal_regularization: bool = True,
    fallback_orthogonality_tol: Optional[float] = None,
    sigma_transform: Optional[SigmaTransform] = None,
    sigma_state: Optional[dict] = None,
    poly_order: int = 1,
    qr_left_vectors: bool = False,
    input_normalize: bool = False,
    sigma_spread: float = 0.0,
    pure_qr: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Streaming power iteration to compute msign(M) with optional f(Σ) transform.

    Algorithm (苏剑林 Part 3 reordered dual orthogonalization):
        gram = M^T M                              # O(nm²), computed ONCE
        For each step:
            first_pass = gram^poly_order @ basis   # polynomial acceleration
            first_gram = basis^T @ first_pass      # precomputed gram for SCQR
            second_pass = SCQR(first_pass, gram=first_gram)
            basis = SCQR(second_pass)
        rotated = M @ basis
        sigma = column_norms(rotated)
        update = U @ diag(f(sigma)) @ basis^T

    Optimizations over basic implementation:
        1. Gram reuse: M^T M computed once, reused across all steps (O(nm²) → 1x)
        2. Polynomial acceleration: gram^p @ V gives convergence of p power iterations
           per double-QR, at cost of (p-1) extra O(m³) matmuls (cheap when n >> m)
        3. Diagonal regularization exploiting V^T (M^T M) V ≈ diag (Part 3)

    Args:
        matrix: (..., n, m) input matrix
        previous_basis: (..., basis_dim, basis_dim) warm-started basis
        steps: number of double-QR iterations (default 1)
        poly_order: polynomial degree for acceleration (default 1).
            poly_order=2 with steps=1 ≈ steps=2 convergence at ~half the QR cost.
        sigma_transform: optional f(Σ) transform
        sigma_state: mutable dict for f(Σ) state
    Returns:
        update: (..., n, m) the update direction
        new_basis: (..., basis_dim, basis_dim) updated basis
        sigma: (..., basis_dim) singular values
    """
    # Pure fp32 unfused path: stable, ~30ms/step on d8
    # (Matmul-only and bf16 fast paths were tried but have precision issues)
    working = matrix.float()
    transposed = working.shape[-2] < working.shape[-1]
    if transposed:
        working = working.mT  # now n >= m

    # Optional: normalize input by Frobenius norm (matches native muon's
    # polar express convention). Keeps gram entries O(1) regardless of ‖g‖.
    if input_normalize:
        fnorm = working.flatten(-2, -1).norm(dim=-1, keepdim=True).unsqueeze(-1)
        working = working / (fnorm * 1.01 + 1e-6)

    basis_dim = working.shape[-1]
    batch_shape = working.shape[:-2]

    if previous_basis is None or previous_basis.shape[-1] != basis_dim or previous_basis.shape[-2] != basis_dim:
        basis = torch.eye(basis_dim, dtype=torch.float32, device=working.device)
        if batch_shape:
            basis = basis.expand(*batch_shape, -1, -1).contiguous()
    else:
        basis = previous_basis.float()

    # Compute gram ONCE — the O(nm²) bottleneck (Part 3 optimization)
    gram_matrix = torch.matmul(working.mT, working)  # (..., m, m)

    for _ in range(steps):
        first_pass = torch.matmul(gram_matrix, basis)
        for _ in range(poly_order - 1):
            first_pass = torch.matmul(gram_matrix, first_pass)
        first_gram = torch.matmul(basis.mT, first_pass)
        second_pass = _orthogonalize_columns(
            first_pass, gram=first_gram,
            ridge_epsilon=ridge_epsilon, fallback_to_qr=fallback_to_qr,
            diagonal_regularization=diagonal_regularization,
            fallback_orthogonality_tol=None,
            pure_qr=pure_qr,
        )
        basis = _orthogonalize_columns(
            second_pass, gram=None,
            ridge_epsilon=ridge_epsilon, fallback_to_qr=fallback_to_qr,
            diagonal_regularization=diagonal_regularization,
            fallback_orthogonality_tol=fallback_orthogonality_tol,
            pure_qr=pure_qr,
        )

    rotated = torch.matmul(working, basis)  # (..., n, m) = M @ V ≈ U Σ

    # Singular values = column norms of rotated (for diagnostics and f(Σ) input)
    sigma = torch.linalg.norm(rotated, dim=-2).clamp(min=muon_eps)

    # Left singular vectors. Two variants:
    #   column-norm (default): fast, but only orthonormal if basis = true V
    #   SCQR (qr_left_vectors): guarantees orthonormality even when power iter
    #     hasn't converged (close singular values, per 苏剑林 Part 4)
    if qr_left_vectors or pure_qr:
        left_vectors = _orthogonalize_columns(
            rotated, gram=None,
            ridge_epsilon=ridge_epsilon, fallback_to_qr=fallback_to_qr,
            diagonal_regularization=diagonal_regularization,
            fallback_orthogonality_tol=None,
            pure_qr=pure_qr,
        )
    else:
        left_vectors = rotated / sigma.unsqueeze(-2)

    # Compute per-column scaling
    if sigma_transform is not None and sigma_state is not None:
        scaling = sigma_transform(sigma, sigma_state)
        scaling = torch.where(torch.isfinite(scaling), scaling, torch.ones_like(scaling))
    else:
        scaling = torch.ones_like(sigma)

    if sigma_spread > 0.0:
        # Simulate polar express's σ ∈ [1-s, 1+s] per-direction noise.
        noise = torch.empty_like(sigma).uniform_(1.0 - sigma_spread, 1.0 + sigma_spread)
        scaling = scaling * noise

    update = torch.matmul(left_vectors * scaling.unsqueeze(-2), basis.mT)

    if transposed:
        update = update.mT

    return update.to(matrix.dtype), basis, sigma


# =============================================================================
# StreamingMuonAdamW: drop-in single-GPU optimizer (replaces nanochat MuonAdamW)
# =============================================================================

class StreamingMuonAdamW(torch.optim.Optimizer):
    """
    Combined optimizer: StreamingMuon for 2D matrix params, AdamW for others.
    Drop-in replacement for nanochat's MuonAdamW.

    Param groups use 'kind': 'adamw', 'streaming_muon', or 'muon'.
    'muon' is auto-upgraded to 'streaming_muon' with identity transform.
    """

    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        # Streaming muon 0-D CPU tensors (avoid recompilation on value change)
        self._sm_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._sm_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._sm_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _get_sigma_transform(self, group: dict) -> SigmaTransform:
        st = group.get('sigma_transform', 'identity')
        kwargs = group.get('sigma_transform_kwargs', {})
        if callable(st):
            return st
        if st in SIGMA_TRANSFORMS:
            return SIGMA_TRANSFORMS[st](**kwargs)
        raise ValueError(f"Unknown sigma_transform: {st}")

    def _step_adamw(self, group: dict) -> None:
        from nanochat.optim import adamw_step_fused
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(
                p, grad, state['exp_avg'], state['exp_avg_sq'],
                self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t,
            )

    def _step_streaming_muon(self, group: dict) -> None:
        params: list[Tensor] = group['params']
        if not params:
            return

        p0 = params[0]
        state = self.state[p0]
        num_params = len(params)
        shape, device, dtype = p0.shape, p0.device, p0.dtype
        n, m = shape[-2], shape[-1]

        # Effective dimension for basis
        effective_dim = min(n, m)

        # Initialize state
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "basis_state" not in state:
            state["basis_state"] = torch.eye(effective_dim, dtype=torch.float32, device=device) \
                .unsqueeze(0).expand(num_params, -1, -1).clone()
        if "sigma_state" not in state:
            state["sigma_state"] = {}
        if "sigma_transform_obj" not in state:
            state["sigma_transform_obj"] = self._get_sigma_transform(group)
        if "step_count" not in state:
            state["step_count"] = 0

        momentum_buffer = state["momentum_buffer"]
        basis_state = state["basis_state"]
        sigma_state = state["sigma_state"]
        sigma_transform = state["sigma_transform_obj"]

        # Optional: reset basis to identity on specified interval (diagnostic for warm-start drift)
        basis_reset_every = group.get("basis_reset_every", 0)
        if basis_reset_every > 0 and state["step_count"] > 0 and state["step_count"] % basis_reset_every == 0:
            basis_state.copy_(torch.eye(effective_dim, dtype=torch.float32, device=device)
                              .unsqueeze(0).expand(num_params, -1, -1))

        # Sigma history: ring buffer of shape (window, num_params, k)
        # Automatically maintained; f can access via state['sigma_history']
        sigma_window = group.get("sigma_window", 100)

        # Stack grads and params
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)

        momentum_val = group["momentum"]
        lr = group["lr"] * max(1.0, n / m) ** 0.5
        wd = group["weight_decay"]
        num_iters = group.get("num_iters", 1)
        poly_order = group.get("poly_order", 1)
        ridge_epsilon = group.get("ridge_epsilon", 1e-9)
        # Default tol=0.02: SCQR numerical drift accumulates in the warm-started
        # basis and causes divergence after ~3000 steps at matrix_lr=0.02
        # (validated on d8/1B). tol=0.02 catches the drift early and falls back
        # to Householder QR for those steps; tol>=0.05 is not safe. Pass
        # fallback_orthogonality_tol=None in the group dict to opt into the
        # fused SCQR fast path (unsafe at lr=0.02 long-horizon training).
        fallback_orthogonality_tol = group.get("fallback_orthogonality_tol", 0.02)

        # Fast path: fully-fused compiled step (all fp32 SCQR in one graph)
        # Eliminates Python overhead between SCQR calls. Falls through to
        # unfused path if Cholesky fails or conditions not met.
        pure_qr = group.get("pure_qr", False)
        use_fast_path = (
            num_iters in (1, 2)
            and poly_order == 1
            and fallback_orthogonality_tol is None
            and isinstance(sigma_transform, IdentityTransform)
            and device.type == 'cuda'
            and not pure_qr
            and not group.get("_capture_metrics", False)
        )

        if use_fast_path:
            self._sm_momentum_t.fill_(momentum_val)
            self._sm_lr_t.fill_(lr)
            self._sm_wd_t.fill_(wd)
            # Transpose wide matrices so fused path always sees n >= m
            if n < m:
                stacked_grads = stacked_grads.mT.contiguous()
                stacked_params_t = stacked_params.mT.contiguous()
                momentum_buffer_orig = momentum_buffer
                if state.get("_transposed_mb") is None:
                    state["_transposed_mb"] = torch.zeros_like(stacked_grads)
                mb_t = state["_transposed_mb"]
            else:
                stacked_params_t = stacked_params
                mb_t = momentum_buffer
            try:
                fnorm_update = group.get("fnorm_update", False)
                pe_cleanup = group.get("pe_cleanup", False)
                qr_left = group.get("qr_left_vectors", False)
                if qr_left and num_iters == 1:
                    new_basis = _streaming_muon_fused_1iter_qr(
                        stacked_grads, stacked_params_t, mb_t, basis_state,
                        self._sm_momentum_t, self._sm_lr_t, self._sm_wd_t,
                        ridge_epsilon, 1e-8,
                    )
                elif pe_cleanup and num_iters == 1:
                    new_basis = _streaming_muon_fused_1iter_pe(
                        stacked_grads, stacked_params_t, mb_t, basis_state,
                        self._sm_momentum_t, self._sm_lr_t, self._sm_wd_t,
                        ridge_epsilon, 1e-8,
                    )
                elif fnorm_update and num_iters == 1:
                    m_eff, n_eff = stacked_grads.shape[-2], stacked_grads.shape[-1]
                    expected_fnorm = float(min(m_eff, n_eff)) ** 0.5
                    new_basis = _streaming_muon_fused_1iter_fnorm(
                        stacked_grads, stacked_params_t, mb_t, basis_state,
                        self._sm_momentum_t, self._sm_lr_t, self._sm_wd_t,
                        ridge_epsilon, 1e-8, expected_fnorm,
                    )
                else:
                    fused_fn = _streaming_muon_fused_1iter if num_iters == 1 else _streaming_muon_fused_2iter
                    new_basis = fused_fn(
                        stacked_grads, stacked_params_t, mb_t, basis_state,
                        self._sm_momentum_t, self._sm_lr_t, self._sm_wd_t,
                        ridge_epsilon, 1e-8,
                    )
                state["basis_state"] = new_basis
                if n < m:
                    stacked_params.copy_(stacked_params_t.mT)
                torch._foreach_copy_(params, list(stacked_params.unbind(0)))
                return
            except (RuntimeError, torch.linalg.LinAlgError):
                pass

        # Slow path: unfused, supports transpose / poly_order / ortho check / f(Σ)
        capture_metrics = bool(group.get("_capture_metrics", False))
        metrics_raw_grad = stacked_grads.detach().clone() if capture_metrics else None
        momentum_buffer.lerp_(stacked_grads, 1 - momentum_val)
        g = stacked_grads.lerp_(momentum_buffer, momentum_val)

        update, new_basis, sigma = streaming_msign(
            g, basis_state,
            steps=num_iters,
            poly_order=poly_order,
            ridge_epsilon=ridge_epsilon,
            fallback_to_qr=True,
            diagonal_regularization=True,
            fallback_orthogonality_tol=fallback_orthogonality_tol,
            sigma_transform=sigma_transform,
            sigma_state=sigma_state,
            qr_left_vectors=group.get("qr_left_vectors", False),
            pure_qr=pure_qr,
        )
        state["basis_state"] = new_basis
        if capture_metrics:
            state["metrics_cache"] = {
                "param_start_idx": 0,
                "num_owned": num_params,
                "grad": metrics_raw_grad,
                "momentum": momentum_buffer.detach().clone(),
                "momentum_after_nesterov": g.detach().clone(),
                "sigma": sigma.detach().clone(),
            }

        # Auto-track sigma history as ring buffer in sigma_state
        # f can access: state['sigma_history'] (T, num_params, k), state['step']
        state["step_count"] += 1
        sigma_state['step'] = state["step_count"]
        if 'sigma_history' not in sigma_state:
            sigma_state['sigma_history'] = sigma.unsqueeze(0).clone()
        else:
            hist = sigma_state['sigma_history']
            if hist.shape[0] < sigma_window:
                sigma_state['sigma_history'] = torch.cat([hist, sigma.unsqueeze(0)], dim=0)
            else:
                # Ring buffer: shift and append
                sigma_state['sigma_history'] = torch.cat([hist[1:], sigma.unsqueeze(0)], dim=0)
        stacked_params.sub_(lr * update + lr * wd * stacked_params)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            kind = group['kind']
            if kind == 'adamw':
                self._step_adamw(group)
            elif kind == 'streaming_muon':
                self._step_streaming_muon(group)
            elif kind == 'muon':
                group['kind'] = 'streaming_muon'
                group.setdefault('sigma_transform', 'identity')
                self._step_streaming_muon(group)
            else:
                raise ValueError(f"Unknown optimizer kind: {kind}")


# =============================================================================
# DistStreamingMuonAdamW: distributed multi-GPU version
# =============================================================================

class DistStreamingMuonAdamW(torch.optim.Optimizer):
    """
    Distributed StreamingMuonAdamW: 3-phase async communication.
    Drop-in replacement for nanochat's DistMuonAdamW.
    """

    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _get_sigma_transform(self, group: dict) -> SigmaTransform:
        st = group.get('sigma_transform', 'identity')
        kwargs = group.get('sigma_transform_kwargs', {})
        if callable(st):
            return st
        if st in SIGMA_TRANSFORMS:
            return SIGMA_TRANSFORMS[st](**kwargs)
        raise ValueError(f"Unknown sigma_transform: {st}")

    # --- Phase 1: launch async reduces ---

    def _reduce_adamw(self, group, world_size):
        import torch.distributed as dist
        param_infos = {}
        for p in group['params']:
            grad = p.grad
            if p.numel() < 1024:
                future = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                param_infos[p] = dict(future=future, grad_slice=grad, is_small=True)
            else:
                assert grad.shape[0] % world_size == 0
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                future = dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                param_infos[p] = dict(future=future, grad_slice=grad_slice, is_small=False)
        return dict(param_infos=param_infos)

    def _reduce_streaming_muon(self, group, world_size):
        import torch.distributed as dist
        params = group['params']
        chunk_size = (len(params) + world_size - 1) // world_size
        padded_num_params = chunk_size * world_size
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype

        grad_stack = torch.stack([p.grad for p in params])
        stacked_grads = torch.empty(padded_num_params, *shape, dtype=dtype, device=device)
        stacked_grads[:len(params)].copy_(grad_stack)
        if len(params) < padded_num_params:
            stacked_grads[len(params):].zero_()

        grad_chunk = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
        future = dist.reduce_scatter_tensor(grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True).get_future()
        return dict(future=future, grad_chunk=grad_chunk, stacked_grads=stacked_grads, chunk_size=chunk_size)

    # --- Phase 2: compute updates ---

    def _compute_adamw(self, group, info, gather_list, rank, world_size):
        import torch.distributed as dist
        from nanochat.optim import adamw_step_fused
        param_infos = info['param_infos']
        for p in group['params']:
            pinfo = param_infos[p]
            pinfo['future'].wait()
            grad_slice = pinfo['grad_slice']
            state = self.state[p]
            if pinfo['is_small']:
                p_slice = p
            else:
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p_slice)
                state['exp_avg_sq'] = torch.zeros_like(p_slice)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(
                p_slice, grad_slice, state['exp_avg'], state['exp_avg_sq'],
                self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t,
            )
            if not pinfo['is_small']:
                future = dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future()
                gather_list.append(dict(future=future, params=None))

    def _compute_streaming_muon(self, group, info, gather_list, rank):
        import torch.distributed as dist
        info['future'].wait()
        params = group['params']
        chunk_size = info['chunk_size']
        grad_chunk = info['grad_chunk']
        p0 = params[0]
        shape, device, dtype = p0.shape, p0.device, p0.dtype
        n, m = shape[-2], shape[-1]

        start_idx = rank * chunk_size
        num_owned = min(chunk_size, max(0, len(params) - start_idx))

        effective_dim = min(n, m)
        num_iters = group.get("num_iters", 1)
        poly_order = group.get("poly_order", 1)
        ridge_epsilon = group.get("ridge_epsilon", 1e-9)
        # Default tol=0.02: SCQR numerical drift accumulates in the warm-started
        # basis and causes divergence after ~3000 steps at matrix_lr=0.02
        # (validated on d8/1B). tol=0.02 catches the drift early and falls back
        # to Householder QR for those steps; tol>=0.05 is not safe. Pass
        # fallback_orthogonality_tol=None in the group dict to opt into the
        # fused SCQR fast path (unsafe at lr=0.02 long-horizon training).
        fallback_orthogonality_tol = group.get("fallback_orthogonality_tol", 0.02)

        state = self.state[p0]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(chunk_size, *shape, dtype=dtype, device=device)
        if "basis_state" not in state:
            state["basis_state"] = torch.eye(effective_dim, dtype=torch.float32, device=device) \
                .unsqueeze(0).expand(chunk_size, -1, -1).clone()
        if "sigma_state" not in state:
            state["sigma_state"] = {}
        if "sigma_transform_obj" not in state:
            state["sigma_transform_obj"] = self._get_sigma_transform(group)

        momentum_buffer = state["momentum_buffer"]
        basis_state = state["basis_state"]
        sigma_state = state["sigma_state"]
        sigma_transform = state["sigma_transform_obj"]

        # Optional: reset basis periodically (diagnostic for warm-start drift)
        basis_reset_every = group.get("basis_reset_every", 0)
        step_count = sigma_state.get('step', 0)
        if basis_reset_every > 0 and step_count > 0 and step_count % basis_reset_every == 0:
            basis_state.copy_(torch.eye(effective_dim, dtype=torch.float32, device=device)
                              .unsqueeze(0).expand(chunk_size, -1, -1))

        updated_params = torch.empty(chunk_size, *shape, dtype=dtype, device=device)

        if num_owned > 0:
            owned_params = [params[start_idx + i] for i in range(num_owned)]
            stacked_owned = torch.stack(owned_params)

            momentum_val = group["momentum"]
            lr = group["lr"] * max(1.0, n / m) ** 0.5
            wd = group["weight_decay"]

            mb = momentum_buffer[:num_owned]
            capture_metrics = bool(group.get("_capture_metrics", False))
            metrics_raw_grad = grad_chunk[:num_owned].detach().clone() if capture_metrics else None
            mb.lerp_(grad_chunk[:num_owned], 1 - momentum_val)
            g = grad_chunk[:num_owned].lerp_(mb, momentum_val)

            update, new_basis, sigma = streaming_msign(
                g, basis_state[:num_owned],
                steps=num_iters,
                poly_order=poly_order,
                ridge_epsilon=ridge_epsilon,
                fallback_to_qr=True,
                diagonal_regularization=True,
                fallback_orthogonality_tol=fallback_orthogonality_tol,
                sigma_transform=sigma_transform,
                sigma_state=sigma_state,
                qr_left_vectors=group.get("qr_left_vectors", False),
                input_normalize=group.get("input_normalize", False),
                sigma_spread=group.get("sigma_spread", 0.0),
                pure_qr=group.get("pure_qr", False),
            )
            basis_state[:num_owned].copy_(new_basis)
            if capture_metrics:
                state["metrics_cache"] = {
                    "param_start_idx": start_idx,
                    "num_owned": num_owned,
                    "grad": metrics_raw_grad,
                    "momentum": mb.detach().clone(),
                    "momentum_after_nesterov": g.detach().clone(),
                    "sigma": sigma.detach().clone(),
                }

            # Optional: Frobenius-normalize update to expected UV^T scale
            if group.get("fnorm_update", False):
                expected_fnorm = float(min(n, m)) ** 0.5
                fnorm = update.flatten(-2, -1).norm(dim=-1, keepdim=True).unsqueeze(-1).clamp(min=1e-8)
                update = update * (expected_fnorm / fnorm)

            # Optional: Polar Express cleanup (robust to close/equal σ where power iter fails)
            if group.get("pe_cleanup", False):
                X = update.float()
                X = X / (X.flatten(-2, -1).norm(dim=-1, keepdim=True).unsqueeze(-1) * 1.01 + 1e-6)
                tall = X.shape[-2] >= X.shape[-1]
                for a, b, c in _POLAR_EXPRESS_COEFFS:
                    if tall:
                        A = torch.matmul(X.mT, X)
                        B = b * A + c * torch.matmul(A, A)
                        X = a * X + torch.matmul(X, B)
                    else:
                        A = torch.matmul(X, X.mT)
                        B = b * A + c * torch.matmul(A, A)
                        X = a * X + torch.matmul(B, X)
                update = X.to(stacked_owned.dtype)

            stacked_owned.sub_(lr * update + lr * wd * stacked_owned)
            updated_params[:num_owned].copy_(stacked_owned)

        if num_owned < chunk_size:
            updated_params[num_owned:].zero_()
            if group.get("_capture_metrics", False) and num_owned == 0:
                state["metrics_cache"] = {
                    "param_start_idx": start_idx,
                    "num_owned": 0,
                    "grad": None,
                    "momentum": None,
                    "momentum_after_nesterov": None,
                    "sigma": None,
                }

        stacked_params = info["stacked_grads"]
        future = dist.all_gather_into_tensor(stacked_params, updated_params, async_op=True).get_future()
        gather_list.append(dict(future=future, stacked_params=stacked_params, params=params))

    # --- Phase 3: finish gathers ---

    def _finish_gathers(self, gather_list):
        for info in gather_list:
            info["future"].wait()
            if info["params"] is not None:
                torch._foreach_copy_(info["params"], list(info["stacked_params"][:len(info["params"])].unbind(0)))

    @torch.no_grad()
    def step(self):
        import torch.distributed as dist
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        reduce_infos = []
        for group in self.param_groups:
            kind = group['kind']
            if kind == 'muon':
                group['kind'] = 'streaming_muon'
                group.setdefault('sigma_transform', 'identity')
            if group['kind'] == 'adamw':
                reduce_infos.append(self._reduce_adamw(group, world_size))
            elif group['kind'] == 'streaming_muon':
                reduce_infos.append(self._reduce_streaming_muon(group, world_size))
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

        gather_list = []
        for group, info in zip(self.param_groups, reduce_infos):
            if group['kind'] == 'adamw':
                self._compute_adamw(group, info, gather_list, rank, world_size)
            elif group['kind'] == 'streaming_muon':
                self._compute_streaming_muon(group, info, gather_list, rank)

        self._finish_gathers(gather_list)


# =============================================================================
# Utilities
# =============================================================================

def patch_setup_optimizer(model, sigma_transform='identity', sigma_transform_kwargs=None, k=None):
    """Monkey-patch a nanochat GPT model to use StreamingMuon."""
    original_setup = model.setup_optimizer.__func__

    def new_setup_optimizer(self, **kwargs):
        optimizer = original_setup(self, **kwargs)
        new_param_groups = []
        for group in optimizer.param_groups:
            if group['kind'] == 'muon':
                group['kind'] = 'streaming_muon'
                group['sigma_transform'] = sigma_transform
                group['sigma_transform_kwargs'] = sigma_transform_kwargs or {}
                if k is not None:
                    group['k'] = k
            new_param_groups.append(group)
        return StreamingMuonAdamW(new_param_groups)

    import types
    model.setup_optimizer = types.MethodType(new_setup_optimizer, model)


def get_spectral_diagnostics(optimizer) -> dict:
    """Extract spectral diagnostics from a StreamingMuonAdamW optimizer."""
    diagnostics = {'sigma': [], 'scaling': [], 'sigma_stats': []}

    for group in optimizer.param_groups:
        if group['kind'] != 'streaming_muon':
            continue
        p0 = group['params'][0]
        state = optimizer.state[p0]
        if 'basis_state' not in state:
            continue

        basis = state['basis_state']
        momentum_buffer = state['momentum_buffer']

        # Recompute sigma
        transposed = p0.shape[-2] < p0.shape[-1]
        M = momentum_buffer.mT if transposed else momentum_buffer
        rotated = torch.matmul(M.float(), basis.float())
        sigma = torch.linalg.norm(rotated, dim=-2).clamp(min=1e-8)

        sigma_transform = state.get('sigma_transform_obj', IdentityTransform())
        sigma_state = state.get('sigma_state', {})
        scaling = sigma_transform(sigma, sigma_state)

        diagnostics['sigma'].append(sigma.detach().cpu())
        diagnostics['scaling'].append(scaling.detach().cpu())
        diagnostics['sigma_stats'].append({
            'mean': sigma.mean().item(),
            'max': sigma.max().item(),
            'min': sigma.min().item(),
            'median': sigma.median().item(),
        })

    return diagnostics
