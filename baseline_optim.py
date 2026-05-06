"""Baseline optimizers for the clean nanochat optimizer-sweep runner.

This file intentionally prioritizes correctness and handoff clarity over the
ZeRO-style sharding used by StreamingMuon. In distributed runs, gradients are
averaged once and every rank applies the same replicated update.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.distributed as dist
from torch import Tensor


STRUCTURED_KINDS = {"soap", "shampoo", "kl_shampoo", "kl_soap"}
BASELINE_KINDS = STRUCTURED_KINDS | {"plain_muon"}


def _finite_power(values: Tensor, exponent: float, eps: float) -> Tensor:
    return values.clamp_min(eps).pow(exponent).nan_to_num_(0.0, posinf=0.0, neginf=0.0)


def _eigh_basis(matrix: Tensor, eps: float) -> tuple[Tensor, Tensor]:
    matrix_f = matrix.float()
    eye = torch.eye(matrix_f.shape[0], device=matrix_f.device, dtype=matrix_f.dtype)
    vals, vecs = torch.linalg.eigh(matrix_f + eps * eye)
    vals = torch.flip(vals, dims=[0]).contiguous()
    vecs = torch.flip(vecs, dims=[1]).contiguous()
    return vecs, vals


def _qr_basis(matrix: Tensor, old_q: Tensor) -> Tensor:
    q, r = torch.linalg.qr(matrix.float() @ old_q.float(), mode="reduced")
    signs = torch.sign(torch.diagonal(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return (q * signs.unsqueeze(0)).contiguous()


def _project_2d(x: Tensor, q_left: Tensor, q_right: Tensor) -> Tensor:
    return q_left.T @ x.float() @ q_right


def _project_back_2d(x: Tensor, q_left: Tensor, q_right: Tensor) -> Tensor:
    return q_left @ x.float() @ q_right.T


def _matrix_outer_products(grad: Tensor) -> tuple[Tensor, Tensor]:
    grad_f = grad.float()
    return grad_f @ grad_f.T, grad_f.T @ grad_f


def matrix_sign_exact(matrix: Tensor, eps: float = 1e-12) -> Tensor:
    """Exact matrix sign via thin SVD. Kept only for tests/debugging."""
    del eps
    u, _, vh = torch.linalg.svd(matrix.float(), full_matrices=False)
    return (u @ vh).to(dtype=matrix.dtype)


def matrix_sign_ns5(matrix: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Muon-style quintic Newton-Schulz approximation to the matrix sign.

    Supports a single matrix ``(m, n)`` or a stack ``(..., m, n)``. This is the
    ordinary Muon orthogonalization path; unlike nanochat's native Muon group,
    callers decide any dimension-dependent LR scaling outside this function.
    """
    if matrix.ndim < 2:
        raise ValueError(f"matrix_sign_ns5 expects a matrix or stack of matrices, got {tuple(matrix.shape)}")

    a, b, c = (3.4445, -4.7750, 2.0315)
    x = matrix
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.mT

    compute_dtype = torch.bfloat16 if x.is_cuda else torch.float32
    x = x.to(dtype=compute_dtype)
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        gram = x @ x.mT
        update = b * gram + c * (gram @ gram)
        x = a * x + update @ x

    if transposed:
        x = x.mT
    return x.to(dtype=matrix.dtype)


def _all_reduce_gradients(param_groups: Iterable[dict]) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    for group in param_groups:
        for p in group["params"]:
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)


class StructuredAdamW(torch.optim.Optimizer):
    """Combined AdamW plus matrix preconditioner baselines.

    Supported matrix group kinds:
    - ``plain_muon``: ordinary Muon using the NS5 matrix-sign approximation of
      the Nesterov momentum input. It intentionally does not apply nanochat's
      dimension-dependent LR normalization.
    - ``soap``: SOAP/RMSProp in Shampoo's eigenbasis.
    - ``shampoo``: two-sided Shampoo with per-factor ``-1/4`` powers.
    - ``kl_shampoo``: KL-Shampoo-style two-sided factor update plus
      eigenvalue EMA correction.
    - ``kl_soap``: KL-Shampoo basis update plus SOAP/RMSProp in that basis.

    The implementation supports 2D matrix weights, which is all this project
    optimizes with Muon-like methods. Non-matrix parameters should be placed in
    ``adamw`` groups by the caller.
    """

    def __init__(self, param_groups: list[dict], *, average_gradients: bool = False):
        super().__init__(param_groups, defaults={})
        self.average_gradients = average_gradients
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group: dict) -> None:
        from nanochat.optim import adamw_step_fused

        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            self._adamw_step_t.fill_(state["step"])
            self._adamw_lr_t.fill_(group["lr"])
            self._adamw_beta1_t.fill_(group["betas"][0])
            self._adamw_beta2_t.fill_(group["betas"][1])
            self._adamw_eps_t.fill_(group["eps"])
            self._adamw_wd_t.fill_(group["weight_decay"])
            adamw_step_fused(
                p,
                p.grad,
                state["exp_avg"],
                state["exp_avg_sq"],
                self._adamw_step_t,
                self._adamw_lr_t,
                self._adamw_beta1_t,
                self._adamw_beta2_t,
                self._adamw_eps_t,
                self._adamw_wd_t,
            )

    def _init_matrix_state(self, p: Tensor, group: dict) -> dict:
        state = self.state[p]
        if state:
            return state

        m, n = p.shape
        device = p.device
        init_factor = float(group.get("init_factor", 1.0))
        eps = float(group.get("eps", 1e-8))
        left_eye = torch.eye(m, device=device, dtype=torch.float32)
        right_eye = torch.eye(n, device=device, dtype=torch.float32)
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
        if group["kind"] in {"soap", "kl_soap"}:
            state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
        state["GG"] = [init_factor * left_eye.clone(), init_factor * right_eye.clone()]
        state["Q"] = [left_eye.clone(), right_eye.clone()]
        state["eigenvalues"] = [
            torch.full((m,), init_factor, device=device, dtype=torch.float32),
            torch.full((n,), init_factor, device=device, dtype=torch.float32),
        ]
        inv = 1.0 / math.sqrt(max(init_factor, eps))
        state["eigen_sqrt_inv"] = [
            torch.full((m,), inv, device=device, dtype=torch.float32),
            torch.full((n,), inv, device=device, dtype=torch.float32),
        ]
        return state

    def _refresh_eigenbasis(self, state: dict, group: dict) -> None:
        eps = float(group.get("eps", 1e-8))
        old_q = state["Q"]
        new_q = []
        new_vals = []
        for idx, factor in enumerate(state["GG"]):
            if group.get("use_qr", True) and state["step"] > 0:
                q = _qr_basis(factor, old_q[idx])
                vals = torch.diagonal(q.T @ factor.float() @ q).clamp_min(eps)
            else:
                q, vals = _eigh_basis(factor, eps)
            new_q.append(q)
            new_vals.append(vals)
        state["Q"] = new_q
        state["eigenvalues"] = new_vals
        state["eigen_sqrt_inv"] = [_finite_power(vals, -0.5, eps) for vals in new_vals]

    def _update_shampoo_preconditioner(self, grad: Tensor, state: dict, group: dict) -> None:
        beta = float(group.get("shampoo_beta", group.get("betas", (0.9, 0.95))[1]))
        left_outer, right_outer = _matrix_outer_products(grad)
        state["GG"][0].lerp_(left_outer, 1.0 - beta)
        state["GG"][1].lerp_(right_outer, 1.0 - beta)

    def _update_kl_preconditioner(self, grad: Tensor, state: dict, group: dict) -> None:
        beta = float(group.get("shampoo_beta", group.get("betas", (0.9, 0.95))[1]))
        eps = float(group.get("eps", 1e-8))
        max_inv_sqrt = float(group.get("max_inv_sqrt", 4000.0))
        grad_f = grad.float()
        m, n = grad_f.shape
        q_left, q_right = state["Q"]
        inv_left, inv_right = state["eigen_sqrt_inv"]

        grad_right = grad_f @ q_right
        left_half = grad_right * inv_right.view(1, -1)
        left_mat = (left_half @ left_half.T) / max(n, 1)

        grad_left = q_left.T @ grad_f
        right_half = grad_left * inv_left.view(-1, 1)
        right_mat = (right_half.T @ right_half) / max(m, 1)

        state["GG"][0].lerp_(left_mat, 1.0 - beta)
        state["GG"][1].lerp_(right_mat, 1.0 - beta)

        diag_half = q_left.T @ grad_f @ q_right
        left_diag = torch.mean((diag_half * inv_right.view(1, -1)).square(), dim=1)
        right_diag = torch.mean((diag_half * inv_left.view(-1, 1)).square(), dim=0)
        for idx, diag in enumerate((left_diag, right_diag)):
            current = (1.0 / state["eigen_sqrt_inv"][idx].clamp_min(eps).square()).nan_to_num_(
                nan=eps, posinf=eps, neginf=eps
            )
            current.lerp_(diag.clamp_min(eps), 1.0 - beta)
            inv = _finite_power(current, -0.5, eps)
            state["eigen_sqrt_inv"][idx] = torch.clamp(inv, max=max_inv_sqrt)

    def _maybe_update_basis(self, state: dict, group: dict) -> None:
        freq = int(group.get("precondition_frequency", 10))
        if freq <= 0:
            return
        if state["step"] % freq != 0:
            return
        if group["kind"] in {"soap", "kl_soap"}:
            state["exp_avg"] = _project_back_2d(state["exp_avg"], state["Q"][0], state["Q"][1])
            self._refresh_eigenbasis(state, group)
            state["exp_avg"] = _project_2d(state["exp_avg"], state["Q"][0], state["Q"][1])
            # The diagonal RMS accumulator lives in the projected basis. There
            # is no exact cheap rotation for elementwise second moments, so do
            # not silently reuse old-basis statistics after a basis refresh.
            state["exp_avg_sq"].zero_()
        else:
            self._refresh_eigenbasis(state, group)

    def _matrix_update(self, p: Tensor, group: dict) -> Tensor:
        state = self._init_matrix_state(p, group)
        grad = p.grad.float()
        beta1, beta2 = group.get("betas", (0.9, 0.95))
        eps = float(group.get("eps", 1e-8))
        kind = group["kind"]

        state["step"] += 1
        if kind in {"soap", "kl_soap"}:
            q_left, q_right = state["Q"]
            grad_projected = _project_2d(grad, q_left, q_right)
            state["exp_avg"].lerp_(grad_projected, 1.0 - float(beta1))
            state["exp_avg_sq"].lerp_(grad_projected.square(), 1.0 - float(beta2))
            denom = state["exp_avg_sq"].sqrt().add_(eps)
            update = _project_back_2d(state["exp_avg"] / denom, q_left, q_right)
        elif kind == "shampoo":
            state["exp_avg"].lerp_(grad, 1.0 - float(beta1))
            q_left, q_right = state["Q"]
            left_scale = _finite_power(state["eigenvalues"][0], -0.25, eps)
            right_scale = _finite_power(state["eigenvalues"][1], -0.25, eps)
            projected = _project_2d(state["exp_avg"], q_left, q_right)
            update = _project_back_2d(projected * left_scale.view(-1, 1) * right_scale.view(1, -1), q_left, q_right)
        elif kind == "kl_shampoo":
            state["exp_avg"].lerp_(grad, 1.0 - float(beta1))
            q_left, q_right = state["Q"]
            projected = _project_2d(state["exp_avg"], q_left, q_right)
            scale = state["eigen_sqrt_inv"][0].view(-1, 1) * state["eigen_sqrt_inv"][1].view(1, -1)
            update = _project_back_2d(projected * scale, q_left, q_right)
        else:
            raise ValueError(f"Unknown structured optimizer kind: {kind}")

        if kind in {"kl_shampoo", "kl_soap"}:
            self._update_kl_preconditioner(grad, state, group)
        else:
            self._update_shampoo_preconditioner(grad, state, group)
        self._maybe_update_basis(state, group)
        return update.to(dtype=p.dtype)

    def _step_structured(self, group: dict) -> None:
        lr = float(group["lr"])
        wd = float(group.get("weight_decay", 0.0))
        for p in group["params"]:
            if p.grad is None:
                continue
            if p.ndim != 2:
                raise ValueError(f"{group['kind']} only supports 2D matrix params, got shape={tuple(p.shape)}")
            update = self._matrix_update(p, group)
            if wd:
                p.add_(p, alpha=-lr * wd)
            p.add_(update, alpha=-lr)

    def _step_plain_muon(self, group: dict) -> None:
        lr = float(group["lr"])
        wd = float(group.get("weight_decay", 0.0))
        momentum = float(group.get("momentum", 0.95))
        params = [p for p in group["params"] if p.grad is not None]
        if not params:
            return
        shape = params[0].shape
        if len(shape) != 2:
            raise ValueError(f"plain_muon only supports 2D matrix params, got shape={tuple(shape)}")
        if any(p.shape != shape for p in params):
            raise ValueError("plain_muon groups must contain same-shaped matrix params")

        first = params[0]
        state = self.state[first]
        if "momentum_buffer" not in state or state["momentum_buffer"].shape[0] != len(params):
            state["momentum_buffer"] = torch.zeros(len(params), *shape, dtype=first.dtype, device=first.device)
        buf = state["momentum_buffer"]

        grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        buf.lerp_(grads, 1.0 - momentum)
        nesterov = grads.float().lerp(buf.float(), momentum)
        update = matrix_sign_ns5(nesterov, steps=int(group.get("ns_steps", 5))).to(dtype=stacked_params.dtype)
        if wd:
            stacked_params.add_(stacked_params, alpha=-lr * wd)
        stacked_params.add_(update, alpha=-lr)
        for p, updated in zip(params, stacked_params):
            p.copy_(updated)

    @torch.no_grad()
    def step(self):
        if self.average_gradients:
            _all_reduce_gradients(self.param_groups)
        for group in self.param_groups:
            kind = group["kind"]
            if kind == "adamw":
                self._step_adamw(group)
            elif kind == "plain_muon":
                self._step_plain_muon(group)
            elif kind in STRUCTURED_KINDS:
                self._step_structured(group)
            else:
                raise ValueError(f"Unknown optimizer kind: {kind}")
