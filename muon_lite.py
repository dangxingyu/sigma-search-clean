"""
Muon-LITE optimizer (Zhu et al. 2026, arxiv 2602.22681).

Amplifies updates along FLAT (non-top-d_s) directions by factor χ ≥ 1, while
keeping sharp directions unchanged. Motivated by the River-Valley landscape
picture: most of the loss decrease happens along flat directions, so uniform
Muon updates (all σ = 1) under-utilize those directions.

Update formula (first term only; β_{1,2} = 0 for simplicity):
    U_k = polar_factor(M̃_k) · (P_k + χ (I - P_k))
where:
    M̃_k      = Nesterov-corrected momentum
    polar_factor(·) = approximate msign via 5-step Polar Express
    P_k       = projection onto top-d_s eigenspace of M̃_k^T M̃_k (sharp)
    I - P_k   = flat directions

Drop-in replacement for nanochat's MuonAdamW for single-GPU (or with
torchrun + rank-sharded data loader, same pattern as SpectraAdamW).
"""

import torch
from torch import Tensor
from typing import Optional

from nanochat.common import COMPUTE_DTYPE


# Polar Express 5-step coefficients (same as nanochat's muon)
_POLAR_EXPRESS_COEFFS = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)


def _polar_factor(g: Tensor, ns_steps: int = 5) -> Tensor:
    """Compute approximate polar factor UV^T via Polar Express iterations.

    Input is normalized by Frobenius norm first (required for convergence).
    Output has σ ∈ ~[0.5, 1.5] (nominal polar express behavior).
    """
    X = g.bfloat16() if COMPUTE_DTYPE == torch.bfloat16 else g
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    if g.size(-2) > g.size(-1):  # Tall (m > n)
        for a, b, c in _POLAR_EXPRESS_COEFFS[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:  # Wide (m <= n)
        for a, b, c in _POLAR_EXPRESS_COEFFS[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    return X


def _lite_top_eigvecs(M_tilde: Tensor, d_s: int, ridge: float = 1e-6) -> Tensor:
    """Return top-d_s eigenvectors of M̃^T M̃ (tall) or M̃ M̃^T (wide).

    Shapes: M̃ is (..., m, n). Returns (..., min(m,n), d_s) where columns are
    the top-d_s eigenvectors by largest eigenvalue of the gram matrix.
    """
    m, n = M_tilde.shape[-2], M_tilde.shape[-1]
    M = M_tilde.float()
    if m >= n:
        gram = torch.matmul(M.mT, M)  # (..., n, n)
    else:
        gram = torch.matmul(M, M.mT)  # (..., m, m)
    dim = gram.shape[-1]
    # Ridge-stabilized eigh (scaled by gram magnitude to be invariant)
    scale = torch.diagonal(gram, dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).unsqueeze(-1)
    eye = torch.eye(dim, dtype=gram.dtype, device=gram.device)
    gram = gram + ridge * scale * eye
    _, eigvecs = torch.linalg.eigh(gram)  # eigvals ascending
    return eigvecs[..., -d_s:]  # top-d_s columns = largest eigenvalues


def _apply_projection(X: Tensor, V_s: Tensor, sharp_coef: float, flat_coef: float) -> Tensor:
    """Apply X · (sharp_coef · P + flat_coef · (I - P)) where P = V_s V_s^T.

    Equivalent to: flat_coef · X + (sharp_coef - flat_coef) · X · P.

    For tall X (m >= n): P acts on the right (column side = V space).
    For wide X (m < n): P acts on the left (row side = U space).
    """
    m, n = X.shape[-2], X.shape[-1]
    V_s_dt = V_s.to(X.dtype)
    if m >= n:
        XV = torch.matmul(X, V_s_dt)         # (..., m, d_s)
        XP = torch.matmul(XV, V_s_dt.mT)     # (..., m, n)
    else:
        VX = torch.matmul(V_s_dt.mT, X)      # (..., d_s, n)
        XP = torch.matmul(V_s_dt, VX)        # (..., m, n)
    return flat_coef * X + (sharp_coef - flat_coef) * XP


def _apply_lite(X: Tensor, V_s: Tensor, chi: float) -> Tensor:
    """Back-compat: LITE first-term projection X · (I · P + χ (I - P)) = χ X + (1-χ) X P."""
    return _apply_projection(X, V_s, sharp_coef=1.0, flat_coef=chi)


class MuonLiteAdamW(torch.optim.Optimizer):
    """Combined Muon-LITE + AdamW, same param_groups convention as nanochat's
    MuonAdamW. LITE groups use kind='muon_lite' (or 'muon' — auto-promoted).
    AdamW groups use kind='adamw'.

    Per-group LITE hyperparameters:
        lite_chi: flat-direction amplification (≥ 1, default 1 = plain Muon)
        lite_ds: sharp subspace dimension (default 0 = disabled)
        ns_steps: Polar Express iterations (default 5)
    """

    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors for AdamW fused path
        self._adam_step = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adam_lr = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adam_beta1 = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adam_beta2 = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adam_eps = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adam_wd = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group: dict) -> None:
        from nanochat.optim import adamw_step_fused
        for p in group['params']:
            if p.grad is None:
                continue
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adam_step.fill_(state['step'])
            self._adam_lr.fill_(group['lr'])
            self._adam_beta1.fill_(group['betas'][0])
            self._adam_beta2.fill_(group['betas'][1])
            self._adam_eps.fill_(group['eps'])
            self._adam_wd.fill_(group['weight_decay'])
            adamw_step_fused(
                p, p.grad, state['exp_avg'], state['exp_avg_sq'],
                self._adam_step, self._adam_lr, self._adam_beta1,
                self._adam_beta2, self._adam_eps, self._adam_wd,
            )

    def _step_muon_lite(self, group: dict) -> None:
        params = group['params']
        if not params:
            return
        p0 = params[0]
        state = self.state[p0]
        num_params = len(params)
        shape, device, dtype = p0.shape, p0.device, p0.dtype

        if 'momentum_buffer' not in state:
            state['momentum_buffer'] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        mb = state['momentum_buffer']

        stacked_grads = torch.stack([p.grad for p in params])  # (B, m, n)
        stacked_params = torch.stack(params)

        lr = group['lr']
        mom_val = group['momentum']
        wd = group['weight_decay']
        ns_steps = group.get('ns_steps', 5)
        chi = float(group.get('lite_chi', 1.0))
        d_s = int(group.get('lite_ds', 0))
        beta_1 = float(group.get('lite_beta1', 0.0))
        beta_2 = float(group.get('lite_beta2', 0.0))
        use_beta = (beta_1 != 0.0 or beta_2 != 0.0) and d_s > 0

        # Save raw G before Nesterov mutates stacked_grads (for LITE 2nd term)
        raw_G = stacked_grads.clone() if use_beta else None

        # Nesterov momentum (matches nanochat muon_step_fused)
        mb.lerp_(stacked_grads, 1 - mom_val)
        g = stacked_grads.lerp_(mb, mom_val)

        # --- First LITE term: polar_factor(M̃) · (P + χ(I - P)) ---
        X_M = _polar_factor(g, ns_steps=ns_steps)

        need_V_s = (chi != 1.0 or use_beta) and d_s > 0
        V_s = _lite_top_eigvecs(g, d_s) if need_V_s else None

        if V_s is not None and chi != 1.0:
            U = _apply_projection(X_M.float(), V_s, sharp_coef=1.0, flat_coef=chi)
        else:
            U = X_M.float()

        # --- Second LITE term: polar_factor(G_raw) · (β_1 P + χβ_2 (I - P)) ---
        if use_beta:
            X_G = _polar_factor(raw_G, ns_steps=ns_steps)
            U = U + _apply_projection(
                X_G.float(), V_s,
                sharp_coef=beta_1, flat_coef=chi * beta_2
            )

        # Apply weight decay + update (same as nanochat muon)
        U_dt = U.to(stacked_params.dtype)
        stacked_params.sub_(lr * U_dt + lr * wd * stacked_params)

        # Copy back to individual params
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    def _step_native_muon(self, group: dict) -> None:
        """Native Muon matrix update with the same state layout as LITE.

        This is used by one-switch experiments so Muon and LITE phases can
        share the same momentum buffer without reinitializing optimizer state.
        It mirrors nanochat's Muon LR aspect-ratio scaling.
        """
        params = group['params']
        if not params:
            return
        p0 = params[0]
        state = self.state[p0]
        num_params = len(params)
        shape, device, dtype = p0.shape, p0.device, p0.dtype

        if 'momentum_buffer' not in state:
            state['momentum_buffer'] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        mb = state['momentum_buffer']

        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)

        lr = group['lr'] * max(1.0, shape[-2] / shape[-1]) ** 0.5
        mom_val = group['momentum']
        wd = group['weight_decay']
        ns_steps = group.get('ns_steps', 5)

        mb.lerp_(stacked_grads, 1 - mom_val)
        g = stacked_grads.lerp_(mb, mom_val)
        U = _polar_factor(g, ns_steps=ns_steps).float()

        U_dt = U.to(stacked_params.dtype)
        stacked_params.sub_(lr * U_dt + lr * wd * stacked_params)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            kind = group['kind']
            if kind == 'adamw':
                self._step_adamw(group)
            elif kind == 'switch_muon_lite':
                mode = group.get('switch_mode', 'lite')
                if mode == 'muon':
                    self._step_native_muon(group)
                elif mode == 'lite':
                    self._step_muon_lite(group)
                else:
                    raise ValueError(f"Unknown switch_mode: {mode}")
            elif kind in ('muon', 'muon_lite'):
                group['kind'] = 'muon_lite'
                self._step_muon_lite(group)
            else:
                raise ValueError(f"Unknown optimizer kind: {kind}")
