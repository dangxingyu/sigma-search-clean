"""
SPECTRA optimizer (Huang et al. 2026, arXiv:2602.11185).

"Spectra: Rethinking Optimizers for LLMs Under Spectral Anisotropy"

Key idea (Algorithm 1 from the paper):
- Maintain momentum M_t = μ M_{t-1} + G_t
- Estimate rank-k spike subspace (U_k, s_k, V_k) of M via warm-started power iteration
- Replace spike singular values with tail-scale estimate σ_tail:
    M_tail = M_t - U_k diag(s_k) V_k^T
    σ_tail = sqrt(||M_tail||_F^2 / (min(m,n) - k))
    O_t = M_tail + U_k diag(σ_tail · I) V_k^T
- Normalize step by RMS(O): η' = 0.2 η / (RMS(O) + ε)
- Update: W_t = W_{t-1} - η' O_t

Rank k = max(1, round(r · min(m,n))) with r = 0.015 default (1.5%)
T = 1-2 power iterations with warm-started V cache.

Drop-in replacement for nanochat's MuonAdamW matrix path.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


def _thin_qr_positive(A: Tensor) -> Tensor:
    """QR with positive diagonal on R, for sign consistency."""
    Q, R = torch.linalg.qr(A, mode='reduced')
    diag_sign = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))
    diag_sign = torch.where(diag_sign == 0, torch.ones_like(diag_sign), diag_sign)
    return Q * diag_sign.unsqueeze(-2)


def cached_power_iter_svd(
    G: Tensor,               # (B, m, n)
    V_cache: Optional[Tensor],  # (B, n, k) or None
    k: int,
    T: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Algorithm 2: Cached Power-Iteration SVD (rank-k).
    Warm-started from V_cache if available.
    Returns (U_k, s_k, V_k) with shapes (B,m,k), (B,k), (B,n,k).
    """
    B, m, n = G.shape
    Gf = G.float()

    if V_cache is None or V_cache.shape[-1] != k:
        # Bootstrap via torch.svd_lowrank (randomized SVD)
        U_k, s_k, V_k = torch.svd_lowrank(Gf, q=k, niter=2)
        return U_k, s_k, V_k

    V = V_cache.float()
    for _ in range(T):
        P = torch.matmul(Gf, V)          # (B, m, k)
        U = _thin_qr_positive(P)         # (B, m, k)
        W = torch.matmul(Gf.mT, U)       # (B, n, k)
        s = torch.linalg.norm(W, dim=-2) # (B, k)
        V = W / s.unsqueeze(-2).clamp(min=1e-12)
    return U, s, V


class SpectraAdamW(torch.optim.Optimizer):
    """
    Combined optimizer: Spectra for 2D matrix params, AdamW for others.
    Drop-in replacement for nanochat's MuonAdamW (compatible param_groups format).

    Spectra groups use 'kind': 'spectra' (or auto-upgrade from 'muon').
    AdamW groups use 'kind': 'adamw'.
    """

    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors for AdamW hyperparameter passing
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

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

    def _step_spectra(self, group: dict) -> None:
        """Spectra Algorithm 1: spike-aware singular value shrinkage."""
        params: list[Tensor] = group['params']
        if not params:
            return

        p0 = params[0]
        state = self.state[p0]
        num_params = len(params)
        shape, device, dtype = p0.shape, p0.device, p0.dtype
        m, n = shape[-2], shape[-1]

        # Hyperparameters
        lr = group['lr'] * max(1.0, m / n) ** 0.5   # match nanochat LR scaling convention
        mu = group.get('momentum', 0.95)
        wd = group.get('weight_decay', 0.0)
        r = group.get('rank_ratio', 0.015)
        T_iters = group.get('power_iters', 2)
        eps = group.get('eps', 1e-8)

        k = max(1, round(r * min(m, n)))

        # --- Initialize state ---
        if 'momentum_buffer' not in state:
            state['momentum_buffer'] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if 'v_cache' not in state:
            state['v_cache'] = None  # will be bootstrap-initialized on first call

        mb = state['momentum_buffer']

        # Stack grads and params
        stacked_grads = torch.stack([p.grad for p in params])    # (B, m, n)
        stacked_params = torch.stack(params)

        # Step 5: M_t = μ M_{t-1} + G_t
        mb.mul_(mu).add_(stacked_grads)

        # Step 6: rank-k spike SVD via warm-started power iteration
        U_k, s_k, V_k = cached_power_iter_svd(mb, state['v_cache'], k, T_iters)
        state['v_cache'] = V_k.detach()

        # Step 7: M_tail = M_t - U_k diag(s_k) V_k^T
        spike = torch.matmul(U_k * s_k.unsqueeze(-2), V_k.mT)  # (B, m, n)
        M_tail = mb.float() - spike

        # Step 8: σ_tail = sqrt(||M_tail||_F^2 / (min(m,n) - k))
        tail_energy = M_tail.pow(2).sum(dim=(-2, -1))          # (B,)
        denom_min = min(m, n) - k
        sigma_tail = (tail_energy / max(denom_min, 1)).sqrt()  # (B,)

        # Step 9: O_t = M_tail + U_k diag(σ_tail · I_k) V_k^T
        # i.e., replace spike singular values with σ_tail (broadcast over k)
        shaped_spike = torch.matmul(U_k * sigma_tail.view(-1, 1, 1), V_k.mT)
        O = M_tail + shaped_spike

        # Step 10: RMS = ||O||_F / sqrt(m*n)
        rms = O.pow(2).mean(dim=(-2, -1)).sqrt()               # (B,)

        # Step 11: η' = 0.2 η / (RMS + ε)
        lr_per = 0.2 * lr / (rms + eps)                        # (B,)

        # Step 12: W_t = W_{t-1} - η' O_t  (include decoupled weight decay)
        update = O * lr_per.view(-1, 1, 1)
        stacked_params.sub_(update.to(dtype) + (lr * wd) * stacked_params)

        # Copy back
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            kind = group['kind']
            if kind == 'adamw':
                self._step_adamw(group)
            elif kind == 'spectra':
                self._step_spectra(group)
            elif kind == 'muon':
                # Auto-upgrade muon → spectra
                group['kind'] = 'spectra'
                self._step_spectra(group)
            else:
                raise ValueError(f'Unknown optimizer kind: {kind}')
