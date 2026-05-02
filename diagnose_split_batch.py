"""Split-batch SVD alignment diagnostic (D1 from guidance.md).

For each matrix parameter P in a model, compute two independent gradients
G_A, G_B from disjoint halves of a batch, do SVD on both, and report a
per-direction alignment c_i. Sadhika's original rule uses
|u_A^(i)^T u_B^(i)|. For LITE-specific diagnostics, `alignment_side="lite"`
uses the singular-vector side that LITE projects on: V for tall matrices and
U for wide matrices.

Interpretation:
  c_i ≈ 1  →  direction is a real, resolved spike (signal reproducible)
  c_i ≈ 0  →  direction is bulk noise (not reproducible)

Decision rule (Sadhika's original): count top-r directions with c_i > 0.7.
  >50% → LITE regime.  <20% → Muon regime.  middle → ambiguous.

Also reports a heuristic data-dependent threshold c* motivated by BBP/RMT:
call a direction resolved if it is about 20% above the aspect-ratio-dependent
noise edge, then clamp the resulting alignment threshold to [0.25, 0.8].
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import torch


def _sample_two_grads(model, batch_iter, loss_fn, device, num_microbatches: int = 2):
    """Accumulate grads from num_microbatches batches into G_A (first half)
    and G_B (second half), both same total number of microbatches.

    Returns dict[param_name, (G_A, G_B)] for matrix params only.
    """
    assert num_microbatches >= 2 and num_microbatches % 2 == 0
    half = num_microbatches // 2
    matrix_params = {n: p for n, p in model.named_parameters()
                     if p.requires_grad and p.ndim == 2}

    grads_a = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in matrix_params.items()}
    grads_b = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in matrix_params.items()}

    for i in range(num_microbatches):
        model.zero_grad(set_to_none=True)
        batch = next(batch_iter)
        loss = loss_fn(model, batch)
        loss.backward()
        target = grads_a if i < half else grads_b
        for n, p in matrix_params.items():
            if p.grad is not None:
                target[n] += p.grad.detach().to(torch.float32) / half

    return {n: (grads_a[n], grads_b[n]) for n in matrix_params}


@torch.no_grad()
def alignment_from_grad_pair(GA: torch.Tensor, GB: torch.Tensor, top_k: int,
                             alignment_side: str = "left") -> dict:
    """Compute c_i = |u_A^(i)^T u_B^(i)| for the top_k singular directions
    of GA and GB. Uses float32.

    Returns dict with:
      c_top_k: Tensor[top_k]        — per-direction alignment
      frac_above_0.7: float         — fraction of c > 0.7 (Sadhika rule)
      frac_above_c_star: float      — fraction above RMT threshold
      c_star: float                 — the RMT threshold
      sigma_ratio: Tensor[top_k]    — σ_i / σ_1 (decay)
      eff_rank: float               — effective rank of (GA+GB)/2
    """
    m, n = GA.shape
    k = min(top_k, min(m, n))
    # Economy SVD on both halves.
    UA, SA, VAh = torch.linalg.svd(GA.float(), full_matrices=False)
    UB, SB, VBh = torch.linalg.svd(GB.float(), full_matrices=False)

    if alignment_side not in {"left", "right", "lite"}:
        raise ValueError(f"alignment_side must be left|right|lite, got {alignment_side!r}")
    side_used = alignment_side
    if alignment_side == "lite":
        # LITE applies the sharp/flat projection on the smaller side:
        # tall matrices use V-space (right), wide matrices use U-space (left).
        side_used = "right" if m >= n else "left"

    if side_used == "left":
        A_k = UA[:, :k]
        B_k = UB[:, :k]
    else:
        A_k = VAh.mT[:, :k]
        B_k = VBh.mT[:, :k]

    # Pairwise alignment in the top-k subspace. We align by index (not
    # optimal matching) since that's what Sadhika's rule uses; singular
    # directions are ordered by σ, so G_A and G_B should roughly line up.
    c = (A_k * B_k).sum(dim=0).abs()  # (k,)
    # Optional: signed inner product could go negative; abs is correct for
    # direction identification.

    # RMT detection threshold. For a rank-1 spike with population strength θ
    # in bulk noise with operator norm 1, the sample alignment obeys
    #   |<u_hat, u_pop>|^2 ≈ 1 - 1/θ^2     (θ > 1, square case; analog for rect)
    # We want a *detection margin*, not the BBP edge itself (edge → c=0).
    # Rule of thumb: call a direction "resolved" when θ ≥ 1.2·BBP_edge, giving
    # cross-alignment c_AB ≈ (1 - 1/θ^2) (two independent halves, index-matched).
    gamma = min(m, n) / max(m, n)
    theta_margin = 1.2  # 20% above BBP edge
    theta_detect = theta_margin * math.sqrt(max(gamma, 1e-12))
    # For γ<1 (rectangular): edge is √γ, detection at 1.2√γ.
    # For γ=1 (square): edge is 1, detection at 1.2. Either way formula below:
    c_star_raw = max(0.0, 1.0 - 1.0 / max(theta_detect**2, 1e-12))
    # For very rectangular γ<<1, theta_detect<1 and formula can underflow;
    # floor to 0.25 (empirical null is very small so 0.25 is safely above pure noise).
    # Cap at 0.8 to keep threshold achievable even for near-edge spikes.
    c_star = min(0.8, max(0.25, c_star_raw))

    # Effective rank from averaged gradient
    G_avg = 0.5 * (GA + GB)
    S_avg = torch.linalg.svdvals(G_avg.float())
    eff_rank = (S_avg.sum()**2 / (S_avg**2).sum()).item()

    return {
        "c_top_k": c.cpu(),
        "frac_above_0.7": (c > 0.7).float().mean().item(),
        "frac_above_c_star": (c > c_star).float().mean().item(),
        "c_star": c_star,
        "sigma_ratio": (SA[:k] / (SA[0] + 1e-12)).cpu(),
        "eff_rank": eff_rank,
        "shape": (m, n),
        "gamma": gamma,
        "alignment_side": side_used,
    }


def aggregate(per_layer: dict[str, dict]) -> dict:
    """Aggregate per-layer stats into a scalar summary for the model."""
    keys = ["frac_above_0.7", "frac_above_c_star", "eff_rank"]
    agg = {}
    for k in keys:
        vals = [v[k] for v in per_layer.values()]
        agg[f"mean_{k}"] = float(sum(vals) / len(vals))
        agg[f"median_{k}"] = float(sorted(vals)[len(vals) // 2])
        agg[f"min_{k}"] = float(min(vals))
        agg[f"max_{k}"] = float(max(vals))

    # LITE/Muon decision on the median layer
    f = agg["median_frac_above_0.7"]
    if f > 0.5:
        agg["decision_rule_0.7"] = "LITE"
    elif f < 0.2:
        agg["decision_rule_0.7"] = "Muon"
    else:
        agg["decision_rule_0.7"] = "ambiguous"

    f2 = agg["median_frac_above_c_star"]
    if f2 > 0.5:
        agg["decision_rule_rmt"] = "LITE"
    elif f2 < 0.2:
        agg["decision_rule_rmt"] = "Muon"
    else:
        agg["decision_rule_rmt"] = "ambiguous"

    return agg


def diagnose_from_grads(grad_pairs: dict[str, tuple[torch.Tensor, torch.Tensor]],
                        top_k: int = 32, min_grad_norm: float = 1e-10,
                        alignment_side: str = "left") -> dict:
    """Run alignment_from_grad_pair on each layer and aggregate.

    Layers where both G_A and G_B have Frobenius norm below min_grad_norm are
    marked as "inactive" and excluded from the aggregate (this happens at
    nanochat init where c_q/c_k/c_v/mlp.c_fc are zero-initialized).
    """
    per_layer = {}
    for name, (GA, GB) in grad_pairs.items():
        na = GA.float().norm().item()
        nb = GB.float().norm().item()
        if na < min_grad_norm and nb < min_grad_norm:
            per_layer[name] = {"inactive": True, "norm_a": na, "norm_b": nb,
                                "shape": tuple(GA.shape)}
            continue
        stats = alignment_from_grad_pair(GA, GB, top_k=top_k, alignment_side=alignment_side)
        stats["inactive"] = False
        stats["norm_a"] = na
        stats["norm_b"] = nb
        per_layer[name] = stats
    active = {n: v for n, v in per_layer.items() if not v.get("inactive", False)}
    agg = aggregate(active) if active else {"decision_rule_0.7": "no_active_layers",
                                              "decision_rule_rmt": "no_active_layers"}
    agg["n_active"] = len(active)
    agg["n_inactive"] = len(per_layer) - len(active)
    return {"per_layer": per_layer, "aggregate": agg}


def to_json(result: dict) -> dict:
    """Make the result JSON-serializable (drop tensors, keep stats)."""
    clean_per_layer = {}
    for name, v in result["per_layer"].items():
        if v.get("inactive", False):
            clean_per_layer[name] = {
                "inactive": True,
                "norm_a": v["norm_a"],
                "norm_b": v["norm_b"],
                "shape": list(v["shape"]),
            }
            continue
        clean_per_layer[name] = {
            "inactive": False,
            "norm_a": v["norm_a"],
            "norm_b": v["norm_b"],
            "frac_above_0.7": v["frac_above_0.7"],
            "frac_above_c_star": v["frac_above_c_star"],
            "c_star": v["c_star"],
            "eff_rank": v["eff_rank"],
            "shape": list(v["shape"]),
            "gamma": v["gamma"],
            "alignment_side": v.get("alignment_side"),
            "c_top_k_mean": float(v["c_top_k"].mean()),
            "c_top_k_min": float(v["c_top_k"].min()),
            "c_top_k_max": float(v["c_top_k"].max()),
        }
    return {"per_layer": clean_per_layer, "aggregate": result["aggregate"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split-batch SVD alignment diagnostic (D1)."
    )
    parser.add_argument("--smoke", action="store_true",
                        help="Run a synthetic smoke test (no model, no data).")
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.smoke:
        torch.manual_seed(args.seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[smoke] device={device}")

        # Scale noise so that a single noise matrix has operator norm ≈ 1
        # (Ginibre: entries ~ N(0, 1/max(m,n))). This matches the RMT
        # normalization the c_star formula assumes. Spike strength θ is
        # then directly the top-singular-value scale.
        grad_pairs = {}
        m, n = 768, 768
        noise_std = 1.0 / math.sqrt(max(m, n))

        # Case A: 4 strong spikes (θ ≈ 4, 3, 2, 1.5) — well above BBP edge ~1.
        U = torch.linalg.qr(torch.randn(m, 4, device=device))[0]
        V = torch.linalg.qr(torch.randn(n, 4, device=device))[0]
        S = torch.tensor([4.0, 3.0, 2.0, 1.5], device=device)
        signal = U @ torch.diag(S) @ V.T
        grad_pairs["resolved_spikes"] = (
            signal + noise_std * torch.randn(m, n, device=device),
            signal + noise_std * torch.randn(m, n, device=device),
        )

        # Case B: pure noise — c should be ≈ chance (very small for index-matched).
        grad_pairs["pure_noise"] = (
            noise_std * torch.randn(m, n, device=device),
            noise_std * torch.randn(m, n, device=device),
        )

        # Case C: buried spikes (θ < 1 = below BBP edge) — indistinguishable from noise.
        U2 = torch.linalg.qr(torch.randn(m, 4, device=device))[0]
        V2 = torch.linalg.qr(torch.randn(n, 4, device=device))[0]
        S2 = torch.tensor([0.6, 0.5, 0.4, 0.3], device=device)
        sig2 = U2 @ torch.diag(S2) @ V2.T
        grad_pairs["buried_spikes"] = (
            sig2 + noise_std * torch.randn(m, n, device=device),
            sig2 + noise_std * torch.randn(m, n, device=device),
        )

        # Case D: rectangular spike (γ=0.25), same 4 resolved spikes.
        mR, nR = 1024, 256
        noise_std_R = 1.0 / math.sqrt(max(mR, nR))
        U3 = torch.linalg.qr(torch.randn(mR, 4, device=device))[0]
        V3 = torch.linalg.qr(torch.randn(nR, 4, device=device))[0]
        S3 = torch.tensor([4.0, 3.0, 2.0, 1.5], device=device)
        sig3 = U3 @ torch.diag(S3) @ V3.T
        grad_pairs["rectangular_spikes"] = (
            sig3 + noise_std_R * torch.randn(mR, nR, device=device),
            sig3 + noise_std_R * torch.randn(mR, nR, device=device),
        )

        result = diagnose_from_grads(grad_pairs, top_k=args.top_k)
        clean = to_json(result)

        print(json.dumps(clean, indent=2))
        if args.output:
            Path(args.output).write_text(json.dumps(clean, indent=2))
        sys.exit(0)

    parser.error("Non-smoke usage requires integration into training loop; "
                 "import `diagnose_from_grads` from this module.")
