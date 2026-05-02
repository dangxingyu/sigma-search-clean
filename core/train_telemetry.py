"""Light telemetry helpers for training drivers.

Collects:
  - per-step loss (full trajectory)
  - per-group grad_norm before each optimizer step
  - periodic D1 probe (split-batch SVD alignment)
  - periodic per-matrix-layer σ spectrum stats
"""

from __future__ import annotations

import sys
import time
import torch
from typing import Callable


def grad_norm_per_group(optimizer) -> dict:
    """Return {kind: L2 grad norm, summed over all params in groups of that kind}."""
    out = {}
    for g in optimizer.param_groups:
        kind = g.get("kind", "unknown")
        total_sq = 0.0
        for p in g["params"]:
            if p.grad is not None:
                total_sq += float(p.grad.detach().float().pow(2).sum())
        out[kind] = out.get(kind, 0.0) + total_sq
    return {k: v ** 0.5 for k, v in out.items()}


def spectral_stats(model, top_k: int = 8, max_layers: int = 8) -> dict:
    """Quick per-layer spectrum stats on matrix params.

    Returns dict with per-layer top singular values (for tracking).
    Only samples first `max_layers` layers to keep overhead low.
    """
    out = {"per_layer": {}}
    count = 0
    for n, p in model.named_parameters():
        if p.ndim != 2 or "transformer.h." not in n:
            continue
        if count >= max_layers:
            break
        with torch.no_grad():
            S = torch.linalg.svdvals(p.detach().float())
            out["per_layer"][n] = {
                "sigma_top": [float(x) for x in S[:top_k].cpu()],
                "sigma_min": float(S[-1]),
                "r_eff": float((S.sum() ** 2 / (S * S).sum())),
                "frobenius": float(p.detach().float().norm()),
            }
        count += 1
    return out


@torch.no_grad()
def d1_from_buffers(buf_A: dict, buf_B: dict, top_k: int = 16,
                    alignment_side: str = "left") -> dict:
    """D1 probe directly on pre-computed buffers (e.g. auxiliary momenta m_A, m_B).

    Args:
      buf_A, buf_B: dict[layer_name -> 2D Tensor]. Same keys, same shapes per key.
                   Caller is responsible for keeping these on a sensible device/dtype;
                   diagnose_from_grads will cast to float32 internally.
      top_k: number of top singular directions per layer.
      alignment_side: "left" for Sadhika's original U-side rule, "right" for
                      V-side, or "lite" to choose the projection side LITE uses
                      for each matrix shape.

    Returns the same aggregate-summary fields as midtraining_d1_probe so downstream
    analysis can be shared.
    """
    from diagnose_split_batch import diagnose_from_grads, to_json as d1_to_json

    t0 = time.time()
    pairs = {n: (buf_A[n], buf_B[n]) for n in buf_A}
    result = diagnose_from_grads(pairs, top_k=top_k, alignment_side=alignment_side)
    clean = d1_to_json(result)
    agg = clean.get("aggregate", {})

    dt = time.time() - t0
    return {
        "frac_above_0_7_median": agg.get("median_frac_above_0.7"),
        "frac_above_0_7_mean": agg.get("mean_frac_above_0.7"),
        "frac_above_c_star_median": agg.get("median_frac_above_c_star"),
        "eff_rank_median": agg.get("median_eff_rank"),
        "n_active_layers": agg.get("n_active"),
        "alignment_side": alignment_side,
        "elapsed_s": dt,
    }


@torch.enable_grad()
def midtraining_d1_probe(model, loader, micros_per_half: int = 4, top_k: int = 16,
                         alignment_side: str = "left") -> dict:
    """Run a split-batch SVD alignment probe on the current model state.

    Does micros_per_half × 2 microbatches of gradient computation. Does NOT
    call optimizer.step(), so the training trajectory is unaffected.

    Returns aggregate D1 stats (frac_above_0.7, etc.).
    """
    from diagnose_split_batch import diagnose_from_grads, to_json as d1_to_json

    matrix_params = {n: p for n, p in model.named_parameters()
                     if p.requires_grad and p.ndim == 2 and "transformer.h." in n}
    grads_a = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in matrix_params.items()}
    grads_b = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in matrix_params.items()}

    was_training = model.training
    model.train()

    t0 = time.time()
    for half_idx, target in enumerate([grads_a, grads_b]):
        for _ in range(micros_per_half):
            model.zero_grad(set_to_none=True)
            x, y = next(loader)
            loss = model(x, y)
            loss.backward()
            for n, p in matrix_params.items():
                if p.grad is not None:
                    target[n] += p.grad.detach().to(torch.float32) / micros_per_half

    # Zero grads so we don't accidentally apply them
    model.zero_grad(set_to_none=True)
    if not was_training:
        model.eval()

    pairs = {n: (grads_a[n], grads_b[n]) for n in matrix_params}
    result = diagnose_from_grads(pairs, top_k=top_k, alignment_side=alignment_side)
    clean = d1_to_json(result)
    # Summarize aggregate
    agg = clean.get("aggregate", {})

    dt = time.time() - t0
    return {
        "frac_above_0_7_median": agg.get("median_frac_above_0.7"),
        "frac_above_0_7_mean": agg.get("mean_frac_above_0.7"),
        "frac_above_c_star_median": agg.get("median_frac_above_c_star"),
        "eff_rank_median": agg.get("median_eff_rank"),
        "n_active_layers": agg.get("n_active"),
        "alignment_side": alignment_side,
        "elapsed_s": dt,
    }


def should_probe(step: int, num_iterations: int, n_probes: int = 5) -> bool:
    """Check if this step is one of n_probes uniformly-spaced probe points."""
    if num_iterations < n_probes:
        return False
    # fractions: 1/n, 2/n, ..., n/n
    fractions = [(i + 1) / (n_probes + 1) for i in range(n_probes)]
    probe_steps = [int(f * num_iterations) for f in fractions]
    return step in probe_steps
