"""Diagnostic metric logging for Muon-like training runs.

The logger is intentionally opt-in. It reads a short-lived optimizer cache that
`streaming_muon_torch.py` materializes only on requested logging steps, so
normal training does not pay extra memory or SVD cost.
"""

from __future__ import annotations

import math
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch


def _rms(x: torch.Tensor) -> float:
    x = torch.nan_to_num(x.detach().float())
    return float(torch.sqrt(torch.mean(x * x)).cpu())


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    denom = a.norm() * b.norm()
    if float(denom.cpu()) == 0.0:
        return float("nan")
    return float(torch.dot(a, b).div(denom).cpu())


def _to_float_list(x: torch.Tensor) -> list[float]:
    return [float(v) for v in x.detach().float().cpu().tolist()]


def _match_module(name: str, module_regex: str | None) -> bool:
    return module_regex is None or module_regex == "" or re.search(module_regex, name) is not None


def _rank_info() -> tuple[int, int, bool]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size(), True
    return 0, 1, False


def _math_sdpa_context():
    """Force math SDPA for Hessian probes so attention supports double backward."""
    if not torch.cuda.is_available():
        return nullcontext()
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        return sdpa_kernel([SDPBackend.MATH])
    except Exception:
        return torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_mem_efficient=False,
            enable_math=True,
        )


@torch.no_grad()
def collect_muon_metrics(
    optimizer: torch.optim.Optimizer,
    param_name_by_id: dict[int, str],
    *,
    step: int,
    top_k: int = 4,
    module_regex: str | None = r"transformer\.h",
    max_modules: int = 0,
    train_loss: float | None = None,
    lr_multiplier: float | None = None,
    muon_momentum: float | None = None,
    save_components: bool = False,
    component_dir: str | Path | None = None,
) -> tuple[dict[str, Any] | None, dict[str, dict[str, torch.Tensor]]]:
    """Collect per-module scalar/vector diagnostics from cached Muon tensors.

    Returns `(entry, references)`. `entry` is populated only on rank 0. The
    `references` dict is rank-local and is meant for optional Hessian alignment
    probes on rank 0.
    """
    rank, world_size, ddp = _rank_info()
    local_rows: list[dict[str, Any]] = []
    local_components: dict[str, dict[str, torch.Tensor]] = {}
    local_refs: dict[str, dict[str, torch.Tensor]] = {}

    for group_idx, group in enumerate(optimizer.param_groups):
        if group.get("kind") not in ("streaming_muon", "muon"):
            continue
        params = group["params"]
        if not params:
            continue
        state = optimizer.state.get(params[0], {})
        cache = state.get("metrics_cache")
        if not cache or not cache.get("num_owned"):
            continue

        start_idx = int(cache["param_start_idx"])
        num_owned = int(cache["num_owned"])
        grad_stack = cache["grad"]
        momentum_stack = cache["momentum"]
        nesterov_stack = cache["momentum_after_nesterov"]
        sigma_stack = cache.get("sigma")
        optimizer_kind = cache.get("optimizer_kind", group.get("kind", "unknown"))
        if grad_stack is None or momentum_stack is None or nesterov_stack is None:
            continue

        for local_idx in range(num_owned):
            global_idx = start_idx + local_idx
            if global_idx >= len(params):
                continue
            p = params[global_idx]
            name = param_name_by_id.get(id(p), f"group{group_idx}.param{global_idx}")
            if not _match_module(name, module_regex):
                continue

            grad = grad_stack[local_idx]
            momentum = momentum_stack[local_idx]
            m_tilde = nesterov_stack[local_idx]
            has_nonfinite = not (
                torch.isfinite(grad).all()
                and torch.isfinite(momentum).all()
                and torch.isfinite(m_tilde).all()
            )
            grad_f = torch.nan_to_num(grad.float())
            momentum_f = torch.nan_to_num(momentum.float())
            m_tilde_f = torch.nan_to_num(m_tilde.float())

            S_momentum = torch.linalg.svdvals(momentum_f)
            U, S_mtilde, Vh = torch.linalg.svd(m_tilde_f, full_matrices=False)
            k = min(top_k, int(S_mtilde.numel()))
            sigma_vals = sigma_stack[local_idx, :k] if sigma_stack is not None else None

            row = {
                "rank": rank,
                "optimizer_kind": optimizer_kind,
                "module": name,
                "shape": list(p.shape),
                "weight_rms": _rms(p.detach()),
                "grad_rms": _rms(grad_f),
                "momentum_rms": _rms(momentum_f),
                "momentum_after_nesterov_rms": _rms(m_tilde_f),
                "momentum_spectral_norm": float(S_momentum[0].detach().float().cpu()),
                "momentum_after_nesterov_spectral_norm": float(S_mtilde[0].detach().float().cpu()),
                "muon_singular_values": _to_float_list(S_mtilde[:k]),
                "has_nonfinite_optimizer_tensor": bool(has_nonfinite),
            }
            if sigma_vals is not None:
                row["streaming_sigma_values"] = _to_float_list(sigma_vals)
            local_rows.append(row)

            local_refs[name] = {
                "grad": grad_f.detach(),
                "momentum": momentum_f.detach(),
                "momentum_after_nesterov": m_tilde_f.detach(),
            }

            if save_components:
                local_components[name] = {
                    "muon_u_top_k": U[:, :k].detach().cpu(),
                    "muon_v_top_k": Vh[:k, :].mT.detach().cpu(),
                    "muon_singular_values": S_mtilde[:k].detach().cpu(),
                }

    if max_modules > 0:
        local_rows = local_rows[:max_modules]
        keep = {row["module"] for row in local_rows}
        local_refs = {k: v for k, v in local_refs.items() if k in keep}
        local_components = {k: v for k, v in local_components.items() if k in keep}

    component_path = None
    if save_components and local_components and component_dir is not None:
        out_dir = Path(component_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        component_path = out_dir / f"muon_components_step{step:06d}_rank{rank:03d}.pt"
        torch.save(local_components, component_path)
        for row in local_rows:
            row["component_file"] = str(component_path)

    gathered: list[list[dict[str, Any]]] | None = None
    if ddp:
        if rank == 0:
            gathered = [None for _ in range(world_size)]  # type: ignore[list-item]
        torch.distributed.gather_object(local_rows, object_gather_list=gathered, dst=0)
    else:
        gathered = [local_rows]

    if rank != 0:
        return None, {}

    rows = [row for part in (gathered or []) if part for row in part]
    if max_modules > 0:
        rows = rows[:max_modules]

    scalars: dict[str, float] = {}
    vectors: dict[str, list[float]] = {}
    per_module: dict[str, dict[str, Any]] = {}
    for row in rows:
        module = row["module"]
        per_module[module] = row
        scalars[f"weight_norm/{module}"] = row["weight_rms"]
        scalars[f"grad_norm/{module}"] = row["grad_rms"]
        scalars[f"momentum_norm/{module}"] = row["momentum_rms"]
        scalars[f"momentum_after_nesterov_norm/{module}"] = row["momentum_after_nesterov_rms"]
        scalars[f"momentum_spectral_norm/{module}"] = row["momentum_spectral_norm"]
        scalars[f"momentum_after_nesterov_spectral_norm/{module}"] = row["momentum_after_nesterov_spectral_norm"]
        vectors[f"muon_singular_values/{module}"] = row["muon_singular_values"]
        if "streaming_sigma_values" in row:
            vectors[f"streaming_sigma_values/{module}"] = row["streaming_sigma_values"]

    if train_loss is not None:
        scalars["train/loss"] = float(train_loss)
    if lr_multiplier is not None:
        scalars["train/lr_multiplier"] = float(lr_multiplier)
    if muon_momentum is not None:
        scalars["train/muon_momentum"] = float(muon_momentum)

    entry = {
        "step": step,
        "scalars": scalars,
        "vectors": vectors,
        "per_module": per_module,
    }
    if component_path is not None:
        entry["component_file_rank0"] = str(component_path)
    return entry, local_refs


def clear_metric_caches(optimizer: torch.optim.Optimizer) -> None:
    for group in optimizer.param_groups:
        group["_capture_metrics"] = False
        if group.get("kind") not in ("streaming_muon", "muon") or not group.get("params"):
            continue
        state = optimizer.state.get(group["params"][0], {})
        state.pop("metrics_cache", None)


@torch.no_grad()
def split_momentum_alignment(
    m_a: dict[str, torch.Tensor],
    m_b: dict[str, torch.Tensor],
    *,
    top_k: int,
    alignment_side: str = "lite",
    module_regex: str | None = r"transformer\.h",
    max_modules: int = 0,
) -> dict[str, Any]:
    from diagnose_split_batch import diagnose_from_grads, to_json

    names = [name for name in m_a if name in m_b and _match_module(name, module_regex)]
    if max_modules > 0:
        names = names[:max_modules]
    pairs = {name: (m_a[name], m_b[name]) for name in names}
    clean = to_json(diagnose_from_grads(pairs, top_k=top_k, alignment_side=alignment_side))

    vectors: dict[str, list[float]] = {}
    # Keep the exact per-direction c_i for the requested top-k; to_json only
    # stores summaries, so compute the vectors directly for active layers.
    from diagnose_split_batch import alignment_from_grad_pair
    for name, (a, b) in pairs.items():
        if float(a.float().norm().cpu()) == 0.0 and float(b.float().norm().cpu()) == 0.0:
            continue
        stats = alignment_from_grad_pair(a, b, top_k=top_k, alignment_side=alignment_side)
        vectors[f"variance_of_u_i/{name}"] = _to_float_list(stats["c_top_k"])
    clean["vectors"] = vectors
    return clean


@torch.enable_grad()
def hessian_power_probe(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    references: dict[str, dict[str, torch.Tensor]],
    name_to_param: dict[str, torch.nn.Parameter],
    *,
    top_k: int = 1,
    iters: int = 6,
    module_regex: str | None = r"transformer\.h",
    max_modules: int = 1,
) -> dict[str, Any]:
    """Approximate per-module top Hessian directions via block power iteration.

    This is expensive and intentionally supports a small module subset. The
    Hessian is the block Hessian with respect to one weight matrix at a time.
    """
    x, y = batch
    names = [n for n in references if n in name_to_param and _match_module(n, module_regex)]
    if max_modules > 0:
        names = names[:max_modules]

    was_training = model.training
    model.train()
    out: dict[str, Any] = {"per_module": {}, "scalars": {}, "vectors": {}}

    def hvp_for(param: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        model.zero_grad(set_to_none=True)
        with _math_sdpa_context():
            loss = model(x, y)
        grad = torch.autograd.grad(loss, param, create_graph=True, retain_graph=True)[0]
        hvp = torch.autograd.grad((grad.float() * vec).sum(), param, retain_graph=False)[0]
        return hvp.detach().float()

    for name in names:
        param = name_to_param[name]
        ref = references[name]
        evecs: list[torch.Tensor] = []
        evals: list[float] = []

        for _ in range(max(1, top_k)):
            v = torch.randn_like(param, dtype=torch.float32)
            v = v / (v.norm() + 1e-12)
            for _ in range(max(1, iters)):
                hv = hvp_for(param, v)
                for prev in evecs:
                    hv = hv - torch.sum(hv * prev) * prev
                hv_norm = hv.norm()
                if float(hv_norm.cpu()) == 0.0 or not torch.isfinite(hv_norm):
                    break
                v = hv / hv_norm
            hv = hvp_for(param, v)
            lam = float(torch.sum(v * hv).detach().cpu())
            evecs.append(v.detach())
            evals.append(lam)

        m_tilde = ref["momentum_after_nesterov"].to(param.device)
        U, _, Vh = torch.linalg.svd(m_tilde.float(), full_matrices=False)
        rank_count = min(len(evecs), U.shape[1], Vh.shape[0])
        component_alignments = []
        for i in range(rank_count):
            component = torch.outer(U[:, i], Vh[i, :])
            component_alignments.append(_cosine(evecs[i], component))

        grad_align = [_cosine(e, ref["grad"].to(e.device)) for e in evecs]
        momentum_align = [_cosine(e, ref["momentum"].to(e.device)) for e in evecs]
        nesterov_align = [_cosine(e, ref["momentum_after_nesterov"].to(e.device)) for e in evecs]

        module_entry = {
            "hessian_top_eigenvalues": evals,
            "sharpness": max(evals) if evals else float("nan"),
            "gradient_hessian_alignment": grad_align,
            "momentum_hessian_alignment": momentum_align,
            "momentum_after_nesterov_hessian_alignment": nesterov_align,
            "alignment_between_covariance_hessian_at_k_th_component": component_alignments,
        }
        out["per_module"][name] = module_entry
        out["scalars"][f"sharpness/{name}"] = module_entry["sharpness"]
        out["vectors"][f"gradient_hessian_alignment/{name}"] = grad_align
        out["vectors"][f"momentum_hessian_alignment/{name}"] = momentum_align
        out["vectors"][f"momentum_after_nesterov_hessian_alignment/{name}"] = nesterov_align
        out["vectors"][f"alignment_between_covariance_hessian_at_k_th_component/{name}"] = component_alignments

    model.zero_grad(set_to_none=True)
    if not was_training:
        model.eval()
    return out
