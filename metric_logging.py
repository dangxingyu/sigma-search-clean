"""Diagnostic metric logging for Muon-like training runs.

The logger is intentionally opt-in. It reads a short-lived optimizer cache that
`streaming_muon_torch.py` materializes only on requested logging steps, so
normal training does not pay extra memory or SVD cost.
"""

from __future__ import annotations

import math
import re
from contextlib import nullcontext
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


def _projection(unit_vec: torch.Tensor, tensor: torch.Tensor) -> float:
    """Signed Frobenius projection onto an approximately unit-norm direction."""
    unit_vec = unit_vec.detach().float().reshape(-1)
    tensor = torch.nan_to_num(tensor.detach().float()).to(unit_vec.device).reshape(-1)
    return float(torch.dot(unit_vec, tensor).cpu())


def _to_float_list(x: torch.Tensor) -> list[float]:
    return [float(v) for v in x.detach().float().cpu().tolist()]


def _match_module(name: str, module_regex: str | None) -> bool:
    return module_regex is None or module_regex == "" or re.search(module_regex, name) is not None


_NORMAL_MATRIX_RE = re.compile(
    r"^transformer\.h\.(?P<layer>\d+)\."
    r"(?P<block>attn|mlp)\.(?P<leaf>c_q|c_k|c_v|c_proj|c_fc)\.weight$"
)


def _unique_preserve_order(names: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _is_normal_matrix_weight(name: str) -> bool:
    """Transformer block matrix weights used by Muon-like optimizers."""
    return _NORMAL_MATRIX_RE.match(name) is not None


def _limit_modules(names: list[str], max_modules: int) -> list[str]:
    """Limit high-frequency metric rows with deterministic even spacing.

    `max_modules=0` means all matched modules. Hessian probes call this with
    zero so the global selected-subspace HVP covers every normal matrix weight.
    """
    names = _unique_preserve_order(names)
    if max_modules <= 0 or len(names) <= max_modules:
        return names
    if max_modules == 1:
        return [names[len(names) // 2]]
    idxs = [round(i * (len(names) - 1) / (max_modules - 1)) for i in range(max_modules)]
    return [names[i] for i in idxs]


def _selected_metric_names(
    optimizer: torch.optim.Optimizer,
    param_name_by_id: dict[int, str],
    module_regex: str | None,
    max_modules: int,
) -> list[str]:
    names: list[str] = []
    for group in optimizer.param_groups:
        if group.get("kind") not in ("streaming_muon", "muon"):
            continue
        for p in group.get("params", []):
            name = param_name_by_id.get(id(p))
            if name is not None and _is_normal_matrix_weight(name) and _match_module(name, module_regex):
                names.append(name)
    return _limit_modules(names, max_modules)


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
    collect_hessian_refs: bool = False,
) -> tuple[dict[str, Any] | None, dict[str, dict[str, torch.Tensor]]]:
    """Collect per-module scalar/vector diagnostics from cached Muon tensors.

    Returns `(entry, references)`. `entry` is populated only on rank 0. The
    `references` dict is rank-local and is meant for optional Hessian alignment
    probes on rank 0.
    """
    rank, world_size, ddp = _rank_info()
    local_rows: list[dict[str, Any]] = []
    local_refs: dict[str, dict[str, torch.Tensor]] = {}
    all_names = _selected_metric_names(optimizer, param_name_by_id, module_regex, 0)
    metric_names = _limit_modules(all_names, max_modules)
    metric_name_set = set(metric_names)
    ref_name_set = set(all_names if collect_hessian_refs else metric_names)
    active_name_set = metric_name_set | ref_name_set

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
        nesterov_stack = cache["momentum_after_nesterov"]
        sigma_stack = cache.get("sigma")
        basis_stack = cache.get("streaming_basis")
        streaming_transposed = bool(cache.get("streaming_transposed", False))
        streaming_left_uses_qr = bool(cache.get("streaming_left_uses_qr", False))
        optimizer_kind = cache.get("optimizer_kind", group.get("kind", "unknown"))
        if grad_stack is None or nesterov_stack is None:
            continue

        for local_idx in range(num_owned):
            global_idx = start_idx + local_idx
            if global_idx >= len(params):
                continue
            p = params[global_idx]
            name = param_name_by_id.get(id(p), f"group{group_idx}.param{global_idx}")
            if name not in active_name_set:
                continue

            grad = grad_stack[local_idx]
            m_tilde = nesterov_stack[local_idx]
            grad_f = torch.nan_to_num(grad.float())
            m_tilde_f = torch.nan_to_num(m_tilde.float())

            if name in ref_name_set:
                local_refs[name] = {
                    "grad": grad_f.detach(),
                    "momentum_after_nesterov": m_tilde_f.detach(),
                }
                if basis_stack is not None and sigma_stack is not None:
                    local_refs[name]["streaming_basis"] = basis_stack[local_idx].detach().float()
                    local_refs[name]["streaming_sigma"] = sigma_stack[local_idx].detach().float()
                    local_refs[name]["streaming_transposed"] = streaming_transposed
                    local_refs[name]["streaming_left_uses_qr"] = streaming_left_uses_qr
            if name in metric_name_set:
                has_nonfinite = not (
                    torch.isfinite(grad).all()
                    and torch.isfinite(m_tilde).all()
                )
                sigma_vals = sigma_stack[local_idx].detach().float() if sigma_stack is not None else None
                if sigma_vals is not None:
                    # StreamingMuon already computed sigma for its sigma transform.
                    singular_values = torch.sort(sigma_vals, descending=True).values
                    singular_source = "streaming_sigma"
                    mtilde_spectral_norm = float(singular_values[0].detach().float().cpu())
                else:
                    singular_values = torch.empty(0, dtype=torch.float32, device=m_tilde_f.device)
                    singular_source = "unavailable_no_streaming_sigma"
                    mtilde_spectral_norm = None
                k = min(top_k, int(singular_values.numel()))

                row = {
                    "rank": rank,
                    "optimizer_kind": optimizer_kind,
                    "module": name,
                    "shape": list(p.shape),
                    "singular_value_source": singular_source,
                    "weight_rms": _rms(p.detach()),
                    "grad_rms": _rms(grad_f),
                    "momentum_after_nesterov_rms": _rms(m_tilde_f),
                    "muon_singular_values": _to_float_list(singular_values[:k]),
                    "has_nonfinite_optimizer_tensor": bool(has_nonfinite),
                }
                if mtilde_spectral_norm is not None:
                    row["momentum_after_nesterov_spectral_norm"] = mtilde_spectral_norm
                if sigma_vals is not None:
                    row["streaming_sigma_values"] = _to_float_list(singular_values[:k])
                local_rows.append(row)

    gathered: list[list[dict[str, Any]]] | None = None
    gathered_refs: list[dict[str, dict[str, torch.Tensor]]] | None = None
    local_refs_cpu = {
        name: {
            key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for key, value in refs.items()
        }
        for name, refs in local_refs.items()
    }
    if ddp:
        if rank == 0:
            gathered = [None for _ in range(world_size)]  # type: ignore[list-item]
            gathered_refs = [None for _ in range(world_size)]  # type: ignore[list-item]
        torch.distributed.gather_object(local_rows, object_gather_list=gathered, dst=0)
        torch.distributed.gather_object(local_refs_cpu, object_gather_list=gathered_refs, dst=0)
    else:
        gathered = [local_rows]
        gathered_refs = [local_refs_cpu]

    if rank != 0:
        return None, {}

    rows = [row for part in (gathered or []) if part for row in part]
    row_by_module = {row["module"]: row for row in rows}
    rows = [row_by_module[name] for name in metric_names if name in row_by_module]
    rank0_refs: dict[str, dict[str, torch.Tensor]] = {}
    for part in gathered_refs or []:
        if part:
            rank0_refs.update(part)
    ref_names_ordered = all_names if collect_hessian_refs else metric_names
    rank0_refs = {name: rank0_refs[name] for name in ref_names_ordered if name in rank0_refs}

    scalars: dict[str, float] = {}
    vectors: dict[str, list[float]] = {}
    per_module: dict[str, dict[str, Any]] = {}
    for row in rows:
        module = row["module"]
        per_module[module] = {
            "rank": row["rank"],
            "optimizer_kind": row["optimizer_kind"],
            "shape": row["shape"],
            "singular_value_source": row["singular_value_source"],
            "has_nonfinite_optimizer_tensor": row["has_nonfinite_optimizer_tensor"],
        }
        scalars[f"weight_norm/{module}"] = row["weight_rms"]
        scalars[f"grad_norm/{module}"] = row["grad_rms"]
        scalars[f"momentum_after_nesterov_norm/{module}"] = row["momentum_after_nesterov_rms"]
        if "momentum_after_nesterov_spectral_norm" in row:
            scalars[f"momentum_after_nesterov_spectral_norm/{module}"] = row["momentum_after_nesterov_spectral_norm"]
        if row["muon_singular_values"]:
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
        "metadata": {
            "module_selection": "normal_attention_mlp_weights",
            "metric_modules": metric_names,
            "hessian_reference_modules": ref_names_ordered if collect_hessian_refs else [],
            "streaming_singular_values": "reuse_cached_sigma_no_explicit_svd",
        },
        "scalars": scalars,
        "vectors": vectors,
        "per_module": per_module,
    }
    return entry, rank0_refs


def clear_metric_caches(optimizer: torch.optim.Optimizer) -> None:
    for group in optimizer.param_groups:
        group["_capture_metrics"] = False
        if group.get("kind") not in ("streaming_muon", "muon") or not group.get("params"):
            continue
        state = optimizer.state.get(group["params"][0], {})
        state.pop("metrics_cache", None)


@torch.no_grad()
def gradient_projection_onto_hessian_space(
    references: dict[str, dict[str, torch.Tensor]],
    hessian_space: dict[str, Any],
) -> dict[str, Any] | None:
    """Project current gradients onto a fixed previously-computed Hessian space."""
    names = hessian_space.get("names")
    evecs = hessian_space.get("evecs")
    if not names or not evecs:
        return None
    if any(name not in references or "grad" not in references[name] for name in names):
        return None

    grad = [references[name]["grad"].detach().float().cpu() for name in names]
    coeffs: list[float] = []
    projection = [torch.zeros_like(block, dtype=torch.float32) for block in evecs[0]]
    for basis_vec in evecs:
        coeff = torch.zeros((), dtype=torch.float32)
        for e_block, g_block in zip(basis_vec, grad):
            coeff = coeff + torch.sum(e_block.float() * g_block.float())
        coeffs.append(float(coeff))
        for i, e_block in enumerate(basis_vec):
            projection[i] = projection[i] + coeff * e_block.float()

    norm_sq = torch.zeros((), dtype=torch.float32)
    for block in projection:
        norm_sq = norm_sq + torch.sum(block * block)
    coeff_norm = math.sqrt(sum(c * c for c in coeffs))
    return {
        "names": list(names),
        "hessian_step": hessian_space.get("step"),
        "coefficients": coeffs,
        "projection": [block.cpu() for block in projection],
        "norm": float(torch.sqrt(norm_sq).cpu()),
        "top1_abs_fraction": abs(coeffs[0]) / coeff_norm if coeff_norm > 0.0 else float("nan"),
    }


@torch.no_grad()
def projection_coefficients_pearson(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Pearson correlation between two fixed-Hessian-space coefficient vectors.

    This is the statistic we want for `c_t = E^T g_t` when Hessian top-k > 1.
    For top_k=1, component-wise Pearson is undefined; use the rolling lag-1
    Pearson of the signed top-1 coefficient time series instead.
    """
    if a.get("names") != b.get("names") or a.get("hessian_step") != b.get("hessian_step"):
        return float("nan")
    av = [float(x) for x in (a.get("coefficients") or [])]
    bv = [float(x) for x in (b.get("coefficients") or [])]
    if len(av) != len(bv) or len(av) < 2:
        return float("nan")
    av = [x for x in av if math.isfinite(x)]
    bv = [y for y in bv if math.isfinite(y)]
    if len(av) != len(bv) or len(av) < 2:
        return float("nan")
    ma = sum(av) / len(av)
    mb = sum(bv) / len(bv)
    num = sum((x - ma) * (y - mb) for x, y in zip(av, bv))
    va = sum((x - ma) * (x - ma) for x in av)
    vb = sum((y - mb) * (y - mb) for y in bv)
    denom = math.sqrt(va * vb)
    return num / denom if denom > 0.0 else float("nan")


def projection_lag1_pearson(values: list[float], window: int = 16, min_pairs: int = 4) -> float:
    """Lag-1 Pearson correlation for a scalar projection time series."""
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if window > 0:
        vals = vals[-(window + 1) :]
    if len(vals) < min_pairs + 1:
        return float("nan")
    x = vals[:-1]
    y = vals[1:]
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    vx = sum((a - mx) * (a - mx) for a in x)
    vy = sum((b - my) * (b - my) for b in y)
    denom = math.sqrt(vx * vy)
    return num / denom if denom > 0.0 else float("nan")


# Deprecated compatibility name for older analysis imports.
projection_vector_correlation = projection_coefficients_pearson


@torch.enable_grad()
def hessian_power_probe(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor] | list[tuple[torch.Tensor, torch.Tensor]],
    references: dict[str, dict[str, torch.Tensor]],
    name_to_param: dict[str, torch.nn.Parameter],
    *,
    top_k: int = 1,
    iters: int = 6,
    module_regex: str | None = r"transformer\.h",
    max_modules: int = 0,
    selected_names: list[str] | None = None,
    return_stats: bool = True,
) -> dict[str, Any]:
    """Approximate top Hessian directions over the selected-parameter subspace.

    One HVP covers all selected modules at once, including cross-module Hessian
    blocks inside this subspace. The resulting eigenvectors are split back into
    per-module blocks for gradient/momentum/Muon-component projections.
    """
    batches = batch if isinstance(batch, list) else [batch]
    rank, world_size, ddp = _rank_info()
    source_names = selected_names if selected_names is not None else list(references)
    names = [
        n for n in source_names
        if n in name_to_param and _is_normal_matrix_weight(n) and _match_module(n, module_regex)
    ]
    names = _limit_modules(names, max_modules)
    params = [name_to_param[n] for n in names]

    was_training = model.training
    model.train()
    out: dict[str, Any] = {
        "per_module": {},
        "scalars": {},
        "vectors": {},
        "metadata": {
            "hessian_probe_method": "selected_subspace_lanczos_top_algebraic",
            "hessian_probe_scope": "distributed_rank_average_loss_hvp" if ddp else "single_process_loss_hvp",
            "hessian_grad_accum_scope": "caller_supplied_batches",
            "local_hessian_batches": len(batches),
            "world_size": world_size,
            "selected_modules": names,
        },
    }
    if not names:
        return out

    def _global_norm(vecs: list[torch.Tensor]) -> torch.Tensor:
        total = torch.zeros((), device=vecs[0].device, dtype=torch.float32)
        for v in vecs:
            total = total + torch.sum(v.float() * v.float())
        return torch.sqrt(total)

    def _dot_vecs(a: list[torch.Tensor], b: list[torch.Tensor]) -> torch.Tensor:
        total = torch.zeros((), device=a[0].device, dtype=torch.float32)
        for x_i, y_i in zip(a, b):
            total = total + torch.sum(x_i.float() * y_i.float())
        return total

    def _scale_vecs(vecs: list[torch.Tensor], scale: torch.Tensor | float) -> list[torch.Tensor]:
        return [v * scale for v in vecs]

    def _sub_vecs(a: list[torch.Tensor], b: list[torch.Tensor], scale: torch.Tensor | float = 1.0) -> list[torch.Tensor]:
        return [x_i - scale * y_i for x_i, y_i in zip(a, b)]

    def _zeros_like_params() -> list[torch.Tensor]:
        return [torch.zeros_like(p, dtype=torch.float32) for p in params]

    def _broadcast_vecs(vecs: list[torch.Tensor]) -> list[torch.Tensor]:
        if ddp:
            for v in vecs:
                torch.distributed.broadcast(v, src=0)
        return vecs

    def hvp_for(vecs: list[torch.Tensor]) -> list[torch.Tensor]:
        accum = _zeros_like_params()
        for x, y in batches:
            model.zero_grad(set_to_none=True)
            with _math_sdpa_context():
                loss = model(x, y)
            grads = torch.autograd.grad(
                loss,
                params,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )
            dot = torch.zeros((), device=params[0].device, dtype=torch.float32)
            for grad, vec in zip(grads, vecs):
                if grad is not None:
                    dot = dot + torch.sum(grad.float() * vec.to(grad.device).float())
            hvps = torch.autograd.grad(dot, params, retain_graph=False, allow_unused=True)
            for i, (p, hv) in enumerate(zip(params, hvps)):
                accum[i] = accum[i] + (
                    torch.zeros_like(p, dtype=torch.float32) if hv is None else hv.detach().float()
                )
        scale = 1.0 / max(1, len(batches))
        accum = [v * scale for v in accum]
        if ddp:
            for v in accum:
                torch.distributed.all_reduce(v, op=torch.distributed.ReduceOp.AVG)
        return accum

    def lanczos_top_eigenpairs(num_pairs: int, lanczos_steps: int) -> tuple[list[float], list[list[torch.Tensor]]]:
        q = [torch.randn_like(p, dtype=torch.float32) for p in params]
        q = _broadcast_vecs(q)
        q = _scale_vecs(q, 1.0 / (_global_norm(q) + 1e-12))
        q_prev = _zeros_like_params()
        beta_prev = torch.tensor(0.0, device=q[0].device, dtype=torch.float32)
        basis: list[list[torch.Tensor]] = []
        alphas: list[torch.Tensor] = []
        betas: list[torch.Tensor] = []

        for j in range(max(1, lanczos_steps)):
            basis.append([v.detach().clone() for v in q])
            z = hvp_for(q)
            if j > 0:
                z = _sub_vecs(z, q_prev, beta_prev)
            alpha = _dot_vecs(q, z)
            z = _sub_vecs(z, q, alpha)
            # Full reorthogonalization is cheap at our tiny Hessian probe ranks
            # and avoids duplicate Ritz directions from numerical drift.
            for prev in basis:
                z = _sub_vecs(z, prev, _dot_vecs(z, prev))
            beta = _global_norm(z)
            alphas.append(alpha.detach().float())
            if j < max(1, lanczos_steps) - 1:
                betas.append(beta.detach().float())
            if float(beta.cpu()) == 0.0 or not torch.isfinite(beta):
                break
            q_prev = q
            q = _scale_vecs(z, 1.0 / beta)
            beta_prev = beta.detach().float()

        dim = len(alphas)
        if dim == 0:
            return [], []
        diag = torch.stack(alphas)
        T = torch.diag(diag)
        if dim > 1:
            off = torch.stack(betas[: dim - 1])
            T = T + torch.diag(off, diagonal=1) + torch.diag(off, diagonal=-1)
        evals, coeffs = torch.linalg.eigh(T)
        order = torch.argsort(evals, descending=True)[: min(num_pairs, dim)]

        out_evals: list[float] = []
        out_vecs: list[list[torch.Tensor]] = []
        for idx in order:
            coeff = coeffs[:, idx]
            vec = _zeros_like_params()
            for t in range(dim):
                vec = [v_i + coeff[t] * basis[t][j] for j, v_i in enumerate(vec)]
            vec = _scale_vecs(vec, 1.0 / (_global_norm(vec) + 1e-12))
            out_evals.append(float(evals[idx].detach().cpu()))
            out_vecs.append([v.detach() for v in vec])
        return out_evals, out_vecs

    evals, evecs = lanczos_top_eigenpairs(
        num_pairs=max(1, top_k),
        lanczos_steps=max(max(1, iters), max(1, top_k) + 1),
    )
    out["global_hessian_top_eigenvalues"] = evals
    out["global_sharpness"] = evals[0] if evals else float("nan")
    if not return_stats:
        model.zero_grad(set_to_none=True)
        if not was_training:
            model.eval()
        return out

    out["scalars"]["sharpness/selected_subspace"] = out["global_sharpness"]
    out["_transient"] = {
        "hessian_space": {
            "names": names,
            "evecs": [[block.detach().float().cpu() for block in vecs] for vecs in evecs],
        }
    }
    missing_refs = [n for n in names if n not in references]
    if missing_refs:
        out["metadata"]["missing_reference_modules"] = missing_refs
        model.zero_grad(set_to_none=True)
        if not was_training:
            model.eval()
        return out
    global_grad = [references[n]["grad"].to(params[i].device) for i, n in enumerate(names)]
    global_mtilde = [references[n]["momentum_after_nesterov"].to(params[i].device) for i, n in enumerate(names)]

    def _global_projection(vecs: list[torch.Tensor], refs: list[torch.Tensor]) -> float:
        return float(_dot_vecs(vecs, refs).detach().cpu())

    def _global_cosine(vecs: list[torch.Tensor], refs: list[torch.Tensor]) -> float:
        denom = _global_norm(vecs) * _global_norm(refs)
        if float(denom.detach().cpu()) == 0.0:
            return float("nan")
        return float((_dot_vecs(vecs, refs) / denom).detach().cpu())

    out["vectors"]["gradient_hessian_projection/selected_subspace"] = [
        _global_projection(vecs, global_grad) for vecs in evecs
    ]
    out["vectors"]["momentum_after_nesterov_hessian_projection/selected_subspace"] = [
        _global_projection(vecs, global_mtilde) for vecs in evecs
    ]
    out["vectors"]["gradient_hessian_alignment/selected_subspace"] = [
        _global_cosine(vecs, global_grad) for vecs in evecs
    ]
    out["vectors"]["momentum_after_nesterov_hessian_alignment/selected_subspace"] = [
        _global_cosine(vecs, global_mtilde) for vecs in evecs
    ]

    def _component_basis(ref: dict[str, Any], count: int) -> tuple[list[torch.Tensor], str]:
        """Return unit-norm components reconstructed from cached StreamingMuon state.

        If the optimizer cache is unavailable, component metrics are omitted
        rather than running a per-module SVD on Hessian logging steps.
        """
        m_tilde = ref["momentum_after_nesterov"].float()
        basis = ref.get("streaming_basis")
        sigma = ref.get("streaming_sigma")
        if isinstance(basis, torch.Tensor) and isinstance(sigma, torch.Tensor):
            device = m_tilde.device
            basis = basis.to(device=device, dtype=torch.float32)
            sigma = sigma.to(device=device, dtype=torch.float32)
            transposed = bool(ref.get("streaming_transposed", m_tilde.shape[-2] < m_tilde.shape[-1]))
            working = m_tilde.mT if transposed else m_tilde
            rotated = torch.matmul(working, basis)
            left = rotated / sigma.clamp_min(1e-12).unsqueeze(-2)
            order = torch.argsort(sigma, descending=True)[: min(count, int(sigma.numel()))]
            components = []
            for idx in order:
                component = torch.outer(left[:, idx], basis[:, idx])
                if transposed:
                    component = component.mT
                components.append(component / (component.norm() + 1e-12))
            return components, "streaming_basis"

        return [], "unavailable_no_streaming_basis"

    for block_idx, name in enumerate(names):
        ref = references[name]
        block_evecs = [vecs[block_idx] for vecs in evecs]
        block_norms = [float(v.norm().detach().cpu()) for v in block_evecs]
        block_unit_evecs = [v / (v.norm() + 1e-12) for v in block_evecs]

        components, component_source = _component_basis(ref, max(1, top_k))
        muon_rank_count = len(components)
        component_signed_matrix = []
        component_abs_matrix = []
        for e_unit, e_raw in zip(block_unit_evecs, block_evecs):
            signed_row = []
            abs_row = []
            for component in components:
                component = component.to(e_unit.device)
                signed = _projection(e_raw, component)
                cosine = _projection(e_unit, component)
                signed_row.append(signed)
                abs_row.append(abs(cosine))
            component_signed_matrix.append(signed_row)
            component_abs_matrix.append(abs_row)
        diag_count = min(len(block_unit_evecs), muon_rank_count)
        component_alignments = [
            component_abs_matrix[i][i] for i in range(diag_count)
        ]

        grad_cos = [_cosine(e, ref["grad"].to(e.device)) for e in block_unit_evecs]
        nesterov_cos = [_cosine(e, ref["momentum_after_nesterov"].to(e.device)) for e in block_unit_evecs]
        grad_proj = [_projection(e, ref["grad"].to(e.device)) for e in block_evecs]
        nesterov_proj = [_projection(e, ref["momentum_after_nesterov"].to(e.device)) for e in block_evecs]

        module_entry = {
            "hessian_top_eigenvalues": evals,
            "sharpness": evals[0] if evals else float("nan"),
            "hessian_probe_method": "selected_subspace_lanczos_top_algebraic",
            "muon_component_source": component_source,
            "hessian_eigenvector_block_norm": block_norms,
            "gradient_hessian_alignment": grad_cos,
            "momentum_after_nesterov_hessian_alignment": nesterov_cos,
            "gradient_hessian_abs_alignment": [abs(v) for v in grad_cos],
            "momentum_after_nesterov_hessian_abs_alignment": [abs(v) for v in nesterov_cos],
            "gradient_hessian_projection": grad_proj,
            "momentum_after_nesterov_hessian_projection": nesterov_proj,
            "alignment_between_covariance_hessian_at_k_th_component": component_alignments,
            "hessian_muon_component_alignment_matrix": component_abs_matrix,
            "hessian_muon_component_signed_projection_matrix": component_signed_matrix,
        }
        out["per_module"][name] = module_entry
        out["scalars"][f"hessian_eigenvector_block_norm/{name}"] = block_norms[0] if block_norms else float("nan")
        out["vectors"][f"gradient_hessian_alignment/{name}"] = grad_cos
        out["vectors"][f"momentum_after_nesterov_hessian_alignment/{name}"] = nesterov_cos
        out["vectors"][f"gradient_hessian_abs_alignment/{name}"] = module_entry["gradient_hessian_abs_alignment"]
        out["vectors"][f"momentum_after_nesterov_hessian_abs_alignment/{name}"] = module_entry["momentum_after_nesterov_hessian_abs_alignment"]
        out["vectors"][f"gradient_hessian_projection/{name}"] = grad_proj
        out["vectors"][f"momentum_after_nesterov_hessian_projection/{name}"] = nesterov_proj
        out["vectors"][f"alignment_between_covariance_hessian_at_k_th_component/{name}"] = component_alignments
        out["vectors"][f"hessian_muon_component_alignment_matrix/{name}"] = component_abs_matrix
        out["vectors"][f"hessian_eigenvector_block_norm/{name}"] = block_norms

    model.zero_grad(set_to_none=True)
    if not was_training:
        model.eval()
    return out
