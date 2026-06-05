#!/usr/bin/env python3
"""
Eval runner for f(Σ) search: trains a small nanochat model with StreamingMuon
for a short horizon and reports val BPB.

Usage:
    python run_eval.py --candidate-file candidate_code.py --max-steps 500

The candidate file should define a function `f(sigma, state) -> Tensor`.
Results are written to --output-file as JSON.
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import sys
import gc
import json
import time
import math
import argparse
import re
import signal
from pathlib import Path

import torch

# Add nanochat to path
SCRIPT_DIR = Path(__file__).resolve().parent
NANOCHAT_DIR = SCRIPT_DIR / "nanochat" if (SCRIPT_DIR / "nanochat").exists() else None

parser = argparse.ArgumentParser(description="Eval runner for f(Σ) search")
parser.add_argument("--nanochat-dir", type=str, default=str(NANOCHAT_DIR), help="Path to nanochat repo")
parser.add_argument("--optimizer", type=str, default="streaming_muon",
                    choices=["streaming_muon", "plain_muon", "muon", "native_muon", "adamw", "soap", "shampoo", "kl_shampoo", "kl_soap"],
                    help="Matrix-weight optimizer baseline.")
parser.add_argument("--candidate-file", type=str, default="candidates/identity.py",
                    help="Python file defining f(sigma, state); only used by --optimizer streaming_muon.")
parser.add_argument("--candidate-code", type=str, default=None, help="Inline Python code for f(sigma, state)")
parser.add_argument("--candidate-param", action="append", default=[],
                    help="Candidate hyperparameter as key=value. Values are JSON-parsed when possible; repeatable.")
parser.add_argument("--output-file", type=str, default="eval_result.json", help="Output JSON file")
# Model
parser.add_argument("--depth", type=int, default=4, help="Model depth (small for search)")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=64, help="Head dimension")
parser.add_argument("--max-seq-len", type=int, default=512, help="Max sequence length")
parser.add_argument("--architecture", type=str, default="gpt2", choices=["gpt2", "nanochat", "qwen3"],
                    help="Model architecture. gpt2 is the simplified control; nanochat keeps the original modded block; qwen3 uses RoPE, RMSNorm, QK norm, and SwiGLU.")
# Training
parser.add_argument("--max-steps", type=int, default=500, help="Training steps")
parser.add_argument("--device-batch-size", type=int, default=8, help="Per-device batch size")
parser.add_argument("--total-batch-size", type=int, default=-1, help="Total batch size (-1 = auto)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="Muon LR")
parser.add_argument(
    "--matrix-lr-adjust",
    type=str,
    default="moonlight",
    choices=["none", "moonlight"],
    help="Per-matrix LR factor for Muon/structured matrix updates. "
    "moonlight uses 0.2*sqrt(max(rows, cols)) (Moonlight RMS matching).",
)
parser.add_argument(
    "--batch-beta-align",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Rescale EMA betas/momentum so half-life in tokens is constant across batch sizes.",
)
parser.add_argument(
    "--batch-beta-align-mode",
    type=str,
    default="all",
    choices=["all", "beta2_only", "none"],
    help="Which betas to batch-align when --batch-beta-align is set. "
    "all scales beta1 and beta2/shampoo_beta; beta2_only keeps beta1 constant "
    "and scales beta2/shampoo_beta/adam_beta2.",
)
parser.add_argument(
    "--adam-lr-mode",
    type=str,
    default="relative_to_matrix",
    choices=["relative_to_matrix", "nanochat_fixed"],
    help="How to set AdamW peak LRs for embeddings/lm_head. "
    "relative_to_matrix scales them with --matrix-lr; nanochat_fixed keeps legacy constants.",
)
parser.add_argument("--weight-decay", type=float, default=0.28, help="Weight decay")
parser.add_argument("--warmup-steps", type=int, default=40, help="LR warmup steps")
parser.add_argument("--warmdown-ratio", type=float, default=0.65, help="Fraction of training in LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.05, help="Final LR as fraction of peak")
parser.add_argument("--muon-momentum", type=float, default=0.95, help="Static Muon momentum when --muon-momentum-schedule=static.")
parser.add_argument(
    "--muon-momentum-schedule",
    type=str,
    default="nanochat",
    choices=["nanochat", "static"],
    help="Muon momentum schedule. static keeps --muon-momentum fixed for all steps.",
)
parser.add_argument("--mom-decay", action="store_true", default=True, help="Decay momentum 0.97→0.90 during warmdown")
parser.add_argument("--no-mom-decay", dest="mom_decay", action="store_false", help="Keep momentum constant at 0.97")
# StreamingMuon
parser.add_argument("--k", type=int, default=-1, help="Number of singular vectors (-1 = full rank)")
parser.add_argument("--num-iters", type=int, default=1, help="Number of streaming power iterations (1 or 2)")
parser.add_argument("--pure-qr", action="store_true", default=False, help="Use Householder QR instead of SCQR in StreamingMuon orthogonalization.")
parser.add_argument("--fallback-ortho-tol", type=float, default=0.02, help="SCQR fallback orthogonality tolerance: when max ||Q^T Q - I|| > tol, fall back to Householder QR. Pass a negative value to disable the orthogonality check.")
parser.add_argument("--precondition-frequency", type=int, default=5,
                    help="SOAP/Shampoo/KL baseline eigenbasis refresh frequency.")
parser.add_argument("--shampoo-beta", type=float, default=0.95,
                    help="EMA beta for SOAP/Shampoo/KL preconditioner factors.")
parser.add_argument("--optimizer-beta1", type=float, default=0.9,
                    help="First-moment beta for non-Muon matrix optimizer baselines.")
parser.add_argument("--optimizer-beta2", type=float, default=0.95,
                    help="Second-moment/RMS beta for SOAP/KL-SOAP.")
parser.add_argument("--structured-init-factor", type=float, default=1.0,
                    help="Initial diagonal factor for Shampoo/SOAP/KL baselines.")
parser.add_argument("--structured-use-qr", action=argparse.BooleanOptionalAction, default=True,
                    help="Use QR power refreshes for structured optimizer eigenbases after initialization.")
# Eval
parser.add_argument("--eval-every", type=int, default=100, help="Evaluate val BPB every N steps")
parser.add_argument("--eval-tokens", type=int, default=524288, help="Tokens for val eval")
# Checkpoint / resume
parser.add_argument("--checkpoint-dir", type=str, default=None,
                    help="Directory for resumable model/optimizer checkpoints. Disabled if omitted.")
parser.add_argument("--save-every", type=int, default=0,
                    help="Save a resumable checkpoint every N optimizer steps; 0 disables periodic saves.")
parser.add_argument("--keep-last-checkpoints", type=int, default=2,
                    help="Keep only the latest N complete checkpoints; <=0 keeps all.")
parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                    help="Resume from --checkpoint-dir if a complete checkpoint exists.")
parser.add_argument("--resume-from-step", type=int, default=-1,
                    help="-1 means latest complete checkpoint, >=0 means exact step.")
# Diagnostics
parser.add_argument("--save-sigma-profile", type=str, default=None,
                    help="Path to save sigma distribution stats (JSON). Useful for profiling baseline.")
parser.add_argument("--metrics-every", type=int, default=0,
                    help="If >0, log Muon-like diagnostics every N optimizer steps into result.json.")
parser.add_argument("--metrics-top-k", type=int, default=4,
                    help="Top singular directions to summarize in diagnostic logging.")
parser.add_argument("--metrics-module-regex", type=str,
                    default=r"transformer\.h\.(?:[0-9]+)\.(?:attn\.(?:c_q|c_k|c_v|c_proj)|mlp\.(?:c_gate|c_fc|c_proj))\.weight$",
                    help="Regex selecting modules for diagnostic logging.")
parser.add_argument("--metrics-max-modules", type=int, default=0,
                    help="Maximum modules to log per step; 0 means all selected modules.")
parser.add_argument("--metrics-hessian-every", type=int, default=0,
                    help="If >0, run expensive Hessian power diagnostics every N logged steps.")
parser.add_argument("--metrics-hessian-top-k", type=int, default=1,
                    help="Number of Hessian directions for the expensive Hessian probe.")
parser.add_argument("--metrics-hessian-iters", type=int, default=6,
                    help="Power iterations per Hessian direction.")
parser.add_argument("--metrics-hessian-max-modules", type=int, default=0,
                    help="Maximum modules in Hessian selected subspace; 0 means all normal matrix weights.")
parser.add_argument("--metrics-projection-correlation-window", type=int, default=16,
                    help="Window length for lag-1 Pearson correlation of top Hessian-direction gradient projections.")
# Device
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty=auto)")
parser.add_argument("--seed", type=int, default=42, help="Random seed for model init and data")
args = parser.parse_args()

# Set seeds early for reproducibility
import random
random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# Insert nanochat into path
sys.path.insert(0, args.nanochat_dir)
sys.path.insert(0, str(SCRIPT_DIR))

from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import (
    tokenizing_distributed_data_loader_bos_bestfit,
    tokenizing_distributed_data_loader_with_state_bos_bestfit,
)
from nanochat.common import COMPUTE_DTYPE, print0, autodetect_device_type
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.loss_eval import evaluate_bpb
from streaming_muon_torch import StreamingMuonAdamW, DistStreamingMuonAdamW, CustomTransform
from optimizer_recipe import (
    B_REF,
    adam_lr_from_matrix_lr,
    batch_lr_scale as recipe_batch_lr_scale,
    ema_beta_for_batch,
)

# =============================================================================
# Load candidate f(Σ)
# =============================================================================

def load_candidate_fn(candidate_file: str, candidate_code: str = None):
    """Load the candidate f(sigma, state) function."""
    if candidate_code:
        code = candidate_code
    else:
        code = Path(candidate_file).read_text()

    # Execute in a namespace with torch available
    ns = {"torch": torch, "math": math}
    exec(code, ns)

    # Look for 'f' or '_candidate_f'
    if 'f' in ns and callable(ns['f']):
        return ns['f']
    elif '_candidate_f' in ns and callable(ns['_candidate_f']):
        return ns['_candidate_f']
    else:
        # Try treating as a lambda
        code = code.strip()
        if code.startswith("lambda"):
            return eval(code, ns)
        raise ValueError("Candidate file must define f(sigma, state) or _candidate_f(sigma, state)")


def _parse_candidate_value(raw: str):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        lowered = raw.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered in ("none", "null"):
            return None
        return raw


def parse_candidate_params(items: list[str]) -> dict:
    params = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--candidate-param must be key=value, got {item!r}")
        key, raw_value = item.split("=", 1)
        key = key.strip().replace("-", "_")
        if not key:
            raise ValueError(f"empty --candidate-param key in {item!r}")
        params[key] = _parse_candidate_value(raw_value.strip())
    return params


def configure_candidate_fn(fn, params: dict):
    if not params:
        return fn
    frozen_params = dict(params)

    def wrapped(sigma, state):
        state.setdefault("candidate_params", frozen_params)
        for key, value in frozen_params.items():
            state.setdefault(key, value)
        return fn(sigma, state)

    return wrapped


_STOP_REQUESTED = False


def _handle_stop_signal(signum, frame):
    del frame
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(f"Received signal {signum}; will checkpoint after the current optimizer step.", flush=True)


def _install_signal_handlers():
    signal.signal(signal.SIGTERM, _handle_stop_signal)
    signal.signal(signal.SIGINT, _handle_stop_signal)


def _step_from_checkpoint_path(path: Path, prefix: str) -> int | None:
    stem = path.stem
    if not stem.startswith(prefix):
        return None
    try:
        return int(stem.split("_")[1])
    except (IndexError, ValueError):
        return None


def _checkpoint_is_complete(checkpoint_dir: Path, step: int, world_size: int) -> bool:
    if not (checkpoint_dir / f"model_{step:06d}.pt").exists():
        return False
    if not (checkpoint_dir / f"meta_{step:06d}.json").exists():
        return False
    return all((checkpoint_dir / f"optim_{step:06d}_rank{rank:d}.pt").exists() for rank in range(world_size))


def _complete_checkpoint_steps(checkpoint_dir: Path, world_size: int) -> list[int]:
    if not checkpoint_dir.exists():
        return []
    steps = []
    for model_path in checkpoint_dir.glob("model_*.pt"):
        step = _step_from_checkpoint_path(model_path, "model_")
        if step is not None and _checkpoint_is_complete(checkpoint_dir, step, world_size):
            steps.append(step)
    return sorted(set(steps))


def _resolve_resume_step(
    checkpoint_dir: Path | None,
    resume: bool,
    resume_from_step: int,
    world_size: int,
    rank: int,
    ddp: bool,
) -> int | None:
    if checkpoint_dir is None or not resume:
        return None

    step = None
    if rank == 0:
        if resume_from_step >= 0:
            if not _checkpoint_is_complete(checkpoint_dir, resume_from_step, world_size):
                raise FileNotFoundError(
                    f"requested checkpoint step {resume_from_step} is incomplete in {checkpoint_dir}"
                )
            step = resume_from_step
        else:
            steps = _complete_checkpoint_steps(checkpoint_dir, world_size)
            step = steps[-1] if steps else None

    if ddp:
        obj = [step]
        torch.distributed.broadcast_object_list(obj, src=0)
        step = obj[0]
    return step


def _optimizer_state_for_checkpoint(optimizer: torch.optim.Optimizer) -> dict:
    """Drop callable/transient objects so optimizer checkpoints are torch.save-safe."""
    raw = optimizer.state_dict()
    state = {}
    for key, value in raw["state"].items():
        if isinstance(value, dict):
            state[key] = {
                k: v for k, v in value.items()
                if k not in {"sigma_transform_obj", "metrics_cache"}
            }
        else:
            state[key] = value

    param_groups = []
    for group in raw["param_groups"]:
        param_groups.append({
            k: v for k, v in group.items()
            if k not in {"sigma_transform", "_capture_metrics"}
        })
    return {"state": state, "param_groups": param_groups}


def _load_training_checkpoint(
    checkpoint_dir: Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    rank: int,
) -> dict:
    model_path = checkpoint_dir / f"model_{step:06d}.pt"
    optim_path = checkpoint_dir / f"optim_{step:06d}_rank{rank:d}.pt"
    meta_path = checkpoint_dir / f"meta_{step:06d}.json"

    model_data = torch.load(model_path, map_location=device)
    model.load_state_dict(model_data, strict=True)
    del model_data

    optimizer_data = torch.load(optim_path, map_location=device)
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data

    return json.loads(meta_path.read_text())


def _cleanup_old_checkpoints(checkpoint_dir: Path, keep_last: int, world_size: int) -> None:
    if keep_last <= 0:
        return
    steps = _complete_checkpoint_steps(checkpoint_dir, world_size)
    for step in steps[:-keep_last]:
        for path in [
            checkpoint_dir / f"model_{step:06d}.pt",
            checkpoint_dir / f"meta_{step:06d}.json",
            *[checkpoint_dir / f"optim_{step:06d}_rank{rank:d}.pt" for rank in range(world_size)],
        ]:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def _save_training_checkpoint(
    checkpoint_dir: Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    meta: dict,
    rank: int,
    world_size: int,
    ddp: bool,
    keep_last: int,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        torch.save(model.state_dict(), checkpoint_dir / f"model_{step:06d}.pt")
        (checkpoint_dir / f"meta_{step:06d}.json").write_text(json.dumps(meta, indent=2))
    torch.save(
        _optimizer_state_for_checkpoint(optimizer),
        checkpoint_dir / f"optim_{step:06d}_rank{rank:d}.pt",
    )
    if ddp:
        torch.distributed.barrier()
    if rank == 0:
        _cleanup_old_checkpoints(checkpoint_dir, keep_last, world_size)
        print0(f"Saved checkpoint step {step} to {checkpoint_dir}")
    if ddp:
        torch.distributed.barrier()


def _distributed_stop_requested(ddp: bool, device: torch.device) -> bool:
    flag = torch.tensor([1 if _STOP_REQUESTED else 0], device=device, dtype=torch.int32)
    if ddp:
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
    return bool(flag.item())


def _named_trainable_params(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    return {name: p for name, p in model.named_parameters() if p.requires_grad}


def _select_params(named: dict[str, torch.nn.Parameter], predicate) -> list[torch.nn.Parameter]:
    return [p for name, p in named.items() if predicate(name, p)]


def _make_adamw_group(kind: str, params: list[torch.nn.Parameter], lr: float,
                      betas: tuple[float, float], eps: float, weight_decay: float) -> dict:
    return dict(kind=kind, params=params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def _beta_align_flags(batch_beta_align: bool, batch_beta_align_mode: str) -> tuple[bool, bool]:
    if not batch_beta_align or batch_beta_align_mode == "none":
        return False, False
    if batch_beta_align_mode == "beta2_only":
        return False, True
    return True, True


def _beta_for_run(
    beta_ref: float,
    total_batch_size: int,
    *,
    align: bool,
) -> float:
    if not align:
        return beta_ref
    return ema_beta_for_batch(beta_ref, total_batch_size, B_REF)


def build_optimizer_for_run(
    model: torch.nn.Module,
    *,
    optimizer_name: str,
    sigma_transform,
    matrix_lr: float,
    weight_decay: float,
    batch_lr_scale: float,
    dmodel_lr_scale: float,
    total_batch_size: int,
    matrix_lr_adjust: str,
    batch_beta_align: bool,
    batch_beta_align_mode: str,
    adam_lr_mode: str,
    ddp: bool,
) -> tuple[torch.optim.Optimizer, dict[int, str]]:
    """Build the mixed optimizer and return ``param_id -> name`` for matrix params."""
    named = _named_trainable_params(model)
    matrix_params_named = {
        name: p for name, p in named.items()
        if p.ndim == 2 and name.startswith("transformer.h.")
    }
    matrix_params = list(matrix_params_named.values())
    param_name_by_id = {id(p): n for n, p in matrix_params_named.items()}
    matrix_param_ids = {id(p) for p in matrix_params}

    lm_head_params = _select_params(named, lambda n, p: n.startswith("lm_head."))
    embedding_params = _select_params(
        named,
        lambda n, p: n.startswith("transformer.wte.") or n.startswith("transformer.wpe."),
    )
    value_embed_params = _select_params(named, lambda n, p: n.startswith("value_embeds."))
    used_ids = matrix_param_ids | {id(p) for p in lm_head_params + embedding_params + value_embed_params}
    extra_adamw_params = [p for p in named.values() if id(p) not in used_ids]

    def _adam_peak_lr(role: str, nanochat_lr: float) -> float:
        if adam_lr_mode == "relative_to_matrix":
            return adam_lr_from_matrix_lr(
                matrix_lr,
                role,
                dmodel_lr_scale=dmodel_lr_scale,
                batch_lr_scale_value=batch_lr_scale,
            )
        return nanochat_lr

    align_beta1, align_beta2 = _beta_align_flags(batch_beta_align, batch_beta_align_mode)
    adam_betas = (
        _beta_for_run(0.8, total_batch_size, align=align_beta1),
        _beta_for_run(0.96, total_batch_size, align=align_beta2),
    )
    embed_betas = (
        _beta_for_run(0.8, total_batch_size, align=align_beta1),
        _beta_for_run(0.995, total_batch_size, align=align_beta2),
    )
    scalar_betas = (
        _beta_for_run(0.8, total_batch_size, align=align_beta1),
        _beta_for_run(0.95, total_batch_size, align=align_beta2),
    )
    param_groups: list[dict] = []
    if lm_head_params:
        param_groups.append(_make_adamw_group(
            "adamw",
            lm_head_params,
            _adam_peak_lr("lm_head", 0.008 * dmodel_lr_scale * batch_lr_scale),
            adam_betas,
            1e-10,
            0.01,
        ))
    if embedding_params:
        param_groups.append(_make_adamw_group(
            "adamw",
            embedding_params,
            _adam_peak_lr("embedding", 0.3 * dmodel_lr_scale * batch_lr_scale),
            embed_betas,
            1e-10,
            0.001,
        ))
    if value_embed_params:
        param_groups.append(_make_adamw_group(
            "adamw",
            value_embed_params,
            _adam_peak_lr("value_embed", 0.3 * dmodel_lr_scale * batch_lr_scale * 0.5),
            embed_betas,
            1e-10,
            0.01,
        ))
    if extra_adamw_params:
        param_groups.append(_make_adamw_group(
            "adamw",
            extra_adamw_params,
            _adam_peak_lr("scalar", 0.5 * batch_lr_scale * 0.01),
            scalar_betas,
            1e-10,
            0.0,
        ))

    matrix_group_extras = {
        "matrix_lr_adjust": matrix_lr_adjust,
    }
    structured_beta1 = _beta_for_run(args.optimizer_beta1, total_batch_size, align=align_beta1)
    structured_beta2 = _beta_for_run(args.optimizer_beta2, total_batch_size, align=align_beta2)
    structured_shampoo_beta = _beta_for_run(args.shampoo_beta, total_batch_size, align=align_beta2)
    plain_muon_momentum = args.muon_momentum

    if optimizer_name == "adamw":
        if matrix_params:
            matrix_group = _make_adamw_group(
                "adamw",
                matrix_params,
                matrix_lr * batch_lr_scale,
                (structured_beta1, structured_beta2),
                1e-8,
                weight_decay,
            )
            matrix_group["schedule_weight_decay"] = True
            param_groups.append(matrix_group)
        from baseline_optim import StructuredAdamW
        optimizer = StructuredAdamW(param_groups, average_gradients=ddp)
    elif optimizer_name == "plain_muon":
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind="plain_muon",
                params=group_params,
                lr=matrix_lr * batch_lr_scale,
                momentum=plain_muon_momentum,
                ns_steps=5,
                weight_decay=weight_decay,
                **matrix_group_extras,
            ))
        from baseline_optim import StructuredAdamW
        optimizer = StructuredAdamW(param_groups, average_gradients=ddp)
    elif optimizer_name in {"muon", "native_muon"}:
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind="muon",
                params=group_params,
                lr=matrix_lr * batch_lr_scale,
                momentum=plain_muon_momentum,
                ns_steps=5,
                weight_decay=weight_decay,
                **matrix_group_extras,
            ))
        from nanochat.optim import DistMuonAdamW, MuonAdamW
        optimizer = (DistMuonAdamW if ddp else MuonAdamW)(param_groups)
    elif optimizer_name == "streaming_muon":
        from streaming_muon_torch import DistStreamingMuonAdamW, StreamingMuonAdamW

        k_val = args.k if args.k > 0 else None
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind="streaming_muon",
                params=group_params,
                lr=matrix_lr * batch_lr_scale,
                momentum=plain_muon_momentum,
                weight_decay=weight_decay,
                sigma_transform=sigma_transform,
                k=k_val,
                num_iters=args.num_iters,
                pure_qr=args.pure_qr,
                fallback_orthogonality_tol=(
                    None if args.fallback_ortho_tol is not None and args.fallback_ortho_tol < 0
                    else args.fallback_ortho_tol
                ),
                **matrix_group_extras,
            ))
        optimizer = (DistStreamingMuonAdamW if ddp else StreamingMuonAdamW)(param_groups)
    else:
        from baseline_optim import StructuredAdamW

        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind=optimizer_name,
                params=group_params,
                lr=matrix_lr * batch_lr_scale,
                betas=(structured_beta1, structured_beta2),
                shampoo_beta=structured_shampoo_beta,
                eps=1e-8,
                weight_decay=weight_decay,
                precondition_frequency=args.precondition_frequency,
                init_factor=args.structured_init_factor,
                use_qr=args.structured_use_qr,
                **matrix_group_extras,
            ))
        optimizer = StructuredAdamW(param_groups, average_gradients=ddp)

    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    return optimizer, param_name_by_id


# =============================================================================
# Main
# =============================================================================

def main():
    _install_signal_handlers()
    results = {
        "start_time": time.time(),
        "args": vars(args),
    }

    try:
        candidate_params = parse_candidate_params(args.candidate_param)
        results["candidate_params"] = candidate_params

        # --- Distributed setup ---
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
        ddp = world_size > 1

        if ddp:
            torch.distributed.init_process_group(backend="nccl")
            print0(f"Distributed: rank={rank}/{world_size}")

        # --- Device setup ---
        device_type = autodetect_device_type() if args.device_type == "" else args.device_type
        if device_type == "cuda":
            gpu_id = local_rank % torch.cuda.device_count()
            torch.cuda.set_device(gpu_id)
            device = torch.device(f"cuda:{gpu_id}")
        elif device_type == "mps":
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print0(f"Device: {device}")

        checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
        if checkpoint_dir is None and args.save_every > 0:
            checkpoint_dir = Path(args.output_file).parent / "checkpoints"
        if checkpoint_dir is not None:
            checkpoint_dir = checkpoint_dir.resolve()
        results["checkpointing"] = {
            "enabled": checkpoint_dir is not None,
            "checkpoint_dir": None if checkpoint_dir is None else str(checkpoint_dir),
            "save_every": args.save_every,
            "keep_last_checkpoints": args.keep_last_checkpoints,
            "resume": args.resume,
            "resume_from_step": args.resume_from_step,
        }

        # --- Tokenizer ---
        tokenizer = get_tokenizer()
        token_bytes = get_token_bytes(device=device)
        vocab_size = tokenizer.get_vocab_size()

        # --- Load optional StreamingMuon candidate ---
        sigma_transform = None
        if args.optimizer == "streaming_muon":
            sigma_fn = configure_candidate_fn(
                load_candidate_fn(args.candidate_file, args.candidate_code),
                candidate_params,
            )
            from streaming_muon_torch import CustomTransform
            sigma_transform = CustomTransform(sigma_fn)
            print0(f"Loaded candidate f(Σ), params={candidate_params}")

            # Quick smoke test
            sigma_test = torch.rand(4, 32, device=device) * 5
            state_test = {}
            scaling_test = sigma_fn(sigma_test, state_test)
            assert scaling_test.shape == sigma_test.shape, f"Shape mismatch: {scaling_test.shape}"
            assert torch.all(torch.isfinite(scaling_test)), "Non-finite scaling"
            print0(f"Smoke test passed: scaling range [{scaling_test.min().item():.4f}, {scaling_test.max().item():.4f}]")
            results["smoke_test"] = "passed"
        else:
            results["smoke_test"] = "not_applicable_non_streaming_optimizer"

        # --- Build model ---
        base_dim = args.depth * args.aspect_ratio
        model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
        num_heads = model_dim // args.head_dim
        config = GPTConfig(
            sequence_len=args.max_seq_len,
            vocab_size=vocab_size,
            n_layer=args.depth,
            n_head=num_heads,
            n_kv_head=num_heads,
            n_embd=model_dim,
            window_pattern="L",
            architecture=args.architecture,
        )
        with torch.device("meta"):
            model = GPT(config)
        model.to_empty(device=device)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        model.init_weights()
        print0(f"Model: architecture={args.architecture}, depth={args.depth}, dim={model_dim}, heads={num_heads}, "
               f"params={sum(p.numel() for p in model.parameters()):,}")

        # --- Optimizer ---
        dmodel_lr_scale = (model_dim / 768) ** -0.5

        # Batch size (scales with world_size for distributed)
        tokens_per_batch = args.device_batch_size * args.max_seq_len
        total_batch_size = args.total_batch_size if args.total_batch_size > 0 else tokens_per_batch * world_size
        batch_lr_scale = recipe_batch_lr_scale(total_batch_size, B_REF)

        optimizer, param_name_by_id = build_optimizer_for_run(
            model,
            optimizer_name=args.optimizer,
            sigma_transform=sigma_transform,
            matrix_lr=args.matrix_lr,
            weight_decay=args.weight_decay,
            batch_lr_scale=batch_lr_scale,
            dmodel_lr_scale=dmodel_lr_scale,
            total_batch_size=total_batch_size,
            matrix_lr_adjust=args.matrix_lr_adjust,
            batch_beta_align=args.batch_beta_align,
            batch_beta_align_mode=args.batch_beta_align_mode,
            adam_lr_mode=args.adam_lr_mode,
            ddp=ddp,
        )
        matrix_params_named = {
            n: p for n, p in model.named_parameters()
            if p.requires_grad and p.ndim == 2 and n.startswith("transformer.h.")
        }
        print0(f"Optimizer: {args.optimizer}")

        resume_meta = None
        resume_step = _resolve_resume_step(
            checkpoint_dir,
            args.resume,
            args.resume_from_step,
            world_size,
            rank,
            ddp,
        )
        if resume_step is not None:
            print0(f"Resuming from checkpoint step {resume_step} in {checkpoint_dir}")
            resume_meta = _load_training_checkpoint(
                checkpoint_dir,
                resume_step,
                model,
                optimizer,
                device,
                rank,
            )
            # Optimizer checkpoints intentionally omit callable sigma transforms.
            # Restore them from the current run config after load_state_dict().
            for group in optimizer.param_groups:
                if group.get("kind") == "streaming_muon":
                    group["sigma_transform"] = sigma_transform
            results["resumed_from_step"] = resume_step
        else:
            results["resumed_from_step"] = None

        # --- Compile model ---
        # Keep the eager module for optional Hessian diagnostics. Higher-order
        # autograd through the compiled wrapper can fail with donated buffers.
        eager_model = model
        if os.environ.get("NANOCHAT_DISABLE_COMPILE", "0") == "1":
            print0("NANOCHAT_DISABLE_COMPILE=1: using eager model for smoke/debug run.")
        else:
            model = torch.compile(model, dynamic=False)

        # --- Data loader ---
        dataloader_resume_state_dict = (
            None if resume_meta is None else resume_meta.get("dataloader_state_dict")
        )
        train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
            tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device,
            resume_state_dict=dataloader_resume_state_dict,
        )
        build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(
            tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device,
        )
        x, y, dataloader_state_dict = next(train_loader)

        # --- LR / momentum / WD schedulers ---
        num_iterations = args.max_steps

        def get_lr_multiplier(it):
            warmup_iters = args.warmup_steps
            warmdown_iters = round(args.warmdown_ratio * num_iterations)
            if it < warmup_iters:
                return (it + 1) / warmup_iters
            elif it <= num_iterations - warmdown_iters:
                return 1.0
            else:
                progress = (num_iterations - it) / warmdown_iters
                return progress * 1.0 + (1 - progress) * args.final_lr_frac

        def get_muon_momentum(it):
            if args.muon_momentum_schedule == "static":
                return args.muon_momentum
            warmdown_iters = round(args.warmdown_ratio * num_iterations)
            warmdown_start = num_iterations - warmdown_iters
            if it < min(400, num_iterations // 4):
                frac = it / min(400, num_iterations // 4)
                momentum = (1 - frac) * 0.85 + frac * 0.97
            elif args.mom_decay and it >= warmdown_start:
                progress = (it - warmdown_start) / warmdown_iters
                momentum = 0.97 * (1 - progress) + 0.90 * progress
            else:
                momentum = 0.97
            return momentum

        def get_weight_decay(it):
            return args.weight_decay * 0.5 * (1 + math.cos(math.pi * it / num_iterations))

        # --- Training loop ---
        print0(f"Training for {num_iterations} steps, batch_size={total_batch_size}")
        if resume_meta is None:
            start_step = 0
            smooth_train_loss = 0.0
            train_losses = []
            val_bpbs = []
            metric_logs = []
            best_val_bpb = float("inf")
        else:
            start_step = int(resume_meta["step"])
            loop_state = resume_meta.get("loop_state", {})
            smooth_train_loss = float(loop_state.get("smooth_train_loss", 0.0))
            train_losses = loop_state.get("train_losses", [])
            val_bpbs = loop_state.get("val_bpbs", [])
            metric_logs = loop_state.get("metric_logs", [])
            best_val_bpb = float(loop_state.get("best_val_bpb", float("inf")))
            print0(f"Restored loop state at step {start_step}; previous best_val_bpb={best_val_bpb}")

        # One forward/backward micro-step consumes tokens_per_batch on each rank.
        # For DDP, the optimizer averages gradients across ranks, so the global
        # effective batch is grad_accum_steps * tokens_per_batch * world_size.
        world_tokens_per_fwdbwd = tokens_per_batch * world_size
        assert total_batch_size % world_tokens_per_fwdbwd == 0, (
            f"total_batch_size ({total_batch_size}) must be divisible by "
            f"device_batch_size * max_seq_len * world_size ({world_tokens_per_fwdbwd})"
        )
        grad_accum_steps = max(1, total_batch_size // world_tokens_per_fwdbwd)

        metrics_enabled = args.metrics_every > 0
        last_hessian_space = None
        hessian_space_active = False
        previous_gradient_projection = None
        gradient_projection_top1_history = []
        if metrics_enabled:
            print0(f"Metric logging enabled every {args.metrics_every} steps.")

        interrupted_step = None
        completed_steps = start_step
        for step in range(start_step, num_iterations + 1):
            last_step = step == num_iterations

            # Evaluate
            if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
                model.eval()
                val_loader = build_val_loader()
                eval_steps = max(1, args.eval_tokens // world_tokens_per_fwdbwd)
                val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
                val_bpbs.append({"step": step, "val_bpb": val_bpb})
                if val_bpb < best_val_bpb:
                    best_val_bpb = val_bpb
                print0(f"  Step {step:05d} | val_bpb: {val_bpb:.6f} (best: {best_val_bpb:.6f})")
                model.train()

            if last_step:
                break

            # Training step
            t0 = time.time()
            lrm = get_lr_multiplier(step)
            muon_momentum = get_muon_momentum(step)
            muon_wd = get_weight_decay(step)
            metrics_due = metrics_enabled and (step % args.metrics_every == 0)
            hessian_due = (
                metrics_due
                and args.metrics_hessian_every > 0
                and step % args.metrics_hessian_every == 0
            )
            hessian_batches = [] if hessian_due else None
            train_loss_sum = 0.0
            for micro_step in range(grad_accum_steps):
                if hessian_due and hessian_batches is not None:
                    hessian_batches.append((x.detach().clone(), y.detach().clone()))
                loss = model(x, y)
                train_loss_val = loss.detach().item()
                train_loss_sum += train_loss_val
                loss = loss / grad_accum_steps
                loss.backward()
                x, y, dataloader_state_dict = next(train_loader)

            # Schedule
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * lrm
                if group['kind'] in {'streaming_muon', 'muon', 'plain_muon'}:
                    group["momentum"] = muon_momentum
                if group.get("schedule_weight_decay", False) or group['kind'] in {'streaming_muon', 'muon', 'plain_muon', 'soap', 'shampoo', 'kl_shampoo', 'kl_soap'}:
                    group["weight_decay"] = muon_wd
                if group['kind'] == 'streaming_muon':
                    group["_capture_metrics"] = metrics_due

            optimizer.step()

            # Logging
            ema_beta = 0.9
            train_loss_mean = train_loss_sum / grad_accum_steps
            if ddp:
                loss_tensor = torch.tensor(train_loss_mean, device=device, dtype=torch.float32)
                torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.AVG)
                train_loss_mean = float(loss_tensor.detach().cpu())
            smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_mean
            debiased = smooth_train_loss / (1 - ema_beta ** (step + 1))
            dt = time.time() - t0

            if metrics_due:
                from metric_logging import (
                    clear_metric_caches,
                    collect_muon_metrics,
                    gradient_projection_onto_hessian_space,
                    hessian_power_probe,
                    projection_coefficients_pearson,
                    projection_lag1_pearson,
                )
                metric_entry, hessian_refs = collect_muon_metrics(
                    optimizer,
                    param_name_by_id,
                    step=step,
                    top_k=args.metrics_top_k,
                    module_regex=args.metrics_module_regex,
                    max_modules=args.metrics_max_modules,
                    train_loss=train_loss_mean,
                    lr_multiplier=lrm,
                    muon_momentum=muon_momentum,
                    collect_hessian_refs=hessian_due or hessian_space_active,
                )
                clear_metric_caches(optimizer)

                run_hessian = hessian_due
                hessian_stats = None
                hessian_error_text = None
                hessian_success = False
                if run_hessian:
                    model.zero_grad(set_to_none=True)
                    try:
                        hessian_probe_batches = hessian_batches if hessian_batches else [(x.detach(), y.detach())]
                        hessian_stats = hessian_power_probe(
                            eager_model,
                            hessian_probe_batches,
                            hessian_refs,
                            matrix_params_named,
                            top_k=args.metrics_hessian_top_k,
                            iters=args.metrics_hessian_iters,
                            module_regex=args.metrics_module_regex,
                            max_modules=args.metrics_hessian_max_modules,
                            selected_names=list(matrix_params_named.keys()),
                            return_stats=(rank == 0),
                        )
                        hessian_success = True
                    except Exception as hessian_error:
                        hessian_error_text = str(hessian_error)
                    if ddp:
                        success_flag = torch.tensor(
                            1 if hessian_success else 0,
                            device=device,
                            dtype=torch.int32,
                        )
                        torch.distributed.all_reduce(success_flag, op=torch.distributed.ReduceOp.MIN)
                        hessian_success = bool(success_flag.item())
                    hessian_space_active = hessian_success

                if rank == 0 and metric_entry is not None:
                    local_hessian_tokens = (
                        sum(int(batch_y.numel()) for _, batch_y in hessian_batches)
                        if hessian_batches
                        else 0
                    )
                    metric_entry.setdefault("metadata", {})
                    metric_entry["metadata"].update({
                        "train_loss": "raw mean cross-entropy over this optimizer step's grad-accum microbatches",
                        "train_loss_scope": "global_ddp_mean" if ddp else "single_process",
                        "train_loss_ema_scalar": "train/loss_ema",
                        "optimizer_metric_scope": "gathered_parameter_owner_ranks" if ddp else "single_process",
                        "hessian_weight_state": "post_optimizer_step",
                        "hessian_batch_source": "grad_accum_microbatches_same_optimizer_step" if hessian_batches else None,
                        "hessian_batch_scope": (
                            "all_local_grad_accum_microbatches_per_rank"
                            if ddp and hessian_batches
                            else ("all_local_grad_accum_microbatches" if hessian_batches else None)
                        ),
                        "hessian_grad_accum_scope": "full_optimizer_step_grad_accum" if hessian_batches else None,
                        "hessian_local_microbatches": len(hessian_batches) if hessian_batches else 0,
                        "hessian_local_tokens": local_hessian_tokens,
                        "hessian_global_tokens_assuming_equal_ranks": local_hessian_tokens * world_size if ddp else local_hessian_tokens,
                        "hessian_probe_scope": (
                            "distributed_token_weighted_loss_hvp"
                            if ddp and hessian_batches
                            else ("single_process_loss_hvp" if hessian_batches else None)
                        ),
                    })
                    metric_entry["scalars"]["train/loss_ema"] = debiased

                    if run_hessian:
                        if hessian_success and hessian_stats is not None:
                            transient = hessian_stats.pop("_transient", {})
                            new_hessian_space = transient.get("hessian_space")
                            if new_hessian_space is not None:
                                new_hessian_space["step"] = step
                                last_hessian_space = new_hessian_space
                                hessian_space_active = True
                                previous_gradient_projection = None
                                gradient_projection_top1_history = []
                            metric_entry["hessian"] = hessian_stats
                            metric_entry["scalars"].update(hessian_stats.get("scalars", {}))
                            metric_entry["vectors"].update(hessian_stats.get("vectors", {}))
                        else:
                            hessian_space_active = last_hessian_space is not None
                            metric_entry["hessian"] = {
                                "error": hessian_error_text or "hessian_probe_failed_on_at_least_one_rank",
                                "note": "Hessian HVP is best-effort; some attention kernels do not support double backward.",
                            }

                    if not run_hessian and last_hessian_space is not None and hessian_refs:
                        current_projection = gradient_projection_onto_hessian_space(
                            hessian_refs,
                            last_hessian_space,
                        )
                        if current_projection is not None:
                            hstep = current_projection.get("hessian_step")
                            metric_entry["metadata"]["last_hessian_space_step"] = hstep
                            metric_entry["vectors"][
                                "gradient_projection_on_last_hessian_space_coefficients/selected_subspace"
                            ] = current_projection["coefficients"]
                            metric_entry["scalars"][
                                "gradient_projection_on_last_hessian_space_norm/selected_subspace"
                            ] = current_projection["norm"]
                            metric_entry["scalars"][
                                "gradient_projection_on_last_hessian_space_top1_abs_fraction/selected_subspace"
                            ] = current_projection["top1_abs_fraction"]
                            if (
                                previous_gradient_projection is not None
                                and previous_gradient_projection.get("hessian_step") == hstep
                            ):
                                metric_entry["scalars"][
                                    "gradient_projection_on_last_hessian_space_consecutive_pearson/selected_subspace"
                                ] = projection_coefficients_pearson(
                                    previous_gradient_projection,
                                    current_projection,
                                )
                                metric_entry["metadata"][
                                    "gradient_projection_cosine_previous_step"
                                ] = previous_gradient_projection.get("step")
                            coeffs = current_projection.get("coefficients") or []
                            if coeffs:
                                gradient_projection_top1_history.append(float(coeffs[0]))
                                lag1 = projection_lag1_pearson(
                                    gradient_projection_top1_history,
                                    window=args.metrics_projection_correlation_window,
                                )
                                if math.isfinite(lag1):
                                    metric_entry["scalars"][
                                        "gradient_projection_on_last_hessian_top1_lag1_pearson/selected_subspace"
                                    ] = lag1
                                    metric_entry["metadata"][
                                        "gradient_projection_lag1_pearson_window"
                                    ] = args.metrics_projection_correlation_window
                            current_projection["step"] = step
                            previous_gradient_projection = current_projection

                    metric_logs.append(metric_entry)

            model.zero_grad(set_to_none=True)

            if step % 50 == 0:
                train_losses.append({"step": step, "loss": debiased})
                print0(f"  Step {step:05d} | loss: {debiased:.6f} | dt: {dt*1000:.0f}ms")

            # GC management
            if step == 0:
                gc.collect()

            completed_steps = step + 1
            stop_now = _distributed_stop_requested(ddp, device)
            checkpoint_due = (
                checkpoint_dir is not None
                and (
                    stop_now
                    or completed_steps == num_iterations
                    or (args.save_every > 0 and completed_steps % args.save_every == 0)
                )
            )
            if checkpoint_due:
                _save_training_checkpoint(
                    checkpoint_dir,
                    completed_steps,
                    eager_model,
                    optimizer,
                    {
                        "checkpoint_version": 1,
                        "step": completed_steps,
                        "args": vars(args),
                        "candidate_params": candidate_params,
                        "dataloader_state_dict": dataloader_state_dict,
                        "loop_state": {
                            "smooth_train_loss": smooth_train_loss,
                            "best_val_bpb": best_val_bpb,
                            "train_losses": train_losses,
                            "val_bpbs": val_bpbs,
                            "metric_logs": metric_logs,
                        },
                    },
                    rank,
                    world_size,
                    ddp,
                    args.keep_last_checkpoints,
                )
            if stop_now:
                interrupted_step = completed_steps
                print0(f"Stopping after checkpoint step {interrupted_step} due to preemption signal.")
                break

        # --- Sigma profiling (optional) ---
        if interrupted_step is None and args.save_sigma_profile:
            from streaming_muon_torch import get_spectral_diagnostics
            diag = get_spectral_diagnostics(optimizer)
            if diag['sigma']:
                all_sigma = torch.cat([s.flatten() for s in diag['sigma']])
                sigma_profile = {
                    "mean": all_sigma.mean().item(),
                    "std": all_sigma.std().item(),
                    "min": all_sigma.min().item(),
                    "max": all_sigma.max().item(),
                    "median": all_sigma.median().item(),
                    "q25": all_sigma.quantile(0.25).item(),
                    "q75": all_sigma.quantile(0.75).item(),
                    "q90": all_sigma.quantile(0.90).item(),
                    "q99": all_sigma.quantile(0.99).item(),
                    "n_values": len(all_sigma),
                    "per_group": diag['sigma_stats'],
                }
                profile_path = Path(args.save_sigma_profile)
                profile_path.parent.mkdir(parents=True, exist_ok=True)
                profile_path.write_text(json.dumps(sigma_profile, indent=2))
                print0(f"Sigma profile saved to {profile_path}")
                print0(f"  mean={sigma_profile['mean']:.4f} median={sigma_profile['median']:.4f} "
                       f"q90={sigma_profile['q90']:.4f} max={sigma_profile['max']:.4f}")
                results["sigma_profile"] = sigma_profile

        # --- Results ---
        results["val_bpb_final"] = val_bpbs[-1]["val_bpb"] if val_bpbs else None
        results["val_bpb_best"] = best_val_bpb
        results["train_loss_final"] = train_losses[-1]["loss"] if train_losses else None
        results["val_bpbs"] = val_bpbs
        results["train_losses"] = train_losses
        results["metric_logs"] = metric_logs
        results["steps_completed"] = completed_steps
        if interrupted_step is None:
            results["score"] = best_val_bpb
            results["error"] = None
        else:
            results["score"] = None
            results["error"] = f"interrupted_after_checkpoint_step_{interrupted_step}"

    except Exception as e:
        import traceback
        results["error"] = str(e)
        results["traceback"] = traceback.format_exc()
        results["score"] = None
        print0(f"ERROR: {e}")
        traceback.print_exc()

    results["end_time"] = time.time()
    results["eval_time_seconds"] = results["end_time"] - results["start_time"]

    # Write results (rank 0 only)
    if rank == 0:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2))
        print0(f"Results written to {output_path}")
        if results.get("score") is not None:
            print0(f"SCORE: {results['score']:.6f}")

    # Cleanup distributed
    if ddp:
        torch.distributed.destroy_process_group()

    return results


if __name__ == "__main__":
    main()
