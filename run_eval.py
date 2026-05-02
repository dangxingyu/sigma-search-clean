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
from pathlib import Path

import torch

# Add nanochat to path
SCRIPT_DIR = Path(__file__).resolve().parent
NANOCHAT_DIR = SCRIPT_DIR / "nanochat" if (SCRIPT_DIR / "nanochat").exists() else None

parser = argparse.ArgumentParser(description="Eval runner for f(Σ) search")
parser.add_argument("--nanochat-dir", type=str, default=str(NANOCHAT_DIR), help="Path to nanochat repo")
parser.add_argument("--candidate-file", type=str, required=True, help="Python file defining f(sigma, state)")
parser.add_argument("--candidate-code", type=str, default=None, help="Inline Python code for f(sigma, state)")
parser.add_argument("--candidate-param", action="append", default=[],
                    help="Candidate hyperparameter as key=value. Values are JSON-parsed when possible; repeatable.")
parser.add_argument("--output-file", type=str, default="eval_result.json", help="Output JSON file")
# Model
parser.add_argument("--depth", type=int, default=4, help="Model depth (small for search)")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=64, help="Head dimension")
parser.add_argument("--max-seq-len", type=int, default=512, help="Max sequence length")
# Training
parser.add_argument("--max-steps", type=int, default=500, help="Training steps")
parser.add_argument("--device-batch-size", type=int, default=8, help="Per-device batch size")
parser.add_argument("--total-batch-size", type=int, default=-1, help="Total batch size (-1 = auto)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="Muon LR")
parser.add_argument("--weight-decay", type=float, default=0.28, help="Weight decay")
parser.add_argument("--warmup-steps", type=int, default=40, help="LR warmup steps")
parser.add_argument("--warmdown-ratio", type=float, default=0.65, help="Fraction of training in LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.05, help="Final LR as fraction of peak")
parser.add_argument("--mom-decay", action="store_true", default=True, help="Decay momentum 0.97→0.90 during warmdown")
parser.add_argument("--no-mom-decay", dest="mom_decay", action="store_false", help="Keep momentum constant at 0.97")
# StreamingMuon
parser.add_argument("--use-scqr", action="store_true", default=True, help="Use Shifted Cholesky QR")
parser.add_argument("--k", type=int, default=-1, help="Number of singular vectors (-1 = full rank)")
parser.add_argument("--num-iters", type=int, default=1, help="Number of streaming power iterations (1 or 2)")
parser.add_argument("--fnorm-update", action="store_true", default=False, help="Frobenius-normalize update to UV^T scale (test fix for divergence)")
parser.add_argument("--pe-cleanup", action="store_true", default=False, help="Apply 5-step Polar Express cleanup on streaming muon's update output")
parser.add_argument("--qr-left-vectors", action="store_true", default=False, help="Use SCQR instead of column-norm for left_vectors (forces orthonormality)")
parser.add_argument("--pure-qr", action="store_true", default=False, help="Replace all SCQR with Householder QR (diagnostic: test if SCQR drift causes divergence)")
parser.add_argument("--fallback-ortho-tol", type=float, default=0.02, help="SCQR fallback orthogonality tolerance: when max ||Q^T Q - I|| > tol, fall back to Householder QR. Default 0.02 is safe at matrix_lr=0.02/d8/1B; tol>=0.05 diverges. Pass a negative value to disable (routes to unsafe fused fast path).")
parser.add_argument("--basis-reset-every", type=int, default=0, help="Reset basis to I every N steps (0 = never; tests warm-start drift hypothesis)")
parser.add_argument("--input-normalize", action="store_true", default=False, help="Normalize input by Frobenius norm before gram (matches native muon)")
parser.add_argument("--sigma-spread", type=float, default=0.0, help="Multiply left_vectors by per-col uniform[1-s, 1+s] to simulate polar express σ∈[0.5,1.5]")
# Eval
parser.add_argument("--eval-every", type=int, default=100, help="Evaluate val BPB every N steps")
parser.add_argument("--eval-tokens", type=int, default=524288, help="Tokens for val eval")
# Diagnostics
parser.add_argument("--save-sigma-profile", type=str, default=None,
                    help="Path to save sigma distribution stats (JSON). Useful for profiling baseline.")
parser.add_argument("--metrics-every", type=int, default=0,
                    help="If >0, log Muon-like diagnostics every N optimizer steps into result.json.")
parser.add_argument("--metrics-top-k", type=int, default=4,
                    help="Top singular directions to summarize in diagnostic logging.")
parser.add_argument("--metrics-module-regex", type=str, default=r"transformer\.h",
                    help="Regex selecting modules for diagnostic logging.")
parser.add_argument("--metrics-max-modules", type=int, default=0,
                    help="Maximum modules to log per step; 0 means all selected modules.")
parser.add_argument("--metrics-save-components", action="store_true",
                    help="Save top Muon singular vectors to .pt files beside the result JSON.")
parser.add_argument("--metrics-component-dir", type=str, default=None,
                    help="Directory for --metrics-save-components; default is <output-dir>/metric_components.")
parser.add_argument("--metrics-split-momentum", action="store_true",
                    help="Maintain independent split-half momentum buffers and log their SVD alignment.")
parser.add_argument("--metrics-alignment-side", type=str, default="lite", choices=("left", "right", "lite"),
                    help="Singular-vector side for split-momentum alignment.")
parser.add_argument("--metrics-hessian-every", type=int, default=0,
                    help="If >0, run expensive Hessian power diagnostics every N logged steps.")
parser.add_argument("--metrics-hessian-top-k", type=int, default=1,
                    help="Number of Hessian directions for the expensive Hessian probe.")
parser.add_argument("--metrics-hessian-iters", type=int, default=6,
                    help="Power iterations per Hessian direction.")
parser.add_argument("--metrics-hessian-max-modules", type=int, default=1,
                    help="Maximum modules per Hessian probe; keep this small.")
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
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.common import COMPUTE_DTYPE, print0, autodetect_device_type
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.loss_eval import evaluate_bpb
from streaming_muon_torch import StreamingMuonAdamW, DistStreamingMuonAdamW, CustomTransform

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


# =============================================================================
# Main
# =============================================================================

def main():
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

        # --- Tokenizer ---
        tokenizer = get_tokenizer()
        token_bytes = get_token_bytes(device=device)
        vocab_size = tokenizer.get_vocab_size()

        # --- Load candidate ---
        sigma_fn = configure_candidate_fn(
            load_candidate_fn(args.candidate_file, args.candidate_code),
            candidate_params,
        )
        print0(f"Loaded candidate f(Σ), params={candidate_params}")

        # Quick smoke test
        sigma_test = torch.rand(4, 32, device=device) * 5
        state_test = {}
        scaling_test = sigma_fn(sigma_test, state_test)
        assert scaling_test.shape == sigma_test.shape, f"Shape mismatch: {scaling_test.shape}"
        assert torch.all(torch.isfinite(scaling_test)), "Non-finite scaling"
        print0(f"Smoke test passed: scaling range [{scaling_test.min().item():.4f}, {scaling_test.max().item():.4f}]")
        results["smoke_test"] = "passed"

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
        )
        with torch.device("meta"):
            model = GPT(config)
        model.to_empty(device=device)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        model.init_weights()
        print0(f"Model: depth={args.depth}, dim={model_dim}, heads={num_heads}, "
               f"params={sum(p.numel() for p in model.parameters()):,}")

        # --- Optimizer ---
        # Replicate setup_optimizer logic but use StreamingMuonAdamW
        matrix_params = list(model.transformer.h.parameters())
        matrix_params_named = {
            n: p for n, p in model.named_parameters()
            if p.requires_grad and p.ndim == 2 and "transformer.h." in n
        }
        param_name_by_id = {id(p): n for n, p in matrix_params_named.items()}
        value_embeds_params = list(model.value_embeds.parameters())
        embedding_params = list(model.transformer.wte.parameters())
        lm_head_params = list(model.lm_head.parameters())
        resid_params = [model.resid_lambdas]
        x0_params = [model.x0_lambdas]
        smear_params = [model.smear_gate.weight, model.smear_lambda, model.backout_lambda]

        dmodel_lr_scale = (model_dim / 768) ** -0.5

        # Batch size (scales with world_size for distributed)
        tokens_per_batch = args.device_batch_size * args.max_seq_len
        total_batch_size = args.total_batch_size if args.total_batch_size > 0 else tokens_per_batch * world_size
        batch_lr_scale = (total_batch_size / (2**19)) ** 0.5 if total_batch_size != 2**19 else 1.0

        # Match nanochat's setup_optimizer with base_train.py CLI defaults:
        # unembedding_lr=0.008, embedding_lr=0.3, scalar_lr=0.5 (all scaled by batch_lr_scale)
        scalar_lr = 0.5 * batch_lr_scale
        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=0.008 * dmodel_lr_scale * batch_lr_scale,
                 betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=0.3 * dmodel_lr_scale * batch_lr_scale,
                 betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=value_embeds_params, lr=0.3 * dmodel_lr_scale * batch_lr_scale * 0.5,
                 betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]

        # Muon groups → streaming_muon groups
        k_val = args.k if args.k > 0 else None
        sigma_transform = CustomTransform(sigma_fn)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='streaming_muon',
                params=group_params,
                lr=args.matrix_lr * batch_lr_scale,
                momentum=0.95,
                weight_decay=args.weight_decay,
                sigma_transform=sigma_transform,
                use_scqr=args.use_scqr,
                k=k_val,
                num_iters=args.num_iters,
                fnorm_update=args.fnorm_update,
                pe_cleanup=args.pe_cleanup,
                qr_left_vectors=args.qr_left_vectors,
                pure_qr=args.pure_qr,
                fallback_orthogonality_tol=(None if args.fallback_ortho_tol is not None and args.fallback_ortho_tol < 0 else args.fallback_ortho_tol),
                basis_reset_every=args.basis_reset_every,
                input_normalize=args.input_normalize,
                sigma_spread=args.sigma_spread,
            ))

        OptimizerClass = DistStreamingMuonAdamW if ddp else StreamingMuonAdamW
        optimizer = OptimizerClass(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]

        # --- Compile model ---
        # Keep the eager module for optional Hessian diagnostics. Higher-order
        # autograd through the compiled wrapper can fail with donated buffers.
        eager_model = model
        model = torch.compile(model, dynamic=False)

        # --- Data loader ---
        train_loader = tokenizing_distributed_data_loader_bos_bestfit(
            tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device,
        )
        build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(
            tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device,
        )
        x, y = next(train_loader)

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
            warmdown_iters = round(args.warmdown_ratio * num_iterations)
            warmdown_start = num_iterations - warmdown_iters
            if it < min(400, num_iterations // 4):
                frac = it / min(400, num_iterations // 4)
                return (1 - frac) * 0.85 + frac * 0.97
            elif args.mom_decay and it >= warmdown_start:
                progress = (it - warmdown_start) / warmdown_iters
                return 0.97 * (1 - progress) + 0.90 * progress
            else:
                return 0.97

        def get_weight_decay(it):
            return args.weight_decay * 0.5 * (1 + math.cos(math.pi * it / num_iterations))

        # --- Training loop ---
        print0(f"Training for {num_iterations} steps, batch_size={total_batch_size}")
        smooth_train_loss = 0.0
        train_losses = []
        val_bpbs = []
        best_val_bpb = float("inf")

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
        metric_logs = []
        do_split_momentum = metrics_enabled and args.metrics_split_momentum and grad_accum_steps >= 2
        split_momentum_a = split_momentum_b = None
        split_mtilde_a = split_mtilde_b = None
        split_momentum_names = []
        if do_split_momentum:
            module_pattern = re.compile(args.metrics_module_regex) if args.metrics_module_regex else None
            split_momentum_names = [
                n for n in matrix_params_named
                if module_pattern is None or module_pattern.search(n) is not None
            ]
            if args.metrics_max_modules > 0:
                split_momentum_names = split_momentum_names[:args.metrics_max_modules]
            if not split_momentum_names:
                do_split_momentum = False
                print0("Metric logging: split-half momentum alignment disabled; no modules matched.")
            else:
                split_momentum_a = {
                    n: torch.zeros_like(matrix_params_named[n], dtype=torch.float32)
                    for n in split_momentum_names
                }
                split_momentum_b = {
                    n: torch.zeros_like(matrix_params_named[n], dtype=torch.float32)
                    for n in split_momentum_names
                }
                split_mtilde_a = {
                    n: torch.zeros_like(matrix_params_named[n], dtype=torch.float32)
                    for n in split_momentum_names
                }
                split_mtilde_b = {
                    n: torch.zeros_like(matrix_params_named[n], dtype=torch.float32)
                    for n in split_momentum_names
                }
                scope = "global DDP-averaged" if ddp else "single-process"
                print0(f"Metric logging: split-half momentum alignment enabled, "
                       f"top_k={args.metrics_top_k}, side={args.metrics_alignment_side}, "
                       f"tensor=M_tilde, scope={scope}, modules={len(split_momentum_names)}")
        elif metrics_enabled:
            reason = "disabled by flag" if not args.metrics_split_momentum else f"grad_accum={grad_accum_steps} < 2"
            print0(f"Metric logging enabled every {args.metrics_every} steps; split momentum {reason}.")

        for step in range(num_iterations + 1):
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
            half = grad_accum_steps // 2 if do_split_momentum else 0
            split_grad_a = {}
            hessian_batch = None
            train_loss_sum = 0.0
            for micro_step in range(grad_accum_steps):
                if hessian_due and hessian_batch is None:
                    hessian_batch = (x.detach().clone(), y.detach().clone())
                loss = model(x, y)
                train_loss_val = loss.detach().item()
                train_loss_sum += train_loss_val
                loss = loss / grad_accum_steps
                loss.backward()
                x, y = next(train_loader)
                if do_split_momentum and (micro_step + 1) == half:
                    scale = grad_accum_steps / half
                    for n in split_momentum_names:
                        p = matrix_params_named[n]
                        if p.grad is not None:
                            split_grad_a[n] = (scale * p.grad.detach().to(torch.float32)).clone()

            if (
                do_split_momentum
                and split_momentum_a is not None
                and split_momentum_b is not None
                and split_mtilde_a is not None
                and split_mtilde_b is not None
            ):
                rest = grad_accum_steps - half
                for n in split_momentum_names:
                    p = matrix_params_named[n]
                    if n not in split_grad_a or p.grad is None:
                        continue
                    grad_full = p.grad.detach().to(torch.float32)
                    grad_a = split_grad_a[n]
                    grad_b = (grad_accum_steps * grad_full - half * grad_a) / rest
                    if ddp:
                        torch.distributed.all_reduce(grad_a, op=torch.distributed.ReduceOp.AVG)
                        torch.distributed.all_reduce(grad_b, op=torch.distributed.ReduceOp.AVG)
                    split_momentum_a[n].mul_(muon_momentum).add_(grad_a, alpha=1.0 - muon_momentum)
                    split_momentum_b[n].mul_(muon_momentum).add_(grad_b, alpha=1.0 - muon_momentum)
                    split_mtilde_a[n].copy_(grad_a).mul_(1.0 - muon_momentum).add_(
                        split_momentum_a[n], alpha=muon_momentum
                    )
                    split_mtilde_b[n].copy_(grad_b).mul_(1.0 - muon_momentum).add_(
                        split_momentum_b[n], alpha=muon_momentum
                    )

            # Schedule
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * lrm
                if group['kind'] == 'streaming_muon':
                    group["momentum"] = muon_momentum
                    group["weight_decay"] = muon_wd
                    group["_capture_metrics"] = metrics_due

            optimizer.step()

            # Logging
            ema_beta = 0.9
            train_loss_mean = train_loss_sum / grad_accum_steps
            smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_mean
            debiased = smooth_train_loss / (1 - ema_beta ** (step + 1))
            dt = time.time() - t0

            if metrics_due:
                from metric_logging import (
                    clear_metric_caches,
                    collect_muon_metrics,
                    hessian_power_probe,
                    split_momentum_alignment,
                )
                component_dir = args.metrics_component_dir
                if component_dir is None:
                    component_dir = str(Path(args.output_file).parent / "metric_components")
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
                    save_components=args.metrics_save_components,
                    component_dir=component_dir,
                )
                clear_metric_caches(optimizer)

                if rank == 0 and metric_entry is not None:
                    metric_entry.setdefault("metadata", {})
                    metric_entry["metadata"].update({
                        "train_loss": "raw mean cross-entropy over this optimizer step's grad-accum microbatches",
                        "train_loss_ema_scalar": "train/loss_ema",
                        "hessian_weight_state": "post_optimizer_step",
                        "hessian_batch_source": "first_microbatch_same_optimizer_step" if hessian_batch is not None else None,
                        "hessian_batch_scope": "rank0_local_microbatch" if ddp and hessian_batch is not None else "single_process",
                        "split_momentum_alignment_scope": "global_ddp_average" if ddp else "single_process",
                    })
                    metric_entry["scalars"]["train/loss_ema"] = debiased
                    if do_split_momentum and split_mtilde_a is not None and split_mtilde_b is not None:
                        split_stats = split_momentum_alignment(
                            split_mtilde_a,
                            split_mtilde_b,
                            top_k=args.metrics_top_k,
                            alignment_side=args.metrics_alignment_side,
                            module_regex=args.metrics_module_regex,
                            max_modules=args.metrics_max_modules,
                        )
                        split_stats["tensor"] = "momentum_after_nesterov"
                        split_stats["definition"] = "M_tilde = (1 - beta) * G_split + beta * M_split_new"
                        metric_entry["split_momentum_alignment"] = split_stats
                        for key, value in split_stats.get("vectors", {}).items():
                            metric_entry["vectors"][key] = value

                    run_hessian = (
                        args.metrics_hessian_every > 0
                        and step % args.metrics_hessian_every == 0
                        and hessian_refs
                    )
                    if run_hessian:
                        model.zero_grad(set_to_none=True)
                        try:
                            hessian_stats = hessian_power_probe(
                                eager_model,
                                hessian_batch if hessian_batch is not None else (x, y),
                                hessian_refs,
                                matrix_params_named,
                                top_k=args.metrics_hessian_top_k,
                                iters=args.metrics_hessian_iters,
                                module_regex=args.metrics_module_regex,
                                max_modules=args.metrics_hessian_max_modules,
                            )
                            metric_entry["hessian"] = hessian_stats
                            metric_entry["scalars"].update(hessian_stats.get("scalars", {}))
                            metric_entry["vectors"].update(hessian_stats.get("vectors", {}))
                        except Exception as hessian_error:
                            metric_entry["hessian"] = {
                                "error": str(hessian_error),
                                "note": "Hessian HVP is best-effort; some attention kernels do not support double backward.",
                            }

                    metric_logs.append(metric_entry)

            model.zero_grad(set_to_none=True)

            if step % 50 == 0:
                train_losses.append({"step": step, "loss": debiased})
                print0(f"  Step {step:05d} | loss: {debiased:.6f} | dt: {dt*1000:.0f}ms")

            # GC management
            if step == 0:
                gc.collect()

        # --- Sigma profiling (optional) ---
        if args.save_sigma_profile:
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
        results["score"] = best_val_bpb
        results["val_bpb_final"] = val_bpbs[-1]["val_bpb"] if val_bpbs else None
        results["val_bpb_best"] = best_val_bpb
        results["train_loss_final"] = train_losses[-1]["loss"] if train_losses else None
        results["val_bpbs"] = val_bpbs
        results["train_losses"] = train_losses
        results["metric_logs"] = metric_logs
        results["steps_completed"] = num_iterations
        results["error"] = None

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
