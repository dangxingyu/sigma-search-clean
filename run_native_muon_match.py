#!/usr/bin/env python3
"""Native Muon runner matched to run_eval.py for StreamingMuon comparisons."""

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
NANOCHAT_DIR = SCRIPT_DIR / "nanochat"

parser = argparse.ArgumentParser(description="Native Muon runner matched to run_eval.py")
parser.add_argument("--nanochat-dir", type=str, default=str(NANOCHAT_DIR))
parser.add_argument("--output-file", type=str, default="native_muon_match.json")
parser.add_argument("--depth", type=int, default=8)
parser.add_argument("--aspect-ratio", type=int, default=64)
parser.add_argument("--head-dim", type=int, default=64)
parser.add_argument("--max-seq-len", type=int, default=1024)
parser.add_argument("--max-steps", type=int, default=256)
parser.add_argument("--device-batch-size", type=int, default=16)
parser.add_argument("--total-batch-size", type=int, default=262144)
parser.add_argument("--matrix-lr", type=float, default=0.02)
parser.add_argument("--weight-decay", type=float, default=0.28)
parser.add_argument("--warmup-steps", type=int, default=40)
parser.add_argument("--warmdown-ratio", type=float, default=0.65)
parser.add_argument("--final-lr-frac", type=float, default=0.05)
parser.add_argument("--ns-steps", type=int, default=5)
parser.add_argument("--eval-every", type=int, default=64)
parser.add_argument("--eval-tokens", type=int, default=524288)
parser.add_argument("--device-type", type=str, default="")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

sys.path.insert(0, args.nanochat_dir)
sys.path.insert(0, str(SCRIPT_DIR))

from nanochat.common import autodetect_device_type, print0
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.gpt import GPT, GPTConfig
from nanochat.loss_eval import evaluate_bpb
from nanochat.optim import DistMuonAdamW, MuonAdamW
from nanochat.tokenizer import get_token_bytes, get_tokenizer


def main():
    results = {"start_time": time.time(), "args": vars(args), "optimizer": "native_muon"}
    rank = 0
    ddp = False
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
        ddp = world_size > 1
        if ddp:
            torch.distributed.init_process_group(backend="nccl")

        device_type = autodetect_device_type() if args.device_type == "" else args.device_type
        if device_type == "cuda":
            torch.cuda.set_device(local_rank % torch.cuda.device_count())
            device = torch.device(f"cuda:{local_rank % torch.cuda.device_count()}")
        else:
            device = torch.device("cpu")
        print0(f"Device: {device}, rank {rank}/{world_size}")

        tokenizer = get_tokenizer()
        token_bytes = get_token_bytes(device=device)
        vocab_size = tokenizer.get_vocab_size()

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
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        model.init_weights()
        print0(
            f"Model: depth={args.depth}, dim={model_dim}, heads={num_heads}, "
            f"params={sum(p.numel() for p in model.parameters()):,}"
        )

        matrix_params = list(model.transformer.h.parameters())
        value_embeds_params = list(model.value_embeds.parameters())
        embedding_params = list(model.transformer.wte.parameters())
        lm_head_params = list(model.lm_head.parameters())
        resid_params = [model.resid_lambdas]
        x0_params = [model.x0_lambdas]
        smear_params = [model.smear_gate.weight, model.smear_lambda, model.backout_lambda]

        dmodel_lr_scale = (model_dim / 768) ** -0.5
        tokens_per_batch = args.device_batch_size * args.max_seq_len
        total_batch_size = args.total_batch_size if args.total_batch_size > 0 else tokens_per_batch * world_size
        batch_lr_scale = (total_batch_size / (2**19)) ** 0.5 if total_batch_size != 2**19 else 1.0
        scalar_lr = 0.5 * batch_lr_scale
        param_groups = [
            dict(kind="adamw", params=lm_head_params, lr=0.008 * dmodel_lr_scale * batch_lr_scale,
                 betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind="adamw", params=embedding_params, lr=0.3 * dmodel_lr_scale * batch_lr_scale,
                 betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind="adamw", params=value_embeds_params, lr=0.3 * dmodel_lr_scale * batch_lr_scale * 0.5,
                 betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind="adamw", params=resid_params, lr=scalar_lr * 0.01,
                 betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind="adamw", params=x0_params, lr=scalar_lr,
                 betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
            dict(kind="adamw", params=smear_params, lr=0.2,
                 betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(
                dict(kind="muon", params=group_params, lr=args.matrix_lr * batch_lr_scale,
                     momentum=0.95, ns_steps=args.ns_steps, weight_decay=args.weight_decay)
            )

        optimizer = (DistMuonAdamW if ddp else MuonAdamW)(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]

        model = torch.compile(model, dynamic=False)

        train_loader = tokenizing_distributed_data_loader_bos_bestfit(
            tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device
        )
        build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(
            tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device
        )
        x, y = next(train_loader)

        num_iterations = args.max_steps

        def get_lr_multiplier(it):
            warmup_iters = args.warmup_steps
            warmdown_iters = round(args.warmdown_ratio * num_iterations)
            if it < warmup_iters:
                return (it + 1) / warmup_iters
            if it <= num_iterations - warmdown_iters:
                return 1.0
            progress = (num_iterations - it) / warmdown_iters
            return progress * 1.0 + (1 - progress) * args.final_lr_frac

        def get_muon_momentum(it):
            warmdown_iters = round(args.warmdown_ratio * num_iterations)
            warmdown_start = num_iterations - warmdown_iters
            if it < min(400, num_iterations // 4):
                frac = it / min(400, num_iterations // 4)
                return (1 - frac) * 0.85 + frac * 0.97
            if it >= warmdown_start:
                progress = (it - warmdown_start) / warmdown_iters
                return 0.97 * (1 - progress) + 0.90 * progress
            return 0.97

        def get_weight_decay(it):
            return args.weight_decay * 0.5 * (1 + math.cos(math.pi * it / num_iterations))

        world_tokens_per_fwdbwd = tokens_per_batch * world_size
        assert total_batch_size % world_tokens_per_fwdbwd == 0, (
            f"total_batch_size ({total_batch_size}) must be divisible by "
            f"device_batch_size * max_seq_len * world_size ({world_tokens_per_fwdbwd})"
        )
        grad_accum_steps = max(1, total_batch_size // world_tokens_per_fwdbwd)

        print0(
            f"Training for {num_iterations} steps, batch_size={total_batch_size}, "
            f"world_size={world_size}, grad_accum={grad_accum_steps}"
        )

        smooth_train_loss = 0.0
        train_losses = []
        val_bpbs = []
        best_val_bpb = float("inf")

        for step in range(num_iterations + 1):
            last_step = step == num_iterations

            if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
                model.eval()
                val_loader = build_val_loader()
                eval_steps = max(1, args.eval_tokens // world_tokens_per_fwdbwd)
                val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
                val_bpbs.append({"step": step, "val_bpb": val_bpb})
                best_val_bpb = min(best_val_bpb, val_bpb)
                print0(f"  Step {step:05d} | val_bpb: {val_bpb:.6f} (best: {best_val_bpb:.6f})")
                model.train()

            if last_step:
                break

            t0 = time.time()
            for _ in range(grad_accum_steps):
                loss = model(x, y)
                train_loss_val = loss.detach().item()
                (loss / grad_accum_steps).backward()
                x, y = next(train_loader)

            lrm = get_lr_multiplier(step)
            mom = get_muon_momentum(step)
            wd = get_weight_decay(step)
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * lrm
                if group["kind"] == "muon":
                    group["momentum"] = mom
                    group["weight_decay"] = wd

            optimizer.step()
            model.zero_grad(set_to_none=True)

            smooth_train_loss = 0.9 * smooth_train_loss + 0.1 * train_loss_val
            debiased = smooth_train_loss / (1 - 0.9 ** (step + 1))
            if step % 50 == 0:
                train_losses.append({"step": step, "loss": debiased})
                print0(f"  Step {step:05d} | loss: {debiased:.6f} | dt: {(time.time() - t0) * 1000:.0f}ms")
            if step == 0:
                gc.collect()

        results["score"] = best_val_bpb
        results["val_bpb_final"] = val_bpbs[-1]["val_bpb"] if val_bpbs else None
        results["val_bpb_best"] = best_val_bpb
        results["train_loss_final"] = train_losses[-1]["loss"] if train_losses else None
        results["val_bpbs"] = val_bpbs
        results["train_losses"] = train_losses
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
    if rank == 0:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2))
        print0(f"Results written to {output_path}")
        if results.get("score") is not None:
            print0(f"SCORE: {results['score']:.6f}")

    if ddp:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
