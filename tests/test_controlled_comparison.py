#!/usr/bin/env python3
"""
Controlled 对拍: same seed, same data, only optimizer differs.
This isolates the true gap between StreamingMuon and Polar Express Muon.
"""
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import sys
import gc
import json
import time
import math
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "nanochat"))
sys.path.insert(0, os.path.dirname(__file__))

import torch
from nanochat.gpt import GPT, GPTConfig
from nanochat.common import COMPUTE_DTYPE
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb
from nanochat.optim import MuonAdamW
from streaming_muon_torch import StreamingMuonAdamW


import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument("--seed", type=int, default=42)
_parser.add_argument("--depth", type=int, default=4)
_parser.add_argument("--max-steps", type=int, default=500)
_parser.add_argument("--batch", type=int, default=8)
_parser.add_argument("--seq-len", type=int, default=512)
_args, _ = _parser.parse_known_args()

SEED = _args.seed
DEPTH = _args.depth
MAX_STEPS = _args.max_steps
DEVICE_BATCH = _args.batch
SEQ_LEN = _args.seq_len


def make_model(device):
    """Create model with deterministic initialization."""
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    model_dim = DEPTH * 64
    num_heads = model_dim // 64
    tokenizer = get_tokenizer()
    config = GPTConfig(
        sequence_len=SEQ_LEN, vocab_size=tokenizer.get_vocab_size(),
        n_layer=DEPTH, n_head=num_heads, n_kv_head=num_heads,
        n_embd=model_dim, window_pattern="L",
    )

    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    torch.manual_seed(SEED)  # reset seed before init_weights
    torch.cuda.manual_seed_all(SEED)
    model.init_weights()
    return model, config, tokenizer


def make_optimizer(model, kind="streaming_muon", num_iters=1, poly_order=1):
    """Create optimizer matching nanochat's setup_optimizer."""
    model_dim = model.config.n_embd
    matrix_params = list(model.transformer.h.parameters())
    value_embed_params = list(model.value_embeds.parameters())
    embedding_params = list(model.transformer.wte.parameters())
    lm_head_params = list(model.lm_head.parameters())
    resid_params = [model.resid_lambdas]
    x0_params = [model.x0_lambdas]
    smear_params = [model.smear_gate.weight, model.smear_lambda, model.backout_lambda]

    dmodel_lr_scale = (model_dim / 768) ** -0.5
    tokens_per_batch = DEVICE_BATCH * SEQ_LEN
    batch_lr_scale = (tokens_per_batch / (2**19)) ** 0.5

    param_groups = [
        dict(kind='adamw', params=lm_head_params, lr=0.008*dmodel_lr_scale*batch_lr_scale,
             betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
        dict(kind='adamw', params=embedding_params, lr=0.3*dmodel_lr_scale*batch_lr_scale,
             betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
        dict(kind='adamw', params=value_embed_params, lr=0.15*dmodel_lr_scale*batch_lr_scale,
             betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
        dict(kind='adamw', params=resid_params, lr=0.005, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
        dict(kind='adamw', params=x0_params, lr=0.5*batch_lr_scale, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        dict(kind='adamw', params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
    ]

    for shape in sorted({p.shape for p in matrix_params}):
        group_params = [p for p in matrix_params if p.shape == shape]
        if kind == 'muon':
            param_groups.append(dict(
                kind='muon', params=group_params, lr=0.02*batch_lr_scale,
                momentum=0.95, ns_steps=5, weight_decay=0.28,
            ))
        else:
            param_groups.append(dict(
                kind='streaming_muon', params=group_params, lr=0.02*batch_lr_scale,
                momentum=0.95, weight_decay=0.28,
                sigma_transform='identity', num_iters=num_iters,
                poly_order=poly_order,
            ))

    OptClass = MuonAdamW if kind == 'muon' else StreamingMuonAdamW
    optimizer = OptClass(param_groups)
    for g in optimizer.param_groups:
        g["initial_lr"] = g["lr"]
    return optimizer


def get_lr_multiplier(it, num_iterations):
    warmup_iters = 40
    warmdown_iters = round(0.65 * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * 0.05


def get_muon_momentum(it, num_iterations):
    warmdown_iters = round(0.65 * num_iterations)
    warmdown_start = num_iterations - warmdown_iters
    if it < min(400, num_iterations // 4):
        frac = it / min(400, num_iterations // 4)
        return (1 - frac) * 0.85 + frac * 0.97
    elif it >= warmdown_start:
        progress = (it - warmdown_start) / warmdown_iters
        return 0.97 * (1 - progress) + 0.90 * progress
    else:
        return 0.97


def get_weight_decay(it, num_iterations, base_wd=0.28):
    return base_wd * 0.5 * (1 + math.cos(math.pi * it / num_iterations))


def train_run(name, kind, num_iters=1, poly_order=1, device='cuda'):
    """Full training run with val BPB evaluation."""
    print(f"\n{'='*60}")
    print(f"  {name} (kind={kind}, iters={num_iters}, poly={poly_order})")
    print(f"{'='*60}")

    model, config, tokenizer = make_model(device)
    optimizer = make_optimizer(model, kind=kind, num_iters=num_iters, poly_order=poly_order)
    token_bytes = get_token_bytes(device=device)
    tokens_per_batch = DEVICE_BATCH * SEQ_LEN

    # Verify init: print first few param values for determinism check
    p0 = list(model.transformer.h.parameters())[0]
    print(f"  Init check: param[0] hash = {p0.data.flatten()[:8].sum().item():.6f}")

    train_loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, DEVICE_BATCH, SEQ_LEN, split="train", device=device,
    )
    build_val = lambda: tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, DEVICE_BATCH, SEQ_LEN, split="val", device=device,
    )
    x, y = next(train_loader)

    val_bpbs = []
    smooth_loss = 0.0
    muon_kind = 'muon' if kind == 'muon' else 'streaming_muon'

    for step in range(MAX_STEPS + 1):
        last = step == MAX_STEPS

        # Eval
        if step % 100 == 0 or last:
            model.eval()
            vl = build_val()
            eval_steps = max(1, 524288 // tokens_per_batch)
            vb = evaluate_bpb(model, vl, eval_steps, token_bytes)
            val_bpbs.append((step, vb))
            print(f"  Step {step:04d} | val_bpb: {vb:.6f}")
            model.train()

        if last:
            break

        # Train step
        loss = model(x, y)
        loss.backward()
        x, y = next(train_loader)

        lrm = get_lr_multiplier(step, MAX_STEPS)
        mom = get_muon_momentum(step, MAX_STEPS)
        wd = get_weight_decay(step, MAX_STEPS)
        for g in optimizer.param_groups:
            g["lr"] = g["initial_lr"] * lrm
            if g['kind'] == muon_kind:
                g["momentum"] = mom
                g["weight_decay"] = wd

        optimizer.step()
        model.zero_grad(set_to_none=True)
        if step == 0:
            gc.collect()

    best_bpb = min(v for _, v in val_bpbs)
    print(f"  Best val BPB: {best_bpb:.6f}")

    del model, optimizer
    torch.cuda.empty_cache()
    gc.collect()

    return val_bpbs, best_bpb


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Controlled 对拍: SEED={SEED}, DEPTH={DEPTH}, MAX_STEPS={MAX_STEPS}")

    results = {}

    # Run all configs: (name, kind, num_iters, poly_order)
    configs = [
        ("Polar Express", "muon", 5, 1),
        ("Stream i=1 p=1", "streaming_muon", 1, 1),   # baseline streaming
        ("Stream i=1 p=2", "streaming_muon", 1, 2),   # poly accel: gram^2
        ("Stream i=2 p=1", "streaming_muon", 2, 1),   # 2 iters (gram reuse)
        ("Stream i=1 p=3", "streaming_muon", 1, 3),   # poly accel: gram^3
    ]
    for name, kind, iters, poly in configs:
        bpbs, best = train_run(name, kind, num_iters=iters, poly_order=poly, device=device)
        results[name] = {"val_bpbs": bpbs, "best": best}

    # Print comparison
    print(f"\n{'='*60}")
    print(f"  Controlled 对拍 Results (same seed={SEED})")
    print(f"{'='*60}")
    print(f"\n{'Step':>5} |", end="")
    for name in results:
        print(f" {name:>20} |", end="")
    print()
    print("-" * 80)

    muon_best = results["Polar Express"]["best"]
    steps_set = [s for s, _ in results["Polar Express"]["val_bpbs"]]
    for i, step in enumerate(steps_set):
        print(f"{step:5d} |", end="")
        for name in results:
            bpb = results[name]["val_bpbs"][i][1]
            print(f" {bpb:20.6f} |", end="")
        print()

    print(f"\nBest BPB:")
    for name in results:
        best = results[name]["best"]
        diff = (best - muon_best) / muon_best * 100
        print(f"  {name:30s}: {best:.6f} (Δ = {diff:+.3f}%)")

    # Save
    with open("controlled_comparison.json", "w") as f:
        json.dump({name: {"val_bpbs": bpbs, "best": best}
                   for name, (bpbs, best) in [(n, (r["val_bpbs"], r["best"]))
                                               for n, r in results.items()]},
                  f, indent=2)
    print(f"\nResults saved to controlled_comparison.json")


if __name__ == "__main__":
    main()
