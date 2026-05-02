#!/usr/bin/env python3
"""
Muon-LITE training eval runner (arxiv 2602.22681).
Same pipeline as run_native_muon.py but uses Muon-LITE (flat-direction
amplification) for matrix param groups.

Hyperparameters:
    --lite-chi CHI   : flat-direction amplification (default 2.0)
    --lite-ds DS     : sharp subspace dim (default 4)
    --ns-steps N     : Polar Express iterations (default 5)
"""
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import sys, json, time, math, gc, argparse
from pathlib import Path
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
NANOCHAT_DIR = SCRIPT_DIR / "nanochat"

parser = argparse.ArgumentParser()
parser.add_argument("--nanochat-dir", type=str, default=str(NANOCHAT_DIR))
parser.add_argument("--output-file", type=str, default="eval_result.json")
parser.add_argument("--depth", type=int, default=8)
parser.add_argument("--head-dim", type=int, default=64)
parser.add_argument("--aspect-ratio", type=int, default=64)
parser.add_argument("--max-seq-len", type=int, default=1024)
parser.add_argument("--max-steps", type=int, default=7630)
parser.add_argument("--device-batch-size", type=int, default=32)
parser.add_argument("--total-batch-size", type=int, default=-1)
parser.add_argument("--matrix-lr", type=float, default=0.02)
parser.add_argument("--weight-decay", type=float, default=0.28)
parser.add_argument("--warmup-steps", type=int, default=40)
parser.add_argument("--warmdown-ratio", type=float, default=0.65)
parser.add_argument("--final-lr-frac", type=float, default=0.05)
parser.add_argument("--ns-steps", type=int, default=5)
parser.add_argument("--eval-every", type=int, default=1000)
parser.add_argument("--eval-tokens", type=int, default=524288)
parser.add_argument("--device-type", type=str, default="")
parser.add_argument("--seed", type=int, default=42)
# LITE-specific (paper defaults: chi=2, r_s=0.1, warmup=0.5)
parser.add_argument("--lite-chi", type=float, default=2.0, help="Flat-direction amplification (>=1)")
parser.add_argument("--lite-rs", type=float, default=0.1, help="Sharp subspace ratio: d_s = r_s * min(m,n)")
parser.add_argument("--lite-chi-warmup", type=float, default=0.5, help="Fraction of training to warm χ from 1 to target")
parser.add_argument("--lite-chi-schedule", type=str, default="warmup_hold",
                    choices=("warmup_hold", "triangular", "first_d1_gate"),
                    help="χ schedule: warmup_hold preserves fixed-LITE; triangular decays χ back to 1 by the end; first_d1_gate gates after first M_tilde D1 probe")
parser.add_argument("--lite-gate-threshold", type=float, default=0.417,
                    help="D1 threshold for first_d1_gate schedule")
parser.add_argument("--lite-gate-metric", type=str, default="cstar",
                    choices=("cstar", "hard"),
                    help="D1 metric for first_d1_gate: cstar or hard")
parser.add_argument("--lite-gate-low-chi", type=float, default=1.0,
                    help="χ target when first_d1_gate decides unresolved")
parser.add_argument("--lite-beta1", type=float, default=0.0, help="Hessian damping coef for sharp directions (paper LITE-H: -0.25)")
parser.add_argument("--lite-beta2", type=float, default=0.0, help="Hessian damping coef for flat directions (paper LITE-H: 2.0, full LITE: 1.0)")
parser.add_argument("--optimizer-switch-mode", type=str, default="none",
                    choices=("none", "fixed_muon", "fixed_lite", "muon_to_lite", "lite_to_muon"),
                    help="True matrix optimizer mode/switch; fixed_muon is native Muon, not a chi=1 proxy")
parser.add_argument("--switch-metric", type=str, default="cstar",
                    choices=("cstar", "hard"),
                    help="M_tilde D1 metric used for optimizer switching")
parser.add_argument("--switch-threshold", type=float, default=0.417,
                    help="Threshold for optimizer switching")
parser.add_argument("--switch-min-probe-index", type=int, default=1,
                    help="First D1 probe index eligible for switching (1-indexed)")
parser.add_argument("--resume-checkpoint", type=str, default="",
                    help="Resume model/optimizer/train-loader state from a checkpoint saved by this runner")
parser.add_argument("--resume-matrix-mode", type=str, default="checkpoint",
                    choices=("checkpoint", "muon", "lite"),
                    help="Override switch_muon_lite matrix groups after loading a checkpoint")
parser.add_argument("--save-checkpoint-dir", type=str, default="",
                    help="Directory for rank-0 training-state checkpoints")
parser.add_argument("--save-checkpoint-steps", type=str, default="",
                    help="Comma-separated global steps at which to save checkpoints before the optimizer step")
parser.add_argument("--stop-after-last-checkpoint", action="store_true",
                    help="Exit after saving the largest requested checkpoint step")
args = parser.parse_args()

import random; random.seed(args.seed); torch.manual_seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

sys.path.insert(0, args.nanochat_dir)
sys.path.insert(0, str(SCRIPT_DIR))

from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import (tokenizing_distributed_data_loader_bos_bestfit,
                                 tokenizing_distributed_data_loader_with_state_bos_bestfit)
from nanochat.common import COMPUTE_DTYPE, print0, autodetect_device_type
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.loss_eval import evaluate_bpb
from muon_lite import MuonLiteAdamW


def main():
    fixed_mode = args.optimizer_switch_mode in ("fixed_muon", "fixed_lite")
    switching_policy = args.optimizer_switch_mode in ("muon_to_lite", "lite_to_muon")
    if args.optimizer_switch_mode == "none":
        optimizer_name = "Muon-LITE"
    elif fixed_mode:
        optimizer_name = args.optimizer_switch_mode.replace("fixed_", "Fixed-")
    else:
        optimizer_name = f"Switch-{args.optimizer_switch_mode}"

    def parse_step_set(spec: str) -> set[int]:
        if not spec.strip():
            return set()
        return {int(x.strip()) for x in spec.split(",") if x.strip()}

    save_checkpoint_steps = parse_step_set(args.save_checkpoint_steps)
    results = {"start_time": time.time(), "args": vars(args), "optimizer": optimizer_name}
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
        config = GPTConfig(sequence_len=args.max_seq_len, vocab_size=vocab_size,
                           n_layer=args.depth, n_head=num_heads, n_kv_head=num_heads,
                           n_embd=model_dim, window_pattern="L")
        with torch.device("meta"):
            model = GPT(config)
        model.to_empty(device=device)
        torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
        model.init_weights()
        print0(f"Model: depth={args.depth}, dim={model_dim}, params={sum(p.numel() for p in model.parameters()):,}")

        matrix_params = list(model.transformer.h.parameters())
        # Names of 2D matrix params for D1 telemetry (subset of matrix_params)
        matrix_params_named = {n: p for n, p in model.named_parameters()
                               if p.requires_grad and p.ndim == 2 and "transformer.h." in n}
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
        matrix_kind = "switch_muon_lite" if args.optimizer_switch_mode != "none" else "muon_lite"
        initial_switch_mode = "muon" if args.optimizer_switch_mode in ("fixed_muon", "muon_to_lite") else "lite"
        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=0.008*dmodel_lr_scale*batch_lr_scale,
                 betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=0.3*dmodel_lr_scale*batch_lr_scale,
                 betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=value_embeds_params, lr=0.3*dmodel_lr_scale*batch_lr_scale*0.5,
                 betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=resid_params, lr=scalar_lr*0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            gp = [p for p in matrix_params if p.shape == shape]
            # Per-shape sharp subspace dim = r_s * min(m, n)  (paper defaults r_s=0.1)
            d_s_shape = max(1, int(round(args.lite_rs * min(shape[-2], shape[-1]))))
            param_groups.append(dict(
                kind=matrix_kind, params=gp, lr=args.matrix_lr*batch_lr_scale,
                momentum=0.95, ns_steps=args.ns_steps, weight_decay=args.weight_decay,
                lite_chi_target=args.lite_chi, lite_chi=1.0,  # warmup scheduler sets lite_chi
                lite_ds=d_s_shape,
                lite_beta1=args.lite_beta1, lite_beta2=args.lite_beta2,
                switch_mode=initial_switch_mode,
            ))

        optimizer = MuonLiteAdamW(param_groups)
        for g in optimizer.param_groups: g['initial_lr'] = g['lr']

        resume_payload = None
        resume_step = 0
        if args.resume_checkpoint:
            ckpt_path = Path(args.resume_checkpoint)
            print0(f"Loading checkpoint: {ckpt_path}")
            resume_payload = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(resume_payload["model"])
            optimizer.load_state_dict(resume_payload["optimizer"])
            for g in optimizer.param_groups:
                g.setdefault('initial_lr', g['lr'])
            resume_step = int(resume_payload.get("step", 0))
            rng = resume_payload.get("rng", {})
            if "python" in rng:
                random.setstate(rng["python"])
            if "torch" in rng:
                torch.random.set_rng_state(rng["torch"].cpu())
            if torch.cuda.is_available() and "cuda_all" in rng:
                torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda_all"]])
            if args.resume_matrix_mode != "checkpoint":
                for g in optimizer.param_groups:
                    if g.get("kind") == "switch_muon_lite":
                        g["switch_mode"] = args.resume_matrix_mode
            print0(f"Resumed at global step {resume_step}, matrix_mode_override={args.resume_matrix_mode}")
            results["resumed_from_checkpoint"] = str(ckpt_path)
            results["resume_step"] = resume_step

        model = torch.compile(model, dynamic=False)

        resume_train_state = None
        if resume_payload is not None:
            resume_train_state = resume_payload.get("train_loader_state")
        train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
            tokenizer, args.device_batch_size, args.max_seq_len,
            split="train", device=device, resume_state_dict=resume_train_state)
        train_loader_state = resume_train_state
        def next_train_batch():
            nonlocal train_loader_state
            xb, yb, state = next(train_loader)
            train_loader_state = state
            return xb, yb

        build_val = lambda: tokenizing_distributed_data_loader_bos_bestfit(
            tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device)
        if resume_payload is not None and resume_payload.get("prefetched_batch") is not None:
            prefetched = resume_payload["prefetched_batch"]
            x = prefetched["x"].to(device=device, non_blocking=True)
            y = prefetched["y"].to(device=device, non_blocking=True)
        else:
            x, y = next_train_batch()

        num_iterations = args.max_steps
        d1_gate_decided = False
        d1_gate_value = None
        d1_gate_target = args.lite_chi
        d1_gate_use_lite = True
        switch_done = not switching_policy
        switch_probe_index = 0
        current_switch_mode = initial_switch_mode
        for g in optimizer.param_groups:
            if g.get("kind") == "switch_muon_lite":
                current_switch_mode = g.get("switch_mode", current_switch_mode)
                break
        switch_decisions = []
        def get_chi(it, chi_target):
            """Schedule χ. Default matches fixed-LITE after warmup."""
            if args.lite_chi_schedule == "first_d1_gate" and d1_gate_decided and not d1_gate_use_lite:
                return d1_gate_target
            warmup_iters = max(1, int(args.lite_chi_warmup * num_iterations))
            if it < warmup_iters:
                return 1.0 + (chi_target - 1.0) * (it / warmup_iters)
            if args.lite_chi_schedule == "triangular":
                decay_iters = max(1, num_iterations - warmup_iters)
                frac = min(1.0, (it - warmup_iters) / decay_iters)
                return chi_target + (1.0 - chi_target) * frac
            return chi_target
        def get_lr_mul(it):
            warmup = args.warmup_steps; wd_iters = round(args.warmdown_ratio * num_iterations)
            if it < warmup: return (it + 1) / warmup
            if it <= num_iterations - wd_iters: return 1.0
            prog = (num_iterations - it) / wd_iters
            return prog * 1.0 + (1 - prog) * args.final_lr_frac
        def get_mom(it):
            wd_iters = round(args.warmdown_ratio * num_iterations); wd_start = num_iterations - wd_iters
            if it < min(400, num_iterations // 4):
                frac = it / min(400, num_iterations // 4)
                return (1 - frac) * 0.85 + frac * 0.97
            elif it >= wd_start:
                p = (it - wd_start) / wd_iters
                return 0.97 * (1 - p) + 0.90 * p
            return 0.97
        def get_wd(it): return args.weight_decay * 0.5 * (1 + math.cos(math.pi * it / num_iterations))
        def fmt3(x):
            return "None" if x is None else f"{x:.3f}"

        # Match nanochat upstream base_train.py: world_tokens_per_fwdbwd = tokens_per_batch * world_size.
        # Original run_lite.py omits the world_size divisor (effective bsz = world_size × nominal).
        world_tokens_per_fwdbwd = tokens_per_batch * world_size
        assert total_batch_size % world_tokens_per_fwdbwd == 0, (
            f"total_batch_size ({total_batch_size}) must be divisible by "
            f"tokens_per_batch * world_size ({world_tokens_per_fwdbwd})")
        grad_accum = max(1, total_batch_size // world_tokens_per_fwdbwd)
        n_matrix_groups = sum(1 for g in param_groups if g.get('kind') == 'muon_lite')
        n_matrix_groups += sum(1 for g in param_groups if g.get('kind') == 'switch_muon_lite')
        sample_ds = next((g['lite_ds'] for g in param_groups
                          if g.get('kind') in ('muon_lite', 'switch_muon_lite')), 0)
        print0(f"Training {num_iterations} steps, batch={total_batch_size}, "
               f"ns_steps={args.ns_steps}, lite_chi={args.lite_chi}, lite_rs={args.lite_rs} "
               f"(sample d_s={sample_ds}), chi_warmup={args.lite_chi_warmup}, "
               f"chi_schedule={args.lite_chi_schedule}, switch_mode={args.optimizer_switch_mode}, "
               f"n_matrix_groups={n_matrix_groups}")

        # --- telemetry ---
        from train_telemetry import (grad_norm_per_group, midtraining_d1_probe,
                                     d1_from_buffers, should_probe, spectral_stats)
        tel_loss, tel_grad_norms, tel_lr_mul, tel_mom, tel_chi, tel_update_mode = [], [], [], [], [], []
        tel_d1_probes, tel_d1_probes_momentum = [], []
        tel_d1_probes_momentum_left, tel_d1_probes_momentum_buffer, tel_spectra = [], [], []
        tel_gate_decisions = []

        # Auxiliary split-batch momenta for D1(m) probe (V2). These are read-only
        # diagnostics; optimizer keeps using its own buffer fed by g_full. We
        # mirror nanochat's scheduled momentum and probe the Nesterov-corrected
        # direction M_tilde = (1-mom) * g + mom * m_new, which is the actual
        # matrix passed to LITE's polar/projection step.
        do_momentum_d1 = grad_accum >= 2
        if do_momentum_d1:
            m_A = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in matrix_params_named.items()}
            m_B = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in matrix_params_named.items()}
            mt_A = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in matrix_params_named.items()}
            mt_B = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in matrix_params_named.items()}
            print0(f"[V2] Maintaining split-batch scheduled momentum + M_tilde for D1 probe; "
                   f"grad_accum={grad_accum}, half={grad_accum // 2}")
        else:
            m_A = m_B = mt_A = mt_B = None
            print0(f"[V2] grad_accum={grad_accum} < 2, momentum-D1 disabled this run")

        val_bpbs = []; best_val_bpb = float('inf')
        max_checkpoint_step = max(save_checkpoint_steps) if save_checkpoint_steps else None

        def save_training_checkpoint(step: int) -> Path:
            ckpt_dir = Path(args.save_checkpoint_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"checkpoint_step{step:06d}.pt"
            payload = {
                "step": step,
                "model": model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_loader_state": train_loader_state,
                "prefetched_batch": {
                    "x": x.detach().cpu(),
                    "y": y.detach().cpu(),
                },
                "rng": {
                    "python": random.getstate(),
                    "torch": torch.random.get_rng_state(),
                    "cuda_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                },
                "args": vars(args),
            }
            torch.save(payload, ckpt_path)
            return ckpt_path

        for step in range(resume_step, num_iterations + 1):
            last = step == num_iterations
            if args.eval_every > 0 and (last or step % args.eval_every == 0):
                model.eval()
                vl = build_val()
                eval_steps = max(1, args.eval_tokens // tokens_per_batch)
                vb = evaluate_bpb(model, vl, eval_steps, token_bytes)
                val_bpbs.append({"step": step, "val_bpb": vb})
                if vb < best_val_bpb: best_val_bpb = vb
                print0(f"  Step {step:5d} | val_bpb: {vb:.6f} (best: {best_val_bpb:.6f})")
                model.train()

            # mid-training D1 probe (5 uniformly-spaced points)
            if should_probe(step, num_iterations, n_probes=5) and not last:
                probe_loader = build_val()
                probe_micros = int(os.environ.get("PROBE_MICROS", "4"))
                d1_stats = midtraining_d1_probe(model, probe_loader,
                                                micros_per_half=probe_micros,
                                                top_k=16)
                d1_stats["step"] = step
                tel_d1_probes.append(d1_stats)
                print0(f"  Step {step:5d} | D1(g) probe: frac>0.7={d1_stats['frac_above_0_7_median']:.3f}")

                # V2 momentum-D1 probe on the latest split-batch Nesterov direction.
                if do_momentum_d1 and m_A is not None:
                    mb_d1_stats = d1_from_buffers(m_A, m_B, top_k=16, alignment_side="lite")
                    mb_d1_stats["step"] = step
                    tel_d1_probes_momentum_buffer.append(mb_d1_stats)

                    m_left_stats = d1_from_buffers(mt_A, mt_B, top_k=16, alignment_side="left")
                    m_left_stats["step"] = step
                    tel_d1_probes_momentum_left.append(m_left_stats)

                    m_d1_stats = d1_from_buffers(mt_A, mt_B, top_k=16, alignment_side="lite")
                    m_d1_stats["step"] = step
                    tel_d1_probes_momentum.append(m_d1_stats)
                    switch_probe_index += 1
                    print0(f"  Step {step:5d} | D1(M_tilde) probe: "
                           f"frac>0.7={fmt3(m_d1_stats['frac_above_0_7_median'])}  "
                           f"frac>c*={fmt3(m_d1_stats['frac_above_c_star_median'])}  "
                           f"[buffer c*={fmt3(mb_d1_stats['frac_above_c_star_median'])}]")

                    if args.lite_chi_schedule == "first_d1_gate" and not d1_gate_decided:
                        if args.lite_gate_metric == "cstar":
                            d1_gate_value = m_d1_stats["frac_above_c_star_median"]
                        else:
                            d1_gate_value = m_d1_stats["frac_above_0_7_median"]
                        use_lite = d1_gate_value is not None and d1_gate_value >= args.lite_gate_threshold
                        d1_gate_target = args.lite_chi if use_lite else args.lite_gate_low_chi
                        d1_gate_use_lite = use_lite
                        d1_gate_decided = True
                        decision = {
                            "step": step,
                            "metric": args.lite_gate_metric,
                            "value": d1_gate_value,
                            "threshold": args.lite_gate_threshold,
                            "target_chi": d1_gate_target,
                            "use_lite": use_lite,
                        }
                        tel_gate_decisions.append(decision)
                        print0(f"  Step {step:5d} | first_d1_gate: "
                               f"{args.lite_gate_metric}={fmt3(d1_gate_value)} "
                               f"threshold={args.lite_gate_threshold:.3f} -> "
                               f"target_chi={d1_gate_target:.3f}")

                    if (args.optimizer_switch_mode != "none" and not switch_done
                            and switch_probe_index >= args.switch_min_probe_index):
                        if args.switch_metric == "cstar":
                            switch_value = m_d1_stats["frac_above_c_star_median"]
                        else:
                            switch_value = m_d1_stats["frac_above_0_7_median"]
                        if args.optimizer_switch_mode == "muon_to_lite":
                            target_mode = "lite"
                            should_switch = switch_value is not None and switch_value >= args.switch_threshold
                            relation = ">="
                        else:
                            target_mode = "muon"
                            should_switch = switch_value is not None and switch_value < args.switch_threshold
                            relation = "<"
                        decision = {
                            "step": step,
                            "probe_index": switch_probe_index,
                            "switch_mode": args.optimizer_switch_mode,
                            "metric": args.switch_metric,
                            "value": switch_value,
                            "threshold": args.switch_threshold,
                            "relation": relation,
                            "from_mode": current_switch_mode,
                            "target_mode": target_mode,
                            "switched": bool(should_switch),
                        }
                        if should_switch:
                            for g in optimizer.param_groups:
                                if g.get("kind") == "switch_muon_lite":
                                    g["switch_mode"] = target_mode
                            current_switch_mode = target_mode
                            switch_done = True
                        switch_decisions.append(decision)
                        print0(f"  Step {step:5d} | optimizer_switch: "
                               f"{args.switch_metric}={fmt3(switch_value)}; "
                               f"condition {relation} {args.switch_threshold:.3f} "
                               f"{'met' if should_switch else 'not met'} -> "
                               f"{'switch to ' + target_mode if should_switch else 'stay ' + current_switch_mode}")

                tel_spectra.append({"step": step, **spectral_stats(model)})

            if rank == 0 and args.save_checkpoint_dir and step in save_checkpoint_steps:
                ckpt_path = save_training_checkpoint(step)
                print0(f"  Step {step:5d} | saved checkpoint: {ckpt_path}")
                if args.stop_after_last_checkpoint and max_checkpoint_step == step:
                    results.update(dict(
                        score=best_val_bpb if best_val_bpb < float('inf') else None,
                        val_bpb_best=best_val_bpb if best_val_bpb < float('inf') else None,
                        val_bpbs=val_bpbs,
                        stopped_after_checkpoint=str(ckpt_path),
                        error=None,
                    ))
                    break

            if last: break

            lrm = get_lr_mul(step); mom = get_mom(step); wd = get_wd(step)
            t0 = time.time()
            half = grad_accum // 2 if do_momentum_d1 else 0
            g_A_partial = {}
            for i in range(grad_accum):
                loss = model(x, y)
                (loss / grad_accum).backward()
                x, y = next_train_batch()
                if do_momentum_d1 and (i + 1) == half:
                    scale = grad_accum / half
                    for n, p in matrix_params_named.items():
                        if p.grad is not None:
                            g_A_partial[n] = (scale * p.grad.detach().to(torch.float32)).clone()

            if do_momentum_d1:
                rest = grad_accum - half
                for n, p in matrix_params_named.items():
                    if n in g_A_partial and p.grad is not None:
                        g_full = p.grad.detach().to(torch.float32)
                        g_A = g_A_partial[n]
                        g_B = (grad_accum * g_full - half * g_A) / rest
                        m_A[n].mul_(mom).add_(g_A, alpha=1.0 - mom)
                        m_B[n].mul_(mom).add_(g_B, alpha=1.0 - mom)
                        mt_A[n].copy_(g_A).mul_(1.0 - mom).add_(m_A[n], alpha=mom)
                        mt_B[n].copy_(g_B).mul_(1.0 - mom).add_(m_B[n], alpha=mom)
                g_A_partial.clear()

            gn = grad_norm_per_group(optimizer)

            current_chi = None
            for g in optimizer.param_groups:
                g['lr'] = g['initial_lr'] * lrm
                if g['kind'] in ('muon', 'muon_lite', 'switch_muon_lite'):
                    g['momentum'] = mom
                    g['weight_decay'] = wd
                    if 'lite_chi_target' in g:
                        g['lite_chi'] = get_chi(step, g['lite_chi_target'])
                        current_chi = g['lite_chi']
            optimizer.step()
            model.zero_grad(set_to_none=True)
            if step == 0: gc.collect()

            tel_loss.append(loss.item())
            tel_grad_norms.append(gn)
            tel_lr_mul.append(lrm)
            tel_mom.append(mom)
            tel_chi.append(current_chi)
            tel_update_mode.append(current_switch_mode if args.optimizer_switch_mode != "none" else "lite")

            if step % 50 == 0:
                print0(f"  Step {step:5d} | loss: {loss.item():.4f} | dt: {(time.time()-t0)*1000:.0f}ms")

        results.update(dict(
            score=best_val_bpb, val_bpb_best=best_val_bpb, val_bpbs=val_bpbs, error=None,
            telemetry=dict(
                train_loss_per_step=tel_loss,
                grad_norms_per_step=tel_grad_norms,
                lr_mul_per_step=tel_lr_mul,
                momentum_per_step=tel_mom,
                chi_per_step=tel_chi,
                update_mode_per_step=tel_update_mode,
                d1_probes=tel_d1_probes,
                d1_probes_momentum=tel_d1_probes_momentum,
                d1_probes_momentum_left=tel_d1_probes_momentum_left,
                d1_probes_momentum_buffer=tel_d1_probes_momentum_buffer,
                gate_decisions=tel_gate_decisions,
                switch_decisions=switch_decisions,
                spectra=tel_spectra,
            ),
        ))
    except Exception as e:
        import traceback
        results.update(dict(error=str(e), traceback=traceback.format_exc(), score=None))
        print0(f"ERROR: {e}"); traceback.print_exc()

    results["end_time"] = time.time()
    if rank == 0:
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_file).write_text(json.dumps(results, indent=2))
        print0(f"SCORE: {results.get('score')}")
    if ddp: torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
