#!/usr/bin/env python3
"""
Native nanochat Muon (DistMuonAdamW / MuonAdamW) eval runner.
Same schedule as run_eval.py but using nanochat's built-in Polar Express muon.
Used to diagnose whether divergence at d8/1B/lr=0.02 is StreamingMuon-specific
or a general Muon-family issue.
"""
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import sys, json, time, math, gc, argparse, re
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
parser.add_argument("--metrics-every", type=int, default=0,
                    help="If >0, log Muon diagnostics every N optimizer steps into result.json.")
parser.add_argument("--metrics-top-k", type=int, default=4)
parser.add_argument("--metrics-module-regex", type=str,
                    default=r"transformer\.h\.(?:[0-9]+)\.(?:attn\.(?:c_q|c_k|c_v|c_proj)|mlp\.(?:c_fc|c_proj))\.weight$")
parser.add_argument("--metrics-max-modules", type=int, default=0,
                    help="Maximum modules to log per step; 0 means all selected modules.")
parser.add_argument("--metrics-exact-svd", action="store_true",
                    help="Compute exact SVD for metrics. Native Muon is deprecated and always needs SVD for singular values.")
parser.add_argument("--metrics-full-per-module", action="store_true",
                    help="Store full duplicate per-module rows in result.json. Default keeps compact metadata only.")
parser.add_argument("--metrics-save-components", action="store_true")
parser.add_argument("--metrics-component-dir", type=str, default=None)
parser.add_argument("--metrics-split-momentum", action="store_true")
parser.add_argument("--metrics-split-svd-every", type=int, default=0)
parser.add_argument("--metrics-alignment-side", type=str, default="lite", choices=("left", "right", "lite"))
parser.add_argument("--metrics-hessian-every", type=int, default=0)
parser.add_argument("--metrics-hessian-top-k", type=int, default=1)
parser.add_argument("--metrics-hessian-iters", type=int, default=6)
parser.add_argument("--metrics-hessian-max-modules", type=int, default=0)
parser.add_argument("--device-type", type=str, default="")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
if args.metrics_every > 0:
    raise ValueError(
        "Native Muon metrics are disabled in the clean repo. "
        "Use run_eval.py with StreamingMuon for dynamics logging."
    )

import random; random.seed(args.seed); torch.manual_seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

sys.path.insert(0, args.nanochat_dir)
sys.path.insert(0, str(SCRIPT_DIR))

from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.common import COMPUTE_DTYPE, print0, autodetect_device_type
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.loss_eval import evaluate_bpb
from nanochat.optim import MuonAdamW, DistMuonAdamW


def main():
    results = {"start_time": time.time(), "args": vars(args), "optimizer": "NativeMuon"}
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
        param_name_by_id = {id(p): n for n, p in matrix_params_named.items()}
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
            param_groups.append(dict(
                kind='muon', params=gp, lr=args.matrix_lr*batch_lr_scale,
                momentum=0.95, ns_steps=args.ns_steps, weight_decay=args.weight_decay,
            ))

        OptCls = DistMuonAdamW if ddp else MuonAdamW
        optimizer = OptCls(param_groups)
        for g in optimizer.param_groups: g['initial_lr'] = g['lr']

        eager_model = model
        model = torch.compile(model, dynamic=False)

        train_loader = tokenizing_distributed_data_loader_bos_bestfit(
            tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device)
        build_val = lambda: tokenizing_distributed_data_loader_bos_bestfit(
            tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device)
        x, y = next(train_loader)

        num_iterations = args.max_steps
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

        # Match nanochat upstream base_train.py: world_tokens_per_fwdbwd = tokens_per_batch * world_size.
        # Original run_native_muon.py omits the world_size divisor (effective bsz = world_size × nominal).
        world_tokens_per_fwdbwd = tokens_per_batch * world_size
        assert total_batch_size % world_tokens_per_fwdbwd == 0, (
            f"total_batch_size ({total_batch_size}) must be divisible by "
            f"tokens_per_batch * world_size ({world_tokens_per_fwdbwd})")
        grad_accum = max(1, total_batch_size // world_tokens_per_fwdbwd)
        print0(f"Training {num_iterations} steps, batch={total_batch_size} (global), "
               f"world_size={world_size}, grad_accum={grad_accum}, ns_steps={args.ns_steps}")

        # --- telemetry ---
        from train_telemetry import (grad_norm_per_group, midtraining_d1_probe,
                                     d1_from_buffers, should_probe, spectral_stats)
        tel_loss, tel_grad_norms, tel_lr_mul, tel_mom = [], [], [], []
        tel_d1_probes, tel_d1_probes_momentum = [], []
        tel_d1_probes_momentum_left, tel_d1_probes_momentum_buffer, tel_spectra = [], [], []
        metrics_enabled = args.metrics_every > 0
        metric_logs = []

        # Auxiliary split-batch momenta for D1(m) probe (V2). These are read-only
        # diagnostics; optimizer keeps using its own buffer fed by g_full. We
        # mirror nanochat's scheduled momentum and probe the Nesterov-corrected
        # direction M_tilde = (1-mom) * g + mom * m_new, which is the actual
        # matrix passed to Muon's polar step.
        do_momentum_d1 = grad_accum >= 2
        split_reduce_names = set()
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
        if metrics_enabled:
            if ddp and args.metrics_split_momentum and do_momentum_d1:
                module_pattern = re.compile(args.metrics_module_regex) if args.metrics_module_regex else None
                split_reduce = [
                    n for n in matrix_params_named
                    if module_pattern is None or module_pattern.search(n) is not None
                ]
                from metric_logging import _is_normal_matrix_weight, _limit_modules
                split_reduce = [n for n in split_reduce if _is_normal_matrix_weight(n)]
                split_reduce = _limit_modules(split_reduce, args.metrics_max_modules)
                if args.metrics_max_modules > 0:
                    split_reduce = split_reduce[:args.metrics_max_modules]
                split_reduce_names = set(split_reduce)
            split_msg = (
                "enabled" if args.metrics_split_momentum and do_momentum_d1
                else "disabled by flag" if not args.metrics_split_momentum
                else f"disabled because grad_accum={grad_accum} < 2"
            )
            print0(f"Metric logging enabled every {args.metrics_every} steps; "
                   f"hessian_every={args.metrics_hessian_every}; split momentum {split_msg}; "
                   f"split scope={'global DDP-averaged' if split_reduce_names else 'rank-local or single-process'}.")

        val_bpbs = []; best_val_bpb = float('inf')
        for step in range(num_iterations + 1):
            last = step == num_iterations
            if args.eval_every > 0 and (last or step % args.eval_every == 0):
                model.eval()
                vl = build_val()
                eval_steps = max(1, args.eval_tokens // world_tokens_per_fwdbwd)
                vb = evaluate_bpb(model, vl, eval_steps, token_bytes)
                val_bpbs.append({"step": step, "val_bpb": vb})
                if vb < best_val_bpb: best_val_bpb = vb
                print0(f"  Step {step:5d} | val_bpb: {vb:.6f} (best: {best_val_bpb:.6f})")
                model.train()

            # mid-training D1 probe (5 uniformly-spaced points)
            if should_probe(step, num_iterations, n_probes=5) and not last:
                probe_loader = build_val()  # fresh loader, disjoint from train
                # micros_per_half: default 4 (cheap noisy probe). Bump to ~32 to
                # approximate momentum (β=0.95 → effective horizon ~20 steps).
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
                    print0(f"  Step {step:5d} | D1(M_tilde) probe: "
                           f"frac>0.7={m_d1_stats['frac_above_0_7_median']:.3f}  "
                           f"frac>c*={m_d1_stats['frac_above_c_star_median']:.3f}  "
                           f"[buffer c*={mb_d1_stats['frac_above_c_star_median']:.3f}]")

                tel_spectra.append({"step": step, **spectral_stats(model)})

            if last: break

            lrm = get_lr_mul(step); mom = get_mom(step); wd = get_wd(step)
            t0 = time.time()
            metrics_due = metrics_enabled and (step % args.metrics_every == 0)
            hessian_due = (
                metrics_due
                and args.metrics_hessian_every > 0
                and step % args.metrics_hessian_every == 0
            )
            half = grad_accum // 2 if do_momentum_d1 else 0
            g_A_partial = {}
            hessian_batch = None
            train_loss_sum = 0.0
            for i in range(grad_accum):
                if hessian_due and hessian_batch is None:
                    hessian_batch = (x.detach().clone(), y.detach().clone())
                loss = model(x, y)
                train_loss_sum += loss.detach().item()
                (loss / grad_accum).backward()
                x, y = next(train_loader)
                if do_momentum_d1 and (i + 1) == half:
                    # .grad accumulated `half` micros, each scaled by 1/grad_accum.
                    # mean of first half = (grad_accum/half) * .grad
                    scale = grad_accum / half
                    for n, p in matrix_params_named.items():
                        if p.grad is not None:
                            g_A_partial[n] = (scale * p.grad.detach().to(torch.float32)).clone()

            # EMA-update auxiliary momenta from split-batch grads
            if do_momentum_d1:
                rest = grad_accum - half
                for n, p in matrix_params_named.items():
                    if n in g_A_partial and p.grad is not None:
                        g_full = p.grad.detach().to(torch.float32)
                        g_A = g_A_partial[n]
                        # g_full = (half/grad_accum)*g_A + (rest/grad_accum)*g_B
                        # → g_B = (grad_accum*g_full - half*g_A) / rest
                        g_B = (grad_accum * g_full - half * g_A) / rest
                        if n in split_reduce_names:
                            torch.distributed.all_reduce(g_A, op=torch.distributed.ReduceOp.AVG)
                            torch.distributed.all_reduce(g_B, op=torch.distributed.ReduceOp.AVG)
                        m_A[n].mul_(mom).add_(g_A, alpha=1.0 - mom)
                        m_B[n].mul_(mom).add_(g_B, alpha=1.0 - mom)
                        mt_A[n].copy_(g_A).mul_(1.0 - mom).add_(m_A[n], alpha=mom)
                        mt_B[n].copy_(g_B).mul_(1.0 - mom).add_(m_B[n], alpha=mom)
                g_A_partial.clear()
            train_loss_mean = train_loss_sum / grad_accum

            # capture grad norms before step
            gn = grad_norm_per_group(optimizer)

            for g in optimizer.param_groups:
                g['lr'] = g['initial_lr'] * lrm
                g['_capture_metrics'] = False
                if g['kind'] == 'muon':
                    g['momentum'] = mom
                    g['weight_decay'] = wd
                    g['_capture_metrics'] = metrics_due
            optimizer.step()
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
                    muon_momentum=mom,
                    save_components=args.metrics_save_components,
                    component_dir=component_dir,
                    collect_hessian_refs=hessian_due,
                    exact_svd=args.metrics_exact_svd,
                    compact=not args.metrics_full_per_module,
                )
                clear_metric_caches(optimizer)

                if rank == 0 and metric_entry is not None:
                    metric_entry.setdefault("metadata", {})
                    metric_entry["metadata"].update({
                        "train_loss": "raw mean cross-entropy over this optimizer step's grad-accum microbatches",
                        "hessian_weight_state": "post_optimizer_step",
                        "hessian_batch_source": "first_microbatch_same_optimizer_step" if hessian_batch is not None else None,
                        "hessian_batch_scope": "rank0_local_microbatch" if ddp and hessian_batch is not None else "single_process",
                        "split_momentum_alignment_source": "momentum_after_nesterov",
                        "split_momentum_alignment_scope": "global_ddp_average" if split_reduce_names else "rank_local_or_single_process",
                    })
                    split_svd_due = (
                        args.metrics_split_momentum
                        and do_momentum_d1
                        and mt_A is not None
                        and mt_B is not None
                        and (
                            args.metrics_split_svd_every <= 0
                            or step % args.metrics_split_svd_every == 0
                        )
                    )
                    if split_svd_due:
                        split_stats = split_momentum_alignment(
                            mt_A,
                            mt_B,
                            top_k=args.metrics_top_k,
                            alignment_side=args.metrics_alignment_side,
                            module_regex=args.metrics_module_regex,
                            max_modules=args.metrics_max_modules,
                        )
                        split_stats["tensor"] = "momentum_after_nesterov"
                        split_stats["definition"] = "M_tilde = (1 - beta) * G_split + beta * M_split_new"
                        split_stats["svd_every_steps"] = args.metrics_split_svd_every or args.metrics_every
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
            if step == 0: gc.collect()

            # record telemetry each step
            tel_loss.append(train_loss_mean)
            tel_grad_norms.append(gn)
            tel_lr_mul.append(lrm)
            tel_mom.append(mom)

            if step % 50 == 0:
                print0(f"  Step {step:5d} | loss: {train_loss_mean:.4f} | dt: {(time.time()-t0)*1000:.0f}ms")

        results.update(dict(
            score=best_val_bpb, val_bpb_best=best_val_bpb, val_bpbs=val_bpbs, error=None,
            telemetry=dict(
                train_loss_per_step=tel_loss,
                grad_norms_per_step=tel_grad_norms,
                lr_mul_per_step=tel_lr_mul,
                momentum_per_step=tel_mom,
                d1_probes=tel_d1_probes,
                d1_probes_momentum=tel_d1_probes_momentum,
                d1_probes_momentum_left=tel_d1_probes_momentum_left,
                d1_probes_momentum_buffer=tel_d1_probes_momentum_buffer,
                spectra=tel_spectra,
            ),
            metric_logs=metric_logs,
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
