#!/usr/bin/env python3
"""Sweep optimizer baselines and Top-Aware Muon alpha variants.

This module keeps the historical filename for compatibility. Prefer invoking
``run_optimizer_sweep.py`` in new scripts/docs. The engine intentionally
supports only the main optimizer baselines used for current experiments:

    streaming_identity: StreamingMuon + candidates/identity.py
    top_aware_muon: StreamingMuon + candidates/top_aware_muon.py
    plain_muon, muon/native_muon, adamw, soap, shampoo, kl_shampoo, kl_soap

The default baseline remains Top-Aware Muon with alpha=1.0/0.5. Non-streaming
baselines ignore --alphas and --top-ks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
import time
from json import JSONDecodeError
from pathlib import Path
from typing import Any


SEQ = 1024
METHOD_CHOICES = {
    "streaming_identity",
    "top_aware_muon",
    "plain_muon",
    "muon",
    "native_muon",
    "adamw",
    "soap",
    "shampoo",
    "kl_shampoo",
    "kl_soap",
}
RUN_EVAL_OPTIMIZER = {
    "streaming_identity": "streaming_muon",
    "top_aware_muon": "streaming_muon",
    "plain_muon": "plain_muon",
    "muon": "muon",
    "native_muon": "native_muon",
    "adamw": "adamw",
    "soap": "soap",
    "shampoo": "shampoo",
    "kl_shampoo": "kl_shampoo",
    "kl_soap": "kl_soap",
}
STRUCTURED_REFERENCE_CONFIGS = {
    "soap": {
        "precondition_frequency": 1,
        "shampoo_beta": 0.95,
        "optimizer_beta1": 0.95,
        "optimizer_beta2": 0.99,
        "structured_init_factor": 1.0,
    },
    "shampoo": {
        "precondition_frequency": 10,
        "shampoo_beta": 0.98,
        "optimizer_beta1": 0.90,
        "optimizer_beta2": 0.98,
        "structured_init_factor": 0.1,
    },
    "kl_soap": {
        "precondition_frequency": 1,
        "shampoo_beta": 0.90,
        "optimizer_beta1": 0.95,
        "optimizer_beta2": 0.95,
        "structured_init_factor": 0.1,
    },
    "kl_shampoo": {
        "precondition_frequency": 1,
        "shampoo_beta": 0.98,
        "optimizer_beta1": 0.90,
        "optimizer_beta2": 0.98,
        "structured_init_factor": 0.1,
    },
}
CHINCHILLA_1X_TOKENS_BY_DEPTH = {
    # 20x non-embedding/token-budget convention used by this handoff repo.
    # Values are rounded down where needed to stay compatible with the main
    # batch grid.
    8: 402_653_184,
    12: 1_698_693_120,
    16: 4_026_531_840,
}


def parse_list_int(spec: str) -> list[int]:
    return [int(x) for x in spec.replace(",", " ").split() if x]


def parse_list_float(spec: str) -> list[float]:
    return [float(x) for x in spec.replace(",", " ").split() if x]


def parse_list_str(spec: str) -> list[str]:
    return [x.strip() for x in spec.replace(",", " ").split() if x.strip()]


def slug_float(x: float) -> str:
    return f"{x:g}".replace("-", "m").replace(".", "p")


def steps_for_batch(tokens: int, batch: int) -> int:
    steps = tokens // batch
    if steps < 1:
        raise ValueError(f"tokens={tokens} is smaller than batch={batch}")
    return steps


def tokens_from_chinchilla(depth: int, mult: float) -> int:
    if depth not in CHINCHILLA_1X_TOKENS_BY_DEPTH:
        supported = ", ".join(str(k) for k in sorted(CHINCHILLA_1X_TOKENS_BY_DEPTH))
        raise ValueError(
            f"no hard-coded Chinchilla token budget for depth={depth}; "
            f"supported depths: {supported}. Pass --tokens for a custom run."
        )
    if not math.isfinite(float(mult)) or mult <= 0:
        raise ValueError("--chinchilla-mult must be finite and > 0")
    return max(1, int(math.floor(CHINCHILLA_1X_TOKENS_BY_DEPTH[depth] * mult)))


def device_batch_for(batch: int, nproc: int, max_device_batch_size: int) -> int:
    max_without_accum = batch // (SEQ * nproc)
    if max_without_accum < 1:
        raise ValueError(f"batch={batch} is smaller than seq*nproc={SEQ*nproc}")
    return min(max_device_batch_size, max_without_accum)


def eval_every_for_steps(steps: int) -> int:
    return max(1, steps // 8)


def load_score(path: Path) -> tuple[float | None, str | None]:
    if not path.exists():
        return None, "missing"
    try:
        data = json.loads(path.read_text())
    except JSONDecodeError as exc:
        return None, f"invalid_json:{exc.msg}"
    err = data.get("error")
    score = data.get("score")
    if err is not None or score is None:
        return None, err or "score_missing"
    return float(score), None


def case_name(method: str, batch: int, lr: float, seed: int,
              top_k: int | None, alpha: float | None) -> str:
    if method == "top_aware_muon":
        assert top_k is not None and alpha is not None
        return f"top_aware_k{top_k}_a{slug_float(alpha)}_bsz{batch}_lr{slug_float(lr)}_s{seed}"
    return f"{method}_bsz{batch}_lr{slug_float(lr)}_s{seed}"


def result_path(args: argparse.Namespace, method: str, batch: int, lr: float,
                seed: int, top_k: int | None = None, alpha: float | None = None) -> Path:
    return args.out_root / case_name(method, batch, lr, seed, top_k, alpha) / "result.json"


def note_case(args: argparse.Namespace, spec: tuple[str, int, float, int, int | None, float | None]) -> None:
    if not hasattr(args, "_case_specs"):
        args._case_specs = []
        args._case_spec_set = set()
    if spec not in args._case_spec_set:
        args._case_specs.append(spec)
        args._case_spec_set.add(spec)


def group_key(method: str, batch: int, seed: int,
              top_k: int | None, alpha: float | None) -> tuple[Any, ...]:
    return (method, int(batch), int(seed), top_k, alpha)


def common_training_args(args: argparse.Namespace, batch: int, lr: float, seed: int, out: Path) -> list[str]:
    steps = steps_for_batch(args.tokens, batch)
    device_batch = device_batch_for(batch, args.nproc_per_node, args.max_device_batch_size)
    world_tokens = device_batch * SEQ * args.nproc_per_node
    if batch % world_tokens != 0:
        raise ValueError(
            f"batch={batch} must be divisible by device_batch*seq*nproc={world_tokens}; "
            "lower --max-device-batch-size or change the batch grid"
        )

    warmup = max(1, round(args.warmup_ratio * steps))
    eval_every = args.eval_every if args.eval_every > 0 else eval_every_for_steps(steps)
    return [
        "--nanochat-dir", args.nanochat_dir,
        "--output-file", str(out),
        "--depth", str(args.depth),
        "--max-seq-len", str(SEQ),
        "--device-batch-size", str(device_batch),
        "--total-batch-size", str(batch),
        "--max-steps", str(steps),
        "--matrix-lr", f"{lr:g}",
        "--weight-decay", f"{args.weight_decay:g}",
        "--warmup-steps", str(warmup),
        "--warmdown-ratio", f"{args.warmdown_ratio:g}",
        "--final-lr-frac", f"{args.final_lr_frac:g}",
        "--eval-every", str(eval_every),
        "--eval-tokens", str(args.eval_tokens),
        "--seed", str(seed),
        "--architecture", args.architecture,
    ]


def checkpoint_args(args: argparse.Namespace, out: Path) -> list[str]:
    if args.save_every <= 0:
        return []
    return [
        "--checkpoint-dir", str(out.parent / "checkpoints"),
        "--save-every", str(args.save_every),
        "--keep-last-checkpoints", str(args.keep_last_checkpoints),
        "--resume-from-step", str(args.resume_from_step),
        "--resume" if args.resume else "--no-resume",
    ]


def metrics_args(args: argparse.Namespace) -> list[str]:
    if args.metrics_every <= 0:
        return []
    out = [
        "--metrics-every", str(args.metrics_every),
        "--metrics-top-k", str(args.metrics_top_k),
        "--metrics-module-regex", args.metrics_module_regex,
        "--metrics-max-modules", str(args.metrics_max_modules),
    ]
    if args.metrics_hessian_every > 0:
        out += [
            "--metrics-hessian-every", str(args.metrics_hessian_every),
            "--metrics-hessian-top-k", str(args.metrics_hessian_top_k),
            "--metrics-hessian-iters", str(args.metrics_hessian_iters),
            "--metrics-hessian-max-modules", str(args.metrics_hessian_max_modules),
            "--metrics-projection-correlation-window", str(args.metrics_projection_correlation_window),
        ]
    return out


def structured_config(args: argparse.Namespace, method: str) -> dict[str, float | int]:
    if args.structured_config == "auto" and method in STRUCTURED_REFERENCE_CONFIGS:
        return STRUCTURED_REFERENCE_CONFIGS[method]
    return {
        "precondition_frequency": args.precondition_frequency,
        "shampoo_beta": args.shampoo_beta,
        "optimizer_beta1": args.optimizer_beta1,
        "optimizer_beta2": args.optimizer_beta2,
        "structured_init_factor": args.structured_init_factor,
    }


def build_command(args: argparse.Namespace, method: str, batch: int, lr: float, seed: int,
                  out: Path, top_k: int | None, alpha: float | None) -> list[str]:
    candidate = {
        "streaming_identity": "candidates/identity.py",
        "top_aware_muon": "candidates/top_aware_muon.py",
    }.get(method)
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={args.nproc_per_node}", "run_eval.py"]
    cmd += ["--optimizer", RUN_EVAL_OPTIMIZER[method]]
    if candidate is not None:
        cmd += ["--candidate-file", candidate]
    if method == "top_aware_muon":
        assert top_k is not None and alpha is not None
        cmd += ["--candidate-param", f"top_k={top_k}", "--candidate-param", f"alpha={alpha:g}"]
    cmd += common_training_args(args, batch, lr, seed, out)
    cmd += [
        "--k", str(args.streaming_rank_k),
        "--num-iters", str(args.streaming_num_iters),
        "--fallback-ortho-tol", f"{args.fallback_ortho_tol:g}",
        "--matrix-lr-adjust", args.matrix_lr_adjust,
        "--adam-lr-mode", args.adam_lr_mode,
        "--muon-momentum", f"{args.muon_momentum:g}",
        "--muon-momentum-schedule", args.muon_momentum_schedule,
    ]
    cmd.append("--batch-beta-align" if args.batch_beta_align else "--no-batch-beta-align")
    cmd += ["--batch-beta-align-mode", args.batch_beta_align_mode]
    cfg = structured_config(args, method)
    cmd += [
        "--precondition-frequency", str(cfg["precondition_frequency"]),
        "--shampoo-beta", f"{cfg['shampoo_beta']:g}",
        "--optimizer-beta1", f"{cfg['optimizer_beta1']:g}",
        "--optimizer-beta2", f"{cfg['optimizer_beta2']:g}",
        "--structured-init-factor", f"{cfg['structured_init_factor']:g}",
    ]
    if args.pure_qr:
        cmd.append("--pure-qr")
    if args.structured_use_qr:
        cmd.append("--structured-use-qr")
    else:
        cmd.append("--no-structured-use-qr")
    cmd += checkpoint_args(args, out)
    cmd += metrics_args(args)
    return cmd


def iter_case_specs(args: argparse.Namespace, methods: list[str]):
    for batch in args.batches:
        for seed in args.seeds:
            if "streaming_identity" in methods:
                for lr in args.lrs:
                    yield "streaming_identity", batch, lr, seed, None, None
            for method in methods:
                if method in {"streaming_identity", "top_aware_muon"}:
                    continue
                for lr in args.lrs:
                    yield method, batch, lr, seed, None, None
            if "top_aware_muon" in methods:
                for top_k in args.top_ks:
                    for alpha in args.alphas:
                        for lr in args.lrs:
                            yield "top_aware_muon", batch, lr, seed, top_k, alpha


def append_summary(args: argparse.Namespace, row: dict[str, Any]) -> None:
    if not getattr(args, "append_summary", True):
        return
    path = args.out_root / "sweep_rows.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "time", "method", "batch", "lr", "seed", "top_k", "alpha",
        "score", "error", "result_json", "log_file",
    ]
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})


def run_case(args: argparse.Namespace, method: str, batch: int, lr: float, seed: int,
             top_k: int | None = None, alpha: float | None = None) -> tuple[float | None, str | None]:
    spec = (method, int(batch), float(lr), int(seed), top_k, alpha)
    note_case(args, spec)

    out = result_path(args, method, batch, lr, seed, top_k, alpha)
    name = out.parent.name
    log = args.log_root / f"{name}.log"
    out.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)

    if out.exists() and not args.rerun_existing:
        score, err = load_score(out)
        if score is not None and err is None:
            print(f"[skip] {name}: score={score:.6f}", flush=True)
            return score, None
        print(f"[rerun invalid] {name}: score={score} error={err}", flush=True)

    cmd = build_command(args, method, batch, lr, seed, out, top_k, alpha)
    print(f"\n=== {name} ===", flush=True)
    print(" ".join(shlex.quote(x) for x in cmd), flush=True)
    if args.dry_run:
        return None, "dry_run"

    with log.open("w") as f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            f.write(line)
            f.flush()
        rc = proc.wait()
    if rc != 0:
        print(f"[warn] {name}: command exited rc={rc}; checking result JSON", flush=True)

    score, err = load_score(out)
    append_summary(args, {
        "time": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "method": method,
        "batch": batch,
        "lr": lr,
        "seed": seed,
        "top_k": "" if top_k is None else top_k,
        "alpha": "" if alpha is None else alpha,
        "score": "" if score is None else score,
        "error": "" if err is None else err,
        "result_json": out,
        "log_file": log,
    })
    print(f"[done] {name}: score={score} error={err}", flush=True)
    return score, err


def write_manifest(args: argparse.Namespace, methods: list[str]) -> None:
    args.out_root.mkdir(parents=True, exist_ok=True)
    signature = sweep_signature(args, methods)
    manifest_path = args.out_root / "manifest.json"
    if manifest_path.exists() and not args.allow_config_mismatch:
        try:
            previous = json.loads(manifest_path.read_text())
        except JSONDecodeError:
            previous = {}
        previous_signature = previous.get("config_signature")
        if (
            previous_signature is not None
            and comparable_sweep_signature(previous_signature)
            != comparable_sweep_signature(signature)
        ):
            raise ValueError(
                f"{manifest_path} already exists with a different sweep configuration. "
                "Use a new STAMP/OUT_ROOT for a changed recipe, or pass "
                "--allow-config-mismatch if you intentionally want to mix configs."
            )
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "out_root": str(args.out_root),
        "log_root": str(args.log_root),
        "config_signature": signature,
        "methods": methods,
        "batches": args.batches,
        "lrs": args.lrs,
        "top_ks": args.top_ks,
        "alphas": args.alphas,
        "seeds": args.seeds,
        "tokens": args.tokens,
        "token_budget_source": args.token_budget_source,
        "chinchilla_mult": args.chinchilla_mult,
        "chinchilla_1x_tokens": args.chinchilla_1x_tokens,
        "depth": args.depth,
        "architecture": args.architecture,
        "seq": SEQ,
        "nproc_per_node": args.nproc_per_node,
        "max_device_batch_size": args.max_device_batch_size,
        "streaming": {
            "pure_qr": args.pure_qr,
            "num_iters": args.streaming_num_iters,
            "fallback_ortho_tol": args.fallback_ortho_tol,
            "rank_k": args.streaming_rank_k,
        },
        "structured_optimizers": {
            "config": args.structured_config,
            "auto_reference_configs": STRUCTURED_REFERENCE_CONFIGS if args.structured_config == "auto" else {},
            "precondition_frequency": args.precondition_frequency,
            "shampoo_beta": args.shampoo_beta,
            "optimizer_beta1": args.optimizer_beta1,
            "optimizer_beta2": args.optimizer_beta2,
            "init_factor": args.structured_init_factor,
            "use_qr": args.structured_use_qr,
        },
        "checkpointing": {
            "enabled": args.save_every > 0,
            "save_every": args.save_every,
            "keep_last_checkpoints": args.keep_last_checkpoints,
            "resume": args.resume,
            "resume_from_step": args.resume_from_step,
            "layout": "each case writes checkpoints under <case_dir>/checkpoints",
        },
        "metrics": {
            "metrics_every": args.metrics_every,
            "top_k": args.metrics_top_k,
            "module_regex": args.metrics_module_regex,
            "max_modules": args.metrics_max_modules,
            "hessian_every": args.metrics_hessian_every,
            "hessian_top_k": args.metrics_hessian_top_k,
            "hessian_iters": args.metrics_hessian_iters,
            "hessian_max_modules": args.metrics_hessian_max_modules,
            "projection_correlation_window": args.metrics_projection_correlation_window,
        },
        "adaptive_lr": {
            "enabled": args.adaptive_lr,
            "extend_factor": args.lr_extend_factor,
            "min_lr": args.lr_min,
            "max_lr": args.lr_max,
            "max_extension_rounds": args.max_lr_extension_rounds,
            "min_edge_improvement": args.adaptive_min_edge_improvement,
        },
        "top_aware_muon_definition": (
            "For each matrix, scale the top_k largest current singular directions by alpha; "
            "all remaining singular directions use scale 1."
        ),
        "method_definitions": {
            "streaming_identity": "StreamingMuon with f(sigma)=1",
            "top_aware_muon": "StreamingMuon with top-k sigma-direction scale alpha",
            "plain_muon": "ordinary Muon: NS5 matrix-sign orthogonalization of Nesterov momentum, no dimension LR normalization",
            "muon": "nanochat native Muon/NormMuon baseline",
            "native_muon": "nanochat native Muon baseline",
            "adamw": "AdamW on matrix weights plus AdamW on embeddings/scalars",
            "soap": "SOAP/RMSProp in Shampoo eigenbasis",
            "shampoo": "Two-sided Shampoo with -1/4 factor powers",
            "kl_shampoo": "KL-Shampoo-style two-sided factor update and eigenvalue EMA correction",
            "kl_soap": "KL-Shampoo basis update with SOAP/RMSProp augmented diagonal",
        },
    }
    tmp_path = manifest_path.with_name(f"{manifest_path.name}.tmp.{os.getpid()}")
    tmp_path.write_text(json.dumps(manifest, indent=2))
    tmp_path.replace(manifest_path)


def sweep_signature(args: argparse.Namespace, methods: list[str]) -> dict[str, Any]:
    """Configuration fields that must stay fixed inside one OUT_ROOT."""
    return {
        "methods": methods,
        "batches": args.batches,
        "lrs": args.lrs,
        "top_ks": args.top_ks,
        "alphas": args.alphas,
        "seeds": args.seeds,
        "tokens": args.tokens,
        "depth": args.depth,
        "architecture": args.architecture,
        "seq": SEQ,
        "nproc_per_node": args.nproc_per_node,
        "max_device_batch_size": args.max_device_batch_size,
        "eval_tokens": args.eval_tokens,
        "eval_every": args.eval_every,
        "warmup_ratio": args.warmup_ratio,
        "warmdown_ratio": args.warmdown_ratio,
        "final_lr_frac": args.final_lr_frac,
        "muon_momentum": args.muon_momentum,
        "muon_momentum_schedule": args.muon_momentum_schedule,
        "streaming_num_iters": args.streaming_num_iters,
        "streaming_rank_k": args.streaming_rank_k,
        "fallback_ortho_tol": args.fallback_ortho_tol,
        "pure_qr": args.pure_qr,
        "matrix_lr_adjust": args.matrix_lr_adjust,
        "batch_beta_align": args.batch_beta_align,
        "batch_beta_align_mode": args.batch_beta_align_mode,
        "adam_lr_mode": args.adam_lr_mode,
        "structured_config": args.structured_config,
        "precondition_frequency": args.precondition_frequency,
        "shampoo_beta": args.shampoo_beta,
        "optimizer_beta1": args.optimizer_beta1,
        "optimizer_beta2": args.optimizer_beta2,
        "structured_init_factor": args.structured_init_factor,
        "structured_use_qr": args.structured_use_qr,
        "metrics_every": args.metrics_every,
        "metrics_top_k": args.metrics_top_k,
        "metrics_module_regex": args.metrics_module_regex,
        "metrics_max_modules": args.metrics_max_modules,
        "metrics_hessian_every": args.metrics_hessian_every,
        "metrics_hessian_top_k": args.metrics_hessian_top_k,
        "metrics_hessian_iters": args.metrics_hessian_iters,
        "metrics_hessian_max_modules": args.metrics_hessian_max_modules,
        "metrics_projection_correlation_window": args.metrics_projection_correlation_window,
        "adaptive_lr": args.adaptive_lr,
        "lr_extend_factor": args.lr_extend_factor,
        "lr_min": args.lr_min,
        "lr_max": args.lr_max,
        "max_lr_extension_rounds": args.max_lr_extension_rounds,
        "adaptive_min_edge_improvement": args.adaptive_min_edge_improvement,
    }


def comparable_sweep_signature(signature: dict[str, Any]) -> dict[str, Any]:
    """Fields that define the base grid and training recipe.

    Adaptive-LR closure knobs intentionally do not participate in compatibility:
    a fixed-grid SLURM array can write the base manifest, then a later local
    command can collate summaries or run boundary closures under the same STAMP.
    """
    ignored = {
        "adaptive_lr",
        "lr_extend_factor",
        "lr_min",
        "lr_max",
        "max_lr_extension_rounds",
        "adaptive_min_edge_improvement",
    }
    return {k: v for k, v in signature.items() if k not in ignored}


def write_summary_csv(args: argparse.Namespace, methods: list[str]) -> Path:
    path = args.out_root / "sweep_rows.csv"
    fields = [
        "time", "method", "batch", "lr", "seed", "top_k", "alpha",
        "score", "error", "result_json", "log_file",
    ]
    case_specs = getattr(args, "_case_specs", None)
    if case_specs is None:
        case_specs = list(iter_case_specs(args, methods))

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for method, batch, lr, seed, top_k, alpha in case_specs:
            out = result_path(args, method, batch, lr, seed, top_k, alpha)
            log = args.log_root / f"{out.parent.name}.log"
            score, err = load_score(out)
            completed_at = (
                time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(out.stat().st_mtime))
                if out.exists() else ""
            )
            writer.writerow({
                "time": completed_at,
                "method": method,
                "batch": batch,
                "lr": lr,
                "seed": seed,
                "top_k": "" if top_k is None else top_k,
                "alpha": "" if alpha is None else alpha,
                "score": "" if score is None else score,
                "error": "" if err is None else err,
                "result_json": out,
                "log_file": log,
            })
    return path


def edge_is_strong_enough(best_score: float, neighbor_score: float | None, min_improvement: float) -> bool:
    if neighbor_score is None:
        return True
    return (neighbor_score - best_score) >= min_improvement


def maybe_propose_lr_extensions(
    args: argparse.Namespace,
    methods: list[str],
    blocked_edges: set[tuple[tuple[Any, ...], str]],
) -> list[tuple[str, int, float, int, int | None, float | None, str]]:
    grouped: dict[tuple[Any, ...], list[float]] = {}
    case_specs = getattr(args, "_case_specs", list(iter_case_specs(args, methods)))
    for method, batch, lr, seed, top_k, alpha in case_specs:
        grouped.setdefault(group_key(method, batch, seed, top_k, alpha), []).append(float(lr))

    proposals: list[tuple[str, int, float, int, int | None, float | None, str]] = []
    seen = set(getattr(args, "_case_spec_set", set()))
    for key, lrs in sorted(grouped.items(), key=lambda kv: repr(kv[0])):
        method, batch, seed, top_k, alpha = key
        grid_lrs = sorted(set(lrs))
        scored = [
            (lr, *load_score(result_path(args, method, batch, lr, seed, top_k, alpha)))
            for lr in grid_lrs
        ]
        finite = sorted((lr, score) for lr, score, err in scored if score is not None and err is None)
        if len(finite) < 2:
            continue

        best_lr, best_score = min(finite, key=lambda item: item[1])
        finite_lrs = [lr for lr, _ in finite]
        finite_scores = dict(finite)

        low_lr = finite_lrs[0]
        if best_lr == low_lr and (key, "low") not in blocked_edges:
            if any(lr < low_lr for lr in grid_lrs):
                blocked_edges.add((key, "low"))
            elif edge_is_strong_enough(best_score, finite_scores.get(finite_lrs[1]), args.adaptive_min_edge_improvement):
                new_lr = low_lr / args.lr_extend_factor
                spec = (method, batch, new_lr, seed, top_k, alpha)
                if new_lr >= args.lr_min and spec not in seen:
                    proposals.append((*spec, "low"))
                    seen.add(spec)
                else:
                    blocked_edges.add((key, "low"))

        high_lr = finite_lrs[-1]
        if best_lr == high_lr and (key, "high") not in blocked_edges:
            if any(lr > high_lr for lr in grid_lrs):
                blocked_edges.add((key, "high"))
            elif edge_is_strong_enough(best_score, finite_scores.get(finite_lrs[-2]), args.adaptive_min_edge_improvement):
                new_lr = high_lr * args.lr_extend_factor
                spec = (method, batch, new_lr, seed, top_k, alpha)
                if new_lr <= args.lr_max and spec not in seen:
                    proposals.append((*spec, "high"))
                    seen.add(spec)
                else:
                    blocked_edges.add((key, "high"))
    return proposals


def run_adaptive_lr_rounds(args: argparse.Namespace, methods: list[str]) -> None:
    blocked_edges: set[tuple[tuple[Any, ...], str]] = set()
    trace: list[dict[str, Any]] = []
    for round_idx in range(1, args.max_lr_extension_rounds + 1):
        proposals = maybe_propose_lr_extensions(args, methods, blocked_edges)
        trace.append({
            "round": round_idx,
            "num_proposals": len(proposals),
            "proposals": [
                {
                    "method": method,
                    "batch": batch,
                    "lr": lr,
                    "seed": seed,
                    "top_k": top_k,
                    "alpha": alpha,
                    "edge": edge,
                }
                for method, batch, lr, seed, top_k, alpha, edge in proposals
            ],
        })
        (args.out_root / "adaptive_lr_trace.json").write_text(json.dumps(trace, indent=2))
        if not proposals:
            print(f"[adaptive-lr] round {round_idx}: no boundary extensions needed", flush=True)
            break
        print(f"[adaptive-lr] round {round_idx}: running {len(proposals)} boundary extensions", flush=True)
        for method, batch, lr, seed, top_k, alpha, edge in proposals:
            print(
                f"[adaptive-lr] {edge} edge -> method={method} batch={batch} "
                f"alpha={alpha} top_k={top_k} lr={lr:g}",
                flush=True,
            )
            score, err = run_case(args, method, batch, lr, seed, top_k=top_k, alpha=alpha)
            if score is None or err is not None:
                blocked_edges.add((group_key(method, batch, seed, top_k, alpha), edge))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--out-root", type=Path, default=Path(f"search_evals/optimizer_sweep_{stamp}"))
    parser.add_argument("--log-root", type=Path, default=Path(f"logs/optimizer_sweep_{stamp}"))
    parser.add_argument("--nanochat-dir", type=str, default="nanochat")
    parser.add_argument("--methods", type=parse_list_str,
                        default=parse_list_str("top_aware_muon"),
                        help=f"Space/comma separated subset of: {' '.join(sorted(METHOD_CHOICES))}")
    parser.add_argument("--batches", type=parse_list_int, default=parse_list_int("131072"))
    parser.add_argument("--lrs", type=parse_list_float, default=parse_list_float("0.005 0.0075 0.01 0.015 0.02 0.03 0.04"))
    parser.add_argument("--top-ks", type=parse_list_int, default=parse_list_int("1"))
    parser.add_argument("--alphas", type=parse_list_float, default=parse_list_float("1.0 0.5"))
    parser.add_argument("--seeds", type=parse_list_int, default=parse_list_int("42"))
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--architecture", type=str, default="gpt2", choices=["gpt2", "nanochat", "qwen3"])
    parser.add_argument("--tokens", type=int, default=None,
                        help="Exact token budget override. If omitted, use --chinchilla-mult and --depth.")
    parser.add_argument("--chinchilla-mult", type=float, default=2.0,
                        help="Multiplier on the hard-coded 1x Chinchilla token table for this depth.")
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--max-device-batch-size", type=int, default=16)
    parser.add_argument("--eval-tokens", type=int, default=524288)
    parser.add_argument("--eval-every", type=int, default=0, help="0 means steps//8")
    parser.add_argument("--weight-decay", type=float, default=0.28)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--warmdown-ratio", type=float, default=0.65)
    parser.add_argument("--final-lr-frac", type=float, default=0.05)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-momentum-schedule", choices=["nanochat", "static"], default="nanochat")

    parser.add_argument("--save-every", type=int, default=0,
                        help="If >0, enable per-case checkpoints every N optimizer steps.")
    parser.add_argument("--keep-last-checkpoints", type=int, default=2)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-from-step", type=int, default=-1)

    parser.add_argument("--streaming-num-iters", type=int, default=2)
    parser.add_argument("--streaming-rank-k", type=int, default=-1)
    parser.add_argument("--fallback-ortho-tol", type=float, default=0.01)
    parser.add_argument("--pure-qr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--matrix-lr-adjust", choices=["none", "moonlight"], default="moonlight")
    parser.add_argument("--batch-beta-align", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--batch-beta-align-mode",
        choices=["all", "beta2_only", "none"],
        default="all",
        help="When batch-beta-align is on: all scales beta1+beta2; beta2_only keeps beta1 constant.",
    )
    parser.add_argument(
        "--adam-lr-mode",
        choices=["relative_to_matrix", "nanochat_fixed"],
        default="relative_to_matrix",
    )
    parser.add_argument(
        "--structured-config",
        choices=["auto", "global"],
        default="auto",
        help=(
            "auto uses method-specific reference configs for SOAP/Shampoo/KL variants; "
            "global uses the explicit --precondition-frequency/--shampoo-beta/"
            "--optimizer-beta*/--structured-init-factor values for every method."
        ),
    )
    parser.add_argument("--precondition-frequency", type=int, default=5)
    parser.add_argument("--shampoo-beta", type=float, default=0.95)
    parser.add_argument("--optimizer-beta1", type=float, default=0.9)
    parser.add_argument("--optimizer-beta2", type=float, default=0.95)
    parser.add_argument("--structured-init-factor", type=float, default=1.0)
    parser.add_argument("--structured-use-qr", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--metrics-every", type=int, default=0)
    parser.add_argument("--metrics-top-k", type=int, default=4)
    parser.add_argument(
        "--metrics-module-regex",
        type=str,
        default=r"transformer\.h\.(?:[0-9]+)\.(?:attn\.(?:c_q|c_k|c_v|c_proj)|mlp\.(?:c_gate|c_fc|c_proj))\.weight$",
    )
    parser.add_argument("--metrics-max-modules", type=int, default=0)
    parser.add_argument("--metrics-hessian-every", type=int, default=0)
    parser.add_argument("--metrics-hessian-top-k", type=int, default=1)
    parser.add_argument("--metrics-hessian-iters", type=int, default=6)
    parser.add_argument("--metrics-hessian-max-modules", type=int, default=0)
    parser.add_argument("--metrics-projection-correlation-window", type=int, default=16)

    parser.add_argument("--rerun-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--case-index", type=int, default=-1,
                        help="Run only one zero-based case from the full grid. For SLURM arrays.")
    parser.add_argument("--print-case-count", action="store_true",
                        help="Print the number of initial grid cases and exit.")
    parser.add_argument("--summary-only", action="store_true",
                        help="Rebuild sweep_rows.csv from result files and exit.")
    parser.add_argument("--allow-top-k-sweep", action="store_true")
    parser.add_argument("--adaptive-lr", action="store_true")
    parser.add_argument("--lr-extend-factor", type=float, default=2.0)
    parser.add_argument("--lr-min", type=float, default=1e-4)
    parser.add_argument("--lr-max", type=float, default=0.16)
    parser.add_argument("--max-lr-extension-rounds", type=int, default=2)
    parser.add_argument("--adaptive-min-edge-improvement", type=float, default=0.0)
    parser.add_argument("--allow-config-mismatch", action="store_true",
                        help="Allow writing into an OUT_ROOT whose manifest has a different config.")
    args = parser.parse_args()
    if args.tokens is None:
        args.chinchilla_1x_tokens = CHINCHILLA_1X_TOKENS_BY_DEPTH.get(args.depth)
        args.tokens = tokens_from_chinchilla(args.depth, args.chinchilla_mult)
        args.token_budget_source = "chinchilla_table"
    else:
        args.chinchilla_1x_tokens = CHINCHILLA_1X_TOKENS_BY_DEPTH.get(args.depth)
        args.token_budget_source = "manual_tokens"

    unknown = sorted(set(args.methods) - METHOD_CHOICES)
    if unknown:
        raise ValueError(f"unknown methods {unknown}; choices={sorted(METHOD_CHOICES)}")
    validate_args(args)
    if args.top_ks != [1] and not args.allow_top_k_sweep:
        raise ValueError(
            f"current clean recipe fixes top_k=1; got --top-ks {args.top_ks}. "
            "Pass --allow-top-k-sweep only for an explicit ablation."
        )
    return args


def _require_nonempty(name: str, values: list[Any]) -> None:
    if not values:
        raise ValueError(f"--{name.replace('_', '-')} must not be empty")


def _require_finite_positive(name: str, values: list[float], *, allow_zero: bool = False) -> None:
    for value in values:
        if not math.isfinite(float(value)):
            raise ValueError(f"--{name.replace('_', '-')} contains non-finite value {value}")
        if allow_zero:
            if value < 0:
                raise ValueError(f"--{name.replace('_', '-')} values must be >= 0")
        elif value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} values must be > 0")


def validate_args(args: argparse.Namespace) -> None:
    _require_nonempty("methods", args.methods)
    _require_nonempty("batches", args.batches)
    _require_nonempty("lrs", args.lrs)
    _require_nonempty("seeds", args.seeds)
    if "top_aware_muon" in args.methods:
        _require_nonempty("top_ks", args.top_ks)
        _require_nonempty("alphas", args.alphas)

    if any(batch <= 0 for batch in args.batches):
        raise ValueError("--batches values must be > 0")
    _require_finite_positive("lrs", args.lrs)
    _require_finite_positive("alphas", args.alphas, allow_zero=True)
    if any(top_k <= 0 for top_k in args.top_ks):
        raise ValueError("--top-ks values must be > 0")
    if any(seed < 0 for seed in args.seeds):
        raise ValueError("--seeds values must be >= 0")

    if args.tokens <= 0:
        raise ValueError("--tokens must be > 0")
    if args.depth <= 0:
        raise ValueError("--depth must be > 0")
    if not math.isfinite(float(args.chinchilla_mult)) or args.chinchilla_mult <= 0:
        raise ValueError("--chinchilla-mult must be finite and > 0")
    if args.nproc_per_node <= 0:
        raise ValueError("--nproc-per-node must be > 0")
    if args.max_device_batch_size <= 0:
        raise ValueError("--max-device-batch-size must be > 0")
    if args.eval_tokens <= 0:
        raise ValueError("--eval-tokens must be > 0")
    if args.eval_every < 0:
        raise ValueError("--eval-every must be >= 0")
    if args.case_index < -1:
        raise ValueError("--case-index must be -1 or >= 0")
    if args.warmup_ratio < 0:
        raise ValueError("--warmup-ratio must be >= 0")
    if not (0 < args.warmdown_ratio <= 1):
        raise ValueError("--warmdown-ratio must be in (0, 1]")
    if args.final_lr_frac < 0:
        raise ValueError("--final-lr-frac must be >= 0")

    if args.save_every < 0:
        raise ValueError("--save-every must be >= 0")
    if args.keep_last_checkpoints < 0:
        raise ValueError("--keep-last-checkpoints must be >= 0")
    if args.streaming_num_iters <= 0:
        raise ValueError("--streaming-num-iters must be > 0")
    if args.streaming_rank_k == 0 or args.streaming_rank_k < -1:
        raise ValueError("--streaming-rank-k must be -1 or > 0")
    if args.fallback_ortho_tol is not None and not math.isfinite(args.fallback_ortho_tol):
        raise ValueError("--fallback-ortho-tol must be finite")
    if args.precondition_frequency < 0:
        raise ValueError("--precondition-frequency must be >= 0")
    if not (0 <= args.shampoo_beta < 1):
        raise ValueError("--shampoo-beta must be in [0, 1)")
    if not (0 <= args.optimizer_beta1 < 1):
        raise ValueError("--optimizer-beta1 must be in [0, 1)")
    if not (0 <= args.optimizer_beta2 < 1):
        raise ValueError("--optimizer-beta2 must be in [0, 1)")
    if args.structured_init_factor <= 0:
        raise ValueError("--structured-init-factor must be > 0")

    if args.metrics_every < 0:
        raise ValueError("--metrics-every must be >= 0")
    if args.metrics_top_k <= 0:
        raise ValueError("--metrics-top-k must be > 0")
    if args.metrics_max_modules < 0:
        raise ValueError("--metrics-max-modules must be >= 0")
    if args.metrics_hessian_every < 0:
        raise ValueError("--metrics-hessian-every must be >= 0")
    if args.metrics_hessian_every > 0 and args.metrics_every <= 0:
        raise ValueError("--metrics-hessian-every requires --metrics-every > 0")
    if args.metrics_hessian_top_k <= 0:
        raise ValueError("--metrics-hessian-top-k must be > 0")
    if args.metrics_hessian_iters <= 0:
        raise ValueError("--metrics-hessian-iters must be > 0")
    if args.metrics_hessian_max_modules < 0:
        raise ValueError("--metrics-hessian-max-modules must be >= 0")
    if args.metrics_projection_correlation_window < 0:
        raise ValueError("--metrics-projection-correlation-window must be >= 0")

    if args.lr_extend_factor <= 1:
        raise ValueError("--lr-extend-factor must be > 1")
    if args.lr_min <= 0 or args.lr_max <= 0:
        raise ValueError("--lr-min and --lr-max must be > 0")
    if args.lr_min > args.lr_max:
        raise ValueError("--lr-min must be <= --lr-max")
    if args.max_lr_extension_rounds < 0:
        raise ValueError("--max-lr-extension-rounds must be >= 0")
    if args.adaptive_min_edge_improvement < 0:
        raise ValueError("--adaptive-min-edge-improvement must be >= 0")


def main() -> None:
    args = parse_args()
    case_specs = list(iter_case_specs(args, args.methods))
    if args.print_case_count:
        print(len(case_specs))
        return

    args.out_root.mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)

    if args.summary_only:
        # Pure collation: do not check or rewrite manifest, and never launch
        # training/adaptive closure. This only needs completed result.json files.
        print(f"summary_csv={write_summary_csv(args, args.methods)}", flush=True)
        return

    write_manifest(args, args.methods)

    print(f"OUT_ROOT={args.out_root}", flush=True)
    print(f"LOG_ROOT={args.log_root}", flush=True)
    print(f"methods={args.methods}", flush=True)
    print(
        f"depth={args.depth} tokens={args.tokens} "
        f"source={args.token_budget_source} chinchilla_mult={args.chinchilla_mult:g}",
        flush=True,
    )
    print(f"batches={args.batches} lrs={args.lrs} top_ks={args.top_ks} alphas={args.alphas} seeds={args.seeds}", flush=True)

    if args.case_index >= 0:
        if args.case_index >= len(case_specs):
            raise ValueError(f"--case-index {args.case_index} out of range for {len(case_specs)} cases")
        args.append_summary = False
        print(f"case_index={args.case_index}/{len(case_specs)}", flush=True)
        method, batch, lr, seed, top_k, alpha = case_specs[args.case_index]
        score, err = run_case(args, method, batch, lr, seed, top_k=top_k, alpha=alpha)
        if score is None or err is not None:
            raise SystemExit(f"case failed: score={score} error={err}")
        print("\n=== case complete ===", flush=True)
        print("Run the same full-grid command without --case-index to collate CSV and run adaptive LR closure.", flush=True)
        return

    args.append_summary = True
    for method, batch, lr, seed, top_k, alpha in case_specs:
        run_case(args, method, batch, lr, seed, top_k=top_k, alpha=alpha)

    if args.adaptive_lr:
        run_adaptive_lr_rounds(args, args.methods)

    print("\n=== sweep complete ===", flush=True)
    print(f"summary_csv={write_summary_csv(args, args.methods)}", flush=True)


if __name__ == "__main__":
    main()
