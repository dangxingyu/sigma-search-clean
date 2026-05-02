#!/usr/bin/env python3
"""Sweep Top-Aware Muon against Muon/LITE baselines.

Designed for cluster handoff runs where the main grid is:

    batch_size x alpha x matrix_lr, with top_k fixed to 1

The top_k parameter remains in the candidate for future ablations, but the
clean current recipe intentionally holds top_k=1 unless --allow-top-k-sweep is
passed. Baselines are run once per batch/lr/seed. Top-Aware Muon is run for
every alpha/lr point. All StreamingMuon-family methods use run_eval.py.
Native Muon/LITE use run_native_muon_v9.py and run_lite_v9.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any


SEQ = 1024
# d8 Chinchilla-style study budget used by the clean handoff recipes.
# 402,653,184 is near 0.4B tokens and divisible by 262K, 1M, and 4M.
DEFAULT_TOKENS = 402_653_184
METHOD_CHOICES = {
    "streaming_identity",
    "streaming_lite",
    "native_muon",
    "native_lite",
    "top_aware_muon",
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
    data = json.loads(path.read_text())
    err = data.get("error")
    score = data.get("score")
    if err is not None or score is None:
        return None, err or "score_missing"
    return float(score), None


def case_name(method: str, batch: int, lr: float, seed: int, top_k: int | None, alpha: float | None) -> str:
    if method == "top_aware_muon":
        assert top_k is not None and alpha is not None
        return f"top_aware_k{top_k}_a{slug_float(alpha)}_bsz{batch}_lr{slug_float(lr)}_s{seed}"
    return f"{method}_bsz{batch}_lr{slug_float(lr)}_s{seed}"


def result_path(args: argparse.Namespace, method: str, batch: int, lr: float,
                seed: int, top_k: int | None = None, alpha: float | None = None) -> Path:
    return args.out_root / case_name(method, batch, lr, seed, top_k, alpha) / "result.json"


def note_case(args: argparse.Namespace, method: str, batch: int, lr: float,
              seed: int, top_k: int | None, alpha: float | None) -> None:
    spec = (method, int(batch), float(lr), int(seed), top_k, alpha)
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
            f"try lowering --max-device-batch-size"
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
        "--warmup-steps", str(warmup),
        "--warmdown-ratio", f"{args.warmdown_ratio:g}",
        "--final-lr-frac", f"{args.final_lr_frac:g}",
        "--eval-every", str(eval_every),
        "--eval-tokens", str(args.eval_tokens),
        "--seed", str(seed),
    ]


def streaming_metrics_args(args: argparse.Namespace) -> list[str]:
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
        ]
    return out


def build_command(args: argparse.Namespace, method: str, batch: int, lr: float, seed: int,
                  out: Path, top_k: int | None, alpha: float | None) -> list[str]:
    cmd = ["torchrun", "--standalone", f"--nproc_per_node={args.nproc_per_node}"]
    common = common_training_args(args, batch, lr, seed, out)
    metrics = streaming_metrics_args(args)

    if method == "streaming_identity":
        return cmd + [
            "run_eval.py",
            "--candidate-file", "candidates/identity.py",
            *common,
            "--k", str(args.streaming_rank_k),
            "--num-iters", str(args.streaming_num_iters),
            "--fallback-ortho-tol", f"{args.fallback_ortho_tol:g}",
            *(["--pure-qr"] if args.pure_qr else []),
            *metrics,
        ]
    if method == "streaming_lite":
        return cmd + [
            "run_eval.py",
            "--candidate-file", "candidates/lite_chi2_rs01.py",
            *common,
            "--k", str(args.streaming_rank_k),
            "--num-iters", str(args.streaming_num_iters),
            "--fallback-ortho-tol", f"{args.fallback_ortho_tol:g}",
            *(["--pure-qr"] if args.pure_qr else []),
            *metrics,
        ]
    if method == "top_aware_muon":
        assert top_k is not None and alpha is not None
        return cmd + [
            "run_eval.py",
            "--candidate-file", "candidates/top_aware_muon.py",
            "--candidate-param", f"top_k={top_k}",
            "--candidate-param", f"alpha={alpha:g}",
            *common,
            "--k", str(args.streaming_rank_k),
            "--num-iters", str(args.streaming_num_iters),
            "--fallback-ortho-tol", f"{args.fallback_ortho_tol:g}",
            *(["--pure-qr"] if args.pure_qr else []),
            *metrics,
        ]
    if method == "native_muon":
        if args.metrics_every > 0:
            raise ValueError("metrics logging is supported only for StreamingMuon methods in the clean repo")
        return cmd + ["run_native_muon_v9.py", *common, "--ns-steps", str(args.ns_steps), *metrics]
    if method == "native_lite":
        if args.metrics_every > 0:
            raise ValueError("metrics logging is supported only for StreamingMuon methods in the clean repo")
        if args.nproc_per_node != 1:
            raise ValueError(
                "native_lite is single-process only in this repo. "
                "Run it separately with --nproc-per-node 1 / NPROC=1."
            )
        return [
            "python",
            "run_lite_v9.py",
            *common,
            "--ns-steps", str(args.ns_steps),
            "--lite-chi", f"{args.lite_chi:g}",
            "--lite-rs", f"{args.lite_rs:g}",
            "--lite-chi-warmup", f"{args.lite_chi_warmup:g}",
            "--lite-chi-schedule", args.lite_chi_schedule,
        ]
    raise ValueError(f"unknown method {method}")


def append_summary(args: argparse.Namespace, row: dict) -> None:
    path = args.out_root / "top_aware_sweep_rows.csv"
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
    note_case(args, method, batch, lr, seed, top_k, alpha)
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
    printable = " ".join(shlex.quote(x) for x in cmd)
    print(f"\n=== {name} ===", flush=True)
    print(printable, flush=True)
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
        print(f"[warn] {name}: torchrun rc={rc}; checking result JSON", flush=True)

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
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "out_root": str(args.out_root),
        "log_root": str(args.log_root),
        "methods": methods,
        "batches": args.batches,
        "lrs": args.lrs,
        "top_ks": args.top_ks,
        "alphas": args.alphas,
        "seeds": args.seeds,
        "tokens": args.tokens,
        "depth": args.depth,
        "seq": SEQ,
        "nproc_per_node": args.nproc_per_node,
        "max_device_batch_size": args.max_device_batch_size,
        "streaming": {
            "pure_qr": args.pure_qr,
            "num_iters": args.streaming_num_iters,
            "fallback_ortho_tol": args.fallback_ortho_tol,
            "rank_k": args.streaming_rank_k,
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
        "baselines": {
            "streaming_identity": "run_eval.py + candidates/identity.py",
            "streaming_lite": "run_eval.py + candidates/lite_chi2_rs01.py",
            "native_muon": "run_native_muon_v9.py",
            "native_lite": "run_lite_v9.py",
        },
    }
    (args.out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))


def iter_case_specs(args: argparse.Namespace, methods: list[str]):
    baseline_methods = [m for m in methods if m != "top_aware_muon"]
    run_top_aware = "top_aware_muon" in methods
    for batch in args.batches:
        for seed in args.seeds:
            for method in baseline_methods:
                for lr in args.lrs:
                    yield method, batch, lr, seed, None, None
            if run_top_aware:
                for top_k in args.top_ks:
                    for alpha in args.alphas:
                        for lr in args.lrs:
                            yield "top_aware_muon", batch, lr, seed, top_k, alpha


def write_summary_csv(args: argparse.Namespace, methods: list[str]) -> Path:
    path = args.out_root / "top_aware_sweep_rows.csv"
    fields = [
        "time", "method", "batch", "lr", "seed", "top_k", "alpha",
        "score", "error", "result_json", "log_file",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        case_specs = getattr(args, "_case_specs", None)
        if case_specs is None:
            case_specs = list(iter_case_specs(args, methods))
        for method, batch, lr, seed, top_k, alpha in case_specs:
            out = result_path(args, method, batch, lr, seed, top_k, alpha)
            log = args.log_root / f"{out.parent.name}.log"
            score, err = load_score(out)
            completed_at = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(out.stat().st_mtime)) if out.exists() else ""
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
    """Return extra LR cases when the best completed LR is at a grid edge.

    Scores are validation BPB, so lower is better. Extensions are independent
    for each `(method, batch, seed, top_k, alpha)` group.
    """
    grouped: dict[tuple[Any, ...], list[tuple[float, str, int, int | None, float | None]]] = {}
    case_specs = getattr(args, "_case_specs", list(iter_case_specs(args, methods)))
    for method, batch, lr, seed, top_k, alpha in case_specs:
        grouped.setdefault(group_key(method, batch, seed, top_k, alpha), []).append(
            (float(lr), method, int(batch), top_k, alpha)
        )

    proposals: list[tuple[str, int, float, int, int | None, float | None, str]] = []
    seen = set(getattr(args, "_case_spec_set", set()))
    for key, lr_specs in sorted(grouped.items(), key=lambda kv: repr(kv[0])):
        method, batch, seed, top_k, alpha = key
        lrs = sorted({lr for lr, *_ in lr_specs})
        scored: list[tuple[float, float | None, str | None]] = []
        for lr in lrs:
            score, err = load_score(result_path(args, method, batch, lr, seed, top_k, alpha))
            scored.append((lr, score, err))
        finite = [(lr, score) for lr, score, err in scored if score is not None and err is None]
        if len(finite) < 2:
            continue

        finite.sort()
        best_lr, best_score = min(finite, key=lambda item: item[1])
        finite_lrs = [lr for lr, _ in finite]
        finite_scores = {lr: score for lr, score in finite}

        low_lr = finite_lrs[0]
        if best_lr == low_lr and (key, "low") not in blocked_edges:
            attempted_lower = [lr for lr in lrs if lr < low_lr]
            if attempted_lower:
                blocked_edges.add((key, "low"))
            elif len(finite_lrs) >= 2 and edge_is_strong_enough(
                best_score, finite_scores.get(finite_lrs[1]), args.adaptive_min_edge_improvement
            ):
                new_lr = low_lr / args.lr_extend_factor
                if new_lr >= args.lr_min:
                    spec = (method, batch, new_lr, seed, top_k, alpha)
                    if spec not in seen:
                        proposals.append((*spec, "low"))
                        seen.add(spec)
                else:
                    blocked_edges.add((key, "low"))

        high_lr = finite_lrs[-1]
        if best_lr == high_lr and (key, "high") not in blocked_edges:
            attempted_higher = [lr for lr in lrs if lr > high_lr]
            if attempted_higher:
                blocked_edges.add((key, "high"))
            elif len(finite_lrs) >= 2 and edge_is_strong_enough(
                best_score, finite_scores.get(finite_lrs[-2]), args.adaptive_min_edge_improvement
            ):
                new_lr = high_lr * args.lr_extend_factor
                if new_lr <= args.lr_max:
                    spec = (method, batch, new_lr, seed, top_k, alpha)
                    if spec not in seen:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--out-root", type=Path, default=Path(f"search_evals/top_aware_muon_sweep_{stamp}"))
    parser.add_argument("--log-root", type=Path, default=Path(f"logs/top_aware_muon_sweep_{stamp}"))
    parser.add_argument("--nanochat-dir", type=str, default="nanochat")
    parser.add_argument("--methods", type=parse_list_str,
                        default=parse_list_str("streaming_identity top_aware_muon"),
                        help=(
                            "Space/comma separated methods. Default is the current streaming-first recipe: "
                            "streaming_identity top_aware_muon. Native baselines remain available by explicitly "
                            "passing native_muon or native_lite."
                        ))
    parser.add_argument("--batches", type=parse_list_int, default=parse_list_int("131072"))
    parser.add_argument("--lrs", type=parse_list_float, default=parse_list_float("0.005 0.01 0.02 0.04"))
    parser.add_argument("--top-ks", type=parse_list_int, default=parse_list_int("1"))
    parser.add_argument("--alphas", type=parse_list_float, default=parse_list_float("0.5"))
    parser.add_argument("--seeds", type=parse_list_int, default=parse_list_int("42"))
    parser.add_argument("--tokens", type=int, default=DEFAULT_TOKENS)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--max-device-batch-size", type=int, default=16)
    parser.add_argument("--eval-tokens", type=int, default=524288)
    parser.add_argument("--eval-every", type=int, default=0, help="0 means steps//8")
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--warmdown-ratio", type=float, default=0.65)
    parser.add_argument("--final-lr-frac", type=float, default=0.05)
    parser.add_argument("--streaming-num-iters", type=int, default=2)
    parser.add_argument("--streaming-rank-k", type=int, default=-1)
    parser.add_argument("--fallback-ortho-tol", type=float, default=0.01)
    parser.add_argument("--pure-qr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--metrics-every", type=int, default=0,
                        help="Pass diagnostic logging interval to StreamingMuon run_eval.py; 0 disables.")
    parser.add_argument("--metrics-top-k", type=int, default=4)
    parser.add_argument("--metrics-module-regex", type=str,
                        default=r"transformer\.h\.(?:[0-9]+)\.(?:attn\.(?:c_q|c_k|c_v|c_proj)|mlp\.(?:c_fc|c_proj))\.weight$")
    parser.add_argument("--metrics-max-modules", type=int, default=0)
    parser.add_argument("--metrics-hessian-every", type=int, default=0)
    parser.add_argument("--metrics-hessian-top-k", type=int, default=1)
    parser.add_argument("--metrics-hessian-iters", type=int, default=6)
    parser.add_argument("--metrics-hessian-max-modules", type=int, default=0)
    parser.add_argument("--ns-steps", type=int, default=5)
    parser.add_argument("--lite-chi", type=float, default=2.0)
    parser.add_argument("--lite-rs", type=float, default=0.1)
    parser.add_argument("--lite-chi-warmup", type=float, default=0.5)
    parser.add_argument("--lite-chi-schedule", type=str, default="warmup_hold")
    parser.add_argument("--rerun-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-top-k-sweep", action="store_true",
                        help="Allow top_k values other than exactly 1. Current clean recipe keeps top_k fixed to 1.")
    parser.add_argument("--adaptive-lr", action="store_true",
                        help="After the initial LR grid, extend outward if the best finite score is at a grid edge.")
    parser.add_argument("--lr-extend-factor", type=float, default=2.0,
                        help="Multiplier/divider for adaptive LR boundary extensions.")
    parser.add_argument("--lr-min", type=float, default=1e-4,
                        help="Minimum LR allowed for adaptive low-edge extensions.")
    parser.add_argument("--lr-max", type=float, default=0.08,
                        help="Maximum LR allowed for adaptive high-edge extensions.")
    parser.add_argument("--max-lr-extension-rounds", type=int, default=2,
                        help="Maximum adaptive LR extension rounds after the initial grid.")
    parser.add_argument("--adaptive-min-edge-improvement", type=float, default=0.0,
                        help="Require edge BPB to beat the adjacent LR by at least this amount before extending.")
    args = parser.parse_args()

    unknown = sorted(set(args.methods) - METHOD_CHOICES)
    if unknown:
        raise ValueError(f"unknown methods {unknown}; choices={sorted(METHOD_CHOICES)}")
    if args.top_ks != [1] and not args.allow_top_k_sweep:
        raise ValueError(
            f"current clean recipe fixes top_k=1; got --top-ks {args.top_ks}. "
            "Pass --allow-top-k-sweep only for an explicit ablation."
        )

    args.out_root.mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)
    write_manifest(args, args.methods)

    print(f"OUT_ROOT={args.out_root}", flush=True)
    print(f"LOG_ROOT={args.log_root}", flush=True)
    print(f"methods={args.methods}", flush=True)
    print(f"batches={args.batches} lrs={args.lrs} top_ks={args.top_ks} alphas={args.alphas} seeds={args.seeds}", flush=True)

    for method, batch, lr, seed, top_k, alpha in iter_case_specs(args, args.methods):
        run_case(args, method, batch, lr, seed, top_k=top_k, alpha=alpha)

    if args.adaptive_lr:
        run_adaptive_lr_rounds(args, args.methods)

    print("\n=== sweep complete ===", flush=True)
    print(f"summary_csv={write_summary_csv(args, args.methods)}", flush=True)


if __name__ == "__main__":
    main()
