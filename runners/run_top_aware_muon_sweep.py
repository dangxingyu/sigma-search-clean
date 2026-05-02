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


SEQ = 1024
DEFAULT_TOKENS = 1_073_741_824
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
        "--metrics-alignment-side", args.metrics_alignment_side,
    ]
    if args.metrics_split_momentum:
        out.append("--metrics-split-momentum")
    if args.metrics_save_components:
        out.append("--metrics-save-components")
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
        return cmd + ["run_native_muon_v9.py", *common, "--ns-steps", str(args.ns_steps), *metrics]
    if method == "native_lite":
        return cmd + [
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
            print(line, end="")
            f.write(line)
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
            "split_momentum": args.metrics_split_momentum,
            "alignment_side": args.metrics_alignment_side,
            "save_components": args.metrics_save_components,
            "hessian_every": args.metrics_hessian_every,
            "hessian_top_k": args.metrics_hessian_top_k,
            "hessian_iters": args.metrics_hessian_iters,
            "hessian_max_modules": args.metrics_hessian_max_modules,
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
        for method, batch, lr, seed, top_k, alpha in iter_case_specs(args, methods):
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


def main() -> None:
    parser = argparse.ArgumentParser()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--out-root", type=Path, default=Path(f"search_evals/top_aware_muon_sweep_{stamp}"))
    parser.add_argument("--log-root", type=Path, default=Path(f"logs/top_aware_muon_sweep_{stamp}"))
    parser.add_argument("--nanochat-dir", type=str, default="nanochat")
    parser.add_argument("--methods", type=parse_list_str,
                        default=parse_list_str("streaming_identity streaming_lite native_muon native_lite top_aware_muon"))
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
    parser.add_argument("--metrics-module-regex", type=str, default=r"transformer\.h")
    parser.add_argument("--metrics-max-modules", type=int, default=0)
    parser.add_argument("--metrics-split-momentum", action="store_true")
    parser.add_argument("--metrics-alignment-side", type=str, default="lite", choices=("left", "right", "lite"))
    parser.add_argument("--metrics-save-components", action="store_true")
    parser.add_argument("--metrics-hessian-every", type=int, default=0)
    parser.add_argument("--metrics-hessian-top-k", type=int, default=1)
    parser.add_argument("--metrics-hessian-iters", type=int, default=6)
    parser.add_argument("--metrics-hessian-max-modules", type=int, default=1)
    parser.add_argument("--ns-steps", type=int, default=5)
    parser.add_argument("--lite-chi", type=float, default=2.0)
    parser.add_argument("--lite-rs", type=float, default=0.1)
    parser.add_argument("--lite-chi-warmup", type=float, default=0.5)
    parser.add_argument("--lite-chi-schedule", type=str, default="warmup_hold")
    parser.add_argument("--rerun-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-top-k-sweep", action="store_true",
                        help="Allow top_k values other than exactly 1. Current clean recipe keeps top_k fixed to 1.")
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

    print("\n=== sweep complete ===", flush=True)
    print(f"summary_csv={write_summary_csv(args, args.methods)}", flush=True)


if __name__ == "__main__":
    main()
