#!/usr/bin/env python3
"""Adaptive LR sweep for top-1 damping in StreamingMuon."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


BATCHES = [131072, 1048576, 8388608]
METHODS = [
    ("identity", "candidates/identity.py"),
    ("top1_damp_a05", "candidates/top1_damp_alpha05.py"),
]
INITIAL_LRS = [0.01, 0.02, 0.04]
MIN_LR = 0.00125
MAX_LR = 0.16

# Below this, a lower score is not worth another expensive outward probe.
MIN_MEANINGFUL_GAIN = 3e-4

# If a boundary LR is already this much worse than the current best, do not
# expand further in that direction.
CLEARLY_WORSE = 0.006


def lr_key(lr: float) -> str:
    return f"{lr:g}".replace(".", "p")


def steps_for_batch(bsz: int) -> int:
    if bsz == 131072:
        return 8192
    if bsz == 1048576:
        return 1024
    if bsz == 8388608:
        return 128
    raise ValueError(f"unsupported batch {bsz}")


def eval_every_for_steps(steps: int) -> int:
    if steps == 8192:
        return 1024
    if steps == 1024:
        return 128
    if steps == 128:
        return 16
    return max(16, steps // 8)


def result_path(out_root: Path, method: str, bsz: int, lr: float) -> Path:
    return out_root / f"{method}_bsz{bsz}_lr{lr_key(lr)}" / "result.json"


def load_score(path: Path) -> tuple[float | None, str | None]:
    if not path.exists():
        return None, "missing"
    data = json.loads(path.read_text())
    err = data.get("error")
    score = data.get("score")
    if err is not None or score is None:
        return None, err or "score_missing"
    return float(score), None


def warmup_steps_for_run(args: argparse.Namespace, steps: int) -> int:
    if args.fixed_warmup_steps is not None:
        return args.fixed_warmup_steps
    return max(1, round(args.warmup_ratio * steps))


def run_case(args: argparse.Namespace, method: str, candidate: str, bsz: int, lr: float) -> tuple[float | None, str | None]:
    out = result_path(args.out_root, method, bsz, lr)
    log = args.log_root / f"{method}_bsz{bsz}_lr{lr_key(lr)}.log"
    out.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.rerun_existing:
        score, err = load_score(out)
        if err is None and score is not None:
            print(f"skip valid existing {out.parent.name}: score={score}", flush=True)
            return score, err
        print(f"rerun invalid existing {out.parent.name}: score={score} error={err}", flush=True)

    steps = steps_for_batch(bsz)
    eval_every = eval_every_for_steps(steps)
    warmup_steps = warmup_steps_for_run(args, steps)
    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={args.nproc_per_node}",
        "run_eval.py",
        "--candidate-file",
        candidate,
        "--output-file",
        str(out),
        "--depth",
        str(args.depth),
        "--max-seq-len",
        "1024",
        "--device-batch-size",
        str(args.device_batch_size),
        "--total-batch-size",
        str(bsz),
        "--max-steps",
        str(steps),
        "--eval-every",
        str(eval_every),
        "--eval-tokens",
        str(args.eval_tokens),
        "--matrix-lr",
        f"{lr:g}",
        "--warmup-steps",
        str(warmup_steps),
        "--num-iters",
        "1",
        "--fallback-ortho-tol",
        str(args.fallback_ortho_tol),
        "--seed",
        str(args.seed),
    ]
    print(
        f"\n=== run {out.parent.name}: steps={steps} "
        f"warmup={warmup_steps} eval_every={eval_every} ===",
        flush=True,
    )
    print(" ".join(cmd), flush=True)
    with log.open("w") as f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
        rc = proc.wait()
    if rc != 0:
        print(f"torchrun exited rc={rc}; checking JSON if available", flush=True)
    score, err = load_score(out)
    print(f"completed {out.parent.name}: score={score} error={err}", flush=True)
    return score, err


def finite_scores(scores: dict[float, tuple[float | None, str | None]]) -> dict[float, float]:
    return {lr: score for lr, (score, err) in scores.items() if score is not None and err is None}


def best_lr(scores: dict[float, tuple[float | None, str | None]]) -> tuple[float, float] | None:
    finite = finite_scores(scores)
    if not finite:
        return None
    return min(finite.items(), key=lambda kv: kv[1])


def should_expand_high(scores: dict[float, tuple[float | None, str | None]], current_high: float) -> bool:
    finite = finite_scores(scores)
    if current_high not in finite:
        return False
    best = min(finite.values())
    return finite[current_high] <= best + CLEARLY_WORSE


def should_expand_low(scores: dict[float, tuple[float | None, str | None]], current_low: float) -> bool:
    finite = finite_scores(scores)
    if current_low not in finite:
        return False
    best = min(finite.values())
    return finite[current_low] <= best + CLEARLY_WORSE


def sweep_one(args: argparse.Namespace, method: str, candidate: str, bsz: int) -> dict[float, tuple[float | None, str | None]]:
    print(f"\n### adaptive sweep method={method} batch={bsz} ###", flush=True)
    scores: dict[float, tuple[float | None, str | None]] = {}

    for lr in INITIAL_LRS:
        scores[lr] = run_case(args, method, candidate, bsz, lr)

    # Expand upward only if the best is at the current high boundary, or if the
    # high boundary is close enough to best that we have not found the cliff.
    high = max(INITIAL_LRS)
    while high < MAX_LR and should_expand_high(scores, high):
        current_best = best_lr(scores)
        next_lr = round(high * 2, 10)
        if current_best is not None and current_best[0] != high and scores[high][0] is not None:
            # If high is not best and is already mildly worse, don't pay for one
            # more outward point.
            if scores[high][0] > current_best[1] + MIN_MEANINGFUL_GAIN:
                break
        scores[next_lr] = run_case(args, method, candidate, bsz, next_lr)
        new_best = best_lr(scores)
        high = next_lr
        if new_best is not None and new_best[0] != next_lr:
            break

    # Expand downward if the current low boundary looks best or near-best.
    low = min(INITIAL_LRS)
    while low > MIN_LR and should_expand_low(scores, low):
        current_best = best_lr(scores)
        next_lr = round(low / 2, 10)
        if current_best is not None and current_best[0] != low and scores[low][0] is not None:
            if scores[low][0] > current_best[1] + MIN_MEANINGFUL_GAIN:
                break
        scores[next_lr] = run_case(args, method, candidate, bsz, next_lr)
        new_best = best_lr(scores)
        low = next_lr
        if new_best is not None and new_best[0] != next_lr:
            break

    best = best_lr(scores)
    if best is None:
        print(f"NO VALID RUNS method={method} batch={bsz}", flush=True)
    else:
        print(f"BEST method={method} batch={bsz}: lr={best[0]:g} score={best[1]:.6f}", flush=True)
    return scores


def write_manifest(args: argparse.Namespace) -> None:
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "out_root": str(args.out_root),
        "log_root": str(args.log_root),
        "depth": args.depth,
        "seed": args.seed,
        "batches": BATCHES,
        "methods": METHODS,
        "initial_lrs": INITIAL_LRS,
        "min_lr": MIN_LR,
        "max_lr": MAX_LR,
        "warmup_ratio": args.warmup_ratio,
        "fixed_warmup_steps": args.fixed_warmup_steps,
        "warmdown_ratio": 0.65,
        "final_lr_frac": 0.05,
        "fallback_ortho_tol": args.fallback_ortho_tol,
        "decision": {
            "min_meaningful_gain": MIN_MEANINGFUL_GAIN,
            "clearly_worse": CLEARLY_WORSE,
        },
    }
    (args.out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--out-root", type=Path, default=Path(f"search_evals/ddp8_top1_damp_adaptive_{stamp}"))
    parser.add_argument("--log-root", type=Path, default=Path(f"logs/ddp8_top1_damp_adaptive_{stamp}"))
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--device-batch-size", type=int, default=16)
    parser.add_argument("--eval-tokens", type=int, default=524288)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--fixed-warmup-steps", type=int, default=None)
    parser.add_argument("--fallback-ortho-tol", type=float, default=0.01)
    parser.add_argument("--rerun-existing", action="store_true")
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)
    write_manifest(args)

    print(f"OUT_ROOT={args.out_root}", flush=True)
    print(f"LOG_ROOT={args.log_root}", flush=True)
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}", flush=True)

    for bsz in BATCHES:
        for method, candidate in METHODS:
            sweep_one(args, method, candidate, bsz)

    print("\n=== final analysis ===", flush=True)
    subprocess.run([sys.executable, "analyze_top1_damp_lr_sweep.py", str(args.out_root)], check=False)


if __name__ == "__main__":
    main()
