#!/usr/bin/env python3
"""Fair same-driver 32K/64K comparison for StreamingMuon identity, LITE-like, and top1."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


METHODS = [
    ("identity", "candidates/identity.py"),
    ("lite_chi2_rs01", "candidates/lite_chi2_rs01.py"),
    ("top1pm_a05", "candidates/top1_damp_alpha05_per_matrix.py"),
]
DEFAULT_TOKENS = 1_073_741_824
SEQ = 1024
INITIAL_LRS = [0.005, 0.01, 0.02]
MIN_LR = 0.00125
MAX_LR = 0.08
CLEARLY_WORSE = 0.006
MIN_MEANINGFUL_GAIN = 3e-4


def lr_key(lr: float) -> str:
    return f"{lr:g}".replace(".", "p")


def parse_list_int(spec: str) -> list[int]:
    return [int(x.strip()) for x in spec.replace(",", " ").split() if x.strip()]


def parse_list_float(spec: str) -> list[float]:
    return [float(x.strip()) for x in spec.replace(",", " ").split() if x.strip()]


def steps_for_batch(tokens: int, batch: int) -> int:
    return tokens // batch


def device_batch_for(batch: int, nproc: int) -> int:
    denom = SEQ * nproc
    if batch % denom != 0:
        raise ValueError(f"batch {batch} not divisible by seq*nproc={denom}")
    db = batch // denom
    if db < 1:
        raise ValueError(f"batch {batch} gives device_batch_size={db} for nproc={nproc}")
    return db


def eval_every_for_steps(steps: int) -> int:
    return max(16, steps // 8)


def result_path(root: Path, method: str, batch: int, lr: float, seed: int) -> Path:
    return root / f"{method}_bsz{batch}_lr{lr_key(lr)}_s{seed}" / "result.json"


def load_score(path: Path) -> tuple[float | None, str | None]:
    if not path.exists():
        return None, "missing"
    data = json.loads(path.read_text())
    err = data.get("error")
    score = data.get("score")
    if err is not None or score is None:
        return None, err or "score_missing"
    return float(score), None


def finite_scores(scores: dict[float, tuple[float | None, str | None]]) -> dict[float, float]:
    return {lr: s for lr, (s, err) in scores.items() if s is not None and err is None}


def best_lr(scores: dict[float, tuple[float | None, str | None]]) -> tuple[float, float] | None:
    finite = finite_scores(scores)
    if not finite:
        return None
    return min(finite.items(), key=lambda kv: kv[1])


def should_expand(scores: dict[float, tuple[float | None, str | None]], boundary: float) -> bool:
    finite = finite_scores(scores)
    if boundary not in finite:
        return False
    best = min(finite.values())
    return finite[boundary] <= best + CLEARLY_WORSE


def write_manifest(args: argparse.Namespace) -> None:
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "out_root": str(args.out_root),
        "log_root": str(args.log_root),
        "tokens": args.tokens,
        "depth": args.depth,
        "batches": args.batches,
        "seeds": args.seeds,
        "methods": METHODS,
        "initial_lrs": args.lrs,
        "min_lr": MIN_LR,
        "max_lr": MAX_LR,
        "warmup_ratio": args.warmup_ratio,
        "warmdown_ratio": 0.65,
        "final_lr_frac": 0.05,
        "nproc_per_node": args.nproc_per_node,
        "pure_qr": True,
        "num_iters": 2,
        "fallback_ortho_tol": args.fallback_ortho_tol,
        "notes": [
            "Same-driver comparison: all three methods use run_eval.py + StreamingMuon.",
            "identity is Muon-like; lite_chi2_rs01 is LITE-like chi=2 rs=0.1; top1pm_a05 damps the top singular direction per matrix.",
            "All StreamingMuon runs use pure Householder QR and two streaming iterations per optimizer step.",
        ],
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))


def run_case(args: argparse.Namespace, method: str, candidate: str, batch: int, lr: float, seed: int) -> tuple[float | None, str | None]:
    out = result_path(args.out_root, method, batch, lr, seed)
    log = args.log_root / f"{method}_bsz{batch}_lr{lr_key(lr)}_s{seed}.log"
    out.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)

    if out.exists() and not args.rerun_existing:
        score, err = load_score(out)
        if score is not None and err is None:
            print(f"skip valid existing {out.parent.name}: score={score:.6f}", flush=True)
            return score, None
        print(f"rerun invalid existing {out.parent.name}: score={score} error={err}", flush=True)

    steps = steps_for_batch(args.tokens, batch)
    device_batch = device_batch_for(batch, args.nproc_per_node)
    warmup = max(1, round(args.warmup_ratio * steps))
    eval_every = eval_every_for_steps(steps)
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
        str(SEQ),
        "--device-batch-size",
        str(device_batch),
        "--total-batch-size",
        str(batch),
        "--max-steps",
        str(steps),
        "--matrix-lr",
        f"{lr:g}",
        "--warmup-steps",
        str(warmup),
        "--warmdown-ratio",
        "0.65",
        "--final-lr-frac",
        "0.05",
        "--eval-every",
        str(eval_every),
        "--eval-tokens",
        str(args.eval_tokens),
        "--num-iters",
        "2",
        "--pure-qr",
        "--fallback-ortho-tol",
        str(args.fallback_ortho_tol),
        "--seed",
        str(seed),
    ]
    print(
        f"\n=== run {out.parent.name}: steps={steps} warmup={warmup} "
        f"eval_every={eval_every} device_bsz={device_batch} nproc={args.nproc_per_node} ===",
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


def sweep_one(args: argparse.Namespace, method: str, candidate: str, batch: int, seed: int) -> dict[float, tuple[float | None, str | None]]:
    print(f"\n### sweep method={method} batch={batch} seed={seed} ###", flush=True)
    scores: dict[float, tuple[float | None, str | None]] = {}
    for lr in args.lrs:
        scores[lr] = run_case(args, method, candidate, batch, lr, seed)

    high = max(args.lrs)
    while high < MAX_LR and should_expand(scores, high):
        current_best = best_lr(scores)
        if current_best is not None and current_best[0] != high and scores[high][0] is not None:
            if scores[high][0] > current_best[1] + MIN_MEANINGFUL_GAIN:
                break
        next_lr = round(high * 2, 10)
        scores[next_lr] = run_case(args, method, candidate, batch, next_lr, seed)
        high = next_lr
        new_best = best_lr(scores)
        if new_best is not None and new_best[0] != next_lr:
            break

    low = min(args.lrs)
    while low > MIN_LR and should_expand(scores, low):
        current_best = best_lr(scores)
        if current_best is not None and current_best[0] != low and scores[low][0] is not None:
            if scores[low][0] > current_best[1] + MIN_MEANINGFUL_GAIN:
                break
        next_lr = round(low / 2, 10)
        scores[next_lr] = run_case(args, method, candidate, batch, next_lr, seed)
        low = next_lr
        new_best = best_lr(scores)
        if new_best is not None and new_best[0] != next_lr:
            break

    best = best_lr(scores)
    if best is None:
        print(f"NO VALID RUNS method={method} batch={batch} seed={seed}", flush=True)
    else:
        print(f"BEST method={method} batch={batch} seed={seed}: lr={best[0]:g} score={best[1]:.6f}", flush=True)
    return scores


def run_analysis(args: argparse.Namespace) -> None:
    subprocess.run(
        [sys.executable, "analyze_v26_small_bsz_streaming_fair.py", str(args.out_root)],
        check=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--out-root", type=Path, default=Path(f"search_evals/v26_small_bsz_streaming_fair_{stamp}"))
    parser.add_argument("--log-root", type=Path, default=Path(f"logs/v26_small_bsz_streaming_fair_{stamp}"))
    parser.add_argument("--batches", type=parse_list_int, default=parse_list_int("65536 32768"))
    parser.add_argument("--seeds", type=parse_list_int, default=parse_list_int("42"))
    parser.add_argument("--lrs", type=parse_list_float, default=INITIAL_LRS)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=DEFAULT_TOKENS)
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--eval-tokens", type=int, default=524288)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--fallback-ortho-tol", type=float, default=0.01)
    parser.add_argument("--rerun-existing", action="store_true")
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)
    write_manifest(args)

    print(f"OUT_ROOT={args.out_root}", flush=True)
    print(f"LOG_ROOT={args.log_root}", flush=True)
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}", flush=True)
    print(f"batches={args.batches} seeds={args.seeds} lrs={args.lrs}", flush=True)

    for batch in args.batches:
        for seed in args.seeds:
            for method, candidate in METHODS:
                sweep_one(args, method, candidate, batch, seed)
                run_analysis(args)

    print("\n=== final analysis ===", flush=True)
    run_analysis(args)


if __name__ == "__main__":
    main()
