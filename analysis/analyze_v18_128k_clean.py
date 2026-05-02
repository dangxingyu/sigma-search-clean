#!/usr/bin/env python3
"""Analyze v18 clean 128K Muon vs LITE small-batch sweep."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("sweep_results_v18_128k_clean")
OUT = Path("v18_128k_clean_analysis.png")


def load_runs() -> list[dict]:
    pat = re.compile(r"(muon|lite)_bsz(\d+)_lr([\d.]+)_s(\d+)\.json$")
    runs = []
    for path in ROOT.glob("*.json"):
        m = pat.match(path.name)
        if not m:
            continue
        opt, bsz, lr, seed = m.groups()
        data = json.loads(path.read_text())
        runs.append({
            "path": path,
            "opt": opt,
            "bsz": int(bsz),
            "lr": float(lr),
            "seed": int(seed),
            "score": data.get("score", data.get("val_bpb_best")),
            "error": data.get("error"),
            "data": data,
        })
    return runs


def mean_sem(xs: list[float]) -> tuple[float, float]:
    xs = [float(x) for x in xs if x is not None and not math.isnan(float(x))]
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], 0.0
    return float(np.mean(xs)), float(np.std(xs, ddof=1) / np.sqrt(len(xs)))


def main() -> None:
    runs = load_runs()
    done = [r for r in runs if r["error"] is None and r["score"] is not None]
    print(f"Loaded {len(runs)} JSONs, complete {len(done)}")
    if not done:
        return

    by_opt_lr = defaultdict(list)
    for r in done:
        by_opt_lr[(r["opt"], r["lr"])].append(float(r["score"]))

    print("\n=== v18 128K LR table ===")
    print(f"{'opt':>5s} | {'lr':>5s} | {'n':>2s} | {'mean BPB':>9s} | {'sem':>7s}")
    rows = []
    for opt in ["muon", "lite"]:
        for lr in sorted({r["lr"] for r in done if r["opt"] == opt}):
            vals = by_opt_lr[(opt, lr)]
            m, s = mean_sem(vals)
            rows.append({"opt": opt, "lr": lr, "mean": m, "sem": s, "n": len(vals)})
            print(f"{opt:>5s} | {lr:>5.3f} | {len(vals):>2d} | {m:>9.5f} | {s:>7.5f}")

    best_lr = {}
    for opt in ["muon", "lite"]:
        opt_rows = [r for r in rows if r["opt"] == opt and r["n"] > 0]
        if opt_rows:
            best_lr[opt] = min(opt_rows, key=lambda r: r["mean"])
    if set(best_lr) == {"muon", "lite"}:
        delta = best_lr["muon"]["mean"] - best_lr["lite"]["mean"]
        print("\n=== best LR comparison ===")
        print(f"Muon best: lr={best_lr['muon']['lr']:.3f}, BPB={best_lr['muon']['mean']:.5f}")
        print(f"LITE best: lr={best_lr['lite']['lr']:.3f}, BPB={best_lr['lite']['mean']:.5f}")
        print(f"Delta Muon-LITE={delta:+.5f} (positive means LITE wins)")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    for opt, color in [("muon", "tab:blue"), ("lite", "tab:orange")]:
        opt_rows = sorted([r for r in rows if r["opt"] == opt], key=lambda r: r["lr"])
        if not opt_rows:
            continue
        ax.errorbar(
            [r["lr"] for r in opt_rows],
            [r["mean"] for r in opt_rows],
            yerr=[r["sem"] for r in opt_rows],
            marker="o",
            capsize=3,
            label=opt,
            color=color,
        )
    ax.set_xscale("log")
    ax.set_xlabel("base matrix LR")
    ax.set_ylabel("best val BPB")
    ax.set_title("128K LR sweep")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1]
    labels, vals, errs = [], [], []
    for opt in ["muon", "lite"]:
        if opt in best_lr:
            labels.append(f"{opt}\nlr={best_lr[opt]['lr']:.3f}")
            vals.append(best_lr[opt]["mean"])
            errs.append(best_lr[opt]["sem"])
    ax.bar(labels, vals, yerr=errs, capsize=4, color=["tab:blue", "tab:orange"][:len(vals)])
    if vals:
        lo, hi = min(vals), max(vals)
        pad = max(0.003, (hi - lo) * 4)
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_ylabel("best val BPB")
    ax.set_title("128K best LR")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT, dpi=180)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
