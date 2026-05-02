#!/usr/bin/env python3
"""Analyze v26 same-driver small-batch StreamingMuon comparison."""

from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = ["identity", "lite_chi2_rs01", "top1pm_a05"]
METHOD_LABEL = {
    "identity": "identity / Muon-like",
    "lite_chi2_rs01": "LITE-like chi=2 rs=.1",
    "top1pm_a05": "top1 per-matrix alpha=.5",
}
COLORS = {
    "identity": "tab:blue",
    "lite_chi2_rs01": "tab:orange",
    "top1pm_a05": "tab:red",
}
PAT = re.compile(r"(.+)_bsz(\d+)_lr([0-9p]+)_s(\d+)$")


def lr_from_key(key: str) -> float:
    return float(key.replace("p", "."))


def batch_label(batch: int) -> str:
    if batch >= 1048576:
        return f"{batch // 1048576}M"
    return f"{batch // 1024}K"


def load_rows(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("*/result.json")):
        match = PAT.match(path.parent.name)
        if not match:
            continue
        method, batch_s, lr_s, seed_s = match.groups()
        data = json.loads(path.read_text())
        rows.append(
            {
                "name": path.parent.name,
                "method": method,
                "batch": int(batch_s),
                "lr": lr_from_key(lr_s),
                "seed": int(seed_s),
                "score": data.get("score"),
                "final": data.get("val_bpb_final"),
                "error": data.get("error"),
                "vals": data.get("val_bpbs", []),
                "args": data.get("args", {}),
                "path": str(path),
            }
        )
    return rows


def mean_sem(values: list[float]) -> tuple[float, float]:
    xs = np.asarray([x for x in values if x is not None and math.isfinite(float(x))], dtype=float)
    if len(xs) == 0:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return float(xs.mean()), float(xs.std(ddof=1) / math.sqrt(len(xs)))


def aggregate(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        if row["error"] is None and row["score"] is not None:
            groups[(row["method"], row["batch"], row["lr"])].append(float(row["score"]))
    out = []
    for (method, batch, lr), scores in sorted(groups.items(), key=lambda x: (x[0][1], x[0][0], x[0][2])):
        mean, sem = mean_sem(scores)
        out.append({"method": method, "batch": batch, "lr": lr, "n": len(scores), "mean": mean, "sem": sem})
    return out


def best_rows(agg: list[dict]) -> dict[tuple[int, str], dict]:
    out = {}
    for batch in sorted({r["batch"] for r in agg}):
        for method in METHOD_ORDER:
            rows = [r for r in agg if r["batch"] == batch and r["method"] == method]
            if rows:
                out[(batch, method)] = min(rows, key=lambda r: r["mean"])
    return out


def boundary_flags(agg: list[dict], best: dict[tuple[int, str], dict]) -> list[dict]:
    flags = []
    for key, row in best.items():
        batch, method = key
        lrs = sorted(r["lr"] for r in agg if r["batch"] == batch and r["method"] == method)
        if not lrs:
            continue
        if row["lr"] == lrs[0] or row["lr"] == lrs[-1]:
            flags.append({"batch": batch, "method": method, "best_lr": row["lr"], "tested_lrs": lrs})
    return flags


def print_tables(rows: list[dict], agg: list[dict], best: dict[tuple[int, str], dict], flags: list[dict]) -> None:
    print("method,batch,lr,seed,score,final,error")
    for row in sorted(rows, key=lambda r: (r["batch"], r["method"], r["lr"], r["seed"])):
        print(f"{row['method']},{row['batch']},{row['lr']:.5g},{row['seed']},{row['score']},{row['final']},{row['error']}")

    print("\nAggregate:")
    print("method,batch,lr,n,mean,sem")
    for row in agg:
        print(f"{row['method']},{row['batch']},{row['lr']:.5g},{row['n']},{row['mean']:.6f},{row['sem']:.6f}")

    print("\nBest by batch/method:")
    for batch in sorted({b for b, _ in best}):
        for method in METHOD_ORDER:
            row = best.get((batch, method))
            if row:
                print(f"{batch_label(batch):>4s} {method:16s} lr={row['lr']:.5g} mean={row['mean']:.6f} n={row['n']}")

    print("\nBest deltas:")
    for batch in sorted({b for b, _ in best}):
        ident = best.get((batch, "identity"))
        lite = best.get((batch, "lite_chi2_rs01"))
        top1 = best.get((batch, "top1pm_a05"))
        if ident and lite:
            print(f"{batch_label(batch):>4s} identity-lite = {ident['mean'] - lite['mean']:+.6f} (positive: LITE-like wins)")
        if ident and top1:
            print(f"{batch_label(batch):>4s} identity-top1 = {ident['mean'] - top1['mean']:+.6f} (positive: top1 wins)")
        if lite and top1:
            print(f"{batch_label(batch):>4s} lite-top1 = {lite['mean'] - top1['mean']:+.6f} (positive: top1 wins over LITE-like)")

    if flags:
        print("\nLR boundary flags:")
        for f in flags:
            print(f"{batch_label(f['batch'])} {f['method']} best_lr={f['best_lr']:.5g} tested={f['tested_lrs']}")


def plot_lr(root: Path, agg: list[dict]) -> Path:
    batches = sorted({r["batch"] for r in agg})
    if not batches:
        raise ValueError("no aggregate rows")
    fig, axes = plt.subplots(1, len(batches), figsize=(5.4 * len(batches), 4.2), squeeze=False)
    for ax, batch in zip(axes[0], batches):
        rows_b = [r for r in agg if r["batch"] == batch]
        for method in METHOD_ORDER:
            rows = sorted([r for r in rows_b if r["method"] == method], key=lambda r: r["lr"])
            if not rows:
                continue
            ax.errorbar(
                [r["lr"] for r in rows],
                [r["mean"] for r in rows],
                yerr=[r["sem"] for r in rows],
                marker="o",
                capsize=3,
                lw=2,
                color=COLORS[method],
                label=METHOD_LABEL[method],
            )
        vals = [r["mean"] for r in rows_b if math.isfinite(r["mean"])]
        if vals:
            lo, hi = min(vals), max(vals)
            pad = max(0.0015, 0.12 * (hi - lo))
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_xscale("log", base=2)
        ax.set_title(f"{batch_label(batch)} global batch, d8, ~1B tokens")
        ax.set_xlabel("matrix LR")
        ax.set_ylabel("best val BPB")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("v26 same-driver StreamingMuon LR sweep: identity vs LITE-like vs top1", y=1.02)
    fig.tight_layout()
    out = root / "v26_lr_sweep.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    return out


def plot_trajectories(root: Path, rows: list[dict], best: dict[tuple[int, str], dict]) -> Path | None:
    batches = sorted({b for b, _ in best})
    if not batches:
        return None
    fig, axes = plt.subplots(1, len(batches), figsize=(5.4 * len(batches), 4.2), squeeze=False)
    for ax, batch in zip(axes[0], batches):
        for method in METHOD_ORDER:
            b = best.get((batch, method))
            if not b:
                continue
            matches = [
                r for r in rows
                if r["batch"] == batch and r["method"] == method and abs(r["lr"] - b["lr"]) < 1e-12 and r["error"] is None
            ]
            if not matches:
                continue
            # For multi-seed, plot each seed faintly.
            for r in matches:
                vals = r["vals"]
                if not vals:
                    continue
                ax.plot(
                    [v["step"] for v in vals],
                    [v["val_bpb"] for v in vals],
                    marker="o",
                    lw=1.5,
                    alpha=0.8,
                    color=COLORS[method],
                    label=f"{METHOD_LABEL[method]} lr={b['lr']:.5g}" if r is matches[0] else None,
                )
        ax.set_title(f"best-LR trajectories, {batch_label(batch)}")
        ax.set_xlabel("step")
        ax.set_ylabel("val BPB")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    out = root / "v26_best_trajectories.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    return out


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("search_evals/v26_small_bsz_streaming_fair_latest")
    rows = load_rows(root)
    if not rows:
        raise SystemExit(f"no result rows under {root}")
    agg = aggregate(rows)
    best = best_rows(agg)
    flags = boundary_flags(agg, best)
    print_tables(rows, agg, best, flags)

    summary = {"rows": rows, "aggregate": agg, "best": {f"{b}:{m}": r for (b, m), r in best.items()}, "boundary_flags": flags}
    (root / "v26_summary.json").write_text(json.dumps(summary, indent=2))

    fig1 = plot_lr(root, agg)
    fig2 = plot_trajectories(root, rows, best)
    print(f"\nwrote {fig1}")
    if fig2:
        print(f"wrote {fig2}")
    print(f"wrote {root / 'v26_summary.json'}")


if __name__ == "__main__":
    main()
