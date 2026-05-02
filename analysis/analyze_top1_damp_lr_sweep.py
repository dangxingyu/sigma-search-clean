#!/usr/bin/env python3
"""Analyze top-1 damping LR sweep for StreamingMuon."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt


BATCH_LABELS = {
    131072: "128K",
    1048576: "1M",
    8388608: "8M",
}


def parse_name(name: str) -> tuple[str, int, float] | None:
    parts = name.split("_bsz")
    if len(parts) != 2:
        return None
    method = parts[0]
    bsz_s, lr_s = parts[1].split("_lr")
    return method, int(bsz_s), float(lr_s.replace("p", "."))


def load(root: Path) -> list[dict]:
    rows = []
    for p in sorted(root.glob("*/result.json")):
        parsed = parse_name(p.parent.name)
        if parsed is None:
            continue
        method, bsz, lr = parsed
        data = json.loads(p.read_text())
        rows.append(
            {
                "name": p.parent.name,
                "method": method,
                "bsz": bsz,
                "lr": lr,
                "score": data.get("score"),
                "final": data.get("val_bpb_final"),
                "error": data.get("error"),
                "vals": data.get("val_bpbs", []),
                "args": data.get("args", {}),
            }
        )
    return rows


def finite_score(row: dict) -> float:
    score = row["score"]
    if score is None or not math.isfinite(score):
        return float("inf")
    return score


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("search_evals/ddp8_top1_damp_lr_sweep_latest")
    rows = load(root)
    if not rows:
        raise SystemExit(f"no result rows under {root}")

    methods = ["identity", "top1_damp_a05"]
    batches = [131072, 1048576, 8388608]

    print("method,batch,lr,score,final,error")
    for r in sorted(rows, key=lambda x: (x["bsz"], x["method"], x["lr"])):
        print(
            f"{r['method']},{r['bsz']},{r['lr']:.5g},"
            f"{r['score']},{r['final']},{r['error']}"
        )

    print("\nBest by batch/method:")
    best = {}
    for bsz in batches:
        for method in methods:
            subset = [r for r in rows if r["bsz"] == bsz and r["method"] == method]
            if not subset:
                continue
            b = min(subset, key=finite_score)
            best[(bsz, method)] = b
            print(
                f"{BATCH_LABELS[bsz]:>4s} {method:16s} "
                f"lr={b['lr']:.5g} score={b['score']} final={b['final']} error={b['error']}"
            )

    print("\nBest deltas: identity - top1_damp_a05")
    for bsz in batches:
        i = best.get((bsz, "identity"))
        t = best.get((bsz, "top1_damp_a05"))
        if i and t and i["score"] is not None and t["score"] is not None:
            print(
                f"{BATCH_LABELS[bsz]:>4s}: "
                f"identity={i['score']:.6f}@{i['lr']:.5g}, "
                f"top1={t['score']:.6f}@{t['lr']:.5g}, "
                f"delta={i['score'] - t['score']:+.6f}"
            )

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), sharey=False)
    colors = {"identity": "#1f77b4", "top1_damp_a05": "#d62728"}
    labels = {"identity": "identity", "top1_damp_a05": "top-1 damp α=.5"}
    for ax, bsz in zip(axes, batches):
        for method in methods:
            subset = sorted(
                [r for r in rows if r["bsz"] == bsz and r["method"] == method],
                key=lambda r: r["lr"],
            )
            if not subset:
                continue
            xs = [r["lr"] for r in subset]
            ys = [r["score"] if r["score"] is not None else float("nan") for r in subset]
            ax.plot(xs, ys, marker="o", lw=2.2, color=colors[method], label=labels[method])
        ax.set_xscale("log", base=2)
        ax.set_title(f"batch {BATCH_LABELS[bsz]}")
        ax.set_xlabel("matrix LR")
        ax.set_ylabel("best val BPB")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("StreamingMuon d8 / 1.07B tokens: top-1 sharp-direction damping LR sweep", y=1.03)
    fig.tight_layout()
    out = Path("top1_damp_lr_sweep_analysis.png")
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"\nwrote {out}")

    fig2, axes2 = plt.subplots(1, 3, figsize=(14.5, 4.2), sharey=False)
    for ax, bsz in zip(axes2, batches):
        for method in methods:
            r = best.get((bsz, method))
            if not r:
                continue
            vals = r["vals"]
            ax.plot(
                [v["step"] for v in vals],
                [v["val_bpb"] for v in vals],
                marker="o",
                lw=2.1,
                color=colors[method],
                label=f"{labels[method]} lr={r['lr']:.5g}",
            )
        ax.set_title(f"best trajectory, batch {BATCH_LABELS[bsz]}")
        ax.set_xlabel("step")
        ax.set_ylabel("val BPB")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig2.tight_layout()
    out2 = Path("top1_damp_lr_sweep_trajectories.png")
    fig2.savefig(out2, dpi=170, bbox_inches="tight")
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
