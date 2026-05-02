#!/usr/bin/env python3
"""Dashboard for historical large-batch Muon/LITE and top1 experiments."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path("search_evals/historical_large_batch_dashboard")
NATIVE_ROOTS = [
    Path("sweep_results_v18_128k_clean"),
    Path("sweep_results_v13_256k_clean"),
    Path("sweep_results_v9"),
]
STREAM_ROOT = Path("search_evals/ddp8_top1_damp_adaptive_20260429_171235")


def bsz_label(bsz: int) -> str:
    if bsz >= 1048576:
        return f"{bsz // 1048576}M"
    return f"{bsz // 1024}K"


def score(data: dict) -> float | None:
    return data.get("score", data.get("val_bpb_best"))


def mean_sem(vals: list[float]) -> tuple[float, float]:
    xs = np.asarray([v for v in vals if v is not None and math.isfinite(float(v))], dtype=float)
    if len(xs) == 0:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return float(xs.mean()), float(xs.std(ddof=1) / math.sqrt(len(xs)))


def load_native() -> list[dict]:
    rows = []
    pat = re.compile(r"(muon|lite)_bsz(\d+)_lr([\d.]+)_s(\d+)\.json$")
    for root in NATIVE_ROOTS:
        for path in sorted(root.glob("*.json")):
            match = pat.match(path.name)
            if not match:
                continue
            opt, bsz_s, lr_s, seed_s = match.groups()
            data = json.loads(path.read_text())
            if data.get("error") is not None:
                continue
            rows.append(
                {
                    "family": "native_single_gpu",
                    "root": str(root),
                    "optimizer": opt,
                    "batch": int(bsz_s),
                    "batch_label": bsz_label(int(bsz_s)),
                    "lr": float(lr_s),
                    "seed": int(seed_s),
                    "score": score(data),
                    "path": str(path),
                }
            )
    return rows


def load_streaming() -> list[dict]:
    rows = []
    pat = re.compile(r"(.+)_bsz(\d+)_lr([0-9p]+)$")
    for path in sorted(STREAM_ROOT.glob("*/result.json")):
        match = pat.match(path.parent.name)
        if not match:
            continue
        method, bsz_s, lr_s = match.groups()
        data = json.loads(path.read_text())
        if data.get("error") is not None:
            continue
        rows.append(
            {
                "family": "streaming_ddp_top1_old",
                "root": str(STREAM_ROOT),
                "optimizer": method,
                "batch": int(bsz_s),
                "batch_label": bsz_label(int(bsz_s)),
                "lr": float(lr_s.replace("p", ".")),
                "seed": data.get("args", {}).get("seed", 42),
                "score": data.get("score"),
                "path": str(path),
            }
        )
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["family"], row["batch"], row["optimizer"], row["lr"])].append(float(row["score"]))
    out = []
    for (family, batch, optimizer, lr), vals in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3])):
        mean, sem = mean_sem(vals)
        out.append(
            {
                "family": family,
                "batch": batch,
                "batch_label": bsz_label(batch),
                "optimizer": optimizer,
                "lr": lr,
                "n": len(vals),
                "mean": mean,
                "sem": sem,
            }
        )
    return out


def best_by_family(agg: list[dict]) -> dict[tuple[str, int, str], dict]:
    best = {}
    for row in agg:
        key = (row["family"], row["batch"], row["optimizer"])
        if key not in best or row["mean"] < best[key]["mean"]:
            best[key] = row
    return best


def write_csv(rows: list[dict], agg: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "historical_large_batch_rows.csv").open("w", newline="") as f:
        fields = ["family", "batch_label", "batch", "optimizer", "lr", "seed", "score", "path"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})
    with (OUT_DIR / "historical_large_batch_aggregate.csv").open("w", newline="") as f:
        fields = ["family", "batch_label", "batch", "optimizer", "lr", "n", "mean", "sem"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in agg:
            writer.writerow({k: row.get(k) for k in fields})


def fmt(x: float | None, digits: int = 6) -> str:
    if x is None:
        return ""
    if not math.isfinite(float(x)):
        return ""
    return f"{float(x):.{digits}f}"


def write_md(native_rows: list[dict], streaming_rows: list[dict], agg: list[dict], best: dict[tuple[str, int, str], dict]) -> None:
    lines = [
        "# Historical Large-Batch Dashboard",
        "",
        "This dashboard separates experiment families. Do not raw-compare BPB between `native_single_gpu` and `streaming_ddp_top1_old`; data/eval streams differ.",
        "",
        "## Native Single-GPU Muon vs LITE Best Rows",
        "",
        "| batch | Muon best | LITE best | Muon - LITE | winner |",
        "|---:|---:|---:|---:|---|",
    ]
    native_batches = sorted({r["batch"] for r in agg if r["family"] == "native_single_gpu"})
    for batch in native_batches:
        mu = best.get(("native_single_gpu", batch, "muon"))
        li = best.get(("native_single_gpu", batch, "lite"))
        if not (mu and li):
            continue
        delta = mu["mean"] - li["mean"]
        winner = "LITE" if delta > 0 else "Muon/tie"
        lines.append(
            f"| {bsz_label(batch)} | `{fmt(mu['mean'])} @ lr={mu['lr']:g}` | "
            f"`{fmt(li['mean'])} @ lr={li['lr']:g}` | `{delta:+.6f}` | {winner} |"
        )

    lines += [
        "",
        "## Streaming DDP Identity vs Top1 Best Rows",
        "",
        "| batch | identity best | top1 best | identity - top1 | winner |",
        "|---:|---:|---:|---:|---|",
    ]
    stream_batches = sorted({r["batch"] for r in agg if r["family"] == "streaming_ddp_top1_old"})
    for batch in stream_batches:
        ident = best.get(("streaming_ddp_top1_old", batch, "identity"))
        top1 = best.get(("streaming_ddp_top1_old", batch, "top1_damp_a05"))
        if not (ident and top1):
            continue
        delta = ident["mean"] - top1["mean"]
        winner = "top1" if delta > 0 else "identity/tie"
        lines.append(
            f"| {bsz_label(batch)} | `{fmt(ident['mean'])} @ lr={ident['lr']:g}` | "
            f"`{fmt(top1['mean'])} @ lr={top1['lr']:g}` | `{delta:+.6f}` | {winner} |"
        )

    lines += [
        "",
        "## All Aggregate Rows",
        "",
        "| family | batch | optimizer | LR | n | mean BPB | SEM |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in agg:
        lines.append(
            f"| `{row['family']}` | {row['batch_label']} | `{row['optimizer']}` | `{row['lr']:g}` | "
            f"{row['n']} | `{fmt(row['mean'])}` | `{fmt(row['sem'])}` |"
        )

    (OUT_DIR / "historical_large_batch_dashboard.md").write_text("\n".join(lines) + "\n")


def plot_native(agg: list[dict], best: dict[tuple[str, int, str], dict]) -> None:
    batches = sorted({r["batch"] for r in agg if r["family"] == "native_single_gpu"})
    xs = np.arange(len(batches))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    width = 0.36
    mu = [best[("native_single_gpu", b, "muon")]["mean"] for b in batches]
    li = [best[("native_single_gpu", b, "lite")]["mean"] for b in batches]
    axes[0].bar(xs - width / 2, mu, width, label="Muon", color="tab:blue")
    axes[0].bar(xs + width / 2, li, width, label="LITE", color="tab:orange")
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels([bsz_label(b) for b in batches], rotation=30)
    axes[0].set_ylabel("best val BPB")
    axes[0].set_title("Native single-GPU best LR")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend(frameon=False)
    deltas = [m - l for m, l in zip(mu, li)]
    axes[1].axhline(0, color="black", lw=1)
    axes[1].plot(xs, deltas, marker="o", color="tab:green")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([bsz_label(b) for b in batches], rotation=30)
    axes[1].set_ylabel("Muon - LITE BPB")
    axes[1].set_title("Positive means LITE wins")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "historical_native_muon_lite.png", dpi=180)


def plot_streaming_lr(agg: list[dict]) -> None:
    batches = sorted({r["batch"] for r in agg if r["family"] == "streaming_ddp_top1_old"})
    fig, axes = plt.subplots(1, len(batches), figsize=(5.2 * len(batches), 4.1), squeeze=False)
    colors = {"identity": "tab:blue", "top1_damp_a05": "tab:red"}
    for ax, batch in zip(axes[0], batches):
        for method in ["identity", "top1_damp_a05"]:
            rows = sorted(
                [r for r in agg if r["family"] == "streaming_ddp_top1_old" and r["batch"] == batch and r["optimizer"] == method],
                key=lambda r: r["lr"],
            )
            if not rows:
                continue
            ax.plot([r["lr"] for r in rows], [r["mean"] for r in rows], marker="o", lw=2, color=colors[method], label=method)
            b = min(rows, key=lambda r: r["mean"])
            ax.scatter([b["lr"]], [b["mean"]], s=80, facecolors="none", edgecolors=colors[method], linewidths=2)
        vals = [r["mean"] for r in agg if r["family"] == "streaming_ddp_top1_old" and r["batch"] == batch]
        if vals:
            lo, hi = min(vals), max(vals)
            ax.set_ylim(lo - max(0.002, 0.12 * (hi - lo)), hi + max(0.002, 0.12 * (hi - lo)))
        ax.set_xscale("log", base=2)
        ax.set_title(bsz_label(batch))
        ax.set_xlabel("matrix LR")
        ax.set_ylabel("best val BPB")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "historical_streaming_top1_lr_sweep.png", dpi=180)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    native = load_native()
    streaming = load_streaming()
    rows = native + streaming
    agg = aggregate(rows)
    best = best_by_family(agg)
    write_csv(rows, agg)
    write_md(native, streaming, agg, best)
    plot_native(agg, best)
    plot_streaming_lr(agg)
    (OUT_DIR / "historical_large_batch_summary.json").write_text(json.dumps({"rows": rows, "aggregate": agg, "best": {f"{a}:{b}:{c}": v for (a, b, c), v in best.items()}}, indent=2))
    print(f"wrote {OUT_DIR / 'historical_large_batch_dashboard.md'}")
    print(f"wrote {OUT_DIR / 'historical_native_muon_lite.png'}")
    print(f"wrote {OUT_DIR / 'historical_streaming_top1_lr_sweep.png'}")
    print(f"wrote {OUT_DIR / 'historical_large_batch_aggregate.csv'}")


if __name__ == "__main__":
    main()
