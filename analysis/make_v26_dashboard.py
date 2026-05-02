#!/usr/bin/env python3
"""Build a Markdown table + LR-sweep dashboard for v26/v26b results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PAT = re.compile(r"(.+)_bsz(\d+)_lr([0-9p]+)_s(\d+)$")
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


def batch_label(batch: int) -> str:
    if batch >= 1048576:
        return f"{batch // 1048576}M"
    return f"{batch // 1024}K"


def lr_from_key(key: str) -> float:
    return float(key.replace("p", "."))


def load_rows(roots: list[Path]) -> list[dict]:
    rows = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.glob("*/result.json")):
            match = PAT.match(path.parent.name)
            if not match:
                continue
            method, batch_s, lr_s, seed_s = match.groups()
            data = json.loads(path.read_text())
            args = data.get("args", {})
            rows.append(
                {
                    "root": str(root),
                    "name": path.parent.name,
                    "method": method,
                    "method_label": METHOD_LABEL.get(method, method),
                    "batch": int(batch_s),
                    "batch_label": batch_label(int(batch_s)),
                    "lr": lr_from_key(lr_s),
                    "seed": int(seed_s),
                    "score": data.get("score"),
                    "final": data.get("val_bpb_final"),
                    "error": data.get("error"),
                    "steps": data.get("steps_completed"),
                    "time_min": (data.get("eval_time_seconds") or 0.0) / 60.0,
                    "depth": args.get("depth"),
                    "tokens": (args.get("total_batch_size") or 0) * (args.get("max_steps") or 0),
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
    grouped = defaultdict(list)
    for row in rows:
        if row["error"] is None and row["score"] is not None:
            grouped[(row["batch"], row["method"], row["lr"])].append(float(row["score"]))
    out = []
    for (batch, method, lr), vals in sorted(grouped.items(), key=lambda x: (x[0][0], METHOD_ORDER.index(x[0][1]) if x[0][1] in METHOD_ORDER else 99, x[0][2])):
        mean, sem = mean_sem(vals)
        out.append(
            {
                "batch": batch,
                "batch_label": batch_label(batch),
                "method": method,
                "method_label": METHOD_LABEL.get(method, method),
                "lr": lr,
                "n": len(vals),
                "mean": mean,
                "sem": sem,
            }
        )
    return out


def best_rows(agg: list[dict]) -> dict[tuple[int, str], dict]:
    best = {}
    for batch in sorted({r["batch"] for r in agg}):
        for method in METHOD_ORDER:
            rows = [r for r in agg if r["batch"] == batch and r["method"] == method]
            if rows:
                best[(batch, method)] = min(rows, key=lambda r: r["mean"])
    return best


def write_csv(path: Path, rows: list[dict], agg: list[dict]) -> None:
    with path.open("w", newline="") as f:
        fields = ["batch_label", "batch", "method", "method_label", "lr", "seed", "score", "final", "error", "steps", "time_min", "path"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["batch"], METHOD_ORDER.index(r["method"]) if r["method"] in METHOD_ORDER else 99, r["lr"], r["seed"])):
            writer.writerow({k: row.get(k) for k in fields})

    agg_path = path.with_name(path.stem + "_aggregate.csv")
    with agg_path.open("w", newline="") as f:
        fields = ["batch_label", "batch", "method", "method_label", "lr", "n", "mean", "sem"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in agg:
            writer.writerow({k: row.get(k) for k in fields})


def fmt(x: float | None, digits: int = 6) -> str:
    if x is None:
        return ""
    try:
        if not math.isfinite(float(x)):
            return ""
    except Exception:
        return str(x)
    return f"{float(x):.{digits}f}"


def write_md(path: Path, rows: list[dict], agg: list[dict], best: dict[tuple[int, str], dict]) -> None:
    lines = [
        "# v26/v26b Optimizer LR Sweep Dashboard",
        "",
        "This is an auto-generated snapshot of completed JSONs only. In-progress runs are not included until their `result.json` exists.",
        "",
        "## Best LR By Batch And Method",
        "",
        "| batch | method | best LR | best BPB | n |",
        "|---:|---|---:|---:|---:|",
    ]
    for batch in sorted({b for b, _ in best}):
        for method in METHOD_ORDER:
            row = best.get((batch, method))
            if row is None:
                continue
            lines.append(
                f"| {batch_label(batch)} | `{method}` | `{row['lr']:g}` | `{fmt(row['mean'])}` | {row['n']} |"
            )

    lines += [
        "",
        "## Best-LR Deltas",
        "",
        "Positive `identity - X` means `X` beats identity. Positive `lite - top1` means top1 beats LITE-like.",
        "",
        "| batch | identity - LITE-like | identity - top1 | LITE-like - top1 |",
        "|---:|---:|---:|---:|",
    ]
    for batch in sorted({b for b, _ in best}):
        ident = best.get((batch, "identity"))
        lite = best.get((batch, "lite_chi2_rs01"))
        top1 = best.get((batch, "top1pm_a05"))
        d_il = ident["mean"] - lite["mean"] if ident and lite else None
        d_it = ident["mean"] - top1["mean"] if ident and top1 else None
        d_lt = lite["mean"] - top1["mean"] if lite and top1 else None
        lines.append(f"| {batch_label(batch)} | `{fmt(d_il)}` | `{fmt(d_it)}` | `{fmt(d_lt)}` |")

    lines += [
        "",
        "## All Completed LR Rows",
        "",
        "| batch | method | LR | seed | BPB | final BPB | minutes |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda r: (r["batch"], METHOD_ORDER.index(r["method"]) if r["method"] in METHOD_ORDER else 99, r["lr"], r["seed"])):
        lines.append(
            f"| {row['batch_label']} | `{row['method']}` | `{row['lr']:g}` | {row['seed']} | "
            f"`{fmt(row['score'])}` | `{fmt(row['final'])}` | {row['time_min']:.1f} |"
        )

    lines += [
        "",
        "## Aggregate LR Table",
        "",
        "| batch | method | LR | n | mean BPB | SEM |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in agg:
        lines.append(
            f"| {row['batch_label']} | `{row['method']}` | `{row['lr']:g}` | {row['n']} | "
            f"`{fmt(row['mean'])}` | `{fmt(row['sem'])}` |"
        )

    path.write_text("\n".join(lines) + "\n")


def plot(path: Path, agg: list[dict]) -> None:
    batches = sorted({r["batch"] for r in agg})
    if not batches:
        return
    fig, axes = plt.subplots(1, len(batches), figsize=(5.4 * len(batches), 4.4), squeeze=False)
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
                linewidth=2,
                color=COLORS.get(method),
                label=METHOD_LABEL.get(method, method),
            )
            best = min(rows, key=lambda r: r["mean"])
            ax.scatter([best["lr"]], [best["mean"]], s=90, facecolors="none", edgecolors=COLORS.get(method), linewidths=2.2)
        vals = [r["mean"] for r in rows_b if math.isfinite(r["mean"])]
        if vals:
            lo, hi = min(vals), max(vals)
            pad = max(0.0015, 0.12 * (hi - lo))
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_xscale("log", base=2)
        ax.set_title(f"{batch_label(batch)}")
        ax.set_xlabel("matrix LR")
        ax.set_ylabel("best validation BPB")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("v26/v26b same-driver StreamingMuon LR sweep", y=1.03)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="*", type=Path, default=[
        Path("search_evals/v26_small_bsz_streaming_fair_20260430"),
        Path("search_evals/v26b_large_bsz_streaming_fair_20260430"),
    ])
    parser.add_argument("--out-dir", type=Path, default=Path("search_evals/v26_small_bsz_streaming_fair_20260430"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.roots)
    if not rows:
        raise SystemExit("no completed rows found")
    agg = aggregate(rows)
    best = best_rows(agg)

    md = args.out_dir / "v26_dashboard_table.md"
    csv_path = args.out_dir / "v26_dashboard_rows.csv"
    fig = args.out_dir / "v26_dashboard_lr_sweep.png"
    summary = args.out_dir / "v26_dashboard_summary.json"

    write_md(md, rows, agg, best)
    write_csv(csv_path, rows, agg)
    plot(fig, agg)
    summary.write_text(json.dumps({"rows": rows, "aggregate": agg, "best": {f"{b}:{m}": r for (b, m), r in best.items()}}, indent=2))

    print(f"wrote {md}")
    print(f"wrote {csv_path}")
    print(f"wrote {csv_path.with_name(csv_path.stem + '_aggregate.csv')}")
    print(f"wrote {fig}")
    print(f"wrote {summary}")


if __name__ == "__main__":
    main()
