#!/usr/bin/env python3
"""Summarize the recent d8 StreamingMuon/Top-Aware sweeps.

This intentionally ignores checkpoint/resume smoke tests and only reads the
main clean sweep directories v42-v58.
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "recent_d8_sweeps"
SOURCE_RE = re.compile(r"^v(4[2-9]|5[0-8])_")


def batch_label(batch: int) -> str:
    if batch == 262_144:
        return "262K"
    if batch % 1_048_576 == 0:
        return f"{batch // 1_048_576}M"
    if batch % 1024 == 0:
        return f"{batch // 1024}K"
    return str(batch)


def method_from_case(case_name: str, candidate_params: dict) -> tuple[str, float]:
    if case_name.startswith("top_aware"):
        alpha = candidate_params.get("alpha")
        if alpha is None:
            match = re.search(r"_a([0-9]+p[0-9]+|[0-9]+)", case_name)
            alpha = float(match.group(1).replace("p", ".")) if match else 0.5
        return "top_aware_muon", float(alpha)
    if case_name.startswith("streaming_identity"):
        return "streaming_identity", 1.0
    return case_name.split("_bsz", 1)[0], float("nan")


def read_rows() -> list[dict]:
    rows: list[dict] = []
    for source in sorted((ROOT / "search_evals").iterdir()):
        if not source.is_dir() or not SOURCE_RE.match(source.name):
            continue
        for result_path in sorted(source.glob("*/result.json")):
            data = json.loads(result_path.read_text())
            args = data.get("args", {})
            score = data.get("score")
            error = data.get("error")
            if score is None or error is not None:
                continue
            method, alpha = method_from_case(result_path.parent.name, data.get("candidate_params", {}))
            if method not in {"streaming_identity", "top_aware_muon"}:
                continue
            batch = int(args["total_batch_size"])
            max_steps = int(args["max_steps"])
            tokens = batch * max_steps
            if int(args.get("depth", 0)) != 8:
                continue
            if not (390_000_000 <= tokens <= 410_000_000):
                continue
            rows.append({
                "source": source.name,
                "case": result_path.parent.name,
                "method": method,
                "alpha": alpha,
                "batch": batch,
                "batch_label": batch_label(batch),
                "lr": float(args["matrix_lr"]),
                "seed": int(args.get("seed", 0)),
                "score": float(score),
                "val_bpb_final": float(data.get("val_bpb_final", score)),
                "steps_completed": int(data.get("steps_completed", max_steps)),
                "tokens": tokens,
                "path": str(result_path.relative_to(ROOT)),
            })
    return rows


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def best_by_mean_lr(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["batch"], row["method"], row["alpha"], row["lr"])
        grouped[key].append(row)

    lr_rows = []
    for (batch, method, alpha, lr), group in grouped.items():
        scores = [r["score"] for r in group]
        mean, std = mean_std(scores)
        lr_rows.append({
            "batch": batch,
            "batch_label": batch_label(batch),
            "method": method,
            "alpha": alpha,
            "lr": lr,
            "n": len(scores),
            "seeds": " ".join(str(r["seed"]) for r in sorted(group, key=lambda x: x["seed"])),
            "mean_score": mean,
            "std_score": std,
            "min_score": min(scores),
            "max_score": max(scores),
        })

    best = {}
    for row in lr_rows:
        key = (row["batch"], row["method"], row["alpha"])
        if key not in best or row["mean_score"] < best[key]["mean_score"]:
            best[key] = row
    return sorted(best.values(), key=lambda r: (r["batch"], r["method"], r["alpha"]))


def paired_summary(best: list[dict]) -> list[dict]:
    by_key = {(r["batch"], r["method"]): r for r in best}
    out = []
    for batch in sorted({r["batch"] for r in best}):
        identity = by_key.get((batch, "streaming_identity"))
        top = by_key.get((batch, "top_aware_muon"))
        if identity is None or top is None:
            continue
        delta = top["mean_score"] - identity["mean_score"]
        out.append({
            "batch": batch,
            "batch_label": batch_label(batch),
            "identity_best_lr": identity["lr"],
            "identity_mean_bpb": identity["mean_score"],
            "identity_n": identity["n"],
            "topaware_best_lr": top["lr"],
            "topaware_mean_bpb": top["mean_score"],
            "topaware_n": top["n"],
            "delta_topaware_minus_identity": delta,
            "winner": "top_aware_muon" if delta < 0 else "streaming_identity",
        })
    return out


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def plot_lr(rows: list[dict]) -> None:
    batches = sorted({r["batch"] for r in rows})
    fig, axes = plt.subplots(1, len(batches), figsize=(4.2 * len(batches), 3.6), squeeze=False)
    colors = {"streaming_identity": "#1f77b4", "top_aware_muon": "#d95f02"}
    labels = {"streaming_identity": "StreamingMuon identity", "top_aware_muon": "Top-Aware alpha=0.5"}

    for ax, batch in zip(axes[0], batches):
        subset = [r for r in rows if r["batch"] == batch]
        for method in ["streaming_identity", "top_aware_muon"]:
            method_rows = [r for r in subset if r["method"] == method]
            if not method_rows:
                continue
            by_lr: dict[float, list[float]] = defaultdict(list)
            for row in method_rows:
                by_lr[row["lr"]].append(row["score"])
                ax.scatter(row["lr"], row["score"], s=18, color=colors[method], alpha=0.35)
            xs, ys, yerr = [], [], []
            for lr in sorted(by_lr):
                mean, std = mean_std(by_lr[lr])
                xs.append(lr)
                ys.append(mean)
                yerr.append(std if len(by_lr[lr]) > 1 else 0.0)
            ax.errorbar(xs, ys, yerr=yerr, marker="o", linewidth=1.6, capsize=2.5,
                        color=colors[method], label=labels[method])
        scores = [r["score"] for r in subset]
        lrs = sorted({r["lr"] for r in subset})
        ymin, ymax = min(scores), max(scores)
        pad = max(0.01, (ymax - ymin) * 0.12)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_xscale("log")
        ax.set_xticks(lrs)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{x:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="x", labelrotation=35)
        ax.set_title(batch_label(batch))
        ax.set_xlabel("base matrix LR")
        ax.grid(True, alpha=0.25)
    axes[0][0].set_ylabel("best validation BPB")
    handles, labels_ = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Recent d8 0.4B-token LR sweeps", y=0.98)
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    fig.savefig(OUT / "lr_sweep_by_batch.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT / "lr_sweep_by_batch.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_delta(paired: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    xs = [r["batch_label"] for r in paired]
    ys = [r["delta_topaware_minus_identity"] for r in paired]
    colors = ["#d95f02" if y < 0 else "#1f77b4" for y in ys]
    ax.bar(xs, ys, color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Top-Aware BPB - identity BPB")
    ax.set_xlabel("global batch size")
    ax.set_title("Best-mean LR comparison by batch")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "best_delta_by_batch.png", dpi=180)
    fig.savefig(OUT / "best_delta_by_batch.pdf")
    plt.close(fig)


def write_markdown(rows: list[dict], best: list[dict], paired: list[dict]) -> None:
    lines = [
        "# Recent d8 Sweep Summary",
        "",
        "Scope: clean d8 StreamingMuon identity vs Top-Aware Muon `top_k=1, alpha=0.5`, 0.4B-token recipe, sources `v42` through `v58` only. Lower BPB is better.",
        "",
        "Generated files:",
        "- `all_runs.csv`: every completed result row used.",
        "- `best_by_mean_lr.csv`: best LR per `(batch, method)` after averaging available seeds at that LR.",
        "- `paired_best_summary.csv`: identity vs Top-Aware comparison by batch.",
        "- `lr_sweep_by_batch.png`: LR curves with seed scatter and mean/error bars.",
        "- `best_delta_by_batch.png`: best Top-Aware minus identity BPB.",
        "",
        "## Main Table",
        "",
        "| batch | identity best LR | identity BPB | n | Top-Aware best LR | Top-Aware BPB | n | delta | winner |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in paired:
        lines.append(
            f"| {r['batch_label']} | {r['identity_best_lr']:.4g} | {r['identity_mean_bpb']:.6f} | {r['identity_n']} | "
            f"{r['topaware_best_lr']:.4g} | {r['topaware_mean_bpb']:.6f} | {r['topaware_n']} | "
            f"{r['delta_topaware_minus_identity']:+.6f} | {r['winner']} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- Top-Aware `alpha=0.5` is not uniformly better across the currently swept 0.4B-token d8 recipe.",
        "- The strongest current positive signal is at very large batch (`4M`, `8M`), where Top-Aware wins in the best-mean comparison.",
        "- At `262K`, `1M`, and `2M`, the identity baseline is at least competitive and currently ahead under the best-mean-LR summary.",
        "- Some rows have different seed counts per LR because follow-up boundary/seed-confirm runs were added adaptively; inspect `all_runs.csv` before treating small deltas as final.",
        "",
        "## Figures",
        "",
        "![LR sweep by batch](lr_sweep_by_batch.png)",
        "",
        "![Best delta by batch](best_delta_by_batch.png)",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = sorted(read_rows(), key=lambda r: (r["batch"], r["method"], r["lr"], r["seed"], r["source"]))
    best = best_by_mean_lr(rows)
    paired = paired_summary(best)

    write_csv(OUT / "all_runs.csv", rows, [
        "source", "case", "method", "alpha", "batch", "batch_label", "lr", "seed",
        "score", "val_bpb_final", "steps_completed", "tokens", "path",
    ])
    write_csv(OUT / "best_by_mean_lr.csv", best, [
        "batch", "batch_label", "method", "alpha", "lr", "n", "seeds",
        "mean_score", "std_score", "min_score", "max_score",
    ])
    write_csv(OUT / "paired_best_summary.csv", paired, [
        "batch", "batch_label", "identity_best_lr", "identity_mean_bpb", "identity_n",
        "topaware_best_lr", "topaware_mean_bpb", "topaware_n",
        "delta_topaware_minus_identity", "winner",
    ])
    plot_lr(rows)
    plot_delta(paired)
    write_markdown(rows, best, paired)

    summary = {
        "row_count": len(rows),
        "sources": sorted({r["source"] for r in rows}),
        "batches": [batch_label(b) for b in sorted({r["batch"] for r in rows})],
        "paired_summary": paired,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)} with {len(rows)} rows")


if __name__ == "__main__":
    main()
