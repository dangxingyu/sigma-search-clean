#!/usr/bin/env python3
"""Plot 2M-batch lr=0.04 training curves and the nearby LR sweep."""

from __future__ import annotations

import json
import math
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "debug_2m_lr004"
BATCH = 2_097_152


@dataclass(frozen=True)
class Run:
    method: str
    lr: float
    seed: int
    source: str
    case: str
    score: float
    path: Path
    val_bpbs: list[dict]
    train_losses: list[dict]


def method_from_case(case: str) -> str | None:
    if case.startswith("streaming_identity"):
        return "identity c=1"
    if case.startswith("top_aware_k1_a0p5"):
        return "Top-Aware c=0.5"
    return None


def load_runs() -> list[Run]:
    runs: list[Run] = []
    for path in sorted((ROOT / "search_evals").glob("v*/**/result.json")):
        data = json.loads(path.read_text())
        args = data.get("args", {})
        if data.get("score") is None or data.get("error") is not None:
            continue
        if int(args.get("depth", 0) or 0) != 8:
            continue
        if int(args.get("total_batch_size", 0) or 0) != BATCH:
            continue
        method = method_from_case(path.parent.name)
        if method is None:
            continue
        runs.append(
            Run(
                method=method,
                lr=float(args["matrix_lr"]),
                seed=int(args.get("seed", 0)),
                source=path.parents[1].name,
                case=path.parent.name,
                score=float(data["score"]),
                path=path,
                val_bpbs=list(data.get("val_bpbs") or []),
                train_losses=list(data.get("train_losses") or []),
            )
        )
    return runs


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def setup_plot_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 220,
            "font.size": 9.2,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.2,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_runs(runs: list[Run]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    setup_plot_style()

    methods = ["identity c=1", "Top-Aware c=0.5"]
    method_colors = {"identity c=1": "#1f4e79", "Top-Aware c=0.5": "#c75a21"}
    lr_colors = {
        0.01: "#8c8c8c",
        0.02: "#2b8cbe",
        0.04: "#d7301f",
        0.08: "#238b45",
        0.16: "#756bb1",
    }

    fig, axes = plt.subplots(2, 3, figsize=(14.4, 8.2), constrained_layout=True)
    fig.suptitle("2M batch LR=0.04 debug", fontsize=14, fontweight="bold")

    lr004 = [r for r in runs if abs(r.lr - 0.04) < 1e-12 and r.seed == 42]
    for run in sorted(lr004, key=lambda r: r.method):
        xs = [x["step"] for x in run.val_bpbs]
        ys = [x["val_bpb"] for x in run.val_bpbs]
        axes[0][0].plot(
            xs,
            ys,
            marker="o",
            linewidth=2.0,
            color=method_colors[run.method],
            label=f"{run.method}, final {run.score:.4f}",
        )
    axes[0][0].set_title("LR=0.04 validation BPB")
    axes[0][0].set_xlabel("optimizer step")
    axes[0][0].set_ylabel("validation BPB")
    axes[0][0].set_ylim(1.05, 1.85)
    axes[0][0].legend(frameon=False)

    for run in sorted(lr004, key=lambda r: r.method):
        xs = [x["step"] for x in run.train_losses]
        ys = [x["loss"] for x in run.train_losses]
        axes[0][1].plot(
            xs,
            ys,
            marker="o",
            linewidth=2.0,
            color=method_colors[run.method],
            label=run.method,
        )
    axes[0][1].set_title("LR=0.04 sparse train loss")
    axes[0][1].set_xlabel("optimizer step")
    axes[0][1].set_ylabel("train loss")
    axes[0][1].legend(frameon=False)

    by_method_lr: dict[tuple[str, float], list[Run]] = defaultdict(list)
    for run in runs:
        by_method_lr[(run.method, run.lr)].append(run)

    for method in methods:
        subset = [(lr, group) for (m, lr), group in by_method_lr.items() if m == method]
        for lr, group in sorted(subset):
            scores = [r.score for r in group]
            axes[0][2].scatter(
                [lr] * len(scores),
                scores,
                color=method_colors[method],
                alpha=0.35,
                s=28,
            )
            axes[0][2].plot(
                [lr],
                [mean(scores)],
                marker="o",
                color=method_colors[method],
                markersize=7,
            )
        xs = sorted({lr for lr, _group in subset})
        ys = [mean([r.score for r in by_method_lr[(method, lr)]]) for lr in xs]
        axes[0][2].plot(xs, ys, color=method_colors[method], linewidth=1.8, label=method)
    axes[0][2].set_xscale("log")
    axes[0][2].set_xticks(sorted({r.lr for r in runs}))
    axes[0][2].xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{x:g}"))
    axes[0][2].set_title("Final BPB vs LR, all completed 2M runs")
    axes[0][2].set_xlabel("base matrix LR")
    axes[0][2].set_ylabel("final validation BPB")
    axes[0][2].set_ylim(1.055, 1.225)
    axes[0][2].legend(frameon=False)

    for col, method in enumerate(methods):
        ax = axes[1][col]
        seed42 = [r for r in runs if r.method == method and r.seed == 42 and r.val_bpbs]
        # Prefer the original sweep for lr=0.08 if both original and metrics rerun exist.
        dedup: dict[float, Run] = {}
        for run in sorted(seed42, key=lambda r: (r.lr, "best_metrics" in r.source)):
            dedup.setdefault(run.lr, run)
        for lr, run in sorted(dedup.items()):
            xs = [x["step"] for x in run.val_bpbs]
            ys = [x["val_bpb"] for x in run.val_bpbs]
            ax.plot(
                xs,
                ys,
                marker="o" if abs(lr - 0.04) < 1e-12 else None,
                linewidth=2.4 if abs(lr - 0.04) < 1e-12 else 1.3,
                color=lr_colors.get(lr, "#555"),
                alpha=1.0 if abs(lr - 0.04) < 1e-12 else 0.72,
                label=f"lr={lr:g}, final {run.score:.4f}",
            )
        ax.set_title(f"{method}: seed-42 validation curves")
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("validation BPB")
        ax.set_ylim(1.05, 1.85 if method == "identity c=1" else 1.95)
        ax.legend(frameon=False, ncol=1)

    axes[1][2].axis("off")
    bullets = [
        "LR=0.04 is monotonic during training; it is not a run-level divergence.",
        "The jagged LR sweep is mainly a final-BPB ordering issue: identity lr=0.04 underperforms both lr=0.02 and lr=0.08 in the same seed-42 sweep.",
        "Final validation loss after a fixed token budget is not mathematically convex in LR; optimizer dynamics, warmup/decay schedule, stochasticity, and finite validation noise can create local reversals.",
        "Treat single-seed nonconvexity as a signal to add seeds around the minimum, not as a principled optimizer ranking.",
    ]
    text = "Readout\n" + "\n".join(
        f"{idx}. {textwrap.fill(item, width=46, subsequent_indent='   ')}"
        for idx, item in enumerate(bullets, start=1)
    )
    axes[1][2].text(0.0, 1.0, text, va="top", ha="left", family="monospace", fontsize=9)

    fig.savefig(OUT / "loss_curves_2m_lr004.png", bbox_inches="tight")
    fig.savefig(OUT / "loss_curves_2m_lr004.pdf", bbox_inches="tight")

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6), sharey=True, constrained_layout=True)
    fig.suptitle("2M batch sparse training loss by LR, seed 42", fontsize=14, fontweight="bold")
    for ax, method in zip(axes, methods):
        seed42 = [r for r in runs if r.method == method and r.seed == 42 and r.train_losses]
        dedup: dict[float, Run] = {}
        for run in sorted(seed42, key=lambda r: (r.lr, "best_metrics" in r.source)):
            dedup.setdefault(run.lr, run)
        for lr, run in sorted(dedup.items()):
            xs = [x["step"] for x in run.train_losses]
            ys = [x["loss"] for x in run.train_losses]
            ax.plot(
                xs,
                ys,
                marker="o" if abs(lr - 0.04) < 1e-12 else None,
                linewidth=2.6 if abs(lr - 0.04) < 1e-12 else 1.5,
                color=lr_colors.get(lr, "#555"),
                alpha=1.0 if abs(lr - 0.04) < 1e-12 else 0.72,
                label=f"lr={lr:g}, final BPB {run.score:.4f}",
            )
        ax.set_title(method)
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("train loss")
        ax.legend(frameon=False)
    fig.savefig(OUT / "train_loss_curves_2m_all_lrs.png", bbox_inches="tight")
    fig.savefig(OUT / "train_loss_curves_2m_all_lrs.pdf", bbox_inches="tight")


def write_table(runs: list[Run]) -> None:
    lines = [
        "# 2M LR=0.04 Debug",
        "",
        "![2M LR=0.04 loss curves](loss_curves_2m_lr004.png)",
        "",
        "![2M all-LR train loss curves](train_loss_curves_2m_all_lrs.png)",
        "",
        "| method | lr | seed | final BPB | source | case |",
        "|---|---:|---:|---:|---|---|",
    ]
    for run in sorted(runs, key=lambda r: (r.method, r.lr, r.seed, r.source)):
        lines.append(
            f"| {run.method} | {run.lr:g} | {run.seed} | {run.score:.6f} | {run.source} | `{run.case}` |"
        )
    lines.append("")
    (OUT / "README.md").write_text("\n".join(lines))


def main() -> None:
    runs = load_runs()
    if not runs:
        raise SystemExit("no completed 2M runs found")
    plot_runs(runs)
    write_table(runs)
    print(OUT / "loss_curves_2m_lr004.png")
    print(OUT / "README.md")


if __name__ == "__main__":
    main()
