#!/usr/bin/env python3
"""Regenerate the consolidated d12/d16 sweep figures.

The sweep CSVs keep both normalized progress and raw optimizer step. Plots use
optimizer step because normalized progress hides the different step counts at
different batch sizes.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "d12-d16-sweep"
FIG = OUT / "figures"

DEPTHS = ["d12", "d16"]
BATCHES = ["512K", "2M", "8M"]
ALPHAS = ["1.0", "0.5"]
COLORS = {"1.0": "#1f4e79", "0.5": "#c75a21"}
LABELS = {"1.0": "c=1", "0.5": "c=0.5"}


def finite(value: object) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def read_csv(name: str) -> list[dict[str, str]]:
    with (OUT / name).open(newline="") as f:
        return list(csv.DictReader(f))


def setup_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 220,
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}.png", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def set_tight_ylim(ax: plt.Axes, values: list[float]) -> None:
    values = [v for v in values if math.isfinite(v)]
    if not values:
        return
    lo, hi = min(values), max(values)
    pad = max((hi - lo) * 0.15, 0.001) if hi > lo else max(abs(lo) * 0.01, 0.001)
    ax.set_ylim(lo - pad, hi + pad)


def step_formatter(x: float, _pos: object) -> str:
    if x >= 1000:
        return f"{x / 1000:.0f}k"
    return f"{int(x)}"


def plot_lr_sweeps(rows: list[dict[str, str]]) -> None:
    by_panel: dict[tuple[str, str, str], list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        score = finite(row.get("score"))
        lr = finite(row.get("lr"))
        if score is None or lr is None:
            continue
        by_panel[(row["depth"], row["batch_label"], row["alpha"])].append((lr, score))

    fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.1), constrained_layout=True)
    for r, depth in enumerate(DEPTHS):
        for c, batch in enumerate(BATCHES):
            ax = axes[r][c]
            ys_panel: list[float] = []
            for alpha in ALPHAS:
                points = sorted(by_panel.get((depth, batch, alpha), []))
                if not points:
                    continue
                xs, ys = zip(*points)
                ys_panel.extend(ys)
                ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=3.8, color=COLORS[alpha], label=LABELS[alpha])
            ax.set_title(f"{depth} · {batch}")
            ax.set_xlabel("matrix LR")
            if c == 0:
                ax.set_ylabel("validation BPB")
            set_tight_ylim(ax, ys_panel)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=2, frameon=False)
    save(fig, "d12_d16_lr_sweeps")


def plot_best_by_batch(best_rows: list[dict[str, str]]) -> None:
    scores = {
        (row["depth"], row["batch_label"], row["alpha"]): finite(row.get("score"))
        for row in best_rows
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), constrained_layout=True)
    for ax, depth in zip(axes, DEPTHS):
        ys_panel: list[float] = []
        xs = range(len(BATCHES))
        for alpha in ALPHAS:
            ys = [scores.get((depth, batch, alpha)) for batch in BATCHES]
            vals = [float("nan") if y is None else y for y in ys]
            ys_panel.extend(v for v in vals if math.isfinite(v))
            ax.plot(xs, vals, marker="o", linewidth=2.0, color=COLORS[alpha], label=LABELS[alpha])
        ax.set_xticks(list(xs), BATCHES)
        ax.set_title(depth)
        ax.set_xlabel("global batch size")
        ax.set_ylabel("best validation BPB")
        set_tight_ylim(ax, ys_panel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=2, frameon=False)
    save(fig, "d12_d16_best_bpb_by_batch")


def plot_delta(delta_rows: list[dict[str, str]]) -> None:
    deltas = {
        (row["depth"], row["batch_label"]): finite(row.get("delta_c05_minus_c1"))
        for row in delta_rows
    }
    fig, ax = plt.subplots(figsize=(8.5, 3.8), constrained_layout=True)
    width = 0.36
    xs = list(range(len(BATCHES)))
    all_vals: list[float] = []
    for offset, depth in [(-width / 2, "d12"), (width / 2, "d16")]:
        vals = [deltas.get((depth, batch)) for batch in BATCHES]
        plotted = [0.0 if v is None else v for v in vals]
        all_vals.extend(v for v in plotted if math.isfinite(v))
        ax.bar([x + offset for x in xs], plotted, width=width, label=depth)
    ax.axhline(0.0, color="#333", linewidth=0.9)
    ax.set_xticks(xs, BATCHES)
    ax.set_xlabel("global batch size")
    ax.set_ylabel("best BPB delta, c=0.5 - c=1")
    ax.legend(frameon=False)
    set_tight_ylim(ax, all_vals + [0.0])
    save(fig, "d12_d16_delta_c05_minus_c1")


def plot_timeseries(
    rows: list[dict[str, str]],
    value_key: str,
    ylabel: str,
    title: str,
    stem: str,
    late_zoom: bool = False,
) -> None:
    by_run: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_run[(row["depth"], row["batch_label"], row["alpha"])].append(row)

    fig, axes = plt.subplots(2, 3, figsize=(11.8, 6.2), constrained_layout=True)
    for r, depth in enumerate(DEPTHS):
        for c, batch in enumerate(BATCHES):
            ax = axes[r][c]
            ys_panel: list[float] = []
            for alpha in ALPHAS:
                run_rows = sorted(by_run.get((depth, batch, alpha), []), key=lambda row: int(float(row["step"])))
                points = [
                    (int(float(row["step"])), finite(row.get(value_key)))
                    for row in run_rows
                ]
                points = [(step, val) for step, val in points if val is not None]
                if not points:
                    continue
                max_step = max(step for step, _ in points)
                if late_zoom:
                    cutoff = int(0.7 * max_step)
                    points = [(step, val) for step, val in points if step >= cutoff]
                xs, ys = zip(*points)
                ys_panel.extend(ys)
                lr = run_rows[0].get("lr", "?") if run_rows else "?"
                ax.plot(xs, ys, linewidth=1.75, marker="o", markersize=2.5, color=COLORS[alpha], label=f"{LABELS[alpha]} lr={lr}")
            ax.set_title(f"{depth} · {batch}")
            ax.set_xlabel("optimizer step")
            if c == 0:
                ax.set_ylabel(ylabel)
            ax.xaxis.set_major_formatter(FuncFormatter(step_formatter))
            set_tight_ylim(ax, ys_panel)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.035), ncol=4, frameon=False)
    fig.suptitle(title, fontsize=13.5, fontweight="bold", y=1.08)
    save(fig, stem)


def main() -> None:
    setup_style()
    all_rows = read_csv("all_rows.csv")
    best_rows = read_csv("best_by_depth_batch_alpha.csv")
    delta_rows = read_csv("delta_c05_minus_c1.csv")
    train_rows = read_csv("best_training_loss_timeseries.csv")
    val_rows = read_csv("best_val_bpb_timeseries.csv")

    plot_lr_sweeps(all_rows)
    plot_best_by_batch(best_rows)
    plot_delta(delta_rows)
    plot_timeseries(
        train_rows,
        value_key="loss",
        ylabel="training loss",
        title="Best-LR Training Loss Curves",
        stem="d12_d16_best_training_loss_curves",
    )
    plot_timeseries(
        train_rows,
        value_key="loss",
        ylabel="training loss",
        title="Best-LR Training Loss Curves, Late Zoom",
        stem="d12_d16_best_training_loss_curves_late_zoom",
        late_zoom=True,
    )
    plot_timeseries(
        val_rows,
        value_key="val_bpb",
        ylabel="validation BPB",
        title="Best-LR Validation BPB Curves",
        stem="d12_d16_best_val_bpb_curves",
    )
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
