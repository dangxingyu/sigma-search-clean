#!/usr/bin/env python3
"""Visualize the Qwen3 d12 64K Muon coordinate-descent ledger."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / "results" / "qwen3_d12_muon64_coordinate_descent"
COORDS = ["matrix_lr", "muon_momentum", "adam_lr_multiplier", "adam_beta1", "weight_decay"]
POSITIVE_LOG_COORDS = {"matrix_lr", "adam_lr_multiplier", "weight_decay"}
BETA_COMPLEMENT_COORDS = {"muon_momentum", "adam_beta1"}


def load_state(ledger: Path) -> dict[str, Any]:
    return json.loads((ledger / "state.json").read_text())


def load_rows(ledger: Path, round_idx: int) -> list[dict[str, str]]:
    path = ledger / f"round_{round_idx:02d}_candidates.csv"
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def score(row: dict[str, str]) -> float:
    return float(row["score"])


def value_for(row: dict[str, str]) -> float:
    return float(row[row["coordinate"]])


def center_recipe_for(state: dict[str, Any], round_info: dict[str, Any]) -> dict[str, Any]:
    return dict(round_info.get("center_recipe") or state["base_recipe"])


def center_score_for(state: dict[str, Any], round_info: dict[str, Any]) -> float:
    if "center_score" in round_info:
        return float(round_info["center_score"])
    return float(state["base_recipe"]["score"])


def axis_value(coord: str, value: float) -> float:
    if coord in POSITIVE_LOG_COORDS:
        return math.log10(value)
    if coord in BETA_COMPLEMENT_COORDS:
        return -math.log10(max(1.0 - value, 1e-12))
    return value


def format_value(value: float) -> str:
    return f"{value:.6g}"


def short_coord(coord: str) -> str:
    return {
        "matrix_lr": "matrix lr",
        "muon_momentum": "muon beta1",
        "adam_lr_multiplier": "adam lr mult",
        "adam_beta1": "adam beta1",
        "weight_decay": "weight decay",
    }.get(coord, coord)


def coord_axis_label(coord: str) -> str:
    if coord in POSITIVE_LOG_COORDS:
        return f"{short_coord(coord)} (log scale)"
    if coord in BETA_COMPLEMENT_COORDS:
        return f"{short_coord(coord)} (log scale in 1-beta)"
    return short_coord(coord)


def local_ylim(values: list[float], center_score: float) -> tuple[float, float]:
    vals = values + [center_score]
    lo, hi = min(vals), max(vals)
    span = max(hi - lo, 8e-5)
    return lo - span * 0.22, hi + span * 0.28


def round_best_stamp(round_info: dict[str, Any]) -> tuple[str, str, float]:
    best_coord = ""
    best_stamp = ""
    best_imp = -float("inf")
    for coord, rec in (round_info.get("best_by_coordinate") or {}).items():
        imp = float(rec["improvement"])
        if imp > best_imp:
            best_coord = coord
            best_stamp = rec["stamp"]
            best_imp = imp
    return best_coord, best_stamp, best_imp


def draw_coordinate_sweep(
    ax: Any,
    *,
    coord: str,
    rows: list[dict[str, str]],
    round_info: dict[str, Any],
    state: dict[str, Any],
    annotate: bool,
    legend_label: str | None = None,
    color: str = "#1f6f8b",
) -> None:
    center_recipe = center_recipe_for(state, round_info)
    center_score = center_score_for(state, round_info)
    accepted_coord = round_info.get("accepted_coordinate")
    accepted_stamp = ""
    if accepted_coord:
        accepted_stamp = round_info["best_by_coordinate"][accepted_coord]["stamp"]
    best_coord, best_stamp, _ = round_best_stamp(round_info)

    coord_rows = [row for row in rows if row["coordinate"] == coord and row.get("score")]
    if not coord_rows:
        ax.set_facecolor("#f3f3f3")
        ax.text(0.5, 0.5, "skipped", transform=ax.transAxes, ha="center", va="center", color="#777777", fontsize=13)
        ax.set_xticks([])
        ax.axhline(center_score, color="#1f2933", linewidth=1.0, linestyle="--", alpha=0.8)
        ax.set_ylim(*local_ylim([], center_score))
        return

    coord_rows.sort(key=lambda row: value_for(row))
    xs = [axis_value(coord, value_for(row)) for row in coord_rows]
    ys = [score(row) for row in coord_rows]
    point_colors = ["#2a9d55" if y < center_score else "#c43b3b" for y in ys]
    ax.plot(xs, ys, "o-", color=color, linewidth=2.1, markersize=5.0, alpha=0.92, label=legend_label, zorder=2)
    ax.scatter(xs, ys, s=44, c=point_colors, edgecolor="#1f2933", linewidth=0.5, zorder=3)

    center_x = axis_value(coord, float(center_recipe[coord]))
    ax.axhline(center_score, color="#1f2933", linewidth=1.05, linestyle="--", alpha=0.85)
    ax.axvline(center_x, color="#6b7280", linewidth=0.95, linestyle=":", alpha=0.9)

    best_row = min(coord_rows, key=score)
    best_x = axis_value(coord, value_for(best_row))
    best_y = score(best_row)
    ax.scatter([best_x], [best_y], s=96, marker="D", facecolor="none", edgecolor="#111111", linewidth=1.15, zorder=4)

    for row, x, y in zip(coord_rows, xs, ys):
        if row["stamp"] == accepted_stamp:
            ax.scatter([x], [y], s=180, marker="*", color="#ffd23f", edgecolor="#111111", linewidth=0.85, zorder=5)
        elif row["stamp"] == best_stamp and coord == best_coord and not accepted_coord:
            ax.scatter([x], [y], s=120, marker="D", facecolor="none", edgecolor="#111111", linewidth=1.7, zorder=5)
        if annotate:
            dy = 8 if y <= center_score else -12
            ax.annotate(f"{y:.6f}", (x, y), xytext=(0, dy), textcoords="offset points", ha="center", fontsize=7.2)

    ax.set_xticks(xs, [format_value(value_for(row)) for row in coord_rows], rotation=35, ha="right", fontsize=8)
    ax.set_xlabel(coord_axis_label(coord), fontsize=9)
    ax.set_ylim(*local_ylim(ys, center_score))
    ax.grid(True, which="both", axis="y", alpha=0.25)
    ax.grid(True, which="major", axis="x", alpha=0.12)
    ax.text(
        0.02,
        0.97,
        f"center {center_score:.6f}\nbest {best_y:.6f}\nΔscore {best_y - center_score:+.6f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#dddddd", alpha=0.86),
    )


def collect_round_summary(state: dict[str, Any]) -> list[dict[str, Any]]:
    path_score = float(state["base_recipe"]["score"])
    records: list[dict[str, Any]] = [
        {
            "step": "base",
            "round": -1,
            "center_score": path_score,
            "accepted_coordinate": "",
            "accepted_score": path_score,
            "accepted_improvement": 0.0,
            "accepted_stamp": state["base_recipe"].get("source_stamp", "base"),
            "status": "base",
        }
    ]
    for round_info in state["rounds"]:
        center_score = center_score_for(state, round_info)
        accepted_coord = round_info.get("accepted_coordinate")
        accepted_score = path_score
        accepted_stamp = ""
        if accepted_coord:
            accepted = round_info["best_by_coordinate"][accepted_coord]
            accepted_score = float(accepted["score"])
            accepted_stamp = accepted["stamp"]
            path_score = accepted_score
        records.append(
            {
                "step": f"round {round_info['round']}",
                "round": int(round_info["round"]),
                "center_score": center_score,
                "accepted_coordinate": accepted_coord or "",
                "accepted_score": accepted_score,
                "accepted_improvement": float(round_info.get("accepted_improvement") or 0.0),
                "accepted_stamp": accepted_stamp,
                "status": round_info.get("status", ""),
            }
        )
    return records


def write_summary_csv(state: dict[str, Any], out_dir: Path) -> Path:
    path = out_dir / "qwen3_d12_muon64_cd_round_summary.csv"
    records = collect_round_summary(state)
    fields = [
        "step",
        "round",
        "center_score",
        "accepted_coordinate",
        "accepted_score",
        "accepted_improvement",
        "accepted_stamp",
        "status",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)
    return path


def plot_round_dashboards(state: dict[str, Any], ledger: Path, out_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    for round_info in state["rounds"]:
        round_idx = int(round_info["round"])
        rows = load_rows(ledger, round_idx)
        center_recipe = center_recipe_for(state, round_info)
        center_score = center_score_for(state, round_info)
        accepted_coord = round_info.get("accepted_coordinate")
        best_coord, _, best_imp = round_best_stamp(round_info)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=180)
        axes_flat = list(axes.ravel())
        for ax, coord in zip(axes_flat[:5], COORDS):
            draw_coordinate_sweep(
                ax,
                coord=coord,
                rows=rows,
                round_info=round_info,
                state=state,
                annotate=True,
            )
            ax.set_title(short_coord(coord), fontsize=11, fontweight="bold")
            ax.set_ylabel("final validation BPB")

        ax_summary = axes_flat[5]
        ax_summary.axis("off")
        status = "converged" if not accepted_coord else f"accepted {accepted_coord}"
        recipe_lines = [
            f"matrix_lr={format_value(float(center_recipe['matrix_lr']))}",
            f"muon_momentum={format_value(float(center_recipe['muon_momentum']))}",
            f"adam_lr_multiplier={format_value(float(center_recipe['adam_lr_multiplier']))}",
            f"adam_beta1={format_value(float(center_recipe['adam_beta1']))}",
            f"weight_decay={format_value(float(center_recipe['weight_decay']))}",
        ]
        ax_summary.text(
            0.02,
            0.98,
            "\n".join(
                [
                    f"round {round_idx}",
                    f"center score: {center_score:.6f}",
                    f"decision: {status}",
                    f"round best: {best_coord or 'n/a'} ({best_imp:+.6f})",
                    "",
                    "center recipe:",
                    *recipe_lines,
                    "",
                    "green = better than center",
                    "red = worse than center",
                    "diamond = best in panel",
                    "star = accepted move",
                ]
            ),
            transform=ax_summary.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.55", fc="#f8fafc", ec="#cbd5e1"),
        )

        fig.suptitle(f"Qwen3 d12 64K Muon coordinate sweeps: round {round_idx}", fontsize=15, fontweight="bold")
        fig.tight_layout(rect=[0, 0.02, 1, 0.95])
        out = out_dir / f"qwen3_d12_muon64_cd_round_{round_idx:02d}_lrsweep_style.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out)
    return outputs


def plot_coordinate_overlays(state: dict[str, Any], ledger: Path, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, len(COORDS), figsize=(22, 5.8), dpi=180)
    palette = ["#1f6f8b", "#8b5a2b", "#2a9d55", "#7c3aed", "#c43b3b", "#475569"]
    for ax, coord in zip(axes, COORDS):
        coord_scores: list[float] = []
        tick_values: set[float] = set()
        for round_info, color in zip(state["rounds"], palette):
            round_idx = int(round_info["round"])
            rows = load_rows(ledger, round_idx)
            coord_rows = [row for row in rows if row["coordinate"] == coord and row.get("score")]
            if not coord_rows:
                continue
            coord_rows.sort(key=lambda row: value_for(row))
            tick_values.update(value_for(row) for row in coord_rows)
            xs = [axis_value(coord, value_for(row)) for row in coord_rows]
            ys = [score(row) for row in coord_rows]
            coord_scores.extend(ys)
            center_score = center_score_for(state, round_info)
            coord_scores.append(center_score)
            ax.plot(xs, ys, "o-", color=color, linewidth=1.85, markersize=4.6, label=f"r{round_idx} center {center_score:.6f}")
            accepted_coord = round_info.get("accepted_coordinate")
            accepted_stamp = ""
            if accepted_coord:
                accepted_stamp = round_info["best_by_coordinate"][accepted_coord]["stamp"]
            for row, x, y in zip(coord_rows, xs, ys):
                if row["stamp"] == accepted_stamp:
                    ax.scatter([x], [y], s=150, marker="*", color="#ffd23f", edgecolor="#111111", linewidth=0.8, zorder=6)
            ax.axhline(center_score, color=color, linewidth=0.85, linestyle="--", alpha=0.38)

        ax.set_title(short_coord(coord), fontsize=11, fontweight="bold")
        ax.set_xlabel(coord_axis_label(coord), fontsize=9)
        ax.set_ylabel("final validation BPB")
        if tick_values:
            ticks = sorted(tick_values)
            ax.set_xticks([axis_value(coord, v) for v in ticks], [format_value(v) for v in ticks], rotation=45, ha="right", fontsize=7)
        ax.grid(True, axis="y", alpha=0.25)
        ax.grid(True, axis="x", alpha=0.12)
        if coord_scores:
            lo, hi = min(coord_scores), max(coord_scores)
            span = max(hi - lo, 8e-5)
            ax.set_ylim(lo - 0.18 * span, hi + 0.28 * span)
        ax.legend(fontsize=7, loc="best", framealpha=0.82)

    fig.suptitle(
        "Same coordinate across rounds (contexts differ because other hyperparameters changed)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.01, 1, 0.93])
    out = out_dir / "qwen3_d12_muon64_cd_coordinate_overlays.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_path(state: dict[str, Any], out_dir: Path) -> Path:
    records = collect_round_summary(state)
    labels = [r["step"] for r in records]
    path_scores = [float(r["accepted_score"]) for r in records]
    improvements = [float(r["accepted_improvement"]) for r in records[1:]]
    accepted_labels = [r["accepted_coordinate"] or "stop" for r in records[1:]]

    fig, (ax_score, ax_imp) = plt.subplots(
        2,
        1,
        figsize=(10.5, 7.2),
        dpi=180,
        gridspec_kw={"height_ratios": [2.1, 1.0], "hspace": 0.28},
    )

    x = np.arange(len(labels))
    ax_score.plot(x, path_scores, marker="o", linewidth=2.5, color="#1f6f8b")
    ax_score.scatter(x[-1], path_scores[-1], s=95, marker="X", color="#c43b3b", zorder=4)
    for i, rec in enumerate(records):
        if i == 0:
            note = f"{path_scores[i]:.6f}"
        elif rec["accepted_coordinate"]:
            note = f"{rec['accepted_coordinate']}\n{path_scores[i]:.6f}"
        else:
            note = f"converged\n{path_scores[i]:.6f}"
        ax_score.annotate(note, (i, path_scores[i]), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=8)
    ax_score.set_xticks(x, labels)
    ax_score.set_ylabel("final validation BPB (lower is better)")
    ax_score.set_title("Qwen3 d12 64K Muon coordinate descent: accepted path")
    ax_score.grid(True, axis="y", alpha=0.25)

    bx = np.arange(len(improvements))
    bar_colors = ["#2a9d55" if v > 0 else "#c43b3b" for v in improvements]
    ax_imp.bar(bx, improvements, color=bar_colors, width=0.62)
    ax_imp.axhline(0, color="#222222", linewidth=0.9)
    for i, (v, label) in enumerate(zip(improvements, accepted_labels)):
        ax_imp.annotate(f"{label}\n{v:+.6f}", (i, v), xytext=(0, 5 if v >= 0 else -18), textcoords="offset points", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    ax_imp.set_xticks(bx, [f"round {r['round']}" for r in records[1:]])
    ax_imp.set_ylabel("improvement vs round center")
    ax_imp.grid(True, axis="y", alpha=0.25)

    out = out_dir / "qwen3_d12_muon64_cd_path.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_heatmap(state: dict[str, Any], out_dir: Path) -> Path:
    rounds = state["rounds"]
    data = np.full((len(COORDS), len(rounds)), np.nan)
    labels = [["" for _ in rounds] for _ in COORDS]
    accepted = set()
    best_not_accepted = set()
    for j, round_info in enumerate(rounds):
        by_coord = round_info.get("best_by_coordinate") or {}
        best_coord = None
        best_imp = -float("inf")
        for coord, rec in by_coord.items():
            imp = float(rec["improvement"])
            if coord in COORDS:
                i = COORDS.index(coord)
                data[i, j] = imp
                labels[i][j] = f"{imp:+.6f}"
            if imp > best_imp:
                best_imp = imp
                best_coord = coord
        acc = round_info.get("accepted_coordinate")
        if acc in COORDS:
            accepted.add((COORDS.index(acc), j))
        elif best_coord in COORDS:
            best_not_accepted.add((COORDS.index(best_coord), j))

    finite = data[np.isfinite(data)]
    vmax = max(abs(float(finite.min())), abs(float(finite.max()))) if finite.size else 1.0
    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(8.8, 4.9), dpi=180)
    im = ax.imshow(data, cmap="RdYlGn", norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(rounds)), [f"round {r['round']}" for r in rounds])
    ax.set_yticks(np.arange(len(COORDS)), [short_coord(c) for c in COORDS])
    ax.set_title("Best improvement per coordinate and round")
    for i in range(len(COORDS)):
        for j in range(len(rounds)):
            text = labels[i][j] or "skipped"
            val = data[i, j]
            color = "white" if np.isfinite(val) and abs(val) > vmax * 0.48 else "#222222"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)
    for i, j in accepted:
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#111111", linewidth=2.4))
    for i, j in best_not_accepted:
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#5b5b5b", linewidth=1.8, linestyle="--"))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("score improvement vs round center")
    ax.text(
        0.0,
        -0.18,
        "Solid outline = accepted coordinate. Dashed outline = round best when no improvement was accepted.",
        transform=ax.transAxes,
        fontsize=8,
        color="#333333",
    )
    out = out_dir / "qwen3_d12_muon64_cd_improvement_heatmap.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_sweeps(state: dict[str, Any], ledger: Path, out_dir: Path) -> Path:
    rounds = state["rounds"]
    all_scores: list[float] = [float(state["base_recipe"]["score"])]
    rows_by_round: dict[int, list[dict[str, str]]] = {}
    for round_info in rounds:
        idx = int(round_info["round"])
        rows = load_rows(ledger, idx)
        rows_by_round[idx] = rows
        all_scores.extend(score(row) for row in rows if row.get("score"))
        all_scores.append(center_score_for(state, round_info))
    y_min, y_max = min(all_scores), max(all_scores)
    pad = (y_max - y_min) * 0.07

    fig, axes = plt.subplots(len(rounds), len(COORDS), figsize=(20, 12.5), dpi=180, sharey=True)
    if len(rounds) == 1:
        axes = np.asarray([axes])

    for r_i, round_info in enumerate(rounds):
        round_idx = int(round_info["round"])
        center_recipe = center_recipe_for(state, round_info)
        center_score = center_score_for(state, round_info)
        accepted_coord = round_info.get("accepted_coordinate")
        accepted_stamp = ""
        if accepted_coord:
            accepted_stamp = round_info["best_by_coordinate"][accepted_coord]["stamp"]
        best_coord = None
        best_stamp = ""
        best_imp = -float("inf")
        for coord, rec in (round_info.get("best_by_coordinate") or {}).items():
            imp = float(rec["improvement"])
            if imp > best_imp:
                best_imp = imp
                best_coord = coord
                best_stamp = rec["stamp"]

        for c_i, coord in enumerate(COORDS):
            ax = axes[r_i, c_i]
            coord_rows = [row for row in rows_by_round[round_idx] if row["coordinate"] == coord]
            if not coord_rows:
                ax.set_facecolor("#f2f2f2")
                ax.text(0.5, 0.5, "skipped", transform=ax.transAxes, ha="center", va="center", color="#777777")
                ax.set_xticks([])
            else:
                coord_rows.sort(key=lambda row: value_for(row))
                xs = [axis_value(coord, value_for(row)) for row in coord_rows]
                ys = [score(row) for row in coord_rows]
                marker_colors = ["#2a9d55" if y < center_score else "#c43b3b" for y in ys]
                ax.plot(xs, ys, color="#66828f", linewidth=1.2, alpha=0.9, zorder=1)
                ax.scatter(xs, ys, s=34, c=marker_colors, edgecolor="#222222", linewidth=0.4, zorder=2)
                for row, x, y in zip(coord_rows, xs, ys):
                    if row["stamp"] == accepted_stamp:
                        ax.scatter([x], [y], s=150, marker="*", color="#ffd23f", edgecolor="#111111", linewidth=0.8, zorder=5)
                    elif row["stamp"] == best_stamp and coord == best_coord and not accepted_coord:
                        ax.scatter([x], [y], s=95, marker="D", facecolor="none", edgecolor="#111111", linewidth=1.2, zorder=4)
                center_x = axis_value(coord, float(center_recipe[coord]))
                ax.axvline(center_x, color="#666666", linewidth=0.8, linestyle=":", alpha=0.9)
                ax.set_xticks(xs, [format_value(value_for(row)) for row in coord_rows], rotation=45, ha="right", fontsize=7)

            ax.axhline(center_score, color="#1f2933", linewidth=0.95, linestyle="--", alpha=0.85)
            ax.set_ylim(y_min - pad, y_max + pad)
            ax.grid(True, axis="y", alpha=0.20)
            if r_i == 0:
                ax.set_title(short_coord(coord), fontsize=10)
            if c_i == 0:
                ax.set_ylabel(f"round {round_idx}\nBPB")
            if r_i == len(rounds) - 1:
                if coord in POSITIVE_LOG_COORDS:
                    label = "candidate value (log-spaced axis)"
                elif coord in BETA_COMPLEMENT_COORDS:
                    label = "candidate beta (log(1-beta) axis)"
                else:
                    label = "candidate value"
                ax.set_xlabel(label, fontsize=8)

    fig.suptitle(
        "Coordinate sweeps by round: each panel changes only the named coordinate",
        fontsize=14,
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        "Dashed horizontal line = current center score for that round. Dotted vertical line = center value. "
        "Green point improves the center; red point is worse. Star = accepted move; hollow diamond = best candidate when converged.",
        ha="center",
        fontsize=9,
        color="#333333",
    )
    out = out_dir / "qwen3_d12_muon64_cd_sweeps.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def write_report(state: dict[str, Any], out_dir: Path, outputs: list[Path]) -> Path:
    current = state["current_recipe"]
    lines = [
        "# Qwen3 d12 64K Muon Coordinate Descent Visualization",
        "",
        "Lower score / BPB is better.",
        "",
        "## Final recipe",
        "",
        "```text",
        f"matrix_lr={current['matrix_lr']}",
        f"muon_momentum={current['muon_momentum']}",
        f"adam_lr_multiplier={current['adam_lr_multiplier']}",
        f"adam_beta1={current['adam_beta1']}",
        f"weight_decay={current['weight_decay']}",
        f"score={current['score']}",
        f"source={current.get('source_stamp', '')}",
        "```",
        "",
        "## Files",
        "",
    ]
    for path in outputs:
        lines.append(f"- [{path.name}]({path.name})")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Each sweep panel varies only the coordinate named in the column; all other hyperparameters are fixed at that round's center.",
            "- Positive-valued coordinates use a log-spaced x-axis. Beta-like coordinates use a log(1-beta) transformed x-axis, matching the search grid near 1.",
            "- The horizontal center score is the accepted center for that round; candidate rows at the same hyperparameter value are independent re-evaluations.",
        ]
    )
    path = out_dir / "README.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def resize_to_width(image: Image.Image, width: int) -> Image.Image:
    if image.width == width:
        return image
    height = round(image.height * width / image.width)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def compose_full_dashboard(out_dir: Path, outputs: list[Path]) -> Path:
    by_name = {path.name: path for path in outputs}
    target_width = 3600
    pad = 36
    title_h = 132
    section_h = 54
    bg = "white"
    text = "#111827"

    def title_image(label: str, width: int, height: int = section_h, font_size: int = 34) -> Image.Image:
        img = Image.new("RGB", (width, height), bg)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()
        draw.text((0, 4), label, fill=text, font=font)
        return img

    # Top row: accepted path and improvement heatmap side by side.
    path_img = load_rgb(by_name["qwen3_d12_muon64_cd_path.png"])
    heatmap_img = load_rgb(by_name["qwen3_d12_muon64_cd_improvement_heatmap.png"])
    half_w = (target_width - pad) // 2
    path_img = resize_to_width(path_img, half_w)
    heatmap_img = resize_to_width(heatmap_img, half_w)
    top_h = max(path_img.height, heatmap_img.height)
    top_row = Image.new("RGB", (target_width, top_h), bg)
    top_row.paste(path_img, (0, 0))
    top_row.paste(heatmap_img, (half_w + pad, 0))

    overlay = resize_to_width(load_rgb(by_name["qwen3_d12_muon64_cd_coordinate_overlays.png"]), target_width)
    round_imgs = [
        resize_to_width(load_rgb(by_name[f"qwen3_d12_muon64_cd_round_{idx:02d}_lrsweep_style.png"]), target_width)
        for idx in range(4)
    ]

    header = Image.new("RGB", (target_width, title_h), bg)
    draw = ImageDraw.Draw(header)
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 52)
        sub_font = ImageFont.truetype("DejaVuSans.ttf", 28)
    except OSError:
        title_font = ImageFont.load_default()
        sub_font = ImageFont.load_default()
    draw.text((0, 8), "Qwen3 d12 64K Muon Coordinate Descent Dashboard", fill=text, font=title_font)
    draw.text(
        (0, 78),
        "Lower final validation BPB is better. Round panels vary one coordinate at a time with all other hyperparameters fixed at that round center.",
        fill="#374151",
        font=sub_font,
    )

    sections: list[Image.Image] = [
        header,
        title_image("Accepted path and coordinate-level improvements", target_width),
        top_row,
        title_image("Same coordinate across rounds", target_width),
        overlay,
        title_image("Per-round lrsweep-style coordinate sweeps", target_width),
    ]
    for idx, img in enumerate(round_imgs):
        sections.append(title_image(f"Round {idx}", target_width, height=42, font_size=28))
        sections.append(img)

    total_h = sum(img.height for img in sections) + pad * (len(sections) - 1)
    dashboard = Image.new("RGB", (target_width, total_h), bg)
    y = 0
    for img in sections:
        dashboard.paste(img, (0, y))
        y += img.height + pad

    out = out_dir / "qwen3_d12_muon64_cd_full_dashboard.png"
    dashboard.save(out, optimize=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    ledger = args.ledger.resolve()
    out_dir = (args.out_dir or ledger / "visualizations").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    state = load_state(ledger)
    outputs = [
        plot_path(state, out_dir),
        plot_heatmap(state, out_dir),
        plot_sweeps(state, ledger, out_dir),
        plot_coordinate_overlays(state, ledger, out_dir),
        write_summary_csv(state, out_dir),
    ]
    outputs.extend(plot_round_dashboards(state, ledger, out_dir))
    outputs.append(compose_full_dashboard(out_dir, outputs))
    outputs.append(write_report(state, out_dir, outputs))
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
