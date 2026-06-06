#!/usr/bin/env python3
from __future__ import annotations

import csv
import html
import json
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "qwen3_muon_klsoap_dashboard"
HDFS_BASE = "hdfs://haruna/home/byte_data_seed/hdd_hldy/user/xingyu.dang/sigma-search-runs/search_evals"

METHOD_LABEL = {"plain_muon": "Muon", "kl_soap": "KL-SOAP", "coord_descent": "coord-descent"}
METHOD_COLOR = {"plain_muon": "#1f77b4", "kl_soap": "#d62728", "coord_descent": "#111111"}
METHOD_MARKER = {"plain_muon": "o", "kl_soap": "s", "coord_descent": "*"}
BETA_COLORS = {"0.85": "#1f77b4", "0.90": "#2ca02c", "0.95": "#ff7f0e", "0.97": "#9467bd"}

COORD_DESCENT_ROW = {
    "group": "coord_descent",
    "method": "coord_descent",
    "method_label": METHOD_LABEL["coord_descent"],
    "config": "coordinate descent tuned Muon 64K",
    "beta1": "0.95",
    "batch": 65536,
    "batch_label": "64K",
    "lr": 0.008,
    "loss": 0.856601,
    "architecture": "qwen3",
    "depth": "12",
    "weight_decay": "0.20",
    "muon_momentum": "0.95",
    "muon_momentum_schedule": "static",
    "optimizer_beta1": "0.95",
    "optimizer_beta2": "0.95",
    "shampoo_beta": "0.95",
    "batch_beta_align_mode": "beta2_only",
    "root": "qwen3_d12_muon64_coordinate_descent",
    "path": (
        f"{HDFS_BASE}/qwen3_d12_muon64_cd_r02_adam_lr_multiplier_4_20260605/"
        "plain_muon_bsz65536_lr0p008_s42/result.json"
    ),
    "note": "d12 coordinate descent tuned Muon 64K final point; kept out of d8 sweep axes.",
}

SOURCE_ROOTS = [
    {
        "group": "mainline",
        "method": "plain_muon",
        "config": "Muon momentum=0.95",
        "beta1": "0.95",
        "root": f"{HDFS_BASE}/qwen3_d8_1x_64k512k_20260603_0626",
        "note": "Use Muon rows only; KL rows in this root are obsolete.",
    },
    {
        "group": "mainline",
        "method": "kl_soap",
        "config": "KL-SOAP beta1=0.90",
        "beta1": "0.90",
        "root": f"{HDFS_BASE}/qwen3_d8_1x_64k512k_klsoap_b1p90_b2shamp0p95_20260603_0730",
        "note": "Final KL-SOAP mainline.",
    },
    {
        "group": "mainline",
        "method": None,
        "config": "mainline LR=0.012 extension",
        "beta1": "",
        "root": f"{HDFS_BASE}/qwen3_d8_1x_64k512k_lrext_lr0p012_h20_20260604_0245",
        "note": "One-step high-LR extension for 64K/512K.",
    },
    {
        "group": "mainline",
        "method": None,
        "config": "mainline 128K sweep",
        "beta1": "",
        "root": f"{HDFS_BASE}/qwen3_d8_1x_128k_lr006_012_h20_20260604_0250",
        "note": "128K sweep under final recipes.",
    },
    {
        "group": "mainline",
        "method": "plain_muon",
        "config": "Muon momentum=0.95 high-LR extension",
        "beta1": "0.95",
        "root": f"{HDFS_BASE}/qwen3_d8_1x_muon_lrext_b0p95_64k128k512k_lr016_032_h20_20260604_0325",
        "note": "High-LR extension for Muon momentum=0.95 boundary optima.",
    },
    {
        "group": "fixed_beta1",
        "method": "plain_muon",
        "config": "fixed beta1=0.95",
        "beta1": "0.95",
        "root": f"{HDFS_BASE}/qwen3_d8_1x_64k512k_20260603_0626",
        "note": "Adds lower-LR Muon rows for fixed-beta1=0.95.",
    },
    {
        "group": "fixed_beta1",
        "method": "plain_muon",
        "config": "fixed beta1=0.95",
        "beta1": "0.95",
        "root": f"{HDFS_BASE}/qwen3_d8_1x_64k512k_lrext_lr0p012_h20_20260604_0245",
        "note": "Adds LR=0.012 Muon rows for fixed-beta1=0.95.",
    },
    {
        "group": "fixed_beta1",
        "method": "plain_muon",
        "config": "fixed beta1=0.95",
        "beta1": "0.95",
        "root": f"{HDFS_BASE}/qwen3_d8_1x_128k_lr006_012_h20_20260604_0250",
        "note": "Adds 128K Muon rows for fixed-beta1=0.95.",
    },
    {
        "group": "fixed_beta1",
        "method": "plain_muon",
        "config": "fixed beta1=0.95",
        "beta1": "0.95",
        "root": f"{HDFS_BASE}/qwen3_d8_1x_muon_lrext_b0p95_64k128k512k_lr016_032_h20_20260604_0325",
        "note": "Adds high-LR Muon rows for fixed-beta1=0.95.",
    },
    {
        "group": "fixed_beta1",
        "method": None,
        "config": "fixed beta1=0.85",
        "beta1": "0.85",
        "root": f"{HDFS_BASE}/qwen3_d8_1x_betafixed_b0p85_64k128k512k_lr006_012_h20_20260604_0255",
        "note": "beta2 and shampoo_beta half-life scaled.",
    },
    {
        "group": "fixed_beta1",
        "method": "plain_muon",
        "config": "fixed beta1=0.85 high-LR extension",
        "beta1": "0.85",
        "root": f"{HDFS_BASE}/qwen3_d8_1x_muon_lrext_b0p85_64k128k512k_lr016_032_h20_20260604_0325",
        "note": "High-LR extension for Muon fixed-beta1=0.85.",
    },
    {
        "group": "fixed_beta1",
        "method": None,
        "config": "fixed beta1=0.90",
        "beta1": "0.90",
        "root": f"{HDFS_BASE}/qwen3_d8_1x_betafixed_b0p90_64k128k512k_lr006_012_h20_20260604_0255",
        "note": "beta2 and shampoo_beta half-life scaled.",
    },
    {
        "group": "fixed_beta1",
        "method": "plain_muon",
        "config": "fixed beta1=0.90 high-LR extension",
        "beta1": "0.90",
        "root": f"{HDFS_BASE}/qwen3_d8_1x_muon_lrext_b0p90_64k128k512k_lr016_032_h20_20260604_0325",
        "note": "High-LR extension for Muon fixed-beta1=0.90.",
    },
    {
        "group": "fixed_beta1",
        "method": None,
        "config": "fixed beta1=0.97",
        "beta1": "0.97",
        "root": f"{HDFS_BASE}/qwen3_d8_1x_betafixed_b0p97_64k128k512k_lr006_012_h20_20260604_0255",
        "note": "beta2 and shampoo_beta half-life scaled.",
    },
    {
        "group": "fixed_beta1",
        "method": "plain_muon",
        "config": "fixed beta1=0.97 high-LR extension",
        "beta1": "0.97",
        "root": f"{HDFS_BASE}/qwen3_d8_1x_muon_lrext_b0p97_64k128k512k_lr016_032_h20_20260604_0325",
        "note": "High-LR extension for Muon fixed-beta1=0.97.",
    },
    {
        "group": "backfill",
        "method": "kl_soap",
        "config": "KL-SOAP beta1=0.95 shampoo_beta=0.95 low-LR backfill",
        "beta1": "0.95",
        "root": f"{HDFS_BASE}/qwen3_d8_1x_klsoap_b0p95_shamp0p95_fix_512k_lrlow_h20_20260604_0300",
        "note": "Replacement for completed obsolete shampoo_beta=0.90 rows.",
    },
]


def run_text(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)


def hdfs_result_paths(root: str) -> list[str]:
    output = run_text(["hdfs", "dfs", "-ls", "-R", root])
    return sorted(line.split()[-1] for line in output.splitlines() if line.endswith("/result.json"))


def hdfs_json(path: str) -> dict:
    return json.loads(run_text(["hdfs", "dfs", "-cat", path]))


def slug_batch(batch: int) -> str:
    return f"{batch // 1024}K" if batch % 1024 == 0 else str(batch)


def inferred_beta1(method: str, args: dict, fallback: str) -> str:
    if fallback:
        return fallback
    if method == "plain_muon":
        return f"{float(args.get('muon_momentum', 0)):.2f}"
    return f"{float(args.get('optimizer_beta1', 0)):.2f}"


def row_from_result(spec: dict, path: str, data: dict) -> dict | None:
    if data.get("error"):
        return None
    args = data.get("args", {})
    method = args.get("optimizer")
    if spec["method"] is not None and method != spec["method"]:
        return None
    score = data.get("score", data.get("val_bpb_final"))
    if method not in METHOD_LABEL or score is None:
        return None
    beta1 = inferred_beta1(method, args, spec["beta1"])
    batch = int(args["total_batch_size"])
    return {
        "group": spec["group"],
        "method": method,
        "method_label": METHOD_LABEL[method],
        "config": spec["config"],
        "beta1": beta1,
        "batch": batch,
        "batch_label": slug_batch(batch),
        "lr": float(args["matrix_lr"]),
        "loss": float(score),
        "architecture": args.get("architecture", ""),
        "depth": args.get("depth", ""),
        "weight_decay": args.get("weight_decay", ""),
        "muon_momentum": args.get("muon_momentum", ""),
        "muon_momentum_schedule": args.get("muon_momentum_schedule", ""),
        "optimizer_beta1": args.get("optimizer_beta1", ""),
        "optimizer_beta2": args.get("optimizer_beta2", ""),
        "shampoo_beta": args.get("shampoo_beta", ""),
        "batch_beta_align_mode": args.get("batch_beta_align_mode", ""),
        "root": spec["root"].rsplit("/", 1)[-1],
        "path": path,
        "note": spec["note"],
    }


def collect_rows() -> list[dict]:
    rows: list[dict] = []
    for spec in SOURCE_ROOTS:
        for path in hdfs_result_paths(spec["root"]):
            row = row_from_result(spec, path, hdfs_json(path))
            if row is not None:
                rows.append(row)
    rows = dedupe_rows(rows)
    return with_coord_descent(rows)


def row_sort_key(row: dict) -> tuple:
    return (row["group"], row["method"], row["beta1"], int(row["batch"]), float(row["lr"]))


def with_coord_descent(rows: list[dict]) -> list[dict]:
    rows = [
        row for row in rows
        if row.get("method") != COORD_DESCENT_ROW["method"] and row.get("group") != COORD_DESCENT_ROW["group"]
    ]
    return sorted([*rows, dict(COORD_DESCENT_ROW)], key=row_sort_key)


def read_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            row["batch"] = int(row["batch"])
            row["lr"] = float(row["lr"])
            row["loss"] = float(row["loss"])
            rows.append(row)
    return rows


def dedupe_rows(rows: list[dict]) -> list[dict]:
    # Prefer exact source-group rows over overlapping extension rows when all key
    # fields are identical. This only removes accidental duplicate result files.
    best: dict[tuple, dict] = {}
    for row in rows:
        key = (row["group"], row["method"], row["beta1"], row["batch"], row["lr"])
        best[key] = row
    return list(best.values())


def best_rows(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for group in sorted({r["group"] for r in rows}):
        for method in sorted({r["method"] for r in rows if r["group"] == group}):
            betas = sorted({r["beta1"] for r in rows if r["group"] == group and r["method"] == method})
            for beta in betas:
                batches = sorted({r["batch"] for r in rows if r["group"] == group and r["method"] == method and r["beta1"] == beta})
                for batch in batches:
                    pts = [r for r in rows if r["group"] == group and r["method"] == method and r["beta1"] == beta and r["batch"] == batch]
                    out.append(min(pts, key=lambda r: r["loss"]))
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "group", "method", "method_label", "config", "beta1", "batch",
        "batch_label", "lr", "loss", "architecture", "depth", "weight_decay",
        "muon_momentum", "muon_momentum_schedule", "optimizer_beta1",
        "optimizer_beta2", "shampoo_beta", "batch_beta_align_mode", "root",
        "path", "note",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def plot_mainline(rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    rows = [r for r in rows if r["group"] == "mainline"]
    batches = sorted({r["batch"] for r in rows})
    fig, axes = plt.subplots(1, len(batches), figsize=(6 * len(batches), 4.8), sharey=False)
    if len(batches) == 1:
        axes = [axes]
    fig.suptitle("Qwen3 Mainline LR Sweep", fontsize=18, fontweight="bold")
    for ax, batch in zip(axes, batches):
        batch_rows = [r for r in rows if r["batch"] == batch]
        for method in ["plain_muon", "kl_soap"]:
            pts = sorted([r for r in batch_rows if r["method"] == method], key=lambda r: r["lr"])
            if not pts:
                continue
            ax.plot(
                [r["lr"] for r in pts],
                [r["loss"] for r in pts],
                marker=METHOD_MARKER[method],
                color=METHOD_COLOR[method],
                label=METHOD_LABEL[method],
                linewidth=2.2,
            )
            best = min(pts, key=lambda r: r["loss"])
            ax.scatter([best["lr"]], [best["loss"]], s=130, facecolors="none", edgecolors=METHOD_COLOR[method], linewidths=2)
        ax.set_title(f"batch={slug_batch(batch)}")
        ax.set_xlabel("Matrix LR")
        ax.set_ylabel("Final validation BPB / loss")
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / "qwen3_mainline_lr_dashboard.png", dpi=180)
    fig.savefig(OUT / "qwen3_mainline_lr_dashboard.pdf")
    plt.close(fig)


def plot_fixed_beta(rows: list[dict]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = [r for r in rows if r["group"] == "fixed_beta1"]
    batches = sorted({r["batch"] for r in rows})
    fig, axes = plt.subplots(1, len(batches), figsize=(6.2 * len(batches), 5.2), sharey=False)
    if len(batches) == 1:
        axes = [axes]
    fig.suptitle(
        "Qwen3 Fixed-Beta1 Sweep: Muon vs KL-SOAP (same panel per batch)",
        fontsize=17,
        fontweight="bold",
    )
    for ax, batch in zip(axes, batches):
        panel = [r for r in rows if r["batch"] == batch]
        plot_overlay_batch_panel(ax, panel, title=f"batch={slug_batch(batch)}")
    handles = [
        Line2D([0], [0], color=c, marker="o", lw=2, label=f"beta1/momentum={b}")
        for b, c in BETA_COLORS.items()
    ]
    handles.extend([
        Line2D([0], [0], color="black", lw=2, linestyle="-", marker="o", label="Muon"),
        Line2D([0], [0], color="black", lw=2, linestyle="--", marker="s", label="KL-SOAP"),
    ])
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=9, frameon=True)
    fig.tight_layout(rect=[0, 0.06, 1, 0.92])
    fig.savefig(OUT / "qwen3_fixed_beta1_dashboard.png", dpi=180)
    fig.savefig(OUT / "qwen3_fixed_beta1_dashboard.pdf")
    plt.close(fig)


def plot_best_loss_over_beta(best: list[dict]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = [r for r in best if r["group"] == "fixed_beta1"]
    batches = sorted({r["batch"] for r in rows})
    methods = ["plain_muon", "kl_soap"]
    batch_colors = {65536: "#1f77b4", 131072: "#2ca02c", 524288: "#d62728"}

    fig, axes = plt.subplots(1, len(batches), figsize=(5.8 * len(batches), 5.0), sharey=False)
    if len(batches) == 1:
        axes = [axes]
    fig.suptitle(
        "Qwen3 Best Loss vs Beta1: Muon vs KL-SOAP (one panel per batch)",
        fontsize=17,
        fontweight="bold",
    )
    for ax, batch in zip(axes, batches):
        batch_rows = [r for r in rows if r["batch"] == batch]
        color = batch_colors.get(batch, "#333")
        for method in methods:
            pts = sorted(
                [r for r in batch_rows if r["method"] == method],
                key=lambda r: float(r["beta1"]),
            )
            if not pts:
                continue
            linestyle = "-" if method == "plain_muon" else "--"
            ax.plot(
                [float(r["beta1"]) for r in pts],
                [r["loss"] for r in pts],
                marker=METHOD_MARKER[method],
                color=color,
                linestyle=linestyle,
                linewidth=2.2,
                label=METHOD_LABEL[method],
            )
            best_pt = min(pts, key=lambda r: r["loss"])
            ax.scatter(
                [float(best_pt["beta1"])],
                [best_pt["loss"]],
                s=100,
                facecolors="none",
                edgecolors=color,
                linewidths=1.8,
            )
        ax.set_title(f"batch={slug_batch(batch)}")
        ax.set_xlabel("fixed beta1 / momentum")
        ax.set_ylabel("Best final validation BPB / loss")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)
    handles = [
        Line2D([0], [0], color="black", lw=2, linestyle="-", marker="o", label="Muon"),
        Line2D([0], [0], color="black", lw=2, linestyle="--", marker="s", label="KL-SOAP"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=10)
    fig.tight_layout(rect=[0, 0.08, 1, 0.90])
    fig.savefig(OUT / "qwen3_best_loss_over_beta1.png", dpi=180)
    fig.savefig(OUT / "qwen3_best_loss_over_beta1.pdf")
    plt.close(fig)


def plot_overlay_batch_panel(ax, panel: list[dict], *, title: str, zoom: bool = False) -> None:
    methods = ["plain_muon", "kl_soap"]
    for method in methods:
        method_rows = [r for r in panel if r["method"] == method]
        for beta in sorted({r["beta1"] for r in method_rows}):
            pts = sorted([r for r in method_rows if r["beta1"] == beta], key=lambda r: r["lr"])
            if not pts:
                continue
            color = BETA_COLORS.get(beta, "#777")
            linestyle = "-" if method == "plain_muon" else "--"
            marker = METHOD_MARKER[method]
            ax.plot(
                [r["lr"] for r in pts],
                [r["loss"] for r in pts],
                marker=marker,
                color=color,
                linestyle=linestyle,
                linewidth=1.9 if zoom else 2.1,
                markersize=4.2 if zoom else 4.8,
            )
            best_pt = min(pts, key=lambda r: r["loss"])
            ax.scatter(
                [best_pt["lr"]],
                [best_pt["loss"]],
                s=75 if zoom else 95,
                facecolors="none",
                edgecolors=color,
                linewidths=1.6,
            )
    ax.set_title(title)
    ax.set_xlabel("Matrix LR")
    ax.set_ylabel("Final validation BPB / loss")
    ax.grid(True, alpha=0.22)
    if zoom and panel:
        ys = sorted(r["loss"] for r in panel)
        lo = ys[0]
        hi = ys[min(len(ys) - 1, max(2, int(len(ys) * 0.55)))]
        pad = max((hi - lo) * 0.25, 0.0004)
        ax.set_ylim(lo - pad, hi + pad)


def plot_combined_verbose(rows: list[dict], best: list[dict]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fixed = [r for r in rows if r["group"] == "fixed_beta1"]
    methods = ["plain_muon", "kl_soap"]
    batches = sorted({r["batch"] for r in fixed})

    fig = plt.figure(figsize=(22, 19))
    gs = fig.add_gridspec(
        5,
        3,
        height_ratios=[1.0, 1.2, 0.95, 0.9, 0.55],
        left=0.045,
        right=0.985,
        top=0.92,
        bottom=0.04,
        wspace=0.23,
        hspace=0.42,
    )
    fig.suptitle(
        "Qwen3 Muon + KL-SOAP LR sweep dashboard: zoomed 64K/128K + full context",
        fontsize=20,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.895,
        "d8, 1x Chinchilla. beta1/momentum fixed; beta2 and shampoo_beta use batch half-life scaling. "
        "Open markers are per-config optima. Superseded roots excluded.",
        ha="center",
        fontsize=10.5,
        color="#444",
    )

    # Top row: zoom around current best region for 64K/128K.
    for col, batch in enumerate([65536, 131072]):
        ax = fig.add_subplot(gs[0, col])
        panel = [r for r in fixed if r["batch"] == batch and 0.006 <= r["lr"] <= 0.032]
        plot_overlay_batch_panel(
            ax,
            panel,
            title=f"ZOOM: batch={slug_batch(batch)}",
            zoom=True,
        )

    # Top-right: legend / notes.
    ax_note = fig.add_subplot(gs[0, 2])
    ax_note.axis("off")
    handles = []
    for beta, color in BETA_COLORS.items():
        handles.append(Line2D([0], [0], color=color, marker="o", lw=2, label=f"beta1/momentum={beta}"))
    handles.extend([
        Line2D([0], [0], color="black", lw=2, linestyle="-", marker="o", label="Muon"),
        Line2D([0], [0], color="black", lw=2, linestyle="--", marker="s", label="KL-SOAP"),
    ])
    ax_note.legend(handles=handles, loc="upper left", fontsize=10, frameon=True)
    note = (
        "Mainline:\n"
        "- Muon momentum=0.95 + LR ext\n"
        "- KL-SOAP beta1=0.90, b2=shampoo=0.95\n\n"
        "Fixed-beta sweep:\n"
        "- beta1/momentum 0.85, 0.90, 0.95, 0.97\n"
        "- Muon high-LR ext to 0.032\n"
        "- KL beta1=0.95 full sweep omitted\n"
        "  (mainline/backfill covers 512K)\n\n"
        "Extra entry:\n"
        "- coord-descent d12 64K lr=0.008\n"
        "  loss=0.85660; not on d8 axes"
    )
    ax_note.text(0.02, 0.55, note, va="top", fontsize=9.5, family="monospace")

    # Middle row: full-context batch panels with Muon + KL-SOAP overlaid (GPT-2 style).
    for c_idx, batch in enumerate(batches):
        ax = fig.add_subplot(gs[1, c_idx])
        panel = [r for r in fixed if r["batch"] == batch]
        plot_overlay_batch_panel(ax, panel, title=f"batch={slug_batch(batch)}")

    # Summary row: best loss and best LR by batch/config.
    ax_loss = fig.add_subplot(gs[2, 0:2])
    ax_lr = fig.add_subplot(gs[2, 2])
    fixed_best = [r for r in best if r["group"] == "fixed_beta1"]
    for method in methods:
        for beta in sorted({r["beta1"] for r in fixed_best if r["method"] == method}):
            pts = sorted([r for r in fixed_best if r["method"] == method and r["beta1"] == beta], key=lambda r: r["batch"])
            if not pts:
                continue
            label = f"{METHOD_LABEL[method]} b={beta}"
            linestyle = "-" if method == "plain_muon" else "--"
            marker = METHOD_MARKER[method]
            color = BETA_COLORS.get(beta)
            x = [slug_batch(r["batch"]) for r in pts]
            ax_loss.plot(x, [r["loss"] for r in pts], label=label, color=color, linestyle=linestyle, marker=marker, linewidth=2)
            ax_lr.plot(x, [r["lr"] for r in pts], label=label, color=color, linestyle=linestyle, marker=marker, linewidth=2)
    ax_loss.set_title("Best loss by batch (per fixed beta1/momentum)")
    ax_loss.set_xlabel("Batch")
    ax_loss.set_ylabel("Best final validation BPB / loss")
    ax_loss.grid(True, alpha=0.22)
    ax_loss.legend(ncol=4, fontsize=8)
    ax_lr.set_title("Optimal LR by batch")
    ax_lr.set_xlabel("Batch")
    ax_lr.set_ylabel("Best matrix LR")
    ax_lr.grid(True, alpha=0.22)

    # Bottom row: Muon vs KL-SOAP on same axes (one panel per batch).
    batch_colors = {65536: "#1f77b4", 131072: "#2ca02c", 524288: "#d62728"}
    for col, batch in enumerate(batches):
        ax = fig.add_subplot(gs[3, col])
        batch_rows = [r for r in fixed_best if r["batch"] == batch]
        color = batch_colors.get(batch, "#333")
        for method in methods:
            pts = sorted(
                [r for r in batch_rows if r["method"] == method],
                key=lambda r: float(r["beta1"]),
            )
            if not pts:
                continue
            linestyle = "-" if method == "plain_muon" else "--"
            ax.plot(
                [float(r["beta1"]) for r in pts],
                [r["loss"] for r in pts],
                marker=METHOD_MARKER[method],
                color=color,
                linestyle=linestyle,
                linewidth=2.2,
                label=METHOD_LABEL[method],
            )
        ax.set_title(f"best loss vs beta1, batch={slug_batch(batch)}")
        ax.set_xlabel("fixed beta1 / momentum")
        ax.set_ylabel("Best final validation BPB / loss")
        ax.grid(True, alpha=0.22)
        ax.legend(fontsize=8)

    ax_table = fig.add_subplot(gs[4, :])
    ax_table.axis("off")
    lines = ["Best rows (+ extra entries):"]
    coord_best = [r for r in best if r["group"] == "coord_descent"]
    for row in coord_best:
        lines.append(
            f"{METHOD_LABEL[row['method']]:13s} depth={row['depth']} batch={slug_batch(row['batch']):>4s} "
            f"lr={row['lr']:.4g} loss={row['loss']:.5f} (extra entry; not on d8 axes)"
        )
    if coord_best:
        lines.append("")
        lines.append("Fixed-beta sweep:")
    for row in sorted(fixed_best, key=lambda r: (r["method"], r["batch"], float(r["beta1"]))):
        lines.append(
            f"{METHOD_LABEL[row['method']]:7s} b={row['beta1']} batch={slug_batch(row['batch']):>4s} "
            f"lr={row['lr']:.4g} loss={row['loss']:.5f}"
        )
    ax_table.text(0, 1, "\n".join(lines), va="top", fontsize=8.0, family="monospace")

    fig.savefig(OUT / "qwen3_combined_verbose_dashboard.png", dpi=180)
    fig.savefig(OUT / "qwen3_combined_verbose_dashboard.pdf")
    plt.close(fig)


def html_table(rows: list[dict]) -> str:
    fields = ["group", "method_label", "beta1", "batch_label", "lr", "loss", "root"]
    header = "".join(f"<th>{html.escape(field)}</th>" for field in fields)
    body = []
    for row in rows:
        cells = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                value = f"{value:.8g}"
            cells.append(f"<td>{html.escape(str(value))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return "<table><thead><tr>" + header + "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"


def coverage_rows(rows: list[dict]) -> list[dict]:
    expected = {
        ("plain_muon", "0.85"): 28,
        ("plain_muon", "0.90"): 28,
        ("plain_muon", "0.95"): 28,
        ("plain_muon", "0.97"): 28,
        ("kl_soap", "0.85"): 12,
        ("kl_soap", "0.90"): 12,
        ("kl_soap", "0.97"): 12,
    }
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        if row["group"] != "fixed_beta1":
            continue
        key = (row["method"], row["beta1"])
        counts[key] = counts.get(key, 0) + 1
    out = []
    for (method, beta1), want in sorted(expected.items()):
        got = counts.get((method, beta1), 0)
        out.append({
            "method": METHOD_LABEL[method],
            "beta1": beta1,
            "expected": str(want),
            "completed": str(got),
            "status": "complete" if got >= want else "incomplete",
        })
    return out


def write_html(rows: list[dict], best: list[dict]) -> None:
    cov = coverage_rows(rows)
    cov_table = html_table(
        [{"method": r["method"], "beta1": r["beta1"], "expected": r["expected"], "completed": r["completed"], "status": r["status"]} for r in cov],
    )
    doc = f"""<!doctype html>
<meta charset="utf-8">
<title>Qwen3 Muon / KL-SOAP dashboard</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:28px;color:#202124}}
.card{{border:1px solid #ddd;border-radius:12px;padding:16px;margin:18px 0;box-shadow:0 1px 3px rgba(0,0,0,.05)}}
img{{max-width:100%;height:auto;border:1px solid #eee;border-radius:8px;background:white;margin:8px 0}}
table{{border-collapse:collapse;font-size:13px;width:100%;margin:10px 0}}
th,td{{border:1px solid #ddd;padding:6px 8px;text-align:left}}
th{{background:#f6f8fa}}
code{{background:#f6f8fa;padding:2px 4px;border-radius:4px}}
.small{{color:#666;font-size:13px}}
</style>
<h1>Qwen3 Muon / KL-SOAP fixed-beta1 dashboard</h1>
<p class="small">Generated from completed HDFS <code>result.json</code> files. Lower final validation BPB/loss is better. Study: <code>base-optimizer-batchsize-study</code>.</p>
<div class="card"><h2>Combined verbose dashboard (primary)</h2>
<p class="small">Single figure matching the GPT-2 <code>combined_old_new_muon_klsoap_dashboard.png</code> layout: zoom panels, batch LR facets, best-loss/best-LR summaries, and best-loss-over-beta1 curves.</p>
<img src="qwen3_combined_verbose_dashboard.png">
<p class="small">PDF: <code>qwen3_combined_verbose_dashboard.pdf</code></p>
</div>
<div class="card"><h2>Coverage (fixed-beta1 sweep)</h2>{cov_table}</div>
<div class="card"><h2>Muon detail panels</h2><img src="qwen3_fixed_beta1_dashboard.png"><img src="qwen3_best_loss_over_beta1.png"></div>
<div class="card"><h2>Mainline LR sweep</h2><img src="qwen3_mainline_lr_dashboard.png"></div>
<div class="card"><h2>Best rows</h2>{html_table(best)}</div>
</body>"""
    (OUT / "qwen3_full_dashboard.html").write_text(doc)
    (OUT / "dashboard.html").write_text(doc)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "qwen3_full_rows.csv"
    if os.environ.get("QWEN3_FULL_DASHBOARD_FROM_CSV") == "1" and csv_path.exists():
        rows = with_coord_descent(read_csv(csv_path))
    else:
        rows = collect_rows()
    best = best_rows(rows)
    write_csv(csv_path, rows)
    write_csv(OUT / "qwen3_full_best.csv", best)
    plot_mainline(rows)
    plot_fixed_beta(rows)
    plot_best_loss_over_beta(best)
    plot_combined_verbose(rows, best)
    # GPT-2-style alias for the primary combined figure.
    shutil.copyfile(OUT / "qwen3_combined_verbose_dashboard.png", OUT / "qwen3_combined_muon_klsoap_dashboard.png")
    shutil.copyfile(OUT / "qwen3_combined_verbose_dashboard.pdf", OUT / "qwen3_combined_muon_klsoap_dashboard.pdf")
    # Keep the historical/default qwen3 dashboard filename current so opening the
    # obvious PNG does not show the older 0.008-only snapshot.
    shutil.copyfile(OUT / "qwen3_mainline_lr_dashboard.png", OUT / "qwen3_muon_klsoap_dashboard.png")
    shutil.copyfile(OUT / "qwen3_mainline_lr_dashboard.pdf", OUT / "qwen3_muon_klsoap_dashboard.pdf")
    write_html(rows, best)
    print(f"rows={len(rows)} best={len(best)} out={OUT}")
    for row in best:
        print(
            f"best {row['group']} {row['method_label']} beta1={row['beta1']} "
            f"batch={row['batch_label']} lr={row['lr']:g} loss={row['loss']:.6f}"
        )


if __name__ == "__main__":
    main()
