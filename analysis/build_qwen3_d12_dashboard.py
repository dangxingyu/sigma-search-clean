#!/usr/bin/env python3
from __future__ import annotations

import csv
import html
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "qwen3_d12_muon_klsoap_dashboard"
HDFS_BASE = "hdfs://haruna/home/byte_data_seed/hdd_hldy/user/xingyu.dang/sigma-search-runs/search_evals"

METHOD_LABEL = {"plain_muon": "Muon", "kl_soap": "KL-SOAP"}
METHOD_COLOR = {"plain_muon": "#1f77b4", "kl_soap": "#d62728"}
BETA_STYLE = {
    "0.85": ("o", "-"),
    "0.90": ("s", "--"),
    "0.95": ("^", ":"),
}

SOURCE_ROOTS = [
    ("plain_muon", "0.85", f"{HDFS_BASE}/qwen3_d12_1x_h20_betafixed_b0p85_muon_20260604"),
    ("plain_muon", "0.90", f"{HDFS_BASE}/qwen3_d12_1x_h20_betafixed_b0p90_muon_20260604"),
    ("plain_muon", "0.95", f"{HDFS_BASE}/qwen3_d12_1x_h20_betafixed_b0p95_muon_20260604"),
    ("plain_muon", "0.95", f"{HDFS_BASE}/qwen3_d12_1x_h20_muon_lrext_b0p95_64k128k_lr0028_004_20260604"),
    ("kl_soap", "0.85", f"{HDFS_BASE}/qwen3_d12_1x_h20_betafixed_b0p85_klsoap_20260604"),
    ("kl_soap", "0.90", f"{HDFS_BASE}/qwen3_d12_1x_h20_betafixed_b0p90_klsoap_20260604"),
    ("kl_soap", "0.95", f"{HDFS_BASE}/qwen3_d12_1x_h20_betafixed_b0p95_klsoap_20260604"),
]


def run_text(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)


def hdfs_result_paths(root: str) -> list[str]:
    try:
        output = run_text(["hdfs", "dfs", "-ls", "-R", root])
    except subprocess.CalledProcessError:
        return []
    return sorted(line.split()[-1] for line in output.splitlines() if line.endswith("/result.json"))


def hdfs_json(path: str) -> dict:
    return json.loads(run_text(["hdfs", "dfs", "-cat", path]))


def slug_batch(batch: int) -> str:
    return f"{batch // 1024}K" if batch % 1024 == 0 else str(batch)


def row_from_result(expected_method: str, beta1: str, root: str, path: str, data: dict) -> dict | None:
    if data.get("error"):
        return None
    args = data.get("args", {})
    method = args.get("optimizer")
    score = data.get("score", data.get("val_bpb_final"))
    if method != expected_method or score is None:
        return None
    batch = int(args["total_batch_size"])
    return {
        "method": method,
        "method_label": METHOD_LABEL[method],
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
        "train_losses": data.get("train_losses", []),
        "root": root.rsplit("/", 1)[-1],
        "path": path,
    }


def collect_rows() -> list[dict]:
    rows: list[dict] = []
    for method, beta1, root in SOURCE_ROOTS:
        for path in hdfs_result_paths(root):
            row = row_from_result(method, beta1, root, path, hdfs_json(path))
            if row is not None:
                rows.append(row)
    return sorted(rows, key=lambda r: (r["method"], r["beta1"], r["batch"], r["lr"]))


def best_rows(rows: list[dict]) -> list[dict]:
    best: dict[tuple[str, str, int], dict] = {}
    for row in rows:
        key = (row["method"], row["beta1"], row["batch"])
        if key not in best or row["loss"] < best[key]["loss"]:
            best[key] = row
    return sorted(best.values(), key=lambda r: (r["method"], r["batch"], r["beta1"]))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "method", "method_label", "beta1", "batch", "batch_label", "lr", "loss",
        "architecture", "depth", "weight_decay", "muon_momentum",
        "muon_momentum_schedule", "optimizer_beta1", "optimizer_beta2",
        "shampoo_beta", "batch_beta_align_mode", "root", "path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def plot_overlay_batch_panel(ax, panel: list[dict], *, title: str, zoom: bool = False) -> None:
    for method in ["plain_muon", "kl_soap"]:
        method_rows = [r for r in panel if r["method"] == method]
        for beta in sorted({r["beta1"] for r in method_rows}):
            pts = sorted([r for r in method_rows if r["beta1"] == beta], key=lambda r: r["lr"])
            if not pts:
                continue
            color = METHOD_COLOR[method]
            marker, linestyle = BETA_STYLE.get(beta, ("o", "-"))
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
    if panel:
        winner = min(panel, key=lambda r: r["loss"])
        ax.scatter(
            [winner["lr"]],
            [winner["loss"]],
            s=155 if zoom else 190,
            marker="*",
            color="black",
            zorder=5,
        )
        ax.annotate(
            f"{METHOD_LABEL[winner['method']]} b={winner['beta1']}\n"
            f"lr={winner['lr']:.4g} loss={winner['loss']:.4f}",
            xy=(winner["lr"], winner["loss"]),
            xytext=(8, 10),
            textcoords="offset points",
            fontsize=8,
            ha="left",
            va="bottom",
        )
    ax.set_title(title)
    ax.set_xlabel("Matrix LR")
    ax.set_ylabel("Final validation BPB / loss")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.22)
    if zoom and panel:
        ys = sorted(r["loss"] for r in panel)
        lo = ys[0]
        hi = ys[min(len(ys) - 1, max(2, int(len(ys) * 0.55)))]
        pad = max((hi - lo) * 0.25, 0.0004)
        ax.set_ylim(lo - pad, hi + pad)


def plot_dashboard(rows: list[dict], best: list[dict]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if not rows:
        return
    batches = sorted({r["batch"] for r in rows})
    methods = ["plain_muon", "kl_soap"]

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
        "Qwen3 d12 Muon + KL-SOAP LR sweep dashboard: 1x Chinchilla",
        fontsize=20,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.895,
        "beta1/momentum fixed at 0.85/0.90/0.95; beta2, shampoo_beta, and AdamW beta2 use batch half-life scaling.",
        ha="center",
        fontsize=10.5,
        color="#444",
    )

    for col, batch in enumerate([65536, 131072]):
        ax = fig.add_subplot(gs[0, col])
        panel = [r for r in rows if r["batch"] == batch]
        plot_overlay_batch_panel(ax, panel, title=f"ZOOM: batch={slug_batch(batch)}", zoom=True)

    ax_note = fig.add_subplot(gs[0, 2])
    ax_note.axis("off")
    handles = [
        Line2D([0], [0], color=color, marker="o", lw=2, label=METHOD_LABEL[method])
        for method, color in METHOD_COLOR.items()
    ]
    handles.extend(
        Line2D([0], [0], color="black", marker=marker, linestyle=linestyle, lw=2, label=f"beta={beta}")
        for beta, (marker, linestyle) in BETA_STYLE.items()
    )
    handles.append(Line2D([0], [0], color="black", marker="*", linestyle="None", markersize=11, label="winner"))
    ax_note.legend(handles=handles, loc="upper left", fontsize=10, frameon=True)
    ax_note.text(
        0.02,
        0.55,
        "d12 config:\n"
        "- Qwen3-like architecture\n"
        "- depth=12, dim=768\n"
        "- 1x Chinchilla tokens\n"
        "- batches 64K/128K/512K\n"
        "- weight_decay=0.28\n"
        "- KL b2=shampoo=0.95 ref\n"
        "- batch_beta_align_mode=beta2_only",
        va="top",
        fontsize=9.5,
        family="monospace",
    )

    for col, batch in enumerate(batches):
        ax = fig.add_subplot(gs[1, col])
        panel = [r for r in rows if r["batch"] == batch]
        plot_overlay_batch_panel(ax, panel, title=f"batch={slug_batch(batch)}")

    ax_loss = fig.add_subplot(gs[2, 0:2])
    ax_lr = fig.add_subplot(gs[2, 2])
    for method in methods:
        for beta in sorted({r["beta1"] for r in best if r["method"] == method}):
            pts = sorted([r for r in best if r["method"] == method and r["beta1"] == beta], key=lambda r: r["batch"])
            if not pts:
                continue
            marker, linestyle = BETA_STYLE.get(beta, ("o", "-"))
            color = METHOD_COLOR[method]
            label = f"{METHOD_LABEL[method]} b={beta}"
            x = [slug_batch(r["batch"]) for r in pts]
            ax_loss.plot(x, [r["loss"] for r in pts], label=label, color=color, linestyle=linestyle, marker=marker, linewidth=2)
            ax_lr.plot(x, [r["lr"] for r in pts], label=label, color=color, linestyle=linestyle, marker=marker, linewidth=2)
    ax_loss.set_title("Best loss by batch")
    ax_loss.set_xlabel("Batch")
    ax_loss.set_ylabel("Best final validation BPB / loss")
    ax_loss.grid(True, alpha=0.22)
    ax_loss.legend(ncol=3, fontsize=8)
    ax_lr.set_title("Optimal LR by batch")
    ax_lr.set_xlabel("Batch")
    ax_lr.set_ylabel("Best matrix LR")
    ax_lr.set_yscale("log")
    ax_lr.grid(True, alpha=0.22)

    for col, batch in enumerate(batches):
        ax = fig.add_subplot(gs[3, col])
        for method in methods:
            pts = sorted(
                [r for r in best if r["batch"] == batch and r["method"] == method],
                key=lambda r: float(r["beta1"]),
            )
            if not pts:
                continue
            ax.plot(
                [float(r["beta1"]) for r in pts],
                [r["loss"] for r in pts],
                marker="o",
                color=METHOD_COLOR[method],
                linestyle="-",
                linewidth=2.2,
                label=METHOD_LABEL[method],
            )
        if any(r["batch"] == batch for r in best):
            winner = min([r for r in best if r["batch"] == batch], key=lambda r: r["loss"])
            ax.scatter(
                [float(winner["beta1"])],
                [winner["loss"]],
                marker="*",
                s=150,
                color="black",
                zorder=5,
            )
        ax.set_title(f"best loss vs beta1, batch={slug_batch(batch)}")
        ax.set_xlabel("fixed beta1 / momentum")
        ax.set_ylabel("Best final validation BPB / loss")
        ax.grid(True, alpha=0.22)
        ax.legend(fontsize=8)

    ax_table = fig.add_subplot(gs[4, :])
    ax_table.axis("off")
    lines = ["Best rows:"]
    for row in sorted(best, key=lambda r: (r["method"], r["batch"], float(r["beta1"]))):
        lines.append(
            f"{METHOD_LABEL[row['method']]:7s} b={row['beta1']} batch={slug_batch(row['batch']):>4s} "
            f"lr={row['lr']:.4g} loss={row['loss']:.5f}"
        )
    ax_table.text(0, 1, "\n".join(lines), va="top", fontsize=8.3, family="monospace")

    fig.savefig(OUT / "qwen3_d12_combined_verbose_dashboard.png", dpi=180)
    fig.savefig(OUT / "qwen3_d12_combined_verbose_dashboard.pdf")
    plt.close(fig)


def plot_training_health(best: list[dict]) -> None:
    import matplotlib.pyplot as plt

    candidates = [row for row in best if row.get("train_losses")]
    if not candidates:
        return
    batches = sorted({row["batch"] for row in candidates})
    fig, axes = plt.subplots(1, len(batches), figsize=(6.2 * len(batches), 4.6), sharey=False)
    if len(batches) == 1:
        axes = [axes]
    fig.suptitle(
        "Qwen3 d12 training-curve health: best completed run per optimizer/batch",
        fontsize=16,
        fontweight="bold",
    )
    for ax, batch in zip(axes, batches):
        batch_rows = [row for row in candidates if row["batch"] == batch]
        for method in ["plain_muon", "kl_soap"]:
            method_rows = [row for row in batch_rows if row["method"] == method]
            if not method_rows:
                continue
            row = min(method_rows, key=lambda r: r["loss"])
            curve = row.get("train_losses") or []
            points = [
                (point.get("step", idx), point.get("loss"))
                for idx, point in enumerate(curve)
                if point.get("loss") is not None
            ]
            if not points:
                continue
            xs, ys = zip(*points)
            ax.plot(
                xs,
                ys,
                color=METHOD_COLOR[method],
                linewidth=2,
                label=f"{METHOD_LABEL[method]} b={row['beta1']} lr={row['lr']:.4g}",
            )
            if len(ys) >= 4:
                deltas = [ys[i] - ys[i - 1] for i in range(1, len(ys))]
                positive = sorted(d for d in deltas if d > 0)
                if positive:
                    threshold = max(0.15, positive[int(0.9 * (len(positive) - 1))] * 3)
                    spikes = [(xs[i], ys[i]) for i in range(1, len(ys)) if ys[i] - ys[i - 1] > threshold]
                    if spikes:
                        sx, sy = zip(*spikes)
                        ax.scatter(sx, sy, color="red", marker="x", s=80, linewidths=2)
        ax.set_title(f"batch={slug_batch(batch)}")
        ax.set_xlabel("Optimizer step")
        ax.set_ylabel("Training loss")
        ax.grid(True, alpha=0.22)
        ax.legend(fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(OUT / "qwen3_d12_training_health.png", dpi=180)
    fig.savefig(OUT / "qwen3_d12_training_health.pdf")
    plt.close(fig)


def html_table(rows: list[dict]) -> str:
    fields = ["method_label", "beta1", "batch_label", "lr", "loss", "root"]
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


def write_html(rows: list[dict], best: list[dict]) -> None:
    doc = f"""<!doctype html>
<meta charset="utf-8">
<title>Qwen3 d12 Muon / KL-SOAP dashboard</title>
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
<h1>Qwen3 d12 Muon / KL-SOAP fixed-beta1 dashboard</h1>
<p class="small">Rows: {len(rows)}. Lower final validation BPB/loss is better.</p>
<div class="card"><h2>Combined verbose dashboard</h2><img src="qwen3_d12_combined_verbose_dashboard.png"></div>
<div class="card"><h2>Training-curve health</h2><img src="qwen3_d12_training_health.png"></div>
<div class="card"><h2>Best rows</h2>{html_table(best)}</div>
</body>"""
    (OUT / "dashboard.html").write_text(doc)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = collect_rows()
    best = best_rows(rows)
    write_csv(OUT / "qwen3_d12_rows.csv", rows)
    write_csv(OUT / "qwen3_d12_best.csv", best)
    plot_dashboard(rows, best)
    plot_training_health(best)
    write_html(rows, best)
    print(f"rows={len(rows)} best={len(best)} out={OUT}")


if __name__ == "__main__":
    main()
