#!/usr/bin/env python3
from __future__ import annotations

import csv
import html
import json
import math
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "qwen3_muon_klsoap_dashboard"
HDFS_BASE = "hdfs://haruna/home/byte_data_seed/hdd_hldy/user/xingyu.dang/sigma-search-runs/search_evals"

RUN_ROOTS = [
    {
        "method": "plain_muon",
        "label": "Muon",
        "recipe": "momentum=0.95 static; beta2-only batch scaling",
        "root": f"{HDFS_BASE}/qwen3_d8_1x_64k512k_20260603_0626",
    },
    {
        "method": "kl_soap",
        "label": "KL-SOAP",
        "recipe": "beta1=0.90; beta2=shampoo_beta=0.95; beta2-only batch scaling",
        "root": f"{HDFS_BASE}/qwen3_d8_1x_64k512k_klsoap_b1p90_b2shamp0p95_20260603_0730",
    },
]

METHOD_LABEL = {
    "plain_muon": "Muon",
    "kl_soap": "KL-SOAP",
}
METHOD_COLOR = {
    "plain_muon": "#1f77b4",
    "kl_soap": "#d62728",
}
METHOD_MARKER = {
    "plain_muon": "o",
    "kl_soap": "s",
}


def run_text(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)


def hdfs_result_paths(root: str) -> list[str]:
    try:
        output = run_text(["hdfs", "dfs", "-ls", "-R", root])
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"failed to list HDFS root: {root}") from exc
    paths: list[str] = []
    for line in output.splitlines():
        if line.endswith("/result.json"):
            parts = line.split()
            if parts:
                paths.append(parts[-1])
    return sorted(paths)


def hdfs_json(path: str) -> dict:
    return json.loads(run_text(["hdfs", "dfs", "-cat", path]))


def slug_batch(batch: int) -> str:
    if batch >= 1_048_576 and batch % 1_048_576 == 0:
        return f"{batch // 1_048_576}M"
    if batch % 1024 == 0:
        return f"{batch // 1024}K"
    return str(batch)


def row_from_result(spec: dict, path: str, data: dict) -> dict | None:
    if data.get("error"):
        return None
    args = data.get("args", {})
    method = args.get("optimizer")
    if method != spec["method"]:
        return None
    score = data.get("score", data.get("val_bpb_final"))
    if score is None:
        return None
    batch = int(args["total_batch_size"])
    lr = float(args["matrix_lr"])
    return {
        "method": method,
        "method_label": METHOD_LABEL.get(method, method),
        "batch": batch,
        "batch_label": slug_batch(batch),
        "lr": lr,
        "loss": float(score),
        "architecture": args.get("architecture", ""),
        "depth": args.get("depth", ""),
        "chinchilla_mult": args.get("chinchilla_mult", ""),
        "weight_decay": args.get("weight_decay", ""),
        "muon_momentum": args.get("muon_momentum", ""),
        "muon_momentum_schedule": args.get("muon_momentum_schedule", ""),
        "optimizer_beta1": args.get("optimizer_beta1", ""),
        "optimizer_beta2": args.get("optimizer_beta2", ""),
        "shampoo_beta": args.get("shampoo_beta", ""),
        "batch_beta_align_mode": args.get("batch_beta_align_mode", ""),
        "recipe": spec["recipe"],
        "root": spec["root"].rsplit("/", 1)[-1],
        "path": path,
    }


def collect_rows() -> list[dict]:
    rows: list[dict] = []
    for spec in RUN_ROOTS:
        for path in hdfs_result_paths(spec["root"]):
            data = hdfs_json(path)
            row = row_from_result(spec, path, data)
            if row is not None:
                rows.append(row)
    return sorted(rows, key=lambda r: (r["method"], r["batch"], r["lr"]))


def best_rows(rows: list[dict]) -> list[dict]:
    best: list[dict] = []
    for method in sorted({r["method"] for r in rows}):
        for batch in sorted({r["batch"] for r in rows if r["method"] == method}):
            candidates = [r for r in rows if r["method"] == method and r["batch"] == batch]
            if candidates:
                best.append(min(candidates, key=lambda r: r["loss"]))
    return best


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "method", "method_label", "batch", "batch_label", "lr", "loss",
        "architecture", "depth", "chinchilla_mult", "weight_decay",
        "muon_momentum", "muon_momentum_schedule", "optimizer_beta1",
        "optimizer_beta2", "shampoo_beta", "batch_beta_align_mode",
        "recipe", "root", "path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def build_dashboard(rows: list[dict], best: list[dict]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    OUT.mkdir(parents=True, exist_ok=True)
    batches = sorted({r["batch"] for r in rows})
    methods = ["plain_muon", "kl_soap"]

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.05, 1.05, 0.9],
        left=0.06,
        right=0.985,
        bottom=0.065,
        top=0.875,
        wspace=0.22,
        hspace=0.38,
    )
    fig.suptitle(
        "Qwen3-like Architecture Optimizer Sweep: Muon vs KL-SOAP",
        fontsize=20,
        fontweight="bold",
        y=0.985,
    )
    subtitle = (
        "d8, 1x Chinchilla, batches 64K/512K. "
        "Muon: momentum=0.95 static. KL-SOAP: beta1=0.90, beta2=shampoo_beta=0.95. "
        "beta2/shampoo use batch half-life scaling."
    )
    fig.text(0.5, 0.947, subtitle, ha="center", va="top", fontsize=10.5, color="#444")

    for idx, batch in enumerate(batches):
        ax = fig.add_subplot(gs[0:2, idx])
        batch_rows = [r for r in rows if r["batch"] == batch]
        for method in methods:
            pts = sorted([r for r in batch_rows if r["method"] == method], key=lambda r: r["lr"])
            if not pts:
                continue
            xs = [r["lr"] for r in pts]
            ys = [r["loss"] for r in pts]
            label = METHOD_LABEL[method]
            ax.plot(
                xs,
                ys,
                marker=METHOD_MARKER[method],
                color=METHOD_COLOR[method],
                linewidth=2.4,
                markersize=6.5,
                label=label,
            )
            best_pt = min(pts, key=lambda r: r["loss"])
            ax.scatter(
                [best_pt["lr"]],
                [best_pt["loss"]],
                s=145,
                facecolors="none",
                edgecolors=METHOD_COLOR[method],
                linewidths=2.2,
                zorder=5,
            )
            ax.annotate(
                f"best {label}\nLR={best_pt['lr']:g}\nloss={best_pt['loss']:.4f}",
                xy=(best_pt["lr"], best_pt["loss"]),
                xytext=(-70, 14 if method == "plain_muon" else -46),
                textcoords="offset points",
                fontsize=9,
                color=METHOD_COLOR[method],
                ha="right",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=METHOD_COLOR[method], alpha=0.82),
            )
        ax.set_xscale("log", base=2)
        lrs = sorted({r["lr"] for r in batch_rows})
        ax.set_xticks(lrs)
        ax.set_xticklabels([f"{lr:g}" for lr in lrs])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.grid(True, which="both", alpha=0.24)
        ax.set_title(f"Loss vs LR, batch={slug_batch(batch)}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Matrix LR")
        ax.set_ylabel("Final validation BPB / loss")
        ax.legend(loc="best")

        ys_all = [r["loss"] for r in batch_rows]
        if ys_all:
            ymin, ymax = min(ys_all), max(ys_all)
            pad = max((ymax - ymin) * 0.10, 0.0005)
            ax.set_ylim(ymin - pad, ymax + pad)

    ax_loss = fig.add_subplot(gs[2, 0])
    ax_lr = fig.add_subplot(gs[2, 1])
    for method in methods:
        pts = sorted([r for r in best if r["method"] == method], key=lambda r: r["batch"])
        if not pts:
            continue
        xs = [r["batch"] for r in pts]
        labels = [slug_batch(x) for x in xs]
        ax_loss.plot(
            labels,
            [r["loss"] for r in pts],
            marker=METHOD_MARKER[method],
            color=METHOD_COLOR[method],
            linewidth=2.4,
            markersize=7,
            label=METHOD_LABEL[method],
        )
        ax_lr.plot(
            labels,
            [r["lr"] for r in pts],
            marker=METHOD_MARKER[method],
            color=METHOD_COLOR[method],
            linewidth=2.4,
            markersize=7,
            label=METHOD_LABEL[method],
        )
    ax_loss.set_title("Best Loss vs Batch", fontsize=13, fontweight="bold")
    ax_loss.set_xlabel("Batch")
    ax_loss.set_ylabel("Best final validation BPB / loss")
    ax_loss.grid(True, alpha=0.24)
    ax_loss.legend(loc="best")

    ax_lr.set_title("Best LR vs Batch", fontsize=13, fontweight="bold")
    ax_lr.set_xlabel("Batch")
    ax_lr.set_ylabel("Best matrix LR")
    ax_lr.grid(True, alpha=0.24)
    ax_lr.legend(loc="best")

    png = OUT / "qwen3_muon_klsoap_dashboard.png"
    pdf = OUT / "qwen3_muon_klsoap_dashboard.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)


def table(rows: list[dict]) -> str:
    cols = [
        "method_label", "batch_label", "lr", "loss", "optimizer_beta1",
        "optimizer_beta2", "shampoo_beta", "batch_beta_align_mode", "root",
    ]
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body = []
    for row in rows:
        cells = []
        for col in cols:
            value = row.get(col, "")
            if isinstance(value, float):
                value = f"{value:.8g}"
            cells.append(f"<td>{html.escape(str(value))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return "<table><thead><tr>" + head + "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"


def write_html(rows: list[dict], best: list[dict]) -> None:
    html_doc = f"""<!doctype html>
<meta charset="utf-8">
<title>Qwen3 Muon KL-SOAP Dashboard</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:24px;background:#f7f8fa;color:#202124}}
.card{{background:white;border:1px solid #ddd;border-radius:12px;padding:18px;margin:0 0 18px 0;box-shadow:0 1px 4px #0001}}
img{{max-width:100%;border:1px solid #eee;border-radius:8px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{border-bottom:1px solid #eee;padding:7px;text-align:left}}
th{{background:#fafafa}}
code{{background:#f1f3f4;padding:2px 4px;border-radius:4px}}
</style>
<h1>Qwen3 Muon vs KL-SOAP Dashboard</h1>
<div class="card">
  <p>Valid rows: {len(rows)}. Muon uses original qwen3 sweep; KL-SOAP uses corrected beta1=0.90, beta2=shampoo_beta=0.95 sweep.</p>
  <p>Artifacts: <code>qwen3_muon_klsoap_dashboard.png</code>, <code>qwen3_muon_klsoap_dashboard.pdf</code>, <code>qwen3_muon_klsoap_rows.csv</code>, <code>qwen3_muon_klsoap_best.csv</code>.</p>
</div>
<div class="card"><img src="qwen3_muon_klsoap_dashboard.png"></div>
<div class="card"><h2>Best Rows</h2>{table(best)}</div>
<div class="card"><h2>All Rows</h2>{table(rows)}</div>
"""
    (OUT / "dashboard.html").write_text(html_doc)


def main() -> None:
    rows = collect_rows()
    if not rows:
        raise SystemExit("No rows collected")
    best = best_rows(rows)
    write_csv(OUT / "qwen3_muon_klsoap_rows.csv", rows)
    write_csv(OUT / "qwen3_muon_klsoap_best.csv", best)
    build_dashboard(rows, best)
    write_html(rows, best)
    print(f"rows={len(rows)} best={len(best)} out={OUT}")
    for row in best:
        print(
            f"best {row['method_label']} batch={row['batch_label']} "
            f"lr={row['lr']:g} loss={row['loss']:.6f}"
        )


if __name__ == "__main__":
    main()
