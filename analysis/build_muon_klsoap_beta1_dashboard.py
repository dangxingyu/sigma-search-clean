#!/usr/bin/env python3
from __future__ import annotations

import csv
import html
import json
import math
import re
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "muon_klsoap_beta1_dashboard"
RESULTS = ROOT / "results"
HDFS_BASE = "hdfs://haruna/home/byte_data_seed/hdd_hldy/user/xingyu.dang/sigma-search-runs/search_evals"

CURRENT_ROOTS = [
    ("plain_muon", "Muon fixed beta1=0.85", "0.85", "A100", f"{HDFS_BASE}/beta1_d8_a100_20260530_0817_plain_muon_b0p85"),
    ("plain_muon", "Muon fixed beta1=0.90", "0.90", "A100", f"{HDFS_BASE}/beta1_d8_a100_20260530_0817_plain_muon_b0p9"),
    ("plain_muon", "Muon fixed beta1=0.95", "0.95", "A100", f"{HDFS_BASE}/faircmp_d8_a100_20260530_0733_plain_muon"),
    ("plain_muon", "Muon fixed beta1=0.97", "0.97", "A100", f"{HDFS_BASE}/beta1_d8_a100_20260530_0817_plain_muon_b0p97"),
    ("plain_muon", "Muon fixed beta1=0.85", "0.85", "H20-ext", f"{HDFS_BASE}/lrext_64k128k_8gpu_20260602_0146_us_h20_plain_muon_b0p85"),
    ("plain_muon", "Muon fixed beta1=0.90", "0.90", "H20-ext", f"{HDFS_BASE}/lrext_64k128k_8gpu_20260602_0146_us_h20_plain_muon_b0p9"),
    ("plain_muon", "Muon fixed beta1=0.95", "0.95", "H20-ext", f"{HDFS_BASE}/lrext_64k128k_8gpu_20260602_0146_us_h20_plain_muon_b0p95"),
    ("plain_muon", "Muon fixed beta1=0.97", "0.97", "H20-ext", f"{HDFS_BASE}/lrext_64k128k_8gpu_20260602_0146_us_h20_plain_muon_b0p97"),
    ("kl_soap", "KL-SOAP fixed beta1=0.85", "0.85", "A100/H20", f"{HDFS_BASE}/klbeta1_case_d8_a100_20260530_1018_kl_soap_b0p85"),
    ("kl_soap", "KL-SOAP fixed beta1=0.85", "0.85", "A100/H20", f"{HDFS_BASE}/klbeta1_case_d8_h20_20260530_1023_kl_soap_b0p85"),
    ("kl_soap", "KL-SOAP fixed beta1=0.90", "0.90", "H20", f"{HDFS_BASE}/klbeta1_case_d8_h20_20260530_1023_kl_soap_b0p9"),
    ("kl_soap", "KL-SOAP fixed beta1=0.95", "0.95", "A100", f"{HDFS_BASE}/faircmp_d8_a100_20260530_0733_kl_soap"),
    ("kl_soap", "KL-SOAP fixed beta1=0.97", "0.97", "H20", f"{HDFS_BASE}/klbeta1_case_d8_h20_20260530_1023_kl_soap_b0p97"),
    ("kl_soap", "KL-SOAP fixed beta1=0.85", "0.85", "H20-ext", f"{HDFS_BASE}/lrext_64k128k_8gpu_20260602_0146_seedx_h20_kl_soap_b0p85"),
    ("kl_soap", "KL-SOAP fixed beta1=0.90", "0.90", "H20-ext", f"{HDFS_BASE}/lrext_64k128k_8gpu_20260602_0146_seedx_h20_kl_soap_b0p9"),
    ("kl_soap", "KL-SOAP fixed beta1=0.95", "0.95", "H20-ext", f"{HDFS_BASE}/lrext_64k128k_8gpu_20260602_0146_seedx_h20_kl_soap_b0p95"),
    ("kl_soap", "KL-SOAP fixed beta1=0.97", "0.97", "H20-ext", f"{HDFS_BASE}/lrext_64k128k_8gpu_20260602_0146_seedx_h20_kl_soap_b0p97"),
    ("kl_soap", "KL-SOAP fixed beta1=0.95", "0.95", "H20-ext-moved", f"{HDFS_BASE}/lrext_64k128k_8gpu_usmove_20260602_0620_kl_soap_b0p95"),
    ("kl_soap", "KL-SOAP fixed beta1=0.97", "0.97", "H20-ext-moved", f"{HDFS_BASE}/lrext_64k128k_8gpu_usmove_20260602_0620_kl_soap_b0p97"),
]


def run_text(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)


def hdfs_result_paths(root: str) -> list[str]:
    try:
        output = run_text(["hdfs", "dfs", "-ls", "-R", root])
    except subprocess.CalledProcessError:
        return []
    paths: list[str] = []
    for line in output.splitlines():
        if not line.endswith("/result.json"):
            continue
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


def nice_float(x: float) -> str:
    return f"{x:g}"


def row_from_result(
    *,
    expected_method: str,
    config: str,
    beta1: str,
    hardware: str,
    root: str,
    path: str,
    data: dict,
) -> dict:
    args = data.get("args", {})
    method = args.get("optimizer") or expected_method
    batch = int(args.get("total_batch_size"))
    lr = float(args.get("matrix_lr"))
    score = float(data.get("score", data.get("val_bpb_final")))
    return {
        "method": method,
        "config": config,
        "beta1": beta1,
        "batch": str(batch),
        "batch_label": slug_batch(batch),
        "lr": f"{lr:g}",
        "loss": f"{score:.12g}",
        "weight_decay": str(args.get("weight_decay", "")),
        "momentum": str(args.get("muon_momentum", "")) if method == "plain_muon" else "",
        "momentum_schedule": str(args.get("muon_momentum_schedule", "")) if method == "plain_muon" else "",
        "optimizer_beta1": str(args.get("optimizer_beta1", "")),
        "optimizer_beta2": str(args.get("optimizer_beta2", "")),
        "shampoo_beta": str(args.get("shampoo_beta", "")),
        "adam_lr_mode": str(args.get("adam_lr_mode", "")),
        "batch_beta_align_mode": str(args.get("batch_beta_align_mode", "")),
        "hardware": hardware,
        "source": "current_beta1_sweep",
        "root": root.rsplit("/", 1)[-1],
        "path": path,
    }


def collect_current_rows() -> list[dict]:
    rows: list[dict] = []
    for method, config, beta1, hardware, root in CURRENT_ROOTS:
        for path in hdfs_result_paths(root):
            data = hdfs_json(path)
            if data.get("error"):
                continue
            rows.append(row_from_result(
                expected_method=method,
                config=config,
                beta1=beta1,
                hardware=hardware,
                root=root,
                path=path,
                data=data,
            ))
    # If the same case exists in both an aborted A100 root and a replacement H20 root,
    # prefer the replacement H20/current result for that method/config/batch/lr.
    best: dict[tuple[str, str, str, str], dict] = {}
    priority = {"H20-ext-moved": 5, "H20-ext": 4, "H20": 3, "A100/H20": 2, "A100": 1}
    for row in rows:
        key = (row["method"], row["config"], row["batch"], row["lr"])
        old = best.get(key)
        if old is None or priority.get(row["hardware"], 0) >= priority.get(old["hardware"], 0):
            best[key] = row
    return sorted(best.values(), key=lambda r: (r["method"], float(r["beta1"]), int(r["batch"]), float(r["lr"])))


def read_old_rows() -> list[dict]:
    path = RESULTS / "plain_muon_klsoap_lr_dashboard_rows.csv"
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                **r,
                "batch_label": slug_batch(int(r["batch"])),
                "beta1": r.get("momentum") or "",
                "optimizer_beta1": "",
                "optimizer_beta2": "",
                "shampoo_beta": "",
                "adam_lr_mode": "",
                "batch_beta_align_mode": "",
                "hardware": "",
                "source": r.get("source") or "historical",
            })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "method", "config", "beta1", "batch", "batch_label", "lr", "loss",
        "weight_decay", "momentum", "momentum_schedule", "optimizer_beta1",
        "optimizer_beta2", "shampoo_beta", "adam_lr_mode", "batch_beta_align_mode",
        "hardware", "source", "root", "path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def flt(row: dict, key: str) -> float:
    return float(row[key])


def color_for(config: str) -> str:
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    return colors[abs(hash(config)) % len(colors)]


def svg_header(w: int, h: int) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
<rect width="100%" height="100%" fill="white"/>
<style>text{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;fill:#202124}} .small{{font-size:11px;fill:#5f6368}} .title{{font-size:18px;font-weight:700}} .subtitle{{font-size:12px;fill:#5f6368}} .axis{{stroke:#555;stroke-width:1}} .grid{{stroke:#ddd;stroke-width:.7}}</style>
'''


def plot_lr_facets(rows: list[dict], *, method: str, title: str, path: Path) -> None:
    sub_rows = [r for r in rows if r["method"] == method]
    if not sub_rows:
        return
    batches = sorted({int(r["batch"]) for r in sub_rows})
    configs = sorted({r["config"] for r in sub_rows})
    cols = min(3, len(batches))
    panel_w, panel_h = 470, 330
    top, legend_h = 75, 120
    w = cols * panel_w + 40
    h = top + math.ceil(len(batches) / cols) * panel_h + legend_h
    parts = [svg_header(w, h), f'<text x="24" y="30" class="title">{esc(title)}</text>',
             f'<text x="24" y="50" class="subtitle">x=LR, y=final val BPB/loss. Open marker is best LR per config within each batch.</text>']
    for idx, batch in enumerate(batches):
        row, col = divmod(idx, cols)
        x0, y0 = 30 + col * panel_w, top + row * panel_h
        gx, gy = x0 + 62, y0 + 25
        pw, ph = 350, 235
        rows_b = [r for r in sub_rows if int(r["batch"]) == batch]
        xs = [math.log2(flt(r, "lr")) for r in rows_b]
        ys = [flt(r, "loss") for r in rows_b]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        if xmin == xmax:
            xmax = xmin + 1
        pad = (ymax - ymin) * 0.08 or 0.001
        ymin -= pad
        ymax += pad
        sx = lambda lr: gx + (math.log2(lr) - xmin) / (xmax - xmin) * pw
        sy = lambda y: gy + ph - (y - ymin) / (ymax - ymin) * ph
        parts.append(f'<text x="{gx}" y="{y0 + 8}" font-size="14" font-weight="700">batch={slug_batch(batch)}</text>')
        for lr in sorted({flt(r, "lr") for r in rows_b}):
            x = sx(lr)
            parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{gy}" y2="{gy+ph}" class="grid"/><text x="{x:.1f}" y="{gy+ph+16}" class="small" text-anchor="middle">{nice_float(lr)}</text>')
        for i in range(5):
            val = ymin + (ymax - ymin) * i / 4
            y = sy(val)
            parts.append(f'<line x1="{gx}" x2="{gx+pw}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/><text x="{gx-8}" y="{y+4:.1f}" class="small" text-anchor="end">{val:.4f}</text>')
        parts.append(f'<line x1="{gx}" x2="{gx}" y1="{gy}" y2="{gy+ph}" class="axis"/><line x1="{gx}" x2="{gx+pw}" y1="{gy+ph}" y2="{gy+ph}" class="axis"/>')
        for config in configs:
            pts = sorted([r for r in rows_b if r["config"] == config], key=lambda r: flt(r, "lr"))
            if not pts:
                continue
            c = color_for(config)
            coords = [(sx(flt(r, "lr")), sy(flt(r, "loss"))) for r in pts]
            if len(coords) > 1:
                parts.append('<polyline fill="none" stroke="{}" stroke-width="2" points="{}"/>'.format(c, " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)))
            for x, y in coords:
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.2" fill="{c}"/>')
            best = min(pts, key=lambda r: flt(r, "loss"))
            parts.append(f'<circle cx="{sx(flt(best, "lr")):.1f}" cy="{sy(flt(best, "loss")):.1f}" r="7" fill="white" stroke="{c}" stroke-width="2"/>')
    lx, ly = 35, h - 88
    x = lx
    for config in configs:
        if x > w - 250:
            x = lx
            ly += 24
        c = color_for(config)
        parts.append(f'<circle cx="{x+7}" cy="{ly}" r="5" fill="{c}"/><text x="{x+20}" y="{ly+4}" class="small">{esc(config)}</text>')
        x += 245
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def best_rows(rows: list[dict]) -> list[dict]:
    best: dict[tuple[str, str, str], dict] = {}
    for r in rows:
        key = (r["method"], r["config"], r["batch"])
        if key not in best or flt(r, "loss") < flt(best[key], "loss"):
            best[key] = r
    return sorted(best.values(), key=lambda r: (r["method"], r["config"], int(r["batch"])))


def plot_optimal(best: list[dict], *, method: str, metric: str, title: str, path: Path) -> None:
    sub = [r for r in best if r["method"] == method]
    if not sub:
        return
    batches = sorted({int(r["batch"]) for r in sub})
    configs = sorted({r["config"] for r in sub})
    w, h = 930, 500
    gx, gy, pw, ph = 80, 75, 760, 310
    vals = [flt(r, metric) for r in sub]
    if metric == "lr":
        vals = [math.log2(v) for v in vals]
    ymin, ymax = min(vals), max(vals)
    pad = (ymax - ymin) * 0.08 or 0.01
    ymin -= pad
    ymax += pad
    sx = lambda b: gx + (batches.index(int(b)) / max(1, len(batches) - 1)) * pw
    def sy(v: float) -> float:
        vv = math.log2(v) if metric == "lr" else v
        return gy + ph - (vv - ymin) / (ymax - ymin) * ph
    parts = [svg_header(w, h), f'<text x="24" y="32" class="title">{esc(title)}</text>']
    for batch in batches:
        x = sx(batch)
        parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{gy}" y2="{gy+ph}" class="grid"/><text x="{x:.1f}" y="{gy+ph+18}" class="small" text-anchor="middle">{slug_batch(batch)}</text>')
    for i in range(5):
        vv = ymin + (ymax - ymin) * i / 4
        label = nice_float(2 ** vv) if metric == "lr" else f"{vv:.5f}"
        y = gy + ph - (vv - ymin) / (ymax - ymin) * ph
        parts.append(f'<line x1="{gx}" x2="{gx+pw}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/><text x="{gx-8}" y="{y+4:.1f}" class="small" text-anchor="end">{label}</text>')
    parts.append(f'<line x1="{gx}" x2="{gx}" y1="{gy}" y2="{gy+ph}" class="axis"/><line x1="{gx}" x2="{gx+pw}" y1="{gy+ph}" y2="{gy+ph}" class="axis"/>')
    for config in configs:
        pts = sorted([r for r in sub if r["config"] == config], key=lambda r: int(r["batch"]))
        c = color_for(config)
        coords = [(sx(int(r["batch"])), sy(flt(r, metric))) for r in pts]
        if len(coords) > 1:
            parts.append('<polyline fill="none" stroke="{}" stroke-width="2" points="{}"/>'.format(c, " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)))
        for x, y in coords:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{c}"/>')
    lx, ly = 55, h - 78
    x = lx
    for config in configs:
        if x > w - 250:
            x = lx
            ly += 23
        c = color_for(config)
        parts.append(f'<circle cx="{x+7}" cy="{ly}" r="5" fill="{c}"/><text x="{x+20}" y="{ly+4}" class="small">{esc(config)}</text>')
        x += 245
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def plot_loss_over_momentum(best: list[dict], *, method: str, title: str, path: Path) -> None:
    sub = [r for r in best if r["method"] == method and r.get("beta1")]
    if not sub:
        return
    batches = sorted({int(r["batch"]) for r in sub})
    betas = sorted({float(r["beta1"]) for r in sub})
    w, h = 980, 540
    gx, gy, pw, ph = 90, 80, 760, 320
    losses = [flt(r, "loss") for r in sub]
    ymin, ymax = min(losses), max(losses)
    pad = (ymax - ymin) * 0.10 or 0.001
    ymin -= pad
    ymax += pad

    def sx(beta: float) -> float:
        if len(betas) == 1:
            return gx + pw / 2
        return gx + (betas.index(beta) / (len(betas) - 1)) * pw

    def sy(loss: float) -> float:
        return gy + ph - (loss - ymin) / (ymax - ymin) * ph

    parts = [
        svg_header(w, h),
        f'<text x="24" y="32" class="title">{esc(title)}</text>',
        '<text x="24" y="53" class="subtitle">For each batch and fixed beta1/momentum, y is the best loss over the LR grid.</text>',
    ]
    for beta in betas:
        x = sx(beta)
        parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{gy}" y2="{gy+ph}" class="grid"/><text x="{x:.1f}" y="{gy+ph+22}" class="small" text-anchor="middle">{beta:.2f}</text>')
    for i in range(5):
        val = ymin + (ymax - ymin) * i / 4
        y = sy(val)
        parts.append(f'<line x1="{gx}" x2="{gx+pw}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/><text x="{gx-8}" y="{y+4:.1f}" class="small" text-anchor="end">{val:.5f}</text>')
    parts.append(f'<line x1="{gx}" x2="{gx}" y1="{gy}" y2="{gy+ph}" class="axis"/><line x1="{gx}" x2="{gx+pw}" y1="{gy+ph}" y2="{gy+ph}" class="axis"/>')
    parts.append(f'<text x="{gx + pw / 2:.1f}" y="{gy+ph+50}" class="small" text-anchor="middle">fixed beta1 / momentum</text>')
    parts.append(f'<text x="20" y="{gy + ph / 2:.1f}" class="small" text-anchor="middle" transform="rotate(-90 20 {gy + ph / 2:.1f})">best final validation BPB / loss</text>')

    for batch in batches:
        pts = sorted([r for r in sub if int(r["batch"]) == batch], key=lambda r: float(r["beta1"]))
        c = color_for(slug_batch(batch))
        coords = [(sx(float(r["beta1"])), sy(flt(r, "loss"))) for r in pts]
        if len(coords) > 1:
            parts.append('<polyline fill="none" stroke="{}" stroke-width="2.4" points="{}"/>'.format(c, " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)))
        for x, y in coords:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.2" fill="{c}"/>')

    lx, ly = 55, h - 76
    x = lx
    for batch in batches:
        c = color_for(slug_batch(batch))
        parts.append(f'<circle cx="{x+7}" cy="{ly}" r="5" fill="{c}"/><text x="{x+20}" y="{ly+4}" class="small">batch={slug_batch(batch)}</text>')
        x += 150
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def table(rows: list[dict], fields: list[str]) -> str:
    lines = ["<table><thead><tr>" + "".join(f"<th>{esc(f)}</th>" for f in fields) + "</tr></thead><tbody>"]
    for r in rows:
        lines.append("<tr>" + "".join(f"<td>{esc(r.get(f, ''))}</td>" for f in fields) + "</tr>")
    lines.append("</tbody></table>")
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    current = collect_current_rows()
    all_rows = sorted(read_old_rows() + current, key=lambda r: (r["method"], r["config"], int(r["batch"]), float(r["lr"])))
    best = best_rows(current)
    write_csv(RESULTS / "muon_klsoap_beta1_dashboard_rows.csv", current)
    write_csv(OUT / "current_beta1_rows.csv", current)
    write_csv(OUT / "all_rows_with_historical.csv", all_rows)
    write_csv(OUT / "current_beta1_best.csv", best)

    plot_lr_facets(current, method="plain_muon", title="Current Muon fixed-beta1 sweep", path=OUT / "current_muon_lr_loss.svg")
    plot_lr_facets(current, method="kl_soap", title="Current KL-SOAP fixed-beta1 sweep", path=OUT / "current_klsoap_lr_loss.svg")
    plot_lr_facets(all_rows, method="plain_muon", title="Muon LR-loss with historical context", path=OUT / "historical_muon_lr_loss.svg")
    plot_lr_facets(all_rows, method="kl_soap", title="KL-SOAP LR-loss with historical context", path=OUT / "historical_klsoap_lr_loss.svg")
    plot_optimal(best, method="plain_muon", metric="loss", title="Muon best loss vs batch", path=OUT / "current_muon_best_loss.svg")
    plot_optimal(best, method="plain_muon", metric="lr", title="Muon optimal LR vs batch", path=OUT / "current_muon_best_lr.svg")
    plot_optimal(best, method="kl_soap", metric="loss", title="KL-SOAP best loss vs batch", path=OUT / "current_klsoap_best_loss.svg")
    plot_optimal(best, method="kl_soap", metric="lr", title="KL-SOAP optimal LR vs batch", path=OUT / "current_klsoap_best_lr.svg")
    plot_loss_over_momentum(best, method="plain_muon", title="Muon best loss vs fixed momentum", path=OUT / "current_muon_best_loss_over_momentum.svg")
    plot_loss_over_momentum(best, method="kl_soap", title="KL-SOAP best loss vs fixed beta1", path=OUT / "current_klsoap_best_loss_over_beta1.svg")

    counts = defaultdict(int)
    for r in current:
        counts[(r["method"], r["config"])] += 1
    expected = {
        "Muon fixed beta1=0.85": 15,
        "Muon fixed beta1=0.90": 15,
        "Muon fixed beta1=0.95": 15,
        "Muon fixed beta1=0.97": 15,
        "KL-SOAP fixed beta1=0.85": 15,
        "KL-SOAP fixed beta1=0.90": 15,
        "KL-SOAP fixed beta1=0.95": 15,
        "KL-SOAP fixed beta1=0.97": 15,
    }
    # The 64K/128K LR-extension adds four higher-LR points for each batch.
    # 32K intentionally remains at the original five-point grid for now.
    for config in list(expected):
        expected[config] = 23
    coverage_rows = []
    for config, want in expected.items():
        method = "plain_muon" if config.startswith("Muon") else "kl_soap"
        got = counts.get((method, config), 0)
        coverage_rows.append({
            "method": method,
            "config": config,
            "expected": str(want),
            "completed": str(got),
            "status": "complete" if got == want else "incomplete",
        })
    with (OUT / "coverage.csv").open("w", newline="") as f:
        fields = ["method", "config", "expected", "completed", "status"]
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(coverage_rows)

    html_doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>Muon / KL-SOAP beta1 dashboard</title><style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:28px;color:#202124}} h1{{margin-bottom:.2em}} .card{{border:1px solid #ddd;border-radius:12px;padding:16px;margin:18px 0;box-shadow:0 1px 3px rgba(0,0,0,.05)}} img{{max-width:100%;height:auto;border:1px solid #eee;border-radius:8px;background:white;margin:8px 0}} table{{border-collapse:collapse;font-size:13px;width:100%;margin:10px 0}} th,td{{border:1px solid #ddd;padding:6px 8px;text-align:left}} th{{background:#f6f8fa}} code{{background:#f6f8fa;padding:2px 4px;border-radius:4px}} .small{{color:#666;font-size:13px}}
</style></head><body>
<h1>Muon / KL-SOAP fixed-beta1 dashboard</h1>
<p class="small">Generated from completed HDFS <code>result.json</code> files. Lower final validation BPB/loss is better.</p>
<div class="card"><h2>Coverage</h2>{table(coverage_rows, ["method", "config", "expected", "completed", "status"])}</div>
<div class="card"><h2>Current Muon fixed-beta1 sweep</h2><img src="current_muon_lr_loss.svg"><img src="current_muon_best_loss.svg"><img src="current_muon_best_lr.svg"><img src="current_muon_best_loss_over_momentum.svg"></div>
<div class="card"><h2>Current KL-SOAP fixed-beta1 sweep</h2><img src="current_klsoap_lr_loss.svg"><img src="current_klsoap_best_loss.svg"><img src="current_klsoap_best_lr.svg"><img src="current_klsoap_best_loss_over_beta1.svg"></div>
<div class="card"><h2>Best rows from current sweep</h2>{table(best, ["method", "config", "batch_label", "lr", "loss", "hardware", "path"])}</div>
<div class="card"><h2>Historical context</h2><p class="small">Includes older rows from <code>results/plain_muon_klsoap_lr_dashboard_rows.csv</code> plus current fixed-beta1 rows.</p><img src="historical_muon_lr_loss.svg"><img src="historical_klsoap_lr_loss.svg"></div>
</body></html>"""
    (OUT / "dashboard.html").write_text(html_doc)
    print(json.dumps({
        "out_dir": str(OUT),
        "dashboard": str(OUT / "dashboard.html"),
        "current_rows": len(current),
        "historical_plus_current_rows": len(all_rows),
        "coverage": coverage_rows,
    }, indent=2))


if __name__ == "__main__":
    main()
