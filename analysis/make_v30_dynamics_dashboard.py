#!/usr/bin/env python3
"""Summarize v30 Muon-family dynamics metric logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


CASE_RE = re.compile(
    r"(?P<method>native_muon|streaming_identity|top_aware(?:_k(?P<top_k>\d+)_a(?P<alpha>[\dp]+))?)"
    r"_bsz(?P<batch>\d+)_lr(?P<lr>[\dp]+)_s(?P<seed>\d+)"
)


def parse_float_slug(text: str | None) -> float | None:
    if text is None:
        return None
    return float(text.replace("p", "."))


def median(values: list[float]) -> float:
    clean = sorted(v for v in values if v is not None and math.isfinite(v))
    if not clean:
        return float("nan")
    mid = len(clean) // 2
    if len(clean) % 2:
        return clean[mid]
    return 0.5 * (clean[mid - 1] + clean[mid])


def summarize_entry(entry: dict) -> dict:
    per_module = entry.get("per_module", {})
    spectra = []
    rms = []
    ratios = []
    for row in per_module.values():
        svals = row.get("muon_singular_values") or []
        if svals:
            spectra.append(float(svals[0]))
            denom = sum(float(x) for x in svals) / len(svals)
            ratios.append(float(svals[0]) / denom if denom else float("nan"))
        if "momentum_after_nesterov_rms" in row:
            rms.append(float(row["momentum_after_nesterov_rms"]))

    split_vals = []
    for values in entry.get("vectors", {}).items():
        key, vec = values
        if key.startswith("variance_of_u_i/") and vec:
            split_vals.append(float(vec[0]))

    sharpness = []
    hess = entry.get("hessian", {})
    for module in hess.get("per_module", {}).values():
        if "sharpness" in module:
            sharpness.append(float(module["sharpness"]))

    return {
        "step": int(entry["step"]),
        "train_loss": entry.get("scalars", {}).get("train/loss", float("nan")),
        "median_s1": median(spectra),
        "median_s1_over_topk_mean": median(ratios),
        "median_mtilde_rms": median(rms),
        "median_split_top1_alignment": median(split_vals),
        "median_sharpness": median(sharpness),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    series = {}
    for result in sorted(args.root.glob("*/result.json")):
        match = CASE_RE.search(result.parent.name)
        if not match:
            continue
        data = json.loads(result.read_text())
        info = match.groupdict()
        method = info["method"]
        if method.startswith("top_aware"):
            method = "top_aware_muon"
        meta = {
            "case": result.parent.name,
            "method": method,
            "batch": int(info["batch"]),
            "lr": parse_float_slug(info["lr"]),
            "seed": int(info["seed"]),
            "top_k": "" if info.get("top_k") is None else int(info["top_k"]),
            "alpha": "" if info.get("alpha") is None else parse_float_slug(info["alpha"]),
            "score": data.get("score"),
            "error": data.get("error"),
            "num_metric_entries": len(data.get("metric_logs", [])),
        }
        rows.append(meta)
        series[result.parent.name] = [summarize_entry(e) for e in data.get("metric_logs", [])]

    out_dir = args.root / "v30_dashboard"
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["case", "method", "batch", "lr", "seed", "top_k", "alpha", "score", "error", "num_metric_entries"]
    with (out_dir / "summary_rows.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with (out_dir / "metric_series.json").open("w") as f:
        json.dump(series, f, indent=2)

    if not series:
        print(f"no metric logs found under {args.root}")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    batches = sorted({r["batch"] for r in rows})
    metrics = [
        ("train_loss", "Train loss"),
        ("median_s1", "Median top singular value of M'"),
        ("median_s1_over_topk_mean", "Median s1 / mean(top-k singular values)"),
        ("median_split_top1_alignment", "Median split top-1 alignment"),
        ("median_sharpness", "Median Hessian sharpness"),
    ]

    fig, axes = plt.subplots(len(batches), len(metrics), figsize=(4.6 * len(metrics), 3.1 * len(batches)), squeeze=False)
    for row_idx, batch in enumerate(batches):
        batch_rows = [r for r in rows if r["batch"] == batch]
        for col_idx, (metric, title) in enumerate(metrics):
            ax = axes[row_idx][col_idx]
            for meta in sorted(batch_rows, key=lambda r: r["case"]):
                points = series.get(meta["case"], [])
                xs = [p["step"] for p in points if math.isfinite(float(p.get(metric, float("nan"))))]
                ys = [float(p[metric]) for p in points if math.isfinite(float(p.get(metric, float("nan"))))]
                if not xs:
                    continue
                label = meta["method"]
                if meta["method"] == "top_aware_muon":
                    label += f" a={meta['alpha']}"
                ax.plot(xs, ys, label=label, linewidth=1.2)
            ax.set_title(f"{batch // 1024}K | {title}")
            ax.set_xlabel("step")
            ax.grid(True, alpha=0.25)
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "dynamics_dashboard.png", dpi=180)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
