#!/usr/bin/env python3
"""Plot dense Hessian/projection dynamics with matplotlib."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "metrics_cross_batch"


@dataclass(frozen=True)
class RunSpec:
    batch: str
    method: str
    path: Path


RUNS = [
    RunSpec("262K", "identity c=1", ROOT / "search_evals/v58_262k_1m_best_metrics_20260502_095901/streaming_identity_bsz262144_lr0p04_s42/result.json"),
    RunSpec("262K", "Top-Aware c=0.5", ROOT / "search_evals/v58_262k_1m_best_metrics_20260502_095901/top_aware_k1_a0p5_bsz262144_lr0p02_s42/result.json"),
    RunSpec("1M", "identity c=1", ROOT / "search_evals/v58_262k_1m_best_metrics_20260502_095901/streaming_identity_bsz1048576_lr0p08_s42/result.json"),
    RunSpec("1M", "Top-Aware c=0.5", ROOT / "search_evals/v58_262k_1m_best_metrics_20260502_095901/top_aware_k1_a0p5_bsz1048576_lr0p08_s42/result.json"),
    RunSpec("2M", "identity c=1", ROOT / "search_evals/v46_2m_best_metrics/streaming_identity_bsz2097152_lr0p08_s42/result.json"),
    RunSpec("2M", "Top-Aware c=0.5", ROOT / "search_evals/v46_2m_best_metrics/top_aware_k1_a0p5_bsz2097152_lr0p08_s42/result.json"),
    RunSpec("4M", "identity c=1", ROOT / "search_evals/v43_4m_best_metrics/streaming_identity_bsz4194304_lr0p02_s42/result.json"),
    RunSpec("4M", "Top-Aware c=0.5", ROOT / "search_evals/v43_4m_best_metrics/top_aware_k1_a0p5_bsz4194304_lr0p02_s42/result.json"),
    RunSpec("8M", "identity c=1", ROOT / "search_evals/v48_8m_best_metrics/streaming_identity_bsz8388608_lr0p02_s42/result.json"),
    RunSpec("8M", "Top-Aware c=0.5", ROOT / "search_evals/v48_8m_best_metrics/top_aware_k1_a0p5_bsz8388608_lr0p02_s42/result.json"),
]


def finite(value: object) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def first_finite(values: object) -> float | None:
    if not isinstance(values, list):
        return finite(values)
    for value in values:
        out = finite(value)
        if out is not None:
            return out
    return None


def load_series(spec: RunSpec) -> dict[str, object]:
    data = json.loads(spec.path.read_text())
    sharpness: list[tuple[int, float]] = []
    grad_align: list[tuple[int, float]] = []
    proj_corr: list[tuple[int, float]] = []

    for log in data.get("metric_logs") or []:
        step = int(log["step"])
        scalars = log.get("scalars") or {}
        corr = finite(scalars.get("gradient_projection_on_last_hessian_space_consecutive_correlation/selected_subspace"))
        if corr is not None:
            proj_corr.append((step, corr))

        hessian = log.get("hessian") or {}
        h_scalars = hessian.get("scalars") or {}
        sharp = finite(h_scalars.get("sharpness/selected_subspace"))
        if sharp is not None:
            sharpness.append((step, sharp))
        align = first_finite((hessian.get("vectors") or {}).get("gradient_hessian_alignment/selected_subspace"))
        if align is not None:
            grad_align.append((step, align))

    return {
        "batch": spec.batch,
        "method": spec.method,
        "path": str(spec.path.relative_to(ROOT)),
        "score": finite(data.get("score")),
        "steps_completed": data.get("steps_completed"),
        "metric_logs": len(data.get("metric_logs") or []),
        "sharpness": sharpness,
        "grad_align": grad_align,
        "proj_corr": proj_corr,
    }


def xy(points: list[tuple[int, float]]) -> tuple[list[int], list[float]]:
    return [p[0] for p in points], [p[1] for p in points]


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}" if abs(value) >= 0.01 else f"{value:.2e}"


def write_summary(rows: list[dict[str, object]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "hessian_dynamics_overview.json").write_text(json.dumps(rows, indent=2))
    lines = [
        "# Hessian Dynamics Overview",
        "",
        "![Hessian dynamics overview](hessian_dynamics_overview.png)",
        "",
        "| batch | method | val BPB | metric logs | sharpness last | sharpness max | grad-Hessian align last | projection-corr mean | projection-corr last10 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        sharp = [v for _, v in row["sharpness"]]  # type: ignore[index]
        align = [v for _, v in row["grad_align"]]  # type: ignore[index]
        corr = [v for _, v in row["proj_corr"]]  # type: ignore[index]
        lines.append(
            "| {batch} | {method} | {score} | {logs} | {sharp_last} | {sharp_max} | {align_last} | {corr_mean} | {corr_last} |".format(
                batch=row["batch"],
                method=row["method"],
                score=fmt(row["score"]),  # type: ignore[arg-type]
                logs=row["metric_logs"],
                sharp_last=fmt(sharp[-1] if sharp else None),
                sharp_max=fmt(max(sharp) if sharp else None),
                align_last=fmt(align[-1] if align else None),
                corr_mean=fmt(mean(corr)),
                corr_last=fmt(mean(corr[-10:])),
            )
        )
    lines.append("")
    (OUT_DIR / "hessian_dynamics_overview.md").write_text("\n".join(lines))


def plot(rows: list[dict[str, object]]) -> None:
    batches = ["262K", "1M", "2M", "4M", "8M"]
    metrics = [
        ("sharpness", "Hessian max eigenvalue", None),
        ("grad_align", "gradient-Hessian top-1 cosine", (-1.05, 1.05)),
        ("proj_corr", "consecutive projection cosine", (-1.05, 1.05)),
    ]
    colors = {"identity c=1": "#174a7c", "Top-Aware c=0.5": "#c94f1a"}

    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 220,
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.labelsize": 9.5,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("default")

    fig, axes = plt.subplots(3, 5, figsize=(18.0, 8.6), sharex="col", constrained_layout=True)
    fig.suptitle("Hessian dynamics overview: identity vs Top-Aware c=0.5", fontsize=15, fontweight="bold")
    for col, batch in enumerate(batches):
        batch_rows = [r for r in rows if r["batch"] == batch]
        for row_idx, (key, ylabel, ylim) in enumerate(metrics):
            ax = axes[row_idx][col]
            for run in batch_rows:
                xs, ys = xy(run[key])  # type: ignore[arg-type]
                if not xs:
                    continue
                method = str(run["method"])
                ax.plot(
                    xs,
                    ys,
                    color=colors[method],
                    label=method,
                    marker="o" if key != "proj_corr" else None,
                    markersize=3.2,
                    linewidth=2.0 if key != "proj_corr" else 1.1,
                    alpha=0.92 if key != "proj_corr" else 0.78,
                )
            ax.axhline(0.0, color="#777", linewidth=0.8, linestyle="--", alpha=0.55)
            if ylim is not None:
                ax.set_ylim(*ylim)
            if row_idx == 0:
                ax.set_title(f"batch {batch}")
            if col == 0:
                ax.set_ylabel(ylabel)
            if row_idx == 2:
                ax.set_xlabel("optimizer step")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=2, frameon=False)
    fig.savefig(OUT_DIR / "hessian_dynamics_overview.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "hessian_dynamics_overview.pdf", bbox_inches="tight")


def main() -> None:
    rows = [load_series(spec) for spec in RUNS]
    write_summary(rows)
    plot(rows)
    print(OUT_DIR / "hessian_dynamics_overview.png")
    print(OUT_DIR / "hessian_dynamics_overview.pdf")
    print(OUT_DIR / "hessian_dynamics_overview.md")


if __name__ == "__main__":
    main()
