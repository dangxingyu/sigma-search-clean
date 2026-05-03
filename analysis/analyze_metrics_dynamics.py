#!/usr/bin/env python3
"""Analyze dense optimizer-dynamics metrics from clean d8 runs."""

from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "metrics_dynamics_analysis"


@dataclass(frozen=True)
class RunSpec:
    batch: int
    batch_label: str
    method: str
    lr: float
    path: Path


RUNS = [
    RunSpec(
        262_144,
        "262K",
        "identity c=1",
        0.04,
        ROOT
        / "search_evals/v58_262k_1m_best_metrics_20260502_095901/streaming_identity_bsz262144_lr0p04_s42/result.json",
    ),
    RunSpec(
        262_144,
        "262K",
        "Top-Aware c=0.5",
        0.02,
        ROOT
        / "search_evals/v58_262k_1m_best_metrics_20260502_095901/top_aware_k1_a0p5_bsz262144_lr0p02_s42/result.json",
    ),
    RunSpec(
        1_048_576,
        "1M",
        "identity c=1",
        0.08,
        ROOT
        / "search_evals/v58_262k_1m_best_metrics_20260502_095901/streaming_identity_bsz1048576_lr0p08_s42/result.json",
    ),
    RunSpec(
        1_048_576,
        "1M",
        "Top-Aware c=0.5",
        0.08,
        ROOT
        / "search_evals/v58_262k_1m_best_metrics_20260502_095901/top_aware_k1_a0p5_bsz1048576_lr0p08_s42/result.json",
    ),
    RunSpec(
        2_097_152,
        "2M",
        "identity c=1",
        0.08,
        ROOT / "search_evals/v46_2m_best_metrics/streaming_identity_bsz2097152_lr0p08_s42/result.json",
    ),
    RunSpec(
        2_097_152,
        "2M",
        "Top-Aware c=0.5",
        0.08,
        ROOT / "search_evals/v46_2m_best_metrics/top_aware_k1_a0p5_bsz2097152_lr0p08_s42/result.json",
    ),
    RunSpec(
        4_194_304,
        "4M",
        "identity c=1",
        0.02,
        ROOT / "search_evals/v43_4m_best_metrics/streaming_identity_bsz4194304_lr0p02_s42/result.json",
    ),
    RunSpec(
        4_194_304,
        "4M",
        "Top-Aware c=0.5",
        0.02,
        ROOT / "search_evals/v43_4m_best_metrics/top_aware_k1_a0p5_bsz4194304_lr0p02_s42/result.json",
    ),
    RunSpec(
        8_388_608,
        "8M",
        "identity c=1",
        0.02,
        ROOT / "search_evals/v48_8m_best_metrics/streaming_identity_bsz8388608_lr0p02_s42/result.json",
    ),
    RunSpec(
        8_388_608,
        "8M",
        "Top-Aware c=0.5",
        0.02,
        ROOT / "search_evals/v48_8m_best_metrics/top_aware_k1_a0p5_bsz8388608_lr0p02_s42/result.json",
    ),
]


COLORS = {
    "identity c=1": "#1f4e79",
    "Top-Aware c=0.5": "#c75a21",
}
METHOD_ORDER = ["identity c=1", "Top-Aware c=0.5"]
BATCH_ORDER = ["262K", "1M", "2M", "4M", "8M"]


def finite(value: object) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def finite_list(values: object) -> list[float]:
    if isinstance(values, list):
        return [out for value in values if (out := finite(value)) is not None]
    out = finite(values)
    return [] if out is None else [out]


def mean(values: Iterable[float]) -> float | None:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else None


def stdev(values: Iterable[float]) -> float | None:
    vals = [v for v in values if math.isfinite(v)]
    if len(vals) < 2:
        return 0.0 if vals else None
    return statistics.stdev(vals)


def fmt(value: object, digits: int = 6) -> str:
    out = finite(value)
    if out is None:
        return ""
    if abs(out) < 1e-3 and out != 0.0:
        return f"{out:.2e}"
    return f"{out:.{digits}f}"


def scalar_by_prefix(scalars: dict[str, object], prefix: str) -> dict[str, float]:
    out = {}
    for key, value in scalars.items():
        if not key.startswith(prefix):
            continue
        val = finite(value)
        if val is not None:
            out[key[len(prefix) :]] = val
    return out


def mean_prefix(scalars: dict[str, object], prefix: str) -> float | None:
    return mean(scalar_by_prefix(scalars, prefix).values())


def mean_ratio(scalars: dict[str, object], numerator: str, denominator: str) -> float | None:
    nums = scalar_by_prefix(scalars, numerator)
    dens = scalar_by_prefix(scalars, denominator)
    ratios = []
    for module, num in nums.items():
        den = dens.get(module)
        if den is not None and den > 0:
            ratios.append(num / den)
    return mean(ratios)


def first(values: object) -> float | None:
    vals = finite_list(values)
    return vals[0] if vals else None


def l2(values: object) -> float | None:
    vals = finite_list(values)
    if not vals:
        return None
    return math.sqrt(sum(v * v for v in vals))


def hessian_muon_alignment_stats(vectors: dict[str, object]) -> tuple[float | None, float | None, float | None]:
    top1_abs = []
    all_abs = []
    for key, values in vectors.items():
        if not key.startswith("alignment_between_covariance_hessian_at_k_th_component/"):
            continue
        vals = finite_list(values)
        if not vals:
            continue
        top1_abs.append(abs(vals[0]))
        all_abs.extend(abs(v) for v in vals)
    return mean(top1_abs), max(top1_abs) if top1_abs else None, mean(all_abs)


def load_run(spec: RunSpec) -> tuple[list[dict[str, object]], dict[str, object]]:
    data = json.loads(spec.path.read_text())
    logs = data.get("metric_logs") or []
    max_step = int(data.get("steps_completed") or max((int(log["step"]) for log in logs), default=0) or 1)
    rows: list[dict[str, object]] = []

    for log in logs:
        step = int(log["step"])
        scalars = log.get("scalars") or {}
        row: dict[str, object] = {
            "batch": spec.batch,
            "batch_label": spec.batch_label,
            "method": spec.method,
            "lr": spec.lr,
            "path": str(spec.path.relative_to(ROOT)),
            "score": finite(data.get("score")),
            "step": step,
            "progress": step / max(max_step, 1),
            "train_loss_ema": finite(scalars.get("train/loss_ema")),
            "train_lr_multiplier": finite(scalars.get("train/lr_multiplier")),
            "projection_corr": finite(
                scalars.get("gradient_projection_on_last_hessian_space_consecutive_correlation/selected_subspace")
            ),
            "projection_norm": finite(
                scalars.get("gradient_projection_on_last_hessian_space_norm/selected_subspace")
            ),
            "weight_norm_mean": mean_prefix(scalars, "weight_norm/"),
            "grad_norm_mean": mean_prefix(scalars, "grad_norm/"),
            "momentum_nesterov_norm_mean": mean_prefix(scalars, "momentum_after_nesterov_norm/"),
            "momentum_nesterov_spectral_mean": mean_prefix(
                scalars, "momentum_after_nesterov_spectral_norm/"
            ),
            "momentum_spectral_to_rms_mean": mean_ratio(
                scalars, "momentum_after_nesterov_spectral_norm/", "momentum_after_nesterov_norm/"
            ),
        }

        hessian = log.get("hessian") or {}
        h_scalars = hessian.get("scalars") or {}
        h_vectors = hessian.get("vectors") or {}
        component_top1_mean, component_top1_max, component_all_mean = hessian_muon_alignment_stats(h_vectors)
        row.update(
            {
                "sharpness": finite(h_scalars.get("sharpness/selected_subspace")),
                "grad_hessian_top1_alignment": first(
                    h_vectors.get("gradient_hessian_alignment/selected_subspace")
                ),
                "grad_hessian_alignment_l2": l2(
                    h_vectors.get("gradient_hessian_alignment/selected_subspace")
                ),
                "momentum_hessian_top1_alignment": first(
                    h_vectors.get("momentum_after_nesterov_hessian_alignment/selected_subspace")
                ),
                "momentum_hessian_alignment_l2": l2(
                    h_vectors.get("momentum_after_nesterov_hessian_alignment/selected_subspace")
                ),
                "hessian_muon_component_top1_abs_mean": component_top1_mean,
                "hessian_muon_component_top1_abs_max": component_top1_max,
                "hessian_muon_component_all_abs_mean": component_all_mean,
            }
        )
        rows.append(row)

    summary = summarize_run(spec, data, rows)
    return rows, summary


def late_values(rows: list[dict[str, object]], key: str, min_progress: float = 0.7) -> list[float]:
    vals = []
    for row in rows:
        if float(row["progress"]) < min_progress:
            continue
        value = finite(row.get(key))
        if value is not None:
            vals.append(value)
    return vals


def all_values(rows: list[dict[str, object]], key: str) -> list[float]:
    return [value for row in rows if (value := finite(row.get(key))) is not None]


def last_value(rows: list[dict[str, object]], key: str) -> float | None:
    for row in reversed(rows):
        value = finite(row.get(key))
        if value is not None:
            return value
    return None


def summarize_run(spec: RunSpec, data: dict[str, object], rows: list[dict[str, object]]) -> dict[str, object]:
    corr = all_values(rows, "projection_corr")
    corr_late = late_values(rows, "projection_corr")
    sharp = all_values(rows, "sharpness")
    return {
        "batch": spec.batch,
        "batch_label": spec.batch_label,
        "method": spec.method,
        "lr": spec.lr,
        "score": finite(data.get("score")),
        "steps_completed": int(data.get("steps_completed") or 0),
        "metric_logs": len(rows),
        "hessian_probes": len(sharp),
        "train_loss_ema_last": last_value(rows, "train_loss_ema"),
        "sharpness_last": sharp[-1] if sharp else None,
        "sharpness_max": max(sharp) if sharp else None,
        "projection_corr_mean": mean(corr),
        "projection_corr_late_mean": mean(corr_late),
        "projection_corr_late_std": stdev(corr_late),
        "projection_corr_negative_fraction": mean([1.0 if v < 0 else 0.0 for v in corr]),
        "projection_norm_late_mean": mean(late_values(rows, "projection_norm")),
        "grad_hessian_top1_last": last_value(rows, "grad_hessian_top1_alignment"),
        "grad_hessian_l2_last": last_value(rows, "grad_hessian_alignment_l2"),
        "momentum_hessian_top1_last": last_value(rows, "momentum_hessian_top1_alignment"),
        "momentum_hessian_l2_last": last_value(rows, "momentum_hessian_alignment_l2"),
        "hessian_muon_component_top1_abs_mean_last": last_value(
            rows, "hessian_muon_component_top1_abs_mean"
        ),
        "hessian_muon_component_top1_abs_max_last": last_value(
            rows, "hessian_muon_component_top1_abs_max"
        ),
        "momentum_spectral_to_rms_last": last_value(rows, "momentum_spectral_to_rms_mean"),
        "grad_norm_late_mean": mean(late_values(rows, "grad_norm_mean")),
        "momentum_norm_late_mean": mean(late_values(rows, "momentum_nesterov_norm_mean")),
        "path": str(spec.path.relative_to(ROOT)),
    }


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


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
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def xy(rows: list[dict[str, object]], key: str) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for row in rows:
        val = finite(row.get(key))
        if val is None:
            continue
        xs.append(int(row["step"]))
        ys.append(val)
    return xs, ys


def rolling_mean(xs: list[int], ys: list[float], window: int) -> tuple[list[int], list[float]]:
    if window <= 1 or len(ys) <= window:
        return xs, ys
    out_x, out_y = [], []
    running = 0.0
    for idx, y in enumerate(ys):
        running += y
        if idx >= window:
            running -= ys[idx - window]
        if idx + 1 >= window:
            out_x.append(xs[idx])
            out_y.append(running / window)
    return out_x, out_y


def set_dynamic_ylim(ax: plt.Axes, series: list[list[float]], zero_line: bool = False) -> None:
    values = [v for ys in series for v in ys if math.isfinite(v)]
    if not values:
        return
    lo, hi = min(values), max(values)
    if zero_line:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    if lo == hi:
        pad = max(abs(lo) * 0.08, 1e-6)
    else:
        pad = max((hi - lo) * 0.08, 1e-6)
    ax.set_ylim(lo - pad, hi + pad)


def plot_grid(
    rows: list[dict[str, object]],
    metrics: list[tuple[str, str, str]],
    output_stem: str,
    title: str,
) -> None:
    fig, axes = plt.subplots(
        len(metrics),
        len(BATCH_ORDER),
        figsize=(3.65 * len(BATCH_ORDER), 2.05 * len(metrics)),
        sharex=True,
        constrained_layout=True,
    )
    if len(metrics) == 1:
        axes = [axes]  # type: ignore[assignment]

    by_run: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        by_run.setdefault((str(row["batch_label"]), str(row["method"])), []).append(row)

    for col, batch in enumerate(BATCH_ORDER):
        for row_idx, (key, ylabel, transform) in enumerate(metrics):
            ax = axes[row_idx][col]
            plotted_ys = []
            for method in METHOD_ORDER:
                run_rows = by_run.get((batch, method), [])
                xs, ys = xy(run_rows, key)
                if not xs:
                    continue
                plotted_ys.append(ys)
                if key == "projection_corr":
                    ax.plot(xs, ys, color=COLORS[method], linewidth=0.55, alpha=0.18)
                    sx, sy = rolling_mean(xs, ys, max(3, min(35, len(ys) // 18)))
                    ax.plot(sx, sy, color=COLORS[method], linewidth=1.8, label=method)
                else:
                    ax.plot(xs, ys, color=COLORS[method], linewidth=1.8, marker="o", markersize=2.7, label=method)
            if transform == "log":
                ax.set_yscale("log")
            if transform == "signed":
                ax.axhline(0.0, color="#666", linewidth=0.75, linestyle="--", alpha=0.55)
                ax.set_ylim(-1.05, 1.05)
            elif transform == "dynamic_signed":
                ax.axhline(0.0, color="#666", linewidth=0.75, linestyle="--", alpha=0.55)
                set_dynamic_ylim(ax, plotted_ys, zero_line=True)
            elif transform != "log":
                set_dynamic_ylim(ax, plotted_ys)
            if row_idx == 0:
                ax.set_title(batch)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row_idx == len(metrics) - 1:
                ax.set_xlabel("optimizer step")
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{int(x)}"))

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.015), ncol=2, frameon=False)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.055)
    fig.savefig(OUT / f"{output_stem}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{output_stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_summary(summary: list[dict[str, object]]) -> None:
    by_run = {(row["batch_label"], row["method"]): row for row in summary}
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.2), constrained_layout=True)
    axes_flat = list(axes.ravel())

    def method_series(key: str) -> tuple[list[float], list[float]]:
        identity = [finite(by_run.get((batch, "identity c=1"), {}).get(key)) for batch in BATCH_ORDER]
        topaware = [finite(by_run.get((batch, "Top-Aware c=0.5"), {}).get(key)) for batch in BATCH_ORDER]
        return (
            [float("nan") if v is None else v for v in identity],
            [float("nan") if v is None else v for v in topaware],
        )

    xs = list(range(len(BATCH_ORDER)))

    ax = axes_flat[0]
    for method in METHOD_ORDER:
        vals = [finite(by_run.get((batch, method), {}).get("score")) for batch in BATCH_ORDER]
        ax.plot(xs, vals, marker="o", linewidth=2.0, color=COLORS[method], label=method)
    ax.set_title("final validation BPB")
    ax.set_ylabel("BPB")
    ax.set_xticks(xs, BATCH_ORDER)
    ax.legend(frameon=False)

    ax = axes_flat[1]
    deltas = []
    for batch in BATCH_ORDER:
        ident = finite(by_run.get((batch, "identity c=1"), {}).get("score"))
        top = finite(by_run.get((batch, "Top-Aware c=0.5"), {}).get("score"))
        deltas.append(float("nan") if ident is None or top is None else top - ident)
    ax.bar(xs, deltas, color=["#c75a21" if d < 0 else "#1f4e79" for d in deltas])
    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.set_title("Top-Aware minus identity")
    ax.set_ylabel("BPB delta")
    ax.set_xticks(xs, BATCH_ORDER)

    for ax, key, title, ylabel, use_log in [
        (axes_flat[2], "sharpness_last", "late sharpness", "top Hessian eigenvalue", True),
        (axes_flat[3], "projection_corr_late_mean", "late projection correlation", "cosine", False),
        (
            axes_flat[4],
            "hessian_muon_component_top1_abs_mean_last",
            "Hessian vs cached Muon component",
            "mean |alignment|",
            False,
        ),
        (
            axes_flat[5],
            "momentum_spectral_to_rms_last",
            "momentum spectral/RMS ratio",
            "mean ratio",
            False,
        ),
    ]:
        for method in METHOD_ORDER:
            vals = [finite(by_run.get((batch, method), {}).get(key)) for batch in BATCH_ORDER]
            ax.plot(xs, vals, marker="o", linewidth=2.0, color=COLORS[method], label=method)
        if use_log:
            ax.set_yscale("log")
        if "correlation" in title:
            ax.axhline(0.0, color="#666", linewidth=0.8, linestyle="--", alpha=0.55)
            ax.set_ylim(-1.05, 1.05)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(xs, BATCH_ORDER)

    fig.suptitle("Metrics summary across batch size", fontsize=14, fontweight="bold")
    fig.savefig(OUT / "metrics_summary_by_batch.png", bbox_inches="tight")
    fig.savefig(OUT / "metrics_summary_by_batch.pdf", bbox_inches="tight")
    plt.close(fig)


def write_markdown(summary: list[dict[str, object]]) -> None:
    by_run = {(row["batch_label"], row["method"]): row for row in summary}
    lines = [
        "# Metrics Dynamics Analysis",
        "",
        "Scope: d8 clean metrics runs comparing StreamingMuon identity `c=1` against Top-Aware Muon `top_k=1, alpha=0.5`. The runs use dense per-step metrics and periodic global Hessian probes. Lower BPB is better.",
        "",
        "Generated files:",
        "- `timeseries.csv`: per-step metrics used for plotting.",
        "- `summary.csv`: per-run final and late-window aggregates.",
        "- `metrics_summary_by_batch.png`: cross-batch metric summary.",
        "- `hessian_timeseries_grid.png`: sharpness, Hessian alignment, and projection-correlation dynamics.",
        "- `optimizer_state_timeseries_grid.png`: loss and optimizer-state norm dynamics.",
        "",
        "## Main Table",
        "",
        "| batch | method | LR | BPB | probes | sharp last | corr late | corr neg frac | grad-H top1 last | mom-H top1 last | Hessian-Muon abs last | spec/RMS last |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for batch in BATCH_ORDER:
        for method in METHOD_ORDER:
            row = by_run.get((batch, method))
            if row is None:
                continue
            lines.append(
                f"| {batch} | {method} | {fmt(row['lr'], 4)} | {fmt(row['score'])} | "
                f"{row['hessian_probes']} | {fmt(row['sharpness_last'])} | "
                f"{fmt(row['projection_corr_late_mean'])} | {fmt(row['projection_corr_negative_fraction'])} | "
                f"{fmt(row['grad_hessian_top1_last'])} | {fmt(row['momentum_hessian_top1_last'])} | "
                f"{fmt(row['hessian_muon_component_top1_abs_mean_last'])} | "
                f"{fmt(row['momentum_spectral_to_rms_last'])} |"
            )

    lines += [
        "",
        "## Paired Deltas",
        "",
        "| batch | Top-Aware BPB - identity BPB | sharpness ratio Top-Aware / identity | late corr delta |",
        "|---:|---:|---:|---:|",
    ]
    for batch in BATCH_ORDER:
        identity = by_run.get((batch, "identity c=1"))
        topaware = by_run.get((batch, "Top-Aware c=0.5"))
        if identity is None or topaware is None:
            continue
        ident_score = finite(identity.get("score"))
        top_score = finite(topaware.get("score"))
        ident_sharp = finite(identity.get("sharpness_last"))
        top_sharp = finite(topaware.get("sharpness_last"))
        ident_corr = finite(identity.get("projection_corr_late_mean"))
        top_corr = finite(topaware.get("projection_corr_late_mean"))
        lines.append(
            f"| {batch} | {fmt(None if ident_score is None or top_score is None else top_score - ident_score)} | "
            f"{fmt(None if not ident_sharp or top_sharp is None else top_sharp / ident_sharp)} | "
            f"{fmt(None if ident_corr is None or top_corr is None else top_corr - ident_corr)} |"
        )

    lines += [
        "",
        "## Figures",
        "",
        "![Metrics summary by batch](metrics_summary_by_batch.png)",
        "",
        "![Hessian timeseries grid](hessian_timeseries_grid.png)",
        "",
        "![Optimizer state timeseries grid](optimizer_state_timeseries_grid.png)",
        "",
        "## Current Readout",
        "",
        "- The metrics runs reproduce the main qualitative batch trend: identity is better at `262K`, `1M`, and `2M`, while Top-Aware is better at `4M` and `8M` in these selected settings.",
        "- Sharpness alone is not a winner predictor. Top-Aware can have larger measured sharpness both where it loses (`262K`, `1M`) and where it wins (`4M`, `8M`).",
        "- The consecutive projection correlation is usually negative, meaning projected gradients often flip direction inside the latest Hessian eigenspace. The least-negative late value is at `8M` Top-Aware, but this does not by itself explain the `4M` improvement.",
        "- Hessian-vs-cached-Muon-component alignment is moderate and batch-dependent. Treat it as a diagnostic for dynamics, not yet as a selection rule.",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    setup_plot_style()

    all_rows: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    for spec in RUNS:
        if not spec.path.exists():
            raise FileNotFoundError(spec.path)
        rows, run_summary = load_run(spec)
        all_rows.extend(rows)
        summary.append(run_summary)

    timeseries_fields = [
        "batch",
        "batch_label",
        "method",
        "lr",
        "score",
        "step",
        "progress",
        "train_loss_ema",
        "train_lr_multiplier",
        "projection_corr",
        "projection_norm",
        "weight_norm_mean",
        "grad_norm_mean",
        "momentum_nesterov_norm_mean",
        "momentum_nesterov_spectral_mean",
        "momentum_spectral_to_rms_mean",
        "sharpness",
        "grad_hessian_top1_alignment",
        "grad_hessian_alignment_l2",
        "momentum_hessian_top1_alignment",
        "momentum_hessian_alignment_l2",
        "hessian_muon_component_top1_abs_mean",
        "hessian_muon_component_top1_abs_max",
        "hessian_muon_component_all_abs_mean",
        "path",
    ]
    summary_fields = [
        "batch",
        "batch_label",
        "method",
        "lr",
        "score",
        "steps_completed",
        "metric_logs",
        "hessian_probes",
        "train_loss_ema_last",
        "sharpness_last",
        "sharpness_max",
        "projection_corr_mean",
        "projection_corr_late_mean",
        "projection_corr_late_std",
        "projection_corr_negative_fraction",
        "projection_norm_late_mean",
        "grad_hessian_top1_last",
        "grad_hessian_l2_last",
        "momentum_hessian_top1_last",
        "momentum_hessian_l2_last",
        "hessian_muon_component_top1_abs_mean_last",
        "hessian_muon_component_top1_abs_max_last",
        "momentum_spectral_to_rms_last",
        "grad_norm_late_mean",
        "momentum_norm_late_mean",
        "path",
    ]
    write_csv(OUT / "timeseries.csv", all_rows, timeseries_fields)
    write_csv(OUT / "summary.csv", summary, summary_fields)

    plot_summary(summary)
    plot_grid(
        all_rows,
        [
            ("sharpness", "sharpness", "log"),
            ("grad_hessian_top1_alignment", "grad-H top1 cosine", "signed"),
            ("momentum_hessian_top1_alignment", "momentum-H top1 cosine", "signed"),
            ("hessian_muon_component_top1_abs_mean", "mean |H vs Muon comp|", "dynamic"),
            ("projection_corr", "projection corr", "signed"),
        ],
        "hessian_timeseries_grid",
        "Hessian and projection dynamics",
    )
    plot_grid(
        all_rows,
        [
            ("train_loss_ema", "train loss EMA", "dynamic"),
            ("weight_norm_mean", "mean weight RMS", "dynamic"),
            ("grad_norm_mean", "mean grad RMS", "log"),
            ("momentum_nesterov_norm_mean", "mean M' RMS", "log"),
            ("momentum_spectral_to_rms_mean", "mean spectral/RMS", "dynamic"),
        ],
        "optimizer_state_timeseries_grid",
        "Optimizer-state dynamics",
    )
    write_markdown(summary)

    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
