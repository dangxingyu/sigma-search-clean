#!/usr/bin/env python3
"""Plot Hessian dynamics from clean metrics result JSONs.

This script intentionally uses only the Python standard library so it can run
in a minimal cluster environment without matplotlib.
"""

from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "metrics_cross_batch"


@dataclass(frozen=True)
class RunSpec:
    batch_label: str
    method_label: str
    color: str
    path: Path


RUNS = [
    RunSpec(
        "2M",
        "identity c=1",
        "#174a7c",
        ROOT / "search_evals/v46_2m_best_metrics/streaming_identity_bsz2097152_lr0p08_s42/result.json",
    ),
    RunSpec(
        "2M",
        "Top-Aware c=0.5",
        "#c94f1a",
        ROOT / "search_evals/v46_2m_best_metrics/top_aware_k1_a0p5_bsz2097152_lr0p08_s42/result.json",
    ),
    RunSpec(
        "4M",
        "identity c=1",
        "#174a7c",
        ROOT / "search_evals/v43_4m_best_metrics/streaming_identity_bsz4194304_lr0p02_s42/result.json",
    ),
    RunSpec(
        "4M",
        "Top-Aware c=0.5",
        "#c94f1a",
        ROOT / "search_evals/v43_4m_best_metrics/top_aware_k1_a0p5_bsz4194304_lr0p02_s42/result.json",
    ),
    RunSpec(
        "8M",
        "identity c=1",
        "#174a7c",
        ROOT / "search_evals/v48_8m_best_metrics/streaming_identity_bsz8388608_lr0p02_s42/result.json",
    ),
    RunSpec(
        "8M",
        "Top-Aware c=0.5",
        "#c94f1a",
        ROOT / "search_evals/v48_8m_best_metrics/top_aware_k1_a0p5_bsz8388608_lr0p02_s42/result.json",
    ),
]


def finite_float(value: object) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def first_finite(values: object) -> float | None:
    if not isinstance(values, list):
        return finite_float(values)
    for value in values:
        out = finite_float(value)
        if out is not None:
            return out
    return None


def load_series(spec: RunSpec) -> dict[str, object]:
    data = json.loads(spec.path.read_text())
    logs = data.get("metric_logs", [])
    sharpness: list[tuple[int, float]] = []
    grad_align: list[tuple[int, float]] = []
    proj_corr: list[tuple[int, float]] = []

    for log in logs:
        step = int(log["step"])
        scalars = log.get("scalars") or {}
        corr = finite_float(
            scalars.get("gradient_projection_on_last_hessian_space_consecutive_correlation/selected_subspace")
        )
        if corr is not None:
            proj_corr.append((step, corr))

        hessian = log.get("hessian") or {}
        h_scalars = hessian.get("scalars") or {}
        sharp = finite_float(h_scalars.get("sharpness/selected_subspace"))
        if sharp is not None:
            sharpness.append((step, sharp))

        h_vectors = hessian.get("vectors") or {}
        align = first_finite(h_vectors.get("gradient_hessian_alignment/selected_subspace"))
        if align is not None:
            grad_align.append((step, align))

    return {
        "batch": spec.batch_label,
        "method": spec.method_label,
        "color": spec.color,
        "path": str(spec.path.relative_to(ROOT)),
        "score": finite_float(data.get("score")),
        "steps_completed": data.get("steps_completed"),
        "sharpness": sharpness,
        "grad_align": grad_align,
        "proj_corr": proj_corr,
    }


def extent(series: Iterable[list[tuple[int, float]]], pad: float = 0.08) -> tuple[float, float]:
    values = [v for xs in series for _, v in xs if math.isfinite(v)]
    if not values:
        return 0.0, 1.0
    lo, hi = min(values), max(values)
    if lo == hi:
        delta = max(abs(lo) * 0.1, 1e-6)
        return lo - delta, hi + delta
    delta = (hi - lo) * pad
    return lo - delta, hi + delta


def map_points(
    points: list[tuple[int, float]],
    x0: float,
    y0: float,
    width: float,
    height: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> str:
    if not points:
        return ""
    denom_x = max(xmax, 1.0)
    denom_y = max(ymax - ymin, 1e-12)
    mapped = []
    for step, value in points:
        x = x0 + width * (step / denom_x)
        y = y0 + height * (1.0 - (value - ymin) / denom_y)
        mapped.append(f"{x:.1f},{y:.1f}")
    return " ".join(mapped)


def fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    if abs(value) < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def make_svg(rows: list[dict[str, object]]) -> str:
    batches = ["2M", "4M", "8M"]
    metrics = [
        ("sharpness", "Hessian max eigenvalue", None),
        ("grad_align", "gradient-Hessian top-1 cosine", (-0.08, 0.08)),
        ("proj_corr", "consecutive projection cosine", (-1.05, 1.05)),
    ]
    width, height = 1260, 930
    panel_w, panel_h = 335, 205
    left0, top0 = 72, 92
    col_gap, row_gap = 60, 68
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        '<text x="34" y="38" font-family="serif" font-size="25" font-weight="700" fill="#222">Hessian dynamics overview</text>',
        '<text x="34" y="62" font-family="sans-serif" font-size="13" fill="#555">Identity vs Top-Aware c=0.5, d8 0.4B-token metrics runs, seed 42</text>',
        '<line x1="760" y1="36" x2="792" y2="36" stroke="#174a7c" stroke-width="3"/>',
        '<text x="800" y="41" font-family="sans-serif" font-size="13" fill="#222">identity c=1</text>',
        '<line x1="940" y1="36" x2="972" y2="36" stroke="#c94f1a" stroke-width="3"/>',
        '<text x="980" y="41" font-family="sans-serif" font-size="13" fill="#222">Top-Aware c=0.5</text>',
    ]

    for col, batch in enumerate(batches):
        x = left0 + col * (panel_w + col_gap)
        parts.append(
            f'<text x="{x + panel_w / 2:.1f}" y="82" text-anchor="middle" '
            'font-family="sans-serif" font-size="15" font-weight="700" fill="#222">'
            f"batch {batch}</text>"
        )

    for row_idx, (metric_key, metric_title, fixed_ylim) in enumerate(metrics):
        y = top0 + row_idx * (panel_h + row_gap)
        parts.append(
            f'<text x="28" y="{y + panel_h / 2:.1f}" text-anchor="middle" '
            f'transform="rotate(-90 28 {y + panel_h / 2:.1f})" '
            'font-family="sans-serif" font-size="13" font-weight="700" fill="#333">'
            f"{html.escape(metric_title)}</text>"
        )
        for col, batch in enumerate(batches):
            x = left0 + col * (panel_w + col_gap)
            batch_rows = [r for r in rows if r["batch"] == batch]
            xmax = max((pts[-1][0] for r in batch_rows for pts in [r[metric_key]] if pts), default=1)
            if fixed_ylim is None:
                ymin, ymax = extent([r[metric_key] for r in batch_rows])  # type: ignore[list-item]
                if metric_key == "sharpness":
                    ymin = min(0.0, ymin)
            else:
                ymin, ymax = fixed_ylim

            parts.extend(
                [
                    f'<rect x="{x}" y="{y}" width="{panel_w}" height="{panel_h}" fill="#fff" stroke="#d8d0c2"/>',
                    f'<line x1="{x}" y1="{y + panel_h}" x2="{x + panel_w}" y2="{y + panel_h}" stroke="#777"/>',
                    f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y + panel_h}" stroke="#777"/>',
                    f'<text x="{x - 8}" y="{y + 5}" text-anchor="end" font-family="monospace" font-size="10" fill="#555">{fmt(ymax)}</text>',
                    f'<text x="{x - 8}" y="{y + panel_h}" text-anchor="end" font-family="monospace" font-size="10" fill="#555">{fmt(ymin)}</text>',
                    f'<text x="{x + panel_w}" y="{y + panel_h + 18}" text-anchor="end" font-family="monospace" font-size="10" fill="#555">step {xmax}</text>',
                ]
            )

            if ymin < 0.0 < ymax:
                y_zero = y + panel_h * (1.0 - (0.0 - ymin) / (ymax - ymin))
                parts.append(
                    f'<line x1="{x}" y1="{y_zero:.1f}" x2="{x + panel_w}" y2="{y_zero:.1f}" '
                    'stroke="#bbb" stroke-dasharray="4 4"/>'
                )

            for run in batch_rows:
                pts = run[metric_key]  # type: ignore[assignment]
                coords = map_points(pts, x, y, panel_w, panel_h, float(xmax), ymin, ymax)
                if not coords:
                    continue
                stroke = run["color"]
                parts.append(
                    f'<polyline points="{coords}" fill="none" stroke="{stroke}" stroke-width="2.2"/>'
                )
                if metric_key != "proj_corr":
                    for step, value in pts:  # type: ignore[union-attr]
                        dot = map_points([(step, value)], x, y, panel_w, panel_h, float(xmax), ymin, ymax)
                        px, py = dot.split(",")
                        parts.append(f'<circle cx="{px}" cy="{py}" r="3" fill="{stroke}"/>')

    parts.append("</svg>")
    return "\n".join(parts)


def make_markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "# Hessian Dynamics Overview",
        "",
        "![Hessian dynamics overview](hessian_dynamics_overview.svg)",
        "",
        "| batch | method | val BPB | sharpness probes | grad-Hessian top-1 alignment | projection-corr mean | projection-corr last10 |",
        "|---:|---|---:|---|---|---:|---:|",
    ]
    for row in rows:
        sharp = [v for _, v in row["sharpness"]]  # type: ignore[index]
        align = [v for _, v in row["grad_align"]]  # type: ignore[index]
        corr = [v for _, v in row["proj_corr"]]  # type: ignore[index]
        corr_mean = sum(corr) / len(corr) if corr else None
        corr_last = sum(corr[-10:]) / min(10, len(corr)) if corr else None
        lines.append(
            "| {batch} | {method} | {score} | {sharp} | {align} | {corr_mean} | {corr_last} |".format(
                batch=row["batch"],
                method=row["method"],
                score=fmt(row["score"]),  # type: ignore[arg-type]
                sharp=", ".join(fmt(v) for v in sharp),
                align=", ".join(fmt(v) for v in align),
                corr_mean=fmt(corr_mean),
                corr_last=fmt(corr_last),
            )
        )
    lines.extend(
        [
            "",
            "Read: sharpness is the top selected-subspace Hessian eigenvalue. Gradient-Hessian alignment is the signed cosine with the top Hessian direction. Projection correlation is the cosine between consecutive gradients after projecting both into the most recent Hessian top-k subspace.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    rows = [load_series(spec) for spec in RUNS]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "hessian_dynamics_overview.svg").write_text(make_svg(rows))
    (OUT_DIR / "hessian_dynamics_overview.md").write_text(make_markdown(rows))
    (OUT_DIR / "hessian_dynamics_overview.json").write_text(json.dumps(rows, indent=2))
    print(OUT_DIR / "hessian_dynamics_overview.svg")
    print(OUT_DIR / "hessian_dynamics_overview.md")


if __name__ == "__main__":
    main()
