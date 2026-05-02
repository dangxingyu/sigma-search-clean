#!/usr/bin/env python3
"""Dashboard for Top-Aware alpha sweeps and 256K partial v29 results."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if (SCRIPT_DIR / "refactored-repo").exists():
    OUT_DIR = SCRIPT_DIR / "new-figures/topaware_alpha_dashboard"
    REFIG_DIR = SCRIPT_DIR / "refactored-repo/figures/topaware_alpha_dashboard"
else:
    OUT_DIR = SCRIPT_DIR / "figures/topaware_alpha_dashboard"
    REFIG_DIR = None

OLD_RAW_CANDIDATES = [
    SCRIPT_DIR / "figures/native_streaming_topk/native_streaming_topk_raw.csv",
    SCRIPT_DIR / "new-figures/native_streaming_topk/native_streaming_topk_raw.csv",
    SCRIPT_DIR.parent / "new-figures/native_streaming_topk/native_streaming_topk_raw.csv",
]
SEARCH_EVAL_ROOTS = [
    SCRIPT_DIR / "search_evals",
    SCRIPT_DIR.parent / "search_evals",
]


def bsz_label(batch: int) -> str:
    if batch >= 1_048_576:
        return f"{batch // 1_048_576}M"
    return f"{batch // 1024}K"


def slug_float(x: str) -> float:
    return float(x.replace("p", "."))


def score(path: Path) -> float | None:
    data = json.loads(path.read_text())
    if data.get("error") is not None:
        return None
    value = data.get("score", data.get("val_bpb_best"))
    return None if value is None else float(value)


def load_rows() -> list[dict]:
    rows: list[dict] = []

    for old_raw in OLD_RAW_CANDIDATES:
        if not old_raw.exists():
            continue
        with old_raw.open() as f:
            for row in csv.DictReader(f):
                batch = int(row["batch"])
                method = row["method"]
                if batch == 131072 and method in {"streaming_identity", "top_aware_k1_a0.5", "native_muon"}:
                    rows.append({
                        "source": row["source"],
                        "batch": batch,
                        "batch_label": bsz_label(batch),
                        "method": method,
                        "lr": float(row["lr"]),
                        "seed": int(row["seed"]),
                        "score": float(row["score"]),
                        "status": "complete",
                        "path": row["path"],
                    })
        break

    pat = re.compile(r"(streaming_identity|streaming_lite|native_muon|native_lite|top_aware_k\d+_a[0-9p]+)_bsz(\d+)_lr([0-9p]+)_s(\d+)$")
    sweep_dirs: list[Path] = []
    seen_dirs: set[Path] = set()
    for root in SEARCH_EVAL_ROOTS:
        if not root.exists():
            continue
        for d in root.glob("v*_topaware_k1_*"):
            if d not in seen_dirs:
                seen_dirs.add(d)
                sweep_dirs.append(d)

    for sweep_dir in sorted(sweep_dirs):
        for p in sorted(sweep_dir.glob("*/result.json")):
            m = pat.match(p.parent.name)
            if not m:
                continue
            method, batch_s, lr_s, seed_s = m.groups()
            method = re.sub(r"a(\d+)p(\d+)", r"a\1.\2", method)
            # The v29/v31 DDP native_lite rows are not valid exact-LITE
            # controls: the local LITE runner initializes distributed state but
            # does not synchronize matrix gradients. Do not let those rows enter
            # the summary dashboard.
            if method == "native_lite":
                continue
            sc = score(p)
            if sc is None:
                continue
            rows.append({
                "source": sweep_dir.name,
                "batch": int(batch_s),
                "batch_label": bsz_label(int(batch_s)),
                "method": method,
                "lr": slug_float(lr_s),
                "seed": int(seed_s),
                "score": sc,
                "status": "complete",
                "path": str(p),
            })

    return rows


def best_rows(rows: list[dict]) -> list[dict]:
    best: dict[tuple[int, str], dict] = {}
    for row in rows:
        key = (row["batch"], row["method"])
        if key not in best or row["score"] < best[key]["score"]:
            best[key] = row
    return sorted(best.values(), key=lambda r: (r["batch"], r["method"]))


def write_csvs(rows: list[dict], best: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = ["source", "batch_label", "batch", "method", "lr", "seed", "score", "status", "path"]
    for name, data in [("raw_rows.csv", rows), ("best_rows.csv", best)]:
        with (OUT_DIR / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in data:
                w.writerow({k: row.get(k, "") for k in fields})


def write_md(best: list[dict]) -> None:
    by = {(r["batch"], r["method"]): r for r in best}
    lines = [
        "# Top-Aware Alpha Dashboard",
        "",
        "Lower BPB is better. This table auto-scans the completed v28/v29/v31 Top-Aware sweep result JSONs plus the historical 128K alpha=.5 rows.",
        "",
        "| batch | method | best BPB | best LR | delta vs identity | status |",
        "|---:|---|---:|---:|---:|---|",
    ]
    methods = [
        "streaming_identity",
        "streaming_lite",
        "native_muon",
        "top_aware_k1_a0.25",
        "top_aware_k1_a0.5",
        "top_aware_k1_a0.75",
        "top_aware_k1_a0.875",
    ]
    for batch in sorted({r["batch"] for r in best}):
        ident = by.get((batch, "streaming_identity"))
        for method in methods:
            row = by.get((batch, method))
            if row is None:
                continue
            delta = ""
            if ident is not None and method != "streaming_identity":
                delta = f"{row['score'] - ident['score']:+.6f}"
            lines.append(
                f"| {bsz_label(batch)} | `{method}` | `{row['score']:.6f}` | `{row['lr']:g}` | `{delta}` | {row['status']} |"
            )

    lines += [
        "",
        "Current reading should be regenerated after each overnight sweep; use the raw CSV when a batch is still partially complete.",
        "DDP `native_lite` rows from v29/v31 are excluded because that runner is not a valid distributed exact-LITE implementation.",
        "",
    ]
    (OUT_DIR / "best_table.md").write_text("\n".join(lines))


def plot(rows: list[dict], best: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    colors = {
        "streaming_identity": "#2563eb",
        "streaming_lite": "#059669",
        "native_muon": "#111827",
        "native_lite": "#7c2d12",
        "top_aware_k1_a0.5": "#dc2626",
        "top_aware_k1_a0.75": "#ea580c",
        "top_aware_k1_a0.875": "#f59e0b",
    }
    markers = {
        "streaming_identity": "o",
        "streaming_lite": "s",
        "native_muon": "D",
        "native_lite": "P",
        "top_aware_k1_a0.5": "^",
        "top_aware_k1_a0.75": "v",
        "top_aware_k1_a0.875": "X",
    }
    batches = sorted({r["batch"] for r in rows})
    if not batches:
        return
    ncols = min(3, len(batches))
    nrows = math.ceil(len(batches) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.8 * nrows), sharey=False, squeeze=False)
    for ax, batch in zip(axes.flat, batches):
        batch_rows = [r for r in rows if r["batch"] == batch]
        for method in sorted({r["method"] for r in batch_rows}):
            mr = sorted([r for r in batch_rows if r["method"] == method], key=lambda r: r["lr"])
            xs = [r["lr"] for r in mr]
            ys = [r["score"] for r in mr]
            ax.plot(xs, ys, marker=markers.get(method, "o"), color=colors.get(method), label=method, linewidth=2)
        ax.set_xscale("log", base=2)
        ax.set_xticks([0.0025, 0.005, 0.01, 0.02, 0.04])
        ax.get_xaxis().set_major_formatter(lambda x, pos: f"{x:g}")
        ax.set_xlabel("matrix LR")
        ax.set_ylabel("val BPB")
        ax.set_title(f"{bsz_label(batch)} LR sweep")
        ax.grid(True, alpha=0.25)
        # Tight y-axis around the observed range, not 0-3.
        if ys := [r["score"] for r in batch_rows]:
            lo, hi = min(ys), max(ys)
            pad = max(0.001, (hi - lo) * 0.15)
            ax.set_ylim(lo - pad, hi + pad)
        ax.legend(fontsize=8)
    for ax in list(axes.flat)[len(batches):]:
        ax.axis("off")
    fig.suptitle("Top-Aware Muon alpha and baseline LR sweeps")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "lr_sweeps.png", dpi=180)
    plt.close(fig)

    # Best-row delta plot.
    by = {(r["batch"], r["method"]): r for r in best}
    batches = sorted({r["batch"] for r in best})
    methods = sorted({r["method"] for r in best if r["method"] != "streaming_identity"})
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(batches))
    width = 0.8 / max(1, len(methods))
    for i, method in enumerate(methods):
        vals = []
        for batch in batches:
            ident = by.get((batch, "streaming_identity"))
            row = by.get((batch, method))
            vals.append(np.nan if ident is None or row is None else row["score"] - ident["score"])
        ax.bar(x + (i - (len(methods) - 1) / 2) * width, vals, width=width, label=method, color=colors.get(method))
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x, [bsz_label(b) for b in batches])
    ax.set_ylabel("Best BPB delta vs streaming identity")
    ax.set_title("Best tuned rows relative to StreamingMuon identity")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "best_delta_vs_identity.png", dpi=180)
    plt.close(fig)


def copy_to_refactored() -> None:
    if REFIG_DIR is None:
        return
    REFIG_DIR.mkdir(parents=True, exist_ok=True)
    for p in OUT_DIR.glob("*"):
        if p.is_file():
            (REFIG_DIR / p.name).write_bytes(p.read_bytes())


def main() -> None:
    rows = load_rows()
    best = best_rows(rows)
    write_csvs(rows, best)
    write_md(best)
    plot(rows, best)
    copy_to_refactored()
    print(f"wrote {OUT_DIR}")
    print(f"copied {REFIG_DIR}")


if __name__ == "__main__":
    main()
