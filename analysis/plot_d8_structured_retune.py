#!/usr/bin/env python3
"""Plot d8/262K/1x Shampoo and KL-Shampoo retune results."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("results/.matplotlib-cache").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("results")
SHAMPOO_ROOT = ROOT / "modal_d8_1x_h100_shampoo_retune_20260515_190755" / "raw_modal"
KL_SHAMPOO_ROOT = ROOT / "modal_d8_1x_a100_klshampoo_refpatch_20260515_194156" / "raw_modal"
OUT = ROOT / "modal_d8_1x_a100_klshampoo_refpatch_20260515_194156" / "figures"

REFERENCE_LINES = {
    "KL-SOAP best": 1.020622,
    "Plain Muon best": 1.024091,
    "SOAP-KL hparams": 1.022831,
}


def load_result(path: Path, method: str) -> dict:
    with path.open() as f:
        data = json.load(f)
    args = data["args"]
    return {
        "method": method,
        "lr": float(args["matrix_lr"]),
        "val_bpb_final": float(data["val_bpb_final"]),
        "train_loss_final": float(data["train_loss_final"]),
        "path": str(path),
    }


def collect() -> list[dict]:
    rows: list[dict] = []
    for path in sorted(SHAMPOO_ROOT.glob("shampoo_*/result.json")):
        rows.append(load_result(path, "Shampoo"))
    for path in sorted(KL_SHAMPOO_ROOT.glob("kl_shampoo_*/result.json")):
        rows.append(load_result(path, "KL-Shampoo ref-aligned"))
    return sorted(rows, key=lambda r: (r["method"], r["lr"]))


def write_csv(rows: list[dict]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "d8_1x_structured_retune_summary.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "lr", "val_bpb_final", "train_loss_final", "path"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot(rows: list[dict]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)

    styles = {
        "Shampoo": {"color": "#7c3aed", "marker": "s"},
        "KL-Shampoo ref-aligned": {"color": "#059669", "marker": "o"},
    }
    for method in ["Shampoo", "KL-Shampoo ref-aligned"]:
        group = [r for r in rows if r["method"] == method]
        if not group:
            continue
        group = sorted(group, key=lambda r: r["lr"])
        ax.plot(
            [r["lr"] for r in group],
            [r["val_bpb_final"] for r in group],
            linewidth=2.0,
            markersize=6,
            label=method,
            **styles[method],
        )

    for label, y in REFERENCE_LINES.items():
        ax.axhline(y, linestyle="--", linewidth=1.3, alpha=0.75, label=f"{label}: {y:.4f}")

    best = min(rows, key=lambda r: r["val_bpb_final"])
    ax.scatter([best["lr"]], [best["val_bpb_final"]], s=130, facecolors="none", edgecolors="black", linewidths=1.5)
    ax.annotate(
        f"best {best['method']}\n{best['val_bpb_final']:.4f} @ {best['lr']:g}",
        xy=(best["lr"], best["val_bpb_final"]),
        xytext=(best["lr"] * 1.25, best["val_bpb_final"] + 0.035),
        arrowprops={"arrowstyle": "->", "lw": 1.0},
        fontsize=9,
    )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Base matrix LR")
    ax.set_ylabel("Final validation BPB")
    ax.set_title("d8 / 262K / 1x Chinchilla: Shampoo Retune")
    ax.set_ylim(1.0, 1.70)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=8)

    path = OUT / "d8_1x_structured_retune_lr_sweep.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    rows = collect()
    csv_path = write_csv(rows)
    fig_path = plot(rows)
    print(fig_path)
    print(csv_path)


if __name__ == "__main__":
    main()
