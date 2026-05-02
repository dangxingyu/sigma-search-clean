#!/usr/bin/env python3
"""Analyze native Muon vs StreamingMuon identity match runs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("search_evals/ddp8_streaming_native_match_20260429_112759")


def load(name: str):
    path = ROOT / name / "result.json"
    d = json.loads(path.read_text())
    vals = d["val_bpbs"]
    return d, [v["step"] for v in vals], [v["val_bpb"] for v in vals]


def main() -> None:
    pairs = [
        ("32 step", "native_32", "streaming_tol001_32"),
        ("256 step", "native_256", "streaming_tol001_256"),
        ("1024 step", "native_1024", "streaming_tol001_1024"),
    ]

    print("horizon,native_final,streaming_final,native-streaming")
    for label, native_name, stream_name in pairs:
        _, _, yn = load(native_name)
        _, _, ys = load(stream_name)
        print(f"{label},{yn[-1]:.9f},{ys[-1]:.9f},{yn[-1]-ys[-1]:+.9f}")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax, axd = axes
    colors = {"native": "#1f77b4", "stream": "#d62728"}

    for label, native_name, stream_name in pairs:
        _, xn, yn = load(native_name)
        _, xs, ys = load(stream_name)
        if label == "32 step":
            alpha = 0.45
            lw = 1.6
        elif label == "256 step":
            alpha = 0.7
            lw = 2.0
        else:
            alpha = 1.0
            lw = 2.4
        ax.plot(xn, yn, marker="o", color=colors["native"], alpha=alpha, lw=lw, label=f"native {label}")
        ax.plot(xs, ys, marker="s", color=colors["stream"], alpha=alpha, lw=lw, label=f"stream tol=.01 {label}")
        common = sorted(set(xn) & set(xs))
        delta = []
        for step in common:
            delta.append(yn[xn.index(step)] - ys[xs.index(step)])
        axd.plot(common, delta, marker="o", lw=lw, alpha=alpha, label=label)

    ax.set_title("Native Muon vs StreamingMuon identity")
    ax.set_xlabel("step")
    ax.set_ylabel("val BPB")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    axd.axhline(0.0, color="black", lw=1, alpha=0.5)
    axd.set_title("Delta: native - streaming")
    axd.set_xlabel("step")
    axd.set_ylabel("BPB")
    axd.grid(True, alpha=0.3)
    axd.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig("streaming_native_match_analysis.png", dpi=160, bbox_inches="tight")
    print("wrote streaming_native_match_analysis.png")


if __name__ == "__main__":
    main()
