#!/usr/bin/env python3
"""Analyze the d12 1B-token native Muon vs StreamingMuon identity run."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("search_evals/ddp8_streaming_native_d12_1b_20260429_140201")
OUT = Path("streaming_native_d12_1b_analysis.png")


def load(name: str) -> tuple[list[int], list[float], dict]:
    data = json.loads((ROOT / name / "result.json").read_text())
    vals = data["val_bpbs"]
    return [v["step"] for v in vals], [v["val_bpb"] for v in vals], data


def main() -> None:
    xn, yn, native = load("native_d12_4096")
    xs, ys, stream = load("streaming_identity_tol001_d12_4096")
    common = sorted(set(xn) & set(xs))
    native_by_step = dict(zip(xn, yn))
    stream_by_step = dict(zip(xs, ys))
    delta = [native_by_step[s] - stream_by_step[s] for s in common]

    print("step,native,streaming,native-streaming")
    for s in common:
        print(f"{s},{native_by_step[s]:.9f},{stream_by_step[s]:.9f},{native_by_step[s]-stream_by_step[s]:+.9f}")
    print(f"native_score={native['score']:.9f}")
    print(f"streaming_score={stream['score']:.9f}")
    print(f"final_delta_native_minus_streaming={native['score'] - stream['score']:+.9f}")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax, axd = axes
    ax.plot(xn, yn, marker="o", lw=2.4, label="native Muon")
    ax.plot(xs, ys, marker="s", lw=2.4, label="StreamingMuon identity tol=.01")
    ax.set_title("d12, 1.07B tokens, 8-GPU DDP")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("val BPB")
    ax.set_ylim(0.84, 1.10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    axd.axhline(0.0, color="black", lw=1, alpha=0.5)
    axd.plot(common, delta, marker="o", lw=2.4, color="#d62728")
    axd.set_title("Delta: native - streaming")
    axd.set_xlabel("optimizer step")
    axd.set_ylabel("BPB")
    axd.set_ylim(-0.0025, 0.0025)
    axd.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT, dpi=170, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
