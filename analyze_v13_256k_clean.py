"""Analyze v13 clean 256K baselines and compare D1 against v9."""

import glob
import json
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


V9_DIR = "sweep_results_v9"
V13_DIR = "sweep_results_v13_256k_clean"
OUT = "v13_256k_d1_analysis.png"


def label(bsz):
    return f"{bsz // 1024}K" if bsz < 1_000_000 else f"{bsz // 1_000_000}M"


def score(data):
    return data.get("val_bpb_best", data.get("score"))


def parse_run(path):
    base = os.path.basename(path)
    m = re.match(r"(muon|lite)_bsz(\d+)_lr([\d.]+)_s(\d+)\.json", base)
    if not m:
        return None
    return m.group(1), int(m.group(2)), float(m.group(3)), int(m.group(4))


def load_runs(*dirs):
    runs = {}
    for directory in dirs:
        for path in glob.glob(f"{directory}/*.json"):
            key = parse_run(path)
            if key is None:
                continue
            data = json.load(open(path))
            if score(data) is None:
                continue
            opt, bsz, lr, seed = key
            runs[(opt, bsz, seed)] = data
    return runs


def mean_std(vals):
    arr = np.asarray(vals, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.nan, np.nan
    return float(np.mean(arr)), float(np.std(arr))


def telemetry_series(data, key, metric):
    probes = (data.get("telemetry") or {}).get(key) or []
    xs, ys = [], []
    for probe in probes:
        val = probe.get(metric)
        if val is not None:
            xs.append(probe.get("step", len(xs)))
            ys.append(float(val))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def summarize_d1(data, key, metric):
    _, ys = telemetry_series(data, key, metric)
    if len(ys) == 0:
        return np.nan, np.nan, np.nan
    return float(ys[0]), float(np.mean(ys)), float(ys[-1])


def main():
    runs = load_runs(V9_DIR, V13_DIR)
    print(f"Loaded paired candidate runs: {len(runs)}")

    print("\n=== Clean 256K result ===")
    bsz = 262144
    paired = [
        seed for seed in (42, 43, 44)
        if ("muon", bsz, seed) in runs and ("lite", bsz, seed) in runs
    ]
    if paired:
        muon = [score(runs[("muon", bsz, seed)]) for seed in paired]
        lite = [score(runs[("lite", bsz, seed)]) for seed in paired]
        delta = [m - l for m, l in zip(muon, lite)]
        print(
            f"256K n={len(paired)} Muon={np.mean(muon):.4f} "
            f"LITE={np.mean(lite):.4f} Δ={np.mean(delta):+.4f} ± {np.std(delta):.4f}"
        )
    else:
        print("No fully paired 256K rows yet.")

    print("\n=== Muon D1(M_tilde) LITE-side trajectory summary ===")
    rows = []
    for bsz in [262144, 524288, 1048576, 4194304, 16777216]:
        vals_cstar, vals_hard = [], []
        for seed in (42, 43, 44):
            data = runs.get(("muon", bsz, seed))
            if not data:
                continue
            vals_cstar.append(summarize_d1(data, "d1_probes_momentum", "frac_above_c_star_median"))
            vals_hard.append(summarize_d1(data, "d1_probes_momentum", "frac_above_0_7_median"))
        if not vals_cstar:
            continue
        arr_c = np.asarray(vals_cstar, dtype=float)
        arr_h = np.asarray(vals_hard, dtype=float)
        row = {
            "bsz": bsz,
            "c_first": np.nanmean(arr_c[:, 0]),
            "c_mean": np.nanmean(arr_c[:, 1]),
            "c_last": np.nanmean(arr_c[:, 2]),
            "h_first": np.nanmean(arr_h[:, 0]),
            "h_mean": np.nanmean(arr_h[:, 1]),
            "h_last": np.nanmean(arr_h[:, 2]),
        }
        rows.append(row)
        print(
            f"{label(bsz):>6s}: c* first/mean/last="
            f"{row['c_first']:.3f}/{row['c_mean']:.3f}/{row['c_last']:.3f}; "
            f">0.7 first/mean/last="
            f"{row['h_first']:.3f}/{row['h_mean']:.3f}/{row['h_last']:.3f}"
        )

    if rows:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        xs = np.arange(len(rows))
        labels = [label(row["bsz"]) for row in rows]
        axes[0].plot(xs, [row["c_first"] for row in rows], marker="o", label="first")
        axes[0].plot(xs, [row["c_mean"] for row in rows], marker="o", label="mean")
        axes[0].plot(xs, [row["c_last"] for row in rows], marker="o", label="last")
        axes[0].set_title("Muon D1(M_tilde), c*")
        axes[0].set_xticks(xs, labels, rotation=20)
        axes[0].set_ylim(0, 1.05)
        axes[0].grid(True, alpha=0.25)
        axes[0].legend()

        axes[1].plot(xs, [row["h_first"] for row in rows], marker="o", label="first")
        axes[1].plot(xs, [row["h_mean"] for row in rows], marker="o", label="mean")
        axes[1].plot(xs, [row["h_last"] for row in rows], marker="o", label="last")
        axes[1].set_title("Muon D1(M_tilde), c > 0.7")
        axes[1].set_xticks(xs, labels, rotation=20)
        axes[1].set_ylim(0, 1.05)
        axes[1].grid(True, alpha=0.25)
        axes[1].legend()

        fig.suptitle("v13 adds clean 256K lower-edge D1 calibration")
        fig.tight_layout()
        fig.savefig(OUT, dpi=180)
        print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
