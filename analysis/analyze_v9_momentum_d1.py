"""Analyze v9 momentum-D1 sweep: compare D1(g) vs D1(M_tilde).

For each (bsz, opt, seed) run, both probes were computed at the same 5 mid-training
steps. `D1(M_tilde)` is computed on the split-half Nesterov-corrected momentum
direction that Muon/LITE actually sends into polar/LITE projection. Within a run,
this gives paired values; across seeds, error bars; across bsz, regime sweep.

Questions:
  Q1. Is D1(M_tilde) less noisy than D1(g)?
       — compare std across seeds at each (bsz, opt, probe)
  Q2. Does D1(M_tilde) show stronger contrast between Muon-favored small-bsz and
       LITE-favored large-bsz regimes?
       — compare Muon vs LITE Δ_D1 = (D1_lite - D1_muon) under both probes
  Q3. Does D1(M_tilde) better predict Δ val_bpb?
       — correlate D1 with the bpb gap (Muon - LITE) per bsz
  Q4. How does D1(M_tilde) vary within a run as training progresses?
       — trajectories per (bsz, opt, seed)
"""
import glob, json, re, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_v9(path="sweep_results_v9"):
    runs = []
    for f in glob.glob(f"{path}/*.json"):
        m = re.match(r".*?(\w+)_bsz(\d+)_lr([\d.]+)_s(\d+)\.json", f)
        if not m:
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        opt, bsz, lr, seed = m.group(1), int(m.group(2)), float(m.group(3)), int(m.group(4))
        bpb = d.get("val_bpb_best")
        if bpb is None:
            continue
        tel = d.get("telemetry", {}) or {}
        runs.append({
            "opt": opt, "bsz": bsz, "lr": lr, "seed": seed,
            "bpb": bpb,
            "d1_g": tel.get("d1_probes", []),
            "d1_m": tel.get("d1_probes_momentum", []),
            "d1_ml": tel.get("d1_probes_momentum_left", []),
            "d1_mb": tel.get("d1_probes_momentum_buffer", []),
        })
    return runs


def bsz_str(bsz):
    return f"{bsz//1024}K" if bsz < 1_000_000 else f"{bsz//1_000_000}M"


def probe_metric(run, key, probe_index=-1, metric="frac_above_c_star_median"):
    probes = run.get(key) or []
    if not probes:
        return float("nan")
    try:
        val = probes[probe_index].get(metric)
    except IndexError:
        return float("nan")
    return float(val) if val is not None else float("nan")


def probe_mean(run, key, metric="frac_above_c_star_median"):
    vals = [p.get(metric) for p in (run.get(key) or [])]
    vals = [float(v) for v in vals if v is not None]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    runs = load_v9()
    print(f"Loaded {len(runs)} runs")
    if not runs:
        print("No v9 runs yet; nothing to analyze.")
        return

    # Group by (bsz, opt) for pair analysis
    by_bo = defaultdict(list)
    for r in runs:
        by_bo[(r["bsz"], r["opt"])].append(r)

    bszs = sorted({r["bsz"] for r in runs})
    print(f"\nBatch sizes: {[bsz_str(b) for b in bszs]}")

    # ---- Q1: noise comparison D1(g) vs D1(m) at the LAST probe per run ----
    print("\n=== Q1: D1 mean/std per (bsz, opt) at LAST probe ===")
    print(f"{'bsz':>6s} | {'opt':>5s} | {'D1(g) mean ± std':>22s} | {'D1(M_tilde) mean ± std':>28s} | n")
    for bsz in bszs:
        for opt in ("muon", "lite"):
            rs = by_bo.get((bsz, opt), [])
            g_last = [r["d1_g"][-1].get("frac_above_c_star_median") for r in rs if r["d1_g"]]
            m_last = [r["d1_m"][-1].get("frac_above_c_star_median") for r in rs if r["d1_m"]]
            g_last = [x for x in g_last if x is not None]
            m_last = [x for x in m_last if x is not None]
            if not (g_last or m_last):
                continue
            print(f"{bsz_str(bsz):>6s} | {opt:>5s} | "
                  f"{np.mean(g_last):.3f} ± {np.std(g_last):.3f} (n={len(g_last):>2d})  | "
                  f"{np.mean(m_last) if m_last else float('nan'):.3f} ± "
                  f"{np.std(m_last) if m_last else float('nan'):.3f} (n={len(m_last):>2d})")

    print("\n=== Q1b: side/buffer ablation at LAST probe ===")
    print(f"{'bsz':>6s} | {'opt':>5s} | {'M_tilde lite-side':>18s} | {'M_tilde left-U':>16s} | {'buffer lite-side':>18s}")
    for bsz in bszs:
        for opt in ("muon", "lite"):
            rs = by_bo.get((bsz, opt), [])
            primary = [r["d1_m"][-1].get("frac_above_c_star_median") for r in rs if r["d1_m"]]
            left = [r["d1_ml"][-1].get("frac_above_c_star_median") for r in rs if r.get("d1_ml")]
            buf = [r["d1_mb"][-1].get("frac_above_c_star_median") for r in rs if r.get("d1_mb")]
            if not (primary or left or buf):
                continue
            mean = lambda xs: np.mean([x for x in xs if x is not None]) if xs else float("nan")
            print(f"{bsz_str(bsz):>6s} | {opt:>5s} | {mean(primary):>18.3f} | {mean(left):>16.3f} | {mean(buf):>18.3f}")

    print("\n=== Q1c: hard 0.7 vs c* threshold for D1(M_tilde) at LAST probe ===")
    print(f"{'bsz':>6s} | {'opt':>5s} | {'M_tilde >0.7':>14s} | {'M_tilde >c*':>13s} | {'buffer >0.7':>12s} | {'buffer >c*':>11s}")
    for bsz in bszs:
        for opt in ("muon", "lite"):
            rs = by_bo.get((bsz, opt), [])
            m07 = [r["d1_m"][-1].get("frac_above_0_7_median") for r in rs if r["d1_m"]]
            mcs = [r["d1_m"][-1].get("frac_above_c_star_median") for r in rs if r["d1_m"]]
            b07 = [r["d1_mb"][-1].get("frac_above_0_7_median") for r in rs if r.get("d1_mb")]
            bcs = [r["d1_mb"][-1].get("frac_above_c_star_median") for r in rs if r.get("d1_mb")]
            if not (m07 or mcs or b07 or bcs):
                continue
            mean = lambda xs: np.mean([x for x in xs if x is not None]) if xs else float("nan")
            print(f"{bsz_str(bsz):>6s} | {opt:>5s} | {mean(m07):>14.3f} | {mean(mcs):>13.3f} | "
                  f"{mean(b07):>12.3f} | {mean(bcs):>11.3f}")

    # ---- Q2: D1 contrast Muon - LITE per bsz, both probes ----
    print("\n=== Q2: contrast Δ_D1 = D1(LITE) - D1(Muon), both probes (last probe) ===")
    print(f"{'bsz':>6s} | {'Δ_D1(g)':>10s} | {'Δ_D1(m)':>10s} | {'Δ_bpb':>10s}")
    delta_table = []
    for bsz in bszs:
        muon_runs = by_bo.get((bsz, "muon"), [])
        lite_runs = by_bo.get((bsz, "lite"), [])
        if not (muon_runs and lite_runs):
            continue
        common_seeds = sorted(set(r["seed"] for r in muon_runs) &
                              set(r["seed"] for r in lite_runs))
        if not common_seeds:
            continue
        muon_by_seed = {r["seed"]: r for r in muon_runs}
        lite_by_seed = {r["seed"]: r for r in lite_runs}
        d_g_list, d_m_list, d_bpb_list = [], [], []
        for s in common_seeds:
            mr, lr = muon_by_seed[s], lite_by_seed[s]
            if mr["d1_g"] and lr["d1_g"]:
                d_g_list.append(lr["d1_g"][-1].get("frac_above_c_star_median", float("nan"))
                              - mr["d1_g"][-1].get("frac_above_c_star_median", float("nan")))
            if mr["d1_m"] and lr["d1_m"]:
                d_m_list.append(lr["d1_m"][-1].get("frac_above_c_star_median", float("nan"))
                              - mr["d1_m"][-1].get("frac_above_c_star_median", float("nan")))
            d_bpb_list.append(mr["bpb"] - lr["bpb"])
        delta_table.append({
            "bsz": bsz,
            "d_g_mean": float(np.mean(d_g_list)) if d_g_list else float("nan"),
            "d_g_std": float(np.std(d_g_list)) if d_g_list else float("nan"),
            "d_m_mean": float(np.mean(d_m_list)) if d_m_list else float("nan"),
            "d_m_std": float(np.std(d_m_list)) if d_m_list else float("nan"),
            "d_bpb_mean": float(np.mean(d_bpb_list)) if d_bpb_list else float("nan"),
            "d_bpb_std": float(np.std(d_bpb_list)) if d_bpb_list else float("nan"),
            "n": len(common_seeds),
        })
        d = delta_table[-1]
        print(f"{bsz_str(bsz):>6s} | "
              f"{d['d_g_mean']:>+.3f}±{d['d_g_std']:.3f} | "
              f"{d['d_m_mean']:>+.3f}±{d['d_m_std']:.3f} | "
              f"{d['d_bpb_mean']:>+.4f}±{d['d_bpb_std']:.4f}")

    # ---- Q3: correlation D1 ↔ Δ_bpb across bsz ----
    if len(delta_table) >= 3:
        print("\n=== Q3: correlation between D1 contrast and Δ_bpb across bsz ===")
        bszs_v = [d["bsz"] for d in delta_table]
        d_g_v = [d["d_g_mean"] for d in delta_table]
        d_m_v = [d["d_m_mean"] for d in delta_table]
        d_bpb_v = [d["d_bpb_mean"] for d in delta_table]
        # Pearson correlation (clean rank-based not used)
        def corr(a, b):
            a, b = np.array(a), np.array(b)
            mask = ~(np.isnan(a) | np.isnan(b))
            if mask.sum() < 2:
                return float("nan")
            return float(np.corrcoef(a[mask], b[mask])[0, 1])
        print(f"  corr(Δ_D1(g), Δ_bpb) across bsz = {corr(d_g_v, d_bpb_v):+.3f}")
        print(f"  corr(Δ_D1(M_tilde), Δ_bpb) across bsz = {corr(d_m_v, d_bpb_v):+.3f}")
        print(f"  corr(D1(M_tilde), D1(g))   across bsz = {corr(d_m_v, d_g_v):+.3f}")

        print("\n=== Q3b: absolute D1 as a deployable switch signal ===")
        print("Using paired seeds; positive Δ_bpb means LITE beats Muon.")
        print(f"{'bsz':>6s} | {'Muon D1_m first':>15s} | {'Muon D1_m mean':>14s} | {'Muon D1_m last':>14s} | {'LITE D1_m last':>14s} | {'Δ_bpb':>10s}")
        abs_rows = []
        for bsz in bszs:
            muon_runs = by_bo.get((bsz, "muon"), [])
            lite_runs = by_bo.get((bsz, "lite"), [])
            common_seeds = sorted(set(r["seed"] for r in muon_runs) &
                                  set(r["seed"] for r in lite_runs))
            if not common_seeds:
                continue
            muon_by_seed = {r["seed"]: r for r in muon_runs}
            lite_by_seed = {r["seed"]: r for r in lite_runs}
            rows = []
            for s in common_seeds:
                mr, lr = muon_by_seed[s], lite_by_seed[s]
                rows.append({
                    "muon_m_first": probe_metric(mr, "d1_m", 0),
                    "muon_m_mean": probe_mean(mr, "d1_m"),
                    "muon_m_last": probe_metric(mr, "d1_m", -1),
                    "lite_m_last": probe_metric(lr, "d1_m", -1),
                    "muon_g_last": probe_metric(mr, "d1_g", -1),
                    "delta_bpb": mr["bpb"] - lr["bpb"],
                })
            def clean_mean(name):
                vals = np.array([r[name] for r in rows], dtype=float)
                vals = vals[~np.isnan(vals)]
                return float(np.mean(vals)) if len(vals) else float("nan")
            abs_rows.append({
                "bsz": bsz,
                "muon_m_first": clean_mean("muon_m_first"),
                "muon_m_mean": clean_mean("muon_m_mean"),
                "muon_m_last": clean_mean("muon_m_last"),
                "lite_m_last": clean_mean("lite_m_last"),
                "muon_g_last": clean_mean("muon_g_last"),
                "delta_bpb": clean_mean("delta_bpb"),
            })
            row = abs_rows[-1]
            print(f"{bsz_str(bsz):>6s} | {row['muon_m_first']:>15.3f} | "
                  f"{row['muon_m_mean']:>14.3f} | {row['muon_m_last']:>14.3f} | "
                  f"{row['lite_m_last']:>14.3f} | {row['delta_bpb']:>+10.4f}")

        if len(abs_rows) >= 3:
            d_bpb_abs = [r["delta_bpb"] for r in abs_rows]
            print("  correlations against Δ_bpb across bsz:")
            for key, label in [
                ("muon_g_last", "Muon D1(g) last"),
                ("muon_m_first", "Muon D1(M_tilde) first"),
                ("muon_m_mean", "Muon D1(M_tilde) mean"),
                ("muon_m_last", "Muon D1(M_tilde) last"),
                ("lite_m_last", "LITE D1(M_tilde) last"),
            ]:
                print(f"    corr({label}, Δ_bpb) = {corr([r[key] for r in abs_rows], d_bpb_abs):+.3f}")

    # ---- Q4 + plots ----
    if len(delta_table) >= 2:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Panel A: D1 trajectories per (bsz, opt) — average across seeds
        ax = axes[0, 0]
        colors = plt.cm.viridis(np.linspace(0, 1, len(bszs)))
        for i, bsz in enumerate(bszs):
            for opt, ls in [("muon", "-"), ("lite", "--")]:
                rs = by_bo.get((bsz, opt), [])
                if not rs:
                    continue
                # Stack probes across seeds
                steps_per_run = [[p.get("step") for p in r["d1_m"]] for r in rs if r["d1_m"]]
                vals_per_run = [[p.get("frac_above_c_star_median") for p in r["d1_m"]] for r in rs if r["d1_m"]]
                if not steps_per_run:
                    continue
                # Align by probe index (assume all runs have same probe schedule)
                arr = np.array(vals_per_run, dtype=float)
                mean = np.nanmean(arr, axis=0)
                std = np.nanstd(arr, axis=0)
                idx = np.arange(len(mean))
                ax.errorbar(idx, mean, yerr=std, fmt=ls + "o", color=colors[i], lw=1.5,
                            label=f"{bsz_str(bsz)} {opt}", alpha=0.85, capsize=3)
        ax.set_xlabel("probe index (1/6 → 5/6 of training)")
        ax.set_ylabel("D1(M_tilde) frac > c* (median across layers)")
        ax.set_title("D1(M_tilde) trajectory by (bsz, opt) — bands across seeds")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)

        # Panel B: D1(g) vs D1(M_tilde) scatter (paired, last probe per run)
        ax = axes[0, 1]
        for opt, marker in [("muon", "o"), ("lite", "s")]:
            xs, ys = [], []
            for r in runs:
                if r["opt"] != opt: continue
                if not (r["d1_g"] and r["d1_m"]): continue
                xs.append(r["d1_g"][-1].get("frac_above_c_star_median", float("nan")))
                ys.append(r["d1_m"][-1].get("frac_above_c_star_median", float("nan")))
            ax.scatter(xs, ys, marker=marker, alpha=0.7, label=opt.upper(), s=50)
        lim = [0, 1]
        ax.plot(lim, lim, "k--", alpha=0.4, lw=0.5, label="y = x")
        ax.set_xlabel("D1(g) frac > c*")
        ax.set_ylabel("D1(M_tilde) frac > c*")
        ax.set_title("D1(M_tilde) vs D1(g) at last probe (paired per run)")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.legend()
        ax.grid(alpha=0.3)

        # Panel C: Δ_D1 vs bsz (both probes) and Δ_bpb
        ax = axes[1, 0]
        bszs_v = [d["bsz"] for d in delta_table]
        log_bsz = np.log2(bszs_v)
        ax.errorbar(log_bsz, [d["d_g_mean"] for d in delta_table],
                    yerr=[d["d_g_std"] for d in delta_table],
                    fmt="o-", lw=2, label="Δ_D1(g) = LITE − Muon", capsize=3)
        ax.errorbar(log_bsz, [d["d_m_mean"] for d in delta_table],
                    yerr=[d["d_m_std"] for d in delta_table],
                    fmt="s-", lw=2, label="Δ_D1(M_tilde) = LITE − Muon", capsize=3)
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xticks(log_bsz)
        ax.set_xticklabels([bsz_str(b) for b in bszs_v])
        ax.set_xlabel("batch size (log2 spaced)")
        ax.set_ylabel("Δ_D1 (LITE − Muon)")
        ax.set_title("D1 contrast vs bsz — does D1(M_tilde) discriminate better than D1(g)?")
        ax.legend()
        ax.grid(alpha=0.3)

        # Panel D: D1(M_tilde) at last probe vs Δ_bpb (validates predictive value)
        ax = axes[1, 1]
        d_bpb_v = [d["d_bpb_mean"] for d in delta_table]
        d_bpb_e = [d["d_bpb_std"] for d in delta_table]
        d_m_v = [d["d_m_mean"] for d in delta_table]
        d_m_e = [d["d_m_std"] for d in delta_table]
        d_g_v = [d["d_g_mean"] for d in delta_table]
        d_g_e = [d["d_g_std"] for d in delta_table]
        ax.errorbar(d_g_v, d_bpb_v, xerr=d_g_e, yerr=d_bpb_e,
                    fmt="o", lw=1, ms=10, label="Δ_D1(g)", capsize=3, alpha=0.7)
        ax.errorbar(d_m_v, d_bpb_v, xerr=d_m_e, yerr=d_bpb_e,
                    fmt="s", lw=1, ms=10, label="Δ_D1(M_tilde)", capsize=3, alpha=0.7)
        ax.axhline(0, color='k', lw=0.5)
        ax.axvline(0, color='k', lw=0.5)
        for i, b in enumerate(bszs_v):
            ax.annotate(bsz_str(b), (d_m_v[i], d_bpb_v[i]),
                        xytext=(5, 5), textcoords="offset points", fontsize=8)
        ax.set_xlabel("D1 contrast (LITE − Muon) at last probe")
        ax.set_ylabel("Δ_bpb (Muon − LITE)")
        ax.set_title("D1 contrast predicts Δ_bpb? Larger Δ_D1 → larger LITE win?")
        ax.legend()
        ax.grid(alpha=0.3)

        fig.tight_layout()
        out = "v9_momentum_d1_analysis.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        print(f"\nwrote {out}")

    # ---- Sanity: cross-validate v9 D1(g) vs v6/v7 D1(g) at same bsz ----
    print("\n=== Sanity: v9 D1(g) vs v6/v7 D1(g) at same nominal bsz ===")
    v6v7_runs = []
    for path in ("sweep_results_v6", "sweep_results_v7", "modal_results"):
        for f in glob.glob(f"{path}/*.json"):
            m = re.match(r".*?(\w+)_bsz(\d+)_lr([\d.]+)_s(\d+)\.json", f)
            if not m:
                continue
            try:
                d = json.load(open(f))
            except Exception:
                continue
            tel = d.get("telemetry", {}) or {}
            probes = tel.get("d1_probes", [])
            if not probes:
                continue
            v6v7_runs.append({
                "src": path, "opt": m.group(1), "bsz": int(m.group(2)),
                "seed": int(m.group(4)),
                "d1_last": probes[-1].get("frac_above_c_star_median"),
            })
    print(f"  loaded {len(v6v7_runs)} prior runs with D1 telemetry")
    for bsz in bszs:
        for opt in ("muon", "lite"):
            v9_vals = [r["d1_g"][-1].get("frac_above_c_star_median") for r in by_bo.get((bsz, opt), [])
                       if r["d1_g"]]
            prior_vals = [p["d1_last"] for p in v6v7_runs
                          if p["bsz"] == bsz and p["opt"] == opt and p["d1_last"] is not None]
            if v9_vals and prior_vals:
                print(f"  {bsz_str(bsz):>6s} {opt}: v9 D1(g) {np.mean(v9_vals):.3f}±{np.std(v9_vals):.3f} (n={len(v9_vals)})  "
                      f"vs prior {np.mean(prior_vals):.3f}±{np.std(prior_vals):.3f} (n={len(prior_vals)})")


if __name__ == "__main__":
    main()
