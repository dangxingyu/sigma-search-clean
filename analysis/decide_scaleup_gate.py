#!/usr/bin/env python3
"""Decide whether v26/v26b satisfy the d12 scale-up gate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


IDENTITY = "identity"
LITE = "lite_chi2_rs01"
TOP1 = "top1pm_a05"


def ensure_summary(root: Path) -> dict:
    summary = root / "v26_summary.json"
    if not summary.exists():
        subprocess.run([sys.executable, "analyze_v26_small_bsz_streaming_fair.py", str(root)], check=True)
    return json.loads(summary.read_text())


def best_map(summary: dict) -> dict[tuple[int, str], float]:
    out = {}
    for key, row in summary.get("best", {}).items():
        batch_s, method = key.split(":", 1)
        out[(int(batch_s), method)] = float(row["mean"])
    return out


def has_boundary_flags(summary: dict) -> bool:
    return bool(summary.get("boundary_flags"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("small_root", type=Path)
    parser.add_argument("large_root", type=Path)
    parser.add_argument("--tie-eps", type=float, default=3e-4)
    parser.add_argument("--allow-boundary", action="store_true")
    parser.add_argument("--decision-json", type=Path, default=Path("scaleup_gate_decision.json"))
    args = parser.parse_args()

    small_summary = ensure_summary(args.small_root)
    large_summary = ensure_summary(args.large_root)
    small = best_map(small_summary)
    large = best_map(large_summary)

    reasons = []
    pass_small = True
    pass_large = True

    small_batches = sorted({b for b, method in small if method == IDENTITY})
    large_batches = sorted({b for b, method in large if method == IDENTITY})

    if not small_batches:
        pass_small = False
        reasons.append("no small batches found")
    if not large_batches:
        pass_large = False
        reasons.append("no large batches found")

    small_checks = []
    for batch in small_batches:
        if (batch, LITE) not in small:
            pass_small = False
            reasons.append(f"small batch {batch} missing LITE-like")
            continue
        ident = small[(batch, IDENTITY)]
        lite = small[(batch, LITE)]
        ok = ident <= lite + args.tie_eps
        small_checks.append({"batch": batch, "identity": ident, "lite": lite, "identity_le_lite": ok})
        pass_small = pass_small and ok

    large_checks = []
    for batch in large_batches:
        missing = [m for m in (LITE, TOP1) if (batch, m) not in large]
        if missing:
            pass_large = False
            reasons.append(f"large batch {batch} missing {missing}")
            continue
        ident = large[(batch, IDENTITY)]
        lite = large[(batch, LITE)]
        top1 = large[(batch, TOP1)]
        ok = top1 < lite and lite < ident
        large_checks.append({"batch": batch, "identity": ident, "lite": lite, "top1": top1, "top1_lt_lite_lt_identity": ok})
        pass_large = pass_large and ok

    boundary_ok = args.allow_boundary or (not has_boundary_flags(small_summary) and not has_boundary_flags(large_summary))
    if not boundary_ok:
        reasons.append("LR boundary flags present")

    launch = bool(pass_small and pass_large and boundary_ok)
    decision = {
        "launch_v27": launch,
        "small_root": str(args.small_root),
        "large_root": str(args.large_root),
        "small_checks": small_checks,
        "large_checks": large_checks,
        "small_pass": pass_small,
        "large_pass": pass_large,
        "boundary_ok": boundary_ok,
        "reasons": reasons,
    }
    args.decision_json.write_text(json.dumps(decision, indent=2))
    print(json.dumps(decision, indent=2))
    return 0 if launch else 2


if __name__ == "__main__":
    raise SystemExit(main())
