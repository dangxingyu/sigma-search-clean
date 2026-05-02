#!/usr/bin/env python3
"""Build a compact historical catalog from curated sweep CSV/JSON summaries."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "sweep_catalog"

FIELDS = [
    "catalog_source",
    "source",
    "family",
    "batch_label",
    "batch",
    "method",
    "variant",
    "alpha",
    "top_k",
    "lr",
    "seed",
    "score",
    "status",
    "quality",
    "tokens",
    "depth",
    "path",
    "notes",
]


def label_batch(batch: str | int) -> str:
    b = int(batch)
    if b % 1_048_576 == 0:
        return f"{b // 1_048_576}M"
    if b % 1024 == 0:
        return f"{b // 1024}K"
    return str(b)


def parse_float_token(token: str) -> float | None:
    if token is None or token == "":
        return None
    token = str(token).strip().replace("p", ".")
    if token.isdigit() and token.startswith("0") and len(token) > 1:
        return float("0." + token[1:])
    try:
        return float(token)
    except ValueError:
        pass
    return None


def parse_alpha(raw: str) -> float | None:
    raw = raw.lower()
    number = r"([0-9]*\.[0-9]+|[0-9]+p[0-9]+|[0-9]+)"
    for pattern in (rf"alpha[_=]?{number}", rf"_a{number}"):
        match = re.search(pattern, raw)
        if match:
            return parse_float_token(match.group(1))
    return None


def normalize_method(raw: str, family_hint: str = "") -> tuple[str, str, float | None, int | None]:
    raw = (raw or "").strip()
    low = raw.lower()
    if low in {"identity", "streaming_identity"}:
        return "streaming_identity", "alpha=1 identity", 1.0, None
    if low in {"streaming_lite", "lite_chi2_rs01"}:
        return "streaming_lite", "LITE-like chi2 tail", None, None
    if low in {"native_muon", "muon"} or (low == "muon" and family_hint):
        return "native_muon", "native Muon", None, None
    if low in {"native_lite", "lite"}:
        return "native_lite", "native LITE", None, None
    if "top" in low:
        alpha = parse_alpha(low)
        top_k = 1
        match = re.search(r"k([0-9]+)", low)
        if match:
            top_k = int(match.group(1))
        if alpha is None and "a05" in low:
            alpha = 0.5
        return "top_aware_muon", f"top_k={top_k}, alpha={alpha}", alpha, top_k
    return low or "unknown", raw, None, None


def infer_family(source: str, method: str, csv_family: str = "") -> str:
    if csv_family:
        return csv_family
    source = source.lower()
    if "native_single" in source:
        return "native_single_gpu"
    if method.startswith("native"):
        return "native_ddp_or_control"
    if "dynamics" in source:
        return "dynamics_smoke"
    return "same_driver_streaming_ddp"


def infer_quality(path: str, tokens: str = "") -> str:
    if tokens:
        try:
            t = int(float(tokens))
            if 350_000_000 <= t <= 450_000_000:
                return "full_0p4b"
            if t >= 900_000_000:
                return "full_1b"
        except ValueError:
            pass
    if "200step" in path or "_200" in path:
        return "smoke_200step"
    if "v34" in path:
        return "smoke_200step"
    return "full_or_historical"


def add(rows: list[dict], *, catalog_source: str, source: str, raw_method: str,
        batch: str | int, lr: str = "", seed: str = "", score: str = "",
        status: str = "complete", path: str = "", family: str = "",
        tokens: str = "", depth: str = "8", notes: str = "") -> None:
    method, variant, alpha, top_k = normalize_method(raw_method, family)
    fam = infer_family(source or catalog_source, method, family)
    rows.append({
        "catalog_source": catalog_source,
        "source": source,
        "family": fam,
        "batch_label": label_batch(batch),
        "batch": str(int(batch)),
        "method": method,
        "variant": variant,
        "alpha": "" if alpha is None else f"{alpha:g}",
        "top_k": "" if top_k is None else str(top_k),
        "lr": "" if lr in (None, "") else f"{float(lr):g}",
        "seed": "" if seed in (None, "") else str(seed),
        "score": "" if score in (None, "") else f"{float(score):.12g}",
        "status": status or "complete",
        "quality": infer_quality(path, tokens),
        "tokens": tokens,
        "depth": depth,
        "path": path,
        "notes": notes,
    })


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_catalog_rows() -> list[dict]:
    rows: list[dict] = []

    for r in read_csv_rows(ROOT / "figures" / "topaware_alpha_dashboard" / "raw_rows.csv"):
        add(rows, catalog_source="figures/topaware_alpha_dashboard/raw_rows.csv",
            source=r.get("source", ""), raw_method=r["method"], batch=r["batch"],
            lr=r.get("lr", ""), seed=r.get("seed", ""), score=r.get("score", ""),
            status=r.get("status", "complete"), path=r.get("path", ""))

    for r in read_csv_rows(ROOT / "figures" / "current_same_driver_v26" / "raw_rows.csv"):
        add(rows, catalog_source="figures/current_same_driver_v26/raw_rows.csv",
            source="v26_streaming_small", raw_method=r["method"], batch=r["batch"],
            lr=r.get("lr", ""), seed=r.get("seed", ""), score=r.get("score", ""),
            status="complete", path=r.get("path", ""))

    for r in read_csv_rows(ROOT / "figures" / "historical_large_batch" / "raw_rows.csv"):
        add(rows, catalog_source="figures/historical_large_batch/raw_rows.csv",
            source=r.get("family", ""), family=r.get("family", ""),
            raw_method=r["optimizer"], batch=r["batch"], lr=r.get("lr", ""),
            seed=r.get("seed", ""), score=r.get("score", ""), path=r.get("path", ""),
            notes="historical native Muon/LITE or older streaming Top-Aware control")

    for r in read_csv_rows(ROOT / "figures" / "native_streaming_topk" / "native_streaming_topk_raw.csv"):
        add(rows, catalog_source="figures/native_streaming_topk/native_streaming_topk_raw.csv",
            source=r.get("source", ""), raw_method=r["method"], batch=r["batch"],
            lr=r.get("lr", ""), seed=r.get("seed", ""), score=r.get("score", ""),
            status="complete", path=r.get("path", ""), notes=r.get("lr_status", ""))

    for summary in sorted((ROOT / "results").glob("*/summary.json")):
        data = json.loads(summary.read_text())
        recipe = data.get("recipe", {})
        records = data.get("records", [])
        for r in records:
            batch = r.get("batch_size", recipe.get("batch_size"))
            if batch is None:
                continue
            raw_method = r.get("method", recipe.get("method", ""))
            alpha = r.get("alpha", recipe.get("alpha"))
            if raw_method == "top_aware_muon" and alpha is not None:
                raw_method = f"top_aware_k{r.get('top_k', recipe.get('top_k', 1))}_a{alpha}"
            add(rows, catalog_source=str(summary.relative_to(ROOT)),
                source=data.get("id", summary.parent.name), raw_method=raw_method,
                batch=batch, lr=r.get("lr", recipe.get("lr", "")),
                seed=r.get("seed", recipe.get("seed", "")), score=r.get("score", ""),
                status=r.get("status", "complete"), path=r.get("source", ""),
                tokens=str(recipe.get("tokens", "")), depth=str(recipe.get("depth", "8")),
                notes=data.get("description", ""))

    # Deduplicate identical raw rows imported from multiple figure dashboards.
    deduped: list[dict] = []
    seen = set()
    for r in rows:
        key = (r["path"], r["method"], r["batch"], r["lr"], r["seed"], r["alpha"], r["score"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    return deduped


def write_csv(path: Path, rows: list[dict], fields: list[str] = FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})


def best_rows(rows: list[dict]) -> list[dict]:
    best: dict[tuple, dict] = {}
    for r in rows:
        if not r["score"] or r["status"] != "complete":
            continue
        key = (r["family"], r["batch"], r["method"], r["alpha"])
        if key not in best or float(r["score"]) < float(best[key]["score"]):
            best[key] = r
    return sorted(best.values(), key=lambda r: (int(r["batch"]), r["family"], r["method"], r["alpha"]))


def write_readme(rows: list[dict], best: list[dict]) -> None:
    by_source = Counter(r["catalog_source"] for r in rows)
    by_method = Counter(r["method"] for r in rows)
    batches = sorted({int(r["batch"]) for r in rows})
    alpha_values = sorted({r["alpha"] for r in rows if r["alpha"]}, key=lambda x: float(x))

    lines = [
        "# Sweep Catalog",
        "",
        "Curated index of existing optimizer sweeps. Lower validation BPB is better.",
        "",
        "Generated files:",
        "- `all_sweeps.csv`: normalized row-level results.",
        "- `best_by_batch_method.csv`: best observed row per `(family, batch, method, alpha)`.",
        "- `summary.json`: counts and coverage.",
        "",
        "Important comparison rule: compare rows within the same `family` first. Historical native single-GPU rows and newer same-driver StreamingMuon DDP rows can differ in absolute BPB.",
        "",
        "## Coverage",
        "",
        f"- Batch sizes present: {', '.join(label_batch(b) for b in batches)}.",
        f"- Alpha values present: {', '.join(alpha_values) if alpha_values else 'none'}.",
        f"- Methods present: {', '.join(f'{k}={v}' for k, v in sorted(by_method.items()))}.",
        "",
        "## Sources",
        "",
    ]
    lines += [f"- `{k}`: {v} rows." for k, v in sorted(by_source.items())]
    lines += [
        "",
        "## Best Rows Snapshot",
        "",
        "| family | batch | method | alpha | lr | seed | BPB |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for r in best[:80]:
        lines.append(
            f"| {r['family']} | {r['batch_label']} | {r['method']} | {r['alpha'] or '-'} | "
            f"{r['lr'] or '-'} | {r['seed'] or '-'} | {float(r['score']):.6f} |"
        )
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    rows = load_catalog_rows()
    rows = sorted(rows, key=lambda r: (int(r["batch"]), r["family"], r["method"], r["alpha"], r["lr"], r["seed"], r["path"]))
    best = best_rows(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "all_sweeps.csv", rows)
    write_csv(OUT / "best_by_batch_method.csv", best)
    summary = {
        "schema_version": 1,
        "row_count": len(rows),
        "best_row_count": len(best),
        "batches": [label_batch(b) for b in sorted({int(r["batch"]) for r in rows})],
        "methods": dict(sorted(Counter(r["method"] for r in rows).items())),
        "families": dict(sorted(Counter(r["family"] for r in rows).items())),
        "alpha_values": sorted({r["alpha"] for r in rows if r["alpha"]}, key=lambda x: float(x)),
        "sources": dict(sorted(Counter(r["catalog_source"] for r in rows).items())),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_readme(rows, best)
    print(f"wrote {OUT.relative_to(ROOT)} with {len(rows)} rows")


if __name__ == "__main__":
    main()
