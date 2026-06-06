#!/usr/bin/env python3
"""Manage the Qwen3 d12 64K Muon coordinate-descent sweep.

This script is intentionally narrow: it follows the handoff ledger under
``results/qwen3_d12_muon64_coordinate_descent`` and uses the existing Merlin
submission wrapper for any new candidates.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "results" / "qwen3_d12_muon64_coordinate_descent"
STATE_PATH = LEDGER / "state.json"
HDFS_BASE = "hdfs://haruna/home/byte_data_seed/hdd_hldy/user/xingyu.dang/sigma-search-runs/search_evals"

MERLIN = os.environ.get("MERLIN_CLI", "/home/tiger/.merlin-cli/bin/merlin-cli")
HDFS = os.environ.get("HDFS_BIN", "/opt/tiger/yarn_deploy/hadoop/bin/hdfs")
HDFS_SOURCE_CONF_DIR = Path(
    os.environ.get("SIGMA_HDFS_SOURCE_CONF_DIR")
    or os.environ.get("HADOOP_CONF_DIR")
    or "/opt/tiger/arnold/hdfs_client/conf/celer_china-north5"
)
HDFS_CONF_DIR = Path(os.environ.get("SIGMA_HDFS_CONF_DIR", "/tmp/sigma_hadoop_conf"))
CONTROL_PLANE = "cn-seed"

COORDS = ["matrix_lr", "muon_momentum", "adam_lr_multiplier", "adam_beta1", "weight_decay"]
FAILED_STATUSES = {
    "CANCELED",
    "CANCELLED",
    "FAILED",
    "FAILED_TO_LAUNCH",
    "RESULT_ERROR",
    "STOPPED",
    "TIMEOUT",
}
SUCCESS_TERMINAL_STATUSES = {"COMPLETED", "DONE", "FINISHED", "SUCCEEDED", "SUCCESS"}
SCORE_PATTERNS = [
    re.compile(r"^SCORE:\s*([0-9.eE+-]+)\s*$", re.MULTILINE),
    re.compile(r"\[done\].*?\bscore=([0-9.eE+-]+)"),
]
CSV_FIELDS = [
    "round",
    "coordinate",
    "candidate",
    "matrix_lr",
    "muon_momentum",
    "adam_lr_multiplier",
    "adam_beta1",
    "weight_decay",
    "stamp",
    "status",
    "score",
    "path",
]

GRID_CONFIG = {
    # Multiplicative log grids around the current center.
    "matrix_lr": {"kind": "positive", "radius": math.log(1.6), "min": 0.001, "max": 0.064},
    "adam_lr_multiplier": {"kind": "positive", "radius": math.log(2.0), "min": 0.125, "max": 4.0},
    "weight_decay": {"kind": "positive", "radius": math.log(1.75), "min": 0.04, "max": 0.50},
    # Log grids in complement space, i.e. q = 1 - beta, to behave sensibly near 1.
    "muon_momentum": {"kind": "beta_complement", "radius": math.log(2.0), "min": 0.75, "max": 0.995},
    "adam_beta1": {"kind": "beta_complement", "radius": math.log(2.5), "min": 0.30, "max": 0.98},
}

SUBMIT_DEFAULTS = {
    "image_url": "hub.byted.org/reckon/data.reckon.mlx.image_11319:50545093b9947634c8247599da45c75e",
    "hdfs_code_tgz": "hdfs://haruna/home/byte_data_seed/hdd_hldy/user/xingyu.dang/sigma-search/sigma-search-clean-qwen3-d12-muon64-cd-20260605.tgz",
    "hdfs_runtime_tgz": "hdfs://haruna/home/byte_data_seed/hdd_hldy/user/xingyu.dang/sigma-search/runtime/nanochat_venv_torch291_cu128.tgz",
    "nanochat_base_dir": "/mnt/hdfs/user/xingyu.dang/nanochat",
    "merlin_output_base": "/mnt/hdfs/user/xingyu.dang/sigma-search-runs",
    "group_ids": "914",
    "cluster_id": "34",
    "queue_name": "a100-sxm-80gb.i110617261685691719770.ai",
    "gpuv": "A100_SXM_80GB",
    "gpu": "8",
    "cpu": "120",
    "memory": "1941504",
    "hdfs_volume_json": '{"path":"hdfs://haruna/home/byte_data_seed/hdd_hldy/user/xingyu.dang/","mnt":"/mnt/hdfs/user/xingyu.dang","roles":["worker"],"access_mode":"RW"}',
}


def run(cmd: list[str], *, timeout: int = 120, check: bool = True, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=check,
        env=env,
    )


def ensure_hdfs_conf_dir() -> str:
    """Use a writable Hadoop conf dir so HDFS commands do not depend on RO paths."""
    HDFS_CONF_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("core-site.xml", "hdfs-site.xml"):
        src = HDFS_SOURCE_CONF_DIR / name
        dst = HDFS_CONF_DIR / name
        if src.exists() and (not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime):
            shutil.copy2(src, dst)
    return str(HDFS_CONF_DIR)


def hdfs_env() -> dict[str, str]:
    env = os.environ.copy()
    conf_dir = ensure_hdfs_conf_dir()
    env["HADOOP_CONF_DIR"] = conf_dir
    core_site = Path(conf_dir) / "core-site.xml"
    hdfs_site = Path(conf_dir) / "hdfs-site.xml"
    if core_site.exists() and hdfs_site.exists():
        env["CPP_HDFS_CONF"] = f"{core_site}:{hdfs_site}"
    return env


def hdfs_cmd(*args: str) -> list[str]:
    return [HDFS, "--config", ensure_hdfs_conf_dir(), "dfs", *args]


def clean_hdfs_output(output: str) -> str:
    # The local wrapper can print harmless conf-generation warnings on this host.
    lines = [
        line
        for line in output.splitlines()
        if not line.startswith("Failed to generate conf:")
    ]
    return "\n".join(lines).strip()


def load_state() -> dict[str, Any]:
    return json.loads(STATE_PATH.read_text())


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=False) + "\n")


def round_csv(round_idx: int) -> Path:
    return LEDGER / f"round_{round_idx:02d}_candidates.csv"


def read_rows(round_idx: int) -> list[dict[str, str]]:
    path = round_csv(round_idx)
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def write_rows(round_idx: int, rows: list[dict[str, Any]]) -> None:
    path = round_csv(round_idx)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def latest_round(state: dict[str, Any]) -> int:
    return max(int(r["round"]) for r in state.get("rounds", []))


def current_recipe(state: dict[str, Any]) -> dict[str, Any]:
    return dict(state.get("current_recipe") or state["base_recipe"])


def fmt_float(value: float) -> str:
    return f"{float(value):.6g}"


def slug_float(value: float) -> str:
    return fmt_float(value).replace("-", "m").replace(".", "p")


def parse_score(value: str) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        score = float(value)
    except ValueError:
        return None
    return score if math.isfinite(score) else None


def case_result_path(row: dict[str, str]) -> str:
    lr_slug = slug_float(float(row["matrix_lr"]))
    return f"{HDFS_BASE}/{row['stamp']}/plain_muon_bsz65536_lr{lr_slug}_s42/result.json"


def hdfs_cat_json(path: str) -> dict[str, Any] | None:
    env = hdfs_env()
    listing = run(hdfs_cmd("-ls", path), check=False, timeout=120, env=env)
    listing_out = clean_hdfs_output(listing.stdout)
    if listing.returncode != 0:
        if "No such file or directory" in listing_out or "File does not exist" in listing_out:
            return None
        raise RuntimeError(f"hdfs ls failed rc={listing.returncode}: {listing_out[:500]}")

    cat = run(hdfs_cmd("-cat", path), check=False, timeout=120, env=env)
    cat_out = clean_hdfs_output(cat.stdout)
    if cat.returncode != 0:
        raise RuntimeError(f"hdfs cat failed rc={cat.returncode}: {cat_out[:500]}")

    json_start = cat_out.find("{")
    if json_start < 0:
        json_start = cat_out.find("[")
    if json_start < 0:
        raise RuntimeError(f"hdfs cat did not return JSON: {cat_out[:500]}")
    data = cat_out[json_start:]
    if not data:
        return None
    return json.loads(data)


def merlin_runs(stamp: str) -> list[dict[str, Any]]:
    proc = run(
        [
            MERLIN,
            "--control-plane",
            CONTROL_PLANE,
            "job",
            "list-run",
            "--json",
            json.dumps({"job_name": stamp, "pageSize": 50, "current": 1}),
        ],
        timeout=120,
    )
    return json.loads(proc.stdout).get("list") or []


def merlin_trial_logs(run_obj: dict[str, Any]) -> list[dict[str, Any]]:
    trial_id = run_obj.get("latest_trial_id") or (run_obj.get("meta") or {}).get("arnold_trial_id")
    if not run_obj.get("id") or not trial_id:
        return []
    proc = run(
        [
            MERLIN,
            "--control-plane",
            CONTROL_PLANE,
            "job",
            "list-trial-logs",
            "--json",
            json.dumps({"job_run_id": run_obj["id"], "trial_id": str(trial_id)}),
        ],
        timeout=120,
    )
    return json.loads(proc.stdout).get("log_list") or []


def score_from_run_logs(run_obj: dict[str, Any]) -> tuple[float, str] | None:
    for log in merlin_trial_logs(run_obj):
        if str(log.get("type", "")).lower() != "stdout" or not log.get("url"):
            continue
        proc = run(["curl", "-L", "-sS", "--max-time", "60", log["url"]], check=False, timeout=90)
        if proc.returncode != 0:
            raise RuntimeError(f"curl stdout log failed rc={proc.returncode}: {proc.stdout[:500]}")
        for pattern in SCORE_PATTERNS:
            matches = pattern.findall(proc.stdout)
            if matches:
                return float(matches[-1]), log["url"]
    return None


def summarize_resource(run_obj: dict[str, Any]) -> tuple[Any, ...]:
    meta = run_obj.get("meta") or {}
    arnold = (((meta.get("job_def_version") or {}).get("resource") or {}).get("arnold_config") or {})
    role = (arnold.get("roles") or [{}])[0]
    vol = (arnold.get("hdfsVolumes") or [{}])[0]
    return (
        arnold.get("clusterId"),
        arnold.get("clusterName"),
        role.get("gpuv"),
        role.get("gpu"),
        role.get("queueName"),
        vol.get("accessMode"),
    )


def collect_round(round_idx: int, *, write: bool) -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows = read_rows(round_idx)
    counts: dict[str, int] = {}
    resources: dict[str, int] = {}
    scored = 0
    errors: list[dict[str, str]] = []
    failed_rows: list[dict[str, str]] = []

    for row in rows:
        status = row.get("status", "")
        run_obj: dict[str, Any] | None = None
        try:
            runs = merlin_runs(row["stamp"])
            if runs:
                # All current submissions have exactly one run per stamp.
                run_obj = runs[0]
                status = run_obj.get("status") or "UNKNOWN"
                key = repr(summarize_resource(run_obj))
                resources[key] = resources.get(key, 0) + 1
            else:
                status = "NO_RUN"
        except Exception as exc:
            status = "QUERY_ERROR"
            errors.append({"stamp": row["stamp"], "error": str(exc)[:500]})

        path = case_result_path(row)
        hdfs_error: str | None = None
        try:
            data = hdfs_cat_json(path)
            if data:
                row["path"] = path
                if data.get("error") is not None:
                    status = "RESULT_ERROR"
                    errors.append({"stamp": row["stamp"], "error": f"result_json:{data.get('error')}"[:500]})
                elif data.get("score") is not None:
                    row["score"] = fmt_float(float(data["score"]))
                    status = "scored"
        except Exception as exc:
            hdfs_error = f"hdfs:{str(exc)[:500]}"

        if parse_score(row.get("score", "")) is None and status in SUCCESS_TERMINAL_STATUSES and run_obj:
            try:
                recovered = score_from_run_logs(run_obj)
                if recovered:
                    score, log_url = recovered
                    row["score"] = fmt_float(score)
                    row["path"] = path
                    status = "scored"
            except Exception as exc:
                errors.append({"stamp": row["stamp"], "error": f"stdout_log:{str(exc)[:500]}"})

        if hdfs_error and parse_score(row.get("score", "")) is None:
            errors.append({"stamp": row["stamp"], "error": hdfs_error})

        if status in SUCCESS_TERMINAL_STATUSES and parse_score(row.get("score", "")) is None:
            status = "RESULT_ERROR"
            errors.append({"stamp": row["stamp"], "error": "terminal_success_without_score"})

        if parse_score(row.get("score", "")) is not None and status not in FAILED_STATUSES:
            status = "scored"

        if status == "scored":
            scored += 1

        row["status"] = status
        counts[status] = counts.get(status, 0) + 1
        if status in FAILED_STATUSES and parse_score(row.get("score", "")) is None:
            failed_rows.append({"stamp": row["stamp"], "status": status})

    if write:
        write_rows(round_idx, rows)
        state = load_state()
        for round_state in state.get("rounds", []):
            if int(round_state["round"]) == round_idx:
                if scored == len(rows):
                    round_state["status"] = "scored"
                elif counts:
                    round_state["status"] = "running"
        if state.get("status", "").startswith(f"round_{round_idx}_"):
            state["status"] = f"round_{round_idx}_running" if scored < len(rows) else f"round_{round_idx}_scored"
        save_state(state)

    summary = {
        "round": round_idx,
        "status_counts": counts,
        "resources": resources,
        "scored": scored,
        "expected": len(rows),
        "errors": errors,
        "failed_rows": failed_rows,
    }
    return rows, summary


def unique_sorted(values: list[float]) -> list[float]:
    out: list[float] = []
    seen: set[str] = set()
    for value in sorted(values):
        key = fmt_float(value)
        if key not in seen:
            out.append(float(key))
            seen.add(key)
    return out


def grid_for_coordinate(coord: str, center: float) -> list[float]:
    cfg = GRID_CONFIG[coord]
    radius = float(cfg["radius"])
    lo = float(cfg["min"])
    hi = float(cfg["max"])
    offsets = [-1.0, -0.5, 0.0, 0.5, 1.0]
    values: list[float] = []
    if cfg["kind"] == "positive":
        values = [center * math.exp(radius * off) for off in offsets]
    elif cfg["kind"] == "beta_complement":
        complement = max(1.0 - center, 1e-4)
        values = [1.0 - complement * math.exp(-radius * off) for off in offsets]
    else:
        raise ValueError(f"unknown grid kind for {coord}: {cfg['kind']}")

    values = [min(max(v, lo), hi) for v in values]
    values = unique_sorted(values)
    if len(values) < 5:
        for off in [-2.0, -1.5, 1.5, 2.0, -2.5, 2.5]:
            if cfg["kind"] == "positive":
                v = center * math.exp(radius * off)
            else:
                complement = max(1.0 - center, 1e-4)
                v = 1.0 - complement * math.exp(-radius * off)
            values = unique_sorted(values + [min(max(v, lo), hi)])
            if len(values) >= 5:
                break
    return values[:5]


def candidate_rows(round_idx: int, center: dict[str, Any], active_coords: list[str], date_tag: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    grids = {coord: grid_for_coordinate(coord, float(center[coord])) for coord in active_coords}
    for coord in active_coords:
        for candidate_idx, value in enumerate(grids[coord]):
            recipe = dict(center)
            recipe[coord] = value
            stamp = f"qwen3_d12_muon64_cd_r{round_idx:02d}_{coord}_{slug_float(value)}_{date_tag}"
            rows.append(
                {
                    "round": str(round_idx),
                    "coordinate": coord,
                    "candidate": str(candidate_idx),
                    "matrix_lr": fmt_float(float(recipe["matrix_lr"])),
                    "muon_momentum": fmt_float(float(recipe["muon_momentum"])),
                    "adam_lr_multiplier": fmt_float(float(recipe["adam_lr_multiplier"])),
                    "adam_beta1": fmt_float(float(recipe["adam_beta1"])),
                    "weight_decay": fmt_float(float(recipe["weight_decay"])),
                    "stamp": stamp,
                    "status": "planned",
                    "score": "",
                    "path": "",
                }
            )
    return rows


def plan_next_round(*, write: bool, min_improvement: float, date_tag: str | None) -> dict[str, Any]:
    state = load_state()
    round_idx = latest_round(state)
    rows = read_rows(round_idx)
    missing = [row["stamp"] for row in rows if parse_score(row.get("score", "")) is None]
    if missing:
        return {"planned": False, "reason": "round_not_fully_scored", "missing": len(missing), "round": round_idx}

    center = current_recipe(state)
    baseline = float(center["score"])
    best_by_coord: dict[str, dict[str, Any]] = {}
    for coord in sorted(set(row["coordinate"] for row in rows)):
        coord_rows = [row for row in rows if row["coordinate"] == coord]
        best = min(coord_rows, key=lambda row: float(row["score"]))
        best_by_coord[coord] = {
            "score": float(best["score"]),
            "improvement": baseline - float(best["score"]),
            "row": best,
        }

    accepted_coord, accepted = max(best_by_coord.items(), key=lambda item: item[1]["improvement"])
    improvement = float(accepted["improvement"])
    for round_state in state.get("rounds", []):
        if int(round_state["round"]) == round_idx:
            round_state["status"] = "completed"
            round_state["best_by_coordinate"] = {
                coord: {
                    "score": info["score"],
                    "improvement": info["improvement"],
                    "stamp": info["row"]["stamp"],
                }
                for coord, info in best_by_coord.items()
            }
            round_state["accepted_coordinate"] = accepted_coord if improvement > min_improvement else None
            round_state["accepted_improvement"] = improvement

    if improvement <= min_improvement:
        state["status"] = "converged"
        state["stop_reason"] = f"best improvement {improvement:.8g} <= min_improvement {min_improvement:.8g}"
        if write:
            save_state(state)
        return {"planned": False, "reason": "converged", "improvement": improvement, "best_coordinate": accepted_coord}

    best_row = accepted["row"]
    new_center = {coord: float(best_row[coord]) for coord in COORDS}
    new_center.update(
        {
            "architecture": center.get("architecture", "qwen3"),
            "depth": int(center.get("depth", 12)),
            "batch": int(center.get("batch", 65536)),
            "chinchilla_mult": float(center.get("chinchilla_mult", 1.0)),
            "optimizer": center.get("optimizer", "plain_muon"),
            "score": float(best_row["score"]),
            "source_stamp": best_row["stamp"],
            "source_path": best_row.get("path", ""),
        }
    )
    state["current_recipe"] = new_center
    state["last_changed_coordinate"] = accepted_coord

    next_round = round_idx + 1
    if next_round >= int(state.get("max_rounds", 6)):
        state["status"] = "max_rounds_reached"
        state["stop_reason"] = f"completed {next_round} rounds"
        if write:
            save_state(state)
        return {"planned": False, "reason": "max_rounds_reached", "accepted": new_center}

    if any(int(r["round"]) == next_round for r in state.get("rounds", [])):
        return {"planned": False, "reason": "next_round_already_exists", "round": next_round}

    active = [coord for coord in COORDS if coord != accepted_coord]
    date_tag = date_tag or dt.datetime.now().strftime("%Y%m%d")
    rows_next = candidate_rows(next_round, new_center, active, date_tag)
    state["rounds"].append(
        {
            "round": next_round,
            "status": "planned",
            "active_coordinates": active,
            "skipped_coordinate": accepted_coord,
            "center_score": new_center["score"],
            "center_recipe": {coord: new_center[coord] for coord in COORDS},
            "grids": {
                coord: [float(row[coord]) for row in rows_next if row["coordinate"] == coord]
                for coord in active
            },
        }
    )
    state["status"] = f"round_{next_round}_planned"

    if write:
        write_rows(next_round, rows_next)
        save_state(state)
    return {
        "planned": True,
        "round": next_round,
        "accepted_coordinate": accepted_coord,
        "accepted_score": new_center["score"],
        "improvement": improvement,
        "active_coordinates": active,
        "candidate_count": len(rows_next),
    }


def submit_command(row: dict[str, str], args: argparse.Namespace) -> list[str]:
    defaults = SUBMIT_DEFAULTS | {
        "image_url": args.image_url or SUBMIT_DEFAULTS["image_url"],
        "hdfs_code_tgz": args.hdfs_code_tgz or SUBMIT_DEFAULTS["hdfs_code_tgz"],
        "hdfs_runtime_tgz": args.hdfs_runtime_tgz or SUBMIT_DEFAULTS["hdfs_runtime_tgz"],
        "group_ids": args.group_ids or SUBMIT_DEFAULTS["group_ids"],
        "cluster_id": str(args.cluster_id or SUBMIT_DEFAULTS["cluster_id"]),
        "queue_name": args.queue_name or SUBMIT_DEFAULTS["queue_name"],
        "gpuv": args.gpuv or SUBMIT_DEFAULTS["gpuv"],
        "cpu": str(args.cpu or SUBMIT_DEFAULTS["cpu"]),
        "memory": str(args.memory or SUBMIT_DEFAULTS["memory"]),
    }
    return [
        sys.executable,
        "scripts/submit_merlin_sweep.py",
        "--submit",
        "--preset",
        "d12_2x",
        "--stamp",
        row["stamp"],
        "--methods",
        "plain_muon",
        "--batches",
        "65536",
        "--lrs",
        row["matrix_lr"],
        "--alphas",
        "1.0",
        "--top-ks",
        "1",
        "--depth",
        "12",
        "--chinchilla-mult",
        "1.0",
        "--architecture",
        "qwen3",
        "--nproc",
        "8",
        "--max-device-batch-size",
        "16",
        "--weight-decay",
        row["weight_decay"],
        "--muon-momentum",
        row["muon_momentum"],
        "--muon-momentum-schedule",
        "static",
        "--adam-lr-multiplier",
        row["adam_lr_multiplier"],
        "--adam-beta1",
        row["adam_beta1"],
        "--batch-beta-align",
        "--batch-beta-align-mode",
        "beta2_only",
        "--structured-config",
        "global",
        "--precondition-frequency",
        "1",
        "--shampoo-beta",
        "0.95",
        "--optimizer-beta1",
        row["muon_momentum"],
        "--optimizer-beta2",
        "0.95",
        "--structured-init-factor",
        "0.1",
        "--save-every",
        "100",
        "--keep-last-checkpoints",
        "2",
        "--streaming-num-iters",
        "2",
        "--fallback-ortho-tol",
        "0.01",
        "--metrics-every",
        "0",
        "--metrics-hessian-every",
        "0",
        "--image-url",
        defaults["image_url"],
        "--hdfs-code-tgz",
        defaults["hdfs_code_tgz"],
        "--hdfs-runtime-tgz",
        defaults["hdfs_runtime_tgz"],
        "--nanochat-base-dir",
        defaults["nanochat_base_dir"],
        "--merlin-output-base",
        defaults["merlin_output_base"],
        "--group-ids",
        defaults["group_ids"],
        "--cluster-id",
        defaults["cluster_id"],
        "--queue-name",
        defaults["queue_name"],
        "--gpuv",
        defaults["gpuv"],
        "--gpu",
        defaults["gpu"],
        "--cpu",
        defaults["cpu"],
        "--memory",
        defaults["memory"],
        "--hdfs-volume-json",
        defaults["hdfs_volume_json"],
    ]


def submit_round(round_idx: int, args: argparse.Namespace) -> dict[str, Any]:
    rows = read_rows(round_idx)
    submitted = 0
    skipped = 0
    env = os.environ.copy()
    env["PATH"] = str(Path(MERLIN).parent) + os.pathsep + env.get("PATH", "")
    for row in rows:
        existing = merlin_runs(row["stamp"])
        if existing and not args.rerun_existing:
            row["status"] = existing[0].get("status") or "submitted"
            skipped += 1
            continue
        cmd = submit_command(row, args)
        print("+ " + " ".join(cmd), flush=True)
        if args.dry_run:
            skipped += 1
            continue
        run(cmd, timeout=300, env=env)
        row["status"] = "submitted"
        submitted += 1
        write_rows(round_idx, rows)
    if not args.dry_run:
        write_rows(round_idx, rows)
    return {"round": round_idx, "submitted": submitted, "skipped": skipped, "dry_run": args.dry_run}


def cmd_status(args: argparse.Namespace) -> None:
    round_idx = args.round if args.round is not None else latest_round(load_state())
    rows, summary = collect_round(round_idx, write=args.write)
    print(json.dumps(summary, indent=2, sort_keys=True))
    for row in rows:
        print(f"{row['coordinate']}\t{row['status']}\t{row['score']}\t{row['stamp']}")


def cmd_plan_next(args: argparse.Namespace) -> None:
    result = plan_next_round(write=args.write, min_improvement=args.min_improvement, date_tag=args.date_tag)
    print(json.dumps(result, indent=2, sort_keys=True))


def cmd_advance(args: argparse.Namespace) -> None:
    state = load_state()
    round_idx = latest_round(state)
    rows, summary = collect_round(round_idx, write=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["failed_rows"]:
        state = load_state()
        state["status"] = f"round_{round_idx}_failed"
        state["failed_rows"] = summary["failed_rows"]
        save_state(state)
        return
    if summary["scored"] < summary["expected"]:
        needs_submit = any(row.get("status") in {"planned", "NO_RUN"} for row in rows)
        if args.submit and needs_submit:
            result = submit_round(round_idx, args)
            print(json.dumps(result, indent=2, sort_keys=True))
        return
    planned = plan_next_round(write=True, min_improvement=args.min_improvement, date_tag=args.date_tag)
    print(json.dumps(planned, indent=2, sort_keys=True))
    if args.submit and planned.get("planned"):
        submit_args = argparse.Namespace(**vars(args))
        result = submit_round(int(planned["round"]), submit_args)
        print(json.dumps(result, indent=2, sort_keys=True))


def cmd_submit_round(args: argparse.Namespace) -> None:
    print(json.dumps(submit_round(args.round, args), indent=2, sort_keys=True))


def cmd_watch(args: argparse.Namespace) -> None:
    while True:
        cmd_advance(args)
        state = load_state()
        status = state.get("status")
        if status in {"converged", "max_rounds_reached"} or str(status).endswith("_failed"):
            return
        time.sleep(args.interval)


def add_submit_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun-existing", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--image-url", default="")
    parser.add_argument("--hdfs-code-tgz", default="")
    parser.add_argument("--hdfs-runtime-tgz", default="")
    parser.add_argument("--group-ids", default="")
    parser.add_argument("--cluster-id", default="")
    parser.add_argument("--queue-name", default="")
    parser.add_argument("--gpuv", default="")
    parser.add_argument("--cpu", default="")
    parser.add_argument("--memory", default="")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status", help="Query Merlin/HDFS and print round status.")
    p_status.add_argument("--round", type=int, default=None)
    p_status.add_argument("--write", action="store_true", help="Update round CSV/status fields from external state.")
    p_status.set_defaults(func=cmd_status)

    p_plan = sub.add_parser("plan-next", help="Choose the next coordinate move and create the next round CSV.")
    p_plan.add_argument("--write", action="store_true")
    p_plan.add_argument("--min-improvement", type=float, default=0.0)
    p_plan.add_argument("--date-tag", default=None)
    p_plan.set_defaults(func=cmd_plan_next)

    p_advance = sub.add_parser("advance", help="Collect current round, plan next if complete, optionally submit it.")
    p_advance.add_argument("--min-improvement", type=float, default=0.0)
    p_advance.add_argument("--date-tag", default=None)
    add_submit_options(p_advance)
    p_advance.set_defaults(func=cmd_advance)

    p_submit = sub.add_parser("submit-round", help="Submit all planned candidates for a round.")
    p_submit.add_argument("--round", type=int, required=True)
    add_submit_options(p_submit)
    p_submit.set_defaults(func=cmd_submit_round)

    p_watch = sub.add_parser("watch", help="Loop: collect, plan, and optionally submit until convergence/max rounds.")
    p_watch.add_argument("--interval", type=int, default=600)
    p_watch.add_argument("--min-improvement", type=float, default=0.0)
    p_watch.add_argument("--date-tag", default=None)
    add_submit_options(p_watch)
    p_watch.set_defaults(func=cmd_watch)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
