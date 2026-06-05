#!/usr/bin/env python3
"""Submit sigma-search sweep cases as parallel Merlin jobs.

The repo's sweep engine already supports ``--case-index``. This script turns a
single sweep grid into many independent Merlin ``create-run`` payloads, one per
case, all writing to the same persistent OUT_ROOT/LOG_ROOT.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

PRESETS: dict[str, dict[str, Any]] = {
    "d12_2x": {
        "depth": 12,
        "chinchilla_mult": 2.0,
        "batches": "524288 2097152 8388608",
        "alphas": "1.0 0.5",
        "lrs": "0.005 0.0075 0.01 0.015 0.02 0.03 0.04",
        "script": "scripts/run_d12_sweep.sh",
    },
    "d16_2x": {
        "depth": 16,
        "chinchilla_mult": 2.0,
        "batches": "524288 2097152 8388608",
        "alphas": "1.0 0.5",
        "lrs": "0.005 0.0075 0.01 0.015 0.02 0.03 0.04",
        "script": "scripts/run_d12_sweep.sh",
    },
    "d8_quality": {
        "depth": 8,
        "chinchilla_mult": 2.0,
        "batches": "262144 1048576 4194304",
        "alphas": "1.0 0.5",
        "lrs": "0.005 0.0075 0.01 0.015 0.02 0.03 0.04",
        "script": "scripts/run_d12_sweep.sh",
    },
    "d8_metrics": {
        "depth": 8,
        "tokens": 402_653_184,
        "batches": "262144 1048576 4194304",
        "alphas": "0.5 1.0",
        "lrs": "0.02",
        "script": "scripts/run_d8_metrics_grid.sh",
        "metrics_every": 1,
        "metrics_top_k": 4,
        "metrics_max_modules": 8,
        "metrics_hessian_every": 50,
        "metrics_hessian_top_k": 1,
        "metrics_hessian_iters": 2,
        "metrics_hessian_max_modules": 8,
    },
    "d12_adamw": {
        "depth": 12,
        "chinchilla_mult": 2.0,
        "methods": "adamw",
        "batches": "524288 2097152 8388608",
        "alphas": "1.0",
        "lrs": "0.0005 0.001 0.002 0.003 0.004 0.005 0.0075",
        "script": "scripts/run_d12_sweep.sh",
        "weight_decay": 0.1,
    },
    "d16_adamw": {
        "depth": 16,
        "chinchilla_mult": 2.0,
        "methods": "adamw",
        "batches": "524288 2097152 8388608",
        "alphas": "1.0",
        "lrs": "0.0005 0.001 0.002 0.003 0.004 0.005 0.0075",
        "script": "scripts/run_d12_sweep.sh",
        "weight_decay": 0.1,
    },
    "d12_soap": {
        "depth": 12,
        "chinchilla_mult": 2.0,
        "methods": "soap",
        "batches": "524288 2097152 8388608",
        "alphas": "1.0",
        "lrs": "0.001 0.002 0.003 0.004 0.006 0.008 0.012",
        "script": "scripts/run_d12_sweep.sh",
        "weight_decay": 0.1,
    },
    "d16_soap": {
        "depth": 16,
        "chinchilla_mult": 2.0,
        "methods": "soap",
        "batches": "524288 2097152 8388608",
        "alphas": "1.0",
        "lrs": "0.001 0.002 0.003 0.004 0.006 0.008 0.012",
        "script": "scripts/run_d12_sweep.sh",
        "weight_decay": 0.1,
    },
    "d12_kl_soap": {
        "depth": 12,
        "chinchilla_mult": 2.0,
        "methods": "kl_soap",
        "batches": "524288 2097152 8388608",
        "alphas": "1.0",
        "lrs": "0.001 0.002 0.003 0.004 0.006 0.008 0.012",
        "script": "scripts/run_d12_sweep.sh",
        "weight_decay": 0.1,
    },
    "d16_kl_soap": {
        "depth": 16,
        "chinchilla_mult": 2.0,
        "methods": "kl_soap",
        "batches": "524288 2097152 8388608",
        "alphas": "1.0",
        "lrs": "0.001 0.002 0.003 0.004 0.006 0.008 0.012",
        "script": "scripts/run_d12_sweep.sh",
        "weight_decay": 0.1,
    },
    "d12_kl_shampoo": {
        "depth": 12,
        "chinchilla_mult": 2.0,
        "methods": "kl_shampoo",
        "batches": "524288 2097152 8388608",
        "alphas": "1.0",
        "lrs": "0.002 0.004 0.006 0.008 0.012 0.016",
        "script": "scripts/run_d12_sweep.sh",
        "weight_decay": 0.1,
    },
    "d16_kl_shampoo": {
        "depth": 16,
        "chinchilla_mult": 2.0,
        "methods": "kl_shampoo",
        "batches": "524288 2097152 8388608",
        "alphas": "1.0",
        "lrs": "0.002 0.004 0.006 0.008 0.012 0.016",
        "script": "scripts/run_d12_sweep.sh",
        "weight_decay": 0.1,
    },
}


def _json_arg(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _split_ints(value: str) -> list[int]:
    return [int(x) for x in value.replace(",", " ").split() if x]


def _preset_value(args: argparse.Namespace, key: str, default: Any = None) -> Any:
    explicit = getattr(args, key)
    parser_default = getattr(args, "_parser_defaults", {}).get(key)
    if explicit is not None and explicit != parser_default:
        return explicit
    return PRESETS[args.preset].get(key, default)


def build_count_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "run_optimizer_sweep.py",
        "--out-root",
        str(args.out_root),
        "--log-root",
        str(args.log_root),
        "--nanochat-dir",
        "nanochat",
        "--methods",
        args.methods,
        "--batches",
        args.batches,
        "--alphas",
        args.alphas,
        "--top-ks",
        args.top_ks,
        "--lrs",
        args.lrs,
        "--seeds",
        args.seeds,
        "--depth",
        str(args.depth),
        "--architecture",
        args.architecture,
        "--nproc-per-node",
        str(args.nproc),
        "--max-device-batch-size",
        str(args.max_device_batch_size),
        "--weight-decay",
        str(args.weight_decay),
        "--streaming-num-iters",
        str(args.streaming_num_iters),
        "--fallback-ortho-tol",
        str(args.fallback_ortho_tol),
        "--metrics-every",
        str(args.metrics_every),
        "--metrics-top-k",
        str(args.metrics_top_k),
        "--metrics-max-modules",
        str(args.metrics_max_modules),
        "--metrics-hessian-every",
        str(args.metrics_hessian_every),
        "--metrics-hessian-top-k",
        str(args.metrics_hessian_top_k),
        "--metrics-hessian-iters",
        str(args.metrics_hessian_iters),
        "--metrics-hessian-max-modules",
        str(args.metrics_hessian_max_modules),
        "--print-case-count",
    ]
    if args.tokens is None:
        cmd += ["--chinchilla-mult", str(args.chinchilla_mult)]
    else:
        cmd += ["--tokens", str(args.tokens)]
    return cmd


def get_case_count(args: argparse.Namespace) -> int:
    output = subprocess.check_output(build_count_command(args), cwd=ROOT, text=True)
    return int(output.strip().splitlines()[-1])


def select_case_indices(args: argparse.Namespace, case_count: int) -> list[int]:
    if args.case_indices:
        indices = _split_ints(args.case_indices)
    else:
        start = args.case_start
        end = case_count if args.case_end is None else args.case_end
        indices = list(range(start, end))
    if args.max_cases is not None:
        indices = indices[: args.max_cases]
    bad = [idx for idx in indices if idx < 0 or idx >= case_count]
    if bad:
        raise ValueError(f"case indices out of range 0..{case_count - 1}: {bad}")
    return indices


def shard_case_indices(indices: list[int], shard_count: int) -> list[list[int]]:
    if shard_count <= 1:
        return [indices]
    shards = [[] for _ in range(shard_count)]
    for offset, idx in enumerate(indices):
        shards[offset % shard_count].append(idx)
    return [shard for shard in shards if shard]


def build_resource_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.resource_config_file:
        return json.loads(Path(args.resource_config_file).read_text())

    arnold: dict[str, Any] = {"roles": []}
    if args.group_ids:
        arnold["group_ids"] = _split_ints(args.group_ids)
    if args.group_names:
        arnold["group_names"] = [x for x in args.group_names.replace(",", " ").split() if x]
    if args.cluster_id is not None:
        arnold["cluster_id"] = args.cluster_id
    if args.cluster_name:
        arnold["cluster_name"] = args.cluster_name

    role = {
        "name": args.role_name,
        "num": args.role_num,
        "gpu": args.gpu,
        "gpuv": args.gpuv.replace("-", "_"),
        "queue_name": args.queue_name,
        "cpu": args.cpu,
        "memory": args.memory,
        "ports": args.ports,
    }
    arnold["roles"].append(role)
    if args.hdfs_volume_json:
        arnold["hdfs_volumes"] = args.hdfs_volume_json
    if args.use_robust_training:
        arnold["use_robust_training"] = True
    return {"backend": "ARNOLD", "arnold_config": arnold}


def bootstrap_script(repo_mnt: str) -> str:
    return f"""set -euo pipefail
export SIGMA_SEARCH_REPO="${{SIGMA_SEARCH_REPO:-{repo_mnt}}}"
if [[ -n "${{HDFS_CODE_TGZ:-}}" ]]; then
  rm -rf "$SIGMA_SEARCH_REPO" /tmp/sigma_search_code
  mkdir -p "$SIGMA_SEARCH_REPO" /tmp/sigma_search_code
  hdfs dfs -get "$HDFS_CODE_TGZ" /tmp/sigma_search_code.tgz
  tar -xzf /tmp/sigma_search_code.tgz -C /tmp/sigma_search_code
  if [[ -f /tmp/sigma_search_code/scripts/merlin_entrypoint.sh ]]; then
    cp -a /tmp/sigma_search_code/. "$SIGMA_SEARCH_REPO"/
  else
    shopt -s nullglob
    dirs=(/tmp/sigma_search_code/*)
    if [[ "${{#dirs[@]}}" -ne 1 || ! -d "${{dirs[0]}}" ]]; then
      echo "Could not infer repo root from HDFS_CODE_TGZ contents" >&2
      exit 2
    fi
    cp -a "${{dirs[0]}}"/. "$SIGMA_SEARCH_REPO"/
  fi
fi
cd "$SIGMA_SEARCH_REPO"
bash scripts/merlin_entrypoint.sh
"""


def common_env(
    args: argparse.Namespace,
    case_index: int | None = None,
    case_indices: list[int] | None = None,
) -> dict[str, str]:
    env = {
        "STAMP": args.stamp,
        "MERLIN_SWEEP_SCRIPT": args.script,
        "MERLIN_OUTPUT_BASE": args.merlin_output_base,
        "NANOCHAT_BASE_DIR": args.nanochat_base_dir,
        "NPROC": str(args.nproc),
        "MAX_DEVICE_BATCH_SIZE": str(args.max_device_batch_size),
        "DEPTH": str(args.depth),
        "METHODS": args.methods,
        "BATCHES": args.batches,
        "ALPHAS": args.alphas,
        "TOP_KS": args.top_ks,
        "LRS": args.lrs,
        "SEEDS": args.seeds,
        "ARCHITECTURE": args.architecture,
        "WEIGHT_DECAY": str(args.weight_decay),
        "MUON_MOMENTUM": str(args.muon_momentum),
        "MUON_MOMENTUM_SCHEDULE": args.muon_momentum_schedule,
        "BATCH_BETA_ALIGN": "1" if args.batch_beta_align else "0",
        "BATCH_BETA_ALIGN_MODE": args.batch_beta_align_mode,
        "STRUCTURED_CONFIG": args.structured_config,
        "PRECONDITION_FREQUENCY": str(args.precondition_frequency),
        "SHAMPOO_BETA": str(args.shampoo_beta),
        "OPTIMIZER_BETA1": str(args.optimizer_beta1),
        "OPTIMIZER_BETA2": str(args.optimizer_beta2),
        "STRUCTURED_INIT_FACTOR": str(args.structured_init_factor),
        "SAVE_EVERY": str(args.save_every),
        "KEEP_LAST_CHECKPOINTS": str(args.keep_last_checkpoints),
        "RESUME": "1" if args.resume else "0",
        "STREAMING_NUM_ITERS": str(args.streaming_num_iters),
        "FALLBACK_ORTHO_TOL": str(args.fallback_ortho_tol),
        "METRICS_EVERY": str(args.metrics_every),
        "METRICS_TOP_K": str(args.metrics_top_k),
        "METRICS_MAX_MODULES": str(args.metrics_max_modules),
        "METRICS_HESSIAN_EVERY": str(args.metrics_hessian_every),
        "METRICS_HESSIAN_TOP_K": str(args.metrics_hessian_top_k),
        "METRICS_HESSIAN_ITERS": str(args.metrics_hessian_iters),
        "METRICS_HESSIAN_MAX_MODULES": str(args.metrics_hessian_max_modules),
        "ADAPTIVE_LR": "0",
    }
    if args.merlin_parallel_cases > 1:
        env["MERLIN_PARALLEL_CASES"] = str(args.merlin_parallel_cases)
        env["MERLIN_GPUS_PER_CASE"] = str(args.merlin_gpus_per_case or args.nproc)
    if args.tokens is None:
        env["CHINCHILLA_MULT"] = str(args.chinchilla_mult)
    else:
        env["TOKENS"] = str(args.tokens)
    if args.hdfs_code_tgz:
        env["HDFS_CODE_TGZ"] = args.hdfs_code_tgz
    if args.hdfs_runtime_tgz:
        env["HDFS_RUNTIME_TGZ"] = args.hdfs_runtime_tgz
    if args.out_root:
        env["OUT_ROOT"] = str(args.out_root)
    if args.log_root:
        env["LOG_ROOT"] = str(args.log_root)
    if case_index is not None:
        env["CASE_INDEX"] = str(case_index)
        env["MERLIN_CASE_INDEX"] = str(case_index)
    if case_indices is not None:
        env["MERLIN_CASE_INDICES"] = " ".join(str(idx) for idx in case_indices)
    return {k: v for k, v in env.items() if v is not None and v != ""}


def build_payload(
    args: argparse.Namespace,
    *,
    case_index: int | None,
    case_indices: list[int] | None = None,
    job_kind: str,
) -> dict[str, Any]:
    caption = f"{args.caption_prefix}-{args.stamp}-{job_kind}"
    if case_index is not None:
        caption = f"{caption}-{case_index:03d}"
    elif case_indices is not None:
        caption = f"{caption}-{case_indices[0]:03d}-{case_indices[-1]:03d}"
    payload: dict[str, Any] = {
        "launch_mode": "from_scratch",
        "caption": caption[:90],
        "entrypoint_full_script": bootstrap_script(args.repo_mnt),
        "env": common_env(args, case_index, case_indices),
        "resource_config": build_resource_config(args),
        "tags": ["sigma-search", "optimizer-sweep", args.preset],
    }
    if args.namespace:
        payload["namespace"] = args.namespace
    if args.image_url:
        payload["image_url"] = args.image_url
    if args.image_vid:
        payload["image_vid"] = args.image_vid
    if args.repo_name:
        git_repo: dict[str, Any] = {"repo_name": args.repo_name, "mnt": args.repo_mnt}
        if args.commit_sha:
            git_repo["commit_sha"] = args.commit_sha
        elif args.tag:
            git_repo["tag"] = args.tag
        else:
            git_repo["branch_name"] = args.branch_name
            git_repo["use_latest_commit"] = args.use_latest_commit
        payload["git_repo"] = git_repo
    if args.notify_disable_all:
        payload["notify_settings"] = {"disable_all": True}
    return payload


def write_payload(payload_dir: Path, payload: dict[str, Any], name: str) -> Path:
    payload_dir.mkdir(parents=True, exist_ok=True)
    path = payload_dir / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return path


def submit_payload(args: argparse.Namespace, path: Path) -> None:
    cmd = [
        "merlin-cli",
        "--control-plane",
        args.control_plane,
        "job",
        "create-run",
        "--from-file",
        str(path),
    ]
    print("+ " + " ".join(cmd), flush=True)
    if args.submit:
        subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="d12_2x")
    parser.add_argument("--stamp", default=f"merlin_sweep_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--caption-prefix", default="sigma-sweep")
    parser.add_argument("--control-plane", default=os.environ.get("MERLIN_CONTROL_PLANE", "cn-seed"))
    parser.add_argument("--namespace", default=os.environ.get("MERLIN_NAMESPACE", ""))
    parser.add_argument("--submit", action="store_true", help="Actually call merlin-cli. Default only writes payloads.")
    parser.add_argument("--payload-dir", type=Path, default=None)

    parser.add_argument("--image-url", default=os.environ.get("MERLIN_IMAGE_URL", ""))
    parser.add_argument("--image-vid", default=os.environ.get("MERLIN_IMAGE_VID", ""))
    parser.add_argument("--repo-name", default=os.environ.get("MERLIN_REPO_NAME", ""))
    parser.add_argument("--branch-name", default=os.environ.get("MERLIN_BRANCH_NAME", "main"))
    parser.add_argument("--commit-sha", default=os.environ.get("MERLIN_COMMIT_SHA", ""))
    parser.add_argument("--tag", default=os.environ.get("MERLIN_GIT_TAG", ""))
    parser.add_argument("--use-latest-commit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--repo-mnt", default=os.environ.get("MERLIN_REPO_MNT", "/opt/tiger/sigma-search-clean"))
    parser.add_argument("--hdfs-code-tgz", default=os.environ.get("HDFS_CODE_TGZ", ""))
    parser.add_argument("--hdfs-runtime-tgz", default=os.environ.get("HDFS_RUNTIME_TGZ", ""))

    parser.add_argument("--resource-config-file", default="")
    parser.add_argument("--group-ids", default=os.environ.get("MERLIN_GROUP_IDS", ""))
    parser.add_argument("--group-names", default=os.environ.get("MERLIN_GROUP_NAMES", ""))
    parser.add_argument("--cluster-id", type=int, default=None)
    parser.add_argument("--cluster-name", default=os.environ.get("MERLIN_CLUSTER_NAME", ""))
    parser.add_argument("--queue-name", default=os.environ.get("MERLIN_QUEUE_NAME", ""))
    parser.add_argument("--gpuv", default=os.environ.get("MERLIN_GPUV", "A100_SXM_80GB"))
    parser.add_argument("--gpu", type=int, default=int(os.environ.get("MERLIN_GPU", "8")))
    parser.add_argument("--cpu", type=int, default=int(os.environ.get("MERLIN_CPU", "120")))
    parser.add_argument("--memory", type=int, default=int(os.environ.get("MERLIN_MEMORY", "1941504")))
    parser.add_argument("--ports", type=int, default=int(os.environ.get("MERLIN_PORTS", "0")))
    parser.add_argument("--role-name", default=os.environ.get("MERLIN_ROLE_NAME", "worker"))
    parser.add_argument("--role-num", type=int, default=1)
    parser.add_argument("--hdfs-volume-json", type=_json_arg, action="append", default=[])
    parser.add_argument("--use-robust-training", action="store_true")
    parser.add_argument("--notify-disable-all", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--nanochat-base-dir", default=os.environ.get("NANOCHAT_BASE_DIR", ""))
    parser.add_argument("--merlin-output-base", default=os.environ.get("MERLIN_OUTPUT_BASE", ""))
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--log-root", type=Path, default=None)

    parser.add_argument("--job-kind", choices=["cases", "summary", "adaptive"], default="cases")
    parser.add_argument("--case-start", type=int, default=0)
    parser.add_argument("--case-end", type=int, default=None)
    parser.add_argument("--case-indices", default="")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument(
        "--shard-count",
        type=int,
        default=0,
        help=(
            "Submit N Merlin jobs and run multiple case indices sequentially inside each job. "
            "0 preserves one-job-per-case behavior."
        ),
    )

    parser.add_argument("--script", default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--chinchilla-mult", type=float, default=None)
    parser.add_argument("--tokens", type=int, default=None)
    parser.add_argument("--methods", default="top_aware_muon")
    parser.add_argument("--batches", default=None)
    parser.add_argument("--alphas", default=None)
    parser.add_argument("--top-ks", default="1")
    parser.add_argument("--lrs", default=None)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--architecture", default="gpt2")
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--max-device-batch-size", type=int, default=16)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-momentum-schedule", choices=["nanochat", "static"], default="nanochat")
    parser.add_argument("--batch-beta-align", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--batch-beta-align-mode",
        choices=["all", "beta2_only", "none"],
        default="all",
    )
    parser.add_argument("--structured-config", choices=["auto", "global"], default="auto")
    parser.add_argument("--precondition-frequency", type=int, default=5)
    parser.add_argument("--shampoo-beta", type=float, default=0.95)
    parser.add_argument("--optimizer-beta1", type=float, default=0.9)
    parser.add_argument("--optimizer-beta2", type=float, default=0.95)
    parser.add_argument("--structured-init-factor", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--keep-last-checkpoints", type=int, default=2)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--streaming-num-iters", type=int, default=2)
    parser.add_argument("--fallback-ortho-tol", type=float, default=0.01)
    parser.add_argument("--metrics-every", type=int, default=None)
    parser.add_argument("--metrics-top-k", type=int, default=None)
    parser.add_argument("--metrics-max-modules", type=int, default=None)
    parser.add_argument("--metrics-hessian-every", type=int, default=None)
    parser.add_argument("--metrics-hessian-top-k", type=int, default=None)
    parser.add_argument("--metrics-hessian-iters", type=int, default=None)
    parser.add_argument("--metrics-hessian-max-modules", type=int, default=None)
    parser.add_argument(
        "--merlin-parallel-cases",
        type=int,
        default=int(os.environ.get("MERLIN_PARALLEL_CASES", "1")),
        help="Run multiple single-process case indices concurrently inside one Merlin node.",
    )
    parser.add_argument(
        "--merlin-gpus-per-case",
        type=int,
        default=int(os.environ.get("MERLIN_GPUS_PER_CASE", "0")),
        help="CUDA devices assigned to each concurrent case; defaults to --nproc.",
    )

    parser_defaults = {
        action.dest: action.default
        for action in parser._actions
        if action.dest != "help"
    }
    args = parser.parse_args()
    args._parser_defaults = parser_defaults

    args.script = _preset_value(args, "script", "scripts/run_d12_sweep.sh")
    args.depth = _preset_value(args, "depth", 12)
    args.chinchilla_mult = _preset_value(args, "chinchilla_mult", 2.0)
    args.tokens = _preset_value(args, "tokens", None)
    args.methods = _preset_value(args, "methods", "top_aware_muon")
    args.batches = _preset_value(args, "batches")
    args.alphas = _preset_value(args, "alphas")
    args.lrs = _preset_value(args, "lrs")
    args.weight_decay = _preset_value(args, "weight_decay", 0.28)
    args.metrics_every = _preset_value(args, "metrics_every", 0)
    args.metrics_top_k = _preset_value(args, "metrics_top_k", 4)
    args.metrics_max_modules = _preset_value(args, "metrics_max_modules", 0)
    args.metrics_hessian_every = _preset_value(args, "metrics_hessian_every", 0)
    args.metrics_hessian_top_k = _preset_value(args, "metrics_hessian_top_k", 1)
    args.metrics_hessian_iters = _preset_value(args, "metrics_hessian_iters", 6)
    args.metrics_hessian_max_modules = _preset_value(args, "metrics_hessian_max_modules", 0)

    if not args.image_url and not args.image_vid:
        raise SystemExit("Provide --image-url or --image-vid (or MERLIN_IMAGE_URL/MERLIN_IMAGE_VID).")
    if not args.nanochat_base_dir:
        raise SystemExit("Provide --nanochat-base-dir or NANOCHAT_BASE_DIR.")
    if not args.merlin_output_base and (args.out_root is None or args.log_root is None):
        raise SystemExit("Provide --merlin-output-base, or both --out-root and --log-root.")
    if args.merlin_output_base:
        args.out_root = args.out_root or Path(args.merlin_output_base) / "search_evals" / args.stamp
        args.log_root = args.log_root or Path(args.merlin_output_base) / "logs" / args.stamp
    if not args.resource_config_file:
        missing = [
            name for name, value in {
                "--queue-name": args.queue_name,
                "--group-ids/--group-names": args.group_ids or args.group_names,
                "--cluster-id/--cluster-name": args.cluster_id if args.cluster_id is not None else args.cluster_name,
            }.items()
            if not value
        ]
        if missing:
            raise SystemExit(f"Missing resource arguments: {', '.join(missing)}")
    return args


def main() -> None:
    args = parse_args()
    payload_dir = args.payload_dir or Path(tempfile.gettempdir()) / f"sigma_merlin_payloads_{args.stamp}"

    if args.job_kind == "cases":
        case_count = get_case_count(args)
        indices = select_case_indices(args, case_count)
        shard_count = args.shard_count or len(indices)
        shards = shard_case_indices(indices, shard_count)
        print(
            f"case_count={case_count} selected={len(indices)} "
            f"shard_jobs={len(shards)} payload_dir={payload_dir}"
        )
        for shard_idx, shard in enumerate(shards):
            if len(shard) == 1 and not args.shard_count:
                payload = build_payload(args, case_index=shard[0], job_kind="case")
                path = write_payload(payload_dir, payload, f"case_{shard[0]:04d}")
            else:
                payload = build_payload(args, case_index=None, case_indices=shard, job_kind=f"shard-{shard_idx:03d}")
                path = write_payload(payload_dir, payload, f"shard_{shard_idx:04d}")
            submit_payload(args, path)
    else:
        payload = build_payload(args, case_index=None, job_kind=args.job_kind)
        if args.job_kind == "summary":
            payload["env"]["SUMMARY_ONLY"] = "1"
            payload["env"]["MERLIN_SKIP_DATA_CHECK"] = "1"
        elif args.job_kind == "adaptive":
            payload["env"]["ADAPTIVE_LR"] = "1"
        path = write_payload(payload_dir, payload, args.job_kind)
        print(f"payload_dir={payload_dir}")
        submit_payload(args, path)

    if not args.submit:
        print("Dry run only. Re-run with --submit to create Merlin jobs.")


if __name__ == "__main__":
    main()
