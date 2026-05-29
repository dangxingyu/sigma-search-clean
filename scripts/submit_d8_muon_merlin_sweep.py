#!/usr/bin/env python3
"""Package and submit the standard d8 plain-Muon sweep to Merlin.

This wrapper captures the current handoff recipe:

- d8, 40TPP (`CHINCHILLA_MULT=2`)
- method: `plain_muon` only
- split 4K and 64K-1M into separate output roots to avoid manifest conflicts
- A100 seed_eval_x by default
- HDFS code tarball plus the known nanochat runtime tarball
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_PARENT = ROOT.parent

A100_QUEUE = "a100-sxm-80gb.i110617261685691719770.ai"
HDFS_BASE = "hdfs://haruna/home/byte_data_seed/hdd_hldy/user/xingyu.dang"
DEFAULT_RUNTIME = f"{HDFS_BASE}/sigma-search/runtime/nanochat_venv_torch291_cu128.tgz"
DEFAULT_OUTPUT_BASE = "/mnt/hdfs/user/xingyu.dang/sigma-search-runs"
DEFAULT_IMAGE = "hub.byted.org/reckon/data.reckon.mlx.image_11319:50545093b9947634c8247599da45c75e"
DEFAULT_HDFS_VOLUME = (
    '{"path":"hdfs://haruna/home/byte_data_seed/hdd_hldy/user/xingyu.dang/",'
    '"mnt":"/mnt/hdfs/user/xingyu.dang","access_mode":"RW","roles":["worker"]}'
)


def run(cmd: list[str], *, dry_run: bool = False) -> None:
    print("+ " + " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=ROOT, check=True)


def package_code(args: argparse.Namespace) -> str:
    hdfs_tgz = f"{args.hdfs_code_dir.rstrip('/')}/sigma-search-clean-{args.stamp}-code.tgz"
    local_tgz = Path("/tmp") / f"sigma-search-clean-{args.stamp}-code.tgz"
    tar_cmd = [
        "tar",
        "--exclude=.git",
        "--exclude=nanochat/.venv",
        "--exclude=results",
        "--exclude=figures",
        "--exclude=logs",
        "--exclude=search_evals",
        "-czf",
        str(local_tgz),
        "-C",
        str(WORKSPACE_PARENT),
        ROOT.name,
    ]
    run(tar_cmd, dry_run=args.dry_run)
    run(["hdfs", "dfs", "-mkdir", "-p", args.hdfs_code_dir], dry_run=args.dry_run)
    run(["hdfs", "dfs", "-put", "-f", str(local_tgz), hdfs_tgz], dry_run=args.dry_run)
    return hdfs_tgz


def submit_group(
    args: argparse.Namespace,
    *,
    hdfs_code_tgz: str,
    group: str,
    batches: str,
    lrs: str,
    nproc: int,
    max_device_batch_size: int,
    shard_count: int,
    merlin_parallel_cases: int = 1,
    merlin_gpus_per_case: int = 0,
) -> None:
    stamp = f"{args.stamp}_{group}"
    cmd = [
        "python",
        "scripts/submit_merlin_sweep.py",
        "--preset",
        "d8_quality",
        "--stamp",
        stamp,
        "--methods",
        "plain_muon",
        "--batches",
        batches,
        "--lrs",
        lrs,
        "--alphas",
        "1.0",
        "--depth",
        "8",
        "--chinchilla-mult",
        "2",
        "--nproc",
        str(nproc),
        "--max-device-batch-size",
        str(max_device_batch_size),
        "--weight-decay",
        str(args.weight_decay),
        "--shard-count",
        str(shard_count),
        "--image-url",
        args.image_url,
        "--hdfs-code-tgz",
        hdfs_code_tgz,
        "--hdfs-runtime-tgz",
        args.hdfs_runtime_tgz,
        "--nanochat-base-dir",
        args.nanochat_base_dir,
        "--merlin-output-base",
        args.merlin_output_base,
        "--group-ids",
        str(args.group_id),
        "--cluster-id",
        str(args.cluster_id),
        "--queue-name",
        args.queue_name,
        "--gpuv",
        args.gpuv,
        "--gpu",
        "8",
        "--cpu",
        str(args.cpu),
        "--memory",
        str(args.memory),
        "--hdfs-volume-json",
        args.hdfs_volume_json,
    ]
    if merlin_parallel_cases > 1:
        cmd += ["--merlin-parallel-cases", str(merlin_parallel_cases)]
        cmd += ["--merlin-gpus-per-case", str(merlin_gpus_per_case or nproc)]
    if args.submit:
        cmd.append("--submit")
    run(cmd, dry_run=False)


def parse_args() -> argparse.Namespace:
    stamp = f"plainmuon_d8_a100_{time.strftime('%Y%m%d_%H%M')}"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stamp", default=stamp)
    parser.add_argument("--submit", action="store_true", help="Submit jobs; default is dry-run payload generation.")
    parser.add_argument("--dry-run", action="store_true", help="Dry-run packaging commands too.")
    parser.add_argument("--skip-package", action="store_true", help="Use --hdfs-code-tgz instead of repackaging.")
    parser.add_argument("--hdfs-code-tgz", default="")
    parser.add_argument("--hdfs-code-dir", default=f"{HDFS_BASE}/sigma-search")
    parser.add_argument("--hdfs-runtime-tgz", default=DEFAULT_RUNTIME)
    parser.add_argument("--image-url", default=DEFAULT_IMAGE)
    parser.add_argument("--nanochat-base-dir", default="/mnt/hdfs/user/xingyu.dang/nanochat")
    parser.add_argument("--merlin-output-base", default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--group-id", type=int, default=914)
    parser.add_argument("--cluster-id", type=int, default=34)
    parser.add_argument("--queue-name", default=A100_QUEUE)
    parser.add_argument("--gpuv", default="A100_SXM_80GB")
    parser.add_argument("--cpu", type=int, default=120)
    parser.add_argument("--memory", type=int, default=1941504)
    parser.add_argument("--hdfs-volume-json", default=DEFAULT_HDFS_VOLUME)
    parser.add_argument("--weight-decay", type=float, default=0.28)
    parser.add_argument("--include-4k", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-main", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lrs-4k", default="0.0005 0.00075 0.001 0.0015 0.002")
    parser.add_argument("--lrs-main", default="0.001 0.0015 0.002 0.003 0.004")
    parser.add_argument("--main-shard-count", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.skip_package:
        if not args.hdfs_code_tgz:
            raise SystemExit("--skip-package requires --hdfs-code-tgz")
        hdfs_code_tgz = args.hdfs_code_tgz
    else:
        hdfs_code_tgz = package_code(args)

    if args.include_4k:
        submit_group(
            args,
            hdfs_code_tgz=hdfs_code_tgz,
            group="4k",
            batches="4096",
            lrs=args.lrs_4k,
            nproc=4,
            max_device_batch_size=1,
            shard_count=1,
            merlin_parallel_cases=2,
            merlin_gpus_per_case=4,
        )

    if args.include_main:
        submit_group(
            args,
            hdfs_code_tgz=hdfs_code_tgz,
            group="64k_1m",
            batches="65536 131072 524288 1048576",
            lrs=args.lrs_main,
            nproc=8,
            max_device_batch_size=64,
            shard_count=args.main_shard_count,
        )


if __name__ == "__main__":
    main()

