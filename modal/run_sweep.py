"""Modal launchers for sigma-search clean sweeps.

These wrappers intentionally call the repo's existing shell/Python entrypoints
instead of duplicating sweep logic. Configure the Modal resource before launch,
for example:

    MODAL_GPU=B200:8 modal run modal/run_sweep.py::sweep --depth 12
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import modal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/workspace/sigma-search")
DATA_ROOT = Path("/data/nanochat")
OUTPUT_ROOT = Path("/outputs")

APP_NAME = os.environ.get("MODAL_APP_NAME", "sigma-search-clean")
DEFAULT_GPU = os.environ.get("MODAL_GPU", "B200:8")
DEFAULT_CPU = float(os.environ.get("MODAL_CPU", "64"))
DEFAULT_MEMORY_MB = int(os.environ.get("MODAL_MEMORY_MB", "262144"))
DEFAULT_TIMEOUT = int(os.environ.get("MODAL_TIMEOUT", str(24 * 60 * 60)))

DATA_VOLUME_NAME = os.environ.get("MODAL_DATA_VOLUME", "sigma-search-nanochat-data")
OUTPUT_VOLUME_NAME = os.environ.get("MODAL_OUTPUT_VOLUME", "sigma-search-runs")


def _optional_secrets() -> list[modal.Secret]:
    names = []
    for env_name in ("MODAL_WANDB_SECRET", "MODAL_HF_SECRET"):
        value = os.environ.get(env_name)
        if value:
            names.append(value)
    return [modal.Secret.from_name(name) for name in names]


def _ignore_upload(path: Path) -> bool:
    ignored_names = {
        ".git",
        ".pytest_cache",
        ".uv-cache",
        "__pycache__",
        "logs",
        "results",
        "search_evals",
        "wandb",
    }
    if any(part in ignored_names for part in path.parts):
        return True
    if path.suffix in {".log", ".pt", ".pth", ".safetensors"}:
        return True
    if "nanochat/.venv" in path.as_posix():
        return True
    return False


NANOCHAT_PACKAGES = [
    "datasets>=4.0.0",
    "fastapi>=0.117.1",
    "kernels>=0.11.7",
    "matplotlib>=3.10.8",
    "psutil>=7.1.0",
    "pytest>=8.0.0",
    "python-dotenv>=1.2.1",
    "rustbpe>=0.1.0",
    "tiktoken>=0.11.0",
    "tokenizers>=0.22.0",
    "torch==2.9.1",
    "transformers>=4.57.3",
    "uvicorn>=0.36.0",
    "wandb>=0.21.3",
]


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "ca-certificates", "curl", "git", "pkg-config")
    .uv_pip_install(
        *NANOCHAT_PACKAGES,
        extra_index_url="https://download.pytorch.org/whl/cu128",
        extra_options="--index-strategy unsafe-best-match",
    )
    .add_local_dir(PROJECT_ROOT, str(REMOTE_ROOT), copy=True, ignore=_ignore_upload)
    .env(
        {
            "NANOCHAT_BASE_DIR": str(DATA_ROOT),
            "PYTHONPATH": f"{REMOTE_ROOT}:{REMOTE_ROOT / 'nanochat'}",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "HF_HOME": str(DATA_ROOT / "hf"),
            "TRANSFORMERS_CACHE": str(DATA_ROOT / "hf" / "transformers"),
        }
    )
)

data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


def _runtime_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    stamp = (extra or {}).get("STAMP") or env.get("STAMP", "modal_run")
    env.update(
        {
            "NANOCHAT_BASE_DIR": str(DATA_ROOT),
            "PYTHONPATH": f"{REMOTE_ROOT}:{REMOTE_ROOT / 'nanochat'}",
            "OUT_ROOT": str(OUTPUT_ROOT / "search_evals" / stamp),
            "LOG_ROOT": str(OUTPUT_ROOT / "logs" / stamp),
            "HF_HOME": str(DATA_ROOT / "hf"),
            "TRANSFORMERS_CACHE": str(DATA_ROOT / "hf" / "transformers"),
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    if extra:
        env.update({key: str(value) for key, value in extra.items() if value is not None})
        env.setdefault("OUT_ROOT", str(OUTPUT_ROOT / "search_evals" / stamp))
        env.setdefault("LOG_ROOT", str(OUTPUT_ROOT / "logs" / stamp))
    return env


def _run_shell(cmd: str, *, cwd: Path, env: dict[str, str]) -> None:
    print(f"+ cd {cwd}")
    print(f"+ {cmd}", flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, shell=True, check=True)


def _has_climbmix_data() -> bool:
    return any((DATA_ROOT / "base_data_climbmix").glob("*.parquet"))


def _has_tokenizer() -> bool:
    tokenizer_dir = DATA_ROOT / "tokenizer"
    return (tokenizer_dir / "tokenizer.pkl").exists() and (tokenizer_dir / "token_bytes.pt").exists()


def _ensure_data(dataset_shards: int, dataset_workers: int, tokenizer_max_chars: int) -> None:
    env = _runtime_env()
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    if not _has_climbmix_data():
        _run_shell(
            f"python -m nanochat.dataset -n {dataset_shards} -w {dataset_workers}",
            cwd=REMOTE_ROOT / "nanochat",
            env=env,
        )
        data_volume.commit()
    else:
        print("ClimbMix data already present in Modal data volume.", flush=True)

    if not _has_tokenizer():
        _run_shell(
            f"python -m scripts.tok_train --max-chars {tokenizer_max_chars}",
            cwd=REMOTE_ROOT / "nanochat",
            env=env,
        )
        data_volume.commit()
    else:
        print("Tokenizer already present in Modal data volume.", flush=True)


@app.function(
    image=image,
    volumes={str(DATA_ROOT): data_volume},
    cpu=16,
    memory=65536,
    timeout=12 * 60 * 60,
    secrets=_optional_secrets(),
)
def _prepare_data(dataset_shards: int, dataset_workers: int, tokenizer_max_chars: int) -> dict[str, str | int | bool]:
    _ensure_data(dataset_shards, dataset_workers, tokenizer_max_chars)
    return {
        "data_volume": DATA_VOLUME_NAME,
        "has_climbmix_data": _has_climbmix_data(),
        "has_tokenizer": _has_tokenizer(),
        "dataset_shards": dataset_shards,
        "tokenizer_max_chars": tokenizer_max_chars,
    }


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    cpu=DEFAULT_CPU,
    memory=DEFAULT_MEMORY_MB,
    timeout=DEFAULT_TIMEOUT,
    volumes={str(DATA_ROOT): data_volume, str(OUTPUT_ROOT): output_volume},
    secrets=_optional_secrets(),
)
def _run_gpu_command(
    cmd: str,
    env: dict[str, str],
    ensure_data: bool,
    dataset_shards: int,
    dataset_workers: int,
    tokenizer_max_chars: int,
) -> dict[str, str | float]:
    started = time.time()
    if ensure_data:
        _ensure_data(dataset_shards, dataset_workers, tokenizer_max_chars)
    run_env = _runtime_env(env)
    try:
        _run_shell(cmd, cwd=REMOTE_ROOT, env=run_env)
    finally:
        output_volume.commit()
    return {
        "cmd": cmd,
        "stamp": run_env.get("STAMP", ""),
        "out_root": run_env.get("OUT_ROOT", ""),
        "log_root": run_env.get("LOG_ROOT", ""),
        "seconds": time.time() - started,
        "output_volume": OUTPUT_VOLUME_NAME,
    }


def _stamp(prefix: str) -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"


def _submit(cmd: str, env: dict[str, str], ensure_data: bool, dataset_shards: int,
            dataset_workers: int, tokenizer_max_chars: int) -> None:
    print(f"Modal app: {APP_NAME}")
    print(f"Modal GPU: {DEFAULT_GPU}")
    print(f"Command: {cmd}")
    result = _run_gpu_command.remote(
        cmd,
        env,
        ensure_data,
        dataset_shards,
        dataset_workers,
        tokenizer_max_chars,
    )
    print(result)


@app.local_entrypoint()
def prepare_data(
    dataset_shards: int = 170,
    dataset_workers: int = 16,
    tokenizer_max_chars: int = 2_000_000_000,
) -> None:
    """Download ClimbMix and train/cache the tokenizer in the Modal data volume."""
    print(_prepare_data.remote(dataset_shards, dataset_workers, tokenizer_max_chars))


@app.local_entrypoint()
def sweep(
    depth: int = 12,
    chinchilla_mult: float = 2.0,
    methods: str = "top_aware_muon",
    batches: str = "524288 2097152 8388608",
    alphas: str = "1.0 0.5",
    top_ks: str = "1",
    lrs: str = "0.005 0.0075 0.01 0.015 0.02 0.03 0.04",
    seeds: str = "42",
    nproc: int = 8,
    max_device_batch_size: int = 16,
    weight_decay: float = 0.28,
    architecture: str = "gpt2",
    stamp: str = "",
    tokens: str = "",
    adaptive_lr: bool = True,
    dry_run: bool = False,
    ensure_data: bool = True,
    dataset_shards: int = 170,
    dataset_workers: int = 16,
    tokenizer_max_chars: int = 2_000_000_000,
    extra_args: str = "",
) -> None:
    """Run the normal Top-Aware/StreamingMuon sweep on Modal."""
    stamp = stamp or _stamp(f"modal_d{depth}_sweep")
    env = {
        "DEPTH": str(depth),
        "CHINCHILLA_MULT": f"{chinchilla_mult:g}",
        "METHODS": methods,
        "BATCHES": batches,
        "ALPHAS": alphas,
        "TOP_KS": top_ks,
        "LRS": lrs,
        "SEEDS": seeds,
        "NPROC": str(nproc),
        "MAX_DEVICE_BATCH_SIZE": str(max_device_batch_size),
        "WEIGHT_DECAY": f"{weight_decay:g}",
        "ARCHITECTURE": architecture,
        "STAMP": stamp,
        "OUT_ROOT": str(OUTPUT_ROOT / "search_evals" / stamp),
        "LOG_ROOT": str(OUTPUT_ROOT / "logs" / stamp),
        "ADAPTIVE_LR": "1" if adaptive_lr else "0",
        "DRY_RUN": "1" if dry_run else "0",
    }
    if tokens:
        env["TOKENS"] = tokens
    cmd = "bash scripts/run_d12_sweep.sh"
    if extra_args:
        cmd += " " + extra_args
    _submit(cmd, env, ensure_data, dataset_shards, dataset_workers, tokenizer_max_chars)


@app.local_entrypoint()
def optimizer_baselines(
    depth: int = 12,
    chinchilla_mult: float = 2.0,
    methods: str = "plain_muon adamw soap shampoo kl_shampoo kl_soap",
    batches: str = "524288 2097152 8388608",
    lrs: str = "0.0005 0.001 0.002 0.004 0.008 0.015 0.02 0.04",
    groups: str = "plain_muon adamw soap_klsoap kl_shampoo shampoo",
    grouped: bool = True,
    seeds: str = "42",
    nproc: int = 8,
    max_device_batch_size: int = 16,
    weight_decay: float = 0.1,
    stamp: str = "",
    dry_run: bool = False,
    ensure_data: bool = True,
    dataset_shards: int = 170,
    dataset_workers: int = 16,
    tokenizer_max_chars: int = 2_000_000_000,
) -> None:
    """Run non-streaming optimizer baselines with reference structured configs."""
    stamp = stamp or _stamp(f"modal_d{depth}_optimizer_baselines")
    env = {
        "DEPTH": str(depth),
        "CHINCHILLA_MULT": f"{chinchilla_mult:g}",
        "METHODS": methods,
        "BATCHES": batches,
        "LRS": lrs,
        "OPTIMIZER_GROUPS": groups,
        "GROUPED": "1" if grouped else "0",
        "SEEDS": seeds,
        "ALPHAS": "1.0",
        "TOP_KS": "1",
        "NPROC": str(nproc),
        "MAX_DEVICE_BATCH_SIZE": str(max_device_batch_size),
        "WEIGHT_DECAY": f"{weight_decay:g}",
        "ARCHITECTURE": "gpt2",
        "STAMP": stamp,
        "OUT_ROOT": str(OUTPUT_ROOT / "search_evals" / stamp),
        "LOG_ROOT": str(OUTPUT_ROOT / "logs" / stamp),
        "ADAPTIVE_LR": "1",
        "STRUCTURED_CONFIG": "auto",
        "DRY_RUN": "1" if dry_run else "0",
    }
    _submit(
        "bash scripts/run_optimizer_baseline_groups.sh",
        env,
        ensure_data,
        dataset_shards,
        dataset_workers,
        tokenizer_max_chars,
    )


@app.local_entrypoint()
def command(
    cmd: str,
    stamp: str = "",
    nproc: int = 1,
    dry_run: bool = False,
    ensure_data: bool = False,
    dataset_shards: int = 170,
    dataset_workers: int = 16,
    tokenizer_max_chars: int = 2_000_000_000,
) -> None:
    """Run an arbitrary shell command in the Modal image."""
    stamp = stamp or _stamp("modal_command")
    env = {
        "STAMP": stamp,
        "NPROC": str(nproc),
        "OUT_ROOT": str(OUTPUT_ROOT / "search_evals" / stamp),
        "LOG_ROOT": str(OUTPUT_ROOT / "logs" / stamp),
        "DRY_RUN": "1" if dry_run else "0",
    }
    _submit(cmd, env, ensure_data, dataset_shards, dataset_workers, tokenizer_max_chars)
