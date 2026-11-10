"""Small reproducibility helpers shared by command-line and notebook runs."""

from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import random
import resource
from typing import Any

import numpy as np
import torch


def portable_path(path: str | os.PathLike[str]) -> str:
    """Render local paths relative to the home directory for user-facing logs."""

    resolved = Path(path).expanduser().resolve()
    try:
        relative = resolved.relative_to(Path.home().resolve())
    except ValueError:
        return str(resolved)
    return "~" if not relative.parts else str(Path("~") / relative)


def set_global_seed(seed: int = 42, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, PyTorch and PyTorch Lightning."""

    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        import pytorch_lightning as pl

        pl.seed_everything(seed, workers=True)
    except ImportError:
        pass


def seed_worker(worker_id: int) -> None:
    """Seed a DataLoader worker from the current PyTorch seed."""

    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


def peak_host_memory_mb() -> float:
    """Return peak resident-set size in MiB for the current process."""

    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes.  DeepHEx's supported cluster runs
    # are Linux, but the guard keeps local smoke tests sensible on macOS.
    if peak_kib > 10**10:
        peak_kib /= 1024
    return float(peak_kib / 1024)


def reset_cuda_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_cuda_memory_mb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated() / (1024**2))


def software_versions() -> dict[str, str | None]:
    packages = [
        "numpy",
        "pandas",
        "scanpy",
        "anndata",
        "scipy",
        "torch",
        "pytorch_lightning",
        "timm",
        "huggingface_hub",
        "Pillow",
        "scikit-image",
        "bin2cell",
    ]
    result: dict[str, str | None] = {}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def environment_metadata() -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": os.sys.version,
        "platform": os.uname().machine if hasattr(os, "uname") else None,
        "cpu_count": os.cpu_count(),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "software_versions": software_versions(),
    }


def redact_config(value: Any) -> Any:
    """Recursively remove credentials before writing run metadata."""

    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(marker in lowered for marker in ("login", "token", "password")):
                redacted[key] = "<redacted>"
            else:
                redacted[key] = redact_config(item)
        return redacted
    if isinstance(value, list):
        return [redact_config(item) for item in value]
    if isinstance(value, tuple):
        return [redact_config(item) for item in value]
    return value


def write_json(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
