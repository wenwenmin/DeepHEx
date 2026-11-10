"""Check the software and optional UNI/GPU prerequisites for DeepHEx."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import sys

from huggingface_hub import get_token

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.reproducibility import environment_metadata


REQUIRED_MODULES = [
    "numpy",
    "pandas",
    "scanpy",
    "anndata",
    "h5py",
    "sklearn",
    "scipy",
    "skimage",
    "shapely",
    "torch",
    "pytorch_lightning",
    "timm",
    "huggingface_hub",
]


def main() -> dict:
    missing = []
    for module in REQUIRED_MODULES:
        try:
            importlib.import_module(module)
        except Exception as exc:  # pragma: no cover - depends on user environment
            missing.append({"module": module, "error": repr(exc)})

    metadata = environment_metadata()
    metadata["missing_modules"] = missing
    metadata["hf_token_configured"] = bool(os.environ.get("HF_TOKEN") or get_token())
    metadata["ready_for_preprocessing"] = not missing
    metadata["ready_for_uni"] = not missing and metadata["hf_token_configured"]
    print(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    main()
