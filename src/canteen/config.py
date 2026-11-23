"""Load params.yaml and expose the repository root."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_params(path: str | os.PathLike | None = None) -> dict:
    """Read params.yaml (path can be overridden with the PARAMS_FILE env var)."""
    path = Path(path or os.environ.get("PARAMS_FILE", ROOT / "params.yaml"))
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(rel: str) -> Path:
    """Turn a path from params.yaml into an absolute path under the repo root."""
    return ROOT / rel


def sha256_file(path: str | os.PathLike) -> str:
    """SHA-256 of a file's raw bytes (used as the data_sha256 MLflow param)."""
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()
