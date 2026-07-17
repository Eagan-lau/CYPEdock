"""Portable discovery of separately distributed model artifacts."""

from __future__ import annotations

import os
from pathlib import Path


def default_model_dir() -> Path:
    """Return the configured or discoverable ``P450model`` directory.

    Large checkpoints intentionally remain outside the Python wheel.  Source
    checkouts work without configuration; installed console scripts can use
    ``P450_MODEL_DIR`` or be launched from the project checkout.
    """

    configured = os.environ.get("P450_MODEL_DIR")
    if configured:
        return Path(configured).expanduser().resolve()

    project_root = Path(__file__).resolve().parent.parent
    candidates = (
        Path.cwd() / "P450model",
        project_root / "P450model",
        project_root.parent / "P450model",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    return candidates[0].resolve()


def default_model_artifact(name: str) -> Path:
    return default_model_dir() / name
