"""Framework-independent provenance helpers."""

from __future__ import annotations

import hashlib
import os


def sha256_file(path: os.PathLike[str] | str, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()
