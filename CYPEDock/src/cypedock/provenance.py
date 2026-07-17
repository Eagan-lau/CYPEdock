"""Hashing, JSON, tabular output, and subprocess provenance."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from .errors import ExternalToolError


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, destination)
    return destination


def write_tsv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fields), delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    os.replace(temporary, destination)
    return destination


@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    elapsed_seconds: float


def run_command(
    argv: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    timeout: int = 3600,
    environment: Optional[Mapping[str, str]] = None,
    log_dir: Optional[Path] = None,
    check: bool = True,
) -> CommandResult:
    command = [str(value) for value in argv]
    if not command:
        raise ExternalToolError("External command is empty")
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=None if cwd is None else str(cwd),
            env=None if environment is None else dict(environment),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ExternalToolError(f"Command could not be executed: {command[0]}: {exc}") from exc
    result = CommandResult(
        argv=tuple(command),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        elapsed_seconds=time.monotonic() - started,
    )
    if log_dir is not None:
        destination = Path(log_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "stdout.log").write_text(result.stdout, encoding="utf-8")
        (destination / "stderr.log").write_text(result.stderr, encoding="utf-8")
        write_json(
            destination / "command.json",
            {
                "argv": command,
                "cwd": str(cwd) if cwd is not None else None,
                "returncode": result.returncode,
                "elapsed_seconds": result.elapsed_seconds,
            },
        )
    if check and result.returncode != 0:
        tail = result.stderr.strip().splitlines()[-1:] or result.stdout.strip().splitlines()[-1:]
        detail = tail[0] if tail else "no diagnostic output"
        raise ExternalToolError(
            f"Command failed with exit code {result.returncode}: {command[0]}: {detail}"
        )
    return result
