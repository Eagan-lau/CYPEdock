"""CYP450 holo-template library manifest handling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

from .errors import InputError
from .provenance import sha256_file


@dataclass(frozen=True)
class LibraryEntry:
    target_id: str
    structure_file: Path
    structure_sha256: str
    protein_chain: str
    cofactor_residues: tuple[Mapping[str, object], ...]


def default_library_root() -> Path:
    return Path(__file__).resolve().parent / "resources" / "cyp450_library"


def library_manifest_path(root: Path) -> Path:
    library_root = Path(root).expanduser().resolve()
    candidates = (
        library_root / "manifest.jsonl",
        library_root / "library_manifest.jsonl",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise InputError(
        "CYP450 library manifest does not exist; expected one of: "
        + ", ".join(str(path) for path in candidates)
    )


def default_database_prefix(root: Path) -> Path:
    library_root = Path(root).expanduser().resolve()
    candidates = (
        library_root / "foldseek" / "cyp450",
        library_root / "cyp450_holo",
    )
    for candidate in candidates:
        if Path(f"{candidate}.dbtype").is_file():
            return candidate
    return candidates[0]


def _contained(root: Path, path: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def load_library(
    root: Path,
    *,
    structure_root: Optional[Path] = None,
    verify_hashes: bool = True,
) -> dict[str, LibraryEntry]:
    library_root = Path(root).expanduser().resolve()
    external_structure_root = (
        Path(structure_root).expanduser().resolve()
        if structure_root is not None
        else None
    )
    manifest = library_manifest_path(library_root)
    entries: dict[str, LibraryEntry] = {}
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise InputError(f"Invalid library JSON at line {line_number}: {exc}") from exc
            if not isinstance(value, dict) or value.get("schema_version") != 1:
                raise InputError(f"Library line {line_number} must use schema_version 1")
            target_id = str(value.get("target_id", "")).strip()
            if not target_id or target_id in entries:
                raise InputError(f"Missing or duplicate library target_id at line {line_number}")
            declared_path = Path(str(value.get("structure_file", ""))).expanduser()
            if external_structure_root is not None:
                relative_structure = (
                    Path(declared_path.name) if declared_path.is_absolute() else declared_path
                )
                structure_file = (external_structure_root / relative_structure).resolve()
                if not _contained(external_structure_root, structure_file):
                    raise InputError(
                        f"Library structure escapes foldseek.structure_root at line "
                        f"{line_number}"
                    )
            elif declared_path.is_absolute():
                structure_file = declared_path.resolve()
            else:
                structure_file = (library_root / declared_path).resolve()
                if not _contained(library_root, structure_file):
                    raise InputError(
                        f"Relative library structure escapes the library root at line "
                        f"{line_number}"
                    )
            if not structure_file.is_file():
                raise InputError(f"Invalid library structure path at line {line_number}")
            expected_hash = str(value.get("structure_sha256", "")).lower()
            if len(expected_hash) != 64:
                raise InputError(f"Missing structure SHA-256 at library line {line_number}")
            if verify_hashes and sha256_file(structure_file) != expected_hash:
                raise InputError(f"Structure hash mismatch at library line {line_number}")
            protein_chain = str(value.get("protein_chain", ""))
            selectors = value.get("cofactor_residues")
            if selectors is None:
                groups = value.get("cofactor_groups")
                if not isinstance(groups, list) or len(groups) != 1:
                    raise InputError(
                        f"Exactly one cofactor group is required at library line {line_number}"
                    )
                group = groups[0]
                if not isinstance(group, dict):
                    raise InputError(f"Invalid cofactor group at library line {line_number}")
                selectors = group.get("residues")
            if not protein_chain or not isinstance(selectors, list) or not selectors:
                raise InputError(
                    f"protein_chain and cofactor_residues are required at line {line_number}"
                )
            for selector in selectors:
                if not isinstance(selector, dict) or "number" not in selector:
                    raise InputError(f"Invalid cofactor selector at library line {line_number}")
            entries[target_id] = LibraryEntry(
                target_id=target_id,
                structure_file=structure_file,
                structure_sha256=expected_hash,
                protein_chain=protein_chain,
                cofactor_residues=tuple(selectors),
            )
    return entries


def validate_database_prefix(prefix: Path) -> None:
    database = Path(prefix).expanduser().resolve()
    required = Path(f"{database}.dbtype")
    if not required.is_file():
        raise InputError(
            f"Foldseek database is unavailable. Expected at least {required}; "
            "install the published CYP450 library before enabling fallback"
        )
