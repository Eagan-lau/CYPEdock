"""Foldseek retrieval and alignment-correspondence parsing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .alignment import SequenceMapping
from .errors import ExternalToolError, InputError
from .provenance import run_command


FOLDSEEK_FIELDS = (
    "query",
    "target",
    "alntmscore",
    "qtmscore",
    "ttmscore",
    "rmsd",
    "fident",
    "alnlen",
    "qcov",
    "tcov",
    "qstart",
    "qend",
    "tstart",
    "tend",
    "qaln",
    "taln",
    "evalue",
    "bits",
)


@dataclass(frozen=True)
class FoldseekHit:
    query: str
    target: str
    alignment_tm_score: float
    query_tm_score: float
    target_tm_score: float
    rmsd_angstrom: float
    sequence_identity: float
    alignment_length: int
    query_coverage: float
    target_coverage: float
    query_start: int
    query_end: int
    target_start: int
    target_end: int
    query_alignment: str
    target_alignment: str
    evalue: float
    bits: float


def _finite_float(value: str, field: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise InputError(f"Foldseek field {field} is not numeric: {value!r}") from exc
    if not math.isfinite(parsed):
        raise InputError(f"Foldseek field {field} is not finite")
    return parsed


def _integer(value: str, field: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise InputError(f"Foldseek field {field} is not an integer: {value!r}") from exc
    if parsed < 0:
        raise InputError(f"Foldseek field {field} cannot be negative")
    return parsed


def parse_hits(path: Path) -> list[FoldseekHit]:
    source = Path(path)
    if not source.is_file():
        raise ExternalToolError(f"Foldseek output does not exist: {source}")
    hits: list[FoldseekHit] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            values = stripped.split("\t")
            if len(values) != len(FOLDSEEK_FIELDS):
                raise InputError(
                    f"Foldseek line {line_number} contains {len(values)} fields; "
                    f"{len(FOLDSEEK_FIELDS)} are required"
                )
            row = dict(zip(FOLDSEEK_FIELDS, values))
            query_alignment = row["qaln"].upper()
            target_alignment = row["taln"].upper()
            if len(query_alignment) != len(target_alignment) or not query_alignment:
                raise InputError(f"Invalid aligned sequences at Foldseek line {line_number}")
            hits.append(
                FoldseekHit(
                    query=row["query"],
                    target=row["target"],
                    alignment_tm_score=_finite_float(row["alntmscore"], "alntmscore"),
                    query_tm_score=_finite_float(row["qtmscore"], "qtmscore"),
                    target_tm_score=_finite_float(row["ttmscore"], "ttmscore"),
                    rmsd_angstrom=_finite_float(row["rmsd"], "rmsd"),
                    sequence_identity=_finite_float(row["fident"], "fident"),
                    alignment_length=_integer(row["alnlen"], "alnlen"),
                    query_coverage=_finite_float(row["qcov"], "qcov"),
                    target_coverage=_finite_float(row["tcov"], "tcov"),
                    query_start=_integer(row["qstart"], "qstart"),
                    query_end=_integer(row["qend"], "qend"),
                    target_start=_integer(row["tstart"], "tstart"),
                    target_end=_integer(row["tend"], "tend"),
                    query_alignment=query_alignment,
                    target_alignment=target_alignment,
                    evalue=_finite_float(row["evalue"], "evalue"),
                    bits=_finite_float(row["bits"], "bits"),
                )
            )
    return sorted(hits, key=lambda hit: (-hit.alignment_tm_score, hit.target))


def mapping_from_hit(
    hit: FoldseekHit,
    *,
    reference_length: int,
    target_length: int,
) -> SequenceMapping:
    if hit.query_start < 1 or hit.target_start < 1:
        raise InputError("Foldseek alignment starts must be one-based positive integers")
    query_index = hit.query_start - 1
    template_index = hit.target_start - 1
    mapping: dict[int, int] = {}
    identical = 0
    for query_letter, template_letter in zip(hit.query_alignment, hit.target_alignment):
        query_present = query_letter != "-"
        template_present = template_letter != "-"
        if query_present and template_present:
            if query_index >= target_length or template_index >= reference_length:
                raise InputError("Foldseek alignment extends beyond a structure sequence")
            mapping[template_index] = query_index
            identical += int(query_letter == template_letter)
        if query_present:
            query_index += 1
        if template_present:
            template_index += 1
    if not mapping:
        raise InputError("Foldseek alignment contains no matched residue pair")
    return SequenceMapping(
        reference_to_target=mapping,
        identity=identical / len(mapping),
        reference_coverage=len(mapping) / reference_length,
        target_coverage=len(mapping) / target_length,
    )


def search(
    *,
    executable: str,
    query_pdb: Path,
    database_prefix: Path,
    output_tsv: Path,
    temporary_directory: Path,
    settings: Mapping[str, object],
    log_dir: Path,
) -> list[FoldseekHit]:
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    temporary_directory.mkdir(parents=True, exist_ok=True)
    format_value = ",".join(FOLDSEEK_FIELDS)
    command: Sequence[str] = (
        executable,
        "easy-search",
        str(query_pdb),
        str(database_prefix),
        str(output_tsv),
        str(temporary_directory),
        "--alignment-type",
        "1",
        "--tmalign-hit-order",
        "3",
        "--max-seqs",
        str(settings["maximum_hits"]),
        "--threads",
        str(settings["threads"]),
        "--format-output",
        format_value,
    )
    run_command(command, timeout=3600, log_dir=log_dir)
    hits = parse_hits(output_tsv)
    return [
        hit
        for hit in hits
        if hit.alignment_tm_score >= float(settings["minimum_alignment_tm_score"])
        and hit.query_coverage >= float(settings["minimum_query_coverage"])
    ][: int(settings["maximum_hits"])]
