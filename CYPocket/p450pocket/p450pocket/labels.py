"""Generate residue labels without losing PDB residue identity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from .structure import ResidueKey, ResidueRecord, StructureRecord


class LabelMappingError(ValueError):
    """Raised when pocket residues cannot be mapped uniquely to the protein."""


@dataclass(frozen=True)
class LabelResult:
    """Binary labels aligned one-to-one with ``protein.residues``."""

    labels: np.ndarray
    positive_keys: Tuple[ResidueKey, ...]
    match_mode: str
    warnings: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        values = np.asarray(self.labels, dtype=np.int64).copy()
        if values.ndim != 1:
            raise LabelMappingError(f"labels must be one-dimensional, got {values.shape}")
        if not np.isin(values, (0, 1)).all():
            raise LabelMappingError("labels may contain only 0 and 1")
        values.setflags(write=False)
        object.__setattr__(self, "labels", values)

    @property
    def n_positive(self) -> int:
        return int(self.labels.sum())


def _position_without_model(residue: ResidueRecord) -> Tuple[str, int, str]:
    # Protein and extracted pocket files each normally start at MODEL 0.  MODEL
    # serialisation is therefore not part of cross-file residue identity.
    return (
        residue.key.chain_id,
        residue.key.auth_seq_id,
        residue.key.insertion_code,
    )


def _ca_distance(left: ResidueRecord, right: ResidueRecord) -> float:
    return float(np.linalg.norm(left.ca_angstrom - right.ca_angstrom))


def _validate_residue_pair(
    protein_residue: ResidueRecord,
    pocket_residue: ResidueRecord,
    *,
    verify_ca_distance_angstrom: Optional[float],
) -> None:
    if protein_residue.canonical_name != pocket_residue.canonical_name:
        raise LabelMappingError(
            "Residue-name mismatch at "
            f"chain={protein_residue.key.chain_id or '_'} "
            f"position={protein_residue.key.auth_seq_id}"
            f"{protein_residue.key.insertion_code}: protein="
            f"{protein_residue.key.residue_name}, pocket="
            f"{pocket_residue.key.residue_name}"
        )
    if verify_ca_distance_angstrom is not None:
        distance = _ca_distance(protein_residue, pocket_residue)
        if distance > verify_ca_distance_angstrom:
            raise LabelMappingError(
                f"CA coordinate mismatch for {protein_residue.key.display()}: "
                f"{distance:.4f} Å exceeds {verify_ca_distance_angstrom:.4f} Å"
            )


def _identity_matches(
    protein: StructureRecord,
    pocket: StructureRecord,
    *,
    verify_ca_distance_angstrom: Optional[float],
) -> Tuple[Dict[int, ResidueRecord], Tuple[ResidueRecord, ...]]:
    protein_by_position: Dict[Tuple[str, int, str], Tuple[int, ResidueRecord]] = {}
    for index, residue in enumerate(protein.residues):
        position = _position_without_model(residue)
        if position in protein_by_position:
            raise LabelMappingError(f"Duplicate protein residue position: {position}")
        protein_by_position[position] = (index, residue)

    matched: Dict[int, ResidueRecord] = {}
    unmatched: list[ResidueRecord] = []
    seen_pocket_positions: set[Tuple[str, int, str]] = set()
    for pocket_residue in pocket.residues:
        position = _position_without_model(pocket_residue)
        if position in seen_pocket_positions:
            raise LabelMappingError(f"Duplicate pocket residue position: {position}")
        seen_pocket_positions.add(position)

        candidate = protein_by_position.get(position)
        if candidate is None:
            unmatched.append(pocket_residue)
            continue
        index, protein_residue = candidate
        _validate_residue_pair(
            protein_residue,
            pocket_residue,
            verify_ca_distance_angstrom=verify_ca_distance_angstrom,
        )
        matched[index] = pocket_residue
    return matched, tuple(unmatched)


def _coordinate_matches(
    protein: StructureRecord,
    pocket_residues: Iterable[ResidueRecord],
    *,
    tolerance_angstrom: float,
    already_matched: Iterable[int] = (),
) -> Dict[int, ResidueRecord]:
    if tolerance_angstrom <= 0:
        raise LabelMappingError("coordinate_tolerance_angstrom must be positive")

    unavailable = set(already_matched)
    matched: Dict[int, ResidueRecord] = {}
    for pocket_residue in pocket_residues:
        candidates = []
        for index, protein_residue in enumerate(protein.residues):
            if index in unavailable or protein_residue.canonical_name != pocket_residue.canonical_name:
                continue
            distance = _ca_distance(protein_residue, pocket_residue)
            if distance <= tolerance_angstrom:
                candidates.append((index, protein_residue, distance))

        if not candidates:
            raise LabelMappingError(
                f"No coordinate match for pocket residue {pocket_residue.key.display()} "
                f"within {tolerance_angstrom:.4f} Å"
            )
        if len(candidates) > 1:
            details = ", ".join(
                f"{candidate.key.display()} ({distance:.4f} Å)"
                for _, candidate, distance in candidates
            )
            raise LabelMappingError(
                f"Ambiguous coordinate match for {pocket_residue.key.display()}: {details}"
            )
        index, _, _ = candidates[0]
        matched[index] = pocket_residue
        unavailable.add(index)
    return matched


def build_labels(
    protein: StructureRecord,
    pocket: StructureRecord,
    *,
    match_mode: str = "identity",
    verify_ca_distance_angstrom: Optional[float] = 0.25,
    coordinate_tolerance_angstrom: float = 0.25,
    require_positive: bool = True,
) -> LabelResult:
    """Build labels aligned with the protein feature rows.

    ``identity`` is the publication-grade default.  Coordinate matching is
    deliberately opt-in and requires a unique, same-residue-type CA match.
    Supported modes are ``identity``, ``identity_then_coordinate`` and
    ``coordinate``.
    """

    modes = {"identity", "identity_then_coordinate", "coordinate"}
    if match_mode not in modes:
        raise LabelMappingError(
            f"Unknown match_mode {match_mode!r}; expected one of {sorted(modes)}"
        )
    if verify_ca_distance_angstrom is not None and verify_ca_distance_angstrom < 0:
        raise LabelMappingError("verify_ca_distance_angstrom must be non-negative or None")

    warnings: list[str] = []
    if match_mode == "coordinate":
        matched = _coordinate_matches(
            protein,
            pocket.residues,
            tolerance_angstrom=coordinate_tolerance_angstrom,
        )
        warnings.append("Labels were mapped by CA coordinates rather than residue identity")
    else:
        matched, unmatched = _identity_matches(
            protein,
            pocket,
            verify_ca_distance_angstrom=verify_ca_distance_angstrom,
        )
        if unmatched and match_mode == "identity":
            missing = ", ".join(residue.key.display() for residue in unmatched[:10])
            suffix = " ..." if len(unmatched) > 10 else ""
            raise LabelMappingError(
                f"{len(unmatched)} pocket residue(s) are absent from the protein: "
                f"{missing}{suffix}"
            )
        if unmatched:
            fallback = _coordinate_matches(
                protein,
                unmatched,
                tolerance_angstrom=coordinate_tolerance_angstrom,
                already_matched=matched,
            )
            matched.update(fallback)
            warnings.append(
                f"Mapped {len(unmatched)} pocket residue(s) using coordinate fallback"
            )

    labels = np.zeros(len(protein), dtype=np.int64)
    for index in matched:
        labels[index] = 1
    if require_positive and not matched:
        raise LabelMappingError("Pocket contains no residues that map to the protein")

    positive_keys = tuple(protein.residues[index].key for index in sorted(matched))
    return LabelResult(
        labels=labels,
        positive_keys=positive_keys,
        match_mode=match_mode,
        warnings=tuple(warnings),
    )
