"""Sequence-guided local structural alignment around a reference cofactor."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping, Optional, Sequence

import numpy as np
from Bio.Align import PairwiseAligner, substitution_matrices

from .errors import AlignmentError
from .structure import Residue, protein_sequence


@dataclass(frozen=True, eq=False)
class RigidTransform:
    """A proper transform applied as target = reference @ rotation.T + translation."""

    rotation: np.ndarray
    translation: np.ndarray
    rmsd_angstrom: float

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=float)
        translation = np.asarray(self.translation, dtype=float)
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise AlignmentError("A rigid transform requires a 3x3 rotation and a 3-vector")
        if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
            raise AlignmentError("Rigid-transform values must be finite")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise AlignmentError("Rotation matrix is not orthonormal")
        if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6):
            raise AlignmentError("Rotation matrix must have determinant +1")
        rotation = rotation.copy()
        translation = translation.copy()
        rotation.setflags(write=False)
        translation.setflags(write=False)
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)
        object.__setattr__(self, "rmsd_angstrom", float(self.rmsd_angstrom))

    def apply(self, coordinates: Sequence[Sequence[float]]) -> np.ndarray:
        values = np.asarray(coordinates, dtype=float)
        if values.shape == (3,):
            return values @ self.rotation.T + self.translation
        if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
            raise AlignmentError("Coordinates must have shape (n, 3) and be finite")
        return values @ self.rotation.T + self.translation

    def to_dict(self) -> dict:
        return {
            "direction": "reference_to_target",
            "rotation": self.rotation.tolist(),
            "translation": self.translation.tolist(),
            "rmsd_angstrom": self.rmsd_angstrom,
        }


def kabsch(
    reference: Sequence[Sequence[float]], target: Sequence[Sequence[float]]
) -> RigidTransform:
    source = np.asarray(reference, dtype=float)
    destination = np.asarray(target, dtype=float)
    if (
        source.ndim != 2
        or source.shape[1:] != (3,)
        or source.shape != destination.shape
        or source.shape[0] < 3
        or not np.all(np.isfinite(source))
        or not np.all(np.isfinite(destination))
    ):
        raise AlignmentError("Kabsch alignment requires equal finite coordinate arrays with n >= 3")
    source_center = np.mean(source, axis=0)
    destination_center = np.mean(destination, axis=0)
    source_centered = source - source_center
    destination_centered = destination - destination_center
    if np.linalg.matrix_rank(source_centered, tol=1e-10) < 2:
        raise AlignmentError("Alignment coordinates are collinear")
    covariance = source_centered.T @ destination_centered
    left, _, right_t = np.linalg.svd(covariance)
    row_rotation = left @ right_t
    if np.linalg.det(row_rotation) < 0:
        left[:, -1] *= -1
        row_rotation = left @ right_t
    rotation = row_rotation.T
    translation = destination_center - source_center @ rotation.T
    fitted = source @ rotation.T + translation
    rmsd = float(np.sqrt(np.mean(np.sum((fitted - destination) ** 2, axis=1))))
    return RigidTransform(rotation, translation, rmsd)


@dataclass(frozen=True)
class SequenceMapping:
    reference_to_target: Mapping[int, int]
    identity: float
    reference_coverage: float
    target_coverage: float


def sequence_mapping(reference: Sequence[Residue], target: Sequence[Residue]) -> SequenceMapping:
    reference_sequence = protein_sequence(reference)
    target_sequence = protein_sequence(target)
    if not reference_sequence or not target_sequence:
        raise AlignmentError("Both structures must contain a protein sequence")
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    alignment = aligner.align(reference_sequence, target_sequence)[0]
    mapping: dict[int, int] = {}
    for reference_block, target_block in zip(alignment.aligned[0], alignment.aligned[1]):
        reference_start, reference_end = map(int, reference_block)
        target_start, target_end = map(int, target_block)
        if reference_end - reference_start != target_end - target_start:
            raise AlignmentError("Unexpected unequal aligned sequence blocks")
        for offset in range(reference_end - reference_start):
            mapping[reference_start + offset] = target_start + offset
    if not mapping:
        raise AlignmentError("Global sequence alignment produced no residue correspondence")
    identical = sum(
        reference_sequence[reference_index] == target_sequence[target_index]
        for reference_index, target_index in mapping.items()
    )
    return SequenceMapping(
        reference_to_target=mapping,
        identity=identical / len(mapping),
        reference_coverage=len(mapping) / len(reference_sequence),
        target_coverage=len(mapping) / len(target_sequence),
    )


@dataclass(frozen=True)
class AlignmentMetrics:
    status: str
    reason: str
    sequence_identity: float
    sequence_reference_coverage: float
    sequence_target_coverage: float
    shell_residue_count: int
    mapped_shell_residue_count: int
    local_coverage: float
    alignment_atom_count: int
    local_rmsd_angstrom: Optional[float]
    maximum_local_residue_rmsd_angstrom: Optional[float]
    global_ca_rmsd_angstrom: Optional[float]
    reference_fe_s_distance_angstrom: Optional[float]
    target_fe_s_distance_angstrom: Optional[float]
    minimum_cofactor_protein_distance_angstrom: Optional[float]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AlignmentResult:
    metrics: AlignmentMetrics
    transform: Optional[RigidTransform]
    shell_reference_indices: tuple[int, ...]
    mapped_residue_pairs: tuple[tuple[int, int], ...]

    @property
    def passed(self) -> bool:
        return self.metrics.status == "PASS" and self.transform is not None


def _local_shell(
    reference: Sequence[Residue],
    cofactor: Sequence[Residue],
    radius_angstrom: float,
) -> list[int]:
    cofactor_coordinates = np.asarray(
        [atom.coordinate for residue in cofactor for atom in residue.heavy_atoms()],
        dtype=float,
    )
    if cofactor_coordinates.size == 0:
        raise AlignmentError("The selected cofactor contains no heavy atoms")
    shell: list[int] = []
    for index, residue in enumerate(reference):
        coordinates = np.asarray([atom.coordinate for atom in residue.heavy_atoms()], dtype=float)
        if coordinates.size == 0:
            continue
        distances = np.linalg.norm(
            coordinates[:, None, :] - cofactor_coordinates[None, :, :], axis=2
        )
        if float(np.min(distances)) <= radius_angstrom:
            shell.append(index)
    return shell


def _global_ca_rmsd(
    reference: Sequence[Residue],
    target: Sequence[Residue],
    mapping: Mapping[int, int],
) -> Optional[float]:
    reference_coordinates: list[np.ndarray] = []
    target_coordinates: list[np.ndarray] = []
    for reference_index, target_index in mapping.items():
        reference_atom = reference[reference_index].atom("CA")
        target_atom = target[target_index].atom("CA")
        if reference_atom is not None and target_atom is not None:
            reference_coordinates.append(reference_atom.coordinate)
            target_coordinates.append(target_atom.coordinate)
    if len(reference_coordinates) < 3:
        return None
    return kabsch(reference_coordinates, target_coordinates).rmsd_angstrom


def _failed(
    reason: str,
    mapping: SequenceMapping,
    shell: Sequence[int],
    pairs: Sequence[tuple[int, int]],
    atom_count: int,
    *,
    local_coverage: float,
    local_rmsd: Optional[float] = None,
    maximum_residue_rmsd: Optional[float] = None,
    global_rmsd: Optional[float] = None,
    reference_fe_s_distance: Optional[float] = None,
    target_fe_s_distance: Optional[float] = None,
    minimum_distance: Optional[float] = None,
) -> AlignmentResult:
    return AlignmentResult(
        metrics=AlignmentMetrics(
            status="SKIP",
            reason=reason,
            sequence_identity=mapping.identity,
            sequence_reference_coverage=mapping.reference_coverage,
            sequence_target_coverage=mapping.target_coverage,
            shell_residue_count=len(shell),
            mapped_shell_residue_count=len(pairs),
            local_coverage=local_coverage,
            alignment_atom_count=atom_count,
            local_rmsd_angstrom=local_rmsd,
            maximum_local_residue_rmsd_angstrom=maximum_residue_rmsd,
            global_ca_rmsd_angstrom=global_rmsd,
            reference_fe_s_distance_angstrom=reference_fe_s_distance,
            target_fe_s_distance_angstrom=target_fe_s_distance,
            minimum_cofactor_protein_distance_angstrom=minimum_distance,
        ),
        transform=None,
        shell_reference_indices=tuple(shell),
        mapped_residue_pairs=tuple(pairs),
    )


def evaluate_local_alignment(
    reference: Sequence[Residue],
    target: Sequence[Residue],
    cofactor: Sequence[Residue],
    mapping: SequenceMapping,
    settings: Mapping[str, object],
) -> AlignmentResult:
    shell = _local_shell(
        reference,
        cofactor,
        float(settings["local_shell_radius_angstrom"]),
    )
    if not shell:
        return _failed(
            "No protein residues lie within the configured cofactor shell",
            mapping,
            shell,
            [],
            0,
            local_coverage=0.0,
        )
    atom_names = [
        str(value).strip().upper()
        for value in settings["alignment_atoms"]  # type: ignore[union-attr]
    ]
    pairs: list[tuple[int, int]] = []
    reference_coordinates: list[np.ndarray] = []
    target_coordinates: list[np.ndarray] = []
    for reference_index in shell:
        target_index = mapping.reference_to_target.get(reference_index)
        if target_index is None:
            continue
        reference_atoms = [reference[reference_index].atom(name) for name in atom_names]
        target_atoms = [target[target_index].atom(name) for name in atom_names]
        if any(atom is None for atom in reference_atoms + target_atoms):
            continue
        pairs.append((reference_index, target_index))
        reference_coordinates.extend(
            atom.coordinate for atom in reference_atoms if atom is not None
        )
        target_coordinates.extend(atom.coordinate for atom in target_atoms if atom is not None)

    coverage = len(pairs) / len(shell)
    global_rmsd = _global_ca_rmsd(reference, target, mapping.reference_to_target)
    minimum_identity = float(settings["minimum_sequence_identity"])
    minimum_sequence_coverage = float(settings["minimum_sequence_coverage"])
    if mapping.identity < minimum_identity:
        return _failed(
            f"Sequence identity {mapping.identity:.3f} is below {minimum_identity:.3f}",
            mapping,
            shell,
            pairs,
            len(reference_coordinates),
            local_coverage=coverage,
            global_rmsd=global_rmsd,
        )
    if min(mapping.reference_coverage, mapping.target_coverage) < minimum_sequence_coverage:
        return _failed(
            "Sequence alignment coverage is below the configured minimum",
            mapping,
            shell,
            pairs,
            len(reference_coordinates),
            local_coverage=coverage,
            global_rmsd=global_rmsd,
        )
    minimum_residues = int(settings["minimum_local_residues"])
    if len(pairs) < minimum_residues:
        return _failed(
            f"Only {len(pairs)} local residues are usable; {minimum_residues} are required",
            mapping,
            shell,
            pairs,
            len(reference_coordinates),
            local_coverage=coverage,
            global_rmsd=global_rmsd,
        )
    minimum_coverage = float(settings["minimum_local_coverage"])
    if coverage < minimum_coverage:
        return _failed(
            f"Local coverage {coverage:.3f} is below {minimum_coverage:.3f}",
            mapping,
            shell,
            pairs,
            len(reference_coordinates),
            local_coverage=coverage,
            global_rmsd=global_rmsd,
        )

    transform = kabsch(reference_coordinates, target_coordinates)
    cutoff = float(settings["local_rmsd_cutoff_angstrom"])
    if transform.rmsd_angstrom > cutoff:
        return _failed(
            f"Local backbone RMSD {transform.rmsd_angstrom:.3f} A exceeds {cutoff:.3f} A",
            mapping,
            shell,
            pairs,
            len(reference_coordinates),
            local_coverage=coverage,
            local_rmsd=transform.rmsd_angstrom,
            global_rmsd=global_rmsd,
        )

    fitted_coordinates = transform.apply(reference_coordinates)
    target_array = np.asarray(target_coordinates, dtype=float)
    atom_count_per_residue = len(atom_names)
    residue_rmsd_values = []
    for start in range(0, len(fitted_coordinates), atom_count_per_residue):
        difference = (
            fitted_coordinates[start : start + atom_count_per_residue]
            - target_array[start : start + atom_count_per_residue]
        )
        residue_rmsd_values.append(
            float(np.sqrt(np.mean(np.sum(difference**2, axis=1))))
        )
    maximum_residue_rmsd = max(residue_rmsd_values)
    maximum_residue_cutoff = float(
        settings["maximum_local_residue_rmsd_angstrom"]
    )
    if maximum_residue_rmsd > maximum_residue_cutoff:
        return _failed(
            f"Maximum local-residue backbone RMSD {maximum_residue_rmsd:.3f} A "
            f"exceeds {maximum_residue_cutoff:.3f} A",
            mapping,
            shell,
            pairs,
            len(reference_coordinates),
            local_coverage=coverage,
            local_rmsd=transform.rmsd_angstrom,
            maximum_residue_rmsd=maximum_residue_rmsd,
            global_rmsd=global_rmsd,
        )

    reference_fe_s_distance: Optional[float] = None
    target_fe_s_distance: Optional[float] = None
    if bool(settings["require_proximal_cysteine"]):
        iron_atoms = [
            atom
            for residue in cofactor
            for atom in residue.heavy_atoms()
            if atom.element == "FE"
        ]
        if len(iron_atoms) != 1:
            return _failed(
                f"Exactly one cofactor Fe atom is required; observed {len(iron_atoms)}",
                mapping,
                shell,
                pairs,
                len(reference_coordinates),
                local_coverage=coverage,
                local_rmsd=transform.rmsd_angstrom,
                maximum_residue_rmsd=maximum_residue_rmsd,
                global_rmsd=global_rmsd,
            )
        reference_fe = iron_atoms[0].coordinate
        cysteine_candidates = [
            (index, residue.atom("SG"))
            for index, residue in enumerate(reference)
            if residue.name == "CYS" and residue.atom("SG") is not None
        ]
        if not cysteine_candidates:
            return _failed(
                "The reference protein contains no CYS SG atom for Fe-S validation",
                mapping,
                shell,
                pairs,
                len(reference_coordinates),
                local_coverage=coverage,
                local_rmsd=transform.rmsd_angstrom,
                maximum_residue_rmsd=maximum_residue_rmsd,
                global_rmsd=global_rmsd,
            )
        proximal_index, proximal_sulfur = min(
            cysteine_candidates,
            key=lambda item: float(
                np.linalg.norm(item[1].coordinate - reference_fe)  # type: ignore[union-attr]
            ),
        )
        assert proximal_sulfur is not None
        reference_fe_s_distance = float(
            np.linalg.norm(proximal_sulfur.coordinate - reference_fe)
        )
        target_index = mapping.reference_to_target.get(proximal_index)
        target_sulfur = (
            target[target_index].atom("SG") if target_index is not None else None
        )
        if (
            target_index is None
            or target[target_index].name != "CYS"
            or target_sulfur is None
        ):
            return _failed(
                "The proximal reference CYS SG atom is not preserved in the target mapping",
                mapping,
                shell,
                pairs,
                len(reference_coordinates),
                local_coverage=coverage,
                local_rmsd=transform.rmsd_angstrom,
                maximum_residue_rmsd=maximum_residue_rmsd,
                global_rmsd=global_rmsd,
                reference_fe_s_distance=reference_fe_s_distance,
            )
        transferred_fe = transform.apply(reference_fe)
        target_fe_s_distance = float(
            np.linalg.norm(target_sulfur.coordinate - transferred_fe)
        )
        minimum_fe_s = float(settings["minimum_fe_s_distance_angstrom"])
        maximum_fe_s = float(settings["maximum_fe_s_distance_angstrom"])
        if not (
            minimum_fe_s <= reference_fe_s_distance <= maximum_fe_s
            and minimum_fe_s <= target_fe_s_distance <= maximum_fe_s
        ):
            return _failed(
                "Fe-S geometry is outside the configured range: "
                f"reference={reference_fe_s_distance:.3f} A, "
                f"target={target_fe_s_distance:.3f} A, "
                f"range=[{minimum_fe_s:.3f}, {maximum_fe_s:.3f}] A",
                mapping,
                shell,
                pairs,
                len(reference_coordinates),
                local_coverage=coverage,
                local_rmsd=transform.rmsd_angstrom,
                maximum_residue_rmsd=maximum_residue_rmsd,
                global_rmsd=global_rmsd,
                reference_fe_s_distance=reference_fe_s_distance,
                target_fe_s_distance=target_fe_s_distance,
            )

    transformed_cofactor = transform.apply(
        [atom.coordinate for residue in cofactor for atom in residue.heavy_atoms()]
    )
    protein_coordinates = np.asarray(
        [atom.coordinate for residue in target for atom in residue.heavy_atoms()],
        dtype=float,
    )
    minimum_distance = float(
        np.min(
            np.linalg.norm(
                transformed_cofactor[:, None, :] - protein_coordinates[None, :, :],
                axis=2,
            )
        )
    )
    clash_cutoff = float(settings["cofactor_clash_distance_angstrom"])
    if minimum_distance < clash_cutoff:
        return _failed(
            f"Transferred cofactor clash distance {minimum_distance:.3f} A is below "
            f"{clash_cutoff:.3f} A",
            mapping,
            shell,
            pairs,
            len(reference_coordinates),
            local_coverage=coverage,
            local_rmsd=transform.rmsd_angstrom,
            maximum_residue_rmsd=maximum_residue_rmsd,
            global_rmsd=global_rmsd,
            reference_fe_s_distance=reference_fe_s_distance,
            target_fe_s_distance=target_fe_s_distance,
            minimum_distance=minimum_distance,
        )

    metrics = AlignmentMetrics(
        status="PASS",
        reason="Local-site quality criteria passed",
        sequence_identity=mapping.identity,
        sequence_reference_coverage=mapping.reference_coverage,
        sequence_target_coverage=mapping.target_coverage,
        shell_residue_count=len(shell),
        mapped_shell_residue_count=len(pairs),
        local_coverage=coverage,
        alignment_atom_count=len(reference_coordinates),
        local_rmsd_angstrom=transform.rmsd_angstrom,
        maximum_local_residue_rmsd_angstrom=maximum_residue_rmsd,
        global_ca_rmsd_angstrom=global_rmsd,
        reference_fe_s_distance_angstrom=reference_fe_s_distance,
        target_fe_s_distance_angstrom=target_fe_s_distance,
        minimum_cofactor_protein_distance_angstrom=minimum_distance,
    )
    return AlignmentResult(
        metrics=metrics,
        transform=transform,
        shell_reference_indices=tuple(shell),
        mapped_residue_pairs=tuple(pairs),
    )
