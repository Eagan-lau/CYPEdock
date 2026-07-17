from __future__ import annotations

import numpy as np
import pytest

from cypedock.alignment import evaluate_local_alignment, sequence_mapping
from cypedock.cofactor import select_cofactor, transfer_cofactor
from cypedock.errors import InputError
from cypedock.structure import Residue, ResidueID

from conftest import cofactor_residue, protein_residues


def settings(cutoff: float = 0.2) -> dict:
    return {
        "local_shell_radius_angstrom": 10.0,
        "local_rmsd_cutoff_angstrom": cutoff,
        "minimum_local_residues": 4,
        "minimum_local_coverage": 0.6,
        "minimum_sequence_identity": 0.2,
        "minimum_sequence_coverage": 0.5,
        "alignment_atoms": ["N", "CA", "C"],
        "maximum_local_residue_rmsd_angstrom": 4.0,
        "cofactor_clash_distance_angstrom": 1.0,
        "require_proximal_cysteine": True,
        "minimum_fe_s_distance_angstrom": 1.8,
        "maximum_fe_s_distance_angstrom": 3.0,
    }


def test_arbitrary_cofactor_name_and_known_transform() -> None:
    angle = np.deg2rad(31.0)
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = np.asarray([4.0, -3.0, 2.0])

    def move(values):
        return values @ rotation.T + translation

    reference = protein_residues()
    target = protein_residues(transform=move)
    cofactor = cofactor_residue()
    assert select_cofactor([*reference, cofactor], []) == [cofactor]
    mapping = sequence_mapping(reference, target)
    result = evaluate_local_alignment(reference, target, [cofactor], mapping, settings())
    assert result.passed
    assert result.metrics.local_rmsd_angstrom < 1e-6
    assert result.metrics.maximum_local_residue_rmsd_angstrom < 1e-6
    assert abs(result.metrics.reference_fe_s_distance_angstrom - 2.3) < 1e-6
    assert abs(result.metrics.target_fe_s_distance_angstrom - 2.3) < 1e-6
    transferred = transfer_cofactor([cofactor], result.transform, target)
    expected = move(np.asarray([atom.coordinate for atom in cofactor.atoms]))
    observed = np.asarray([atom.coordinate for atom in transferred[0].atoms])
    np.testing.assert_allclose(observed, expected, atol=1e-6)


def test_local_rmsd_above_cutoff_is_skipped() -> None:
    reference = protein_residues()
    target = protein_residues()
    distorted = list(target)
    residue = distorted[5]
    distorted[5] = type(residue)(
        residue.name,
        residue.residue_id,
        tuple(
            atom.moved(atom.coordinate + np.asarray([0.0, 4.0, 0.0]))
            for atom in residue.atoms
        ),
        residue.hetero,
    )
    mapping = sequence_mapping(reference, distorted)
    result = evaluate_local_alignment(
        reference,
        distorted,
        [cofactor_residue()],
        mapping,
        settings(cutoff=0.25),
    )
    assert not result.passed
    assert "exceeds" in result.metrics.reason


def test_disrupted_proximal_cysteine_geometry_is_skipped() -> None:
    reference = protein_residues()
    target = list(protein_residues())
    residue = target[5]
    target[5] = type(residue)(
        residue.name,
        residue.residue_id,
        tuple(
            atom.moved(atom.coordinate + np.asarray([0.0, 0.0, 8.0]))
            if atom.name == "SG"
            else atom
            for atom in residue.atoms
        ),
        residue.hetero,
    )
    mapping = sequence_mapping(reference, target)
    result = evaluate_local_alignment(
        reference,
        target,
        [cofactor_residue()],
        mapping,
        settings(),
    )
    assert not result.passed
    assert "Fe-S geometry" in result.metrics.reason


def test_single_residue_outlier_is_rejected_by_maximum_rmsd() -> None:
    reference = protein_residues()
    target = list(protein_residues())
    residue = target[4]
    target[4] = type(residue)(
        residue.name,
        residue.residue_id,
        tuple(
            atom.moved(atom.coordinate + np.asarray([0.0, 2.0, 0.0]))
            for atom in residue.atoms
        ),
        residue.hetero,
    )
    thresholds = settings(cutoff=10.0)
    thresholds["maximum_local_residue_rmsd_angstrom"] = 0.5
    result = evaluate_local_alignment(
        reference,
        target,
        [cofactor_residue()],
        sequence_mapping(reference, target),
        thresholds,
    )
    assert not result.passed
    assert "Maximum local-residue" in result.metrics.reason


def test_explicit_non_heme_cofactor_is_rejected() -> None:
    source = cofactor_residue()
    invalid = Residue(
        "LIG",
        ResidueID("L", 77),
        tuple(atom for atom in source.atoms if atom.element != "FE"),
        hetero=True,
    )
    with pytest.raises(InputError, match="exactly one Fe"):
        select_cofactor(
            [*protein_residues(), invalid],
            [{"chain": "L", "number": 77}],
        )
