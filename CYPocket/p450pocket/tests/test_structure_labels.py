from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from p450pocket.labels import LabelMappingError, build_labels
from p450pocket.prediction import parse_optional_structure
from p450pocket.structure import (
    MissingBackboneAtomError,
    UnsupportedResidueError,
    parse_feature_protein,
    parse_pocket,
)


def _atom_line(
    serial: int,
    atom_name: str,
    residue_name: str,
    chain_id: str,
    residue_number: int,
    xyz: tuple[float, float, float],
    *,
    insertion_code: str = "",
    altloc: str = "",
    occupancy: float = 1.0,
) -> str:
    element = atom_name.strip()[0]
    atom_field = f"{atom_name:^4}"
    return (
        f"ATOM  {serial:5d} {atom_field}{altloc or ' ':1}{residue_name:>3} "
        f"{chain_id or ' ':1}{residue_number:4d}{insertion_code or ' ':1}   "
        f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
        f"{occupancy:6.2f}{20.0:6.2f}          {element:>2}\n"
    )


def _residue_lines(
    serial: int,
    residue_name: str,
    chain_id: str,
    residue_number: int,
    base: float,
    *,
    insertion_code: str = "",
    atoms: tuple[str, ...] = ("N", "CA", "C", "O"),
) -> tuple[list[str], int]:
    offsets = {
        "N": (0.0, 0.0, 0.0),
        "CA": (1.0, 0.0, 0.0),
        "C": (2.0, 1.0, 0.0),
        "O": (2.0, 2.0, 0.0),
    }
    lines = []
    for atom_name in atoms:
        offset = offsets[atom_name]
        lines.append(
            _atom_line(
                serial,
                atom_name,
                residue_name,
                chain_id,
                residue_number,
                (base + offset[0], offset[1], offset[2]),
                insertion_code=insertion_code,
            )
        )
        serial += 1
    return lines, serial


def _write(path: Path, lines: list[str]) -> Path:
    path.write_text("".join(lines) + "TER\nEND\n", encoding="ascii")
    return path


def _three_residue_protein(path: Path) -> Path:
    lines: list[str] = []
    serial = 1
    for name, chain, number, insertion, base in (
        ("ALA", "A", 10, "", 0.0),
        ("GLY", "A", 10, "A", 10.0),
        ("SER", "B", 10, "", 20.0),
    ):
        residue, serial = _residue_lines(
            serial,
            name,
            chain,
            number,
            base,
            insertion_code=insertion,
        )
        lines.extend(residue)
    return _write(path, lines)


def test_parse_preserves_chain_insertion_order_and_nm_units(tmp_path: Path) -> None:
    path = _three_residue_protein(tmp_path / "example_protein.pdb")
    structure = parse_feature_protein(path)

    assert structure.sample_id == "example"
    assert structure.sequence == "AGS"
    assert structure.sequence_indices.tolist() == [0, 7, 15]
    assert structure.chains == ("A", "B")
    assert [
        (key.chain_id, key.auth_seq_id, key.insertion_code, key.residue_name)
        for key in structure.residue_keys
    ] == [
        ("A", 10, "", "ALA"),
        ("A", 10, "A", "GLY"),
        ("B", 10, "", "SER"),
    ]
    assert structure.backbone_coordinates_angstrom.shape == (3, 4, 3)
    np.testing.assert_allclose(
        structure.backbone_coordinates_nm,
        structure.backbone_coordinates_angstrom / 10.0,
        rtol=0.0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        structure.residues[1].ca_angstrom,
        np.asarray([11.0, 0.0, 0.0], dtype=np.float32),
    )


def test_prediction_structure_uses_the_same_chain_selection(tmp_path: Path) -> None:
    path = _three_residue_protein(tmp_path / "example_protein.pdb")
    structure = parse_optional_structure(
        path,
        expected_length=2,
        chains=["A"],
    )
    assert structure is not None
    assert structure.chains == ("A",)
    assert [key.insertion_code for key in structure.residue_keys] == ["", "A"]


def test_altloc_uses_highest_occupancy(tmp_path: Path) -> None:
    lines = [
        _atom_line(1, "N", "ALA", "A", 1, (0.0, 0.0, 0.0)),
        _atom_line(
            2,
            "CA",
            "ALA",
            "A",
            1,
            (1.0, 0.0, 0.0),
            altloc="A",
            occupancy=0.40,
        ),
        _atom_line(
            3,
            "CA",
            "ALA",
            "A",
            1,
            (9.0, 0.0, 0.0),
            altloc="B",
            occupancy=0.60,
        ),
        _atom_line(4, "C", "ALA", "A", 1, (2.0, 1.0, 0.0)),
        _atom_line(5, "O", "ALA", "A", 1, (2.0, 2.0, 0.0)),
    ]
    structure = parse_feature_protein(_write(tmp_path / "altloc.pdb", lines))
    np.testing.assert_allclose(structure.residues[0].ca_angstrom, [9.0, 0.0, 0.0])
    assert any("altloc B" in warning for warning in structure.warnings)


def test_missing_backbone_atom_is_a_hard_error(tmp_path: Path) -> None:
    lines, _ = _residue_lines(
        1, "ALA", "A", 1, 0.0, atoms=("N", "CA", "C")
    )
    path = _write(tmp_path / "missing_o.pdb", lines)
    with pytest.raises(MissingBackboneAtomError, match="Missing required atom 'O'"):
        parse_feature_protein(path)


def test_modified_residue_requires_explicit_mapping(tmp_path: Path) -> None:
    lines, _ = _residue_lines(1, "MSE", "A", 1, 0.0)
    path = _write(tmp_path / "modified.pdb", lines)
    with pytest.raises(UnsupportedResidueError, match="Unsupported polymer residue"):
        parse_feature_protein(path)

    structure = parse_feature_protein(path, modified_residue_map={"MSE": "MET"})
    assert structure.sequence == "M"
    assert structure.residues[0].key.residue_name == "MSE"
    assert structure.residues[0].canonical_name == "MET"


def test_identity_labels_follow_feature_row_order(tmp_path: Path) -> None:
    protein = parse_feature_protein(
        _three_residue_protein(tmp_path / "sample_protein.pdb")
    )
    pocket_lines = [
        _atom_line(1, "CA", "GLY", "A", 10, (11.0, 0.0, 0.0), insertion_code="A"),
        _atom_line(2, "CA", "SER", "B", 10, (21.0, 0.0, 0.0)),
    ]
    pocket = parse_pocket(_write(tmp_path / "sample_pocket.pdb", pocket_lines))

    result = build_labels(protein, pocket)
    assert result.labels.tolist() == [0, 1, 1]
    assert result.n_positive == 2
    assert [key.chain_id for key in result.positive_keys] == ["A", "B"]
    assert result.positive_keys[0].insertion_code == "A"


def test_identity_mapping_verifies_ca_coordinates(tmp_path: Path) -> None:
    protein = parse_feature_protein(
        _three_residue_protein(tmp_path / "sample_protein.pdb")
    )
    pocket = parse_pocket(
        _write(
            tmp_path / "bad_pocket.pdb",
            [_atom_line(1, "CA", "ALA", "A", 10, (5.0, 0.0, 0.0))],
        )
    )
    with pytest.raises(LabelMappingError, match="CA coordinate mismatch"):
        build_labels(protein, pocket)


def test_coordinate_fallback_is_explicit_unique_and_type_safe(tmp_path: Path) -> None:
    protein = parse_feature_protein(
        _three_residue_protein(tmp_path / "sample_protein.pdb")
    )
    pocket = parse_pocket(
        _write(
            tmp_path / "renumbered_pocket.pdb",
            [_atom_line(1, "CA", "GLY", "Z", 999, (11.0, 0.0, 0.0))],
        )
    )

    with pytest.raises(LabelMappingError, match="absent from the protein"):
        build_labels(protein, pocket)

    result = build_labels(
        protein,
        pocket,
        match_mode="identity_then_coordinate",
        coordinate_tolerance_angstrom=0.05,
    )
    assert result.labels.tolist() == [0, 1, 0]
    assert result.warnings == ("Mapped 1 pocket residue(s) using coordinate fallback",)
