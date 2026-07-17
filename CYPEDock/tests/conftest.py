from __future__ import annotations

from pathlib import Path

import numpy as np

from cypedock.structure import Atom, Residue, ResidueID, write_pdb


def protein_residues(count: int = 10, transform=None) -> list[Residue]:
    residues: list[Residue] = []
    for index in range(count):
        ca = np.asarray(
            [index * 1.5, np.sin(index * 0.7), np.cos(index * 0.4) * 0.5],
            dtype=float,
        )
        coordinates = {
            "N": ca + np.asarray([-0.55, 0.20, 0.10]),
            "CA": ca,
            "C": ca + np.asarray([0.60, 0.35, -0.10]),
            "O": ca + np.asarray([0.85, 1.10, -0.15]),
            "CB": ca + np.asarray([0.05, -1.20, 0.65]),
        }
        residue_name = "ALA"
        if index == 5:
            residue_name = "CYS"
            coordinates["SG"] = np.asarray([7.0, 5.5, -2.3], dtype=float)
        if transform is not None:
            coordinates = {
                name: transform(value[None, :])[0] for name, value in coordinates.items()
            }
        elements = {
            "N": "N",
            "CA": "C",
            "C": "C",
            "O": "O",
            "CB": "C",
            "SG": "S",
        }
        residues.append(
            Residue(
                residue_name,
                ResidueID("A", index + 1),
                tuple(
                    Atom(name, coordinate, elements[name])
                    for name, coordinate in coordinates.items()
                ),
            )
        )
    return residues


def cofactor_residue(transform=None) -> Residue:
    center = np.asarray([7.0, 5.5, 0.0])
    atom_values = [
        ("FE", center, "FE"),
        ("NA", center + np.asarray([1.9, 0.0, 0.0]), "N"),
        ("NB", center + np.asarray([-1.9, 0.0, 0.0]), "N"),
        ("NC", center + np.asarray([0.0, 1.9, 0.0]), "N"),
        ("ND", center + np.asarray([0.0, -1.9, 0.0]), "N"),
        ("C1", center + np.asarray([2.7, 0.8, 0.0]), "C"),
        ("C2", center + np.asarray([-2.7, -0.8, 0.0]), "C"),
    ]
    if transform is not None:
        atom_values = [
            (name, transform(coordinate[None, :])[0], element)
            for name, coordinate, element in atom_values
        ]
    return Residue(
        "XYZ",
        ResidueID("H", 501),
        tuple(Atom(name, coordinate, element) for name, coordinate, element in atom_values),
        hetero=True,
    )


def write_synthetic_pair(directory: Path) -> tuple[Path, Path]:
    target = directory / "target.pdb"
    reference = directory / "reference_holo.pdb"
    write_pdb(protein_residues(), target)
    write_pdb([*protein_residues(), cofactor_residue()], reference)
    return target, reference
