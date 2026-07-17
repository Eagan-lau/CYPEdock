"""Reference-cofactor selection and coordinate transfer."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Optional, Sequence

import numpy as np

from .alignment import RigidTransform
from .errors import InputError
from .structure import Atom, Residue, ResidueID, WATER_NAMES, is_protein, protein_chains


def validate_heme_like_cofactor(cofactor: Sequence[Residue]) -> None:
    if not cofactor:
        raise InputError("The selected cofactor is empty")
    identities = [residue.residue_id for residue in cofactor]
    if len(identities) != len(set(identities)):
        raise InputError("The cofactor selection contains duplicate residues")
    atoms = [atom for residue in cofactor for atom in residue.heavy_atoms()]
    iron_count = sum(atom.element == "FE" for atom in atoms)
    nitrogen_count = sum(atom.element == "N" for atom in atoms)
    if iron_count != 1 or nitrogen_count < 4:
        raise InputError(
            "The selected CYP450 cofactor must contain exactly one Fe atom and "
            f"at least four N atoms; observed Fe={iron_count}, N={nitrogen_count}"
        )


def iron_atom(cofactor: Sequence[Residue]) -> Atom:
    atoms = [
        atom
        for residue in cofactor
        for atom in residue.heavy_atoms()
        if atom.element == "FE"
    ]
    if len(atoms) != 1:
        raise InputError(f"Exactly one cofactor Fe atom is required; observed {len(atoms)}")
    return atoms[0]


def select_cofactor(
    residues: Sequence[Residue],
    selectors: Sequence[Mapping[str, object]],
) -> list[Residue]:
    if selectors:
        selected: list[Residue] = []
        for selector in selectors:
            chain = str(selector.get("chain", "") or "")
            number = int(selector["number"])
            insertion = str(selector.get("insertion_code", "") or "")
            name = str(selector.get("name", "") or "").strip().upper()
            matches = [
                residue
                for residue in residues
                if residue.residue_id.chain == chain
                and residue.residue_id.number == number
                and residue.residue_id.insertion_code == insertion
                and (not name or residue.name == name)
            ]
            if len(matches) != 1:
                raise InputError(
                    f"Cofactor selector {chain or '_'}:{number}{insertion} matched "
                    f"{len(matches)} residues"
                )
            if is_protein(matches[0]) or matches[0].name in WATER_NAMES:
                raise InputError("A cofactor selector resolved to protein or water")
            selected.append(matches[0])
        validate_heme_like_cofactor(selected)
        return selected

    candidates: list[Residue] = []
    for residue in residues:
        if is_protein(residue) or residue.name in WATER_NAMES:
            continue
        elements = [atom.element for atom in residue.heavy_atoms()]
        if elements.count("FE") == 1 and elements.count("N") >= 4:
            candidates.append(residue)
    if len(candidates) != 1:
        labels = [f"{residue.name}:{residue.residue_id.label()}" for residue in candidates]
        raise InputError(
            "Automatic cofactor selection requires exactly one non-protein residue "
            f"containing one Fe and at least four N atoms; candidates={labels}"
        )
    validate_heme_like_cofactor(candidates)
    return candidates


def select_reference_chain(
    residues: Sequence[Residue],
    cofactor: Sequence[Residue],
    requested: Optional[str],
) -> list[Residue]:
    chains = protein_chains(residues)
    if requested is not None:
        if requested not in chains:
            raise InputError(f"Reference protein chain {requested!r} is unavailable: {chains}")
        return [
            residue
            for residue in residues
            if is_protein(residue) and residue.residue_id.chain == requested
        ]
    if len(chains) == 1:
        return [
            residue
            for residue in residues
            if is_protein(residue) and residue.residue_id.chain == chains[0]
        ]
    cofactor_coordinates = np.asarray(
        [atom.coordinate for residue in cofactor for atom in residue.heavy_atoms()],
        dtype=float,
    )
    contact_counts: dict[str, int] = {}
    for chain in chains:
        count = 0
        for residue in residues:
            if not is_protein(residue) or residue.residue_id.chain != chain:
                continue
            coordinates = np.asarray(
                [atom.coordinate for atom in residue.heavy_atoms()], dtype=float
            )
            if float(
                np.min(
                    np.linalg.norm(
                        coordinates[:, None, :] - cofactor_coordinates[None, :, :], axis=2
                    )
                )
            ) <= 6.0:
                count += 1
        contact_counts[chain] = count
    if not contact_counts:
        raise InputError("The reference contains no protein chain")
    maximum = max(contact_counts.values())
    winners = sorted(chain for chain, count in contact_counts.items() if count == maximum)
    if maximum == 0 or len(winners) != 1:
        raise InputError(
            "Reference protein chain is ambiguous; specify reference.protein_chain "
            f"(contact counts: {contact_counts})"
        )
    selected_chain = winners[0]
    return [
        residue
        for residue in residues
        if is_protein(residue) and residue.residue_id.chain == selected_chain
    ]


def transfer_cofactor(
    cofactor: Sequence[Residue],
    transform: RigidTransform,
    target_protein: Sequence[Residue],
) -> list[Residue]:
    occupied = {
        (residue.residue_id.chain, residue.residue_id.number, residue.residue_id.insertion_code)
        for residue in target_protein
    }
    preferred_chains = [residue.residue_id.chain for residue in cofactor]
    invalid_chain = any(
        len(chain) != 1 or not chain.strip() or not chain.isascii()
        for chain in preferred_chains
    )
    identifier_collision = any(
        (
            residue.residue_id.chain,
            residue.residue_id.number,
            residue.residue_id.insertion_code,
        )
        in occupied
        for residue in cofactor
    )
    if invalid_chain or identifier_collision:
        available = next(
            (
                chain
                for chain in reversed("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                if all(
                    (
                        chain,
                        residue.residue_id.number,
                        residue.residue_id.insertion_code,
                    )
                    not in occupied
                    for residue in cofactor
                )
            ),
            None,
        )
        if available is None:
            raise InputError("No one-character chain ID is available for the transferred cofactor")
        preferred_chains = [available] * len(cofactor)

    transferred: list[Residue] = []
    for residue, output_chain in zip(cofactor, preferred_chains):
        coordinates = np.asarray([atom.coordinate for atom in residue.atoms], dtype=float)
        moved = transform.apply(coordinates)
        transferred.append(
            replace(
                residue,
                residue_id=ResidueID(
                    chain=output_chain,
                    number=residue.residue_id.number,
                    insertion_code=residue.residue_id.insertion_code,
                ),
                atoms=tuple(
                    atom.moved(coordinate)
                    for atom, coordinate in zip(residue.atoms, moved)
                ),
                hetero=True,
            )
        )
    return transferred
