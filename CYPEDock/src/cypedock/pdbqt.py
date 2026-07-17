"""Self-contained PDBQT writers for Vina scoring."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Sequence

from rdkit import Chem
from rdkit.Chem import Lipinski

from .errors import InputError
from .structure import Residue, is_protein


_SUPPORTED_TYPES = {
    "H",
    "HD",
    "C",
    "A",
    "N",
    "NA",
    "O",
    "OA",
    "F",
    "P",
    "S",
    "SA",
    "Cl",
    "Br",
    "I",
    "Si",
    "B",
    "Mg",
    "Mn",
    "Zn",
    "Ca",
    "Fe",
}

_AROMATIC_PROTEIN_ATOMS = {
    "PHE": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TYR": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TRP": {"CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "HIS": {"CG", "CD2", "CE1"},
}
_PORPHYRIN_AROMATIC_CARBONS = {
    "CHA",
    "CHB",
    "CHC",
    "CHD",
    *(f"C{position}{ring}" for ring in "ABCD" for position in range(1, 5)),
}


def _line(
    *,
    record: str,
    serial: int,
    name: str,
    residue_name: str,
    chain: str,
    residue_number: int,
    x: float,
    y: float,
    z: float,
    charge: float,
    atom_type: str,
) -> str:
    if atom_type not in _SUPPORTED_TYPES:
        raise InputError(f"Unsupported AutoDock atom type: {atom_type}")
    return (
        f"{record:<6}{serial:5d} {name[:4]:>4} {residue_name[:3]:>3} "
        f"{(chain or ' ')[:1]}{residue_number:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{0.00:6.2f}"
        f"    {charge:6.3f} {atom_type:<2}\n"
    )


def _receptor_type(residue: Residue, atom_name: str, element: str) -> str:
    element = element.upper()
    if element in {"H", "D"}:
        hydrogen = residue.atom(atom_name)
        if hydrogen is None:
            return "H"
        heavy_atoms = residue.heavy_atoms()
        if not heavy_atoms:
            return "H"
        parent = min(
            heavy_atoms,
            key=lambda atom: float(sum((atom.coordinate - hydrogen.coordinate) ** 2)),
        )
        return "HD" if parent.element in {"N", "O", "S"} else "H"
    if element == "C":
        if atom_name in _AROMATIC_PROTEIN_ATOMS.get(residue.name, set()):
            return "A"
        if not is_protein(residue):
            return "A" if atom_name in _PORPHYRIN_AROMATIC_CARBONS else "C"
        return "C"
    if element == "N":
        if residue.name == "HIS" and atom_name in {"ND1", "NE2"}:
            return "NA"
        return "N"
    if element == "O":
        return "OA"
    if element == "S":
        return "S"
    translations = {
        "F": "F",
        "P": "P",
        "CL": "Cl",
        "BR": "Br",
        "I": "I",
        "SI": "Si",
        "B": "B",
        "MG": "Mg",
        "MN": "Mn",
        "ZN": "Zn",
        "CA": "Ca",
        "FE": "Fe",
    }
    if element not in translations:
        raise InputError(
            f"No receptor AutoDock type is defined for element {element} "
            f"({residue.name} {residue.residue_id.label()} {atom_name})"
        )
    return translations[element]


def write_receptor_pdbqt(residues: Sequence[Residue], path: Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "REMARK CYPEdock receptor typing policy 1\n",
        "REMARK Partial charges are zero under the Vina scoring function\n",
    ]
    serial = 1
    for residue in residues:
        for atom in residue.atoms:
            atom_type = _receptor_type(residue, atom.name, atom.element)
            x, y, z = atom.coordinate
            lines.append(
                _line(
                    record="ATOM" if is_protein(residue) else "HETATM",
                    serial=serial,
                    name=atom.name,
                    residue_name=residue.name,
                    chain=residue.residue_id.chain,
                    residue_number=residue.residue_id.number,
                    x=float(x),
                    y=float(y),
                    z=float(z),
                    charge=0.0,
                    atom_type=atom_type,
                )
            )
            serial += 1
    if serial == 1:
        raise InputError("Cannot write an empty receptor PDBQT")
    lines.append("TER\n")
    destination.write_text("".join(lines), encoding="ascii")
    return destination


def _ligand_type(atom: Chem.Atom) -> str:
    symbol = atom.GetSymbol()
    if symbol == "H":
        return "HD"
    if symbol == "C":
        return "A" if atom.GetIsAromatic() else "C"
    if symbol == "N":
        if atom.GetIsAromatic():
            return "N" if atom.GetTotalNumHs(includeNeighbors=True) > 0 else "NA"
        if atom.GetFormalCharge() <= 0 and atom.GetTotalValence() <= 3:
            amide = any(
                neighbor.GetSymbol() == "C"
                and any(
                    bond.GetBondType() == Chem.BondType.DOUBLE
                    and bond.GetOtherAtom(neighbor).GetSymbol() == "O"
                    for bond in neighbor.GetBonds()
                )
                for neighbor in atom.GetNeighbors()
            )
            conjugated_to_aromatic_ring = any(
                neighbor.GetIsAromatic() for neighbor in atom.GetNeighbors()
            )
            return "N" if amide or conjugated_to_aromatic_ring else "NA"
        return "N"
    if symbol == "O":
        return "OA"
    if symbol == "S":
        return "SA" if atom.GetFormalCharge() <= 0 else "S"
    translations = {
        "F": "F",
        "P": "P",
        "Cl": "Cl",
        "Br": "Br",
        "I": "I",
        "Si": "Si",
        "B": "B",
        "Mg": "Mg",
        "Mn": "Mn",
        "Zn": "Zn",
        "Ca": "Ca",
        "Fe": "Fe",
    }
    if symbol not in translations:
        raise InputError(f"No ligand AutoDock type is defined for element {symbol}")
    return translations[symbol]


def _charge(atom: Chem.Atom) -> float:
    if not atom.HasProp("_GasteigerCharge"):
        raise InputError(f"Atom {atom.GetIdx()} has no Gasteiger charge")
    try:
        value = float(atom.GetProp("_GasteigerCharge"))
    except ValueError as exc:
        raise InputError(f"Atom {atom.GetIdx()} has an invalid Gasteiger charge") from exc
    if not math.isfinite(value):
        raise InputError(f"Atom {atom.GetIdx()} has a non-finite Gasteiger charge")
    return value


def _retained_atoms(molecule: Chem.Mol) -> tuple[list[int], dict[int, float]]:
    retained: list[int] = []
    charges = {atom.GetIdx(): _charge(atom) for atom in molecule.GetAtoms()}
    for atom in molecule.GetAtoms():
        index = atom.GetIdx()
        if atom.GetAtomicNum() != 1:
            retained.append(index)
            continue
        neighbors = list(atom.GetNeighbors())
        if len(neighbors) == 1 and neighbors[0].GetAtomicNum() in {7, 8, 16}:
            retained.append(index)
        elif len(neighbors) == 1:
            charges[neighbors[0].GetIdx()] += charges[index]
    observed_charge = sum(charges[index] for index in retained)
    formal_charge = float(Chem.GetFormalCharge(molecule))
    if not math.isclose(observed_charge, formal_charge, abs_tol=0.05):
        raise InputError(
            "Retained PDBQT charges do not reproduce the molecular formal charge: "
            f"observed={observed_charge:.4f}, expected={formal_charge:.4f}"
        )
    return retained, charges


def _rotatable_bonds(molecule: Chem.Mol, retained: set[int]) -> set[tuple[int, int]]:
    matches = molecule.GetSubstructMatches(Lipinski.RotatableBondSmarts)
    bonds: set[tuple[int, int]] = set()
    for match in matches:
        if len(match) < 2:
            continue
        first, second = int(match[0]), int(match[1])
        if first in retained and second in retained:
            bonds.add(tuple(sorted((first, second))))
    return bonds


def _components(
    molecule: Chem.Mol,
    retained: set[int],
    rotatable: set[tuple[int, int]],
) -> tuple[list[set[int]], dict[int, int]]:
    adjacency: dict[int, set[int]] = {index: set() for index in retained}
    for bond in molecule.GetBonds():
        first, second = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if first not in retained or second not in retained:
            continue
        if tuple(sorted((first, second))) in rotatable:
            continue
        adjacency[first].add(second)
        adjacency[second].add(first)
    components: list[set[int]] = []
    index_to_component: dict[int, int] = {}
    for start in sorted(retained):
        if start in index_to_component:
            continue
        component: set[int] = set()
        queue = deque([start])
        while queue:
            current = queue.popleft()
            if current in component:
                continue
            component.add(current)
            queue.extend(sorted(adjacency[current] - component))
        component_id = len(components)
        components.append(component)
        for index in component:
            index_to_component[index] = component_id
    return components, index_to_component


def write_ligand_pdbqt(molecule: Chem.Mol, ligand_id: str, path: Path) -> Path:
    if molecule.GetNumConformers() != 1:
        raise InputError("Ligand PDBQT preparation requires exactly one conformer")
    retained_list, charges = _retained_atoms(molecule)
    retained = set(retained_list)
    rotatable = _rotatable_bonds(molecule, retained)
    components, component_of = _components(molecule, retained, rotatable)
    if not components:
        raise InputError("Ligand contains no PDBQT atoms")
    component_edges: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for first, second in rotatable:
        first_component = component_of[first]
        second_component = component_of[second]
        if first_component == second_component:
            continue
        component_edges[first_component].append((second_component, first, second))
        component_edges[second_component].append((first_component, second, first))
    root = min(
        range(len(components)),
        key=lambda index: (-len(components[index]), min(components[index])),
    )
    serial_by_atom = {
        atom_index: serial for serial, atom_index in enumerate(sorted(retained), start=1)
    }
    conformer = molecule.GetConformer()

    def atom_line(atom_index: int) -> str:
        atom = molecule.GetAtomWithIdx(atom_index)
        position = conformer.GetAtomPosition(atom_index)
        name = f"{atom.GetSymbol()}{atom_index + 1}"
        return _line(
            record="HETATM",
            serial=serial_by_atom[atom_index],
            name=name,
            residue_name="LIG",
            chain="L",
            residue_number=1,
            x=position.x,
            y=position.y,
            z=position.z,
            charge=charges[atom_index],
            atom_type=_ligand_type(atom),
        )

    lines = [
        f"REMARK CYPEdock ligand {ligand_id}\n",
        f"REMARK SMILES {Chem.MolToSmiles(Chem.RemoveHs(molecule), isomericSmiles=True)}\n",
        "ROOT\n",
    ]
    for atom_index in sorted(components[root]):
        lines.append(atom_line(atom_index))
    lines.append("ENDROOT\n")
    visited = {root}

    def write_children(component: int) -> None:
        for child, parent_atom, child_atom in sorted(
            component_edges.get(component, []),
            key=lambda edge: (edge[0], edge[1], edge[2]),
        ):
            if child in visited:
                continue
            visited.add(child)
            parent_serial = serial_by_atom[parent_atom]
            child_serial = serial_by_atom[child_atom]
            lines.append(f"BRANCH {parent_serial:3d} {child_serial:3d}\n")
            order = [child_atom] + sorted(components[child] - {child_atom})
            for atom_index in order:
                lines.append(atom_line(atom_index))
            write_children(child)
            lines.append(f"ENDBRANCH {parent_serial:3d} {child_serial:3d}\n")

    write_children(root)
    if len(visited) != len(components):
        raise InputError("Ligand contains disconnected rigid components")
    lines.append(f"TORSDOF {len(rotatable)}\n")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(lines), encoding="ascii")
    return destination
