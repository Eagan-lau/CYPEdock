"""Deterministic PDB/mmCIF parsing and normalized PDB writing."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser

from .errors import InputError


AMINO_ACIDS = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
WATER_NAMES = {"HOH", "WAT", "DOD"}
STRUCTURE_SUFFIXES = {".pdb", ".ent", ".cif", ".mmcif"}


def _guess_element(name: str) -> str:
    value = str(name).strip().upper()
    while value and value[0].isdigit():
        value = value[1:]
    if not value:
        return ""
    two_letter = {
        "BR",
        "CL",
        "CO",
        "CU",
        "FE",
        "MG",
        "MN",
        "MO",
        "NA",
        "NI",
        "SE",
        "SI",
        "ZN",
    }
    return value[:2] if value[:2] in two_letter else value[0]


@dataclass(frozen=True, order=True)
class ResidueID:
    chain: str
    number: int
    insertion_code: str = ""

    def label(self) -> str:
        return f"{self.chain or '_'}:{self.number}{self.insertion_code}"


@dataclass(frozen=True, eq=False)
class Atom:
    name: str
    coordinate: np.ndarray
    element: str
    occupancy: float = 1.0
    bfactor: float = 0.0

    def __post_init__(self) -> None:
        name = str(self.name).strip().upper()
        coordinate = np.asarray(self.coordinate, dtype=float)
        if not name or coordinate.shape != (3,) or not np.all(np.isfinite(coordinate)):
            raise InputError("Atom names and three finite coordinates are required")
        coordinate = coordinate.copy()
        coordinate.setflags(write=False)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "coordinate", coordinate)
        element = str(self.element).strip().upper() or _guess_element(name)
        object.__setattr__(self, "element", element)
        object.__setattr__(self, "occupancy", float(self.occupancy))
        object.__setattr__(self, "bfactor", float(self.bfactor))

    @property
    def is_hydrogen(self) -> bool:
        return self.element in {"H", "D"}

    def moved(self, coordinate: Sequence[float]) -> "Atom":
        return replace(self, coordinate=np.asarray(coordinate, dtype=float))


@dataclass(frozen=True)
class Residue:
    name: str
    residue_id: ResidueID
    atoms: tuple[Atom, ...]
    hetero: bool = False

    def __post_init__(self) -> None:
        name = str(self.name).strip().upper()
        atoms = tuple(self.atoms)
        if not name or not atoms:
            raise InputError("Residues require a name and at least one atom")
        names = [atom.name for atom in atoms]
        if len(names) != len(set(names)):
            raise InputError(f"Duplicate atom names in {name} {self.residue_id.label()}")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "atoms", atoms)

    def atom(self, name: str) -> Optional[Atom]:
        wanted = name.strip().upper()
        return next((atom for atom in self.atoms if atom.name == wanted), None)

    def heavy_atoms(self) -> tuple[Atom, ...]:
        return tuple(atom for atom in self.atoms if not atom.is_hydrogen)

    def moved(self, coordinates: np.ndarray, *, chain: Optional[str] = None) -> "Residue":
        if coordinates.shape != (len(self.atoms), 3):
            raise InputError("Transferred residue coordinate count does not match")
        residue_id = self.residue_id
        if chain is not None:
            residue_id = replace(residue_id, chain=chain)
        return replace(
            self,
            residue_id=residue_id,
            atoms=tuple(atom.moved(coord) for atom, coord in zip(self.atoms, coordinates)),
            hetero=True,
        )


@dataclass(frozen=True)
class ConformationSource:
    state_id: str
    path: Path
    model_index: int


def is_protein(residue: Residue) -> bool:
    return residue.name in AMINO_ACIDS and not residue.hetero


def protein_sequence(residues: Sequence[Residue]) -> str:
    return "".join(AMINO_ACIDS[residue.name] for residue in residues if is_protein(residue))


def protein_chains(residues: Sequence[Residue]) -> list[str]:
    return sorted({residue.residue_id.chain for residue in residues if is_protein(residue)})


def select_protein_chain(residues: Sequence[Residue], requested: Optional[str]) -> list[Residue]:
    chains = protein_chains(residues)
    if requested is None:
        if len(chains) != 1:
            raise InputError(
                f"Exactly one protein chain is required when no chain is specified; found {chains}"
            )
        requested = chains[0]
    if requested not in chains:
        raise InputError(f"Protein chain {requested!r} is unavailable; found {chains}")
    selected = [
        residue
        for residue in residues
        if is_protein(residue) and residue.residue_id.chain == requested
    ]
    if not selected:
        raise InputError(f"Protein chain {requested!r} contains no standard amino acids")
    return selected


def remap_chain(residues: Sequence[Residue], chain: str) -> list[Residue]:
    """Return residues with a PDB-compatible chain identifier."""
    if len(chain) != 1 or not chain.strip() or not chain.isascii():
        raise InputError(f"Output chain must be one non-blank ASCII character: {chain!r}")
    return [
        replace(
            residue,
            residue_id=replace(residue.residue_id, chain=chain),
        )
        for residue in residues
    ]


def _format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".pdb", ".ent"}:
        return "pdb"
    if suffix in {".cif", ".mmcif"}:
        return "mmcif"
    raise InputError(f"Unsupported structure format: {path}")


def _parser(path: Path):
    return (
        PDBParser(QUIET=True, PERMISSIVE=True)
        if _format(path) == "pdb"
        else MMCIFParser(QUIET=True)
    )


def _bio_structure(path: Path):
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise InputError(f"Structure file does not exist: {source}")
    try:
        return _parser(source).get_structure(source.stem, str(source))
    except Exception as exc:
        raise InputError(f"Failed to parse structure {source}: {exc}") from exc


def model_count(path: Path) -> int:
    count = len(list(_bio_structure(path).get_models()))
    if count < 1:
        raise InputError(f"Structure contains no models: {path}")
    return count


def _selected_atoms(bio_residue: object) -> list[object]:
    try:
        candidates = list(bio_residue.get_unpacked_list())  # type: ignore[attr-defined]
    except AttributeError:
        candidates = list(bio_residue)  # type: ignore[arg-type]
    grouped: dict[str, list[object]] = {}
    for atom in candidates:
        grouped.setdefault(atom.get_name().strip().upper(), []).append(atom)

    def rank(atom: object) -> tuple[float, int, str]:
        occupancy = atom.get_occupancy()  # type: ignore[attr-defined]
        altloc = str(atom.get_altloc() or "").strip()  # type: ignore[attr-defined]
        preference = 2 if not altloc else (1 if altloc == "A" else 0)
        return (float(occupancy or 0.0), preference, altloc)

    return [max(grouped[name], key=rank) for name in sorted(grouped)]


def read_structure(path: Path, *, model_index: int = 0) -> list[Residue]:
    source = Path(path).expanduser().resolve()
    structure = _bio_structure(source)
    models = list(structure.get_models())
    if model_index < 0 or model_index >= len(models):
        raise InputError(
            f"Model index {model_index} is invalid for {source}; model count is {len(models)}"
        )
    records: list[Residue] = []
    for chain in models[model_index]:
        chain_id = str(chain.id).strip()
        for bio_residue in chain:
            hetero_flag, number, insertion_code = bio_residue.id
            atoms: list[Atom] = []
            for bio_atom in _selected_atoms(bio_residue):
                atoms.append(
                    Atom(
                        name=bio_atom.get_name(),
                        coordinate=np.asarray(bio_atom.get_coord(), dtype=float),
                        element=getattr(bio_atom, "element", ""),
                        occupancy=bio_atom.get_occupancy() or 1.0,
                        bfactor=bio_atom.get_bfactor() or 0.0,
                    )
                )
            if atoms:
                records.append(
                    Residue(
                        name=bio_residue.get_resname(),
                        residue_id=ResidueID(
                            chain=chain_id,
                            number=int(number),
                            insertion_code=str(insertion_code or "").strip(),
                        ),
                        atoms=tuple(atoms),
                        hetero=str(hetero_flag or "").strip() not in {"", " "},
                    )
                )
    if not records:
        raise InputError(f"Structure contains no atoms: {source}")
    return records


def _atom_name_field(name: str, element: str) -> str:
    if len(name) >= 4 or (name and name[0].isdigit()):
        return f"{name[:4]:<4}"
    if len(element) == 2:
        return f"{name:<4}"
    return f" {name:<3}"


def write_pdb(residues: Sequence[Residue], path: Path) -> Path:
    records = list(residues)
    if not records:
        raise InputError("Cannot write an empty PDB structure")
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    serial = 1
    with temporary.open("w", encoding="ascii") as handle:
        for residue in records:
            chain = residue.residue_id.chain or " "
            if len(chain) != 1:
                raise InputError(f"PDB output requires a one-character chain ID: {chain!r}")
            record_name = "ATOM" if is_protein(residue) else "HETATM"
            for atom in residue.atoms:
                x, y, z = atom.coordinate
                atom_name = _atom_name_field(atom.name, atom.element)
                handle.write(
                    f"{record_name:<6}{serial:5d} {atom_name} {residue.name:>3} "
                    f"{chain}{residue.residue_id.number:4d}"
                    f"{residue.residue_id.insertion_code[:1] or ' '}   "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}"
                    f"{atom.occupancy:6.2f}{atom.bfactor:6.2f}"
                    f"          {atom.element:>2}\n"
                )
                serial += 1
        handle.write("END\n")
    os.replace(temporary, destination)
    return destination


def _safe_state_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not normalized:
        raise InputError(f"Cannot derive a state identifier from {value!r}")
    return normalized


def _manifest_paths(path: Path) -> Iterator[Path]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                candidate = path.parent / candidate
            candidate = candidate.resolve()
            if candidate.suffix.lower() not in STRUCTURE_SUFFIXES:
                raise InputError(
                    f"Unsupported structure in ensemble manifest at line {line_number}: {candidate}"
                )
            yield candidate


def discover_ensemble(path: Path) -> list[ConformationSource]:
    source = Path(path).expanduser().resolve()
    if source.is_dir():
        files = sorted(
            item.resolve()
            for item in source.iterdir()
            if item.is_file() and item.suffix.lower() in STRUCTURE_SUFFIXES
        )
    elif source.is_file() and source.suffix.lower() in {".txt", ".list"}:
        files = list(_manifest_paths(source))
    elif source.is_file() and source.suffix.lower() in STRUCTURE_SUFFIXES:
        files = [source]
    else:
        raise InputError(
            "The ensemble input must be a structure, a directory, or a .txt/.list manifest"
        )
    if not files:
        raise InputError(f"No supported structures were found in the ensemble: {source}")

    discovered: list[ConformationSource] = []
    identifiers: set[str] = set()
    for file_index, structure_path in enumerate(files, start=1):
        count = model_count(structure_path)
        for model_index in range(count):
            base = _safe_state_id(structure_path.stem)
            state_id = base if count == 1 else f"{base}_m{model_index + 1:04d}"
            if state_id in identifiers:
                state_id = f"{state_id}_f{file_index:04d}"
            if state_id in identifiers:
                raise InputError(f"Duplicate state identifier: {state_id}")
            identifiers.add(state_id)
            discovered.append(
                ConformationSource(
                    state_id=state_id,
                    path=structure_path,
                    model_index=model_index,
                )
            )
    return discovered
