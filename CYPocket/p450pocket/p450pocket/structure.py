"""Strict, deterministic parsing of protein PDB files.

The downstream PocketMiner checkpoint was trained with four backbone atoms in
the order ``N, CA, C, O`` and with coordinates expressed in nanometres.  This
module makes that otherwise implicit contract explicit and carries a stable
residue key alongside every feature row.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa


BACKBONE_ATOMS: Tuple[str, ...] = ("N", "CA", "C", "O")

THREE_TO_ONE: Mapping[str, str] = MappingProxyType(
    {
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
)

# CYM was explicitly accepted by the legacy feature script.  Other modified
# residues must be opted into by the caller so that an input distribution
# change is never silent.
LEGACY_MODIFIED_RESIDUES: Mapping[str, str] = MappingProxyType({"CYM": "CYS"})

# PocketMiner's exact amino-acid embedding order.
LEGACY_AA_INDEX: Mapping[str, int] = MappingProxyType(
    {
        "A": 0,
        "R": 1,
        "N": 2,
        "D": 3,
        "C": 4,
        "Q": 5,
        "E": 6,
        "G": 7,
        "H": 8,
        "I": 9,
        "L": 10,
        "K": 11,
        "M": 12,
        "F": 13,
        "P": 14,
        "S": 15,
        "T": 16,
        "W": 17,
        "Y": 18,
        "V": 19,
    }
)


class StructureParseError(ValueError):
    """Base class for invalid or unsupported protein structures."""


class UnsupportedResidueError(StructureParseError):
    """Raised when an ATOM residue cannot be mapped to the legacy alphabet."""


class MissingBackboneAtomError(StructureParseError):
    """Raised when a residue lacks an atom required by the caller."""


@dataclass(frozen=True, order=True)
class ResidueKey:
    """Unambiguous identity of one polymer residue.

    ``auth_seq_id`` and ``insertion_code`` are the author-provided PDB fields;
    they must not be replaced by an array index.  Blank chain and insertion
    identifiers are represented by the empty string.
    """

    model_index: int
    chain_id: str
    auth_seq_id: int
    insertion_code: str
    residue_name: str

    @property
    def position(self) -> Tuple[int, str, int, str]:
        """Identity excluding residue name, useful for mismatch diagnostics."""

        return (
            self.model_index,
            self.chain_id,
            self.auth_seq_id,
            self.insertion_code,
        )

    def display(self) -> str:
        insertion = self.insertion_code or ""
        chain = self.chain_id or "_"
        return f"model={self.model_index} chain={chain} {self.residue_name} " \
            f"{self.auth_seq_id}{insertion}"


@dataclass(frozen=True)
class ResidueRecord:
    """A parsed residue and the exact atoms selected for numerical use."""

    key: ResidueKey
    canonical_name: str
    one_letter: str
    row_index: int
    atom_coordinates_angstrom: Mapping[str, np.ndarray] = field(
        repr=False, compare=False
    )

    def __post_init__(self) -> None:
        copied: Dict[str, np.ndarray] = {}
        for name, value in self.atom_coordinates_angstrom.items():
            coordinate = np.asarray(value, dtype=np.float32).copy()
            if coordinate.shape != (3,):
                raise StructureParseError(
                    f"Atom {name!r} for {self.key.display()} has shape "
                    f"{coordinate.shape}, expected (3,)"
                )
            if not np.isfinite(coordinate).all():
                raise StructureParseError(
                    f"Atom {name!r} for {self.key.display()} has non-finite coordinates"
                )
            coordinate.setflags(write=False)
            copied[name] = coordinate
        object.__setattr__(self, "atom_coordinates_angstrom", MappingProxyType(copied))

    def atom(self, name: str) -> np.ndarray:
        try:
            return self.atom_coordinates_angstrom[name]
        except KeyError as exc:
            raise MissingBackboneAtomError(
                f"{self.key.display()} does not contain atom {name!r}"
            ) from exc

    @property
    def ca_angstrom(self) -> np.ndarray:
        return self.atom("CA")

    @property
    def backbone_angstrom(self) -> np.ndarray:
        return np.stack([self.atom(name) for name in BACKBONE_ATOMS]).astype(
            np.float32, copy=False
        )


@dataclass(frozen=True)
class StructureRecord:
    """Protein residues in the precise order used by feature and label arrays."""

    source_path: Path
    sample_id: str
    model_index: int
    residues: Tuple[ResidueRecord, ...]
    warnings: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.residues:
            raise StructureParseError(f"No supported protein residues in {self.source_path}")
        expected = list(range(len(self.residues)))
        actual = [residue.row_index for residue in self.residues]
        if actual != expected:
            raise StructureParseError(
                f"Non-contiguous residue row indices in {self.source_path}: {actual[:10]}"
            )

    def __len__(self) -> int:
        return len(self.residues)

    @property
    def residue_keys(self) -> Tuple[ResidueKey, ...]:
        return tuple(residue.key for residue in self.residues)

    @property
    def chains(self) -> Tuple[str, ...]:
        return tuple(dict.fromkeys(residue.key.chain_id for residue in self.residues))

    @property
    def sequence(self) -> str:
        return "".join(residue.one_letter for residue in self.residues)

    @property
    def sequence_indices(self) -> np.ndarray:
        values = np.asarray(
            [LEGACY_AA_INDEX[residue.one_letter] for residue in self.residues],
            dtype=np.int32,
        )
        values.setflags(write=False)
        return values

    @property
    def backbone_coordinates_angstrom(self) -> np.ndarray:
        return np.stack([residue.backbone_angstrom for residue in self.residues]).astype(
            np.float32, copy=False
        )

    @property
    def backbone_coordinates_nm(self) -> np.ndarray:
        # MDTraj, used by the legacy code, parses the PDB's 8.3 coordinate
        # fields in double precision, converts Å -> nm, then stores float32.
        # Bio.PDB stores float32 Å first; dividing that array directly changes
        # a few last bits. Restoring the three-decimal PDB representation before
        # conversion reproduces MDTraj exactly without coupling this module to it.
        pdb_angstrom = np.round(
            self.backbone_coordinates_angstrom.astype(np.float64), decimals=3
        )
        return np.asarray(pdb_angstrom / 10.0, dtype=np.float32)

    def chain_residues(self, chain_id: str) -> Tuple[ResidueRecord, ...]:
        return tuple(
            residue for residue in self.residues if residue.key.chain_id == chain_id
        )


def _normalise_identifier(value: object) -> str:
    return str(value or "").strip()


def _sample_id(path: Path) -> str:
    stem = path.stem
    return stem[: -len("_protein")] if stem.endswith("_protein") else stem


def _choose_disordered_residue(residue: object) -> Tuple[object, Optional[str]]:
    """Choose a point-mutation conformer deterministically, if present."""

    if residue.__class__.__name__ != "DisorderedResidue":
        return residue, None

    candidates = list(residue.child_dict.values())
    if not candidates:
        raise StructureParseError("Encountered an empty DisorderedResidue")

    def score(candidate: object) -> Tuple[float, str]:
        occupancies = [
            atom.get_occupancy()
            for atom in candidate.get_atoms()
            if atom.get_occupancy() is not None
        ]
        mean_occupancy = float(np.mean(occupancies)) if occupancies else -1.0
        # max() is deterministic because residue name breaks occupancy ties.
        return mean_occupancy, candidate.get_resname().strip().upper()

    selected = max(candidates, key=score)
    warning = (
        "Selected residue conformer "
        f"{selected.get_resname().strip().upper()} among a disordered residue"
    )
    return selected, warning


def _altloc_priority(altloc: str) -> int:
    value = _normalise_identifier(altloc)
    if value == "":
        return 2
    if value == "A":
        return 1
    return 0


def _select_atom(atom: object) -> Tuple[object, Optional[str]]:
    """Resolve an alternate-location atom by occupancy, then blank/A preference."""

    if not atom.is_disordered() or not hasattr(atom, "child_dict"):
        return atom, None

    candidates = list(atom.child_dict.values())
    if not candidates:
        raise StructureParseError("Encountered an empty disordered atom")

    def score(candidate: object) -> Tuple[float, int]:
        occupancy = candidate.get_occupancy()
        occupancy_score = float(occupancy) if occupancy is not None else -1.0
        altloc = _normalise_identifier(candidate.get_altloc())
        return occupancy_score, _altloc_priority(altloc)

    # max() retains the first item on an exact key tie, so this ascending sort
    # makes the final fallback deterministic after occupancy and blank/A rules.
    candidates.sort(key=lambda candidate: _normalise_identifier(candidate.get_altloc()))
    selected = max(candidates, key=score)
    warning = (
        f"Selected altloc {_normalise_identifier(selected.get_altloc()) or '<blank>'} "
        f"for atom {selected.get_name().strip()}"
    )
    return selected, warning


def parse_pdb(
    path: Path | str,
    *,
    model_index: int = 0,
    chains: Optional[Iterable[str]] = None,
    required_atoms: Sequence[str] = BACKBONE_ATOMS,
    modified_residue_map: Optional[Mapping[str, str]] = None,
    reject_multiple_models: bool = True,
) -> StructureRecord:
    """Parse one PDB with publication-grade validation.

    Parameters
    ----------
    path:
        Input PDB path.
    model_index:
        Zero-based model position, not the PDB ``MODEL`` serial number.
    chains:
        Optional author chain identifiers.  Use ``""`` for a blank chain.
    required_atoms:
        Atoms each selected residue must contain.  Protein feature extraction
        uses the default backbone tuple; a pocket file may require only ``CA``.
    modified_residue_map:
        Explicit original-to-canonical three-letter mappings.  CYM->CYS is
        always retained for compatibility with the legacy extractor.
    reject_multiple_models:
        Reject ensembles by default rather than silently mixing conformations.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"PDB file not found: {source}")
    if model_index < 0:
        raise StructureParseError("model_index must be non-negative")

    selected_chains = None
    if chains is not None:
        selected_chains = {_normalise_identifier(chain) for chain in chains}
        if not selected_chains:
            raise StructureParseError("chains was provided but contains no chain identifiers")

    required = tuple(str(name).strip().upper() for name in required_atoms)
    if not required:
        raise StructureParseError("required_atoms must not be empty")
    if len(set(required)) != len(required):
        raise StructureParseError(f"required_atoms contains duplicates: {required}")

    residue_map: Dict[str, str] = dict(LEGACY_MODIFIED_RESIDUES)
    if modified_residue_map:
        residue_map.update(
            {
                str(original).strip().upper(): str(canonical).strip().upper()
                for original, canonical in modified_residue_map.items()
            }
        )
    invalid_targets = sorted(set(residue_map.values()) - set(THREE_TO_ONE))
    if invalid_targets:
        raise StructureParseError(
            f"Modified-residue mapping targets are not canonical: {invalid_targets}"
        )

    parser = PDBParser(PERMISSIVE=False, QUIET=True)
    try:
        structure = parser.get_structure(source.stem, str(source))
    except Exception as exc:
        raise StructureParseError(f"Failed to parse {source}: {exc}") from exc

    models = list(structure.get_models())
    if not models:
        raise StructureParseError(f"No MODEL found in {source}")
    if reject_multiple_models and len(models) != 1:
        raise StructureParseError(
            f"{source} contains {len(models)} models; select one explicitly with "
            "reject_multiple_models=False"
        )
    if model_index >= len(models):
        raise StructureParseError(
            f"model_index={model_index} is out of range for {len(models)} model(s)"
        )

    model = models[model_index]
    parsed: list[ResidueRecord] = []
    warnings: list[str] = []
    occupied_positions: Dict[Tuple[int, str, int, str], str] = {}
    present_chain_ids: set[str] = set()

    for chain in model:
        chain_id = _normalise_identifier(chain.id)
        present_chain_ids.add(chain_id)
        if selected_chains is not None and chain_id not in selected_chains:
            continue

        for raw_residue in chain:
            residue, residue_warning = _choose_disordered_residue(raw_residue)
            if residue_warning:
                warnings.append(f"{chain_id or '_'}: {residue_warning}")

            hetero_flag, auth_seq_id, insertion_code = residue.id
            original_name = residue.get_resname().strip().upper()
            canonical_name = residue_map.get(original_name, original_name)

            is_canonical = canonical_name in THREE_TO_ONE
            is_polymer_record = str(hetero_flag).strip() == ""
            is_explicit_modified = original_name in residue_map
            if not is_polymer_record and not is_explicit_modified:
                if is_aa(residue, standard=False):
                    raise UnsupportedResidueError(
                        f"Modified amino acid {original_name!r} at chain "
                        f"{chain_id or '_'} residue {auth_seq_id} requires an explicit "
                        "modified_residue_map entry"
                    )
                # Waters, ligands and ions are not feature rows.
                continue
            if not is_canonical:
                raise UnsupportedResidueError(
                    f"Unsupported polymer residue {original_name!r} at chain "
                    f"{chain_id or '_'} residue {auth_seq_id}"
                )

            position = (
                model_index,
                chain_id,
                int(auth_seq_id),
                _normalise_identifier(insertion_code),
            )
            if position in occupied_positions:
                raise StructureParseError(
                    f"Duplicate residue position {position} for {original_name} and "
                    f"{occupied_positions[position]}"
                )
            occupied_positions[position] = original_name

            coordinates: Dict[str, np.ndarray] = {}
            atom_warnings: list[str] = []
            for atom_name in required:
                if atom_name not in residue:
                    raise MissingBackboneAtomError(
                        f"Missing required atom {atom_name!r} in chain "
                        f"{chain_id or '_'} {original_name} {auth_seq_id}"
                        f"{_normalise_identifier(insertion_code)} ({source})"
                    )
                selected_atom, atom_warning = _select_atom(residue[atom_name])
                coordinates[atom_name] = np.asarray(
                    selected_atom.get_coord(), dtype=np.float32
                )
                if atom_warning:
                    atom_warnings.append(atom_warning)

            key = ResidueKey(
                model_index=model_index,
                chain_id=chain_id,
                auth_seq_id=int(auth_seq_id),
                insertion_code=_normalise_identifier(insertion_code),
                residue_name=original_name,
            )
            if atom_warnings:
                warnings.extend(f"{key.display()}: {message}" for message in atom_warnings)
            parsed.append(
                ResidueRecord(
                    key=key,
                    canonical_name=canonical_name,
                    one_letter=THREE_TO_ONE[canonical_name],
                    row_index=len(parsed),
                    atom_coordinates_angstrom=coordinates,
                )
            )

    if selected_chains is not None:
        missing_chains = sorted(selected_chains - present_chain_ids)
        if missing_chains:
            raise StructureParseError(
                f"Requested chain(s) absent from {source}: {missing_chains}"
            )

    return StructureRecord(
        source_path=source,
        sample_id=_sample_id(source),
        model_index=model_index,
        residues=tuple(parsed),
        warnings=tuple(warnings),
    )


def parse_feature_protein(
    path: Path | str,
    **kwargs: object,
) -> StructureRecord:
    """Parse a protein under the exact four-backbone-atom model contract."""

    return parse_pdb(path, required_atoms=BACKBONE_ATOMS, **kwargs)


def parse_pocket(path: Path | str, **kwargs: object) -> StructureRecord:
    """Parse an extracted pocket; only CA is required for identity validation."""

    return parse_pdb(path, required_atoms=("CA",), **kwargs)
