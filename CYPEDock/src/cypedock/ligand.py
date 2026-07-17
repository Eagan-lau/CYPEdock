"""Ligand input parsing and deterministic RDKit three-dimensional preparation."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

from .errors import InputError
from .provenance import write_json


SMILES_SUFFIXES = {".smi", ".smiles", ".txt"}
STRUCTURE_SUFFIXES = {".sdf", ".sd", ".mol", ".mol2"}
SUPPORTED_LIGAND_SUFFIXES = SMILES_SUFFIXES | STRUCTURE_SUFFIXES


@dataclass(frozen=True)
class LigandInput:
    ligand_id: str
    smiles: str
    canonical_smiles: str
    molecule: object
    source_path: Path
    source_format: str
    source_record: int
    input_has_3d: bool


@dataclass
class PreparedLigand:
    ligand: LigandInput
    molecule: object
    sdf_path: Path
    force_field: str
    optimization_status: int
    energy_kcal_mol: float


def _safe_identifier(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not normalized:
        raise InputError(f"Invalid ligand identifier: {value!r}")
    return normalized


def _has_3d_coordinates(molecule: object) -> bool:
    return bool(
        molecule.GetNumConformers() > 0 and molecule.GetConformer(0).Is3D()
    )


def _from_molecule(
    molecule: object,
    *,
    source: Path,
    source_format: str,
    source_record: int,
    proposed_identifier: Optional[str] = None,
    input_smiles: Optional[str] = None,
) -> LigandInput:
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise InputError(
            f"RDKit rejected ligand record {source_record} in {source}"
        )
    try:
        Chem.SanitizeMol(molecule)
    except Exception as exc:
        raise InputError(
            f"RDKit could not sanitize ligand record {source_record} in {source}: {exc}"
        ) from exc
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise InputError(
            f"Disconnected ligand record {source_record} is not accepted: {source}"
        )
    heavy = Chem.RemoveHs(Chem.Mol(molecule))
    canonical = Chem.MolToSmiles(heavy, canonical=True, isomericSmiles=True)
    if not canonical:
        raise InputError(
            f"RDKit could not derive a canonical SMILES for record {source_record}: {source}"
        )
    identifier = (
        _safe_identifier(proposed_identifier)
        if proposed_identifier and proposed_identifier.strip()
        else "LIG_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    )
    return LigandInput(
        ligand_id=identifier,
        smiles=input_smiles or canonical,
        canonical_smiles=canonical,
        molecule=Chem.Mol(molecule),
        source_path=source,
        source_format=source_format,
        source_record=source_record,
        input_has_3d=_has_3d_coordinates(molecule),
    )


def _read_smiles_file(source: Path) -> list[LigandInput]:
    ligands: list[LigandInput] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            smiles = fields[0]
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                raise InputError(
                    f"RDKit rejected SMILES at line {line_number}: {smiles!r}"
                )
            ligands.append(
                _from_molecule(
                    molecule,
                    source=source,
                    source_format="smiles",
                    source_record=line_number,
                    proposed_identifier=fields[1] if len(fields) > 1 else None,
                    input_smiles=smiles,
                )
            )
    return ligands


def _read_sdf(source: Path) -> list[LigandInput]:
    supplier = Chem.SDMolSupplier(str(source), removeHs=False, sanitize=True)
    ligands: list[LigandInput] = []
    for index, molecule in enumerate(supplier, start=1):
        if molecule is None:
            raise InputError(f"RDKit rejected SDF record {index} in {source}")
        name = molecule.GetProp("_Name").strip() if molecule.HasProp("_Name") else None
        ligands.append(
            _from_molecule(
                molecule,
                source=source,
                source_format="sdf",
                source_record=index,
                proposed_identifier=name,
            )
        )
    return ligands


def _read_mol(source: Path) -> list[LigandInput]:
    molecule = Chem.MolFromMolFile(str(source), removeHs=False, sanitize=True)
    name = None
    if molecule is not None and molecule.HasProp("_Name"):
        name = molecule.GetProp("_Name").strip()
    return [
        _from_molecule(
            molecule,
            source=source,
            source_format="mol",
            source_record=1,
            proposed_identifier=name or source.stem,
        )
    ]


def _read_mol2(source: Path) -> list[LigandInput]:
    text = source.read_text(encoding="utf-8", errors="replace")
    blocks = [
        block
        for block in re.split(r"(?=@<TRIPOS>MOLECULE)", text)
        if block.lstrip().startswith("@<TRIPOS>MOLECULE")
    ]
    if not blocks:
        raise InputError(f"No @<TRIPOS>MOLECULE records were found in {source}")
    ligands: list[LigandInput] = []
    for index, block in enumerate(blocks, start=1):
        lines = block.splitlines()
        name = lines[1].strip() if len(lines) > 1 else None
        molecule = Chem.MolFromMol2Block(
            block,
            sanitize=True,
            removeHs=False,
            cleanupSubstructures=True,
        )
        ligands.append(
            _from_molecule(
                molecule,
                source=source,
                source_format="mol2",
                source_record=index,
                proposed_identifier=name,
            )
        )
    return ligands


def read_ligands(path: Path) -> list[LigandInput]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise InputError(f"Ligand input file does not exist: {source}")
    suffix = source.suffix.lower()
    if suffix in SMILES_SUFFIXES:
        ligands = _read_smiles_file(source)
    elif suffix in {".sdf", ".sd"}:
        ligands = _read_sdf(source)
    elif suffix == ".mol":
        ligands = _read_mol(source)
    elif suffix == ".mol2":
        ligands = _read_mol2(source)
    else:
        supported = ", ".join(sorted(SUPPORTED_LIGAND_SUFFIXES))
        raise InputError(
            f"Unsupported ligand input format {suffix or '<none>'}: {source}; "
            f"supported suffixes are {supported}"
        )
    if not ligands:
        raise InputError(f"No ligands were found in {source}")
    identifiers: set[str] = set()
    for ligand in ligands:
        if ligand.ligand_id in identifiers:
            raise InputError(f"Duplicate ligand identifier: {ligand.ligand_id}")
        identifiers.add(ligand.ligand_id)
    return ligands


def read_smiles(path: Path) -> list[LigandInput]:
    """Backward-compatible SMILES reader."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise InputError(f"SMILES file does not exist: {source}")
    ligands = _read_smiles_file(source)
    identifiers: set[str] = set()
    for ligand in ligands:
        if ligand.ligand_id in identifiers:
            raise InputError(f"Duplicate ligand identifier: {ligand.ligand_id}")
        identifiers.add(ligand.ligand_id)
    if not ligands:
        raise InputError(f"No ligands were found in {source}")
    return ligands


def prepare_ligand(
    ligand: LigandInput,
    output_dir: Path,
    settings: Mapping[str, object],
    *,
    random_seed: int,
) -> PreparedLigand:
    if ligand.input_has_3d:
        molecule = Chem.AddHs(Chem.Mol(ligand.molecule), addCoords=True)
        conformer_id = 0
        coordinate_source = "input_3d"
    else:
        molecule = Chem.AddHs(Chem.MolFromSmiles(ligand.canonical_smiles))
        parameters = AllChem.ETKDGv3()
        parameters.randomSeed = int(random_seed & 0x7FFFFFFF)
        parameters.maxIterations = int(settings["embedding_attempts"])
        parameters.useRandomCoords = False
        conformer_id = AllChem.EmbedMolecule(molecule, parameters)
        if conformer_id < 0:
            raise InputError(f"RDKit could not embed ligand {ligand.ligand_id}")
        coordinate_source = "rdkit_etkdg_v3"
    iterations = int(settings["optimization_iterations"])
    if AllChem.MMFFHasAllMoleculeParams(molecule):
        force_field = "MMFF94s"
        properties = AllChem.MMFFGetMoleculeProperties(molecule, mmffVariant="MMFF94s")
        field = AllChem.MMFFGetMoleculeForceField(
            molecule, properties, confId=conformer_id
        )
    else:
        force_field = "UFF"
        field = AllChem.UFFGetMoleculeForceField(molecule, confId=conformer_id)
    if field is None:
        raise InputError(f"No RDKit force field is available for {ligand.ligand_id}")
    status = int(field.Minimize(maxIts=iterations))
    if status != 0:
        raise InputError(
            f"RDKit {force_field} optimization did not converge for "
            f"{ligand.ligand_id} within {iterations} iterations"
        )
    energy = float(field.CalcEnergy())
    if not math.isfinite(energy):
        raise InputError(f"RDKit produced a non-finite energy for {ligand.ligand_id}")
    AllChem.ComputeGasteigerCharges(molecule)
    molecule.SetProp("_Name", ligand.ligand_id)
    molecule.SetProp("CYPEdockCanonicalSMILES", ligand.canonical_smiles)
    molecule.SetProp("CYPEdockForceField", force_field)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    sdf_path = destination / f"{ligand.ligand_id}.sdf"
    writer = Chem.SDWriter(str(sdf_path))
    writer.write(molecule, confId=conformer_id)
    writer.close()
    if not sdf_path.is_file() or sdf_path.stat().st_size == 0:
        raise InputError(f"RDKit failed to write {sdf_path}")
    write_json(
        destination / f"{ligand.ligand_id}.json",
        {
            "ligand_id": ligand.ligand_id,
            "input_source": str(ligand.source_path),
            "input_format": ligand.source_format,
            "input_record": ligand.source_record,
            "input_smiles": ligand.smiles,
            "canonical_smiles": ligand.canonical_smiles,
            "coordinate_source": coordinate_source,
            "rdkit_version": rdBase.rdkitVersion,
            "random_seed": random_seed,
            "force_field": force_field,
            "optimization_status": status,
            "energy_kcal_mol": energy,
        },
    )
    return PreparedLigand(
        ligand=ligand,
        molecule=molecule,
        sdf_path=sdf_path,
        force_field=force_field,
        optimization_status=status,
        energy_kcal_mol=energy,
    )
