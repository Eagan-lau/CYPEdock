"""End-to-end CYPEdock workflow."""

from __future__ import annotations

import copy
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

import Bio
import numpy as np
from rdkit import rdBase

from . import foldseek
from .alignment import (
    AlignmentResult,
    SequenceMapping,
    evaluate_local_alignment,
    sequence_mapping,
)
from .cofactor import iron_atom, select_cofactor, select_reference_chain, transfer_cofactor
from .config import Config
from .cypocket import PocketResult, predict as predict_pocket, software_record
from .errors import AlignmentError, CYPEdockError, ExternalToolError, InputError
from .library import (
    LibraryEntry,
    default_database_prefix,
    default_library_root,
    library_manifest_path,
    load_library,
    validate_database_prefix,
)
from .ligand import PreparedLigand, prepare_ligand, read_ligands
from .pdbqt import write_ligand_pdbqt, write_receptor_pdbqt
from .provenance import run_command, sha256_file, write_json, write_tsv
from .structure import (
    ConformationSource,
    Residue,
    discover_ensemble,
    read_structure,
    remap_chain,
    select_protein_chain,
    write_pdb,
)
from .vina import DockingResult, dock, materialize_vina, version as vina_version


STATE_FIELDS = (
    "state_id",
    "input_structure",
    "model_index",
    "input_chain",
    "output_chain",
    "status",
    "template_source",
    "template_id",
    "local_rmsd_angstrom",
    "local_coverage",
    "pocket_residue_count",
    "center_x",
    "center_y",
    "center_z",
    "size_x",
    "size_y",
    "size_z",
    "minimum_fe_box_margin_angstrom",
    "reason",
)
ALIGNMENT_FIELDS = (
    "state_id",
    "attempt",
    "template_source",
    "template_id",
    "status",
    "reason",
    "sequence_identity",
    "sequence_reference_coverage",
    "sequence_target_coverage",
    "shell_residue_count",
    "mapped_shell_residue_count",
    "local_coverage",
    "alignment_atom_count",
    "local_rmsd_angstrom",
    "maximum_local_residue_rmsd_angstrom",
    "global_ca_rmsd_angstrom",
    "reference_fe_s_distance_angstrom",
    "target_fe_s_distance_angstrom",
    "minimum_cofactor_protein_distance_angstrom",
)
DOCKING_FIELDS = (
    "task_id",
    "state_id",
    "ligand_id",
    "replicate",
    "seed",
    "pose_rank",
    "affinity_kcal_mol",
    "rmsd_lower_bound",
    "rmsd_upper_bound",
    "status",
    "output_pdbqt",
    "output_out",
    "error",
)


@dataclass(frozen=True)
class Template:
    template_id: str
    source_type: str
    path: Path
    protein: tuple[Residue, ...]
    cofactor: tuple[Residue, ...]


@dataclass(frozen=True)
class AcceptedState:
    state_id: str
    protein_pdb: Path
    receptor_pdb: Path
    receptor_pdbqt: Path
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    template_source: str
    pocket: PocketResult


def docking_box(
    pocket: PocketResult,
    cofactor: Sequence[Residue],
    settings: dict,
    *,
    details: Optional[dict] = None,
) -> tuple[tuple[float, float, float], float]:
    """Calculate a fixed or pocket-enclosing box and validate the heme Fe position."""
    center = np.asarray(pocket.center, dtype=float)
    coordinates = np.asarray(pocket.coordinates, dtype=float)
    if center.shape != (3,) or coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise InputError("Pocket coordinates are invalid for docking-box calculation")
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(coordinates)):
        raise InputError("Pocket coordinates must be finite")
    padding = float(settings["pocket_padding_angstrom"])
    minimum = np.asarray(settings["size_angstrom"], dtype=float)
    maximum = np.asarray(settings["maximum_size_angstrom"], dtype=float)
    iron = iron_atom(cofactor).coordinate
    required_margin = float(settings["minimum_fe_margin_angstrom"])
    if bool(settings["auto_size"]):
        pocket_required = (
            2.0 * np.max(np.abs(coordinates - center), axis=0) + 2.0 * padding
        )
        iron_required = 2.0 * np.abs(iron - center) + 2.0 * required_margin
        required = np.maximum(pocket_required, iron_required)
        requested_size = np.maximum(minimum, required)
        clipped_mask = requested_size > maximum + 1e-9
        size = np.minimum(requested_size, maximum)
    else:
        requested_size = minimum
        clipped_mask = np.zeros(3, dtype=bool)
        size = minimum
    margins = size / 2.0 - np.abs(iron - center)
    minimum_margin = float(np.min(margins))
    if minimum_margin < required_margin:
        raise InputError(
            "Transferred Fe is not sufficiently enclosed by the docking box: "
            f"minimum margin={minimum_margin:.3f} A, required={required_margin:.3f} A; "
            "enable --auto-box or provide a larger --box-size"
        )
    if details is not None:
        details.update(
            {
                "requested_size_angstrom": [
                    float(value) for value in requested_size
                ],
                "clipped_to_maximum": bool(np.any(clipped_mask)),
                "clipped_axes": [
                    axis
                    for axis, clipped in zip(("x", "y", "z"), clipped_mask)
                    if bool(clipped)
                ],
            }
        )
    return tuple(float(value) for value in size), minimum_margin


def _input_file_record(path: Path) -> dict:
    source = Path(path).expanduser().resolve()
    return {"path": str(source), "sha256": sha256_file(source)}


def _reject_destructive_output(output: Path, protected_paths: Sequence[Path]) -> None:
    for value in protected_paths:
        protected = Path(value).expanduser().resolve()
        try:
            protected.relative_to(output)
        except ValueError:
            continue
        raise InputError(
            f"Output directory {output} would contain and delete protected input {protected}"
        )


class Pipeline:
    def __init__(self, config: Config, *, console: Callable[[str], None] = print):
        self.config = config
        self.data = copy.deepcopy(config.data)
        self._config_source = config.source
        self.console = console
        self._foldseek_used = False
        self._reference_preparation_error: Optional[str] = None
        self._library_cache: Optional[tuple[Path, Path, dict[str, LibraryEntry]]] = None

    def _message(self, value: str) -> None:
        self.console(value)

    def _reference_template(self, path: Path) -> Template:
        records = read_structure(path, model_index=0)
        cofactor = select_cofactor(
            records, self.data["reference"]["cofactor_residues"]
        )
        protein = select_reference_chain(
            records,
            cofactor,
            self.data["reference"]["protein_chain"],
        )
        return Template(
            template_id=Path(path).stem,
            source_type="supplied_holo",
            path=Path(path).expanduser().resolve(),
            protein=tuple(protein),
            cofactor=tuple(cofactor),
        )

    def _alignment_attempt(
        self,
        *,
        state_id: str,
        attempt: int,
        template: Template,
        target: Sequence[Residue],
        mapping: SequenceMapping,
    ) -> tuple[AlignmentResult, dict]:
        result = evaluate_local_alignment(
            template.protein,
            target,
            template.cofactor,
            mapping,
            self.data["alignment"],
        )
        row = {
            "state_id": state_id,
            "attempt": attempt,
            "template_source": template.source_type,
            "template_id": template.template_id,
            **result.metrics.to_dict(),
        }
        return result, row

    def _library_settings(self) -> tuple[Path, Path, dict[str, LibraryEntry]]:
        if self._library_cache is not None:
            return self._library_cache
        settings = self.data["foldseek"]
        root = (
            Path(settings["library_root"]).expanduser().resolve()
            if settings["library_root"]
            else default_library_root().resolve()
        )
        prefix = (
            Path(settings["database_prefix"]).expanduser().resolve()
            if settings["database_prefix"]
            else default_database_prefix(root)
        )
        structure_root = (
            Path(settings["structure_root"]).expanduser().resolve()
            if settings["structure_root"]
            else None
        )
        entries = load_library(root, structure_root=structure_root)
        if not entries:
            raise InputError(
                "The CYP450 library is empty. Supply --reference-holo or install the "
                "published Foldseek database and manifest"
            )
        validate_database_prefix(prefix)
        self._library_cache = (root, prefix, entries)
        return self._library_cache

    def _fallback_template(
        self,
        *,
        state_id: str,
        protein_pdb: Path,
        target: Sequence[Residue],
        state_dir: Path,
        first_attempt: int,
        alignment_rows: list[dict],
    ) -> tuple[Optional[Template], Optional[AlignmentResult]]:
        settings = self.data["foldseek"]
        root, prefix, entries = self._library_settings()
        self._foldseek_used = True
        hits = foldseek.search(
            executable=str(settings["executable"]),
            query_pdb=protein_pdb,
            database_prefix=prefix,
            output_tsv=state_dir / "foldseek" / "hits.tsv",
            temporary_directory=state_dir / "foldseek" / "tmp",
            settings=settings,
            log_dir=state_dir / "logs" / "foldseek",
        )
        if not hits:
            self._message(f"WARNING: {state_id}: Foldseek returned no eligible library hit")
            return None, None
        for offset, hit in enumerate(hits):
            entry = entries.get(hit.target)
            if entry is None:
                self._message(
                    f"WARNING: {state_id}: Foldseek target {hit.target!r} has no manifest entry"
                )
                continue
            records = read_structure(entry.structure_file, model_index=0)
            cofactor = select_cofactor(records, entry.cofactor_residues)
            protein = select_reference_chain(records, cofactor, entry.protein_chain)
            template = Template(
                template_id=entry.target_id,
                source_type="foldseek_library",
                path=entry.structure_file,
                protein=tuple(protein),
                cofactor=tuple(cofactor),
            )
            try:
                mapping = foldseek.mapping_from_hit(
                    hit,
                    reference_length=len(template.protein),
                    target_length=len(target),
                )
                result, row = self._alignment_attempt(
                    state_id=state_id,
                    attempt=first_attempt + offset,
                    template=template,
                    target=target,
                    mapping=mapping,
                )
            except CYPEdockError as exc:
                alignment_rows.append(
                    {
                        "state_id": state_id,
                        "attempt": first_attempt + offset,
                        "template_source": "foldseek_library",
                        "template_id": entry.target_id,
                        "status": "SKIP",
                        "reason": str(exc),
                    }
                )
                continue
            alignment_rows.append(row)
            if result.passed:
                return template, result
        return None, None

    def _foldseek_record(self) -> dict:
        root, prefix, entries = self._library_settings()
        configured = str(self.data["foldseek"]["executable"])
        resolved = shutil.which(configured)
        executable = Path(resolved or configured).expanduser().resolve()
        if not executable.is_file():
            raise InputError(f"Foldseek executable is unavailable: {configured}")
        probe = run_command([str(executable), "version"], timeout=30)
        version_text = (probe.stdout + "\n" + probe.stderr).strip()
        database_files = sorted(
            path
            for path in prefix.parent.glob(f"{prefix.name}*")
            if path.is_file()
        )
        return {
            "path": str(executable),
            "version": version_text,
            "sha256": sha256_file(executable),
            "library_root": str(root),
            "structure_root": self.data["foldseek"]["structure_root"],
            "library_manifest": str(library_manifest_path(root)),
            "library_manifest_sha256": sha256_file(library_manifest_path(root)),
            "library_entry_count": len(entries),
            "database_prefix": str(prefix),
            "database_files": [
                {
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in database_files
            ],
        }

    def _process_state(
        self,
        source: ConformationSource,
        reference: Optional[Template],
        output: Path,
        state_rows: list[dict],
        alignment_rows: list[dict],
    ) -> Optional[AcceptedState]:
        state_dir = output / "states" / source.state_id
        state_dir.mkdir(parents=True)
        base_row = {
            "state_id": source.state_id,
            "input_structure": str(source.path),
            "model_index": source.model_index,
            "input_chain": "",
            "output_chain": "",
            "status": "SKIP",
            "template_source": "",
            "template_id": "",
            "reason": "",
        }
        try:
            records = read_structure(source.path, model_index=source.model_index)
            target = select_protein_chain(records, self.data["input"]["chain"])
            input_chain = target[0].residue_id.chain
            chain = (
                input_chain
                if len(input_chain) == 1 and input_chain.isascii() and input_chain.strip()
                else "A"
            )
            if chain != input_chain:
                self._message(
                    f"INFO: {source.state_id}: remapped input chain {input_chain!r} "
                    f"to PDB-compatible chain {chain!r}"
                )
                target = remap_chain(target, chain)
            base_row["input_chain"] = input_chain
            base_row["output_chain"] = chain
            protein_pdb = write_pdb(
                target, state_dir / f"{source.state_id}_protein.pdb"
            )
            selected_template: Optional[Template] = None
            selected_alignment: Optional[AlignmentResult] = None
            attempt = 1
            if reference is not None:
                try:
                    mapping = sequence_mapping(reference.protein, target)
                    selected_alignment, row = self._alignment_attempt(
                        state_id=source.state_id,
                        attempt=attempt,
                        template=reference,
                        target=target,
                        mapping=mapping,
                    )
                    alignment_rows.append(row)
                    if selected_alignment.passed:
                        selected_template = reference
                    else:
                        self._message(
                            f"WARNING: {source.state_id}: supplied holo rejected: "
                            f"{selected_alignment.metrics.reason}"
                        )
                except (AlignmentError, InputError) as exc:
                    alignment_rows.append(
                        {
                            "state_id": source.state_id,
                            "attempt": attempt,
                            "template_source": "supplied_holo",
                            "template_id": reference.template_id,
                            "status": "SKIP",
                            "reason": str(exc),
                        }
                    )
                    self._message(
                        f"WARNING: {source.state_id}: supplied holo alignment failed: {exc}"
                    )
                attempt += 1
            if selected_template is None and bool(self.data["foldseek"]["enabled"]):
                self._message(f"INFO: {source.state_id}: attempting Foldseek library fallback")
                selected_template, selected_alignment = self._fallback_template(
                    state_id=source.state_id,
                    protein_pdb=protein_pdb,
                    target=target,
                    state_dir=state_dir,
                    first_attempt=attempt,
                    alignment_rows=alignment_rows,
                )
            if (
                selected_template is None
                or selected_alignment is None
                or selected_alignment.transform is None
            ):
                reason = "No supplied or library template passed local-site quality control"
                self._message(f"WARNING: {source.state_id}: {reason}; state skipped")
                base_row["reason"] = reason
                state_rows.append(base_row)
                return None

            cofactor = transfer_cofactor(
                selected_template.cofactor,
                selected_alignment.transform,
                target,
            )
            receptor_records = [*target, *cofactor]
            receptor_pdb = write_pdb(
                receptor_records, state_dir / f"{source.state_id}_receptor.pdb"
            )
            receptor_pdbqt = write_receptor_pdbqt(
                receptor_records, state_dir / f"{source.state_id}_receptor.pdbqt"
            )
            write_json(
                state_dir / "alignment.json",
                {
                    "state_id": source.state_id,
                    "template_source": selected_template.source_type,
                    "template_id": selected_template.template_id,
                    "template_file": str(selected_template.path),
                    "template_sha256": sha256_file(selected_template.path),
                    "metrics": selected_alignment.metrics.to_dict(),
                    "transform": selected_alignment.transform.to_dict(),
                    "mapped_residue_pairs": selected_alignment.mapped_residue_pairs,
                    "shell_reference_indices": selected_alignment.shell_reference_indices,
                },
            )
            pocket = predict_pocket(
                protein_pdb=protein_pdb,
                state_id=source.state_id,
                chain=chain,
                output_dir=state_dir / "cypocket",
                settings=self.data["cypocket"],
                center_mode=self.data["box"]["center_atoms"],
                log_dir=state_dir / "logs" / "cypocket",
            )
            center = tuple(float(value) for value in pocket.center)
            box_details: dict = {}
            size, minimum_fe_margin = docking_box(
                pocket,
                cofactor,
                self.data["box"],
                details=box_details,
            )
            if box_details["clipped_to_maximum"]:
                self._message(
                    f"WARNING: {source.state_id}: automatically requested box "
                    f"{box_details['requested_size_angstrom']} A was clipped on axes "
                    f"{box_details['clipped_axes']} to {list(size)} A"
                )
            write_json(
                state_dir / "box.json",
                {
                    "state_id": source.state_id,
                    "auto_size": self.data["box"]["auto_size"],
                    "center_angstrom": list(center),
                    "size_angstrom": list(size),
                    "minimum_size_angstrom": self.data["box"]["size_angstrom"],
                    "maximum_size_angstrom": self.data["box"]["maximum_size_angstrom"],
                    "pocket_padding_angstrom": self.data["box"][
                        "pocket_padding_angstrom"
                    ],
                    "transferred_fe_angstrom": iron_atom(cofactor).coordinate.tolist(),
                    "minimum_fe_box_margin_angstrom": minimum_fe_margin,
                    "center_definition": self.data["box"]["center_atoms"],
                    "pocket_residue_count": pocket.residue_count,
                    **box_details,
                },
            )
            base_row.update(
                {
                    "status": "READY",
                    "template_source": selected_template.source_type,
                    "template_id": selected_template.template_id,
                    "local_rmsd_angstrom": selected_alignment.metrics.local_rmsd_angstrom,
                    "local_coverage": selected_alignment.metrics.local_coverage,
                    "pocket_residue_count": pocket.residue_count,
                    "center_x": center[0],
                    "center_y": center[1],
                    "center_z": center[2],
                    "size_x": size[0],
                    "size_y": size[1],
                    "size_z": size[2],
                    "minimum_fe_box_margin_angstrom": minimum_fe_margin,
                    "reason": "",
                }
            )
            state_rows.append(base_row)
            return AcceptedState(
                state_id=source.state_id,
                protein_pdb=protein_pdb,
                receptor_pdb=receptor_pdb,
                receptor_pdbqt=receptor_pdbqt,
                center=center,
                size=size,
                template_source=selected_template.source_type,
                pocket=pocket,
            )
        except CYPEdockError as exc:
            base_row["reason"] = str(exc)
            state_rows.append(base_row)
            self._message(f"WARNING: {source.state_id}: {exc}; state skipped")
            return None

    def run(
        self,
        *,
        ensemble: Path,
        output_dir: Path,
        ligand_input: Optional[Path] = None,
        smiles_file: Optional[Path] = None,
        reference_holo: Optional[Path] = None,
        overwrite: bool = False,
    ) -> Path:
        self._foldseek_used = False
        self._reference_preparation_error = None
        self._library_cache = None
        output = Path(output_dir).expanduser().resolve()
        started = datetime.now(timezone.utc).isoformat()
        sources = discover_ensemble(ensemble)
        if ligand_input is not None and smiles_file is not None:
            raise InputError("Provide only one of ligand_input or smiles_file")
        ligand_source = ligand_input if ligand_input is not None else smiles_file
        if ligand_source is None:
            raise InputError("A ligand input file is required")
        ligand_source = Path(ligand_source).expanduser().resolve()
        ligands = read_ligands(ligand_source)
        reference: Optional[Template] = None
        if reference_holo is not None:
            reference_path = Path(reference_holo).expanduser().resolve()
            if not reference_path.is_file():
                raise InputError(f"Reference holo structure does not exist: {reference_path}")
            try:
                reference = self._reference_template(reference_path)
            except CYPEdockError as exc:
                if not bool(self.data["foldseek"]["enabled"]):
                    raise
                self._reference_preparation_error = str(exc)
                self._message(
                    "WARNING: supplied holo could not be prepared: "
                    f"{exc}; Foldseek library fallback will be used"
                )
        if reference is None and not bool(self.data["foldseek"]["enabled"]):
            raise InputError(
                "A reference holo structure is required when Foldseek fallback is disabled"
            )
        protected_paths = [
            Path(ensemble),
            ligand_source,
            Path(__file__),
            *(source.path for source in sources),
        ]
        if self._config_source is not None:
            protected_paths.append(self._config_source)
        if reference_holo is not None:
            protected_paths.append(Path(reference_holo))
        dependency_values = [
            self.data["foldseek"]["executable"],
            self.data["foldseek"]["library_root"],
            self.data["foldseek"]["database_prefix"],
            self.data["foldseek"]["structure_root"],
            self.data["cypocket"]["python"],
            self.data["cypocket"]["root"],
            self.data["cypocket"]["feature_python"],
            self.data["cypocket"]["classifier_python"],
            self.data["cypocket"]["pocketminer_checkpoint"],
            self.data["cypocket"]["classifier_checkpoint"],
            self.data["docking"]["vina_executable"],
        ]
        for value in dependency_values:
            if value and Path(str(value)).expanduser().is_absolute():
                protected_paths.append(Path(str(value)))
        _reject_destructive_output(output, protected_paths)
        if output.exists():
            if not overwrite:
                raise InputError(f"Output directory already exists: {output}")
            if not output.is_dir():
                raise InputError(f"Output path exists and is not a directory: {output}")
            shutil.rmtree(output)
        output.mkdir(parents=True)

        state_rows: list[dict] = []
        alignment_rows: list[dict] = []
        accepted: list[AcceptedState] = []
        for source in sources:
            self._message(f"INFO: processing conformation {source.state_id}")
            result = self._process_state(
                source, reference, output, state_rows, alignment_rows
            )
            if result is not None:
                accepted.append(result)
        write_tsv(output / "state_results.tsv", state_rows, STATE_FIELDS)
        write_tsv(
            output / "alignment_attempts.tsv", alignment_rows, ALIGNMENT_FIELDS
        )
        if not accepted:
            raise InputError("No conformation passed alignment and CYPocket quality control")

        prepared_ligands: list[PreparedLigand] = []
        for index, ligand in enumerate(ligands, start=1):
            seed = int(self.data["project"]["random_seed"]) + index
            prepared = prepare_ligand(
                ligand,
                output / "ligands" / ligand.ligand_id,
                self.data["ligand"],
                random_seed=seed,
            )
            write_ligand_pdbqt(
                prepared.molecule,
                ligand.ligand_id,
                output / "ligands" / ligand.ligand_id / f"{ligand.ligand_id}.pdbqt",
            )
            prepared_ligands.append(prepared)

        vina_executable = materialize_vina(
            output / "tools", self.data["docking"]["vina_executable"]
        )
        vina_record = {
            "path": str(vina_executable),
            "version": vina_version(vina_executable),
            "sha256": sha256_file(vina_executable),
        }
        cypocket_record = software_record(self.data["cypocket"])
        docking_futures = []
        docking_results: list[DockingResult] = []
        with ThreadPoolExecutor(max_workers=int(self.data["docking"]["workers"])) as executor:
            for state in accepted:
                for ligand in prepared_ligands:
                    ligand_pdbqt = (
                        output
                        / "ligands"
                        / ligand.ligand.ligand_id
                        / f"{ligand.ligand.ligand_id}.pdbqt"
                    )
                    for replicate in range(
                        1, int(self.data["docking"]["replicates"]) + 1
                    ):
                        docking_futures.append(
                            executor.submit(
                                dock,
                                executable=vina_executable,
                                receptor_pdbqt=state.receptor_pdbqt,
                                ligand_pdbqt=ligand_pdbqt,
                                center=state.center,
                                size=state.size,
                                state_id=state.state_id,
                                ligand_id=ligand.ligand.ligand_id,
                                replicate=replicate,
                                base_seed=int(self.data["project"]["random_seed"]),
                                output_dir=output / "docking",
                                settings=self.data["docking"],
                            )
                        )
            for future in as_completed(docking_futures):
                docking_results.append(future.result())
        docking_results.sort(key=lambda value: value.task_id)
        docking_rows: list[dict] = []
        for result in docking_results:
            if result.scores:
                for score in result.scores:
                    docking_rows.append(
                        {
                            "task_id": result.task_id,
                            "state_id": result.state_id,
                            "ligand_id": result.ligand_id,
                            "replicate": result.replicate,
                            "seed": result.seed,
                            "pose_rank": score.pose_rank,
                            "affinity_kcal_mol": score.affinity_kcal_mol,
                            "rmsd_lower_bound": score.rmsd_lower_bound,
                            "rmsd_upper_bound": score.rmsd_upper_bound,
                            "status": result.status,
                            "output_pdbqt": str(result.output_pdbqt),
                            "output_out": str(result.output_out),
                            "error": result.error,
                        }
                    )
            else:
                docking_rows.append(
                    {
                        "task_id": result.task_id,
                        "state_id": result.state_id,
                        "ligand_id": result.ligand_id,
                        "replicate": result.replicate,
                        "seed": result.seed,
                        "status": result.status,
                        "output_pdbqt": str(result.output_pdbqt),
                        "output_out": str(result.output_out),
                        "error": result.error,
                    }
                )
        write_tsv(output / "docking_results.tsv", docking_rows, DOCKING_FIELDS)
        successes = sum(result.status == "SUCCESS" for result in docking_results)
        structure_hashes = {
            source.path.resolve(): sha256_file(source.path.resolve()) for source in sources
        }
        software = {
            "cypedock": "0.1.0",
            "python_packages": {
                "biopython": Bio.__version__,
                "numpy": np.__version__,
                "rdkit": rdBase.rdkitVersion,
            },
            "vina": vina_record,
            "cypocket": cypocket_record,
        }
        if self._foldseek_used:
            software["foldseek"] = self._foldseek_record()
        reference_record = (
            _input_file_record(reference_holo) if reference_holo is not None else None
        )
        if reference_record is not None and self._reference_preparation_error is not None:
            reference_record["preparation_error"] = self._reference_preparation_error
        ligand_record = _input_file_record(ligand_source)
        ligand_record["formats"] = sorted({ligand.source_format for ligand in ligands})
        manifest = {
            "schema_version": 1,
            "software": software,
            "inputs": {
                "ensemble": str(Path(ensemble).expanduser().resolve()),
                "conformations": [
                    {
                        "state_id": source.state_id,
                        "path": str(source.path.resolve()),
                        "model_index": source.model_index,
                        "sha256": structure_hashes[source.path.resolve()],
                    }
                    for source in sources
                ],
                "ligand_input": ligand_record,
                "smiles_file": ligand_record if smiles_file is not None else None,
                "reference_holo": reference_record,
            },
            "configuration": self.data,
            "started_at": started,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "counts": {
                "input_conformations": len(sources),
                "accepted_conformations": len(accepted),
                "ligands": len(prepared_ligands),
                "docking_tasks": len(docking_results),
                "successful_docking_tasks": successes,
            },
        }
        write_json(output / "run_manifest.json", manifest)
        if successes == 0:
            raise ExternalToolError(
                f"All {len(docking_results)} Vina tasks failed; inspect docking logs"
            )
        self._message(
            f"INFO: completed {successes}/{len(docking_results)} Vina tasks; output={output}"
        )
        return output
