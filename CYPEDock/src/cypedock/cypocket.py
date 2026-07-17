"""CYPocket subprocess integration and pocket-box calculation."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from .errors import ConfigurationError, ExternalToolError
from .provenance import run_command, sha256_file
from .structure import read_structure


@dataclass(frozen=True)
class CYPocketPaths:
    launcher_python: Path
    root: Path
    script: Path
    feature_python: Path
    classifier_python: Path
    pocketminer_checkpoint: Path
    classifier_checkpoint: Path


@dataclass(frozen=True)
class PocketResult:
    pocket_pdb: Path
    residue_table: Path
    summary_json: Path
    center: np.ndarray
    coordinates: np.ndarray
    residue_count: int


def _executable(value: object, field: str, fallback: Optional[str] = None) -> Path:
    raw = str(value or fallback or "").strip()
    if not raw:
        raise ConfigurationError(f"{field} must be configured")
    path = Path(raw).expanduser().resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ConfigurationError(f"{field} is not an executable file: {path}")
    return path


def resolve_paths(settings: Mapping[str, object]) -> CYPocketPaths:
    root_value = str(settings.get("root") or "").strip()
    if not root_value:
        raise ConfigurationError("cypocket.root must identify the CYPocket installation")
    root = Path(root_value).expanduser().resolve()
    script = root / "predict_pdb.py"
    if not script.is_file():
        raise ConfigurationError(f"CYPocket entry script does not exist: {script}")
    launcher = _executable(settings.get("python"), "cypocket.python", sys.executable)
    feature = _executable(
        settings.get("feature_python"), "cypocket.feature_python", str(launcher)
    )
    classifier = _executable(
        settings.get("classifier_python"), "cypocket.classifier_python", str(launcher)
    )
    model_root = root / "P450model"
    if not model_root.is_dir() and (root.parent / "P450model").is_dir():
        model_root = root.parent / "P450model"
    pocketminer = Path(
        str(settings.get("pocketminer_checkpoint") or model_root / "pocketminer")
    ).expanduser().resolve()
    if not Path(f"{pocketminer}.index").is_file() or not list(
        pocketminer.parent.glob(f"{pocketminer.name}.data-*")
    ):
        raise ConfigurationError(f"PocketMiner checkpoint prefix is incomplete: {pocketminer}")
    classifier_checkpoint = Path(
        str(
            settings.get("classifier_checkpoint")
            or model_root / "best_protein_distilbert_classifier.pth"
        )
    ).expanduser().resolve()
    if not classifier_checkpoint.is_file():
        raise ConfigurationError(
            f"CYPocket classifier checkpoint does not exist: {classifier_checkpoint}"
        )
    return CYPocketPaths(
        launcher_python=launcher,
        root=root,
        script=script,
        feature_python=feature,
        classifier_python=classifier,
        pocketminer_checkpoint=pocketminer,
        classifier_checkpoint=classifier_checkpoint,
    )


def checkpoint_prefix_sha256(prefix: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    files = [Path(f"{prefix}.index"), *sorted(prefix.parent.glob(f"{prefix.name}.data-*"))]
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _center(pocket_pdb: Path, mode: str) -> tuple[np.ndarray, np.ndarray, int]:
    residues = read_structure(pocket_pdb)
    if mode == "CA":
        coordinates = [
            residue.atom("CA").coordinate
            for residue in residues
            if residue.atom("CA") is not None
        ]
    else:
        coordinates = [
            atom.coordinate
            for residue in residues
            for atom in residue.heavy_atoms()
        ]
    if not coordinates:
        raise ExternalToolError(
            f"CYPocket predicted no usable pocket coordinates in {pocket_pdb}"
        )
    coordinate_array = np.asarray(coordinates, dtype=float)
    return np.mean(coordinate_array, axis=0), coordinate_array, len(residues)


def predict(
    *,
    protein_pdb: Path,
    state_id: str,
    chain: str,
    output_dir: Path,
    settings: Mapping[str, object],
    center_mode: str,
    log_dir: Path,
) -> PocketResult:
    paths = resolve_paths(settings)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(paths.launcher_python),
        str(paths.script),
        "--protein",
        str(protein_pdb),
        "--output-dir",
        str(output_dir),
        "--pocketminer-checkpoint",
        str(paths.pocketminer_checkpoint),
        "--checkpoint",
        str(paths.classifier_checkpoint),
        "--feature-python",
        str(paths.feature_python),
        "--classifier-python",
        str(paths.classifier_python),
        "--classifier-cuda-visible-devices",
        str(settings.get("classifier_cuda_visible_devices") or ""),
        "--device",
        str(settings.get("device") or "cpu"),
        "--chains",
        chain,
        "--overwrite",
    ]
    threshold = settings.get("threshold")
    if threshold is not None:
        command.extend(["--threshold", str(threshold)])
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    run_command(
        command,
        cwd=paths.root,
        environment=environment,
        timeout=7200,
        log_dir=log_dir,
    )

    sample_id = state_id
    pocket_pdb = output_dir / f"{sample_id}_predicted_pocket.pdb"
    residue_table = output_dir / f"{sample_id}_residues.csv"
    summary_path = output_dir / f"{sample_id}_summary.json"
    for path in (pocket_pdb, residue_table, summary_path):
        if not path.is_file():
            raise ExternalToolError(f"CYPocket did not produce the required output: {path}")
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("sample_id") != sample_id:
        raise ExternalToolError("CYPocket summary sample_id does not match the state")
    if summary.get("protein_pdb_sha256") != sha256_file(protein_pdb):
        raise ExternalToolError("CYPocket summary is not bound to the protein input hash")
    count = int(summary.get("n_predicted_pocket", -1))
    if count < 1:
        raise ExternalToolError(f"CYPocket predicted no pocket residues for {state_id}")
    center, coordinates, parsed_count = _center(pocket_pdb, center_mode)
    if parsed_count != count:
        raise ExternalToolError(
            f"CYPocket pocket residue count mismatch: summary={count}, PDB={parsed_count}"
        )
    return PocketResult(
        pocket_pdb=pocket_pdb,
        residue_table=residue_table,
        summary_json=summary_path,
        center=center,
        coordinates=coordinates,
        residue_count=count,
    )


def software_record(settings: Mapping[str, object]) -> dict:
    paths = resolve_paths(settings)
    result = run_command(
        [
            str(paths.launcher_python),
            "-c",
            (
                "import json,sys; sys.path.insert(0,sys.argv[1]); "
                "import p450pocket; print(json.dumps({"
                "'version':getattr(p450pocket,'__version__','unknown')}))"
            ),
            str(paths.root),
        ],
        timeout=10,
    )
    return {
        "python": str(paths.launcher_python),
        "root": str(paths.root),
        "entry_sha256": sha256_file(paths.script),
        "classifier_checkpoint_sha256": sha256_file(paths.classifier_checkpoint),
        "pocketminer_checkpoint_sha256": checkpoint_prefix_sha256(
            paths.pocketminer_checkpoint
        ),
        "version_probe": result.stdout.strip(),
    }
