"""Validated inference and machine-readable residue-level output."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .checkpoint import sha256_file
from .model import ResidueBertClassifier, positive_probabilities
from .structure import ResidueKey, StructureRecord, parse_feature_protein


@dataclass(frozen=True)
class PredictionResult:
    probabilities: np.ndarray
    labels: np.ndarray
    decision_rule: str
    threshold: float

    def __post_init__(self) -> None:
        probabilities = np.asarray(self.probabilities, dtype=np.float32).copy()
        labels = np.asarray(self.labels, dtype=np.int64).copy()
        if probabilities.ndim != 1 or labels.shape != probabilities.shape:
            raise ValueError(
                f"prediction arrays must be aligned 1D arrays, got "
                f"{probabilities.shape} and {labels.shape}"
            )
        if not np.isfinite(probabilities).all():
            raise ValueError("probabilities contain NaN or infinity")
        if not np.isin(labels, (0, 1)).all():
            raise ValueError("predicted labels must be binary")
        probabilities.setflags(write=False)
        labels.setflags(write=False)
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "labels", labels)


def load_feature_array(path: os.PathLike[str] | str) -> np.ndarray:
    values = np.load(path)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 2 or values.shape[1] != 100:
        raise ValueError(f"features must have shape [L, 100] or [1, L, 100], got {values.shape}")
    if values.shape[0] < 1:
        raise ValueError("feature array contains no residues")
    if not np.isfinite(values).all():
        raise ValueError("features contain NaN or infinity")
    return np.asarray(values, dtype=np.float32)


def predict_array(
    model: ResidueBertClassifier,
    features: np.ndarray,
    device: torch.device,
    *,
    decision_rule: str = "argmax",
    threshold: float = 0.5,
    mixed_precision: bool = False,
) -> PredictionResult:
    values = np.asarray(features)
    if values.ndim != 2 or values.shape[1] != model.model_config.feature_dim:
        raise ValueError(
            f"features must have shape [L, {model.model_config.feature_dim}], got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("features contain NaN or infinity")
    if decision_rule not in {"argmax", "probability-threshold"}:
        raise ValueError("decision_rule must be argmax or probability-threshold")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32)).unsqueeze(0).to(device)
    mask = torch.ones((1, tensor.shape[1]), dtype=torch.long, device=device)
    model.eval()
    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type,
            enabled=mixed_precision and device.type == "cuda",
        ):
            logits = model.forward_chunked(tensor, mask)
        probabilities = positive_probabilities(logits).float().cpu().numpy()[0]
        if decision_rule == "argmax":
            labels = torch.argmax(logits, dim=-1).cpu().numpy()[0]
        else:
            labels = (probabilities >= threshold).astype(np.int64)
    return PredictionResult(probabilities, labels, decision_rule, threshold)


def _residue_rows(
    sample_id: str,
    result: PredictionResult,
    structure: Optional[StructureRecord],
):
    if structure is not None and len(structure) != len(result.labels):
        raise ValueError(
            f"PDB/feature length mismatch: PDB={len(structure)}, features={len(result.labels)}"
        )
    for index, (probability, label) in enumerate(
        zip(result.probabilities, result.labels)
    ):
        if structure is None:
            yield {
                "sample_id": sample_id,
                "row_index": index,
                "model_index": "",
                "chain_id": "",
                "auth_seq_id": "",
                "insertion_code": "",
                "residue_name": "",
                "pocket_probability": float(probability),
                "predicted_label": int(label),
            }
        else:
            key = structure.residues[index].key
            yield {
                "sample_id": sample_id,
                "row_index": index,
                "model_index": key.model_index,
                "chain_id": key.chain_id,
                "auth_seq_id": key.auth_seq_id,
                "insertion_code": key.insertion_code,
                "residue_name": key.residue_name,
                "pocket_probability": float(probability),
                "predicted_label": int(label),
            }


def _write_predicted_pocket(
    input_pdb: Path,
    output_pdb: Path,
    selected_keys: Sequence[ResidueKey],
) -> None:
    """Preserve original atom records for predicted polymer residues."""

    selected = {
        (
            key.chain_id,
            key.auth_seq_id,
            key.insertion_code,
            key.residue_name,
        )
        for key in selected_keys
    }
    lines = []
    with input_pdb.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line[:6].strip() not in {"ATOM", "HETATM"} or len(line) < 27:
                continue
            try:
                identity = (
                    line[21].strip(),
                    int(line[22:26]),
                    line[26].strip(),
                    line[17:20].strip().upper(),
                )
            except ValueError:
                continue
            if identity in selected:
                lines.append(line.rstrip("\n") + "\n")
    if selected and not lines:
        raise ValueError(f"no selected residue atom records could be recovered from {input_pdb}")
    temporary = output_pdb.with_suffix(output_pdb.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.writelines(lines)
        handle.write("END\n")
    os.replace(temporary, output_pdb)


def write_prediction_outputs(
    output_dir: os.PathLike[str] | str,
    sample_id: str,
    result: PredictionResult,
    *,
    structure: Optional[StructureRecord] = None,
    checkpoint_path: Optional[os.PathLike[str] | str] = None,
    feature_path: Optional[os.PathLike[str] | str] = None,
    feature_metadata_path: Optional[os.PathLike[str] | str] = None,
    overwrite: bool = False,
) -> Dict[str, str]:
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    feature_digest = sha256_file(feature_path) if feature_path else None
    metadata_source = None
    feature_metadata_values: Optional[Dict[str, Any]] = None
    if feature_metadata_path:
        if feature_digest is None:
            raise ValueError("feature_metadata_path requires feature_path")
        metadata_source = Path(feature_metadata_path).expanduser().resolve()
        if not metadata_source.is_file():
            raise FileNotFoundError(f"feature metadata not found: {metadata_source}")
        with metadata_source.open("r", encoding="utf-8") as handle:
            feature_metadata_values = json.load(handle)
        if not isinstance(feature_metadata_values, dict):
            raise ValueError("feature metadata must be a JSON object")
        if feature_metadata_values.get("features_sha256") != feature_digest:
            raise ValueError(
                "feature metadata SHA-256 does not match the supplied feature array"
            )
        if int(feature_metadata_values.get("n_residues", -1)) != len(result.labels):
            raise ValueError(
                "feature metadata residue count does not match the prediction array"
            )
        if not feature_metadata_values.get("checkpoint_sha256"):
            raise ValueError("feature metadata does not identify its extractor checkpoint")
    probability_path = destination / f"{sample_id}_probabilities.npy"
    label_path = destination / f"{sample_id}_predictions.npy"
    csv_path = destination / f"{sample_id}_residues.csv"
    summary_path = destination / f"{sample_id}_summary.json"
    pocket_path = (
        destination / f"{sample_id}_predicted_pocket.pdb"
        if structure is not None
        else None
    )
    planned = [probability_path, label_path, csv_path, summary_path]
    if pocket_path is not None:
        planned.append(pocket_path)
    existing = [path for path in planned if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"prediction outputs already exist: {existing}; pass --overwrite to replace them"
        )

    for path, values in (
        (probability_path, result.probabilities),
        (label_path, result.labels),
    ):
        with tempfile.NamedTemporaryFile(
            dir=destination, suffix=".npy", delete=False
        ) as handle:
            temporary = Path(handle.name)
            np.save(handle, values, allow_pickle=False)
        try:
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    rows = list(_residue_rows(sample_id, result, structure))
    csv_temporary = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with csv_temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(csv_temporary, csv_path)

    outputs = {
        "probabilities": str(probability_path),
        "predictions": str(label_path),
        "residue_table": str(csv_path),
        "summary": str(summary_path),
    }
    if structure is not None:
        assert pocket_path is not None
        selected_keys = [
            structure.residues[index].key
            for index, label in enumerate(result.labels)
            if label == 1
        ]
        _write_predicted_pocket(structure.source_path, pocket_path, selected_keys)
        outputs["predicted_pocket_pdb"] = str(pocket_path)

    summary: Dict[str, Any] = {
        "sample_id": sample_id,
        "n_residues": int(len(result.labels)),
        "n_predicted_pocket": int(result.labels.sum()),
        "decision_rule": result.decision_rule,
        "threshold": result.threshold,
        "outputs": outputs,
    }
    if checkpoint_path:
        summary["checkpoint"] = str(Path(checkpoint_path).resolve())
        summary["checkpoint_sha256"] = sha256_file(checkpoint_path)
    if feature_path:
        summary["features"] = str(Path(feature_path).resolve())
        summary["features_sha256"] = feature_digest
    if feature_metadata_path:
        assert metadata_source is not None
        assert feature_metadata_values is not None
        summary["feature_metadata"] = str(metadata_source)
        summary["feature_metadata_sha256"] = sha256_file(metadata_source)
        summary["feature_extractor"] = {
            "checkpoint_prefix": feature_metadata_values.get("checkpoint_prefix"),
            "checkpoint_sha256": feature_metadata_values["checkpoint_sha256"],
            "software": feature_metadata_values.get("software"),
        }
    if structure is not None:
        summary["protein_pdb"] = str(structure.source_path)
        summary["protein_pdb_sha256"] = sha256_file(structure.source_path)
        summary["chains"] = list(structure.chains)
        summary["structure_warnings"] = list(structure.warnings)

    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, summary_path)
    return outputs


def parse_optional_structure(
    protein_pdb: Optional[os.PathLike[str] | str],
    expected_length: int,
    *,
    model_index: int = 0,
    chains: Optional[Sequence[str]] = None,
    modified_residue_map: Optional[Mapping[str, str]] = None,
    reject_multiple_models: bool = True,
) -> Optional[StructureRecord]:
    if protein_pdb is None:
        return None
    structure = parse_feature_protein(
        protein_pdb,
        model_index=model_index,
        chains=chains,
        modified_residue_map=modified_residue_map,
        reject_multiple_models=reject_multiple_models,
    )
    if len(structure) != expected_length:
        raise ValueError(
            f"PDB has {len(structure)} feature residues but array has {expected_length} rows"
        )
    return structure
