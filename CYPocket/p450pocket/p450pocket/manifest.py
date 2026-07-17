"""Framework-independent dataset manifests and processed-data discovery."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .provenance import sha256_file


@dataclass
class SampleRecord:
    sample_id: str
    split: str
    feature_path: str
    label_path: Optional[str] = None
    protein_path: Optional[str] = None
    pocket_path: Optional[str] = None
    residue_map_path: Optional[str] = None
    n_residues: Optional[int] = None
    n_positive: Optional[int] = None
    sequence_sha256: Optional[str] = None
    feature_sha256: Optional[str] = None
    label_sha256: Optional[str] = None
    feature_extractor_sha256: Optional[str] = None
    protein_sha256: Optional[str] = None
    pocket_sha256: Optional[str] = None
    schema_version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


def _sample_id(filename: str, suffix: str) -> str:
    if not filename.endswith(suffix):
        raise ValueError(f"{filename!r} does not end with {suffix!r}")
    return filename[: -len(suffix)]


def discover_processed_split(split_dir: os.PathLike[str] | str, split: str) -> List[SampleRecord]:
    base = Path(split_dir)
    feature_dir = base / "tokens"
    label_dir = base / "labels"
    protein_dir = base / "protein"
    pocket_dir = base / "pocket"
    if not feature_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"expected tokens/ and labels/ under {base}")
    features = {
        _sample_id(path.name, "_protein-tokens.npy"): path
        for path in feature_dir.glob("*_protein-tokens.npy")
    }
    labels = {
        _sample_id(path.name, "_labels.npy"): path
        for path in label_dir.glob("*_labels.npy")
    }
    if features.keys() != labels.keys():
        missing_features = sorted(labels.keys() - features.keys())[:10]
        missing_labels = sorted(features.keys() - labels.keys())[:10]
        raise ValueError(
            f"feature/label IDs differ; missing_features={missing_features}, "
            f"missing_labels={missing_labels}"
        )
    records: List[SampleRecord] = []
    for sample_id in sorted(features):
        feature = np.load(features[sample_id], mmap_mode="r")
        label = np.load(labels[sample_id], mmap_mode="r")
        feature_shape = feature.shape[1:] if feature.ndim == 3 and feature.shape[0] == 1 else feature.shape
        if len(feature_shape) != 2 or feature_shape[1] != 100:
            raise ValueError(f"invalid feature shape for {sample_id}: {feature.shape}")
        if feature_shape[0] < 1 or not np.issubdtype(feature.dtype, np.floating):
            raise ValueError(f"features must be a non-empty floating array for {sample_id}")
        if not np.isfinite(feature).all():
            raise ValueError(f"features contain NaN or infinity for {sample_id}")
        if label.ndim != 1 or feature_shape[0] != label.shape[0]:
            raise ValueError(
                f"length mismatch for {sample_id}: features={feature.shape}, labels={label.shape}"
            )
        if not np.issubdtype(label.dtype, np.integer):
            raise ValueError(f"labels must have integer dtype for {sample_id}: {label.dtype}")
        unique = np.unique(label)
        if not set(unique.tolist()).issubset({0, 1}):
            raise ValueError(f"invalid labels for {sample_id}: {unique.tolist()}")
        protein = protein_dir / f"{sample_id}_protein.pdb"
        pocket = pocket_dir / f"{sample_id}_pocket.pdb"
        records.append(
            SampleRecord(
                sample_id=sample_id,
                split=split,
                feature_path=str(features[sample_id].resolve()),
                label_path=str(labels[sample_id].resolve()),
                protein_path=str(protein.resolve()) if protein.exists() else None,
                pocket_path=str(pocket.resolve()) if pocket.exists() else None,
                n_residues=int(label.shape[0]),
                n_positive=int(np.count_nonzero(label == 1)),
                feature_sha256=sha256_file(features[sample_id]),
                label_sha256=sha256_file(labels[sample_id]),
                protein_sha256=sha256_file(protein) if protein.exists() else None,
                pocket_sha256=sha256_file(pocket) if pocket.exists() else None,
            )
        )
    return records


def write_manifest(records: Iterable[SampleRecord], output_path: os.PathLike[str] | str) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_root = destination.parent.resolve()
    path_fields = (
        "feature_path",
        "label_path",
        "protein_path",
        "pocket_path",
        "residue_map_path",
    )
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            values = record.to_dict()
            # Paths inside the prepared dataset are made relative so copying
            # the dataset and manifest together does not invalidate the run.
            for field in path_fields:
                value = values.get(field)
                if not value:
                    continue
                try:
                    values[field] = str(Path(value).resolve().relative_to(manifest_root))
                except ValueError:
                    # Raw source PDBs can intentionally live outside the
                    # prepared dataset and remain explicit absolute paths.
                    values[field] = str(Path(value).resolve())
            handle.write(json.dumps(values, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, destination)


def load_manifest(
    path: os.PathLike[str] | str,
    splits: Optional[Sequence[str]] = None,
) -> List[SampleRecord]:
    manifest = Path(path).resolve()
    wanted = set(splits) if splits else None
    records: List[SampleRecord] = []
    path_fields = ("feature_path", "label_path", "protein_path", "pocket_path", "residue_map_path")
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            values = json.loads(line)
            if wanted is not None and values.get("split") not in wanted:
                continue
            for field in path_fields:
                value = values.get(field)
                if value and not os.path.isabs(value):
                    values[field] = str((manifest.parent / value).resolve())
            try:
                records.append(SampleRecord(**values))
            except TypeError as error:
                raise ValueError(f"invalid manifest line {line_number}: {error}") from error
    if not records:
        raise ValueError(f"no matching samples in {manifest}")
    return records


def verify_record_hashes(
    records: Sequence[SampleRecord],
    *,
    require_feature_and_label_hashes: bool = True,
) -> Dict[str, int]:
    """Verify immutable training arrays against hashes stored in a manifest."""

    checked = 0
    for record in records:
        fields = (("feature", record.feature_path, record.feature_sha256),)
        if record.label_path:
            fields += (("label", record.label_path, record.label_sha256),)
        for kind, path_value, expected in fields:
            path = Path(path_value)
            if not path.is_file():
                raise FileNotFoundError(f"missing {kind} file for {record.sample_id}: {path}")
            if expected is None:
                if require_feature_and_label_hashes:
                    raise ValueError(
                        f"manifest has no {kind}_sha256 for {record.sample_id}; rebuild "
                        "it with prepare_dataset.py or explicitly skip integrity checking"
                    )
                continue
            actual = sha256_file(path)
            if actual != expected:
                raise ValueError(
                    f"{kind} SHA-256 mismatch for {record.sample_id}: "
                    f"manifest={expected}, actual={actual}"
                )
            checked += 1
    return {"n_records": len(records), "n_files_checked": checked}
