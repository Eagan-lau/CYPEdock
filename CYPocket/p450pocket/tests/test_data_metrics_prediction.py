from pathlib import Path

import numpy as np
import pytest
import torch

from p450pocket.data import ProteinFeatureDataset, collate_proteins
from p450pocket.manifest import (
    SampleRecord,
    load_manifest,
    verify_record_hashes,
    write_manifest,
)
from p450pocket.metrics import binary_metrics, macro_protein_metrics
from p450pocket.prediction import (
    PredictionResult,
    load_feature_array,
    write_prediction_outputs,
)
from p450pocket.provenance import sha256_file


def test_manifest_dataset_and_padding(tmp_path: Path) -> None:
    records = []
    for sample_id, length in (("a", 3), ("b", 5)):
        feature_path = tmp_path / f"{sample_id}_features.npy"
        label_path = tmp_path / f"{sample_id}_labels.npy"
        np.save(feature_path, np.full((length, 100), length, dtype=np.float32))
        np.save(label_path, np.arange(length, dtype=np.int64) % 2)
        records.append(
            SampleRecord(sample_id, "train", str(feature_path), str(label_path))
        )
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(records, manifest)
    loaded = load_manifest(manifest, splits=["train"])
    batch = collate_proteins([ProteinFeatureDataset(loaded)[0], ProteinFeatureDataset(loaded)[1]])
    assert tuple(batch["features"].shape) == (2, 5, 100)
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]
    assert batch["labels"][0].tolist() == [0, 1, 0, -1, -1]


def test_manifest_hash_verification_detects_changed_arrays(tmp_path: Path) -> None:
    feature_path = tmp_path / "features.npy"
    label_path = tmp_path / "labels.npy"
    np.save(feature_path, np.zeros((2, 100), dtype=np.float32))
    np.save(label_path, np.array([0, 1], dtype=np.int64))
    record = SampleRecord(
        "sample",
        "train",
        str(feature_path),
        str(label_path),
        feature_sha256=sha256_file(feature_path),
        label_sha256=sha256_file(label_path),
    )
    assert verify_record_hashes([record])["n_files_checked"] == 2
    np.save(label_path, np.array([1, 1], dtype=np.int64))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_record_hashes([record])


def test_metrics_use_continuous_probabilities() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.35, 0.8])
    metrics = binary_metrics(labels, probabilities, threshold=0.5)
    assert metrics["confusion_matrix"] == [[2, 0], [1, 1]]
    assert metrics["roc_auc"] == pytest.approx(0.75)
    assert metrics["pr_auc"] == pytest.approx(5.0 / 6.0)


def test_macro_auc_skips_proteins_where_auc_is_undefined() -> None:
    metrics = macro_protein_metrics(
        [np.array([0, 1]), np.array([0, 0])],
        [np.array([0.1, 0.9]), np.array([0.2, 0.3])],
    )
    assert metrics["macro_roc_auc"] == pytest.approx(1.0)
    assert metrics["macro_pr_auc"] == pytest.approx(1.0)
    assert metrics["n_proteins_with_roc_auc"] == 1
    assert metrics["n_proteins_with_pr_auc"] == 1


def test_feature_array_and_prediction_validation(tmp_path: Path) -> None:
    path = tmp_path / "features.npy"
    np.save(path, np.zeros((1, 4, 100), dtype=np.float32))
    assert load_feature_array(path).shape == (4, 100)
    result = PredictionResult(
        probabilities=np.array([0.2, 0.9], dtype=np.float32),
        labels=np.array([0, 1]),
        decision_rule="argmax",
        threshold=0.5,
    )
    assert result.labels.tolist() == [0, 1]
    with pytest.raises(ValueError, match="NaN"):
        PredictionResult(
            probabilities=np.array([np.nan]),
            labels=np.array([0]),
            decision_rule="argmax",
            threshold=0.5,
        )


def test_prediction_outputs_require_explicit_overwrite(tmp_path: Path) -> None:
    result = PredictionResult(
        probabilities=np.array([0.2, 0.9], dtype=np.float32),
        labels=np.array([0, 1]),
        decision_rule="argmax",
        threshold=0.5,
    )
    write_prediction_outputs(tmp_path, "sample", result)
    with pytest.raises(FileExistsError, match="--overwrite"):
        write_prediction_outputs(tmp_path, "sample", result)
    write_prediction_outputs(tmp_path, "sample", result, overwrite=True)
