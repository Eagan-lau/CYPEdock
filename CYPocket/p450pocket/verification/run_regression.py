#!/usr/bin/env python3
"""Run the self-contained CYPocket protein/pocket regression fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_ROOT = Path(
    os.environ.get("P450_MODEL_DIR", PROJECT_ROOT.parent / "P450model")
).expanduser().resolve()
FIXTURE_ROOT = Path(__file__).resolve().parent / "regression_v1"
SAMPLE_ID = "AF-A0A010RW94-F1-model_v4"
PROTEIN_NAME = f"{SAMPLE_ID}_protein.pdb"
POCKET_NAME = f"{SAMPLE_ID}_pocket.pdb"
BEST_SHA256 = "548135d13a62b0d9ce9fa925563f8a52b2f61f18271f599c5eec41a6e1596690"
POCKETMINER_SHA256 = "c3e659f33b8caf4fd1c2a79ca8f9b24a106628d81296a5fcc877ae3801512c56"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recompute one real P450 fixture and compare all core artifacts"
    )
    parser.add_argument(
        "--feature-python",
        default=os.environ.get("P450_FEATURE_PYTHON", sys.executable),
    )
    parser.add_argument(
        "--classifier-python",
        default=os.environ.get("P450_CLASSIFIER_PYTHON", sys.executable),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--classifier-cuda-visible-devices",
        default=os.environ.get("P450_CLASSIFIER_GPU", ""),
    )
    parser.add_argument(
        "--output-dir",
        help="Keep generated artifacts here; otherwise a temporary directory is used",
    )
    parser.add_argument("--feature-atol", type=float, default=1.0e-3)
    parser.add_argument("--probability-atol", type=float, default=2.0e-5)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_digest(prefix: Path) -> str:
    files = (Path(f"{prefix}.index"), *sorted(prefix.parent.glob(f"{prefix.name}.data-*")))
    digest = hashlib.sha256()
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"missing PocketMiner checkpoint shard: {path}")
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _run(command: Sequence[str], environment: Dict[str, str] | None = None) -> None:
    print("RUN " + " ".join(command), flush=True)
    subprocess.run(
        list(command),
        cwd=str(PROJECT_ROOT),
        env=environment,
        check=True,
    )


def _compare(work_root: Path, feature_atol: float, probability_atol: float) -> Dict[str, Any]:
    expected = FIXTURE_ROOT / "expected"
    predicted = work_root / "prediction"
    prepared = work_root / "prepared"

    generated_features = np.load(
        predicted / "features" / f"{SAMPLE_ID}_protein-tokens.npy"
    )
    prepared_features = np.load(
        prepared / "test" / "tokens" / f"{SAMPLE_ID}_protein-tokens.npy"
    )
    expected_features = np.load(expected / "features.npy")
    generated_labels = np.load(
        prepared / "test" / "labels" / f"{SAMPLE_ID}_labels.npy"
    )
    expected_labels = np.load(expected / "labels.npy")
    generated_predictions = np.load(predicted / f"{SAMPLE_ID}_predictions.npy")
    expected_predictions = np.load(expected / "predictions.npy")
    generated_probabilities = np.load(predicted / f"{SAMPLE_ID}_probabilities.npy")
    expected_probabilities = np.load(expected / "probabilities.npy")

    feature_max_abs = float(np.max(np.abs(generated_features - expected_features)))
    probability_max_abs = float(
        np.max(np.abs(generated_probabilities - expected_probabilities))
    )
    residue_map = predicted / "features" / f"{SAMPLE_ID}_protein-residues.tsv"
    predicted_pocket = predicted / f"{SAMPLE_ID}_predicted_pocket.pdb"
    generated_csv = predicted / f"{SAMPLE_ID}_residues.csv"

    checks = {
        "feature_shape": list(generated_features.shape),
        "features_match_expected": bool(feature_max_abs <= feature_atol),
        "features_match_raw_prepare_bitwise": bool(
            np.array_equal(generated_features, prepared_features)
        ),
        "feature_max_abs": feature_max_abs,
        "labels_match_expected_bitwise": bool(
            np.array_equal(generated_labels, expected_labels)
        ),
        "predictions_match_expected_bitwise": bool(
            np.array_equal(generated_predictions, expected_predictions)
        ),
        "probabilities_match_expected": bool(probability_max_abs <= probability_atol),
        "probability_max_abs": probability_max_abs,
        "residue_map_match_expected_bitwise": bool(
            residue_map.read_bytes() == (expected / "residue_map.tsv").read_bytes()
        ),
        "predicted_pocket_match_expected_bitwise": bool(
            predicted_pocket.read_bytes()
            == (expected / "predicted_pocket.pdb").read_bytes()
        ),
        "residue_csv_exists": generated_csv.is_file(),
        "n_reference_positive": int(generated_labels.sum()),
        "n_predicted_positive": int(generated_predictions.sum()),
        "generated_feature_sha256": _sha256(
            predicted / "features" / f"{SAMPLE_ID}_protein-tokens.npy"
        ),
        "generated_prediction_sha256": _sha256(
            predicted / f"{SAMPLE_ID}_predictions.npy"
        ),
    }
    required = [
        value
        for key, value in checks.items()
        if key.endswith("_bitwise")
        or key.endswith("_expected")
        or key == "residue_csv_exists"
    ]
    checks["passed"] = all(bool(value) for value in required)
    return checks


def _execute(args: argparse.Namespace, work_root: Path) -> Dict[str, Any]:
    input_root = FIXTURE_ROOT / "input"
    protein = input_root / PROTEIN_NAME
    pocket = input_root / POCKET_NAME
    expected_files = (
        protein,
        pocket,
        FIXTURE_ROOT / "expected" / "features.npy",
        FIXTURE_ROOT / "expected" / "labels.npy",
        FIXTURE_ROOT / "expected" / "predictions.npy",
        FIXTURE_ROOT / "expected" / "probabilities.npy",
    )
    missing = [str(path) for path in expected_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"regression fixture is incomplete: {missing}")

    best = MODEL_ROOT / "best_protein_distilbert_classifier.pth"
    pocketminer = MODEL_ROOT / "pocketminer"
    if _sha256(best) != BEST_SHA256:
        raise RuntimeError("best classifier SHA-256 does not match the release fixture")
    if _checkpoint_digest(pocketminer) != POCKETMINER_SHA256:
        raise RuntimeError("PocketMiner composite SHA-256 does not match the release fixture")

    raw = work_root / "raw" / "test"
    (raw / "protein").mkdir(parents=True, exist_ok=True)
    (raw / "pocket").mkdir(parents=True, exist_ok=True)
    shutil.copy2(protein, raw / "protein" / PROTEIN_NAME)
    shutil.copy2(pocket, raw / "pocket" / POCKET_NAME)

    prediction_dir = work_root / "prediction"
    classifier_environment = os.environ.copy()
    classifier_environment["CUDA_VISIBLE_DEVICES"] = args.classifier_cuda_visible_devices
    feature_environment = os.environ.copy()
    feature_environment["CUDA_VISIBLE_DEVICES"] = "-1"
    _run(
        [
            sys.executable,
            str(PROJECT_ROOT / "predict_pdb.py"),
            "--protein",
            str(protein),
            "--output-dir",
            str(prediction_dir),
            "--feature-python",
            args.feature_python,
            "--classifier-python",
            args.classifier_python,
            "--classifier-cuda-visible-devices",
            args.classifier_cuda_visible_devices,
            "--device",
            args.device,
            "--overwrite",
        ],
        classifier_environment,
    )
    _run(
        [
            args.feature_python,
            str(PROJECT_ROOT / "prepare_dataset.py"),
            "--dataset-root",
            str(work_root / "raw"),
            "--output-root",
            str(work_root / "prepared"),
            "--splits",
            "test",
            "--overwrite",
        ],
        feature_environment,
    )
    result = _compare(work_root, args.feature_atol, args.probability_atol)
    result.update(
        {
            "sample_id": SAMPLE_ID,
            "work_root": str(work_root),
            "classifier_sha256": BEST_SHA256,
            "pocketminer_sha256": POCKETMINER_SHA256,
            "feature_atol": args.feature_atol,
            "probability_atol": args.probability_atol,
        }
    )
    return result


def main() -> int:
    args = _parser().parse_args()
    if args.feature_atol < 0 or args.probability_atol < 0:
        raise ValueError("comparison tolerances must be non-negative")
    if args.output_dir:
        work_root = Path(args.output_dir).expanduser().resolve()
        work_root.mkdir(parents=True, exist_ok=True)
        result = _execute(args, work_root)
    else:
        with tempfile.TemporaryDirectory(prefix="cypocket-regression-") as temporary:
            result = _execute(args, Path(temporary))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
