#!/usr/bin/env python3
"""One-command PDB inference using isolated feature and classifier processes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from p450pocket.paths import default_model_artifact


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CLASSIFIER = default_model_artifact("best_protein_distilbert_classifier.pth")
DEFAULT_POCKETMINER = default_model_artifact("pocketminer")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract PocketMiner features on CPU, then predict P450 pocket residues "
            "in a separate PyTorch process."
        )
    )
    parser.add_argument("--protein", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--pocketminer-checkpoint",
        default=os.environ.get(
            "P450_POCKETMINER_CHECKPOINT", str(DEFAULT_POCKETMINER)
        ),
        help="TensorFlow checkpoint prefix; may also set P450_POCKETMINER_CHECKPOINT",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("P450_CLASSIFIER_CHECKPOINT", str(DEFAULT_CLASSIFIER)),
        help="Classifier checkpoint; may also set P450_CLASSIFIER_CHECKPOINT",
    )
    parser.add_argument(
        "--feature-python",
        default=os.environ.get("P450_FEATURE_PYTHON", sys.executable),
    )
    parser.add_argument(
        "--classifier-python",
        default=os.environ.get("P450_CLASSIFIER_PYTHON", sys.executable),
    )
    parser.add_argument(
        "--classifier-cuda-visible-devices",
        default=os.environ.get("P450_CLASSIFIER_GPU", ""),
        help="Examples: empty for CPU, 0, or 1",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--chains", nargs="*", default=None)
    parser.add_argument("--model-index", type=int, default=0)
    parser.add_argument(
        "--modified-residue",
        action="append",
        default=[],
        metavar="ORIGINAL=CANONICAL",
        help="Explicit noncanonical residue mapping; may be repeated",
    )
    parser.add_argument(
        "--allow-multiple-models",
        action="store_true",
        help="Select --model-index from an ensemble instead of rejecting it",
    )
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _sample_id(path: Path) -> str:
    stem = path.stem
    for suffix in ("_protein", "-protein"):
        if stem.lower().endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _run(command: list[str], environment: dict[str, str]) -> None:
    print("RUN " + " ".join(command), flush=True)
    subprocess.run(command, cwd=str(PROJECT_ROOT), env=environment, check=True)


def main() -> None:
    args = _parser().parse_args()
    if args.chains is not None and len(args.chains) != 1:
        raise ValueError("--chains must select exactly one chain for this single-chain model")
    if not args.pocketminer_checkpoint:
        raise ValueError(
            "PocketMiner checkpoint is required. Pass --pocketminer-checkpoint PREFIX "
            "(PREFIX.index and PREFIX.data-* must both exist)."
        )
    protein = Path(args.protein).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if not protein.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(
            f"protein/checkpoint missing: protein={protein}, checkpoint={checkpoint}"
        )
    output.mkdir(parents=True, exist_ok=True)
    feature_dir = output / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    sample_id = _sample_id(protein)
    feature_path = feature_dir / f"{protein.stem}-tokens.npy"
    feature_metadata_path = feature_dir / f"{protein.stem}-features.json"

    extract_command = [
        args.feature_python,
        str(PROJECT_ROOT / "extract_features.py"),
        "--protein",
        str(protein),
        "--output-dir",
        str(feature_dir),
        "--checkpoint-prefix",
        args.pocketminer_checkpoint,
        "--model-index",
        str(args.model_index),
    ]
    if args.chains is not None:
        extract_command.extend(["--chains", *args.chains])
    for mapping in args.modified_residue:
        extract_command.extend(["--modified-residue", mapping])
    if args.allow_multiple_models:
        extract_command.append("--allow-multiple-models")
    if args.overwrite:
        extract_command.append("--overwrite")
    feature_environment = os.environ.copy()
    feature_environment["CUDA_VISIBLE_DEVICES"] = "-1"
    _run(extract_command, feature_environment)

    predict_command = [
        args.classifier_python,
        str(PROJECT_ROOT / "predict.py"),
        "--features",
        str(feature_path),
        "--feature-metadata",
        str(feature_metadata_path),
        "--protein",
        str(protein),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output),
        "--sample-id",
        sample_id,
        "--device",
        args.device,
        "--model-index",
        str(args.model_index),
    ]
    if args.chains is not None:
        predict_command.extend(["--chains", *args.chains])
    for mapping in args.modified_residue:
        predict_command.extend(["--modified-residue", mapping])
    if args.allow_multiple_models:
        predict_command.append("--allow-multiple-models")
    if args.threshold is not None:
        predict_command.extend(
            ["--decision-rule", "probability-threshold", "--threshold", str(args.threshold)]
        )
    if args.mixed_precision:
        predict_command.append("--mixed-precision")
    if args.overwrite:
        predict_command.append("--overwrite")
    classifier_environment = os.environ.copy()
    classifier_environment["CUDA_VISIBLE_DEVICES"] = args.classifier_cuda_visible_devices
    _run(predict_command, classifier_environment)

    summary_path = output / f"{sample_id}_summary.json"
    print(
        json.dumps(
            {
                "sample_id": sample_id,
                "feature_path": str(feature_path),
                "summary": str(summary_path),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
