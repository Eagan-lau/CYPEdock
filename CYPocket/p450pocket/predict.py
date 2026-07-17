#!/usr/bin/env python3
"""Run the existing classifier on a compatible 100D residue-feature array."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

from p450pocket.checkpoint import load_model
from p450pocket.prediction import (
    load_feature_array,
    parse_optional_structure,
    predict_array,
    write_prediction_outputs,
)
from p450pocket.training import resolve_device


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict pocket residues from PocketMiner-compatible [L,100] features"
    )
    parser.add_argument("--features", required=True, help=".npy array shaped [L,100] or [1,L,100]")
    parser.add_argument("--checkpoint", required=True, help="P450 PyTorch classifier checkpoint")
    parser.add_argument(
        "--feature-metadata",
        help="Optional JSON sidecar produced by extract_features.py",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--protein",
        help="Optional source PDB; enables residue numbering and predicted-pocket PDB output",
    )
    parser.add_argument("--model-index", type=int, default=0)
    parser.add_argument(
        "--chains",
        nargs="*",
        default=None,
        help="Optional author chain IDs; must match feature extraction",
    )
    parser.add_argument(
        "--modified-residue",
        action="append",
        default=[],
        metavar="ORIGINAL=CANONICAL",
        help="Residue mapping used during feature extraction; may be repeated",
    )
    parser.add_argument(
        "--allow-multiple-models",
        action="store_true",
        help="Select --model-index from an ensemble instead of rejecting it",
    )
    parser.add_argument("--sample-id", help="Output basename (default: inferred from feature file)")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument(
        "--decision-rule",
        choices=["checkpoint", "argmax", "probability-threshold"],
        default="checkpoint",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help=(
            "Probability threshold. With the default --decision-rule checkpoint, "
            "supplying this option selects probability-threshold decisions."
        ),
    )
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _parse_modified_residues(values: Iterable[str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"modified residue mapping must be ORIGINAL=CANONICAL: {value}"
            )
        original, canonical = value.split("=", 1)
        original, canonical = original.strip().upper(), canonical.strip().upper()
        if not original or not canonical:
            raise ValueError(f"invalid modified residue mapping: {value}")
        result[original] = canonical
    return result


def _infer_sample_id(path: Path) -> str:
    stem = path.stem
    for suffix in ("_protein-tokens", "-tokens", "_tokens", "_features"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def main() -> None:
    args = _parser().parse_args()
    if args.chains is not None and len(args.chains) != 1:
        raise ValueError("--chains must select exactly one chain for this single-chain model")
    feature_path = Path(args.features).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    features = load_feature_array(feature_path)
    device = resolve_device(args.device)
    model, metadata = load_model(checkpoint_path, device=device, strict=True)
    checkpoint_rule = str(metadata.get("decision_rule", "argmax"))
    decision_rule = checkpoint_rule if args.decision_rule == "checkpoint" else args.decision_rule
    if args.threshold is not None and args.decision_rule == "checkpoint":
        decision_rule = "probability-threshold"
    elif args.threshold is not None and decision_rule == "argmax":
        raise ValueError("--threshold cannot be combined with explicit --decision-rule argmax")
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(metadata.get("threshold", 0.5))
    )
    modified_residue_map = _parse_modified_residues(args.modified_residue)
    structure = parse_optional_structure(
        args.protein,
        expected_length=features.shape[0],
        model_index=args.model_index,
        chains=args.chains,
        modified_residue_map=modified_residue_map,
        reject_multiple_models=not args.allow_multiple_models,
    )
    if structure is not None and len(structure.chains) != 1:
        raise ValueError(
            f"protein contains {len(structure.chains)} chains; select exactly one with --chains"
        )
    result = predict_array(
        model,
        features,
        device,
        decision_rule=decision_rule,
        threshold=threshold,
        mixed_precision=args.mixed_precision,
    )
    sample_id = args.sample_id or _infer_sample_id(feature_path)
    outputs = write_prediction_outputs(
        args.output_dir,
        sample_id,
        result,
        structure=structure,
        checkpoint_path=checkpoint_path,
        feature_path=feature_path,
        feature_metadata_path=args.feature_metadata,
        overwrite=args.overwrite,
    )
    print(json.dumps(outputs, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
