#!/usr/bin/env python3
"""Extract legacy-compatible PocketMiner features from one or more PDBs."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from p450pocket.features import PocketMinerExtractor
from p450pocket.paths import default_model_artifact
from p450pocket.provenance import sha256_file
from p450pocket.structure import StructureRecord, parse_feature_protein


DEFAULT_POCKETMINER = default_model_artifact("pocketminer")


def _parse_modified_residues(values: Iterable[str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError(
                f"Modified residue mapping {value!r} must have ORIGINAL=CANONICAL form"
            )
        original, canonical = value.split("=", 1)
        original, canonical = original.strip().upper(), canonical.strip().upper()
        if not original or not canonical:
            raise argparse.ArgumentTypeError(
                f"Invalid modified residue mapping: {value!r}"
            )
        result[original] = canonical
    return result


def _input_files(inputs: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for value in inputs:
        path = Path(value).expanduser()
        if path.is_dir():
            files.extend(
                sorted(
                    candidate
                    for candidate in path.iterdir()
                    if candidate.is_file() and candidate.suffix.lower() == ".pdb"
                )
            )
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"Protein path does not exist: {path}")
    resolved = [path.resolve() for path in files]
    if not resolved:
        raise FileNotFoundError("No .pdb files were found")
    if len(set(resolved)) != len(resolved):
        raise ValueError("The same protein PDB was supplied more than once")
    return resolved


def _atomic_save_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npy", delete=False) as handle:
        temporary = Path(handle.name)
        np.save(handle, values, allow_pickle=False)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_residues(path: Path, structure: StructureRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".tsv", mode="w", newline="", delete=False
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "row_index",
                "model_index",
                "chain_id",
                "auth_seq_id",
                "insertion_code",
                "residue_name",
                "canonical_name",
                "one_letter",
            ]
        )
        for residue in structure.residues:
            writer.writerow(
                [
                    residue.row_index,
                    residue.key.model_index,
                    residue.key.chain_id,
                    residue.key.auth_seq_id,
                    residue.key.insertion_code,
                    residue.key.residue_name,
                    residue.canonical_name,
                    residue.one_letter,
                ]
            )
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, values: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(values, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract [residues, 100] PocketMiner features from protein PDB files"
    )
    parser.add_argument(
        "--protein",
        "--source-path",
        "--source_path",
        nargs="+",
        required=True,
        help="PDB file(s) or non-recursive directories containing PDB files",
    )
    parser.add_argument(
        "--output-dir",
        "--target-path",
        "--target_path",
        required=True,
        help="Directory for *-tokens.npy and *-residues.tsv",
    )
    parser.add_argument(
        "--checkpoint-prefix",
        "--checkpoint",
        default=os.environ.get("P450_POCKETMINER_CHECKPOINT", str(DEFAULT_POCKETMINER)),
        help=(
            "TensorFlow checkpoint prefix (without .index/.data suffix); may also "
            "set P450_POCKETMINER_CHECKPOINT"
        ),
    )
    parser.add_argument("--model-index", type=int, default=0)
    parser.add_argument(
        "--chains",
        nargs="*",
        default=None,
        help="Optional author chain IDs; use an empty quoted string for a blank chain",
    )
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
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        modified = _parse_modified_residues(args.modified_residue)
        proteins = _input_files(args.protein)
    except (ValueError, FileNotFoundError, argparse.ArgumentTypeError) as exc:
        parser.error(str(exc))
    if args.chains is not None and len(args.chains) != 1:
        parser.error("--chains must select exactly one chain for this single-chain model")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    extractor = PocketMinerExtractor(args.checkpoint_prefix)

    for protein_path in proteins:
        token_path = output_dir / f"{protein_path.stem}-tokens.npy"
        residue_path = output_dir / f"{protein_path.stem}-residues.tsv"
        metadata_path = output_dir / f"{protein_path.stem}-features.json"
        if not args.overwrite and any(
            path.exists() for path in (token_path, residue_path, metadata_path)
        ):
            raise FileExistsError(
                f"Output already exists for {protein_path}; pass --overwrite to replace it"
            )
        structure = parse_feature_protein(
            protein_path,
            model_index=args.model_index,
            chains=args.chains,
            modified_residue_map=modified,
            reject_multiple_models=not args.allow_multiple_models,
        )
        if len(structure.chains) != 1:
            raise ValueError(
                f"{protein_path} contains {len(structure.chains)} protein chains. "
                "The classifier was trained on single-chain AlphaFold structures; "
                "select exactly one chain with --chains CHAIN_ID."
            )
        result = extractor.extract(structure)
        _atomic_save_npy(token_path, result.features)
        _atomic_write_residues(residue_path, structure)
        _atomic_write_json(
            metadata_path,
            {
                "schema_version": 1,
                "protein_pdb": str(protein_path),
                "protein_pdb_sha256": sha256_file(protein_path),
                "features": str(token_path),
                "features_sha256": sha256_file(token_path),
                "residue_map": str(residue_path),
                "residue_map_sha256": sha256_file(residue_path),
                "n_residues": len(structure),
                "model_index": args.model_index,
                "chains": list(structure.chains),
                "modified_residue_map": modified,
                "coordinate_unit": "nanometre",
                "checkpoint_prefix": str(result.checkpoint_prefix),
                "checkpoint_sha256": result.checkpoint_sha256,
                "warnings": list(result.warnings),
                "software": {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "numpy": np.__version__,
                    "biopython": _distribution_version("biopython"),
                    "tensorflow": _distribution_version("tensorflow"),
                    "keras": _distribution_version("keras"),
                },
            },
        )
        print(
            f"{protein_path.name}: residues={len(structure)} "
            f"features={token_path} metadata={metadata_path} "
            f"checkpoint_sha256={result.checkpoint_sha256}"
        )
        for warning in result.warnings:
            print(f"WARNING {protein_path.name}: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
