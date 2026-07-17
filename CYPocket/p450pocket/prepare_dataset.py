#!/usr/bin/env python3
"""Build an auditable manifest from protein/pocket PDB pairs or existing arrays."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

from p450pocket.features import PocketMinerExtractor
from p450pocket.labels import build_labels
from p450pocket.manifest import SampleRecord, discover_processed_split, write_manifest
from p450pocket.paths import default_model_artifact
from p450pocket.provenance import sha256_file
from p450pocket.structure import StructureRecord, parse_feature_protein, parse_pocket


DEFAULT_POCKETMINER = default_model_artifact("pocketminer")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare train/val/test data. Raw mode expects SPLIT/protein and "
            "SPLIT/pocket; --index-existing validates existing tokens/labels only."
        )
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--output-root",
        help="Prepared output root (required for raw PDB extraction)",
    )
    parser.add_argument(
        "--manifest-out",
        help="Manifest path (default: OUTPUT_ROOT/manifest.jsonl or DATASET_ROOT/manifest.jsonl)",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument(
        "--index-existing",
        action="store_true",
        help="Validate and index existing SPLIT/tokens and SPLIT/labels",
    )
    parser.add_argument(
        "--pocketminer-checkpoint",
        default=os.environ.get("P450_POCKETMINER_CHECKPOINT", str(DEFAULT_POCKETMINER)),
        help=(
            "TensorFlow checkpoint prefix required in raw PDB mode; may also set "
            "P450_POCKETMINER_CHECKPOINT"
        ),
    )
    parser.add_argument("--chains", nargs="*", default=None)
    parser.add_argument("--model-index", type=int, default=0)
    parser.add_argument(
        "--match-mode",
        choices=["identity", "identity_then_coordinate", "coordinate"],
        default="identity",
    )
    parser.add_argument("--verify-ca-distance", type=float, default=0.25)
    parser.add_argument("--coordinate-tolerance", type=float, default=0.25)
    parser.add_argument(
        "--modified-residue",
        action="append",
        default=[],
        metavar="ORIGINAL=CANONICAL",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _modified_map(values: Iterable[str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"modified residue mapping must be ORIGINAL=CANONICAL: {value}")
        original, canonical = value.split("=", 1)
        original, canonical = original.strip().upper(), canonical.strip().upper()
        if not original or not canonical:
            raise ValueError(f"invalid modified residue mapping: {value}")
        result[original] = canonical
    return result


def _sample_id(path: Path, kind: str) -> str:
    stem = path.stem
    candidates = (
        f"_{kind}",
        f"-{kind}",
        "_protein" if kind == "pocket" else "_pocket",
        "-protein" if kind == "pocket" else "-pocket",
    )
    for suffix in candidates:
        if stem.lower().endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _pdb_index(directory: Path, kind: str) -> Dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"missing {kind} directory: {directory}")
    result: Dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".pdb":
            continue
        sample_id = _sample_id(path, kind)
        if sample_id in result:
            raise ValueError(
                f"duplicate {kind} sample ID {sample_id}: {result[sample_id]} and {path}"
            )
        result[sample_id] = path.resolve()
    if not result:
        raise ValueError(f"no PDB files found in {directory}")
    return result


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


def _atomic_residue_map(path: Path, structure: StructureRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
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
    os.replace(temporary, path)


def _sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def _prepare_raw_split(
    source_root: Path,
    output_root: Path,
    split: str,
    extractor: PocketMinerExtractor,
    *,
    chains: Sequence[str] | None,
    model_index: int,
    modified_residue_map: Mapping[str, str],
    match_mode: str,
    verify_ca_distance: float,
    coordinate_tolerance: float,
    overwrite: bool,
) -> List[SampleRecord]:
    source_split = source_root / split
    proteins = _pdb_index(source_split / "protein", "protein")
    pockets = _pdb_index(source_split / "pocket", "pocket")
    if proteins.keys() != pockets.keys():
        missing_proteins = sorted(pockets.keys() - proteins.keys())[:10]
        missing_pockets = sorted(proteins.keys() - pockets.keys())[:10]
        raise ValueError(
            f"{split} protein/pocket IDs differ; missing_proteins={missing_proteins}, "
            f"missing_pockets={missing_pockets}"
        )

    destination = output_root / split
    records: List[SampleRecord] = []
    for index, sample_id in enumerate(sorted(proteins), start=1):
        protein_path, pocket_path = proteins[sample_id], pockets[sample_id]
        feature_path = destination / "tokens" / f"{sample_id}_protein-tokens.npy"
        label_path = destination / "labels" / f"{sample_id}_labels.npy"
        residue_map_path = destination / "residue_maps" / f"{sample_id}_residues.tsv"
        existing = [path for path in (feature_path, label_path, residue_map_path) if path.exists()]
        if existing and not overwrite:
            raise FileExistsError(
                f"prepared outputs already exist for {sample_id}: {existing}; pass --overwrite"
            )

        protein = parse_feature_protein(
            protein_path,
            model_index=model_index,
            chains=chains,
            modified_residue_map=modified_residue_map,
        )
        if len(protein.chains) != 1:
            raise ValueError(
                f"{protein_path} contains {len(protein.chains)} protein chains. "
                "This model requires one P450 chain per sample; use --chains CHAIN_ID."
            )
        pocket = parse_pocket(
            pocket_path,
            model_index=model_index,
            chains=chains,
            modified_residue_map=modified_residue_map,
        )
        label_result = build_labels(
            protein,
            pocket,
            match_mode=match_mode,
            verify_ca_distance_angstrom=verify_ca_distance,
            coordinate_tolerance_angstrom=coordinate_tolerance,
        )
        feature_result = extractor.extract(protein)
        if len(feature_result.features) != len(label_result.labels):
            raise RuntimeError(f"internal feature/label mismatch for {sample_id}")

        _atomic_save_npy(feature_path, feature_result.features)
        _atomic_save_npy(label_path, label_result.labels)
        _atomic_residue_map(residue_map_path, protein)
        records.append(
            SampleRecord(
                sample_id=sample_id,
                split=split,
                feature_path=str(feature_path.resolve()),
                label_path=str(label_path.resolve()),
                protein_path=str(protein_path),
                pocket_path=str(pocket_path),
                residue_map_path=str(residue_map_path.resolve()),
                n_residues=len(protein),
                n_positive=label_result.n_positive,
                sequence_sha256=_sequence_sha256(protein.sequence),
                feature_sha256=sha256_file(feature_path),
                label_sha256=sha256_file(label_path),
                feature_extractor_sha256=feature_result.checkpoint_sha256,
                protein_sha256=sha256_file(protein_path),
                pocket_sha256=sha256_file(pocket_path),
            )
        )
        print(
            f"[{split} {index}/{len(proteins)}] {sample_id}: "
            f"residues={len(protein)} positives={label_result.n_positive}",
            flush=True,
        )
    return records


def _write_summary(
    path: Path,
    records: Sequence[SampleRecord],
    *,
    mode: str,
    manifest: Path,
    checkpoint_prefix: str | None,
    checkpoint_sha256: str | None,
) -> None:
    by_split: Dict[str, Dict[str, int | float]] = {}
    for split in sorted({record.split for record in records}):
        selected = [record for record in records if record.split == split]
        n_residues = sum(int(record.n_residues or 0) for record in selected)
        n_positive = sum(int(record.n_positive or 0) for record in selected)
        lengths = np.asarray(
            [int(record.n_residues or 0) for record in selected], dtype=np.int64
        )
        by_split[split] = {
            "n_proteins": len(selected),
            "n_residues": n_residues,
            "n_positive": n_positive,
            "positive_fraction": n_positive / n_residues if n_residues else 0.0,
            "length_min": int(lengths.min()),
            "length_median": float(np.median(lengths)),
            "length_mean": float(lengths.mean()),
            "length_max": int(lengths.max()),
        }
    values = {
        "schema_version": 1,
        "mode": mode,
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "pocketminer_checkpoint_prefix": checkpoint_prefix,
        "pocketminer_checkpoint_sha256": checkpoint_sha256,
        "splits": by_split,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(values, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> None:
    args = _parser().parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {dataset_root}")
    if len(set(args.splits)) != len(args.splits):
        raise ValueError("--splits contains duplicates")
    if args.chains is not None and len(args.chains) != 1:
        raise ValueError("--chains must select exactly one chain for this single-chain model")

    if args.index_existing:
        output_root = (
            Path(args.output_root).expanduser().resolve()
            if args.output_root
            else dataset_root
        )
        records = []
        for split in args.splits:
            records.extend(discover_processed_split(dataset_root / split, split))
        mode = "index-existing"
        checkpoint_prefix = None
        checkpoint_sha256 = None
    else:
        if not args.output_root:
            raise ValueError("raw PDB mode requires --output-root")
        output_root = Path(args.output_root).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        modified = _modified_map(args.modified_residue)
        extractor = PocketMinerExtractor(args.pocketminer_checkpoint)
        records = []
        for split in args.splits:
            records.extend(
                _prepare_raw_split(
                    dataset_root,
                    output_root,
                    split,
                    extractor,
                    chains=args.chains,
                    model_index=args.model_index,
                    modified_residue_map=modified,
                    match_mode=args.match_mode,
                    verify_ca_distance=args.verify_ca_distance,
                    coordinate_tolerance=args.coordinate_tolerance,
                    overwrite=args.overwrite,
                )
            )
        mode = "extract-raw-pdb"
        checkpoint_prefix = str(extractor.checkpoint_prefix)
        checkpoint_sha256 = extractor.checkpoint_sha256

    manifest = (
        Path(args.manifest_out).expanduser().resolve()
        if args.manifest_out
        else output_root / "manifest.jsonl"
    )
    if manifest.exists() and not args.overwrite:
        raise FileExistsError(f"manifest already exists: {manifest}; pass --overwrite")
    identities = [(record.split, record.sample_id) for record in records]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate (split, sample_id) entries were generated")
    sample_splits: Dict[str, set[str]] = {}
    for record in records:
        sample_splits.setdefault(record.sample_id, set()).add(record.split)
    overlap = {sample_id: splits for sample_id, splits in sample_splits.items() if len(splits) > 1}
    if overlap:
        examples = list(sorted(overlap.items()))[:10]
        raise ValueError(f"sample IDs leak across splits; examples={examples}")
    write_manifest(records, manifest)
    _write_summary(
        manifest.with_name("dataset_summary.json"),
        records,
        mode=mode,
        manifest=manifest,
        checkpoint_prefix=checkpoint_prefix,
        checkpoint_sha256=checkpoint_sha256,
    )
    print(
        json.dumps(
            {"manifest": str(manifest), "n_records": len(records), "mode": mode},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
