#!/usr/bin/env python3
"""Evaluate a checkpoint on an immutable manifest split."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from p450pocket.checkpoint import load_model, sha256_file
from p450pocket.data import load_manifest
from p450pocket.manifest import verify_record_hashes
from p450pocket.metrics import split_metric_levels
from p450pocket.training import evaluate_model, make_loader, resolve_device, seed_everything


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate every labeled residue in a split")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-integrity-check",
        action="store_true",
        help="Skip feature/label SHA-256 verification",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative")
    manifest = Path(args.manifest).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    destination = Path(args.output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / f"{args.split}_metrics.json"
    csv_path = destination / f"{args.split}_per_protein.csv"
    existing = [path for path in (json_path, csv_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"evaluation outputs already exist: {existing}; pass --overwrite to replace them"
        )
    seed_everything(args.seed, deterministic=True)
    device = resolve_device(args.device)
    records = load_manifest(manifest, splits=[args.split])
    if not args.skip_integrity_check:
        verify_record_hashes(records)
    loader = make_loader(
        records,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    model, metadata = load_model(checkpoint, device=device, strict=True)
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(metadata.get("threshold", 0.5))
    )
    pooled, per_protein = evaluate_model(
        model,
        loader,
        device,
        threshold=threshold,
        mixed_precision=args.mixed_precision,
    )
    pooled_residue_metrics, protein_macro_metrics = split_metric_levels(pooled)
    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "split": args.split,
        "device": str(device),
        "mixed_precision": bool(args.mixed_precision),
        "pooled_residue_metrics": pooled_residue_metrics,
        "protein_macro_metrics": protein_macro_metrics,
        "n_proteins": len(records),
    }
    temporary = json_path.with_suffix(json_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, json_path)

    if per_protein:
        csv_temporary = csv_path.with_suffix(csv_path.suffix + ".tmp")
        with csv_temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_protein[0].keys()))
            writer.writeheader()
            writer.writerows(per_protein)
        os.replace(csv_temporary, csv_path)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
