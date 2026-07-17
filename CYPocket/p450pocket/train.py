#!/usr/bin/env python3
"""Train or fine-tune the P450 residue-pocket classifier."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from p450pocket.data import load_manifest
from p450pocket.manifest import verify_record_hashes
from p450pocket.training import TrainingConfig, run_training


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train/fine-tune from a JSONL manifest. Use prepare_dataset.py first "
            "when starting from protein and pocket PDB directories."
        )
    )
    parser.add_argument("--manifest", required=True, help="JSONL dataset manifest")
    parser.add_argument("--output-dir", required=True, help="New or resumable run directory")
    parser.add_argument(
        "--config",
        help="Optional training JSON config; built-in documented defaults are used when omitted",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="val")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--init-checkpoint",
        help="Strictly load model weights and start a new fine-tuning run",
    )
    source.add_argument(
        "--resume",
        help="Resume model, optimizer, scheduler, epoch, and global step",
    )
    source.add_argument(
        "--bert-pretrained",
        help="Local bert-base-uncased directory for de novo classifier training",
    )
    source.add_argument(
        "--allow-random-initialization",
        action="store_true",
        help="Explicitly train a randomly initialized BERT (normally inappropriate)",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")

    # High-value overrides. All other settings remain in the archived JSON.
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--positive-class-weight", type=float)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--monitor", choices=["loss", "pr_auc", "roc_auc", "f1", "mcc"])
    parser.add_argument("--patience", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-workers", type=int)
    mixed = parser.add_mutually_exclusive_group()
    mixed.add_argument("--mixed-precision", dest="mixed_precision", action="store_true")
    mixed.add_argument("--no-mixed-precision", dest="mixed_precision", action="store_false")
    deterministic = parser.add_mutually_exclusive_group()
    deterministic.add_argument("--deterministic", dest="deterministic", action="store_true")
    deterministic.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    parser.set_defaults(mixed_precision=None, deterministic=None)
    parser.add_argument("--freeze-policy", choices=["none", "classifier", "last_n"])
    parser.add_argument("--train-last-n-layers", type=int)
    parser.add_argument(
        "--skip-integrity-check",
        action="store_true",
        help="Skip feature/label SHA-256 verification (not recommended for reported runs)",
    )
    return parser


def _load_config(path: str | None) -> Dict[str, Any]:
    if path is None:
        return {}
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"training config not found: {source}")
    with source.open("r", encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, dict):
        raise ValueError("training config must be a JSON object")
    if "training" in values:
        values = values["training"]
    return values


def main() -> None:
    args = _parser().parse_args()
    values = _load_config(args.config)
    overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "positive_class_weight": args.positive_class_weight,
        "threshold": args.threshold,
        "monitor": args.monitor,
        "patience": args.patience,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "mixed_precision": args.mixed_precision,
        "deterministic": args.deterministic,
        "freeze_policy": args.freeze_policy,
        "train_last_n_layers": args.train_last_n_layers,
    }
    values.update({key: value for key, value in overrides.items() if value is not None})
    config = TrainingConfig.from_dict(values)
    manifest = Path(args.manifest).expanduser().resolve()
    train_records = load_manifest(manifest, splits=[args.train_split])
    validation_records = load_manifest(manifest, splits=[args.validation_split])
    if not args.skip_integrity_check:
        verify_record_hashes([*train_records, *validation_records])
    overlap = {record.sample_id for record in train_records} & {
        record.sample_id for record in validation_records
    }
    if overlap:
        raise ValueError(
            f"train/validation sample IDs overlap ({len(overlap)}); examples={sorted(overlap)[:10]}"
        )

    result = run_training(
        train_records,
        validation_records,
        args.output_dir,
        config,
        device=args.device,
        initialization_checkpoint=args.init_checkpoint,
        resume_checkpoint=args.resume,
        manifest_path=manifest,
        bert_pretrained_path=args.bert_pretrained,
        allow_random_initialization=args.allow_random_initialization,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
