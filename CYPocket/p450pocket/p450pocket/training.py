"""Reproducible training and fine-tuning for residue-level pocket prediction."""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .checkpoint import (
    load_model,
    restore_rng_state,
    save_training_checkpoint,
    sha256_file,
)
from .data import ProteinFeatureDataset, SampleRecord, collate_proteins
from .metrics import binary_metrics, macro_protein_metrics, split_metric_levels
from .model import ModelConfig, ResidueBertClassifier, positive_probabilities


@dataclass
class TrainingConfig:
    """All settings which affect optimization or numerical reproducibility."""

    epochs: int = 50
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1.0e-5
    weight_decay: float = 1.0e-4
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    positive_class_weight: Optional[float] = None
    threshold: float = 0.5
    monitor: str = "pr_auc"
    patience: int = 10
    min_delta: float = 0.0
    seed: int = 2026
    num_workers: int = 0
    deterministic: bool = True
    mixed_precision: bool = False
    freeze_policy: str = "none"
    train_last_n_layers: int = 2

    @classmethod
    def from_dict(cls, values: Optional[Dict[str, Any]]) -> "TrainingConfig":
        if not values:
            return cls()
        allowed = cls.__dataclass_fields__.keys()
        unknown = sorted(set(values) - set(allowed))
        if unknown:
            raise ValueError(f"unknown training configuration fields: {unknown}")
        return cls(**{key: values[key] for key in allowed if key in values})

    def validate(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("batch_size and gradient_accumulation_steps must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if self.positive_class_weight is not None and self.positive_class_weight <= 0:
            raise ValueError("positive_class_weight must be positive")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if self.monitor not in {"loss", "pr_auc", "roc_auc", "f1", "mcc"}:
            raise ValueError("monitor must be one of loss, pr_auc, roc_auc, f1, mcc")
        if self.patience < 1 or self.min_delta < 0:
            raise ValueError("patience must be positive and min_delta non-negative")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.freeze_policy not in {"none", "classifier", "last_n"}:
            raise ValueError("freeze_policy must be none, classifier, or last_n")
        if self.train_last_n_layers < 1:
            raise ValueError("train_last_n_layers must be positive")


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {requested}")
    return device


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed all RNGs used by this package and select deterministic kernels."""

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # older PyTorch
            torch.use_deterministic_algorithms(True)


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(
    records: Sequence[SampleRecord],
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        ProteinFeatureDataset(records, require_labels=True),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_proteins,
        worker_init_fn=_seed_worker if num_workers else None,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def configure_trainable(
    model: ResidueBertClassifier,
    policy: str,
    train_last_n_layers: int = 2,
) -> Dict[str, int]:
    """Apply an explicit fine-tuning policy and report parameter counts."""

    for parameter in model.parameters():
        parameter.requires_grad = policy == "none"

    if policy == "classifier":
        for module in (model.feature_projection, model.classifier):
            for parameter in module.parameters():
                parameter.requires_grad = True
    elif policy == "last_n":
        if train_last_n_layers > len(model.bert.encoder.layer):
            raise ValueError(
                f"train_last_n_layers={train_last_n_layers} exceeds "
                f"the {len(model.bert.encoder.layer)} BERT layers"
            )
        for module in (
            model.feature_projection,
            model.classifier,
            *model.bert.encoder.layer[-train_last_n_layers:],
        ):
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad = True
    elif policy != "none":
        raise ValueError(f"unknown freeze policy: {policy}")

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable == 0:
        raise ValueError("fine-tuning policy leaves no trainable parameters")
    return {"total_parameters": int(total), "trainable_parameters": int(trainable)}


def _class_weights(config: TrainingConfig, device: torch.device) -> Optional[torch.Tensor]:
    if config.positive_class_weight is None:
        return None
    return torch.tensor(
        [1.0, float(config.positive_class_weight)], dtype=torch.float32, device=device
    )


def _autocast(device: torch.device, enabled: bool):
    # torch.autocast is available in the currently supported PyTorch versions.
    return torch.autocast(device_type=device.type, enabled=enabled and device.type == "cuda")


def evaluate_model(
    model: ResidueBertClassifier,
    loader: DataLoader,
    device: torch.device,
    *,
    threshold: float = 0.5,
    positive_class_weight: Optional[float] = None,
    mixed_precision: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluate every valid residue; no failed batches or labels are discarded."""

    model.eval()
    class_weights = (
        torch.tensor([1.0, positive_class_weight], device=device)
        if positive_class_weight is not None
        else None
    )
    all_labels: List[np.ndarray] = []
    all_probabilities: List[np.ndarray] = []
    per_sample: List[Dict[str, Any]] = []
    loss_sum = 0.0
    loss_normalizer = 0.0
    residue_count = 0

    with torch.inference_mode():
        for batch in loader:
            features = batch["features"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with _autocast(device, mixed_precision):
                logits = model.forward_chunked(features, attention_mask)
                valid = labels.ne(-1)
                batch_loss = F.cross_entropy(
                    logits[valid], labels[valid], weight=class_weights, reduction="sum"
                )
            loss_sum += float(batch_loss.detach().cpu())
            residue_count += int(valid.sum().item())
            if class_weights is None:
                loss_normalizer += int(valid.sum().item())
            else:
                loss_normalizer += float(class_weights[labels[valid]].sum().item())
            probabilities = positive_probabilities(logits).float().cpu().numpy()
            labels_cpu = labels.cpu().numpy()
            lengths = batch["lengths"].tolist()
            for index, (sample_id, length) in enumerate(
                zip(batch["sample_ids"], lengths)
            ):
                sample_labels = labels_cpu[index, :length].astype(np.int64, copy=True)
                sample_probabilities = probabilities[index, :length].astype(
                    np.float64, copy=True
                )
                all_labels.append(sample_labels)
                all_probabilities.append(sample_probabilities)
                sample_metrics = binary_metrics(
                    sample_labels, sample_probabilities, threshold=threshold
                )
                per_sample.append({"sample_id": sample_id, **sample_metrics})

    if residue_count == 0:
        raise RuntimeError("evaluation loader contained no valid residues")
    pooled = binary_metrics(
        np.concatenate(all_labels), np.concatenate(all_probabilities), threshold=threshold
    )
    pooled["loss"] = loss_sum / loss_normalizer
    pooled.update(macro_protein_metrics(all_labels, all_probabilities, threshold=threshold))
    return pooled, per_sample


def _linear_warmup_decay(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = int(round(total_steps * warmup_ratio))

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        remaining = total_steps - step
        decay_steps = max(1, total_steps - warmup_steps)
        return max(0.0, float(remaining) / float(decay_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _scale_gradients(model: torch.nn.Module, divisor: float) -> None:
    if divisor <= 0:
        raise RuntimeError("cannot normalize gradients by zero residues")
    inverse = 1.0 / float(divisor)
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.mul_(inverse)


def _write_json(path: Path, values: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(values, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _write_history(path: Path, history: List[Dict[str, Any]]) -> None:
    if not history:
        return
    fieldnames = list(history[0].keys())
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    os.replace(temporary, path)


def _read_history(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _monitor_improved(
    name: str,
    current: float,
    best: Optional[float],
    min_delta: float,
) -> bool:
    if not np.isfinite(current):
        return False
    if best is None:
        return True
    if name == "loss":
        return current < best - min_delta
    return current > best + min_delta


def run_training(
    train_records: Sequence[SampleRecord],
    validation_records: Sequence[SampleRecord],
    output_dir: os.PathLike[str] | str,
    config: TrainingConfig,
    *,
    device: str = "auto",
    initialization_checkpoint: Optional[os.PathLike[str] | str] = None,
    resume_checkpoint: Optional[os.PathLike[str] | str] = None,
    manifest_path: Optional[os.PathLike[str] | str] = None,
    bert_pretrained_path: Optional[os.PathLike[str] | str] = None,
    allow_random_initialization: bool = False,
) -> Dict[str, Any]:
    """Train, fine-tune, or resume a model and write an auditable run directory."""

    config.validate()
    initialization_sources = sum(
        value is not None
        for value in (initialization_checkpoint, resume_checkpoint, bert_pretrained_path)
    )
    if initialization_sources > 1:
        raise ValueError("initialization_checkpoint and resume_checkpoint are mutually exclusive")
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    run_artifacts = (
        destination / "run_config.json",
        destination / "history.csv",
        destination / "best.pth",
        destination / "last.pth",
    )
    existing_artifacts = [path for path in run_artifacts if path.exists()]
    if existing_artifacts and not resume_checkpoint:
        raise FileExistsError(
            f"output directory already contains training artifacts: {existing_artifacts}"
        )
    device_object = resolve_device(device)
    seed_everything(config.seed, config.deterministic)

    source_checkpoint = resume_checkpoint or initialization_checkpoint
    checkpoint_metadata: Dict[str, Any] = {}
    if source_checkpoint:
        model, checkpoint_metadata = load_model(source_checkpoint, device=device_object, strict=True)
    elif bert_pretrained_path:
        model = ResidueBertClassifier.from_pretrained_bert(
            str(Path(bert_pretrained_path).expanduser().resolve()),
            ModelConfig(),
            local_files_only=True,
        ).to(device_object)
    elif allow_random_initialization:
        model = ResidueBertClassifier(ModelConfig()).to(device_object)
    else:
        raise ValueError(
            "de novo training requires bert_pretrained_path; for fine-tuning pass "
            "initialization_checkpoint, or explicitly opt into random BERT weights"
        )

    parameter_counts = configure_trainable(
        model, config.freeze_policy, config.train_last_n_layers
    )
    validation_loader = make_loader(
        validation_records,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        seed=config.seed + 1,
    )

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    batches_per_epoch = math.ceil(len(train_records) / config.batch_size)
    steps_per_epoch = math.ceil(batches_per_epoch / config.gradient_accumulation_steps)
    total_steps = steps_per_epoch * config.epochs
    scheduler = _linear_warmup_decay(optimizer, total_steps, config.warmup_ratio)
    scaler = torch.cuda.amp.GradScaler(
        enabled=config.mixed_precision and device_object.type == "cuda"
    )

    manifest_digest = sha256_file(manifest_path) if manifest_path else None

    start_epoch = 0
    global_step = 0
    if resume_checkpoint:
        saved_config = checkpoint_metadata.get("training_config")
        if saved_config != asdict(config):
            raise ValueError(
                "resume configuration differs from the checkpoint. Exact resume "
                "requires the same TrainingConfig; use --init-checkpoint for a new "
                "fine-tuning run with changed hyperparameters"
            )
        saved_manifest_digest = checkpoint_metadata.get("manifest_sha256")
        if saved_manifest_digest != manifest_digest:
            raise ValueError(
                "resume manifest SHA-256 differs from the checkpoint (or one is "
                "missing); exact resume requires the original manifest"
            )
        optimizer_state = checkpoint_metadata.get("optimizer_state_dict")
        scheduler_state = checkpoint_metadata.get("scheduler_state_dict")
        if optimizer_state is None:
            raise ValueError("resume checkpoint does not contain optimizer state")
        optimizer.load_state_dict(optimizer_state)
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
        scaler_state = checkpoint_metadata.get("scaler_state_dict")
        if scaler_state is not None:
            scaler.load_state_dict(scaler_state)
        elif config.mixed_precision:
            raise ValueError(
                "mixed-precision resume checkpoint does not contain GradScaler state"
            )
        start_epoch = int(checkpoint_metadata.get("epoch", -1)) + 1
        global_step = int(checkpoint_metadata.get("global_step", 0))
        if start_epoch >= config.epochs:
            raise ValueError(
                f"resume starts at epoch {start_epoch}, but configured epochs={config.epochs}"
            )
        if checkpoint_metadata.get("rng_state"):
            restore_rng_state(checkpoint_metadata["rng_state"])

    run_configuration: Dict[str, Any] = {
        "training_config": asdict(config),
        "model_config": model.model_config.to_dict(),
        "device": str(device_object),
        "initialization_checkpoint": (
            str(Path(initialization_checkpoint).resolve())
            if initialization_checkpoint
            else None
        ),
        "resume_checkpoint": (
            str(Path(resume_checkpoint).resolve()) if resume_checkpoint else None
        ),
        "bert_pretrained_path": (
            str(Path(bert_pretrained_path).resolve()) if bert_pretrained_path else None
        ),
        "allow_random_initialization": bool(allow_random_initialization),
        "manifest": str(Path(manifest_path).resolve()) if manifest_path else None,
        "manifest_sha256": manifest_digest,
        "n_train_proteins": len(train_records),
        "n_validation_proteins": len(validation_records),
        **parameter_counts,
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": str(torch.__version__),
        },
    }
    _write_json(destination / "run_config.json", run_configuration)

    class_weights = _class_weights(config, device_object)
    history: List[Dict[str, Any]] = (
        _read_history(destination / "history.csv") if resume_checkpoint else []
    )
    if resume_checkpoint and history:
        resumed_epoch = start_epoch - 1
        history = [
            row
            for row in history
            if int(float(row.get("epoch", -1))) <= resumed_epoch
        ]
        _write_history(destination / "history.csv", history)
    best_value: Optional[float] = None
    best_epoch: Optional[int] = None
    epochs_without_improvement = 0
    if resume_checkpoint:
        saved_training_state = checkpoint_metadata.get("training_state") or {}
        if saved_training_state.get("monitor") == config.monitor:
            saved_best = saved_training_state.get("best_value")
            best_value = float(saved_best) if saved_best is not None else None
            saved_best_epoch = saved_training_state.get("best_epoch")
            best_epoch = int(saved_best_epoch) if saved_best_epoch is not None else None
            epochs_without_improvement = int(
                saved_training_state.get("epochs_without_improvement", 0)
            )
        elif checkpoint_metadata.get("metrics", {}).get(config.monitor) is not None:
            # Older project checkpoints predate the explicit early-stopping
            # state. Their current metric is a conservative resume baseline.
            best_value = float(checkpoint_metadata["metrics"][config.monitor])
            best_epoch = int(checkpoint_metadata.get("epoch", start_epoch - 1))
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.time()
        model.train()
        # Epoch-derived shuffling makes an interrupted/resumed run independent
        # of hidden DataLoader generator state.
        train_loader = make_loader(
            train_records,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            seed=config.seed + epoch,
        )
        train_loss_sum = 0.0
        train_loss_normalizer = 0.0
        accumulated_normalizer = 0.0

        for batch_index, batch in enumerate(train_loader):
            features = batch["features"].to(device_object, non_blocking=True)
            labels = batch["labels"].to(device_object, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device_object, non_blocking=True)
            valid = labels.ne(-1)
            n_valid = int(valid.sum().item())
            if n_valid == 0:
                raise RuntimeError("training batch contained no valid residues")
            with _autocast(device_object, config.mixed_precision):
                logits = model.forward_chunked(features, attention_mask)
                loss_sum = F.cross_entropy(
                    logits[valid], labels[valid], weight=class_weights, reduction="sum"
                )
            scaler.scale(loss_sum).backward()
            train_loss_sum += float(loss_sum.detach().cpu())
            if class_weights is None:
                batch_normalizer = float(n_valid)
            else:
                batch_normalizer = float(class_weights[labels[valid]].sum().item())
            train_loss_normalizer += batch_normalizer
            accumulated_normalizer += batch_normalizer

            accumulation_boundary = (
                (batch_index + 1) % config.gradient_accumulation_steps == 0
                or batch_index + 1 == len(train_loader)
            )
            if accumulation_boundary:
                scaler.unscale_(optimizer)
                _scale_gradients(model, accumulated_normalizer)
                torch.nn.utils.clip_grad_norm_(trainable_parameters, config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                accumulated_normalizer = 0.0

        validation_metrics, per_sample = evaluate_model(
            model,
            validation_loader,
            device_object,
            threshold=config.threshold,
            positive_class_weight=config.positive_class_weight,
            mixed_precision=config.mixed_precision,
        )
        raw_monitor_value = validation_metrics[config.monitor]
        if raw_monitor_value is None:
            raise RuntimeError(
                f"validation metric {config.monitor!r} is undefined; the validation "
                "split must contain both classes for ROC-AUC/PR-AUC monitoring"
            )
        monitor_value = float(raw_monitor_value)
        improved = _monitor_improved(
            config.monitor, monitor_value, best_value, config.min_delta
        )
        if improved:
            best_value = monitor_value
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        row: Dict[str, Any] = {
            "epoch": epoch,
            "global_step": global_step,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss_sum / train_loss_normalizer,
            "val_loss": validation_metrics["loss"],
            "val_accuracy": validation_metrics["accuracy"],
            "val_balanced_accuracy": validation_metrics["balanced_accuracy"],
            "val_precision": validation_metrics["precision"],
            "val_recall": validation_metrics["recall"],
            "val_f1": validation_metrics["f1"],
            "val_mcc": validation_metrics["mcc"],
            "val_roc_auc": validation_metrics["roc_auc"],
            "val_pr_auc": validation_metrics["pr_auc"],
            "seconds": time.time() - epoch_start,
        }
        history.append(row)
        _write_history(destination / "history.csv", history)
        pooled_residue_metrics, protein_macro_metrics = split_metric_levels(
            validation_metrics
        )
        _write_json(
            destination / "validation_metrics.json",
            {
                "epoch": epoch,
                "pooled_residue_metrics": pooled_residue_metrics,
                "protein_macro_metrics": protein_macro_metrics,
                "per_protein": per_sample,
            },
        )

        checkpoint_arguments = dict(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            metrics=validation_metrics,
            training_config=asdict(config),
            manifest_sha256=manifest_digest,
            threshold=config.threshold,
            training_state={
                "monitor": config.monitor,
                "best_value": best_value,
                "best_epoch": best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
            },
            scaler=scaler,
        )
        save_training_checkpoint(destination / "last.pth", **checkpoint_arguments)
        if improved:
            save_training_checkpoint(destination / "best.pth", **checkpoint_arguments)

        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": row["train_loss"],
                    "val_loss": row["val_loss"],
                    config.monitor: monitor_value,
                    "best": improved,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        if epochs_without_improvement >= config.patience:
            break

    destination_best = destination / "best.pth"
    source_best = (
        Path(resume_checkpoint).expanduser().resolve().parent / "best.pth"
        if resume_checkpoint
        else None
    )
    if destination_best.is_file():
        best_checkpoint_path = destination_best
    elif source_best is not None and source_best.is_file():
        best_checkpoint_path = source_best
    else:
        best_checkpoint_path = destination / "last.pth"
    result = {
        "best_epoch": best_epoch,
        "best_monitor_value": best_value,
        "monitor": config.monitor,
        "epochs_completed": len(history),
        "global_step": global_step,
        "best_checkpoint": str(best_checkpoint_path),
        "last_checkpoint": str(destination / "last.pth"),
    }
    _write_json(destination / "result.json", result)
    return result
