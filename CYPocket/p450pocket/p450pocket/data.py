"""Validated PyTorch datasets and variable-length collation."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .manifest import (
    SampleRecord,
    discover_processed_split,
    load_manifest,
    write_manifest,
)

__all__ = [
    "SampleRecord",
    "discover_processed_split",
    "load_manifest",
    "write_manifest",
    "ProteinFeatureDataset",
    "collate_proteins",
]


class ProteinFeatureDataset(Dataset):
    def __init__(self, records: Sequence[SampleRecord], require_labels: bool = True) -> None:
        self.records = list(records)
        self.require_labels = require_labels
        if not self.records:
            raise ValueError("dataset is empty")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        features = np.load(record.feature_path)
        if features.ndim == 3 and features.shape[0] == 1:
            features = features[0]
        if features.ndim != 2 or features.shape[1] != 100:
            raise ValueError(f"invalid feature shape for {record.sample_id}: {features.shape}")
        if not np.isfinite(features).all():
            raise ValueError(f"non-finite features for {record.sample_id}")
        item: Dict[str, Any] = {
            "sample_id": record.sample_id,
            "features": torch.from_numpy(np.asarray(features, dtype=np.float32)),
            "record": record,
        }
        if self.require_labels:
            if not record.label_path:
                raise ValueError(f"missing label_path for {record.sample_id}")
            labels = np.load(record.label_path)
            if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
                raise ValueError(
                    f"feature/label mismatch for {record.sample_id}: {features.shape}, {labels.shape}"
                )
            if not set(np.unique(labels).tolist()).issubset({0, 1}):
                raise ValueError(f"labels must be binary for {record.sample_id}")
            item["labels"] = torch.from_numpy(np.asarray(labels, dtype=np.int64))
        return item


def collate_proteins(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not batch:
        raise ValueError("cannot collate an empty batch")
    features = [item["features"] for item in batch]
    lengths = torch.tensor([value.shape[0] for value in features], dtype=torch.long)
    padded_features = pad_sequence(features, batch_first=True, padding_value=0.0)
    positions = torch.arange(padded_features.shape[1]).unsqueeze(0)
    attention_mask = (positions < lengths.unsqueeze(1)).long()
    result: Dict[str, Any] = {
        "sample_ids": [item["sample_id"] for item in batch],
        "records": [item["record"] for item in batch],
        "features": padded_features,
        "attention_mask": attention_mask,
        "lengths": lengths,
    }
    if "labels" in batch[0]:
        result["labels"] = pad_sequence(
            [item["labels"] for item in batch],
            batch_first=True,
            padding_value=-1,
        )
    return result
