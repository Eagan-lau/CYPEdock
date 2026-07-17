"""Unbiased residue-level and protein-level evaluation metrics."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    y_true = np.asarray(labels, dtype=np.int64).reshape(-1)
    y_prob = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if y_true.shape != y_prob.shape:
        raise ValueError(f"shape mismatch: labels={y_true.shape}, probabilities={y_prob.shape}")
    if y_true.size == 0:
        raise ValueError("cannot evaluate empty arrays")
    if not set(np.unique(y_true).tolist()).issubset({0, 1}):
        raise ValueError("labels must be binary")
    if not np.isfinite(y_prob).all():
        raise ValueError("probabilities contain NaN or infinity")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if np.any((y_prob < 0.0) | (y_prob > 1.0)):
        raise ValueError("probabilities must be in [0, 1]")
    y_pred = (y_prob >= threshold).astype(np.int64)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics: Dict[str, Any] = {
        "n_residues": int(y_true.size),
        "n_positive": int(y_true.sum()),
        "prevalence": float(y_true.mean()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": matrix.tolist(),
    }
    if np.unique(y_true).size == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        # JSON has no portable NaN representation.  ``None`` makes an
        # undefined single-class metric explicit in machine-readable output.
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
    return metrics


def macro_protein_metrics(
    labels: Sequence[np.ndarray],
    probabilities: Sequence[np.ndarray],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("labels and probabilities must be non-empty lists of equal length")
    per_protein = [binary_metrics(y, p, threshold) for y, p in zip(labels, probabilities)]
    keys = (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "mcc",
        "roc_auc",
        "pr_auc",
    )
    output: Dict[str, Any] = {"n_proteins": len(per_protein)}
    for key in keys:
        values = [float(item[key]) for item in per_protein if item[key] is not None]
        output[f"macro_{key}"] = float(np.mean(values)) if values else None
        if key in {"roc_auc", "pr_auc"}:
            output[f"n_proteins_with_{key}"] = len(values)
    return output


def split_metric_levels(metrics: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Separate pooled-residue metrics from protein-macro aggregates."""

    macro_keys = {
        key
        for key in metrics
        if key == "n_proteins"
        or key.startswith("macro_")
        or key.startswith("n_proteins_with_")
    }
    pooled = {key: value for key, value in metrics.items() if key not in macro_keys}
    macro = {key: metrics[key] for key in metrics if key in macro_keys}
    return pooled, macro
