"""Safe loading and metadata-rich saving of model checkpoints."""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
import transformers

from .model import ModelConfig, ResidueBertClassifier
from .provenance import sha256_file


_FLOAT_TAG = "__p450pocket_float__"


def _encode_weights_only(value: Any) -> Any:
    """Encode Python floats for PyTorch 1.13's restricted unpickler.

    That early ``weights_only`` implementation accepts tensors and basic
    containers but not the BINFLOAT opcode.  Tagged decimal strings keep our
    structured checkpoints safe-loadable without losing optimizer metadata.
    """

    if isinstance(value, float):
        return {_FLOAT_TAG: repr(value)}
    if isinstance(value, dict):
        return {key: _encode_weights_only(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_encode_weights_only(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_encode_weights_only(item) for item in value)
    return value


def _decode_weights_only(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {_FLOAT_TAG}:
            return float(value[_FLOAT_TAG])
        return {key: _decode_weights_only(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_weights_only(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_decode_weights_only(item) for item in value)
    return value


def _torch_load(
    path: os.PathLike[str] | str,
    map_location: Any = "cpu",
    *,
    trusted: bool = False,
) -> Any:
    if trusted:
        return torch.load(path, map_location=map_location, weights_only=False)
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError as error:  # very old PyTorch without weights_only
        raise RuntimeError(
            "this PyTorch version cannot safely load checkpoints; upgrade PyTorch "
            "or explicitly mark a trusted local checkpoint"
        ) from error
    except Exception as error:
        raise RuntimeError(
            f"checkpoint could not be loaded in weights-only mode: {path}. "
            "Only use trusted=True for a checkpoint whose provenance you control."
        ) from error


def unpack_checkpoint(payload: Any) -> Tuple[Mapping[str, torch.Tensor], Dict[str, Any]]:
    payload = _decode_weights_only(payload)
    if not isinstance(payload, Mapping):
        raise TypeError(f"checkpoint must be a mapping, got {type(payload)!r}")
    if "model_state_dict" in payload:
        state = payload["model_state_dict"]
        metadata = {key: value for key, value in payload.items() if key != "model_state_dict"}
    elif "state_dict" in payload and isinstance(payload["state_dict"], Mapping):
        state = payload["state_dict"]
        metadata = {key: value for key, value in payload.items() if key != "state_dict"}
    else:
        state = payload
        metadata = {"format": "legacy-raw-state-dict"}
    if not isinstance(state, Mapping):
        raise TypeError("model_state_dict is not a mapping")
    if state and all(str(key).startswith("module.") for key in state):
        state = {str(key)[7:]: value for key, value in state.items()}
    return state, metadata


def load_model(
    checkpoint_path: os.PathLike[str] | str,
    device: torch.device | str = "cpu",
    strict: bool = True,
    trusted: bool = False,
) -> Tuple[ResidueBertClassifier, Dict[str, Any]]:
    payload = _torch_load(checkpoint_path, map_location="cpu", trusted=trusted)
    state, metadata = unpack_checkpoint(payload)
    config = ModelConfig.from_dict(metadata.get("model_config"))
    model = ResidueBertClassifier(config)
    incompatible = model.load_state_dict(state, strict=strict)
    if not strict and (incompatible.missing_keys or incompatible.unexpected_keys):
        metadata["missing_keys"] = list(incompatible.missing_keys)
        metadata["unexpected_keys"] = list(incompatible.unexpected_keys)
    model.to(device)
    metadata.setdefault("model_config", config.to_dict())
    metadata.setdefault("decision_rule", "argmax")
    metadata.setdefault("threshold", 0.5)
    metadata["checkpoint_path"] = str(Path(checkpoint_path).resolve())
    return model, metadata


def rng_state() -> Dict[str, Any]:
    numpy_state = np.random.get_state()
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            # PyTorch 1.13 cannot construct tensors from NumPy uint32 arrays.
            "state": torch.from_numpy(numpy_state[1].astype(np.int64, copy=True)),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore RNGs saved by :func:`rng_state` at an epoch boundary."""

    if not state:
        return
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            numpy_state["state"].cpu().numpy().astype(np.uint32, copy=False),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_training_checkpoint(
    path: os.PathLike[str] | str,
    model: ResidueBertClassifier,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    epoch: int,
    global_step: int,
    metrics: Dict[str, Any],
    training_config: Dict[str, Any],
    manifest_sha256: Optional[str] = None,
    threshold: float = 0.5,
    training_state: Optional[Dict[str, Any]] = None,
    scaler: Optional[Any] = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "architecture": model.model_config.architecture,
        "model_config": model.model_config.to_dict(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "metrics": metrics,
        "training_config": training_config,
        "manifest_sha256": manifest_sha256,
        "decision_rule": "probability-threshold",
        "threshold": float(threshold),
        "rng_state": rng_state(),
        "training_state": training_state or {},
        "software": {
            "python": sys.version,
            "numpy": str(np.__version__),
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
        },
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(_encode_weights_only(payload), temporary)
    os.replace(temporary, destination)
