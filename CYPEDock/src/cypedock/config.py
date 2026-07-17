"""Strict configuration loading for CYPEdock."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional

import yaml

from .errors import ConfigurationError


DEFAULT_CONFIG: dict[str, Any] = {
    "project": {"random_seed": 20260716},
    "input": {"chain": None},
    "reference": {"protein_chain": None, "cofactor_residues": []},
    "alignment": {
        "local_shell_radius_angstrom": 10.0,
        "local_rmsd_cutoff_angstrom": 2.5,
        "minimum_local_residues": 8,
        "minimum_local_coverage": 0.60,
        "minimum_sequence_identity": 0.20,
        "minimum_sequence_coverage": 0.50,
        "alignment_atoms": ["N", "CA", "C"],
        "maximum_local_residue_rmsd_angstrom": 4.0,
        "cofactor_clash_distance_angstrom": 1.20,
        "require_proximal_cysteine": True,
        "minimum_fe_s_distance_angstrom": 1.80,
        "maximum_fe_s_distance_angstrom": 3.00,
    },
    "foldseek": {
        "enabled": True,
        "executable": "foldseek",
        "library_root": None,
        "database_prefix": None,
        "structure_root": None,
        "maximum_hits": 20,
        "minimum_alignment_tm_score": 0.50,
        "minimum_query_coverage": 0.60,
        "threads": 8,
    },
    "cypocket": {
        "python": None,
        "root": None,
        "feature_python": None,
        "classifier_python": None,
        "pocketminer_checkpoint": None,
        "classifier_checkpoint": None,
        "classifier_cuda_visible_devices": "",
        "device": "cpu",
        "threshold": None,
    },
    "box": {
        "auto_size": False,
        "center_atoms": "CA",
        "size_angstrom": [25.0, 25.0, 25.0],
        "maximum_size_angstrom": [40.0, 40.0, 40.0],
        "pocket_padding_angstrom": 4.0,
        "minimum_fe_margin_angstrom": 2.0,
    },
    "ligand": {"embedding_attempts": 1000, "optimization_iterations": 1000},
    "docking": {
        "vina_executable": None,
        "exhaustiveness": 8,
        "num_modes": 9,
        "energy_range": 3,
        "replicates": 1,
        "workers": 1,
        "cpu_per_task": 1,
        "timeout_seconds": 3600,
    },
}

_PATH_FIELDS = {
    ("foldseek", "executable"),
    ("foldseek", "library_root"),
    ("foldseek", "database_prefix"),
    ("foldseek", "structure_root"),
    ("cypocket", "python"),
    ("cypocket", "root"),
    ("cypocket", "feature_python"),
    ("cypocket", "classifier_python"),
    ("cypocket", "pocketminer_checkpoint"),
    ("cypocket", "classifier_checkpoint"),
    ("docking", "vina_executable"),
}


@dataclass(frozen=True)
class Config:
    data: dict[str, Any]
    source: Optional[Path]

    @property
    def base_dir(self) -> Path:
        return self.source.parent if self.source is not None else Path.cwd()

    def section(self, name: str) -> dict[str, Any]:
        return self.data[name]


def _merge_strict(target: MutableMapping[str, Any], update: Mapping[str, Any], path: str) -> None:
    for key, value in update.items():
        location = f"{path}.{key}" if path else key
        if key not in target:
            raise ConfigurationError(f"Unknown configuration key: {location}")
        if isinstance(target[key], dict):
            if not isinstance(value, Mapping):
                raise ConfigurationError(f"{location} must be a mapping")
            _merge_strict(target[key], value, location)
        else:
            target[key] = value


def _finite_number(value: Any, name: str, *, minimum: Optional[float] = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        raise ConfigurationError(f"{name} must be finite and >= {minimum}")
    return number


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigurationError(f"{name} must be a positive integer")
    return value


def _validate_selector(value: Any, index: int) -> None:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"reference.cofactor_residues[{index}] must be a mapping")
    allowed = {"chain", "number", "insertion_code", "name"}
    unknown = set(value) - allowed
    if unknown:
        raise ConfigurationError(
            f"Unknown cofactor selector keys at index {index}: {sorted(unknown)}"
        )
    if "number" not in value or isinstance(value["number"], bool) or not isinstance(
        value["number"], int
    ):
        raise ConfigurationError(f"reference.cofactor_residues[{index}].number is required")


def validate_config(data: Mapping[str, Any]) -> None:
    alignment = data["alignment"]
    for key in (
        "local_shell_radius_angstrom",
        "local_rmsd_cutoff_angstrom",
        "maximum_local_residue_rmsd_angstrom",
        "cofactor_clash_distance_angstrom",
        "minimum_fe_s_distance_angstrom",
        "maximum_fe_s_distance_angstrom",
    ):
        _finite_number(alignment[key], f"alignment.{key}", minimum=0.0)
    _positive_integer(alignment["minimum_local_residues"], "alignment.minimum_local_residues")
    for key in (
        "minimum_local_coverage",
        "minimum_sequence_identity",
        "minimum_sequence_coverage",
    ):
        value = _finite_number(alignment[key], f"alignment.{key}", minimum=0.0)
        if value > 1.0:
            raise ConfigurationError(f"alignment.{key} must be <= 1.0")
    atoms = alignment["alignment_atoms"]
    if not isinstance(atoms, list) or len(atoms) < 1 or not all(
        isinstance(atom, str) and atom.strip() for atom in atoms
    ):
        raise ConfigurationError("alignment.alignment_atoms must be a non-empty string list")
    if not isinstance(alignment["require_proximal_cysteine"], bool):
        raise ConfigurationError("alignment.require_proximal_cysteine must be boolean")
    if (
        float(alignment["minimum_fe_s_distance_angstrom"])
        >= float(alignment["maximum_fe_s_distance_angstrom"])
    ):
        raise ConfigurationError(
            "alignment.minimum_fe_s_distance_angstrom must be less than the maximum"
        )

    selectors = data["reference"]["cofactor_residues"]
    if not isinstance(selectors, list):
        raise ConfigurationError("reference.cofactor_residues must be a list")
    for index, selector in enumerate(selectors):
        _validate_selector(selector, index)

    foldseek = data["foldseek"]
    _positive_integer(foldseek["maximum_hits"], "foldseek.maximum_hits")
    _positive_integer(foldseek["threads"], "foldseek.threads")
    for key in ("minimum_alignment_tm_score", "minimum_query_coverage"):
        value = _finite_number(foldseek[key], f"foldseek.{key}", minimum=0.0)
        if value > 1.0:
            raise ConfigurationError(f"foldseek.{key} must be <= 1.0")

    for field in ("size_angstrom", "maximum_size_angstrom"):
        size = data["box"][field]
        if not isinstance(size, list) or len(size) != 3:
            raise ConfigurationError(f"box.{field} must contain three values")
        for index, value in enumerate(size):
            _finite_number(value, f"box.{field}[{index}]", minimum=1.0)
    for minimum, maximum in zip(
        data["box"]["size_angstrom"], data["box"]["maximum_size_angstrom"]
    ):
        if float(minimum) > float(maximum):
            raise ConfigurationError(
                "Each box.size_angstrom value must not exceed maximum_size_angstrom"
            )
    if not isinstance(data["box"]["auto_size"], bool):
        raise ConfigurationError("box.auto_size must be boolean")
    _finite_number(
        data["box"]["pocket_padding_angstrom"],
        "box.pocket_padding_angstrom",
        minimum=0.0,
    )
    _finite_number(
        data["box"]["minimum_fe_margin_angstrom"],
        "box.minimum_fe_margin_angstrom",
        minimum=0.0,
    )
    if data["box"]["center_atoms"] not in {"CA", "heavy"}:
        raise ConfigurationError("box.center_atoms must be CA or heavy")

    for key in ("embedding_attempts", "optimization_iterations"):
        _positive_integer(data["ligand"][key], f"ligand.{key}")
    for key in (
        "exhaustiveness",
        "num_modes",
        "replicates",
        "workers",
        "cpu_per_task",
        "timeout_seconds",
    ):
        _positive_integer(data["docking"][key], f"docking.{key}")
    if isinstance(data["docking"]["energy_range"], bool) or not isinstance(
        data["docking"]["energy_range"], int
    ) or data["docking"]["energy_range"] < 0:
        raise ConfigurationError("docking.energy_range must be a non-negative integer")


def _resolve_declared_paths(data: dict[str, Any], base: Path) -> None:
    for section, key in _PATH_FIELDS:
        value = data[section][key]
        if value is None or not isinstance(value, str) or not value.strip():
            continue
        expanded = Path(value).expanduser()
        if expanded.is_absolute():
            data[section][key] = str(expanded)
            continue
        if section == "foldseek" and key == "executable" and "/" not in value:
            continue
        data[section][key] = str((base / expanded).resolve())


def load_config(path: Optional[Path] = None) -> Config:
    data = copy.deepcopy(DEFAULT_CONFIG)
    source = None
    if path is not None:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise ConfigurationError(f"Configuration file does not exist: {source}")
        with source.open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle)
        if values is None:
            values = {}
        if not isinstance(values, Mapping):
            raise ConfigurationError("Configuration root must be a mapping")
        _merge_strict(data, values, "")
    _resolve_declared_paths(data, source.parent if source else Path.cwd())
    validate_config(data)
    return Config(data=data, source=source)


def set_override(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    section, key = dotted_key.split(".", 1)
    if section not in data or key not in data[section]:
        raise ConfigurationError(f"Unknown override: {dotted_key}")
    data[section][key] = value
    validate_config(data)
