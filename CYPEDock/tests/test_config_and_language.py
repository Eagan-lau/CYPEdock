from __future__ import annotations

import re
from pathlib import Path

import pytest

from cypedock.cli import main
from cypedock.config import load_config
from cypedock.errors import ConfigurationError


def test_unknown_configuration_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("alignment:\n  invented_key: 1\n", encoding="ascii")
    with pytest.raises(ConfigurationError, match="Unknown configuration key"):
        load_config(path)


def test_init_config_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "production.yaml"
    assert main(["init-config", "--output", str(output)]) == 0
    loaded = load_config(output)
    assert loaded.data["alignment"]["local_rmsd_cutoff_angstrom"] == 2.5


def test_repository_text_and_paths_contain_no_han_characters() -> None:
    root = Path(__file__).resolve().parents[1]
    pattern = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
    text_suffixes = {
        ".py",
        ".md",
        ".toml",
        ".yaml",
        ".yml",
        ".json",
        ".jsonl",
        ".cff",
        ".txt",
    }
    violations: list[str] = []
    for path in root.rglob("*"):
        relative = str(path.relative_to(root))
        if pattern.search(relative):
            violations.append(relative)
        if path.is_file() and path.suffix.lower() in text_suffixes:
            text = path.read_text(encoding="utf-8")
            if pattern.search(text):
                violations.append(relative)
    assert not violations
