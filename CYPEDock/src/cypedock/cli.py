"""Command-line interface for CYPEdock."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional, Sequence

import yaml

from . import __version__
from .config import DEFAULT_CONFIG, Config, load_config, set_override, validate_config
from .cypocket import software_record
from .errors import CYPEdockError, ConfigurationError
from .library import (
    default_database_prefix,
    default_library_root,
    library_manifest_path,
    load_library,
    validate_database_prefix,
)
from .pipeline import Pipeline
from .provenance import run_command, sha256_file
from .vina import materialize_vina, version as vina_version


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cypedock",
        description="Local-site-guided cytochrome P450 ensemble docking",
    )
    parser.add_argument("--version", action="version", version=f"CYPEdock {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run the complete ensemble-docking workflow")
    run.add_argument("--ensemble", type=Path, required=True)
    run.add_argument("--reference-holo", type=Path)
    ligand_input = run.add_mutually_exclusive_group(required=True)
    ligand_input.add_argument(
        "--ligands",
        type=Path,
        help="Ligand input in SMILES, SDF, MOL, or MOL2 format",
    )
    ligand_input.add_argument(
        "--smiles-file",
        type=Path,
        help="Deprecated alias for a SMILES ligand input",
    )
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--config", type=Path)
    run.add_argument("--chain")
    run.add_argument("--reference-chain")
    run.add_argument(
        "--cofactor",
        action="append",
        default=[],
        metavar="CHAIN:NUMBER[:ICODE[:NAME]]",
        help="Explicit cofactor residue selector; may be repeated",
    )
    run.add_argument("--local-rmsd-cutoff", type=float)
    run.add_argument("--foldseek-bin")
    run.add_argument("--library-root", type=Path)
    run.add_argument("--database-prefix", type=Path)
    run.add_argument("--structure-root", type=Path)
    run.add_argument("--disable-library-fallback", action="store_true")
    run.add_argument("--cypocket-python")
    run.add_argument("--cypocket-root", type=Path)
    run.add_argument("--cypocket-feature-python")
    run.add_argument("--cypocket-classifier-python")
    run.add_argument("--cypocket-device")
    run.add_argument("--vina-bin")
    run.add_argument(
        "--box-size",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Fixed box size, or per-axis minimum when --auto-box is enabled",
    )
    run.add_argument(
        "--auto-box",
        action="store_true",
        help="Automatically expand each box axis to enclose the pocket and heme Fe",
    )
    run.add_argument("--workers", type=int)
    run.add_argument("--overwrite", action="store_true")

    doctor = subparsers.add_parser("doctor", help="Check required software and model files")
    doctor.add_argument("--config", type=Path)
    doctor.add_argument("--require-library", action="store_true")

    library = subparsers.add_parser(
        "validate-library", help="Validate a CYP450 Foldseek library"
    )
    library.add_argument("--library-root", type=Path, required=True)
    library.add_argument("--database-prefix", type=Path)
    library.add_argument("--structure-root", type=Path)

    initialize = subparsers.add_parser(
        "init-config", help="Write the default configuration"
    )
    initialize.add_argument("--output", type=Path, required=True)

    vina = subparsers.add_parser("vina", help="Run the bundled Vina information action")
    vina.add_argument("--version", action="store_true", required=True)
    vina.add_argument("--vina-bin")
    return parser


def _cofactor_selector(value: str) -> dict:
    fields = value.split(":")
    if len(fields) not in {2, 3, 4}:
        raise ConfigurationError(
            f"Invalid cofactor selector {value!r}; expected CHAIN:NUMBER[:ICODE[:NAME]]"
        )
    chain = "" if fields[0] == "_" else fields[0]
    try:
        number = int(fields[1])
    except ValueError as exc:
        raise ConfigurationError(f"Invalid cofactor residue number: {fields[1]!r}") from exc
    selector = {
        "chain": chain,
        "number": number,
        "insertion_code": fields[2] if len(fields) >= 3 else "",
    }
    if len(fields) == 4 and fields[3]:
        selector["name"] = fields[3]
    return selector


def _run_config(args: argparse.Namespace) -> Config:
    loaded = load_config(args.config)
    data = copy.deepcopy(loaded.data)
    overrides = {
        "input.chain": args.chain,
        "reference.protein_chain": args.reference_chain,
        "alignment.local_rmsd_cutoff_angstrom": args.local_rmsd_cutoff,
        "foldseek.executable": args.foldseek_bin,
        "foldseek.library_root": (
            str(args.library_root.expanduser().resolve()) if args.library_root else None
        ),
        "foldseek.database_prefix": (
            str(args.database_prefix.expanduser().resolve())
            if args.database_prefix
            else None
        ),
        "foldseek.structure_root": (
            str(args.structure_root.expanduser().resolve())
            if args.structure_root
            else None
        ),
        "cypocket.python": args.cypocket_python,
        "cypocket.root": (
            str(args.cypocket_root.expanduser().resolve()) if args.cypocket_root else None
        ),
        "cypocket.feature_python": args.cypocket_feature_python,
        "cypocket.classifier_python": args.cypocket_classifier_python,
        "cypocket.device": args.cypocket_device,
        "docking.vina_executable": args.vina_bin,
        "docking.workers": args.workers,
    }
    for key, value in overrides.items():
        if value is not None:
            set_override(data, key, value)
    if args.cofactor:
        data["reference"]["cofactor_residues"] = [
            _cofactor_selector(value) for value in args.cofactor
        ]
    if args.disable_library_fallback:
        data["foldseek"]["enabled"] = False
    if args.box_size is not None:
        data["box"]["size_angstrom"] = list(args.box_size)
    if args.auto_box:
        data["box"]["auto_size"] = True
    validate_config(data)
    return Config(data=data, source=loaded.source)


def _run(args: argparse.Namespace) -> int:
    config = _run_config(args)
    Pipeline(config).run(
        ensemble=args.ensemble,
        reference_holo=args.reference_holo,
        ligand_input=args.ligands,
        smiles_file=args.smiles_file,
        output_dir=args.output,
        overwrite=args.overwrite,
    )
    return 0


def _doctor(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    report: dict[str, object] = {"cypedock": __version__}
    failures: list[str] = []
    try:
        report["cypocket"] = software_record(config.data["cypocket"])
    except CYPEdockError as exc:
        report["cypocket"] = {"status": "FAIL", "error": str(exc)}
        failures.append("CYPocket")

    foldseek_value = str(config.data["foldseek"]["executable"])
    foldseek_path = shutil.which(foldseek_value)
    if foldseek_path is None and Path(foldseek_value).is_file():
        foldseek_path = str(Path(foldseek_value).resolve())
    if foldseek_path is None:
        report["foldseek"] = {"status": "UNAVAILABLE", "configured": foldseek_value}
        if args.require_library:
            failures.append("Foldseek")
    else:
        result = run_command([foldseek_path, "version"], timeout=10, check=False)
        report["foldseek"] = {
            "status": "PASS" if result.returncode == 0 else "FAIL",
            "path": foldseek_path,
            "version": (result.stdout + result.stderr).strip(),
            "sha256": sha256_file(Path(foldseek_path)),
        }
        if result.returncode != 0 and args.require_library:
            failures.append("Foldseek")

    root = (
        Path(config.data["foldseek"]["library_root"]).resolve()
        if config.data["foldseek"]["library_root"]
        else default_library_root().resolve()
    )
    try:
        structure_root = (
            Path(config.data["foldseek"]["structure_root"]).resolve()
            if config.data["foldseek"]["structure_root"]
            else None
        )
        entries = load_library(root, structure_root=structure_root)
        prefix = (
            Path(config.data["foldseek"]["database_prefix"]).resolve()
            if config.data["foldseek"]["database_prefix"]
            else default_database_prefix(root)
        )
        if entries:
            validate_database_prefix(prefix)
        report["cyp450_library"] = {
            "status": "PASS" if entries else "EMPTY",
            "root": str(root),
            "manifest": str(library_manifest_path(root)),
            "database_prefix": str(prefix),
            "structure_root": str(structure_root) if structure_root else None,
            "entry_count": len(entries),
        }
        if args.require_library and not entries:
            failures.append("CYP450 library")
    except CYPEdockError as exc:
        report["cyp450_library"] = {"status": "FAIL", "error": str(exc)}
        if args.require_library:
            failures.append("CYP450 library")

    try:
        with tempfile.TemporaryDirectory(prefix="cypedock-doctor-") as temporary:
            executable = materialize_vina(
                Path(temporary), config.data["docking"]["vina_executable"]
            )
            report["vina"] = {
                "status": "PASS",
                "version": vina_version(executable),
                "sha256": sha256_file(executable),
            }
    except CYPEdockError as exc:
        report["vina"] = {"status": "FAIL", "error": str(exc)}
        failures.append("Vina")
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    if failures:
        print("FAILED: " + ", ".join(sorted(set(failures))), file=sys.stderr)
        return 1
    return 0


def _validate_library(args: argparse.Namespace) -> int:
    root = args.library_root.expanduser().resolve()
    structure_root = (
        args.structure_root.expanduser().resolve() if args.structure_root else None
    )
    entries = load_library(root, structure_root=structure_root)
    prefix = (
        args.database_prefix.expanduser().resolve()
        if args.database_prefix
        else default_database_prefix(root)
    )
    if entries:
        validate_database_prefix(prefix)
    print(
        json.dumps(
            {
                "library_root": str(root),
                "manifest": str(library_manifest_path(root)),
                "database_prefix": str(prefix),
                "structure_root": str(structure_root) if structure_root else None,
                "entry_count": len(entries),
                "status": "PASS" if entries else "EMPTY",
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _vina(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory(prefix="cypedock-vina-") as temporary:
        executable = materialize_vina(Path(temporary), args.vina_bin)
        print(vina_version(executable))
    return 0


def _init_config(args: argparse.Namespace) -> int:
    destination = args.output.expanduser().resolve()
    if destination.exists():
        raise ConfigurationError(f"Configuration output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(DEFAULT_CONFIG, sort_keys=False),
        encoding="ascii",
    )
    print(destination)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "run":
            return _run(args)
        if args.command == "doctor":
            return _doctor(args)
        if args.command == "validate-library":
            return _validate_library(args)
        if args.command == "init-config":
            return _init_config(args)
        if args.command == "vina":
            return _vina(args)
        raise ConfigurationError(f"Unknown command: {args.command}")
    except CYPEdockError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
