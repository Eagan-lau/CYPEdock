"""Bundled AutoDock Vina execution and result parsing."""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

from .errors import ConfigurationError, ExternalToolError
from .provenance import run_command, sha256_file


BUNDLED_VINA_VERSION = "AutoDock Vina v1.2.3"
BUNDLED_VINA_SHA256 = "04a665729f86cee39e0c18fdbc397a85bda24e81b76685ed6d3e3beec405aa0c"


@dataclass(frozen=True)
class VinaScore:
    pose_rank: int
    affinity_kcal_mol: float
    rmsd_lower_bound: float
    rmsd_upper_bound: float


@dataclass(frozen=True)
class DockingResult:
    task_id: str
    state_id: str
    ligand_id: str
    replicate: int
    seed: int
    output_pdbqt: Path
    output_out: Path
    scores: tuple[VinaScore, ...]
    status: str
    error: str


def bundled_vina_source() -> Path:
    return Path(__file__).resolve().parent / "resources" / "vina" / "linux-x86_64" / "vina"


def materialize_vina(destination_dir: Path, override: Optional[str] = None) -> Path:
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise ConfigurationError(f"Configured Vina executable is unavailable: {path}")
        return path
    machine = platform.machine().lower()
    if platform.system() != "Linux" or machine not in {"x86_64", "amd64"}:
        raise ConfigurationError(
            "The bundled Vina executable requires Linux x86-64; configure "
            "docking.vina_executable for this platform"
        )
    source = bundled_vina_source()
    if not source.is_file():
        raise ConfigurationError(f"Bundled Vina executable is missing: {source}")
    observed = sha256_file(source)
    if observed != BUNDLED_VINA_SHA256:
        raise ConfigurationError(
            f"Bundled Vina SHA-256 mismatch: expected {BUNDLED_VINA_SHA256}, observed {observed}"
        )
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / "vina"
    shutil.copyfile(source, destination)
    destination.chmod(0o755)
    return destination


def version(executable: Path) -> str:
    result = run_command([str(executable), "--version"], timeout=10)
    value = (result.stdout + "\n" + result.stderr).strip()
    if not value:
        raise ExternalToolError("Vina version probe returned empty output")
    return value.splitlines()[0].strip()


def stable_seed(base_seed: int, task_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{task_id}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") % 2_147_483_647 + 1


def parse_scores(path: Path) -> tuple[VinaScore, ...]:
    source = Path(path)
    if not source.is_file():
        raise ExternalToolError(f"Vina output does not exist: {source}")
    scores: list[VinaScore] = []
    with source.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.upper().startswith("REMARK VINA RESULT:"):
                continue
            fields = line.split(":", 1)[1].split()
            if len(fields) < 3:
                raise ExternalToolError(f"Malformed Vina result record in {source}")
            try:
                values = [float(value) for value in fields[:3]]
            except ValueError as exc:
                raise ExternalToolError(f"Non-numeric Vina result in {source}") from exc
            scores.append(
                VinaScore(
                    pose_rank=len(scores) + 1,
                    affinity_kcal_mol=values[0],
                    rmsd_lower_bound=values[1],
                    rmsd_upper_bound=values[2],
                )
            )
    if not scores:
        raise ExternalToolError(f"Vina output contains no poses: {source}")
    return tuple(scores)


def _write_vina_output(path: Path, stdout: str, stderr: str) -> Path:
    """Persist the complete Vina console output beside its pose file."""

    sections = [stdout.rstrip()] if stdout.strip() else []
    if stderr.strip():
        sections.extend(("", "[stderr]", stderr.rstrip()))
    if not sections:
        sections.append("AutoDock Vina produced no console output.")
    destination = Path(path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text("\n".join(sections) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return destination


def dock(
    *,
    executable: Path,
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    state_id: str,
    ligand_id: str,
    replicate: int,
    base_seed: int,
    output_dir: Path,
    settings: Mapping[str, object],
) -> DockingResult:
    task_id = f"{state_id}__{ligand_id}__r{replicate:02d}"
    if "__" in state_id or "__" in ligand_id:
        identity = f"{len(state_id)}:{state_id}:{len(ligand_id)}:{ligand_id}:{replicate}"
        task_id += "__" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    seed = stable_seed(base_seed, task_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pdbqt = output_dir / f"{task_id}.pdbqt"
    output_out = output_dir / f"{task_id}.out"
    command = [
        str(executable),
        "--receptor",
        str(receptor_pdbqt),
        "--ligand",
        str(ligand_pdbqt),
        "--scoring",
        "vina",
        "--center_x",
        f"{center[0]:.6f}",
        "--center_y",
        f"{center[1]:.6f}",
        "--center_z",
        f"{center[2]:.6f}",
        "--size_x",
        f"{size[0]:.6f}",
        "--size_y",
        f"{size[1]:.6f}",
        "--size_z",
        f"{size[2]:.6f}",
        "--exhaustiveness",
        str(settings["exhaustiveness"]),
        "--num_modes",
        str(settings["num_modes"]),
        "--energy_range",
        str(settings["energy_range"]),
        "--cpu",
        str(settings["cpu_per_task"]),
        "--seed",
        str(seed),
        "--out",
        str(output_pdbqt),
    ]
    try:
        result = run_command(
            command,
            timeout=int(settings["timeout_seconds"]),
            log_dir=output_dir / "logs" / task_id,
            check=False,
        )
    except ExternalToolError as exc:
        return DockingResult(
            task_id=task_id,
            state_id=state_id,
            ligand_id=ligand_id,
            replicate=replicate,
            seed=seed,
            output_pdbqt=output_pdbqt,
            output_out=output_out,
            scores=(),
            status="FAIL",
            error=str(exc),
        )
    try:
        _write_vina_output(output_out, result.stdout, result.stderr)
    except OSError as exc:
        return DockingResult(
            task_id=task_id,
            state_id=state_id,
            ligand_id=ligand_id,
            replicate=replicate,
            seed=seed,
            output_pdbqt=output_pdbqt,
            output_out=output_out,
            scores=(),
            status="FAIL",
            error=f"Could not write Vina output file {output_out}: {exc}",
        )
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()[-1:] or ["Vina returned no diagnostic"]
        return DockingResult(
            task_id=task_id,
            state_id=state_id,
            ligand_id=ligand_id,
            replicate=replicate,
            seed=seed,
            output_pdbqt=output_pdbqt,
            output_out=output_out,
            scores=(),
            status="FAIL",
            error=detail[0],
        )
    try:
        scores = parse_scores(output_pdbqt)
    except ExternalToolError as exc:
        return DockingResult(
            task_id=task_id,
            state_id=state_id,
            ligand_id=ligand_id,
            replicate=replicate,
            seed=seed,
            output_pdbqt=output_pdbqt,
            output_out=output_out,
            scores=(),
            status="FAIL",
            error=str(exc),
        )
    return DockingResult(
        task_id=task_id,
        state_id=state_id,
        ligand_id=ligand_id,
        replicate=replicate,
        seed=seed,
        output_pdbqt=output_pdbqt,
        output_out=output_out,
        scores=scores,
        status="SUCCESS",
        error="",
    )
