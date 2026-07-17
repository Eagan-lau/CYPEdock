from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import numpy as np

from cypedock.config import Config, DEFAULT_CONFIG
from cypedock.cypocket import PocketResult
from cypedock.errors import InputError
from cypedock.pipeline import Pipeline, docking_box
from cypedock.structure import write_pdb

from conftest import cofactor_residue, protein_residues, write_synthetic_pair


FAKE_CYPOCKET = r"""#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--protein", required=True)
parser.add_argument("--output-dir", required=True)
args, _ = parser.parse_known_args()
protein = Path(args.protein).resolve()
output = Path(args.output_dir).resolve()
output.mkdir(parents=True, exist_ok=True)
sample = protein.stem
if sample.endswith("_protein"):
    sample = sample[:-8]
lines = [
    line for line in protein.read_text(encoding="ascii").splitlines()
    if line.startswith(("ATOM", "HETATM"))
]
keys = {
    (line[21].strip(), line[22:26], line[26], line[17:20])
    for line in lines
}
pocket = output / f"{sample}_predicted_pocket.pdb"
pocket.write_text("\n".join(lines) + "\nEND\n", encoding="ascii")
(output / f"{sample}_residues.csv").write_text(
    "sample_id,predicted_label\n", encoding="ascii"
)
digest = hashlib.sha256(protein.read_bytes()).hexdigest()
(output / f"{sample}_summary.json").write_text(
    json.dumps({
        "sample_id": sample,
        "protein_pdb_sha256": digest,
        "n_predicted_pocket": len(keys)
    }),
    encoding="ascii"
)
"""


def fake_cypocket_root(tmp_path: Path) -> Path:
    root = tmp_path / "CYPocket"
    project = root / "p450pocket"
    package = project / "p450pocket"
    model = root / "P450model"
    package.mkdir(parents=True)
    model.mkdir()
    (project / "predict_pdb.py").write_text(FAKE_CYPOCKET, encoding="ascii")
    (package / "__init__.py").write_text('__version__ = "test"\n', encoding="ascii")
    (model / "pocketminer.index").write_text("index", encoding="ascii")
    (model / "pocketminer.data-00000-of-00001").write_text("data", encoding="ascii")
    (model / "best_protein_distilbert_classifier.pth").write_text(
        "weights", encoding="ascii"
    )
    return project


def test_complete_simplified_pipeline(tmp_path: Path) -> None:
    ensemble, reference = write_synthetic_pair(tmp_path)
    write_pdb([*protein_residues(), cofactor_residue()], ensemble)
    smiles = tmp_path / "ligands.smi"
    smiles.write_text("CCO ethanol\n", encoding="ascii")
    root = fake_cypocket_root(tmp_path)
    data = copy.deepcopy(DEFAULT_CONFIG)
    data["input"]["chain"] = "A"
    data["reference"]["protein_chain"] = "A"
    data["alignment"]["minimum_local_residues"] = 4
    data["alignment"]["cofactor_clash_distance_angstrom"] = 1.0
    data["foldseek"]["enabled"] = False
    data["cypocket"]["python"] = sys.executable
    data["cypocket"]["root"] = str(root)
    data["cypocket"]["feature_python"] = sys.executable
    data["cypocket"]["classifier_python"] = sys.executable
    data["box"]["auto_size"] = True
    data["box"]["size_angstrom"] = [12.0, 12.0, 12.0]
    data["docking"]["exhaustiveness"] = 1
    data["docking"]["num_modes"] = 1
    output = tmp_path / "run"
    Pipeline(Config(data=data, source=None)).run(
        ensemble=ensemble,
        reference_holo=reference,
        smiles_file=smiles,
        output_dir=output,
    )
    assert (output / "run_manifest.json").is_file()
    assert (output / "state_results.tsv").read_text(encoding="utf-8").count("READY") == 1
    protein_input = output / "states" / "target" / "target_protein.pdb"
    assert "HETATM" not in protein_input.read_text(encoding="ascii")
    receptor = output / "states" / "target" / "target_receptor.pdb"
    assert "HETATM" in receptor.read_text(encoding="ascii")
    box = json.loads(
        (output / "states" / "target" / "box.json").read_text(encoding="utf-8")
    )
    assert box["size_angstrom"][0] > 12.0
    assert box["minimum_fe_box_margin_angstrom"] >= 2.0
    docking = (output / "docking_results.tsv").read_text(encoding="utf-8")
    assert "SUCCESS" in docking
    pose_files = sorted((output / "docking").glob("*.pdbqt"))
    score_files = sorted((output / "docking").glob("*.out"))
    assert [path.stem for path in score_files] == [path.stem for path in pose_files]
    assert all("(kcal/mol)" in path.read_text(encoding="utf-8") for path in score_files)
    alignment_header = (output / "alignment_attempts.tsv").read_text(
        encoding="utf-8"
    ).splitlines()[0]
    assert "maximum_local_residue_rmsd_angstrom" in alignment_header
    assert "target_fe_s_distance_angstrom" in alignment_header
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["inputs"]["smiles_file"]["sha256"]
    assert manifest["inputs"]["conformations"][0]["sha256"]
    assert manifest["software"]["python_packages"]["biopython"]


def test_overwrite_rejects_output_that_contains_inputs(tmp_path: Path) -> None:
    ensemble, reference = write_synthetic_pair(tmp_path)
    smiles = tmp_path / "ligands.smi"
    smiles.write_text("CCO ethanol\n", encoding="ascii")
    data = copy.deepcopy(DEFAULT_CONFIG)
    data["foldseek"]["enabled"] = False
    pipeline = Pipeline(Config(data=data, source=None))
    with pytest.raises(InputError, match="protected input"):
        pipeline.run(
            ensemble=ensemble,
            reference_holo=reference,
            smiles_file=smiles,
            output_dir=tmp_path,
            overwrite=True,
        )
    assert ensemble.is_file()
    assert reference.is_file()
    assert smiles.is_file()


def test_fixed_and_automatic_box_modes(tmp_path: Path) -> None:
    pocket = PocketResult(
        pocket_pdb=tmp_path / "pocket.pdb",
        residue_table=tmp_path / "residues.csv",
        summary_json=tmp_path / "summary.json",
        center=np.asarray([0.0, 0.0, 0.0]),
        coordinates=np.asarray([[-20.0, 0.0, 0.0], [20.0, 0.0, 0.0]]),
        residue_count=2,
    )
    fixed = copy.deepcopy(DEFAULT_CONFIG["box"])
    size, margin = docking_box(pocket, [cofactor_residue()], fixed)
    assert size == (25.0, 25.0, 25.0)
    assert margin >= 2.0

    automatic = copy.deepcopy(fixed)
    automatic["auto_size"] = True
    automatic["maximum_size_angstrom"] = [40.0, 60.0, 60.0]
    details: dict = {}
    size, margin = docking_box(
        pocket, [cofactor_residue()], automatic, details=details
    )
    assert size == (40.0, 25.0, 25.0)
    assert margin >= 2.0
    assert details["requested_size_angstrom"] == [48.0, 25.0, 25.0]
    assert details["clipped_to_maximum"] is True
    assert details["clipped_axes"] == ["x"]


def test_state_ligand_cartesian_product(tmp_path: Path) -> None:
    ensemble_dir = tmp_path / "ensemble"
    ensemble_dir.mkdir()
    write_pdb(protein_residues(), ensemble_dir / "state_a.pdb")
    write_pdb(protein_residues(), ensemble_dir / "state_b.pdb")
    reference = tmp_path / "reference_holo.pdb"
    write_pdb([*protein_residues(), cofactor_residue()], reference)
    smiles = tmp_path / "ligands.smi"
    smiles.write_text("CCO ethanol\nCCN ethylamine\n", encoding="ascii")
    root = fake_cypocket_root(tmp_path)
    data = copy.deepcopy(DEFAULT_CONFIG)
    data["input"]["chain"] = "A"
    data["reference"]["protein_chain"] = "A"
    data["alignment"]["minimum_local_residues"] = 4
    data["alignment"]["cofactor_clash_distance_angstrom"] = 1.0
    data["foldseek"]["enabled"] = False
    data["cypocket"]["python"] = sys.executable
    data["cypocket"]["root"] = str(root)
    data["cypocket"]["feature_python"] = sys.executable
    data["cypocket"]["classifier_python"] = sys.executable
    data["docking"]["exhaustiveness"] = 1
    data["docking"]["num_modes"] = 1
    data["docking"]["workers"] = 2
    output = tmp_path / "cartesian_run"
    Pipeline(Config(data=data, source=None)).run(
        ensemble=ensemble_dir,
        reference_holo=reference,
        ligand_input=smiles,
        output_dir=output,
    )
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["accepted_conformations"] == 2
    assert manifest["counts"]["ligands"] == 2
    assert manifest["counts"]["docking_tasks"] == 4
    assert manifest["counts"]["successful_docking_tasks"] == 4
