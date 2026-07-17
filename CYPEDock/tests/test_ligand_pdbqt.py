from __future__ import annotations

import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

from cypedock.ligand import prepare_ligand, read_ligands, read_smiles
from cypedock.pdbqt import write_ligand_pdbqt, write_receptor_pdbqt
from cypedock.vina import dock, materialize_vina

from conftest import protein_residues


def test_rdkit_embedding_pdbqt_and_bundled_vina(tmp_path: Path) -> None:
    smiles = tmp_path / "ligands.smi"
    smiles.write_text("CCO ethanol\n", encoding="ascii")
    ligand = read_smiles(smiles)[0]
    prepared = prepare_ligand(
        ligand,
        tmp_path / "ligand",
        {"embedding_attempts": 1000, "optimization_iterations": 1000},
        random_seed=7,
    )
    ligand_pdbqt = write_ligand_pdbqt(
        prepared.molecule, ligand.ligand_id, tmp_path / "ligand.pdbqt"
    )
    text = ligand_pdbqt.read_text(encoding="ascii")
    assert "ROOT" in text
    assert "TORSDOF" in text
    receptor_pdbqt = write_receptor_pdbqt(
        protein_residues(2), tmp_path / "receptor.pdbqt"
    )
    vina = materialize_vina(tmp_path / "tools")
    result = dock(
        executable=vina,
        receptor_pdbqt=receptor_pdbqt,
        ligand_pdbqt=ligand_pdbqt,
        center=(1.0, 0.0, 0.0),
        size=(12.0, 12.0, 12.0),
        state_id="state",
        ligand_id="ethanol",
        replicate=1,
        base_seed=9,
        output_dir=tmp_path / "docking",
        settings={
            "exhaustiveness": 1,
            "num_modes": 1,
            "energy_range": 3,
            "cpu_per_task": 1,
            "timeout_seconds": 120,
        },
    )
    assert result.status == "SUCCESS"
    assert len(result.scores) == 1
    assert result.output_out == result.output_pdbqt.with_suffix(".out")
    assert result.output_out.is_file()
    vina_output = result.output_out.read_text(encoding="utf-8")
    assert "mode |   affinity | dist from best mode" in vina_output
    assert "(kcal/mol)" in vina_output


def test_aniline_nitrogen_uses_nonacceptor_autodock_type(tmp_path: Path) -> None:
    smiles = tmp_path / "aniline.smi"
    smiles.write_text("Nc1ccccc1 aniline\n", encoding="ascii")
    ligand = read_smiles(smiles)[0]
    prepared = prepare_ligand(
        ligand,
        tmp_path / "aniline",
        {"embedding_attempts": 1000, "optimization_iterations": 1000},
        random_seed=11,
    )
    path = write_ligand_pdbqt(
        prepared.molecule,
        ligand.ligand_id,
        tmp_path / "aniline.pdbqt",
    )
    nitrogen_lines = [
        line
        for line in path.read_text(encoding="ascii").splitlines()
        if line.startswith("HETATM") and line[12:16].strip().startswith("N")
    ]
    assert len(nitrogen_lines) == 1
    assert nitrogen_lines[0].split()[-1] == "N"


def test_vina_launch_failure_is_a_failed_task(tmp_path: Path) -> None:
    result = dock(
        executable=tmp_path / "missing-vina",
        receptor_pdbqt=tmp_path / "receptor.pdbqt",
        ligand_pdbqt=tmp_path / "ligand.pdbqt",
        center=(0.0, 0.0, 0.0),
        size=(12.0, 12.0, 12.0),
        state_id="state__ambiguous",
        ligand_id="ligand",
        replicate=1,
        base_seed=9,
        output_dir=tmp_path / "docking",
        settings={
            "exhaustiveness": 1,
            "num_modes": 1,
            "energy_range": 3,
            "cpu_per_task": 1,
            "timeout_seconds": 1,
        },
    )
    assert result.status == "FAIL"
    assert "could not be executed" in result.error
    assert len(result.task_id.split("__")[-1]) == 12
    assert not result.output_out.exists()


def test_sdf_mol_and_mol2_inputs(tmp_path: Path) -> None:
    ethanol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    assert AllChem.EmbedMolecule(ethanol, randomSeed=7) == 0
    ethanol.SetProp("_Name", "ethanol_sdf")
    ethylamine = Chem.AddHs(Chem.MolFromSmiles("CCN"))
    assert AllChem.EmbedMolecule(ethylamine, randomSeed=11) == 0
    ethylamine.SetProp("_Name", "ethylamine_sdf")
    sdf = tmp_path / "ligands.sdf"
    writer = Chem.SDWriter(str(sdf))
    writer.write(ethanol)
    writer.write(ethylamine)
    writer.close()
    sdf_ligands = read_ligands(sdf)
    assert [value.ligand_id for value in sdf_ligands] == [
        "ethanol_sdf",
        "ethylamine_sdf",
    ]
    assert all(value.input_has_3d for value in sdf_ligands)

    mol = tmp_path / "ethanol.mol"
    Chem.MolToMolFile(ethanol, str(mol))
    mol_ligand = read_ligands(mol)[0]
    assert mol_ligand.source_format == "mol"
    assert mol_ligand.input_has_3d

    mol2 = tmp_path / "ethanol.mol2"
    mol2.write_text(
        """@<TRIPOS>MOLECULE
ethanol_mol2
3 2 1 0 0
SMALL
NO_CHARGES

@<TRIPOS>ATOM
1 C1 0.0000 0.0000 0.0000 C.3 1 LIG 0.0000
2 C2 1.5000 0.0000 0.0000 C.3 1 LIG 0.0000
3 O3 2.1000 1.2000 0.0000 O.3 1 LIG 0.0000
@<TRIPOS>BOND
1 1 2 1
2 2 3 1
@<TRIPOS>SUBSTRUCTURE
1 LIG 1
""",
        encoding="ascii",
    )
    mol2_ligand = read_ligands(mol2)[0]
    assert mol2_ligand.ligand_id == "ethanol_mol2"
    assert mol2_ligand.source_format == "mol2"
    assert mol2_ligand.input_has_3d

    for index, ligand in enumerate(
        [sdf_ligands[0], mol_ligand, mol2_ligand], start=1
    ):
        destination = tmp_path / f"prepared_{index}"
        prepared = prepare_ligand(
            ligand,
            destination,
            {"embedding_attempts": 1000, "optimization_iterations": 1000},
            random_seed=20 + index,
        )
        assert prepared.sdf_path.is_file()
        metadata = json.loads(
            (destination / f"{ligand.ligand_id}.json").read_text(encoding="utf-8")
        )
        assert metadata["coordinate_source"] == "input_3d"
