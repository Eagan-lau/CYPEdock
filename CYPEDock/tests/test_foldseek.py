from __future__ import annotations

import json
from pathlib import Path

from cypedock.foldseek import FoldseekHit, mapping_from_hit
from cypedock.library import (
    default_database_prefix,
    library_manifest_path,
    load_library,
    validate_database_prefix,
)
from cypedock.provenance import sha256_file
from cypedock.structure import write_pdb

from conftest import cofactor_residue, protein_residues


def test_foldseek_gap_mapping_uses_one_based_starts() -> None:
    hit = FoldseekHit(
        query="query",
        target="template",
        alignment_tm_score=0.8,
        query_tm_score=0.8,
        target_tm_score=0.8,
        rmsd_angstrom=1.2,
        sequence_identity=0.75,
        alignment_length=5,
        query_coverage=1.0,
        target_coverage=1.0,
        query_start=1,
        query_end=4,
        target_start=1,
        target_end=4,
        query_alignment="AC-DE",
        target_alignment="A-CDE",
        evalue=1e-8,
        bits=80.0,
    )
    mapping = mapping_from_hit(hit, reference_length=4, target_length=4)
    assert mapping.reference_to_target == {0: 0, 2: 2, 3: 3}
    assert mapping.identity == 1.0


def test_generated_workstation_library_schema_is_supported(tmp_path: Path) -> None:
    root = tmp_path / "foldseek_library"
    structures = tmp_path / "structures"
    root.mkdir()
    structures.mkdir()
    structure = write_pdb(
        [*protein_residues(), cofactor_residue()],
        structures / "template.pdb",
    )
    record = {
        "schema_version": 1,
        "target_id": "template.pdb",
        "structure_file": "/relocated/original/template.pdb",
        "structure_sha256": sha256_file(structure),
        "protein_chain": "A",
        "cofactor_groups": [
            {
                "detection_mode": "single_residue",
                "residues": [
                    {
                        "chain": "H",
                        "number": "501",
                        "insertion_code": "",
                        "name": "XYZ",
                    }
                ],
            }
        ],
    }
    manifest = root / "library_manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="ascii")
    Path(f"{root / 'cyp450_holo'}.dbtype").write_bytes(b"\x00\x00\x00\x00")

    entries = load_library(root, structure_root=structures)
    assert library_manifest_path(root) == manifest
    assert entries["template.pdb"].structure_file == structure
    assert entries["template.pdb"].cofactor_residues[0]["number"] == "501"
    prefix = default_database_prefix(root)
    assert prefix == root / "cyp450_holo"
    validate_database_prefix(prefix)
