# Verification Record

Verification date: 2026-07-17

## Automated tests

```text
18 passed
```

The suite covers local Kabsch alignment, arbitrary cofactor residue names,
heme-like validation of selected cofactors, aggregate and maximum
per-residue local-RMSD rejection, proximal Cys Fe-S geometry, Foldseek gap
correspondence, strict configuration, English-only source validation, SMILES,
multi-record SDF, MOL, and MOL2 ingestion, RDKit PDBQT preparation, bundled
Vina execution, paired PDBQT and Vina `.out` score reports, protein-only
CYPocket input, cofactor restoration, fixed and
automatically clipped box construction, and the complete simplified pipeline. A real
two-conformation by two-ligand test verifies four Cartesian-product Vina tasks.
Destructive output-path overlap and Vina launch-failure handling are covered by
dedicated regression tests.

## External Foldseek library validation

The self-contained workstation library at `CYP450_foldseek_db` was validated
against its generated manifest and its relative `templates/` coordinate
directory. The Foldseek database symlinks were also normalized to relative
targets so that the library can be relocated as one unit.

```text
Input candidate structures: 2,633
Retained heme-bearing templates: 2,497
Excluded structures: 136
Single-residue cofactors: 2,492
Split porphyrin/iron cofactors: 5
Manifest target IDs matching Foldseek lookup IDs: 2,497/2,497
Relative template paths resolving inside the library: 2,497/2,497
Manifest and coordinate SHA-256 validation: PASS
Database prefix: cyp450_holo
Foldseek version: 1cb3a80d9566820bc223bdf4d34d30d85ffbc4f4
Status: PASS
```

A complete run was then performed without `--reference-holo`. Foldseek
retrieved `4egm_B.pdb` as the first template for the protein-only `4egm_B`
query, after which local-site validation, CYPocket, box construction, and Vina
docking completed successfully.

```text
Template source: foldseek_library
Template ID: 4egm_B.pdb
Alignment TM-score: 1.0
Local backbone RMSD: 3.66e-15 A
Reference and target Fe-S distance: 2.279 A
CYPocket pocket residues: 53
Pocket coordinates outside the docking box: 0
Successful Vina tasks: 1/1
Best ethanol affinity: -2.607 kcal/mol
Status: PASS
```

## CYPocket model smoke test

The packaged CYPocket 0.1.0 entry point and the existing PocketMiner and
classifier checkpoints were used without retraining.

```text
Input: AF-A0A010RW94-F1-model_v4_protein.pdb
Predicted pocket residues: 71
Pocket center: 55.620578, -1.949704, 8.295028 A
Status: PASS
```

## Real holo end-to-end smoke test

The crystal structure `4egm_B.pdb` was used as both the input conformation
and the reference holo structure. The input contains HEM, allowing direct
verification of the protein-only CYPocket boundary and cofactor restoration.

```text
Reference source: supplied_holo
Sequence identity: 1.0
Local shell residues: 120
Local coverage: 1.0
Local backbone RMSD: 3.66e-15 A
Maximum local-residue backbone RMSD: 7.94e-15 A
Reference proximal Cys Fe-S distance: 2.279 A
Target proximal Cys Fe-S distance: 2.279 A
Transferred cofactor minimum protein distance: 2.279 A
CYPocket pocket residues: 53
Protein-only CYPocket input HETATM count: 0
Docking receptor HEM atom count: 43
Dynamic docking-box dimensions: 31.821, 37.013, 33.130 A
Minimum Fe-to-box-face distance: 9.064 A
Pocket coordinates outside the docking box: 0
Vina tasks: 1
Successful Vina tasks: 1
Best ethanol affinity: -2.607 kcal/mol
Status: PASS
```

AutoDock Vina reported version 1.2.3 and matched the expected SHA-256 digest.
The bundled workstation copy was also compared with the official upstream
`vina_1.2.3_linux_x86_64` release asset and matched its 4,979,416-byte size
and SHA-256 digest exactly.
The local-site test passed the configured 4.0 A maximum per-residue RMSD and
1.80-3.00 A Fe-S criteria. The box expanded beyond its 25 A per-axis minimum,
enclosed every predicted-pocket coordinate with the requested padding, and
retained more than the required 2.0 A Fe margin.

## Interpretation boundary

These checks establish software execution, interface contracts, coordinate
transfer, and quality-control behavior. They do not establish docking-pose
fidelity, affinity ranking, or prospective predictive performance. The
internal PDBQT preparation pathway remains a research-preview implementation:
it does not add missing receptor hydrogens or determine system-specific
protonation and tautomeric states. No docking result generated through this
pathway should support a publication-level scientific conclusion until the
receptor and heme preparation protocol has been independently justified and
retrospective redocking has reproduced experimentally resolved CYP-ligand
poses representative of the intended application domain.

## Remaining unverified branch

A curated CYP redocking benchmark, including pose-RMSD acceptance criteria and
predeclared treatment of protonation, axial coordination, structural waters,
and cofactor atom types, has not yet been completed. The Vina smoke test above
therefore verifies executable integration only.
