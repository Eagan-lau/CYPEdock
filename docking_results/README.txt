CYPEdock docking results
=======================
Each of the 32 Cell ID folders contains receptor.pdb, ligand.pdb, ligand.sdf,
complex.pdb, and run_command.txt. The ligand and complex are the selected
retrospective minimum-RMSD pose in the original receptor coordinate frame.
PDB/SDF coordinates were not optimized. Chemical validity was not reassessed.

Reproduce filtering and RMSD (Python 3.10+):
  python -m pip install -r analysis_workflow/requirements.txt
  python analysis_workflow/run_workflow.py
The script validates input_data.zip and extracts it into a new run directory.
Results are written under analysis_workflow/outputs/run_TIMESTAMP/.

To repeat an original Vina search, first run from this directory:
  python -m zipfile -e input_data.zip input_data
Then enter a Cell ID folder and execute its single line in run_command.txt.
A compatible Vina executable must be on PATH. The prepared PDBQT inputs retain
the original bytes, seed, box and search arguments. Re-docking was not run here;
exact pose reproduction can depend on the Vina build and runtime environment.

Scope and definitions
---------------------
The table combines historical crystal/initial/five-state searches with the
frozen 40-state expanded snapshot for D/E of 3WVS, 1OXA and 3AWM. These are not
uniform-budget comparisons. E is the gate-passed subset of D, not a new search.
F in this delivery is historical; later expanded-F calculations are excluded.
RMSD uses whole-protein CA alignment, no separate ligand fitting, and permitted
chemical symmetries. Fixed-name RMSD is reported separately. 3AWM uses altA as
the primary reference; 3WVS excludes crystal residues 290-293 from CA alignment.
The archive contains only the selected call per cell, including all its returned
modes; E aliases are preserved. It is not the complete scenario pool and cannot
reconstruct scenario-wide hit rates or prove its minima against omitted calls.
The supplied subset reproduces 32 cells / 309 arm-pose records / 29 physical calls /
279 unique physical poses. 3AWM D is 4.137314 A (4.14 to two decimals, not 4.13).

Public JSON copies omit machine-specific paths. Raw coordinate/chemical inputs
are unchanged; origin and public SHA-256 bindings are provided in the manifests.
