# CYPEdock Runtime Workspace

The `workspace/` directory in the repository root separates user inputs from
generated run artifacts.

## Input locations

- `inputs/conformations/`: one protein conformation, a conformational ensemble,
  or a text manifest accepted by `--ensemble`.
- `inputs/reference_holo/`: an optional AF3 or experimental holo structure
  accepted by `--reference-holo`. The coordinate file must contain the protein
  and a heme-like cofactor; PDB and mmCIF formats are supported.
- `inputs/ligands/`: a SMILES, SDF, MOL, or MOL2 ligand input. SMILES files use
  one ligand per line as `SMILES [identifier]`; SDF and MOL2 may contain
  multiple records.

## Output location

Write each run to a distinct subdirectory of `outputs/`, for example
`outputs/run_001`. CYPEdock creates all result files below that directory.

The workspace paths are conventions rather than fixed program requirements.
Absolute input and output paths remain supported.
