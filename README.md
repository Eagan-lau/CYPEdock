# CYPEdock Installation and User Manual

CYPEdock is a local-site-guided workflow for ensemble docking to cytochrome
P450 proteins. It aligns every input protein conformation to a supplied holo
reference or to a validated local CYP450 Foldseek library, transfers the
cofactor position, predicts pocket residues with CYPocket, constructs the
docking box, prepares one or more ligands, and executes AutoDock Vina.

This document describes installation and routine use of the complete project.
The current workstation checkout is `/media/yugengliu/DATA2/CYPEDock`;
component-level implementation, training, and validation details are provided
in the linked manuals.

## 1. Project Layout

```text
CYPEDock/
|-- README.md                         This installation and user manual
|-- ASSETS.md                         External-asset download and validation
|-- CYPEDock/                         CYPEdock workflow source and release
|   |-- configs/                      Default and workstation configurations
|   |-- release/                      Installable wheel and source archive
|   |-- src/cypedock/                 Python package
|   `-- tests/                        Workflow test suite
|-- CYPocket/                         Pocket-prediction model package
|   |-- P450model/                    Released model checkpoints only
|   `-- p450pocket/                   Training, fine-tuning, and inference code
|-- CYP450_foldseek_db/               Local CYP450 holo Foldseek database
`-- workspace/
    |-- inputs/                       Runtime input data
    `-- outputs/                      One directory per docking run
```

The three computational components have distinct roles:

- `CYPEDock/` provides alignment, quality control, pocket transfer, ligand
  preparation, docking, provenance recording, and result aggregation.
- `CYPocket/` provides residue-level pocket prediction. Its source project is
  `CYPocket/p450pocket`, whereas `CYPocket/P450model` contains only weights and
  model metadata.
- `CYP450_foldseek_db/` provides a local holo-template fallback when a suitable
  reference structure is unavailable or fails local-site quality control.

## 2. External Scientific Assets

The GitHub repository contains the complete source, release packages, tests,
small verification fixtures, example inputs, PocketMiner checkpoint, and model
metadata. Two large asset groups are distributed separately:

- the generated CYP450 Foldseek database; and
- the two trained CYPocket classifier checkpoints.

Download both archives from:

- Baidu Netdisk: <https://pan.baidu.com/s/17dh949GLClk_QC1WUDejkw?pwd=7qnp>
- Extraction code: `7qnp`

Archive names, SHA-256 checksums, extraction commands, expected paths, and
validation steps are provided in [ASSETS.md](ASSETS.md). Install these assets
before running end-to-end docking.

## 3. Workflow Summary

For each protein conformation, CYPEdock performs the following operations:

1. Selects the requested protein chain.
2. Aligns the conformation to the supplied holo reference.
3. Evaluates local structural similarity around the reference cofactor.
4. Searches the local CYP450 Foldseek database when reference-based transfer is
   unavailable or unacceptable.
5. Transfers the accepted cofactor coordinates and generates a protein-only
   structure for CYPocket.
6. Predicts pocket residues and calculates their geometric center.
7. Generates three-dimensional ligand coordinates from SMILES when required,
   or reads supported molecular structure files.
8. Docks every accepted protein conformation against every accepted ligand
   with AutoDock Vina.
9. Writes pose files, complete Vina reports, structured tables, logs, and a
   reproducibility manifest.

For `N` accepted protein conformations, `M` accepted ligands, and `R` docking
replicates, the workflow creates `N x M x R` Vina tasks.

## 4. Platform and Software Requirements

- Linux is the supported operating system.
- Python 3.9 or later is required for CYPEdock.
- Python 3.9 is required for the validated CYPocket environment.
- Conda or Mamba is recommended for CYPocket and Foldseek installation.
- Foldseek is required only for local-library fallback.
- The distributed Linux x86-64 CYPEdock wheel contains AutoDock Vina 1.2.3.
  A separate Vina installation is unnecessary on that platform.

CYPEdock and CYPocket should use separate Python environments because the
validated CYPocket TensorFlow and PyTorch dependencies are version constrained.

## 5. CYPEdock Installation

Run the following commands from the project root:

```bash
cd /media/yugengliu/DATA2/CYPEDock

python3 -m venv .venv-cypedock
.venv-cypedock/bin/python -m pip install --upgrade pip
.venv-cypedock/bin/python -m pip install \
  CYPEDock/release/cypedock-0.1.0-py3-none-linux_x86_64.whl
```

Verify the command and bundled Vina executable:

```bash
.venv-cypedock/bin/cypedock --version
.venv-cypedock/bin/cypedock vina --version
```

The source tree may be installed in editable mode for development:

```bash
.venv-cypedock/bin/python -m pip install -e CYPEDock
```

After activating the environment, the shorter `cypedock` command may be used:

```bash
source .venv-cypedock/bin/activate
cypedock --version
```

If `cypedock: command not found` is reported, either activate the environment
or invoke `/absolute/path/to/environment/bin/cypedock` directly. Installing the
wheel in one interpreter does not place the command in unrelated Conda or
virtual environments.

## 6. CYPocket Installation

CYPocket includes its compatible PocketMiner/GVP implementation and released
PocketMiner checkpoint. A separate PocketMiner source installation is not
required. End-to-end prediction still requires the dependencies listed in
`requirements-inference.txt`.

Create the validated inference environment:

```bash
cd /media/yugengliu/DATA2/CYPEDock/CYPocket/p450pocket

conda create --name cypocket-inference --channel conda-forge \
  python=3.9.9 pip=23.2.1 setuptools=68.0.0 wheel=0.41.2 -y
conda activate cypocket-inference

python -m pip install -r requirements-inference.txt
python -m pip install --no-deps \
  release/cypocket-0.1.0-py3-none-any.whl
cypocket-predict-pdb --help
```

Keep the released artifacts under:

```text
/media/yugengliu/DATA2/CYPEDock/CYPocket/P450model
```

For a relocated installation, set the model directory explicitly:

```bash
export P450_MODEL_DIR=/absolute/path/to/CYPocket/P450model
```

The complete CYPocket data-preparation, training, fine-tuning, evaluation, and
inference instructions are in the [CYPocket model manual](CYPocket/README.md).

## 7. Foldseek Installation and Library Validation

Foldseek is not redistributed with CYPEdock. Install it only if local-library
fallback will be used:

```bash
conda create --name cypedock-foldseek \
  --channel conda-forge \
  --channel bioconda \
  foldseek -y
conda run --name cypedock-foldseek foldseek version
```

The current local library is:

```text
/media/yugengliu/DATA2/CYPEDock/CYP450_foldseek_db
```

Its Foldseek database prefix is:

```text
/media/yugengliu/DATA2/CYPEDock/CYP450_foldseek_db/cyp450_holo
```

Validate the database, manifest, template checksums, and identifier mapping:

```bash
/media/yugengliu/DATA2/CYPEDock/.venv-cypedock/bin/cypedock \
  validate-library \
  --library-root /media/yugengliu/DATA2/CYPEDock/CYP450_foldseek_db \
  --database-prefix /media/yugengliu/DATA2/CYPEDock/CYP450_foldseek_db/cyp450_holo
```

An existing Foldseek executable may be selected in the configuration or with
`--foldseek-bin /absolute/path/to/foldseek`.

## 8. Input Organization

The recommended runtime layout is already present:

```text
workspace/inputs/
|-- conformations/                    Protein conformational ensemble
|-- reference_holo/                   Optional holo reference structure
`-- ligands/                          One or more ligands
workspace/outputs/                    Run output directories
```

These paths are conventions, not hard-coded requirements. Absolute paths may
be supplied for every input and output.

### 8.1 Protein Conformations

`--ensemble` accepts:

- one PDB or mmCIF file;
- a directory containing PDB or mmCIF files; or
- a text manifest containing one structure path per line.

Every model in a multi-model structure is treated as an independent
conformation. Use `--chain` when an input structure contains multiple protein
chains.

### 8.2 Holo Reference

Place an optional AF3-predicted or experimental holo structure in
`workspace/inputs/reference_holo/`. The coordinate file must contain both the
protein and a heme-like cofactor, but the cofactor residue name does not need to
be `HEM`. PDB and mmCIF formats are supported.

Automatic cofactor selection requires exactly one non-protein group containing
one Fe atom and at least four N atoms. For an ambiguous structure, select the
cofactor explicitly:

```bash
--cofactor A:501::HEM
```

The selector format is
`CHAIN:RESIDUE_NUMBER[:INSERTION_CODE[:RESIDUE_NAME]]`. Repeat `--cofactor`
when a cofactor spans multiple residues.

If no reference is supplied, CYPEdock uses the configured Foldseek library.
The same fallback is available when the supplied reference fails local-site
quality control.

### 8.3 Ligands

`--ligands` accepts the following formats:

- SMILES: `.smi`, `.smiles`, or `.txt`;
- structure data: `.sdf` or `.sd`;
- MDL molecule: `.mol`; and
- Tripos molecule: `.mol2`.

A SMILES file uses one ligand per line:

```text
CCO ethanol
CC(=O)OC1=CC=CC=C1C(=O)O aspirin
```

The optional second field is the ligand identifier. SDF and MOL2 files may
contain multiple molecules. Existing three-dimensional coordinates are used as
the starting conformation; SMILES and two-dimensional records are embedded
with RDKit ETKDGv3 and force-field minimized. Disconnected molecular records
are rejected.

## 9. Configuration

The current workstation configuration is:

```text
CYPEDock/configs/workstation.yaml
```

It contains the local Foldseek executable, Foldseek library, CYPocket
interpreter, source root, and model paths for this workstation. Inspect and
update these paths when the project is relocated.

To create a portable configuration template:

```bash
cypedock init-config --output workspace/production.yaml
```

The most important configuration groups are:

- `alignment`: local-shell radius, RMSD limits, coverage thresholds, and
  proximal-cysteine quality control;
- `foldseek`: executable, database paths, hit limits, and fallback behavior;
- `cypocket`: Python interpreters, source root, checkpoints, device, and
  classification threshold;
- `box`: fixed or automatic docking-box dimensions; and
- `docking`: Vina executable, search effort, output modes, replicates, workers,
  and timeout.

The default docking box is `25 x 25 x 25` angstrom. With `--auto-box`, each
axis expands as needed to enclose the predicted pocket while retaining the
configured cofactor margin. Each requested axis is capped independently at the
default maximum of `40` angstrom. For example, a calculated size of
`33 x 45 x 32` angstrom becomes `33 x 40 x 32` angstrom instead of causing the
protein conformation to be skipped.

Use a fixed box from the command line with:

```bash
--box-size 25 25 25
```

## 10. Preflight Validation

Before a production run, verify the effective configuration and all external
components:

```bash
cd /media/yugengliu/DATA2/CYPEDock

.venv-cypedock/bin/cypedock doctor \
  --config CYPEDock/configs/workstation.yaml \
  --require-library
```

Resolve every reported error before docking. Warnings should be reviewed in
the context of the intended protein family and structural data quality.

## 11. Running CYPEdock

### 11.1 Holo-Reference Workflow

The repository contains example inputs at the paths used below:

```bash
cd /media/yugengliu/DATA2/CYPEDock

.venv-cypedock/bin/cypedock run \
  --ensemble workspace/inputs/conformations \
  --reference-holo workspace/inputs/reference_holo/ref.pdb \
  --ligands workspace/inputs/ligands/ligands.smi \
  --config CYPEDock/configs/workstation.yaml \
  --auto-box \
  --workers 4 \
  --output workspace/outputs/run_001
```

For an SDF library, replace the ligand argument, for example:

```bash
--ligands workspace/inputs/ligands/ligands.sdf
```

### 11.2 Foldseek-Fallback Workflow

Omit `--reference-holo` and retain the enabled, validated library configuration:

```bash
.venv-cypedock/bin/cypedock run \
  --ensemble workspace/inputs/conformations \
  --ligands workspace/inputs/ligands/ligands.smi \
  --config CYPEDock/configs/workstation.yaml \
  --auto-box \
  --workers 4 \
  --output workspace/outputs/run_foldseek
```

### 11.3 Selected Runtime Overrides

```bash
# Change the accepted mean local RMSD threshold.
cypedock run ... --local-rmsd-cutoff 2.0

# Use an external Vina executable.
cypedock run ... --vina-bin /absolute/path/to/vina

# Select a protein chain in ensemble structures.
cypedock run ... --chain A

# Replace an existing run directory after safety checks.
cypedock run ... --overwrite
```

Warnings are emitted separately for each protein conformation. A conformation
that fails alignment, local-site, cofactor-transfer, or pocket quality control
is skipped while the remaining ensemble continues. The run fails when no
conformation passes quality control or when every docking task fails.

## 12. Output Files

Each run has the following principal layout:

```text
workspace/outputs/<run_id>/
|-- run_manifest.json
|-- state_results.tsv
|-- alignment_attempts.tsv
|-- docking_results.tsv
|-- states/<state_id>/
|   |-- <state_id>_protein.pdb
|   |-- <state_id>_receptor.pdb
|   |-- <state_id>_receptor.pdbqt
|   |-- alignment.json
|   |-- box.json
|   `-- cypocket/
|-- ligands/<ligand_id>/
`-- docking/
    |-- <state_id>__<ligand_id>__rNN.pdbqt
    `-- <state_id>__<ligand_id>__rNN.out
```

The files have the following interpretation:

- `run_manifest.json` records effective configuration, input hashes, software
  versions, executable hashes, checkpoint hashes, timestamps, and task counts.
- `state_results.tsv` reports the terminal status of each conformation.
- `alignment_attempts.tsv` reports reference and Foldseek alignment attempts
  with their quality-control metrics.
- `docking_results.tsv` provides task status, parsed affinities, RMSD bounds,
  pose paths, and Vina report paths.
- `box.json` records the pocket-derived center, final dimensions, and any
  maximum-size clipping.
- Each successful `.pdbqt` file contains the docked poses for one task.
- The same-stem `.out` file contains the complete Vina console report,
  including affinity in kcal/mol and RMSD bounds for every reported pose.

The `.pdbqt` and `.out` files are intentionally placed together in the
top-level `docking/` directory for direct downstream analysis.

## 13. Verification and Troubleshooting

Run the CYPEdock test suite after source modifications:

```bash
cd /media/yugengliu/DATA2/CYPEDock/CYPEDock
/media/yugengliu/DATA2/CYPEDock/.venv-cypedock/bin/python -m pip install -e '.[test]'
/media/yugengliu/DATA2/CYPEDock/.venv-cypedock/bin/python -m pytest -q tests
```

Useful diagnostic commands are:

```bash
cypedock doctor --config /absolute/path/to/config.yaml --require-library
cypedock validate-library \
  --library-root /absolute/path/to/CYP450_foldseek_db \
  --database-prefix /absolute/path/to/CYP450_foldseek_db/cyp450_holo
cypedock vina --version
```

When a run is incomplete, inspect `state_results.tsv` and
`alignment_attempts.tsv` first, then examine the corresponding state and tool
logs. For a failed Vina task, use `docking_results.tsv` to locate its retained
stdout, stderr, and task metadata.

## 14. Technical Documentation

- [CYPEdock technical README](CYPEDock/README.md)
- [CYPocket model manual](CYPocket/README.md)
- [External scientific assets](ASSETS.md)
- [CYPEdock verification record](CYPEDock/VERIFICATION.md)
- [Citation metadata](CYPEDock/CITATION.cff)

Users should report the effective configuration, CYPEdock version, CYPocket
checkpoint identity, Foldseek database identity, and Vina version in published
computational methods.


## Data and Model Availability

The data and model checkpoints supporting the article are archived on Zenodo:
[DOI: 10.5281/zenodo.23439146](https://doi.org/10.5281/zenodo.23439146).

The principal models used in the article are:

| Model | Role |
|---|---|
| **CYPocket (GVP + BERT)** | P450 pocket prediction; the deployed best/final classifiers are provided in `P450model.tar.xz`. |
| **Sequence-only BERT** | Sequence-only ablation model. |
| **Fixed GVP + MLP** | Structural-feature baseline. |
| **AlphaFlow** (`alphaflow_12l_md_templates_base_202406.pt`) | Protein conformational ensemble sampling. |
| **PocketMiner / GVP** | Pretrained structural feature extractor used by CYPocket. |
