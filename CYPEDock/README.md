# CYPEdock

CYPEdock is a reproducible workflow for docking small molecules against an
ensemble of cytochrome P450 conformations. It transfers a reference heme
position by local structural alignment, predicts pocket residues with
CYPocket, derives a docking box from the predicted pocket, generates
three-dimensional ligands with RDKit, and runs AutoDock Vina.

## Workflow

1. Read one or more protein conformations from PDB or mmCIF files.
2. Read a reference holo structure. An AlphaFold 3 holo prediction or an
   experimentally determined crystal structure is recommended.
3. Identify and validate a heme-like reference cofactor independently of its
   residue name.
4. Map the reference protein to each conformation and fit the local backbone
   surrounding the cofactor.
5. Reject a conformation when its local coverage, mean local RMSD, maximum
   per-residue local RMSD, or proximal Cys Fe-S geometry does not satisfy the
   configured criteria.
6. If the supplied reference is absent or rejected, optionally search a
   configured CYP450 holo library with Foldseek.
7. Write a protein-only structure and run CYPocket for each accepted
   conformation.
8. Calculate the docking-box center from the predicted pocket residues and use
   either a fixed 25 A box or optional automatic per-axis sizing.
9. Read ligands from SMILES, SDF, MOL, or MOL2, prepare them with RDKit, and
   dock every ligand against every accepted conformation with AutoDock Vina
   1.2.3.

The wheel contains only a versioned library-layout placeholder. The validated
workstation library is stored externally at
`/media/yugengliu/DATA2/CYPEDock/CYP450_foldseek_db`; it is not copied into the
wheel because the indexed database is large. The library is self-contained:
its holo coordinate files are stored under `templates/` and referenced with
relative paths. A supplied holo structure is not required when a compatible
external library is configured.

The public source repository distributes the generated Foldseek database and
large CYPocket classifier checkpoints through the external data deposit. See
[`../ASSETS.md`](../ASSETS.md) for the download URL, checksums, extraction
commands, and validation procedure.

## Installation

### CYPEdock

Python 3.9 or later is required. A dedicated environment is recommended.

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install .
.venv/bin/cypedock --version
```

The Linux x86-64 package contains AutoDock Vina 1.2.3. The bundled executable
has SHA-256
`04a665729f86cee39e0c18fdbc397a85bda24e81b76685ed6d3e3beec405aa0c`.
It was verified byte-for-byte against the official upstream
`vina_1.2.3_linux_x86_64` release asset.
On another platform, provide a compatible executable with `--vina-bin`.

### Foldseek

Foldseek is required only when library fallback is used. It is not
redistributed with CYPEdock.

```bash
conda create --name cypedock-foldseek \
  --channel conda-forge \
  --channel bioconda \
  foldseek
conda run --name cypedock-foldseek foldseek version
```

An existing executable may be selected with
`--foldseek-bin /absolute/path/to/foldseek`.

Two version-1 library layouts are supported. The portable layout uses
`manifest.jsonl`, relative coordinate paths, and the default Foldseek prefix
`foldseek/cyp450`. The generated workstation layout uses
`library_manifest.jsonl`, `cofactor_groups[].residues`, the prefix
`cyp450_holo`, and relative coordinate paths under `templates/`. Every
coordinate file is verified against its manifest SHA-256 digest. The optional
`structure_root` setting is retained only for compatibility with legacy
manifests whose coordinate files are stored outside the library root.

Validate the workstation library before use:

```bash
cypedock validate-library \
  --library-root /media/yugengliu/DATA2/CYPEDock/CYP450_foldseek_db \
  --database-prefix /media/yugengliu/DATA2/CYPEDock/CYP450_foldseek_db/cyp450_holo
```

Set `foldseek.library_root` in the configuration file or pass `--library-root`
to `cypedock run`. The corresponding database option is `--database-prefix`.
CYPEdock verifies template hashes, manifest uniqueness, and the database
prefix before searching; retrieved hit identifiers must resolve to manifest
entries. Internal symbolic links and manifest coordinate paths are relative,
so the complete library directory can be relocated as one unit.

### CYPocket

CYPocket should be installed in an independent Python 3.9 environment because
its validated TensorFlow and PyTorch dependencies are version constrained.

```bash
conda create --name cypocket-inference --channel conda-forge python=3.9
conda activate cypocket-inference
python -m pip install \
  -r /absolute/path/to/CYPocket/p450pocket/requirements-inference.txt
python -m pip install --no-deps \
  /absolute/path/to/CYPocket/p450pocket/release/cypocket-0.1.0-py3-none-any.whl
cypocket-predict-pdb --help
```

The PocketMiner and classifier checkpoints must be retained under
`CYPocket/P450model`, or their paths must be set in the configuration file.
CYPEdock invokes `predict_pdb.py` through the dedicated CYPocket interpreter
and records the checkpoint hashes in the run manifest.

## Workspace layout

The source package uses a sibling runtime workspace at
`/media/yugengliu/DATA2/CYPEDock/workspace`. These locations are recommended
for clarity but are not hard-coded; every CLI path may point to a different
absolute location.

```text
../workspace/
  inputs/
    conformations/   Protein ensemble in PDB or mmCIF format
    reference_holo/  AF3 or experimental holo reference
    ligands/         SMILES, SDF, MOL, or MOL2 input files
  outputs/           One subdirectory per CYPEdock run
```

For an AlphaFold 3 reference, place the single holo coordinate file containing
the protein and its modeled heme-like cofactor in `reference_holo/`. PDB and
mmCIF are supported, and the cofactor residue name is not required to be
`HEM`. Auxiliary AF3 confidence and metadata files are not CYPEdock inputs.

An example using the workstation configuration is:

```bash
cypedock run \
  --ensemble ../workspace/inputs/conformations \
  --reference-holo ../workspace/inputs/reference_holo/af3_holo.cif \
  --ligands ../workspace/inputs/ligands/ligands.smi \
  --config configs/workstation.yaml \
  --output ../workspace/outputs/run_001
```

If no AF3 or experimental holo reference is available, omit
`--reference-holo`; the configured CYP450 Foldseek library will be used.

## Input

### Conformational ensemble

`--ensemble` accepts:

- one PDB or mmCIF file;
- a directory containing PDB or mmCIF files;
- a text manifest containing one structure path per line.

Every model in a multi-model structure is treated as an independent
conformation. A conformation must contain one selected standard-amino-acid
chain. Use `--chain` when a file contains multiple protein chains.

### Reference holo structure

The reference may contain any cofactor residue name. Automatic selection
requires exactly one non-protein residue containing one Fe atom and at least
four N atoms. An explicitly selected, potentially multiresidue cofactor is
subject to the same heme-like composition criterion: its unique selected
residues must collectively contain exactly one Fe atom and at least four N
atoms. Ambiguous structures require an explicit selector:

```bash
--cofactor A:501::HEM
```

The format is `CHAIN:NUMBER[:INSERTION_CODE[:RESIDUE_NAME]]`. Repeat the
option when the cofactor spans multiple residues.

### Ligands

`--ligands` accepts `.smi`, `.smiles`, `.txt`, `.sdf`, `.sd`, `.mol`, and
`.mol2` files. A SMILES file contains one ligand per line:

```text
CCO ethanol
CC(=O)OC1=CC=CC=C1C(=O)O aspirin
```

The optional second field is the ligand identifier. SDF and MOL2 may contain
multiple molecule records; MOL contains one molecule. Existing
three-dimensional coordinates are retained as the starting conformer, whereas
SMILES and two-dimensional records are embedded with ETKDGv3. All ligands are
subsequently force-field minimized. Disconnected molecules are rejected
because salt and component selection require an explicit chemical policy.

For `N` accepted protein conformations, `M` ligands, and `R` configured
replicates, CYPEdock creates `N x M x R` independent Vina tasks. The legacy
`--smiles-file` option remains available as an alias for SMILES input.

## Configuration

Copy `configs/default.yaml`, or create it from an installed wheel, and set
the CYPocket installation:

```bash
cypedock init-config --output production.yaml
```

```yaml
cypocket:
  python: /absolute/envs/cypocket/bin/python
  root: /absolute/path/to/CYPocket/p450pocket
  feature_python: /absolute/envs/cypocket/bin/python
  classifier_python: /absolute/envs/cypocket/bin/python
  device: cpu

foldseek:
  executable: /absolute/path/to/foldseek
  enabled: true
  library_root: /absolute/path/to/CYP450_foldseek_db
  database_prefix: /absolute/path/to/CYP450_foldseek_db/cyp450_holo
  structure_root: null

alignment:
  local_shell_radius_angstrom: 10.0
  local_rmsd_cutoff_angstrom: 2.5
  maximum_local_residue_rmsd_angstrom: 4.0
  minimum_local_residues: 8
  minimum_local_coverage: 0.60
  require_proximal_cysteine: true
  minimum_fe_s_distance_angstrom: 1.80
  maximum_fe_s_distance_angstrom: 3.00

box:
  auto_size: false
  size_angstrom: [25.0, 25.0, 25.0]
  maximum_size_angstrom: [40.0, 40.0, 40.0]
  pocket_padding_angstrom: 4.0
  minimum_fe_margin_angstrom: 2.0
```

The default 2.5 A mean local RMSD, 4.0 A maximum per-residue local RMSD, and
1.80-3.00 A Fe-S distance limits are operational starting points. They should
be calibrated for the intended CYP family, reference quality, and
conformational sampling protocol. By default, `size_angstrom` is the fixed box
size. With `auto_size: true` or `--auto-box`, it becomes the per-axis minimum;
the box expands to enclose the predicted pocket and retain the configured Fe
margin. Any requested axis longer than `maximum_size_angstrom` is clipped to
that maximum and recorded in `box.json`.

## Execution

```bash
cypedock doctor --config production.yaml

cypedock run \
  --ensemble /absolute/path/to/conformations \
  --reference-holo /absolute/path/to/reference_holo.cif \
  --ligands /absolute/path/to/ligands.sdf \
  --config production.yaml \
  --output /absolute/path/to/run
```

To use library fallback without a supplied holo structure, omit
`--reference-holo` and retain `foldseek.enabled: true` with a validated library
configuration.

The local threshold can be changed without editing the configuration:

```bash
cypedock run ... --local-rmsd-cutoff 2.0
```

Use a fixed non-default box or enable automatic sizing from the command line:

```bash
cypedock run ... --box-size 25 25 25
cypedock run ... --auto-box
```

Warnings are emitted per conformation. A failed conformation is skipped, while
the remaining ensemble continues. The run fails only when no conformation
passes structural and pocket quality control or when all docking tasks fail.
`--overwrite` is refused when the output directory would contain an input,
the active package, the configuration file, or a configured tool/model path.

## Structural method

Reference and target sequences are aligned globally with BLOSUM62 and affine
gap penalties. The local reference shell contains every protein residue with a
heavy atom within 10 A of a selected cofactor heavy atom. Matched residues with
complete `N`, `CA`, and `C` atoms define a proper Kabsch transform from
the reference into the target coordinate frame. The cofactor is transferred
with this local transform; target protein coordinates are not modified. In
addition to the aggregate local RMSD, CYPEdock computes the backbone RMSD of
each mapped shell residue after fitting and rejects a state when the maximum
value exceeds its configured limit. When proximal-cysteine validation is
enabled, the reference Cys SG atom nearest to the heme Fe defines the putative
axial ligand. The corresponding target residue must retain a Cys SG atom, and
both the reference Fe-S distance and the distance between the transferred Fe
and target SG must lie within the configured interval.

CYPocket always receives a newly written protein-only PDB. The docking receptor
contains the target protein and the transferred cofactor. By default, the box
center is the arithmetic mean of predicted-pocket CA coordinates and the box
size is fixed at 25 A per axis. The transferred Fe must retain the configured
minimum distance from every box face. In automatic mode, each dimension starts
at 25 A and expands as needed to enclose every selected pocket coordinate with
the configured padding and to retain the required Fe-to-face margin. Any
automatically requested dimension above the configured maximum is clipped on
that axis. CYPEdock records the unclipped dimensions and clipped axes and emits
a warning while continuing the run. A state is still rejected if clipping
prevents the transferred Fe from retaining its required box-face margin.

Ligands without three-dimensional coordinates are embedded with ETKDGv3.
Three-dimensional SDF, MOL, and MOL2 coordinates are used as the starting
conformer. Every ligand is minimized with MMFF94s when parameters are
available, otherwise with UFF. RDKit Gasteiger charges and AutoDock atom types
are written directly to PDBQT. Rotatable bonds are represented as a Vina
torsion tree. Receptor atoms use deterministic element- and residue-aware
AutoDock typing. Receptor partial charges are written as zero because the
`vina` scoring function is used; this policy must not be substituted for an
AutoDock4 electrostatic parameterization.

### Scientific limitations of PDBQT preparation

The internal PDBQT preparation pathway is a deterministic research-preview
implementation. In particular, receptor preparation retains only hydrogens
already present in the structural input; it neither adds missing hydrogens nor
assigns system-specific protonation or tautomeric states. Heme/cofactor atom
typing, axial coordination, receptor protonation, and the treatment of
structural waters therefore require application-specific assessment. Docking
scores and poses produced by this pathway must not be interpreted as
publication-grade evidence without an independently justified receptor
preparation protocol and retrospective pose-fidelity validation, including
redocking against experimentally resolved CYP-ligand complexes representative
of the intended chemical and protein domain.

## Output

```text
run_manifest.json
state_results.tsv
alignment_attempts.tsv
docking_results.tsv
states/<state_id>/
  <state_id>_protein.pdb
  <state_id>_receptor.pdb
  <state_id>_receptor.pdbqt
  alignment.json
  box.json
  cypocket/
ligands/<ligand_id>/
docking/
  <state_id>__<ligand_id>__rNN.pdbqt
  <state_id>__<ligand_id>__rNN.out
```

`run_manifest.json` records software versions, executable and checkpoint
hashes, structure and ligand-input hashes, effective configuration, timestamps,
and task counts. When Foldseek fallback is used, it also records the Foldseek
binary, library manifest, and database-component hashes. Per-tool stdout,
stderr, and command arguments are retained under the corresponding log
directories.

Each successful docking task writes a `.pdbqt` pose file and a same-stem `.out`
file directly under `docking/`. The `.out` file preserves the complete Vina
console report, including the affinity in kcal/mol and RMSD bounds for every
reported pose. `docking_results.tsv` records both output paths and provides the
same parsed scores in tabular form.

## References

1. van Kempen M, et al. Fast and accurate protein structure search with
   Foldseek. *Nature Biotechnology*. 2024;42:243-246.
   [doi:10.1038/s41587-023-01773-0](https://doi.org/10.1038/s41587-023-01773-0).
2. Eberhardt J, et al. AutoDock Vina 1.2.0: New docking methods, expanded force
   field, and Python bindings. *Journal of Chemical Information and Modeling*.
   2021;61:3891-3898.
   [doi:10.1021/acs.jcim.1c00203](https://doi.org/10.1021/acs.jcim.1c00203).
3. Trott O, Olson AJ. AutoDock Vina: improving the speed and accuracy of
   docking. *Journal of Computational Chemistry*. 2010;31:455-461.
   [doi:10.1002/jcc.21334](https://doi.org/10.1002/jcc.21334).
4. Abramson J, et al. Accurate structure prediction of biomolecular
   interactions with AlphaFold 3. *Nature*. 2024;630:493-500.
   [doi:10.1038/s41586-024-07487-w](https://doi.org/10.1038/s41586-024-07487-w).
