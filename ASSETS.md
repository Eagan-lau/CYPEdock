# External Scientific Assets

The CYPEdock source repository excludes two large scientific asset groups from
Git history. Both archives are distributed through the project data deposit.

## Download

- Baidu Netdisk: <https://pan.baidu.com/s/17dh949GLClk_QC1WUDejkw?pwd=7qnp>
- Extraction code: `7qnp`

Download the following archives:

| Archive | Size in bytes | SHA-256 |
|---|---:|---|
| `CYP450_foldseek_db.tar.xz` | 146243920 | `984056598b0d45714eeec3f504629ecf6977c58c9e6f19cb1212296370d0827a` |
| `P450model.tar.xz` | 777813864 | `154f9eca6b5b564c8e1b8c54d8bb230ceaa921721896fb00d7d8da7198e631d8` |

The recorded checksums correspond to the release archives used to prepare this
repository. Verify both files before extraction:

```bash
sha256sum CYP450_foldseek_db.tar.xz P450model.tar.xz
```

## Installation

From the repository root, extract the Foldseek database directly into the
checkout:

```bash
tar -xJf /path/to/CYP450_foldseek_db.tar.xz
```

Extract the classifier checkpoints under `CYPocket/`:

```bash
tar -xJf /path/to/P450model.tar.xz -C CYPocket
```

Extraction merges the downloaded files with the placeholder directories
already present in the repository. It must produce the following layout:

```text
CYP450_foldseek_db/
|-- library_manifest.jsonl
|-- cyp450_holo.dbtype
|-- cyp450_holo.index
|-- templates/
`-- ...

CYPocket/P450model/
|-- README.md
|-- best_protein_distilbert_classifier.pth
|-- final_protein_distilbert_classifier.pth
|-- model_metadata.json
|-- pocketminer.index
`-- pocketminer.data-00000-of-00001
```

The PocketMiner checkpoint and model metadata are lightweight and are tracked
in the source repository. The external model archive supplies the two large
classifier checkpoints.

## Validation

Validate the Foldseek database after extraction:

```bash
cypedock validate-library \
  --library-root "$PWD/CYP450_foldseek_db" \
  --database-prefix "$PWD/CYP450_foldseek_db/cyp450_holo"
```

Confirm that the model artifacts are present:

```bash
test -s CYPocket/P450model/best_protein_distilbert_classifier.pth
test -s CYPocket/P450model/final_protein_distilbert_classifier.pth
test -s CYPocket/P450model/pocketminer.index
test -s CYPocket/P450model/pocketminer.data-00000-of-00001
```

The local Foldseek library is required only when reference-based alignment is
unavailable or fails quality control. The CYPocket model artifacts are required
for end-to-end pocket prediction and CYPEdock execution.
