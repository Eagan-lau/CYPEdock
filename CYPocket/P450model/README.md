# CYPocket Model Assets

The two large classifier checkpoints are intentionally excluded from Git and
are distributed through the project data deposit. The lightweight PocketMiner
checkpoint and model metadata remain versioned in this directory.

Download `P450model.tar.xz` from:

- Preferred source: Zenodo [DOI: 10.5281/zenodo.23439146](https://doi.org/10.5281/zenodo.23439146).
- [Download P450model.tar.xz](https://zenodo.org/api/records/23439146/files/P450model.tar.xz/content)
- Backup mirror: [Baidu Netdisk](https://pan.baidu.com/s/17dh949GLClk_QC1WUDejkw?pwd=7qnp)
  (extraction code: `7qnp`).
- SHA-256: `154f9eca6b5b564c8e1b8c54d8bb230ceaa921721896fb00d7d8da7198e631d8`

From the repository root, install the classifier checkpoints with:

```bash
tar -xJf /path/to/P450model.tar.xz -C CYPocket
```

The completed directory must contain:

```text
best_protein_distilbert_classifier.pth
final_protein_distilbert_classifier.pth
model_metadata.json
pocketminer.index
pocketminer.data-00000-of-00001
```

Complete asset installation and validation instructions are provided in
[`../../ASSETS.md`](../../ASSETS.md).
