# Reproducibility verification

Audit date: 2026-07-15. The machine-readable values are frozen in [`verification/verification_report.json`](verification/verification_report.json).

After integration into the standalone `CYPocket/p450pocket/` project root with model artifacts in the sibling `CYPocket/P450model/` directory, both the source-package regression and an installed `cypocket-0.1.0` wheel completed end-to-end inference with feature and prediction hashes identical to the frozen fixture.

## What was actually run

- Strictly loaded both supplied classifier state dictionaries and reconstructed the full 12-layer BERT-base architecture without downloading BERT weights.
- Restored the bundled TensorFlow PocketMiner checkpoint with model-variable matching and generated `[436,100]` features from a real test PDB.
- Ran `protein+pocket → features+labels+manifest` on the same real sample.
- Ran `protein → features → classifier → probability/label/CSV/pocket PDB` through `predict_pdb.py` from both the source checkout and an installed wheel.
- Ran the automated suite, including 511/512/513-residue chunk boundaries, strict safe checkpoint loading, a tiny synthetic one-epoch training smoke test, PDB edge cases, labels, hashes, metrics and overwrite protection.

No production model was retrained, no real-data hyperparameter was changed, and the historical `test/pred` directory was not used as a trusted result cache.

## Real-sample regression

Sample `AF-A0A010RW94-F1-model_v4` contains 436 residues. Its protein PDB SHA-256 is `15e2dacf07ea975f91ecd94727a0ad5b2a0456285ec0c13f924d933ad27e5f10`.

| Check | Result |
|---|---:|
| Refactored vs same-environment legacy PocketMiner features | bitwise equal |
| Feature SHA-256 from both implementations | `8a7818963fc6877ad53b5ca0b06e5fb8b6456d6cbe698d27bff4254b7e770f62` |
| Raw-prepare vs standalone-extractor features | bitwise equal |
| Rebuilt vs archived labels | bitwise equal; 36 positives |
| End-to-end vs archived-feature classifier labels | identical; 71 predicted positives |
| End-to-end probability max / mean absolute difference | `1.8626451e-05` / `1.0947346e-06` |
| PyTorch 1.13 vs 2.1 classifier labels | identical |
| PyTorch 1.13 vs 2.1 probability max / mean absolute difference | `5.9604645e-07` / `4.0212619e-08` |

The archived token file was produced in an earlier numerical environment. Recomputing the same legacy feature code now differs from that archived array by at most `5.7792664e-04` (mean `8.0603031e-06`, RMSE `1.8992254e-05`), but does not change any final label on this regression sample. The stronger bitwise claim applies to the refactored and legacy implementations evaluated from the same PDB in the same audited environment.

## Portable regression command

The package contains the real input PDB, pocket PDB and frozen expected artifacts. From the `CYPocket/p450pocket/` project root, one command recomputes both inference and data preparation without the original dataset mount:

```bash
python verification/run_regression.py
```

With separate TensorFlow and PyTorch environments:

```bash
python verification/run_regression.py \
  --feature-python /path/to/feature-env/bin/python \
  --classifier-python /path/to/classifier-env/bin/python
```

Automated tests:

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  python -m pytest -p no:cacheprovider -q tests
```

The fixture inventory is [`verification/regression_v1/SHA256SUMS`](verification/regression_v1/SHA256SUMS), and the executable comparison logic is [`verification/run_regression.py`](verification/run_regression.py). Pass `--output-dir verification/latest_run` only when generated runtime summaries need to be retained; that directory is intentionally not part of the frozen fixture.

## Publication boundary

The full dataset audit found 11,933 proteins, 5,252,969 residues and 528,989 positive labels with no file-ID, dtype, shape, label-domain or feature/label-length anomaly. Exact split statistics are in `DATASET_CARD.md`.

This validates the software and supplied artifacts; it is not a new model-performance experiment. A paper should still report a frozen test-set evaluation, confidence intervals and subgroup analyses, and must establish the dataset/checkpoint redistribution permissions described in `THIRD_PARTY_NOTICES.md`.
