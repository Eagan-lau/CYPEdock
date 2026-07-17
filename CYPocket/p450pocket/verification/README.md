# CYPocket regression fixture

`regression_v1` is a self-contained real P450 fixture. Its input PDB and pocket PDB are regular files, not links to the original dataset. The expected directory stores the audited PocketMiner feature matrix, reconstructed labels and classifier outputs.

From the CYPocket project root, run:

```bash
python verification/run_regression.py
```

For separate environments:

```bash
python verification/run_regression.py \
  --feature-python /path/to/feature-env/bin/python \
  --classifier-python /path/to/classifier-env/bin/python
```

The script verifies the bundled model hashes, recomputes `protein → features → prediction`, recomputes `protein+pocket → features+labels`, and compares all core outputs. By default temporary artifacts are deleted; pass `--output-dir verification/latest_run` to retain them.

Feature probabilities may differ in the last floating-point bits across TensorFlow/PyTorch builds, so the script requires exact labels and residue identities while applying explicitly recorded numeric tolerances to features and probabilities. `SHA256SUMS` identifies the frozen input and expected-output files.

The real structure/checkpoint redistribution permission must be confirmed before uploading this fixture or the model files publicly; see `../THIRD_PARTY_NOTICES.md`.
