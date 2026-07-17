# Legacy provenance

This directory contains the smallest useful subset of the original experimental code and logs needed to audit the refactor. These files are historical evidence and are not production entry points.

- `model_bert.py`, `train_bert.py`, `train_bert.log`: original residue-classifier architecture and training record.
- `models.py`, `gvp.py`, `utils.py`, `get_features.py`: historical PocketMiner/GVP feature path.
- `get_labels.py`, `dataset.py`: historical label/data assumptions.
- `test_bert.py`: historical inference entry, which points to a different checkpoint and must not be used for the bundled best model.

Comments were translated to English for distribution. Computational statements and historical absolute paths remain preserved except for correction of an archival syntax artifact. Use the top-level `prepare_dataset.py`, `train.py`, `predict_pdb.py` and `evaluate.py` for all new work.
