# CYPocket package contents

## Runtime

- `p450pocket/` and the six top-level CLI scripts implement feature extraction, data preparation, training/fine-tuning, prediction and evaluation.
- `configs/`, `environment.yml` and `requirements*.txt` define the model and reproducible environments.
- `../P450model/best_protein_distilbert_classifier.pth` plus the two `pocketminer.*` shards are required for direct PDB inference.

## Archival and publication support

- `../P450model/final_protein_distilbert_classifier.pth` is archival rather than a runtime requirement; `docs/artifacts/training_validation_curves.*` records the historical training curves.
- `MODEL_CARD.md`, `DATASET_CARD.md`, `docs/METHODS.md`, `VERIFICATION.md` and `THIRD_PARTY_NOTICES.md` document provenance and limitations.
- `verification/regression_v1/` is a self-contained input/expected-output regression fixture.
- `legacy/` preserves the minimum historical evidence referenced by the model card.
- `release/cypocket-0.1.0-py3-none-any.whl` is the tested code-only wheel; model artifacts intentionally remain outside the wheel.

The large model artifacts are intentionally external to both the Python wheel and the `p450pocket/` source project. They are stored in the sibling `P450model/` directory. Before public redistribution, confirm the unresolved software, checkpoint and fixture permissions described in `THIRD_PARTY_NOTICES.md`.
