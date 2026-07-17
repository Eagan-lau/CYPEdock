# P450 Residue Pocket Classifier - Model Card

## Model Description

The model predicts a binary ligand-pocket label for every residue in a cytochrome P450 protein. Its input is not an amino-acid token sequence. Each residue is represented by a 100-dimensional structural feature vector generated from a PDB structure by the compatible PocketMiner/GVP encoder. The classifier projects each vector from 100 to 768 dimensions, processes the sequence with a complete 12-layer BERT-base encoder, and produces two logits per residue.

The default checkpoint is `../P450model/best_protein_distilbert_classifier.pth`. The historical filename contains `distilbert`, but parameter names and tensor shapes identify a complete BERT-base encoder rather than DistilBERT. Both supplied classifier files are raw PyTorch state dictionaries. Architecture and inference conventions are fixed by `configs/model.json` and `../P450model/model_metadata.json`.

## Intended Use

The intended uses are:

- Candidate pocket-residue ranking or binary classification for structurally complete, single-chain P450 proteins near the training distribution.
- Reproducible evaluation, continued fine-tuning, and ablation studies using compatible PocketMiner features.
- Candidate-region definition for downstream molecular docking.

The output must not be interpreted directly as binding affinity, catalytic activity, or a clinical conclusion. Multi-chain complexes, severely incomplete structures, unsupported modified residues, non-P450 proteins, and structures outside the training domain require independent validation.

## Data and Label Provenance

The documented data-construction workflow filtered experimental P450 structures by heme and ligand criteria, matched structural templates to the AlphaFold P450 database with Foldseek, retained high-similarity matches, and reduced redundancy with CD-HIT. Template ligands were transferred to aligned AlphaFold structures, and AlphaFold residues within 5 angstrom of a transferred ligand were labeled as pocket residues. The final dataset contains 11,933 proteins divided into train, validation, and test splits.

These are structure-transfer proxy labels rather than independent experimental binding-site measurements for every AlphaFold sample. Directory structure, user-provided documentation, and training logs support the association between the data and supplied weights, but the raw state dictionaries do not contain a training-manifest hash. The original training code revision and random seed were not recorded.

`legacy/train_bert.log` reports the highest validation F1 of 0.7435 at epoch 373, corresponding to the default best checkpoint. The same log reports a final validation F1 of 0.7421 at epoch 400. `docs/artifacts/training_validation_curves.csv` contains loss and accuracy but not F1 and therefore is not the source of the checkpoint-selection claim.

## Input, Output, and Compatibility Behavior

- Feature shape: `[L, 100]`, `float32`, with one row per PDB residue.
- Model output: two logits per residue.
- Pocket probability: `softmax(logits)[..., 1]`.
- Default decision rule: two-class `argmax`.
- Long-protein behavior: independent, non-overlapping 512-residue windows with position indices reset in each window.
- Compatibility dropout behavior: the separately declared historical classifier dropout was not called in the historical forward pass and is not introduced in compatibility mode. Standard BERT dropout remains active during training.

Except for exact logit ties, two-class `argmax` is equivalent to a pocket-probability threshold of 0.5. At an exact tie, PyTorch `argmax` selects class 0, whereas a `p >= 0.5` threshold selects class 1. Prediction summaries therefore record the actual decision rule.

Inference outputs include per-residue probabilities, binary labels, a residue identity table, and a run summary. When a source PDB is supplied, the workflow also writes a predicted-pocket PDB. End-to-end PDB inference records hashes for the structure, features, residue mapping, PocketMiner checkpoint, and classifier checkpoint.

## Checkpoint Integrity

| Checkpoint | Role | Epoch | Validation F1 | Size in bytes | SHA-256 |
|---|---|---:|---:|---:|---|
| `best_protein_distilbert_classifier.pth` | Default inference | 373 | 0.7435 | 438,333,361 | `548135d13a62b0d9ce9fa925563f8a52b2f61f18271f599c5eec41a6e1596690` |
| `final_protein_distilbert_classifier.pth` | Final historical epoch | 400 | 0.7421 | 438,333,568 | `b48376590f532bcaaaf449a56e34fe219de1f1dc97bed31b6c30a176e7a2344d` |

Each checkpoint contains 203 tensors and 109,561,346 parameters. Loading must use strict key and shape validation. A hash mismatch must stop inference pending artifact verification.

## Environment Isolation

Separate classifier and feature-extraction environments are recommended to isolate the legacy TensorFlow stack from PyTorch and modern CUDA libraries.

Classifier and training environment:

```bash
conda create -n cypocket-classifier -c conda-forge \
  python=3.9.9 pip=23.2.1 setuptools=68.0.0 wheel=0.41.2 -y
conda activate cypocket-classifier
python -m pip install -r requirements-classifier.txt
python -m pip install -e . --no-deps
```

PocketMiner feature environment:

```bash
conda create -n cypocket-features -c conda-forge \
  python=3.9.9 pip=23.2.1 setuptools=68.0.0 wheel=0.41.2 -y
conda activate cypocket-features
python -m pip install -r requirements-features.txt
python -m pip install -e . --no-deps
```

PocketMiner extraction should normally run on CPU:

```bash
CUDA_VISIBLE_DEVICES=-1 python extract_features.py \
  --protein example.pdb \
  --output-dir features
```

Classifier inference from existing features:

```bash
python predict.py \
  --features features/example-tokens.npy \
  --protein example.pdb \
  --checkpoint ../P450model/best_protein_distilbert_classifier.pth \
  --output-dir predictions
```

De novo classifier training requires an explicit compatible local BERT-base directory through `--bert-pretrained`. Existing P450 checkpoint inference and fine-tuning do not require that directory and do not download BERT weights.

## Availability Boundary

The compatible PocketMiner/GVP TensorFlow checkpoint is supplied as `../P450model/pocketminer.index` and `../P450model/pocketminer.data-00000-of-00001`. For a real 436-residue P450 sample, the refactored extractor was element-wise identical to the legacy MDTraj/GVP implementation in the same environment. Reconstructed protein-pocket labels matched the archived labels exactly, and end-to-end classifier labels matched inference from archived features.

The supplied checkpoint must not be replaced by a randomly initialized PocketMiner model, an unrelated structural encoder, or rescaled coordinates. Shape compatibility alone does not preserve feature semantics. The historical structural-feature implementation uses coordinates in nanometers.

## Fine-Tuning and Reporting

The default configuration in `configs/train.json` uses deterministic seeding, validation PR-AUC early stopping, gradient clipping, and unweighted cross-entropy. Positive-class weighting is unset because it depends on the training distribution. If enabled, the weight must be estimated only from training data and reported with the experiment.

Reported experiments should include the sample-level split strategy, homology controls, residue counts, positive fraction, ROC-AUC, PR-AUC, MCC, F1, precision, recall, confusion matrix, threshold-selection rule, and protein-macro metrics. Threshold optimization must use validation data only. The test split must remain frozen until the model and threshold are finalized.

## Limitations

- Labels inherit template-selection bias and boundary noise from structural matching and geometric transfer.
- Residues on opposite sides of a 512-residue boundary cannot exchange contextual information.
- A static structure does not fully represent flexible pockets or induced fit.
- The model does not separate multiple pocket instances or predict ligand-specific poses or affinities.
- Historical weights lack complete training provenance that cannot be reconstructed retrospectively.

Independent validation is required before any high-risk downstream application. The model card reports historical validation F1 values from the supplied log; it does not present a new frozen-test performance experiment.

## Release and Licensing

The two P450 classifier checkpoints were supplied by the user, and the PocketMiner checkpoint was recovered from an existing local research installation. The artifacts do not contain an explicit redistribution grant. Before public code, supplementary archive, or model-hub release, the rights holders must confirm the license and citation requirements for the weights and compatibility implementation. Scientific references and known release boundaries are listed in `THIRD_PARTY_NOTICES.md`.
