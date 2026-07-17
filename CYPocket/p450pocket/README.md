# CYPocket

CYPocket is a reproducible residue-level binding-pocket predictor for cytochrome P450 proteins. It combines PocketMiner/GVP structural residue features with a BERT-base residue classifier and provides complete commands for data preparation, training, fine-tuning, evaluation, and inference.

Run all source commands from this project directory:

```bash
cd /path/to/CYPocket/p450pocket
```

The model artifacts are stored in the sibling `../P450model/` directory and are intentionally excluded from the Python wheel.

## Package Contents

```text
../P450model/               Classifier and PocketMiner model artifacts
configs/                    Model and training configurations
p450pocket/                 Importable Python package
  features/                 PocketMiner/GVP compatibility implementation
prepare_dataset.py          Protein and pocket PDBs to features, labels, and manifest
extract_features.py         Protein PDB to 100D features and residue mapping
train.py                    Training, fine-tuning, and exact new-format resume
predict.py                  Prediction from precomputed 100D features
predict_pdb.py              End-to-end PDB prediction in isolated subprocesses
evaluate.py                 Evaluation on a frozen labeled split
tests/                      Automated tests
verification/               Frozen real-sample regression fixture
legacy/                     Archived historical scripts and logs
```

Detailed model usage is documented in [`../README.md`](../README.md). Architecture and limitations are documented in [`MODEL_CARD.md`](MODEL_CARD.md), dataset provenance in [`DATASET_CARD.md`](DATASET_CARD.md), and the methods specification in [`docs/METHODS.md`](docs/METHODS.md).

## Environment

The validated environment uses Python 3.9.9, PyTorch 1.13.0, Transformers 4.45.2, and TensorFlow 2.6.2.

Combined inference environment:

```bash
conda create -n cypocket-inference -c conda-forge \
  python=3.9.9 pip=23.2.1 setuptools=68.0.0 wheel=0.41.2 -y
conda activate cypocket-inference
python -m pip install -r requirements-inference.txt
python -m pip install -e . --no-deps
```

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

A separate PocketMiner project installation is not required. The compatible implementation is included under `p450pocket/features/`, and its TensorFlow checkpoint is stored in `../P450model/`.

## End-to-End PDB Inference

Prediction requires only a protein structure. A pocket structure is required only when constructing training labels.

```bash
python predict_pdb.py \
  --protein /path/to/protein.pdb \
  --output-dir outputs/protein
```

For separate feature and classifier environments:

```bash
python predict_pdb.py \
  --protein /path/to/protein.pdb \
  --output-dir outputs/protein \
  --feature-python /path/to/cypocket-features/bin/python \
  --classifier-python /path/to/cypocket-classifier/bin/python \
  --device cpu
```

Outputs include residue probabilities, binary predictions, a residue identity table, a predicted-pocket PDB, extracted features, and JSON provenance records containing input and checkpoint hashes.

## Prediction from Existing Features

This route does not require TensorFlow or PocketMiner at runtime:

```bash
python predict.py \
  --features /path/to/features.npy \
  --protein /path/to/protein.pdb \
  --checkpoint ../P450model/best_protein_distilbert_classifier.pth \
  --output-dir outputs/feature_prediction \
  --device cpu
```

## Training-Data Preparation

Raw PDB input must use the following layout:

```text
raw_dataset/
  train/{protein,pocket}/
  val/{protein,pocket}/
  test/{protein,pocket}/
```

Matching files may be named `ABC_protein.pdb` and `ABC_pocket.pdb`.

```bash
CUDA_VISIBLE_DEVICES=-1 python prepare_dataset.py \
  --dataset-root /path/to/raw_dataset \
  --output-root /path/to/prepared_dataset \
  --splits train val test
```

The command writes 100D features, binary labels, residue maps, SHA-256 values, `manifest.jsonl`, and `dataset_summary.json`.

Existing split directories containing `tokens/` and `labels/` may be indexed without repeating feature extraction:

```bash
python prepare_dataset.py \
  --dataset-root /path/to/existing_dataset \
  --index-existing \
  --output-root /path/to/manifests \
  --manifest-out /path/to/manifests/dataset.jsonl \
  --overwrite
```

## Fine-Tuning

Use the historical best classifier as an initialization checkpoint:

```bash
python train.py \
  --manifest /path/to/prepared_dataset/manifest.jsonl \
  --init-checkpoint ../P450model/best_protein_distilbert_classifier.pth \
  --config configs/train.json \
  --output-dir runs/p450_finetune \
  --device cuda:0
```

Fine-tuning policies are:

```bash
python train.py ... --freeze-policy none
python train.py ... --freeze-policy classifier
python train.py ... --freeze-policy last_n --train-last-n-layers 2
```

The historical classifier files are raw state dictionaries and must be passed with `--init-checkpoint`. New `last.pth` checkpoints contain optimizer, scheduler, random-state, configuration, and manifest metadata and can be resumed exactly:

```bash
python train.py \
  --manifest /path/to/prepared_dataset/manifest.jsonl \
  --resume runs/p450_finetune/last.pth \
  --config configs/train.json \
  --output-dir runs/p450_finetune \
  --device cuda:0
```

## De Novo Training

De novo classifier training requires an explicit local BERT-base initialization:

```bash
python train.py \
  --manifest /path/to/prepared_dataset/manifest.jsonl \
  --bert-pretrained /path/to/bert-base-uncased \
  --config configs/train.json \
  --output-dir runs/de_novo
```

Existing P450 checkpoint inference and fine-tuning do not require a separate BERT directory.

## Evaluation

```bash
python evaluate.py \
  --manifest /path/to/prepared_dataset/manifest.jsonl \
  --split test \
  --checkpoint runs/p450_finetune/best.pth \
  --output-dir runs/p450_finetune/test_metrics \
  --device cuda:0
```

The report includes residue-pooled and protein-macro metrics. ROC-AUC and PR-AUC are computed from continuous probabilities.

## Verification

```bash
python -m pip install -e ".[test]"
python -m pytest -q tests
python verification/run_regression.py \
  --feature-python /path/to/cypocket-features/bin/python \
  --classifier-python /path/to/cypocket-classifier/bin/python
```

The frozen regression recomputes feature extraction, label preparation, and inference for a real P450 sample and compares every core output with archived expected artifacts.
