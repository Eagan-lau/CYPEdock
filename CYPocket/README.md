# CYPocket Model Package

CYPocket is a residue-level binding-pocket predictor for cytochrome P450 proteins. The workflow converts a protein structure into 100-dimensional residue embeddings with the bundled PocketMiner/GVP implementation and then classifies each residue with a BERT-base encoder.

This directory contains the complete data-preparation, training, fine-tuning, evaluation, and inference workflow. Existing model artifacts are stored separately from the source code.

## 1. Directory Layout

```text
CYPocket/
|-- README.md
|-- P450model/                         Model artifacts and metadata only
|   |-- best_protein_distilbert_classifier.pth
|   |-- final_protein_distilbert_classifier.pth
|   |-- model_metadata.json
|   |-- pocketminer.index
|   `-- pocketminer.data-00000-of-00001
`-- p450pocket/                        Complete source project
    |-- p450pocket/                    Importable Python package
    |-- configs/                       Model and training configurations
    |-- docs/                          Methods and archived training curves
    |-- tests/                         Automated tests
    |-- verification/                  Frozen real-sample regression fixture
    |-- prepare_dataset.py             Protein and pocket PDBs to a manifest
    |-- extract_features.py            Protein PDB to 100D residue features
    |-- train.py                       Training, fine-tuning, and resume
    |-- predict.py                     Prediction from precomputed features
    |-- predict_pdb.py                 End-to-end prediction from a PDB file
    `-- evaluate.py                    Evaluation on a labeled split
```

The naming conventions are:

- Source project directory: `p450pocket/`
- Python distribution name: `cypocket`
- Python import name: `p450pocket`
- Command-line prefix: `cypocket-`

`P450model/` contains only model artifacts. Model weights are intentionally excluded from the Python wheel.
The large classifier checkpoints are distributed through the external project
data deposit; download, checksum, and extraction instructions are provided in
[`../ASSETS.md`](../ASSETS.md).

## 2. Model Overview

| Property | Value |
|---|---|
| Structural input | One protein chain in PDB format |
| Residue representation | 100-dimensional PocketMiner/GVP feature vector |
| Classifier | 12-layer BERT-base encoder |
| Hidden dimension | 768 |
| Output | Two logits per residue: non-pocket and pocket |
| Maximum window length | 512 residues |
| Long-protein behavior | Independent, non-overlapping 512-residue windows |
| Default decision rule | Two-class `argmax` |
| Default classifier | `P450model/best_protein_distilbert_classifier.pth` |

The historical checkpoint filenames contain `distilbert`, but checkpoint-key and tensor-shape audits identify a full BERT-base encoder rather than DistilBERT.

## 3. Software Environment

### 3.1 Requirements

- Python 3.9; Python 3.9.9 is recommended.
- PyTorch 1.13.0 and Transformers 4.45.2 for classification and training.
- TensorFlow 2.6.2, Keras 2.6.0, and MDTraj 1.9.7 for PocketMiner feature extraction.
- CPU execution is recommended for PocketMiner. The classifier may run on CPU or CUDA.

On the current workstation, the unqualified `python` command may resolve to the Python 2.7 interpreter bundled with ADFRsuite. Activate the intended Conda environment or use an explicit Python 3.9 executable.

### 3.2 Single-Environment Installation

A combined environment is the simplest option for end-to-end inference:

```bash
cd /media/yugengliu/DATA2/CYPEDock/CYPocket/p450pocket

conda create -n cypocket-inference -c conda-forge \
  python=3.9.9 pip=23.2.1 setuptools=68.0.0 wheel=0.41.2 -y
conda activate cypocket-inference

python -m pip install -r requirements-inference.txt
python -m pip install -e . --no-deps
```

`predict_pdb.py` still launches feature extraction and classification in separate processes, preventing TensorFlow from retaining classifier-process GPU memory.

### 3.3 Isolated Feature and Classifier Environments

Separate environments provide stricter dependency and device isolation:

```bash
cd /media/yugengliu/DATA2/CYPEDock/CYPocket/p450pocket

conda create -n cypocket-features -c conda-forge \
  python=3.9.9 pip=23.2.1 setuptools=68.0.0 wheel=0.41.2 -y
conda activate cypocket-features
python -m pip install -r requirements-features.txt
python -m pip install -e . --no-deps

conda create -n cypocket-classifier -c conda-forge \
  python=3.9.9 pip=23.2.1 setuptools=68.0.0 wheel=0.41.2 -y
conda activate cypocket-classifier
python -m pip install -r requirements-classifier.txt
python -m pip install -e . --no-deps
```

The prebuilt code-only wheel may be installed instead of the editable source package:

```bash
python -m pip install --no-deps \
  /media/yugengliu/DATA2/CYPEDock/CYPocket/p450pocket/release/cypocket-0.1.0-py3-none-any.whl
```

## 4. PocketMiner Requirement

A separate PocketMiner project installation is not required. This package already includes:

- A compatible implementation under `p450pocket/p450pocket/features/`.
- The TensorFlow checkpoint `P450model/pocketminer.index`.
- The matching parameter shard `P450model/pocketminer.data-00000-of-00001`.

End-to-end prediction from a PDB file requires the PocketMiner feature stage and therefore requires the dependencies in `requirements-features.txt` or `requirements-inference.txt`. Prediction or fine-tuning from existing `[L, 100]` feature arrays does not require TensorFlow or the PocketMiner runtime.

The PocketMiner command-line argument is a checkpoint prefix. Pass `/path/to/P450model/pocketminer`, not the `.index` file.

## 5. Model Artifacts

| File | Purpose |
|---|---|
| `best_protein_distilbert_classifier.pth` | Default inference model and recommended fine-tuning initialization |
| `final_protein_distilbert_classifier.pth` | Classifier state at the final historical training epoch |
| `pocketminer.index` | TensorFlow checkpoint index for structural feature extraction |
| `pocketminer.data-00000-of-00001` | TensorFlow checkpoint parameter shard |
| `model_metadata.json` | Architecture, hashes, environment, and historical metrics |

Source execution automatically discovers the sibling `P450model/` directory. After wheel installation, set the model directory explicitly:

```bash
export P450_MODEL_DIR=/media/yugengliu/DATA2/CYPEDock/CYPocket/P450model
```

Individual artifacts may also be selected with:

```bash
export P450_POCKETMINER_CHECKPOINT=/path/to/P450model/pocketminer
export P450_CLASSIFIER_CHECKPOINT=/path/to/P450model/best_protein_distilbert_classifier.pth
```

## 6. End-to-End Pocket Prediction from a PDB File

### 6.1 Input Requirements

The recommended input is a protein-only PDB file with the following properties:

- One selected cytochrome P450 protein chain.
- Backbone atoms `N`, `CA`, `C`, and `O` for every retained residue.
- No requirement for heme or another ligand.
- An explicit chain selection for multi-chain structures.
- An explicit canonical mapping for supported modified residues.
- An explicit model selection for a multi-model PDB file.

For multi-chain structures, use `--chains A`. For modified residues, repeat `--modified-residue ORIGINAL=CANONICAL` as required. Multi-model input is rejected unless `--allow-multiple-models` and `--model-index` are supplied.

### 6.2 Minimal Command

```bash
cd /media/yugengliu/DATA2/CYPEDock/CYPocket/p450pocket

python predict_pdb.py \
  --protein /path/to/protein.pdb \
  --output-dir /path/to/output
```

Select one chain when necessary:

```bash
python predict_pdb.py \
  --protein /path/to/protein.pdb \
  --chains A \
  --output-dir /path/to/output
```

Use isolated feature and classifier environments as follows:

```bash
/path/to/launcher/bin/python predict_pdb.py \
  --protein /path/to/protein.pdb \
  --output-dir /path/to/output \
  --feature-python /path/to/cypocket-features/bin/python \
  --classifier-python /path/to/cypocket-classifier/bin/python \
  --device cuda:0
```

An explicit probability threshold may replace the default decision rule:

```bash
python predict_pdb.py \
  --protein /path/to/protein.pdb \
  --output-dir /path/to/output \
  --threshold 0.40
```

Threshold selection for reported experiments must use validation data rather than the test set.

### 6.3 Prediction Outputs

For a sample named `protein`, the output directory contains:

| Output | Description |
|---|---|
| `protein_probabilities.npy` | Continuous pocket probability for every residue |
| `protein_predictions.npy` | Binary residue predictions |
| `protein_residues.csv` | Residue identity, numbering, probability, and prediction |
| `protein_predicted_pocket.pdb` | Predicted pocket residues extracted from the input structure |
| `protein_summary.json` | Inputs, settings, hashes, decision rule, and output paths |
| `features/*-tokens.npy` | Generated 100-dimensional residue features |
| `features/*-residues.tsv` | Mapping between feature rows and PDB residues |
| `features/*-features.json` | Feature checkpoint and software provenance |

## 7. Prediction from Precomputed Features

TensorFlow feature extraction can be skipped when a compatible array with shape `[L, 100]` is already available:

```bash
cd /media/yugengliu/DATA2/CYPEDock/CYPocket/p450pocket

python predict.py \
  --features /path/to/sample-tokens.npy \
  --protein /path/to/sample_protein.pdb \
  --checkpoint ../P450model/best_protein_distilbert_classifier.pth \
  --output-dir /path/to/output \
  --device cpu
```

Providing `--protein` enables a strict feature-length check, original residue numbering, and predicted-pocket PDB output.

## 8. Training-Data Preparation

### 8.1 Protein and Pocket PDB Input

Raw data must be organized by split:

```text
raw_dataset/
|-- train/
|   |-- protein/
|   `-- pocket/
|-- val/
|   |-- protein/
|   `-- pocket/
`-- test/
    |-- protein/
    `-- pocket/
```

Protein and pocket filenames must resolve to the same sample identifier. For example:

```text
train/protein/ABC_protein.pdb
train/pocket/ABC_pocket.pdb
```

Generate features, labels, residue maps, and an auditable manifest:

```bash
cd /media/yugengliu/DATA2/CYPEDock/CYPocket/p450pocket

CUDA_VISIBLE_DEVICES=-1 python prepare_dataset.py \
  --dataset-root /path/to/raw_dataset \
  --output-root /path/to/prepared_dataset \
  --splits train val test
```

The default mapping uses exact residue identity and checks that corresponding protein and pocket CA coordinates differ by no more than 0.25 angstrom. Missing pairs, duplicate identifiers, split leakage, invalid labels, and feature-label length mismatches are rejected.

The prepared layout is:

```text
prepared_dataset/
|-- manifest.jsonl
|-- dataset_summary.json
|-- train/{tokens,labels,residue_maps}/
|-- val/{tokens,labels,residue_maps}/
`-- test/{tokens,labels,residue_maps}/
```

### 8.2 Existing Feature and Label Arrays

Existing split directories containing `tokens/` and `labels/` may be validated and indexed without repeating feature extraction:

```bash
python prepare_dataset.py \
  --dataset-root /path/to/existing_dataset \
  --index-existing \
  --output-root /path/to/manifests \
  --manifest-out /path/to/manifests/dataset.jsonl \
  --overwrite
```

## 9. Fine-Tuning the Existing Classifier

Initialize a new fine-tuning run from the historical best checkpoint:

```bash
cd /media/yugengliu/DATA2/CYPEDock/CYPocket/p450pocket

python train.py \
  --manifest /path/to/prepared_dataset/manifest.jsonl \
  --init-checkpoint ../P450model/best_protein_distilbert_classifier.pth \
  --config configs/train.json \
  --output-dir /path/to/runs/p450_finetune \
  --device cuda:0
```

Available fine-tuning policies are:

```bash
# Update all model parameters.
python train.py ... --freeze-policy none

# Update only the 100-to-768 projection and the final classifier.
python train.py ... --freeze-policy classifier

# Update the projection, classifier, and the final two BERT layers.
python train.py ... --freeze-policy last_n --train-last-n-layers 2
```

Common overrides include:

```text
--epochs
--batch-size
--gradient-accumulation-steps
--learning-rate
--positive-class-weight
--monitor {loss,pr_auc,roc_auc,f1,mcc}
--patience
--seed
--mixed-precision
```

The current default configuration is designed for reproducible fine-tuning. It uses AdamW, linear warm-up and decay, gradient clipping, deterministic seeds, and validation PR-AUC model selection. The model architecture remains compatible with the historical checkpoints, but the default optimization procedure is not a line-by-line reproduction of the historical 400-epoch script.

### 9.1 Initialization and Resume Semantics

- The historical `best_*.pth` and `final_*.pth` files are raw model state dictionaries. Use them with `--init-checkpoint`.
- New `best.pth` and `last.pth` files include the model, optimizer, scheduler, random states, training configuration, and manifest hash.
- Only a structured checkpoint produced by the new workflow can be used with `--resume` for an exact continuation.

Resume an interrupted new-format run with:

```bash
python train.py \
  --manifest /path/to/prepared_dataset/manifest.jsonl \
  --resume /path/to/runs/p450_finetune/last.pth \
  --config configs/train.json \
  --output-dir /path/to/runs/p450_finetune \
  --device cuda:0
```

Changing training hyperparameters requires a new run initialized with `--init-checkpoint`; it is not an exact resume.

### 9.2 Training Outputs

| Output | Description |
|---|---|
| `run_config.json` | Complete settings, environment, data, and initialization source |
| `history.csv` | Epoch-level training and validation metrics |
| `validation_metrics.json` | Pooled and per-protein validation metrics |
| `best.pth` | Checkpoint with the best monitored validation metric |
| `last.pth` | Complete checkpoint from the latest epoch |
| `result.json` | Best epoch, best metric, and checkpoint paths |

## 10. De Novo Classifier Training

De novo classifier training requires an explicit local BERT-base initialization:

```bash
python train.py \
  --manifest /path/to/prepared_dataset/manifest.jsonl \
  --bert-pretrained /path/to/bert-base-uncased \
  --config configs/train.json \
  --output-dir /path/to/runs/de_novo \
  --device cuda:0
```

Inference and fine-tuning from an existing P450 checkpoint do not require a separate BERT-base directory. The training command does not silently download BERT or silently initialize a random encoder.

## 11. Evaluation

Evaluate a frozen labeled split with:

```bash
cd /media/yugengliu/DATA2/CYPEDock/CYPocket/p450pocket

python evaluate.py \
  --manifest /path/to/prepared_dataset/manifest.jsonl \
  --split test \
  --checkpoint /path/to/runs/p450_finetune/best.pth \
  --output-dir /path/to/runs/p450_finetune/test_metrics \
  --device cuda:0
```

The report includes residue-pooled accuracy, balanced accuracy, precision, recall, F1, Matthews correlation coefficient, ROC-AUC, PR-AUC, and a confusion matrix. It also includes per-protein metrics and protein-macro summaries. ROC-AUC and PR-AUC are calculated from continuous probabilities.

## 12. CYPEdock Integration and Verification

CYPEdock invokes CYPocket with the following configuration structure:

```yaml
cypocket:
  python: /path/to/cypocket-env/bin/python
  root: /media/yugengliu/DATA2/CYPEDock/CYPocket/p450pocket
  feature_python: /path/to/cypocket-features/bin/python
  classifier_python: /path/to/cypocket-classifier/bin/python
  pocketminer_checkpoint: /media/yugengliu/DATA2/CYPEDock/CYPocket/P450model/pocketminer
  classifier_checkpoint: /media/yugengliu/DATA2/CYPEDock/CYPocket/P450model/best_protein_distilbert_classifier.pth
  device: cpu
```

The workstation configuration is located at:

```text
/media/yugengliu/DATA2/CYPEDock/CYPEDock/configs/workstation.yaml
```

CYPEdock provides CYPocket with a protein-only PDB file. Heme and other hetero residues are excluded from the model input because CYPocket predicts protein pocket residues only.

Check the source command entry points with:

```bash
cd /media/yugengliu/DATA2/CYPEDock/CYPocket/p450pocket
python predict_pdb.py --help
python train.py --help
```

Run the frozen real-sample regression with:

```bash
python verification/run_regression.py \
  --feature-python /path/to/cypocket-features/bin/python \
  --classifier-python /path/to/cypocket-classifier/bin/python
```

The regression recomputes PDB feature extraction, classifier inference, and protein-pocket data preparation, then compares features, labels, probabilities, predictions, residue mappings, and the predicted-pocket PDB with frozen expected artifacts.

Run the automated tests with:

```bash
python -m pip install -e ".[test]"
python -m pytest -q tests
```
