# Methods-ready technical specification

## Structure preprocessing and labels

Each sample contains one cytochrome P450 chain. PDB residues are retained in file/model/chain order and identified by author chain identifier, author residue number, insertion code and residue name. The structural encoder requires the backbone atoms N, CA, C and O for every retained residue. Alternate atom locations are selected deterministically by occupancy and then by blank/A preference. The three-decimal coordinates represented in the PDB are converted from ångström to nanometres in float64 and then cast to float32. This order reproduces the MDTraj conversion used by the legacy PocketMiner implementation and avoids last-bit differences caused by first retaining Biopython's float32 ångström values.

Pocket labels are mapped to protein residues by exact residue identity and residue type. A CA-coordinate deviation of at most 0.25 Å is used only as a consistency check. Coordinate-only matching is disabled by default and, when explicitly requested for legacy recovery, must yield a unique match of the same residue type. Labels are binary and aligned one-to-one with the feature rows.

## Structural residue representation

The feature extractor reproduces the TensorFlow PocketMiner/GVP encoder used by the original scripts. It constructs a 30-nearest-neighbour residue graph from CA coordinates and combines radial-basis distances, relative positional encodings, edge directions, backbone dihedrals, backbone orientations, an approximate side-chain direction and amino-acid identity. The network uses node features `(8 vector, 50 scalar)`, edge features `(1 vector, 32 scalar)`, hidden dimensions `(16 vector, 100 scalar)`, four message-passing layers and dropout 0.1. The 100 scalar channels before the historical classification head form the per-residue feature matrix.

The TensorFlow object checkpoint is restored only after all model variables are built. Existing model variables must match; missing `.index`/`.data-*` shards or incompatible variables terminate extraction. Feature values must have shape `[L,100]`, dtype float32 and no non-finite entries.

## Residue classifier

The classifier projects each 100-dimensional structural feature to 768 dimensions and supplies it as `inputs_embeds` to a BERT-base encoder with 12 layers, hidden size 768, 12 attention heads and feed-forward size 3072. A linear layer maps each contextual residue embedding to two logits. The audited model has 109,561,346 parameters. Although the historical filenames contain “distilbert”, the state-dict keys and tensor shapes identify a full BERT-base model with a pooler.

Proteins longer than 512 residues are divided into independent, non-overlapping 512-residue windows, with positional indices reset in each window. This preserves the behavior used to produce the existing checkpoints; no information crosses a window boundary. The separately declared legacy classifier dropout was never called in the historical forward pass and is therefore not introduced in compatibility mode. Standard BERT internal dropout remains enabled during training.

## Optimization and model selection

The new implementation uses AdamW, gradient clipping and a linear warm-up/decay schedule. The documented default configuration uses batch size 1, 50 maximum epochs, learning rate 1×10⁻⁵, weight decay 1×10⁻⁴, 5% warm-up, gradient-norm limit 1.0, seed 2026, validation PR-AUC monitoring and patience 10. Cross-entropy is computed over every non-padding residue; optional positive-class weighting is normalized by the summed class weight. Gradient accumulation is normalized by the exact number (or total class weight) of residues in the accumulated microbatches. Seeds are set for Python, NumPy and PyTorch, deterministic kernels are requested, and epoch-derived sampler seeds permit reproducible epoch-boundary resume.

Fine-tuning policies are explicitly recorded: all parameters; feature projection plus classifier only; or projection, classifier and the last N BERT layers. De novo classifier training requires an explicitly supplied local BERT-base initialization. Random BERT initialization is rejected unless separately opted into. Model selection defaults to validation PR-AUC with early stopping; the decision threshold is fixed before test evaluation.

New checkpoints include the architecture, optimizer, scheduler and mixed-precision GradScaler states, epoch/global step, RNG states, training configuration, validation metrics, threshold, software version and dataset-manifest SHA-256. Resume rejects a changed configuration or manifest. Generated manifests store feature, label, protein, pocket and feature-extractor hashes when available. The two historical weights are raw state dictionaries and cannot retrospectively supply the missing training seed or code revision.

## Evaluation and inference

Pocket probability is `softmax(logits)[class 1]`. The legacy default decision is the two-class argmax, normally equivalent to threshold 0.5. Threshold optimization, if performed, uses validation data only. No batch or protein is silently skipped.

Evaluation reports pooled residue-level accuracy, balanced accuracy, precision, recall, F1, Matthews correlation coefficient, ROC-AUC, average precision (PR-AUC) and a fixed-label confusion matrix, together with per-protein results and protein-macro averages. ROC-AUC and PR-AUC are calculated from continuous probabilities. Prediction artifacts include probabilities, binary labels, an identity-preserving residue table, a predicted-residue PDB and a JSON provenance record containing input and checkpoint hashes.

## Reproducibility boundary

Classification from existing 100-dimensional arrays is fully reconstructable from the supplied PyTorch state dictionaries and explicit model configuration. The compatible PocketMiner TensorFlow checkpoint is archived in the sibling `../P450model/` directory, making end-to-end inference from a new PDB self-contained in the combined source-and-model distribution. On a real 436-residue sample, the refactored extractor was bitwise identical to the legacy MDTraj/GVP implementation, the protein+pocket labels exactly matched the archived labels, and end-to-end predictions matched classification from the archived features. Exact hashes, environments and comparison tolerances are frozen in `VERIFICATION.md` and `verification/verification_report.json`. Replacing the checkpoint with random weights or a different encoder would preserve shape but invalidate feature semantics and is not considered reproduction.

Formal third-party citations and the unresolved redistribution-license boundary are listed in `THIRD_PARTY_NOTICES.md`. The dataset construction provenance and raw checkpoint metadata do not contain enough information to prove the original train/validation/test split algorithm, software commit or weight-to-manifest association; these are reported as unknown rather than reconstructed by assumption.
