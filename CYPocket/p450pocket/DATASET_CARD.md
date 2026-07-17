# CYPocket Dataset Card

Audit version: 2026-07-15. This card describes the frozen local snapshot. The original dataset version identifier, DOI, maintainer, and license were not supplied with the directory.

## Scale and Layout

The audited dataset contains 11,933 samples: 8,353 training proteins, 2,386 validation proteins, and 1,194 test proteins. Each split contains:

- `protein/`: complete single-chain AlphaFold P450 PDB files.
- `pocket/`: PDB subsets containing mapped pocket residues.
- `tokens/`: PocketMiner/GVP residue features stored as `float32 [L, 100]` arrays.
- `labels/`: binary residue labels stored as `int64 [L]` arrays aligned with feature rows.
- Other directories: intermediate or predicted outputs from legacy processing; these are not treated as training ground truth.

The audit confirmed sample-level feature-label length agreement, finite feature values, binary label domains, and an overall positive-residue fraction near 10 percent. `test/test_tokens` is a historical batch-dimension copy of `test/tokens`; the new loader supports both `[L, 100]` and `[1, L, 100]`, but manifests index only `tokens/`.

| Split | Proteins | Residues | Positive residues | Positive fraction | Length min / median / mean / max |
|---|---:|---:|---:|---:|---:|
| train | 8,353 | 3,673,277 | 370,454 | 10.085109% | 56 / 456 / 439.7554 / 607 |
| val | 2,386 | 1,049,891 | 105,453 | 10.044186% | 61 / 457 / 440.0214 / 590 |
| test | 1,194 | 529,801 | 53,082 | 10.019234% | 46 / 458 / 443.7194 / 588 |
| total | 11,933 | 5,252,969 | 528,989 | 10.070286% | not pooled |

No pairing, dtype, shape, label-domain, finite-value, or feature-label length anomaly was found in the audited splits.

## Data Source and Label Definition

The user-provided workflow describes the following construction procedure:

1. Collect 1,379 experimental P450 structures and separate them into 2,632 chains.
2. Search 366,848 AlphaFold P450 structures with Foldseek, retain the best match above `TM-score > 0.8`, and obtain 225,502 matched pairs.
3. Exclude templates without a ligand, without heme, or with an inactive ligand distant from catalytic iron; apply an `Fe-ligand distance < 10 angstrom` condition to retain 1,795 templates.
4. Reduce AlphaFold sequence redundancy with CD-HIT at `c = 0.5`, producing 11,933 P450 samples.
5. Align template and AlphaFold structures in PyMOL and label AlphaFold residues within 5 angstrom of the transferred ligand as pocket residues.

The labels are therefore structural-homology and ligand-geometry proxy labels rather than independent experimental binding-site measurements for every AlphaFold structure. The approximately 1 angstrom distance in the legacy `get_labels.py` script was used to map a pocket-subset PDB back to the complete protein; it was not the biological pocket-definition radius. The production workflow uses exact residue identity by default and applies a 0.25 angstrom CA-distance consistency check.

The construction description was derived from user-provided documentation and associated files rather than a machine-verifiable original construction manifest. A public dataset release should add experimental structure identifiers, template-to-AlphaFold mappings, ligand-filter tables, software versions, and formal dataset citations.

## Verified Mappings

A feature row is identified by:

```text
model index + author chain ID + author residue number + insertion code + residue name
```

Re-parsing protein and pocket PDB files for three real training samples produced labels element-wise identical to the archived arrays.

The parser explicitly handles blank chain identifiers, insertion codes, and alternate atom locations. Production commands reject missing `N/CA/C/O` backbone atoms, unknown polymer residues, multiple models without explicit selection, and multiple chains without an explicit target chain.

## Split and Leakage Audit

The directory does not contain the original split-generation script, random seed, or grouping key. It is therefore not possible to claim retrospectively whether the split was random, organism-based, sequence-clustered, template-grouped, or constructed by another rule. The audit found no repeated sample identifier or identical sequence across splits, but a publication should document the original split strategy and repeat leakage analysis by sequence and structural clusters where possible.

The raw PyTorch classifier state dictionaries do not contain a training-manifest hash. Directory organization, user documentation, and training logs support an association between the dataset and weights, but the relationship cannot be proven cryptographically from the checkpoints. The new workflow records feature, label, protein, pocket, and extractor hashes in each manifest and records the manifest SHA-256 in structured checkpoints.

The test split must remain frozen until model selection and threshold selection are complete. Historical files under `test/pred/` do not record their generating checkpoint and cannot be treated as ground truth or as a checkpoint-consistency reference.

## Biases and Limitations

- Foldseek thresholds, best-match selection, and CD-HIT alter protein-family and structural-space coverage.
- Template and AlphaFold local errors propagate into the 5 angstrom labels.
- Static structures do not fully represent flexible pockets, induced fit, or ligand specificity.
- The residue-level classes are imbalanced, so accuracy alone is insufficient.
- A single-chain AlphaFold training distribution does not justify unvalidated concatenation of multi-chain complexes.
- `pocket/` records residue subsets rather than ligand coordinates or separate pocket-instance annotations.

A publication should report a dataset version or hash, protein and residue counts per split, positive fractions, length distributions, homology controls, and the number of rejected or unmappable samples.
