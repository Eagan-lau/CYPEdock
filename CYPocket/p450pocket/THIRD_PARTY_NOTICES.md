# Third-party references and release boundary

This file records scientific provenance; it is not a substitute for legal review or the license texts shipped by each dependency.

## Scientific references

- PocketMiner, *Nature Communications* 14, 1177 (2023), DOI: [10.1038/s41467-023-36699-3](https://doi.org/10.1038/s41467-023-36699-3).
- Geometric Vector Perceptrons (GVP), ICLR 2021: [OpenReview 1YLJDvSx6J4](https://openreview.net/forum?id=1YLJDvSx6J4).
- BERT, NAACL 2019, DOI: [10.18653/v1/N19-1423](https://doi.org/10.18653/v1/N19-1423).
- AlphaFold Protein Structure Database, *Nucleic Acids Research* 2022, DOI: [10.1093/nar/gkab1061](https://doi.org/10.1093/nar/gkab1061).
- Foldseek, *Nature Biotechnology*, DOI: [10.1038/s41587-023-01773-0](https://doi.org/10.1038/s41587-023-01773-0).
- CD-HIT, *Bioinformatics* 2006, DOI: [10.1093/bioinformatics/btl158](https://doi.org/10.1093/bioinformatics/btl158).
- Biopython, *Bioinformatics* 2009, DOI: [10.1093/bioinformatics/btp163](https://doi.org/10.1093/bioinformatics/btp163).
- PyMOL was used in the user-provided label-transfer workflow; the exact version and command script were not preserved and must be supplied in the final paper.

The final manuscript should import complete author lists and BibTeX records from the DOI landing pages rather than treating this compact inventory as its bibliography.

## Checkpoints and source code

- `../P450model/best_protein_distilbert_classifier.pth` and `../P450model/final_protein_distilbert_classifier.pth` are user-provided research artifacts. The files contain no embedded owner, license or redistribution grant.
- `../P450model/pocketminer.*` was recovered from an existing local research installation. Its SHA-256 is documented, but the local files do not establish permission to redistribute the checkpoint.
- `p450pocket/features/gvp.py` and `p450pocket/features/pocketminer.py` are compatibility refactors of the legacy implementation in this workspace. Before public release, compare the originating project's license and preserve every required copyright/attribution notice.
- This refactored project currently has no top-level software license. Select and add one only after ownership of the legacy-derived portions is confirmed.

Internal inference and reproducibility testing can use the local artifacts. Public code, wheel, supplementary archive or model-hub publication must omit restricted artifacts or obtain explicit permission from the relevant rights holders.
