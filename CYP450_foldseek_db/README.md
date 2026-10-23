# CYP450 Foldseek Database Placeholder

The generated CYP450 holo-template database is intentionally excluded from Git
because it contains large database indices and coordinate data.

Download `CYP450_foldseek_db.tar.xz` from the project data deposit:

- Preferred source: Zenodo [DOI: 10.5281/zenodo.23439146](https://doi.org/10.5281/zenodo.23439146).
- [Download CYP450_foldseek_db.tar.xz](https://zenodo.org/api/records/23439146/files/CYP450_foldseek_db.tar.xz/content)
- Backup mirror: [Baidu Netdisk](https://pan.baidu.com/s/17dh949GLClk_QC1WUDejkw?pwd=7qnp)
  (extraction code: `7qnp`).
- SHA-256: `984056598b0d45714eeec3f504629ecf6977c58c9e6f19cb1212296370d0827a`

From the repository root, install the database with:

```bash
tar -xJf /path/to/CYP450_foldseek_db.tar.xz
```

The extracted directory must contain `library_manifest.jsonl`, the
`cyp450_holo` Foldseek database prefix, and `templates/`. Complete asset and
validation instructions are provided in [`../ASSETS.md`](../ASSETS.md).
