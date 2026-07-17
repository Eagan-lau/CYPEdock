# CYPEdock CYP450 Holo Library

This directory is intentionally empty in version 0.1.0.

The published library must contain a Foldseek database, the original holo
coordinate files, and a companion `manifest.jsonl`. The database target
identifier must exactly match one manifest record. Each record uses this
schema:

```json
{
  "schema_version": 1,
  "target_id": "template_0001_A",
  "structure_file": "templates/template_0001.cif",
  "structure_sha256": "<sha256>",
  "protein_chain": "A",
  "cofactor_residues": [
    {"chain": "A", "number": 501, "insertion_code": ""}
  ]
}
```

The expected default database prefix is `foldseek/cyp450`.

Generated external libraries may instead provide `library_manifest.jsonl`
with one `cofactor_groups` record per target. Relative coordinate paths under
the library root are recommended for portability. Legacy absolute paths can be
remapped through `foldseek.structure_root`; the basename of each declared
structure is resolved under that directory. A generated database prefix such
as `cyp450_holo` must be configured explicitly or placed at the library root
for automatic detection.
