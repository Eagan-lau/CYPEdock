# Docking results and reproducible RMSD analysis

This package contains 32 Cell ID folders with the selected receptor, ligand,
and complex structures, plus each search command and its prepared PDBQT inputs.

```bash
python -m pip install -r analysis_workflow/requirements.txt
python analysis_workflow/run_workflow.py
```

The workflow validates and extracts `input_data.zip` automatically. Results go
to a fresh `analysis_workflow/outputs/run_TIMESTAMP/` directory. For Vina search
instructions and the cohort, RMSD, symmetry and sampling limitations, read
[README.txt](README.txt). These files preserve the supplied 32-cell snapshot;
later expanded-F searches are not mixed into this delivery.
