From this directory:
  python -m pip install -r requirements.txt
  python run_workflow.py

The bundled ../input_data.zip is validated and extracted into a new
outputs/run_TIMESTAMP/input_data/ directory. Results: outputs/run_TIMESTAMP/results.csv.
Optional: python run_workflow.py --source PATH --output NEW_OUTPUT_DIRECTORY
Existing output directories are never overwritten; no docking is performed.

This is a selected-call subset (32 cells / 309 arm-pose records / 29 physical
calls / 279 unique physical poses), not a full scenario hit-rate calculation.
The stored READY + strict gate=True filter is reused; donor geometry is not
recomputed. Protein-aligned heavy-atom RMSD uses allowed symmetry permutations;
fixed-name RMSD is separate. Read ../README.txt for scope and reference rules.
