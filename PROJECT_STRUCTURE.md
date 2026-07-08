# Project Structure

This repository is organized around four top-level work areas:

- `src/`: core NTD-PL implementation.
- `experiment/`, `scripts/`, `viz/`: experiment runners, postprocessing, and plotting utilities.
- `papers/`: LaTeX and paper-facing projects.
  - `papers/tsp/`: current TSP journal version.
  - `papers/neurips/`: earlier NeurIPS version.
  - `papers/latex-zh/`: Chinese LaTeX draft/materials.
  - `papers/supplementary/`: supplementary package snapshot.
- `artifacts/`: generated experiment outputs and result summaries.
  - `artifacts/multirun/`: Hydra sweep results and collected parquet summaries.
  - `artifacts/outputs/`: per-run Hydra logs/checkpoints.
  - `artifacts/paper-outputs/`: tables and figures produced by postprocessing.
  - `artifacts/results/`: ad hoc or script-specific result folders.

Legacy notes and one-off command records are kept in `archive/legacy-notes/`.
New experiments should write generated files under `artifacts/` and paper files
under `papers/`.
