# Experiment Catalog

This file briefly describes each experiment registered in `experiment/registry.py`.

## CLI entrypoints

These commands assume you are using the project's Python environment (recommended):

```powershell
& ./.venv/Scripts/Activate.ps1
```

### From repo root (no install)

You can run the CLI directly from the repository root via the `experiment` package module:

```powershell
python -m experiment <command> ...
```

### After installing `./experiment`

After installing `./experiment` (editable install is recommended), you can also run the CLI via either:

```powershell
python -m experiment <command> ...
```

or, if installed via `pip install -e ./experiment`, the console script:

```powershell
ntd-experiments <command> ...
```

List available projects:

```powershell
python -m experiment projects --verbose
```

Outputs:
- Multirun sweeps write parquet results under `artifacts/multirun/<exp>/` with mode-specific subdirectories:
  - `artifacts/multirun/<exp>/benchmark/`: Non-NTDPL method results
  - `artifacts/multirun/<exp>/run/`: NTDPL method results
  - Combined `runs.parquet` and `curves.parquet` files at `artifacts/multirun/<exp>/`
- Individual run outputs (logs, checkpoints) under `artifacts/outputs/<exp>/<mode>/` (managed by Hydra)
- Tables/figures are exported under `artifacts/paper-outputs/`
- Some table commands also sync LaTeX inputs into `papers/latex-zh/inputs/`

## Synthetic validation

- `linear-consistency`: Linear consistency validation on controlled synthetic data.
- `nonlinear-approx`: Unified nonlinear link approximation analysis on controlled synthetic data, covering polynomial and smooth non-polynomial links.
- `geometry-visualization`: Geometric surface visualization analysis on controlled synthetic data.

## Real data / downstream

- `cave-representation`: CAVE hyperspectral nonlinear low-rank representation with compression-reconstruction analysis.

## Summary utilities

- `paper-tables`: Generate LaTeX table inputs used in the paper (placeholders and small utilities).

## How to run

### Execution modes

Experiments support three run modes via CLI commands:

- **`benchmark`**: Runs non-NTDPL methods only (tucker, cp, tr, tt). Results stored in `artifacts/multirun/<exp>/benchmark/`.
- **`run`**: Runs both benchmark and NTDPL methods. Results stored in respective `benchmark/` and `run/` subdirectories. This is the typical full experiment run.
- **`ntdpl`**: Runs NTDPL-specific experiments only. Results stored in `artifacts/multirun/<exp>/run/`.

The runner automatically detects which methods are NTDPL and separates execution accordingly. This separation:
- Prevents interference between different method groups
- Enables independent result aggregation per mode
- Keeps output directories organized by experimental intent

Results from both modes are combined when `collect.py` aggregates to parquet files.

### Postprocessing

Postprocessing (plots/tables export under `artifacts/paper-outputs/` and LaTeX inputs sync where applicable) is:

- **Automatic only for `run` mode** (i.e., when both benchmark and NTDPL are executed together).
- **Manual for `benchmark` / `ntdpl`** runs via the dedicated command:

```powershell
python -m experiment postprocess <experiment>
```

This design keeps benchmark/NTDPL runs lightweight and lets you run postprocessing exactly once after both are finished.

### Synthetic validation

`linear-consistency`:

```powershell
python -m experiment linear-consistency run           # Both benchmark and NTDPL
python -m experiment linear-consistency benchmark     # Benchmark only
python -m experiment linear-consistency ntdpl         # NTDPL only

# After you have the results you want:
python -m experiment postprocess linear-consistency
```

`nonlinear-approx`:

```powershell
python -m experiment nonlinear-approx run
python -m experiment nonlinear-approx benchmark
python -m experiment nonlinear-approx ntdpl

python -m experiment postprocess nonlinear-approx
```

`geometry-visualization`:

```powershell
python -m experiment geometry-visualization run
python -m experiment geometry-visualization benchmark
python -m experiment geometry-visualization ntdpl

python -m experiment postprocess geometry-visualization
```

### Real data / downstream

`cave-representation`:

```powershell
python -m experiment cave-representation run
python -m experiment cave-representation benchmark
python -m experiment cave-representation ntdpl

python -m experiment postprocess cave-representation
```

### Summary utilities

`paper-tables` (writes/syncs small paper-facing LaTeX table inputs):

```powershell
python -m experiment paper-tables
```
