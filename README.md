# NTD-PL

This repo contains the core implementation under `src/` and a Python-only experiment/plotting CLI under `experiment/`.

## Install (recommended: editable experiment package)

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -e .\experiment
```

### macOS / Linux

```bash
python -m venv .venv
./.venv/bin/python -m pip install -U pip
./.venv/bin/python -m pip install -e ./experiment
```

Notes:
- `experiment/pyproject.toml` declares the runtime dependencies (NumPy/Pandas/Matplotlib/SciPy/PyArrow).
- Datasets live under `data/`.

## Run experiments

All experiment entrypoints are exposed via:

```powershell
python -m experiment projects --verbose
```

Typical workflow is `run` to generate results and then `summary`/`*-table`/`*-plot` to export paper-facing artifacts.
For example:

```powershell
python -m experiment nonlinear-approx run
python -m experiment postprocess nonlinear-approx
```

Artifacts/figures are written under `experiment/outputs/`. Some commands also sync paper inputs into `latex-zh/inputs/`.

## Experiment catalog

See `experiment/doc.md` for a short description of each experiment.
