# NTD-PL Plotting Pipeline Migration

## Why the old plotting stack had to be retired

The previous paper figures were produced by experiment-specific scripts under `experiment/process/plots/` and helper modules under `experiment/process/helpers/`. That layout had four recurring failures:

1. Data loading, aggregation, plotting, export, and LaTeX syncing were all interleaved inside single functions.
2. Figure style was duplicated across scripts, so changing fonts, line widths, or legend behavior required touching many files.
3. Figure semantics were implicit: multiple scripts hard-coded scene ids, missing-rate filters, and panel layouts instead of declaring them as reusable figure specs.
4. The paper entrypoint was effectively "run a grab-bag of scripts until the right PDFs appear", not "build a controlled set of figures from stable specs".

The new plotting system treats the old plotting code as read-only reference. The active figure path is now:

`multirun/* -> viz/io.py -> viz/aggregate.py -> viz/specs.py -> viz/renderers.py -> viz/export.py -> experiment/outputs/figures/{main,appendix}`

## New directory layout

```text
viz/
  __init__.py
  aggregate.py
  catalog.py
  export.py
  io.py
  pipeline.py
  renderers.py
  specs.py
  style.py

scripts/
  make_figures.py
  make_single_figure.py

experiment/process/plots/
  __init__.py
  paper_figures.py

docs/
  figure_layout_guide.md
  figure_mapping.csv
  plotting_migration.md
```

## Layer responsibilities

### `viz/io.py`

- Loads `runs.parquet`, `curves.parquet`, table CSV summaries, `state.mat`, and CAVE scene cubes.
- Normalizes path handling and reconstructs tensor outputs from `core/factors/beta` when required.
- Contains no figure composition logic.

### `viz/aggregate.py`

- Produces paper-facing tables for each figure family.
- Examples:
  - paired RMSE gaps for linear consistency
  - `alpha` and `p_max` trend summaries with mean/std bands
  - scene-wise gains for full reconstruction and completion
  - difficulty-boundary heatmap tables
  - parameter trend plus contribution map tables
  - image, spectra, and surface payload tables for composite figures
- Rendering code does not read raw runs directly.

### `viz/style.py`

- Centralizes paper presets by role:
  - `single_column`
  - `double_column`
  - `appendix_wide`
  - `compact`
- Controls fonts, line widths, markers, grayscale-safe linestyles, grid behavior, and colorbar style.

### `viz/specs.py` and `viz/catalog.py`

- Every paper figure is declared as a `FigureJob`.
- Each job states:
  - figure id
  - figure family
  - export section and role
  - final output stem
  - LaTeX path
  - final LaTeX-facing `inputs/figures/*` path

### `viz/renderers.py`

- Implements reusable figure families:
  - `paired_boxplot`
  - `boxplot_summary`
  - `line_grid`
  - `step_line_grid`
  - `sorted_gain_grid`
  - `heatmap`
  - `line_plus_heatmap`
  - `image_comparison_grid`
  - `spectra_panel`
  - `geometry_evolution`
  - `geometry_response_maps`

### `viz/export.py`

- Exports stable PDF filenames into `experiment/outputs/figures/main` and `experiment/outputs/figures/appendix`.
- Leaves LaTeX to consume the same artifacts through the `latex-zh/inputs -> experiment/outputs` directory junction.

### `experiment/process/plots/paper_figures.py`

- Provides the only remaining experiment `postprocess` plot registration layer.
- Maps each experiment name to the figure ids owned by the new `viz` registry.
- Keeps `python -m experiment postprocess <exp>` working after deleting the legacy plot scripts.

## How to build figures

Build the whole paper:

```bash
.venv\Scripts\python.exe scripts\make_figures.py
```

Build only main-text figures:

```bash
.venv\Scripts\python.exe scripts\make_figures.py --scope main
```

Build a single figure:

```bash
```

Run experiment-local postprocessing, including tables plus the mapped figures:

```bash
.venv\Scripts\python.exe -m experiment postprocess cave-random-completion
```

## How to add a new figure

1. Add a tidy aggregation function to `viz/aggregate.py`.
2. Choose an existing renderer family, or add a new one to `viz/renderers.py`.
3. Declare the new figure in `viz/catalog.py` with:
   - output role
   - section (`main` or `appendix`)
   - output stem
   - LaTeX path under `inputs/figures/*`
4. If the figure should be produced by `python -m experiment postprocess <exp>`, add its id to `EXPERIMENT_FIGURES` in `experiment/process/plots/paper_figures.py`.
5. Regenerate with `scripts/make_single_figure.py <figure_id>`.

## Changelog

- Added a new standalone `viz/` plotting package.
- Moved paper figure generation to declarative `FigureJob` registry entries.
- Introduced role-driven size presets instead of per-script `figsize` tuning.
- Switched paper figure outputs to `experiment/outputs/figures/main` and `experiment/outputs/figures/appendix`.
- Replaced the old `experiment/process/plots/*.py` stack with a single thin `experiment/process/plots/paper_figures.py` registration layer.
- Removed the unused `experiment/process/bundles/` discovery path.
- Updated LaTeX figure paths in `latex-zh/sections/experiment.tex` and `latex-zh/sections/appendix.tex`.
- Replaced copied LaTeX inputs with a `latex-zh/inputs -> experiment/outputs` junction so the paper always reads the canonical outputs tree.
- Retired the old 3D `geometry_surface_grid` figure and replaced it with a 1D link-evolution figure plus 2D response maps that are easier to read in print.
