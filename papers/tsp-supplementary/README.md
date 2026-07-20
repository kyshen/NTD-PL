# TSP Supplementary Material

This is the standalone LaTeX project for the supplementary material associated
with the TSP paper in `../tsp`.

## Build

Compile the main paper first so that cross-document references are available:

```powershell
cd ..\tsp
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Then run from this directory:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The compiled PDF and all intermediate files are written to `out/`. The root
`.gitignore` already excludes LaTeX intermediates under paper projects and all
`out/` directories.

## Project Layout

- `main.tex`: document entry point and bibliography configuration.
- `preamble.tex`: TSP preamble reuse, supplementary numbering, and asset paths.
- `sections/`: supplementary model details, proofs, and experiments.
- `figures/`: figures used only by the supplementary material.
- `tables/`: tables used only by the supplementary material.

The project contains the detailed model derivations, proofs, extended
experimental protocols, supplementary figures, and supplementary tables. It
reuses `../tsp/macros.tex`, the TSP bibliography, and the IEEEtran
bibliography style. `latexmkrc` adds the TSP directory to the BibTeX search
path. Supplementary-only figures and tables belong in the local `figures/` and
`tables/` directories.

Section labels defined by the supplementary material use the `supp:` prefix.
Figure, table, equation, and theorem labels retain their usual type prefixes.
When `../tsp/main.aux` is available, labels from the main paper can be referenced
with the `paper-` prefix, for example `\ref{paper-sec:theory}`.
