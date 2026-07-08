# TSP LaTeX Project

This directory is a clean IEEE Transactions-style LaTeX project for preparing
the TSP version of NTD-PL.

## Build

```powershell
latexmk -pdf main.tex
```

To clean generated files:

```powershell
latexmk -C main.tex
```

## Template Source

The IEEEtran template package was downloaded into `template_source/` from a
CTAN mirror. The project keeps local copies of:

- `IEEEtran.cls`
- `IEEEtran.bst`
- `IEEEabrv.bib`

The main file uses:

```tex
\documentclass[journal]{IEEEtran}
```

which is the standard IEEE journal manuscript class appropriate for IEEE
Transactions submissions, including IEEE Transactions on Signal Processing.
