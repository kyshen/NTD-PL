This directory contains an English, arXiv-style LaTeX manuscript for NTD-PL.

Notes:
- It is independent from `latex-zh/` and can be edited/compiled separately.
- Figures, tables, and bibliography assets are copied locally so the package is self-contained.
- The manuscript uses a standard `article` + `natbib` setup for better arXiv portability.

Typical build:
- `latexmk -pdf main.tex`

Main entry:
- `main.tex`
