from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from viz.style import PALETTE, apply_style, style_axes


OUT_DIR = ROOT / "papers" / "tsp" / "figures"
OUT_STEM = "controlled_nonlinear_summary_slope"

ROWS = [
    (r"$s^2$", 0.2196, 0.1500, 0.0695),
    (r"$s^2+s^3$", 0.1345, 0.0561, 0.0784),
    (r"$\tanh(\kappa s)$", 0.2385, 0.0707, 0.1678),
    (r"$(e^{\kappa s}-1)/\kappa$", 0.1787, 0.1116, 0.0671),
]


def main() -> None:
    apply_style("compact")

    labels = [row[0] for row in ROWS]
    best_linear = np.asarray([row[1] for row in ROWS], dtype=float)
    ntdpl = np.asarray([row[2] for row in ROWS], dtype=float)
    gaps = np.asarray([row[3] for row in ROWS], dtype=float)
    y = np.arange(len(ROWS))[::-1]

    fig, ax = plt.subplots(figsize=(4.45, 1.58))
    for yi, linear_value, ntdpl_value, gap in zip(y, best_linear, ntdpl, gaps, strict=True):
        ax.plot(
            [linear_value, ntdpl_value],
            [yi, yi],
            color=PALETTE.neutral,
            linewidth=1.05,
            zorder=1,
        )
        ax.scatter(linear_value, yi, s=20, color=PALETTE.tucker, marker="s", label="Best linear" if yi == y[0] else None, zorder=2)
        ax.scatter(ntdpl_value, yi, s=26, color=PALETTE.ntdpl, marker="o", label="NTD-PL" if yi == y[0] else None, zorder=3)
        ax.text(
            linear_value + 0.006,
            yi,
            f"{gap:.3f}",
            va="center",
            ha="left",
            fontsize=6.3,
            color=PALETTE.border,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.02, 0.26)
    ax.set_xticks([0.05, 0.10, 0.15, 0.20, 0.25])
    ax.set_xlabel("RMSE, lower is better")
    ax.set_ylabel("")
    style_axes(ax, grid=True)
    ax.xaxis.grid(True, color=PALETTE.grid, linewidth=0.55, alpha=0.7)
    ax.yaxis.grid(False)
    ax.legend(loc="lower left", ncol=2, bbox_to_anchor=(0.0, 1.01), borderaxespad=0.0, handlelength=1.0)
    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.27, top=0.82)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{ext}", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
