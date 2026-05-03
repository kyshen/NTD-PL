from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from viz.style import PALETTE, apply_style, style_axes


OUT_DIR = ROOT / "neurips" / "figures"
OUT_STEM = "controlled_nonlinear_summary_slope"

ROWS = [
    (r"$s^2$", -13.36, -16.79, 3.42),
    (r"$s^2+s^3$", -17.77, -25.28, 7.51),
    (r"$\sin(\kappa s)$", -11.81, -25.12, 13.31),
    (r"$\tanh(\kappa s)$", -12.48, -23.07, 10.59),
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
            linear_value + 0.45,
            yi,
            f"{gap:.1f} dB",
            va="center",
            ha="left",
            fontsize=6.3,
            color=PALETTE.border,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(-27.0, -9.8)
    ax.set_xticks([-25, -20, -15, -10])
    ax.set_xlabel("NMSE (dB), lower is better")
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
