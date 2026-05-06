from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from viz.aggregate import aggregate_nonlinear_pmax_grid
from viz.style import PALETTE, apply_style, style_axes


OUT_DIR = ROOT / "neurips" / "figures"
OUT_STEM = "controlled_nonlinear_degree_gain_compact"

PANEL_STYLES = {
    "poly3": {"label": r"$s^2+s^3$", "color": "#C97C1A", "marker": "s"},
    "tanh": {"label": r"$\tanh(\kappa s)$", "color": PALETTE.tr, "marker": "D"},
    "exp": {"label": r"$(e^{\kappa s}-1)/\kappa$", "color": PALETTE.tt, "marker": "^"},
}


def _degree_gain_data():
    data = aggregate_nonlinear_pmax_grid()
    pivot = data.pivot_table(index=["panel", "x"], columns="method", values="mean").reset_index()
    pivot["gain"] = pivot["Tucker"] - pivot["NTD-PL"]
    return pivot


def main() -> None:
    apply_style("compact")
    data = _degree_gain_data()

    fig, ax = plt.subplots(figsize=(5.35, 1.85))
    for panel, style in PANEL_STYLES.items():
        sub = data.loc[data["panel"].eq(panel)].sort_values("x")
        ax.plot(
            sub["x"].to_numpy(dtype=float),
            sub["gain"].to_numpy(dtype=float),
            color=style["color"],
            marker=style["marker"],
            markersize=3.8,
            linewidth=1.35,
            alpha=0.92,
            label=style["label"],
        )

    ax.axhline(0.0, color=PALETTE.tucker, linestyle="--", linewidth=0.8)
    ax.set_xlim(0.85, 6.15)
    ax.set_ylim(-0.7, 14.4)
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.set_yticks([0, 5, 10])
    ax.set_xlabel(r"Polynomial degree $P$")
    ax.set_ylabel(r"NMSE gain over Tucker (dB)")
    style_axes(ax, grid=True)
    ax.legend(
        loc="upper left",
        ncol=4,
        columnspacing=0.7,
        handlelength=1.3,
        borderaxespad=0.2,
    )
    fig.subplots_adjust(left=0.12, right=0.995, bottom=0.26, top=0.92)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{ext}", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
