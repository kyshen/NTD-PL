from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from viz.aggregate import aggregate_nonlinear_pmax_grid
from viz.style import PALETTE, apply_style


OUT_DIR = ROOT / "papers" / "tsp" / "figures"
OUT_STEM = "controlled_nonlinear_pmax_heatmap"

PANEL_LABELS = {
    "poly2": r"$s^2$",
    "poly3": r"$s^2+s^3$",
    "tanh": r"$\tanh(\kappa s)$",
    "exp": r"$(e^{\kappa s}-1)/\kappa$",
}


def _degree_gain_matrix(data, panel_order: list[str]) -> tuple[np.ndarray, list[int]]:
    wide = data.pivot_table(index=["panel", "x"], columns="method", values="mean").reset_index()
    wide["gain_rmse"] = wide["Tucker"] - wide["NTD-PL"]
    degrees = sorted(int(x) for x in wide["x"].unique())
    matrix = np.full((len(panel_order), len(degrees)), np.nan, dtype=float)
    for row_idx, panel_key in enumerate(panel_order):
        panel = wide.loc[wide["panel"].eq(panel_key)]
        for col_idx, degree in enumerate(degrees):
            value = panel.loc[panel["x"].eq(degree), "gain_rmse"]
            if not value.empty:
                matrix[row_idx, col_idx] = float(value.iloc[0])
    return matrix, degrees


def main() -> None:
    apply_style("single_column")
    data = aggregate_nonlinear_pmax_grid()
    panel_order = ["poly2", "poly3", "tanh", "exp"]
    missing = [panel for panel in panel_order if panel not in set(data["panel"])]
    if missing:
        raise RuntimeError(f"Missing controlled nonlinear panels: {missing}")
    matrix, degrees = _degree_gain_matrix(data, panel_order)

    cmap = LinearSegmentedColormap.from_list(
        "ntdpl_degree_gain",
        [PALETTE.heat_low, PALETTE.heat_mid, PALETTE.ntdpl],
    )
    fig, ax = plt.subplots(figsize=(7.16, 1.34))
    image = ax.imshow(
        matrix,
        aspect="auto",
        cmap=cmap,
        vmin=0.0,
        vmax=float(np.nanmax(matrix)),
    )

    ax.set_xticks(np.arange(len(degrees)))
    ax.set_xticklabels([str(degree) for degree in degrees])
    ax.set_yticks(np.arange(len(panel_order)))
    ax.set_yticklabels([PANEL_LABELS[key] for key in panel_order])
    ax.set_xlabel(r"degree $P$", labelpad=2.0)
    ax.set_ylabel("residual", labelpad=2.0)
    ax.tick_params(axis="both", length=0, pad=2.0)

    ax.set_xticks(np.arange(-0.5, len(degrees), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(panel_order), 1), minor=True)
    ax.grid(which="minor", color=PALETTE.white, linewidth=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    threshold = 0.55 * float(np.nanmax(matrix))
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            if not np.isfinite(value):
                continue
            ax.text(
                col_idx,
                row_idx,
                f"{value:.3f}",
                ha="center",
                va="center",
                color=PALETTE.white if value >= threshold else PALETTE.border,
                fontsize=7.0,
            )

    cbar = fig.colorbar(image, ax=ax, fraction=0.042, pad=0.025)
    cbar.set_label("RMSE gain", labelpad=4.0)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=2.5, width=0.7, labelsize=7.4, colors=PALETTE.border)
    cbar.formatter.set_powerlimits((-2, 2))
    cbar.update_ticks()
    fig.subplots_adjust(left=0.11, right=0.92, bottom=0.26, top=0.96)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{ext}", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
