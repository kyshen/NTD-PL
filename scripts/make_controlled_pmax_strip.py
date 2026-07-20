from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.tucker import TuckerData
from viz.style import PALETTE, apply_style
from scripts.make_controlled_linear_noise_preview import (
    SEEDS,
    _add_noise,
    _fit_rmse,
    _response,
)


OUT_DIR = ROOT / "papers" / "tsp-supplementary" / "figures"
OUT_STEM = "controlled_nonlinear_pmax_heatmap"
ALPHA_REF = 0.25
P_VALUES = (1, 2, 3, 4, 5, 6)

PANEL_LABELS = {
    "linear": r"$s$",
    "poly3": r"$s^2+s^3$",
    "tanh": r"$\tanh(\kappa s)$",
    "exp": r"$(e^{\kappa s}-1)/\kappa$",
}


def _target_for(seed: int, panel: str, alpha: float) -> np.ndarray:
    data = TuckerData(shape=(10, 10, 10), rank=(4, 4, 4), seed=seed)
    signal = np.asarray(data.get("fit").dense, dtype=np.float32)
    signal = signal / (np.linalg.norm(signal) + 1e-8) * np.sqrt(signal.size)
    clean = _response(signal, panel, alpha)
    panel_index = list(PANEL_LABELS).index(panel)
    noise_seed = 10_000 + 1_000 * seed + 37 * panel_index
    if panel != "linear":
        noise_seed += int(round(alpha * 1000))
    return _add_noise(clean, seed=noise_seed)


def _collect() -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    rank = (4, 4, 4)
    panel_order = list(PANEL_LABELS)
    for panel in panel_order:
        for seed in SEEDS:
            target = _target_for(seed, panel, ALPHA_REF)
            tucker_rmse = _fit_rmse("Tucker", target, rank, seed)
            for p_value in P_VALUES:
                ntd_rmse = _fit_rmse("NTD-PL", target, rank, seed, p_max=p_value)
                rows.append(
                    {
                        "panel": panel,
                        "seed": int(seed),
                        "P": int(p_value),
                        "Tucker": float(tucker_rmse),
                        "NTD-PL": float(ntd_rmse),
                        "gain_rmse": float(tucker_rmse - ntd_rmse),
                    }
                )
    return pd.DataFrame(rows)


def _degree_gain_matrix(data: pd.DataFrame, panel_order: list[str]) -> tuple[np.ndarray, list[int]]:
    summary = data.groupby(["panel", "P"], as_index=False)["gain_rmse"].mean()
    degrees = sorted(int(x) for x in summary["P"].unique())
    matrix = np.full((len(panel_order), len(degrees)), np.nan, dtype=float)
    for row_idx, panel_key in enumerate(panel_order):
        panel = summary.loc[summary["panel"].eq(panel_key)]
        for col_idx, degree in enumerate(degrees):
            value = panel.loc[panel["P"].eq(degree), "gain_rmse"]
            if not value.empty:
                matrix[row_idx, col_idx] = float(value.iloc[0])
    return matrix, degrees


def main() -> None:
    apply_style("single_column")
    panel_order = list(PANEL_LABELS)
    data = _collect()
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
    ax.set_ylabel("response", labelpad=2.0)
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
    data.to_csv(OUT_DIR / f"{OUT_STEM}.csv", index=False)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{ext}", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
