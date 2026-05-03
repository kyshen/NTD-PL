from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from viz.aggregate import aggregate_nonlinear_step_grid
from viz.style import PALETTE, apply_style, style_axes


OUT_DIR = ROOT / "neurips" / "figures"
OUT_STEM = "controlled_nonlinear_step_strip"

PANEL_LABELS = {
    "poly2": r"$s^2$",
    "poly3": r"$s^2+s^3$",
    "sin": r"$\sin(\kappa s)$",
    "tanh": r"$\tanh(\kappa s)$",
}


def _plot_panel(ax: plt.Axes, data, panel_key: str, *, show_ylabel: bool) -> None:
    panel = data.loc[data["panel"].eq(panel_key)].copy()
    curve = panel.loc[panel["kind"].eq("curve")].sort_values("x")
    transitions = panel.loc[panel["kind"].eq("transition")].sort_values("x")

    x = curve["x"].to_numpy(dtype=float)
    y = curve["mean"].to_numpy(dtype=float)
    std = curve["std"].fillna(0).to_numpy(dtype=float)

    ax.fill_between(x, y - std, y + std, color=PALETTE.ntdpl, alpha=0.08, linewidth=0)
    ax.plot(x, y, color=PALETTE.ntdpl, linewidth=1.55)

    for item in transitions.itertuples(index=False):
        xpos = float(item.x)
        nearest = int(np.argmin(np.abs(x - xpos)))
        ax.axvline(xpos, color=PALETTE.highlight, linestyle="--", linewidth=0.85, alpha=0.9)
        ax.plot(x[nearest], y[nearest], marker="o", color=PALETTE.highlight, markersize=3.2, zorder=3)

    ax.set_title(PANEL_LABELS.get(panel_key, panel_key), pad=1.0)
    ax.set_xlim(-35, 1035)
    ax.set_ylim(0.045, 0.31)
    ax.set_xticks([0, 400, 800])
    ax.set_ylabel("RMSE" if show_ylabel else "", labelpad=0.8)
    ax.tick_params(axis="both", pad=0.8)
    style_axes(ax, grid=True)


def main() -> None:
    apply_style("compact")
    data = aggregate_nonlinear_step_grid()
    panel_order = ["poly2", "poly3", "sin", "tanh"]

    fig, axes = plt.subplots(1, 4, figsize=(6.95, 1.25), sharex=True, sharey=True)
    for idx, (ax, panel_key) in enumerate(zip(axes, panel_order, strict=True)):
        _plot_panel(ax, data, panel_key, show_ylabel=idx == 0)

    fig.text(0.54, 0.025, "iteration", ha="center", va="bottom", color=PALETTE.border)
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.27, top=0.79, wspace=0.18)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{ext}", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
