from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from viz.aggregate import aggregate_nonlinear_step_grid
from viz.style import PALETTE, apply_style, method_style, style_axes


OUT_DIR = ROOT / "papers" / "tsp" / "figures"
OUT_STEM = "controlled_nonlinear_step_strip"

PANEL_LABELS = {
    "poly2": r"$s^2$",
    "poly3": r"$s^2+s^3$",
    "tanh": r"$\tanh(\kappa s)$",
    "exp": r"$(e^{\kappa s}-1)/\kappa$",
}


def _plot_panel(
    ax: plt.Axes,
    data,
    panel_key: str,
    *,
    show_ylabel: bool,
    show_xlabel: bool,
    y_limits: tuple[float, float],
    y_ticks: list[float],
) -> None:
    panel = data.loc[data["panel"].eq(panel_key)].copy()
    curve = panel.loc[panel["kind"].eq("curve")].sort_values("x")
    transitions = panel.loc[panel["kind"].eq("transition")].sort_values("x")

    x = curve["x"].to_numpy(dtype=float)
    y = curve["mean"].to_numpy(dtype=float)
    std = curve["std"].fillna(0).to_numpy(dtype=float)

    style = method_style("NTD-PL")
    ax.fill_between(x, y - std, y + std, color=style["color"], alpha=0.10, linewidth=0, zorder=1)
    ax.plot(
        x,
        y,
        color=style["color"],
        linestyle=style["linestyle"],
        marker=None,
        linewidth=1.85,
        zorder=4,
    )

    for transition_idx, item in enumerate(transitions.itertuples(index=False)):
        xpos = float(item.x)
        nearest = int(np.argmin(np.abs(x - xpos)))
        label = "degree activation" if transition_idx == 0 else None
        ax.axvline(
            xpos,
            color=PALETTE.highlight,
            linestyle="--",
            linewidth=1.0,
            alpha=0.9,
            zorder=2,
            label=label,
        )
        ax.plot(x[nearest], y[nearest], marker="o", color=PALETTE.highlight, markersize=3.7, zorder=5)

    ax.set_title(PANEL_LABELS.get(panel_key, panel_key), pad=2.0)
    ax.set_xlim(-35, 1035)
    ax.set_ylim(*y_limits)
    ax.set_yticks(y_ticks)
    ax.set_xticks([0, 400, 800])
    ax.set_xlabel("iteration" if show_xlabel else "", labelpad=1.5)
    ax.set_ylabel("RMSE" if show_ylabel else "", labelpad=1.5)
    ax.tick_params(axis="both", pad=1.0)
    style_axes(ax, grid=True)


def main() -> None:
    apply_style("single_column")
    data = aggregate_nonlinear_step_grid()
    panel_order = ["poly2", "poly3", "tanh", "exp"]
    missing = [panel for panel in panel_order if panel not in set(data["panel"])]
    if missing:
        raise RuntimeError(f"Missing controlled nonlinear panels: {missing}")
    curve = data.loc[data["kind"].eq("curve")].copy()
    y_min = float((curve["mean"] - curve["std"].fillna(0)).min())
    y_max = float((curve["mean"] + curve["std"].fillna(0)).max())
    y_min = max(0.0, y_min - 0.01)
    y_max = y_max + 0.01
    tick_step = 0.1
    y_ticks = np.arange(0.0, np.ceil(y_max / tick_step) * tick_step + 1e-9, tick_step).tolist()
    y_limits = (0.0, y_ticks[-1] if y_ticks else y_max)

    fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.72), sharex=True, sharey=True)
    flat_axes = axes.ravel()
    for idx, (ax, panel_key) in enumerate(zip(flat_axes, panel_order, strict=True)):
        _plot_panel(
            ax,
            data,
            panel_key,
            show_ylabel=idx == 0,
            show_xlabel=True,
            y_limits=y_limits,
            y_ticks=y_ticks,
        )

    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.22, top=0.76, wspace=0.14)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{ext}", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
