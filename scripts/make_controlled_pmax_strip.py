from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from viz.aggregate import aggregate_nonlinear_pmax_grid
from viz.style import PALETTE, apply_style, method_style, style_axes


OUT_DIR = ROOT / "neurips" / "figures"
OUT_STEM = "controlled_nonlinear_pmax_strip"

PANEL_LABELS = {
    "poly2": r"$s^2$",
    "poly3": r"$s^2+s^3$",
    "sin": r"$\sin(\kappa s)$",
    "tanh": r"$\tanh(\kappa s)$",
}


def _plot_panel(ax: plt.Axes, panel_data, panel_key: str, *, show_ylabel: bool) -> None:
    tucker_style = method_style("Tucker")
    tucker_style.update({"marker": None, "linewidth": 1.25, "linestyle": "--", "color": PALETTE.tucker})
    ntdpl_style = method_style("NTD-PL")
    ntdpl_style.update({"linewidth": 1.55, "markersize": 3.0})

    for method, style in (("Tucker", tucker_style), ("NTD-PL", ntdpl_style)):
        sub = panel_data.loc[panel_data["method"].eq(method)].sort_values("x")
        if sub.empty:
            continue
        x = sub["x"].to_numpy(dtype=float)
        y = sub["mean"].to_numpy(dtype=float)
        lower = sub["band_lower"].to_numpy(dtype=float)
        upper = sub["band_upper"].to_numpy(dtype=float)
        if method == "NTD-PL":
            ax.fill_between(x, lower, upper, color=style["color"], alpha=0.10, linewidth=0)
        ax.plot(
            x,
            y,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style.get("marker"),
            linewidth=style["linewidth"],
            markersize=style.get("markersize", 3.0),
            label=method,
        )

    ax.set_title(PANEL_LABELS.get(panel_key, panel_key), pad=1.0)
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.set_ylim(-30, -8)
    ax.set_yticks([-30, -25, -20, -15, -10])
    ax.set_xlabel("")
    ax.set_ylabel("NMSE (dB)" if show_ylabel else "", labelpad=0.8)
    ax.tick_params(axis="both", pad=0.8)
    style_axes(ax, grid=True)


def main() -> None:
    apply_style("compact")
    data = aggregate_nonlinear_pmax_grid()
    panel_order = ["poly2", "poly3", "sin", "tanh"]

    fig, axes = plt.subplots(1, 4, figsize=(6.95, 1.22), sharex=True, sharey=True)
    for idx, (ax, panel_key) in enumerate(zip(axes, panel_order, strict=True)):
        panel_data = data.loc[data["panel"].eq(panel_key)].copy()
        _plot_panel(ax, panel_data, panel_key, show_ylabel=idx == 0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.045),
        handlelength=1.8,
        columnspacing=1.2,
        borderaxespad=0.0,
    )
    fig.text(0.54, 0.025, r"degree $P$", ha="center", va="bottom", color=PALETTE.border)
    fig.subplots_adjust(left=0.062, right=0.995, bottom=0.27, top=0.75, wspace=0.18)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{ext}", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
