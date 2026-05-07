from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from viz.aggregate import aggregate_nonlinear_alpha_grid
from viz.style import PALETTE, apply_style, method_style, style_axes


OUT_DIR = ROOT / "neurips" / "figures"
OUT_STEM = "controlled_nonlinear_alpha_grid"

PANEL_LABELS = {
    "poly3": r"$s^2+s^3$",
    "exp": r"$(e^{\kappa s}-1)/\kappa$",
    "tanh": r"$\tanh(\kappa s)$",
}

METHOD_ORDER = ("Tucker", "CP", "TT", "NTD-PL")


def _compact_method_style(method: str) -> dict:
    style = method_style(method)
    style.update({"linewidth": 1.25, "markersize": 3.0})
    if method == "NTD-PL":
        style.update({"linewidth": 1.85, "markersize": 3.5, "zorder": 4})
    elif method == "Tucker":
        style.update({"linewidth": 1.45, "zorder": 3})
    else:
        style.update({"alpha": 0.88, "zorder": 2})
    return style


def _plot_panel(
    ax: plt.Axes,
    panel_data,
    panel_key: str,
    *,
    show_ylabel: bool,
    show_xlabel: bool,
    y_limits: tuple[float, float],
    y_ticks: list[float],
) -> None:
    for method in METHOD_ORDER:
        sub = panel_data.loc[panel_data["method"].eq(method)].sort_values("x")
        if sub.empty:
            continue

        style = _compact_method_style(method)
        x = sub["x"].to_numpy(dtype=float)
        y = sub["mean"].to_numpy(dtype=float)

        if method == "NTD-PL":
            ax.fill_between(
                x,
                sub["band_lower"].to_numpy(dtype=float),
                sub["band_upper"].to_numpy(dtype=float),
                color=style["color"],
                alpha=0.10,
                linewidth=0,
                zorder=1,
            )

        ax.plot(
            x,
            y,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style.get("marker"),
            linewidth=style["linewidth"],
            markersize=style.get("markersize", 2.5),
            alpha=style.get("alpha", 1.0),
            zorder=style.get("zorder", 2),
            label=method,
        )

    ax.set_title(PANEL_LABELS.get(panel_key, panel_key), pad=2.0)
    ax.set_xticks([0.10, 0.20, 0.30, 0.40])
    ax.set_xticklabels([".10", ".20", ".30", ".40"])
    ax.set_ylim(*y_limits)
    ax.set_yticks(y_ticks)
    ax.set_xlabel(r"residual energy $\alpha$" if show_xlabel else "", labelpad=1.5)
    ax.set_ylabel("RMSE" if show_ylabel else "", labelpad=1.5)
    ax.tick_params(axis="both", pad=1.0)
    style_axes(ax, grid=True)


def main() -> None:
    apply_style("single_column")
    data = aggregate_nonlinear_alpha_grid()
    data = data.loc[data["method"].isin(METHOD_ORDER)].copy()
    panel_order = ["poly3", "tanh", "exp"]
    y_min = float(data["band_lower"].min())
    y_max = float(data["band_upper"].max())
    y_min = max(0.0, y_min - 0.01)
    y_max = y_max + 0.01
    tick_step = 0.1
    y_ticks = np.arange(0.0, np.ceil(y_max / tick_step) * tick_step + 1e-9, tick_step).tolist()
    y_limits = (0.0, y_ticks[-1] if y_ticks else y_max)

    fig, axes = plt.subplots(1, 3, figsize=(5.48, 1.58), sharex=True, sharey=True)
    flat_axes = axes.ravel()
    for idx, (ax, panel_key) in enumerate(zip(flat_axes, panel_order, strict=True)):
        panel_data = data.loc[data["panel"].eq(panel_key)].copy()
        _plot_panel(
            ax,
            panel_data,
            panel_key,
            show_ylabel=idx == 0,
            show_xlabel=True,
            y_limits=y_limits,
            y_ticks=y_ticks,
        )

    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.015),
        handlelength=1.5,
        columnspacing=0.65,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.25, top=0.76, wspace=0.12)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{ext}", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
