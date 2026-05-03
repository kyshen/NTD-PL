from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from viz.aggregate import aggregate_nonlinear_alpha_grid
from viz.style import PALETTE, apply_style, method_style, style_axes


OUT_DIR = ROOT / "neurips" / "figures"
OUT_STEM = "controlled_nonlinear_alpha_grid"

PANEL_LABELS = {
    "poly2": r"$s^2$",
    "poly3": r"$s^2+s^3$",
    "sin": r"$\sin(\kappa s)$",
    "tanh": r"$\tanh(\kappa s)$",
}

METHOD_ORDER = ("Tucker", "CP", "TT", "TR", "NTD-PL")


def _compact_method_style(method: str) -> dict:
    style = method_style(method)
    style.update({"linewidth": 1.05, "markersize": 2.5})
    if method == "NTD-PL":
        style.update({"linewidth": 1.55, "markersize": 3.0, "zorder": 4})
    elif method == "Tucker":
        style.update({"linewidth": 1.25, "zorder": 3})
    else:
        style.update({"alpha": 0.88, "zorder": 2})
    return style


def _plot_panel(ax: plt.Axes, panel_data, panel_key: str, *, show_ylabel: bool) -> None:
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

    ax.set_title(PANEL_LABELS.get(panel_key, panel_key), pad=1.0)
    ax.set_xticks([0.10, 0.20, 0.30, 0.40])
    ax.set_xticklabels([".10", ".20", ".30", ".40"])
    ax.set_ylim(-33, -2)
    ax.set_yticks([-30, -20, -10])
    ax.set_xlabel("")
    ax.set_ylabel("NMSE (dB)" if show_ylabel else "", labelpad=0.8)
    ax.tick_params(axis="both", pad=0.8)
    style_axes(ax, grid=True)


def main() -> None:
    apply_style("compact")
    data = aggregate_nonlinear_alpha_grid()
    panel_order = ["poly2", "poly3", "sin", "tanh"]

    fig, axes = plt.subplots(1, 4, figsize=(6.95, 1.42), sharex=True, sharey=True)
    for idx, (ax, panel_key) in enumerate(zip(axes, panel_order, strict=True)):
        panel_data = data.loc[data["panel"].eq(panel_key)].copy()
        _plot_panel(ax, panel_data, panel_key, show_ylabel=idx == 0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 1.04),
        handlelength=1.5,
        columnspacing=0.8,
        borderaxespad=0.0,
    )
    fig.text(0.54, 0.025, r"nonlinear residual energy $\alpha$", ha="center", va="bottom", color=PALETTE.border)
    fig.subplots_adjust(left=0.062, right=0.995, bottom=0.26, top=0.74, wspace=0.18)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{ext}", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
