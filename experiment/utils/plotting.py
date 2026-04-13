from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from cycler import cycler
import matplotlib as mpl
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


_AUTO_BBOX = object()


@dataclass(frozen=True)
class Palette:
    blue: str = "#2D6EA8"
    orange: str = "#C98A2E"
    green: str = "#2E8B57"
    rose: str = "#B65C3A"
    gray: str = "#6E7681"
    black: str = "#111111"
    lightgray: str = "#D7DCE2"
    grid: str = "#D7DCE2"
    border: str = "#30343A"
    tucker: str = "#7A838E"
    ntdpl: str = "#2D6EA8"
    ground_truth: str = "#111111"
    baseline_1: str = "#C98A2E"
    baseline_2: str = "#2E8B57"
    baseline_3: str = "#B65C3A"
    highlight: str = "#E9B949"
    error_low: str = "#FBE8A6"
    error_high: str = "#1E102B"

    @property
    def cycle(self) -> list[str]:
        return [self.ntdpl, self.baseline_1, self.baseline_2, self.baseline_3, self.tucker, self.black]


PALETTE = Palette()


@dataclass(frozen=True)
class PlotTheme:
    font_family: str = "STIXGeneral"
    math_fontset: str = "stix"
    font_size: float = 10.0
    axes_labelsize: float = 10.2
    axes_titlesize: float = 10.8
    xtick_labelsize: float = 9.1
    ytick_labelsize: float = 9.1
    legend_fontsize: float = 8.9
    legend_title_fontsize: float = 8.9
    grid_alpha: float = 0.40
    grid_linewidth: float = 0.55
    spine_linewidth: float = 0.8
    tick_length: float = 3.2
    tick_width: float = 0.8
    panel_box_aspect: float = 0.68
    savefig_bbox: str | None = "tight"
    savefig_pad_inches: float = 0.02


@dataclass(frozen=True)
class FigureSizePresets:
    single_panel: tuple[float, float] = (4.1, 3.0)
    single_metric: tuple[float, float] = (6.4, 4.1)
    two_panel: tuple[float, float] = (7.1, 3.55)
    four_panel: tuple[float, float] = (7.1, 5.55)


PLOT_THEME = PlotTheme()
FIGURE_SIZES = FigureSizePresets()

METHOD_STYLES = {
    "NTD-PL": {"color": PALETTE.ntdpl, "linestyle": "-", "marker": "o", "linewidth": 2.3, "markersize": 5.4},
    "Tucker": {"color": PALETTE.tucker, "linestyle": "--", "marker": "s", "linewidth": 1.8, "markersize": 4.9},
    "CP": {"color": PALETTE.baseline_1, "linestyle": "-.", "marker": "D", "linewidth": 1.6, "markersize": 4.7},
    "TT": {"color": PALETTE.baseline_2, "linestyle": ":", "marker": "^", "linewidth": 1.7, "markersize": 4.8},
    "TR": {"color": PALETTE.baseline_3, "linestyle": (0, (4, 1.3)), "marker": "x", "linewidth": 1.7, "markersize": 5.0},
    "Ground truth": {"color": PALETTE.ground_truth, "linestyle": "-", "marker": None, "linewidth": 2.2, "markersize": 0.0},
}


def apply_theme(theme: PlotTheme = PLOT_THEME) -> None:
    mpl.rcdefaults()
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": PALETTE.border,
            "axes.labelcolor": PALETTE.black,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": PALETTE.grid,
            "grid.alpha": theme.grid_alpha,
            "grid.linewidth": theme.grid_linewidth,
            "axes.prop_cycle": cycler(color=PALETTE.cycle),
            "font.family": theme.font_family,
            "mathtext.fontset": theme.math_fontset,
            "font.size": theme.font_size,
            "axes.labelsize": theme.axes_labelsize,
            "axes.titlesize": theme.axes_titlesize,
            "xtick.labelsize": theme.xtick_labelsize,
            "ytick.labelsize": theme.ytick_labelsize,
            "legend.frameon": False,
            "legend.fontsize": theme.legend_fontsize,
            "legend.title_fontsize": theme.legend_title_fontsize,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "savefig.bbox": theme.savefig_bbox,
            "savefig.pad_inches": theme.savefig_pad_inches,
        }
    )


def style_axes(ax: Axes, grid: bool = True) -> None:
    ax.tick_params(length=PLOT_THEME.tick_length, width=PLOT_THEME.tick_width, colors=PALETTE.border)
    ax.spines["left"].set_linewidth(PLOT_THEME.spine_linewidth)
    ax.spines["bottom"].set_linewidth(PLOT_THEME.spine_linewidth)
    ax.spines["left"].set_color(PALETTE.border)
    ax.spines["bottom"].set_color(PALETTE.border)
    ax.grid(grid, axis="y")
    ax.set_axisbelow(True)


def method_style(label: str) -> dict[str, object]:
    style = METHOD_STYLES.get(label, {"color": PALETTE.black, "linestyle": "-", "marker": None, "linewidth": 1.6})
    return dict(style)


def legend_style(
    handles: Sequence[Line2D] | None = None,
    labels: Sequence[str] | None = None,
    *,
    ncols: int = 1,
    loc: str = "upper center",
    bbox_to_anchor: tuple[float, float] | None = None,
) -> dict[str, object]:
    options: dict[str, object] = {
        "ncols": ncols,
        "loc": loc,
        "columnspacing": 1.0,
        "handlelength": 2.0,
        "handletextpad": 0.55,
        "borderaxespad": 0.25,
    }
    if handles is not None:
        options["handles"] = handles
    if labels is not None:
        options["labels"] = labels
    if bbox_to_anchor is not None:
        options["bbox_to_anchor"] = bbox_to_anchor
    return options


def style_colorbar(colorbar: Colorbar, *, label: str | None = None) -> None:
    colorbar.outline.set_linewidth(0.8)
    colorbar.outline.set_edgecolor(PALETTE.border)
    colorbar.ax.tick_params(length=2.8, width=0.7, labelsize=8.2, color=PALETTE.border)
    if label:
        colorbar.set_label(label, fontsize=8.6)


def paired_comparison_panel(
    ax: Axes,
    *,
    left_values: Iterable[float],
    right_values: Iterable[float],
    left_label: str,
    right_label: str,
    ylabel: str,
    title: str | None = None,
    lower_is_better: bool = True,
    show_mean: bool = True,
    tie_tol: float = 1e-12,
) -> dict[str, float]:
    left = np.asarray(list(left_values), dtype=float)
    right = np.asarray(list(right_values), dtype=float)
    if left.size == 0 or right.size == 0:
        raise ValueError("Paired comparison requires non-empty left/right values.")
    if left.shape != right.shape:
        raise ValueError("Paired comparison requires left/right arrays with the same shape.")

    right_gain = left - right if lower_is_better else right - left
    right_better = right_gain > tie_tol
    left_better = right_gain < -tie_tol

    left_style = method_style(left_label)
    right_style = method_style(right_label)
    x_positions = np.asarray([0.0, 1.0], dtype=float)

    for left_value, right_value, right_win, left_win in zip(left, right, right_better, left_better, strict=False):
        if right_win:
            color = PALETTE.ntdpl
        elif left_win:
            color = PALETTE.baseline_3
        else:
            color = PALETTE.grid
        ax.plot(
            x_positions,
            [left_value, right_value],
            color=color,
            linewidth=1.45,
            alpha=0.85,
            zorder=2,
        )

    ax.scatter(
        np.zeros_like(left),
        left,
        color=str(left_style["color"]),
        marker=str(left_style.get("marker") or "o"),
        s=34,
        linewidths=0.0,
        alpha=0.95,
        zorder=3,
    )
    ax.scatter(
        np.ones_like(right),
        right,
        color=str(right_style["color"]),
        marker=str(right_style.get("marker") or "o"),
        s=38,
        linewidths=0.0,
        alpha=0.95,
        zorder=3,
    )

    if show_mean:
        ax.plot(
            x_positions,
            [float(left.mean()), float(right.mean())],
            color=PALETTE.black,
            linewidth=2.8,
            marker="o",
            markersize=6.5,
            zorder=4,
        )

    all_values = np.concatenate([left, right], axis=0)
    span = float(all_values.max() - all_values.min())
    pad = max(0.08 * span, max(abs(float(all_values.max())), 1.0) * 0.03, 1e-6)
    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(float(all_values.min()) - pad, float(all_values.max()) + pad)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([left_label, right_label])
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    style_axes(ax, grid=True)

    return {
        "count": int(left.size),
        "right_better": int(np.sum(right_better)),
        "ties": int(np.sum(~(right_better | left_better))),
        "left_better": int(np.sum(left_better)),
        "mean_gap": float(np.mean(right_gain)),
        "median_gap": float(np.median(right_gain)),
    }


def rounded_bounds(values: Iterable[float], step: float) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    lo = np.floor(array.min() / step) * step
    hi = np.ceil(array.max() / step) * step
    if np.isclose(lo, hi):
        lo -= step
        hi += step
    return lo, hi


def save_figure(
    fig: Figure,
    destination: str | Path,
    formats: Iterable[str] = ("pdf", "png"),
    dpi: int = 300,
    *,
    bbox_inches: str | None | object = _AUTO_BBOX,
) -> None:
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    export_formats = [fmt.lower() for fmt in formats]
    if not export_formats:
        export_formats = ["pdf"]
    for fmt in export_formats:
        export_dpi = None if fmt == "pdf" else dpi
        save_kwargs: dict[str, object] = {"dpi": export_dpi}
        if bbox_inches is None:
            with mpl.rc_context({"savefig.bbox": None}):
                fig.savefig(target.with_suffix(f".{fmt}"), **save_kwargs)
            continue
        if bbox_inches is not _AUTO_BBOX:
            save_kwargs["bbox_inches"] = bbox_inches
        fig.savefig(target.with_suffix(f".{fmt}"), **save_kwargs)
