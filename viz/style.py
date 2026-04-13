from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cycler import cycler
import matplotlib as mpl

try:
    import scienceplots  # noqa: F401

    _HAS_SCIENCEPLOTS = True
except ModuleNotFoundError:
    _HAS_SCIENCEPLOTS = False


@dataclass(frozen=True)
class Palette:
    ntdpl: str = "#1E5A8A"
    tucker: str = "#6C757D"
    cp: str = "#B35C1E"
    tt: str = "#2E8B57"
    tr: str = "#8A3B3B"
    highlight: str = "#C97C1A"
    neutral: str = "#AEB5BD"
    grid: str = "#D9DEE4"
    border: str = "#24313C"
    text: str = "#111111"
    white: str = "#FFFFFF"
    heat_low: str = "#F3F5F7"
    heat_mid: str = "#E7B66D"
    heat_high: str = "#A63B2D"

    @property
    def cycle(self) -> list[str]:
        return [self.ntdpl, self.tucker, self.cp, self.tt, self.tr]


PALETTE = Palette()


METHOD_STYLES: dict[str, dict[str, Any]] = {
    "NTD-PL": {"color": PALETTE.ntdpl, "linestyle": "-", "marker": "o", "linewidth": 2.2, "markersize": 4.8},
    "Tucker": {"color": PALETTE.tucker, "linestyle": "--", "marker": "s", "linewidth": 1.8, "markersize": 4.4},
    "Tucker + PolyCal": {"color": PALETTE.highlight, "linestyle": "-.", "marker": "D", "linewidth": 1.9, "markersize": 4.6},
    "CP": {"color": PALETTE.cp, "linestyle": "-.", "marker": "D", "linewidth": 1.6, "markersize": 4.2},
    "TT": {"color": PALETTE.tt, "linestyle": ":", "marker": "^", "linewidth": 1.6, "markersize": 4.4},
    "TR": {"color": PALETTE.tr, "linestyle": (0, (4, 1.2)), "marker": "x", "linewidth": 1.6, "markersize": 4.6},
    "Ground truth": {"color": PALETTE.text, "linestyle": "-", "marker": None, "linewidth": 2.2, "markersize": 0.0},
}


@dataclass(frozen=True)
class RolePreset:
    width: float
    font_size: float
    label_size: float
    title_size: float
    tick_size: float
    legend_size: float
    line_width: float
    marker_size: float
    colorbar_tick_size: float


ROLE_PRESETS: dict[str, RolePreset] = {
    "single_column": RolePreset(3.35, 8.7, 8.7, 9.2, 8.0, 7.8, 1.4, 4.0, 7.4),
    "double_column": RolePreset(6.95, 9.2, 9.2, 9.8, 8.4, 8.0, 1.6, 4.4, 7.8),
    "appendix_wide": RolePreset(7.25, 9.0, 9.0, 9.6, 8.3, 7.9, 1.5, 4.2, 7.6),
    "compact": RolePreset(2.45, 7.8, 7.8, 8.4, 7.2, 7.0, 1.2, 3.2, 7.0),
}


def method_style(label: str) -> dict[str, Any]:
    return dict(METHOD_STYLES.get(label, {"color": PALETTE.text, "linestyle": "-", "marker": None, "linewidth": 1.4}))


def role_preset(role: str) -> RolePreset:
    return ROLE_PRESETS[role]


def apply_style(role: str) -> RolePreset:
    preset = role_preset(role)
    mpl.rcdefaults()
    base_style = ["science", "no-latex"] if _HAS_SCIENCEPLOTS else []
    if base_style:
        mpl.style.use(base_style)
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": preset.font_size,
            "axes.labelsize": preset.label_size,
            "axes.titlesize": preset.title_size,
            "xtick.labelsize": preset.tick_size,
            "ytick.labelsize": preset.tick_size,
            "legend.fontsize": preset.legend_size,
            "legend.frameon": False,
            "legend.handlelength": 2.2,
            "legend.handletextpad": 0.5,
            "axes.prop_cycle": cycler(color=PALETTE.cycle),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": PALETTE.border,
            "axes.labelcolor": PALETTE.text,
            "xtick.color": PALETTE.border,
            "ytick.color": PALETTE.border,
            "grid.color": PALETTE.grid,
            "grid.linewidth": 0.55,
            "grid.alpha": 0.55,
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )
    return preset


def style_axes(ax: Any, *, grid: bool = True) -> None:
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_color(PALETTE.border)
    ax.spines["bottom"].set_color(PALETTE.border)
    ax.tick_params(length=3.0, width=0.8, colors=PALETTE.border)
    if grid:
        ax.yaxis.grid(True, color=PALETTE.grid, linewidth=0.55, alpha=0.7)
        ax.xaxis.grid(False)
    else:
        ax.grid(False)
    ax.set_axisbelow(True)


def style_colorbar(colorbar: Any, *, label: str | None = None, tick_size: float = 8.0) -> None:
    colorbar.outline.set_linewidth(0.8)
    colorbar.outline.set_edgecolor(PALETTE.border)
    colorbar.ax.tick_params(length=2.6, width=0.7, labelsize=tick_size, color=PALETTE.border)
    if label:
        colorbar.set_label(label, fontsize=tick_size + 0.2, labelpad=4.0)
