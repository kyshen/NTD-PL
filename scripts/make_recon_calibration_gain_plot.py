from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from viz.style import PALETTE, apply_style, style_axes


SUMMARY_PATH = ROOT / "neurips" / "tables" / "recon_calibration_baselines.summary.csv"
OUT_DIR = ROOT / "neurips" / "figures"
OUT_STEM = "recon_calibration_gain_plot"

METHOD_LABELS = {
    "polycal": "+PolyCal",
    "mlpcal": "+MLPCal",
    "ntdpl": "NTD-PL",
}
METHOD_COLORS = {
    "polycal": PALETTE.highlight,
    "mlpcal": PALETTE.tt,
    "ntdpl": PALETTE.ntdpl,
}


def _panel(ax, values: np.ndarray, methods: list[str], title: str, xmax: float) -> None:
    y = np.arange(len(methods))[::-1]
    colors = [METHOD_COLORS[method] for method in methods]
    ax.barh(y, values, height=0.48, color=colors, alpha=0.86)
    for yi, value in zip(y, values, strict=True):
        ax.text(
            value + 0.35,
            yi,
            f"{value:.1f}",
            va="center",
            ha="left",
            fontsize=6.5,
            color=PALETTE.border,
        )
    ax.axvline(0.0, color=PALETTE.border, linewidth=0.7)
    ax.set_xlim(0.0, xmax)
    ax.set_xticks([0, 5, 10, 15])
    ax.set_title(title, loc="left", pad=2.0, fontweight="bold")
    style_axes(ax, grid=False)
    ax.xaxis.grid(True, color=PALETTE.grid, linewidth=0.55, alpha=0.7)
    ax.yaxis.grid(False)
    ax.tick_params(axis="y", length=0)


def main() -> None:
    apply_style("compact")
    summary = pd.read_csv(SUMMARY_PATH)
    methods = ["polycal", "mlpcal", "ntdpl"]
    data = summary.set_index("method").loc[methods]
    rmse_gain = data["RMSE_gain_pct"].to_numpy(dtype=float)
    sam_gain = data["SAM_gain_pct"].to_numpy(dtype=float)
    labels = [METHOD_LABELS[method] for method in methods]
    y = np.arange(len(methods))[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(5.15, 1.42), sharey=True)
    _panel(axes[0], rmse_gain, methods, "RMSE gain", 16.0)
    _panel(axes[1], sam_gain, methods, "SAM gain", 14.0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[1].tick_params(axis="y", labelleft=False)
    fig.supxlabel("Gain over Tucker (%)", y=0.03, fontsize=7.4)
    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.30, top=0.82, wspace=0.22)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{ext}", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
