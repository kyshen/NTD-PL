from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from viz.style import PALETTE, apply_style, style_axes


SUMMARY_PATH = ROOT / "neurips" / "tables" / "cave_tucker_rank_sweep.summary.csv"
OUT_DIR = ROOT / "neurips" / "figures"
OUT_STEM = "cave_tucker_rank_sweep_compact"

NTDPL_REFS = [
    {"rank": r"NTD-PL $(18,18,3)$", "k": 3, "x0": 19, "x1": 30, "rmse": 0.0319, "sam": 19.23},
    {"rank": r"NTD-PL $(24,24,4)$", "k": 4, "x0": 24, "x1": 38, "rmse": 0.0256, "sam": 15.15},
    {"rank": r"NTD-PL $(33,33,4)$", "k": 4, "x0": 38, "x1": 50, "rmse": 0.0223, "sam": 14.41},
]


def _plot_metric(ax, data: pd.DataFrame, metric: str, ylabel: str) -> None:
    styles = {
        3: {"color": PALETTE.highlight, "marker": "o", "label": r"Tucker $r_3=3$"},
        4: {"color": PALETTE.ntdpl, "marker": "s", "label": r"Tucker $r_3=4$"},
    }
    for k, style in styles.items():
        sub = data.loc[data["rank_r3"].eq(k)].sort_values("rank_r1")
        ax.plot(
            sub["rank_r1"],
            sub[metric],
            color=style["color"],
            marker=style["marker"],
            markersize=3.2,
            linewidth=1.25,
            alpha=0.92,
            label=style["label"],
        )

    ref_key = "rmse" if metric == "RMSE_mean" else "sam"
    for ref in NTDPL_REFS:
        ax.hlines(
            ref[ref_key],
            ref["x0"],
            ref["x1"],
            colors=PALETTE.border,
            linestyles=(0, (3, 2)),
            linewidth=0.8,
            alpha=0.72,
        )

    ax.set_xlim(18.4, 50.6)
    ax.set_xticks([20, 30, 40, 50])
    ax.set_xlabel("Spatial Tucker rank")
    ax.set_ylabel(ylabel)
    style_axes(ax, grid=True)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, color=PALETTE.grid, linewidth=0.55, alpha=0.7)


def main() -> None:
    apply_style("compact")
    data = pd.read_csv(SUMMARY_PATH)
    data = data.loc[data["rank_r3"].isin([3, 4])].copy()

    fig, axes = plt.subplots(1, 2, figsize=(5.25, 1.62))
    _plot_metric(axes[0], data, "RMSE_mean", "RMSE")
    _plot_metric(axes[1], data, "SAM_mean", "SAM")
    axes[0].set_ylim(0.021, 0.0375)
    axes[1].set_ylim(14.0, 22.8)
    axes[0].legend(loc="upper right", frameon=False, handlelength=1.4)
    axes[1].text(
        0.98,
        0.08,
        "dashed: NTD-PL refs",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=6.4,
        color=PALETTE.border,
    )
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.30, top=0.94, wspace=0.26)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{ext}", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
