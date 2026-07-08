from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "papers" / "neurips" / "figures"

COLORS = {
    "(12,12,3)": "#6C757D",
    "(16,16,3)": "#1E5A8A",
    "(20,20,4)": "#2E8B57",
    "(24,24,4)": "#C97C1A",
    "grid": "#D9DEE4",
    "border": "#24313C",
}


def _save(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=240, bbox_inches="tight")


def _plot_panel(ax: plt.Axes, data: pd.DataFrame, protocol: str, metric: str, title: str, ylabel: str) -> None:
    panel = data.loc[data["protocol"].eq(protocol)].copy()
    for rank, sub in panel.groupby("rank", sort=False):
        sub = sub.sort_values("missing_rate")
        ax.plot(
            sub["missing_rate"],
            sub[f"{metric}_gain_median_pct"],
            marker="o",
            linewidth=1.8,
            markersize=4.2,
            color=COLORS.get(str(rank), COLORS["border"]),
            label=str(rank),
        )
    ax.axhline(0.0, color=COLORS["border"], linewidth=0.8)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel(r"Missing rate $\rho$")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(panel["missing_rate"].unique()))
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CAVE low-rank core gains across missing rates.")
    parser.add_argument("--paired-csv", default="papers/neurips/tables/lowrank_core_cave_rates.paired.csv")
    parser.add_argument("--stem", default="lowrank_core_cave_rates")
    args = parser.parse_args()

    data = pd.read_csv(PROJECT_ROOT / args.paired_csv)
    data = data.loc[data["method"].eq("ntdpl")].copy()
    rank_order = ["(12,12,3)", "(16,16,3)", "(20,20,4)", "(24,24,4)"]
    data["rank"] = pd.Categorical(data["rank"], categories=rank_order, ordered=True)
    data = data.sort_values(["protocol", "rank", "missing_rate"])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.25), constrained_layout=True, sharex=True)
    _plot_panel(axes[0, 0], data, "random", "RMSE_missing", "A. Random masks: RMSE", "Median RMSE gain (%)")
    _plot_panel(axes[0, 1], data, "random", "SAM_missing", "B. Random masks: SAM", "Median SAM gain (%)")
    _plot_panel(axes[1, 0], data, "block", "RMSE_missing", "C. Block masks: RMSE", "Median RMSE gain (%)")
    _plot_panel(axes[1, 1], data, "block", "SAM_missing", "D. Block masks: SAM", "Median SAM gain (%)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=4, frameon=False, bbox_to_anchor=(0.5, 1.04))
    _save(fig, args.stem)
    print(f"Wrote {OUT_DIR / (args.stem + '.pdf')} and .png")


if __name__ == "__main__":
    main()
