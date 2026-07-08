from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "papers" / "neurips" / "figures"

COLORS = {
    "random": "#1E5A8A",
    "block": "#C97C1A",
    "dot": "#24313C",
    "grid": "#D9DEE4",
}


def _label(row: pd.Series) -> str:
    rank = str(row["rank"]).replace("(", "").replace(")", "")
    return f"{row['protocol']}\n{rank}"


def _save(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=240, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot paired low-rank core gains.")
    parser.add_argument("--paired-csv", default="papers/neurips/tables/lowrank_core_cave_seed0_mr05.paired.csv")
    parser.add_argument("--stem", default="lowrank_core_cave_seed0_mr05")
    args = parser.parse_args()

    paired = pd.read_csv(PROJECT_ROOT / args.paired_csv)
    paired = paired.loc[paired["method"].eq("ntdpl")].copy()
    order = {"random": 0, "block": 1}
    paired["protocol_order"] = paired["protocol"].map(order).fillna(99)
    paired = paired.sort_values(["protocol_order", "rank"]).reset_index(drop=True)
    x = np.arange(len(paired))
    colors = [COLORS.get(str(protocol), COLORS["dot"]) for protocol in paired["protocol"]]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55), constrained_layout=True)
    panels = [
        ("RMSE_missing", "Median RMSE gain (%)", "A. Missing-value accuracy"),
        ("SAM_missing", "Median SAM gain (%)", "B. Spectral angle"),
    ]
    for ax, (metric, ylabel, title) in zip(axes, panels, strict=True):
        med = paired[f"{metric}_gain_median_pct"].to_numpy(dtype=float)
        mean = paired[f"{metric}_gain_mean_pct"].to_numpy(dtype=float)
        ax.bar(x, med, color=colors, width=0.68, edgecolor="white", linewidth=0.8)
        ax.scatter(x, mean, s=26, color=COLORS["dot"], zorder=4, label="mean")
        ax.axhline(0.0, color="#222222", linewidth=0.8)
        ax.set_xticks(x, [_label(row) for _, row in paired.iterrows()])
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        for idx, row in paired.iterrows():
            wins = int(row[f"{metric}_wins"])
            n_pairs = int(row["n_pairs"])
            ypos = med[idx] + (1.2 if med[idx] >= 0 else -2.4)
            va = "bottom" if med[idx] >= 0 else "top"
            ax.text(idx, ypos, f"{wins}/{n_pairs}", ha="center", va=va, fontsize=7.2)
    axes[1].legend(frameon=False, loc="upper right")
    _save(fig, args.stem)
    print(f"Wrote {OUT_DIR / (args.stem + '.pdf')} and .png")


if __name__ == "__main__":
    main()
