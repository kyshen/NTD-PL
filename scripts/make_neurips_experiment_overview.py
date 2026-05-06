from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "neurips" / "figures"


COLORS = {
    "tucker": "#5B6472",
    "polycal": "#9AA3AE",
    "ntdpl": "#2F6FDB",
    "accent": "#D26A3A",
    "green": "#2B8A6E",
    "grid": "#D9DEE7",
}


def _save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.png", dpi=240, bbox_inches="tight")


def _mechanism_panel(ax: plt.Axes) -> None:
    data = pd.read_csv(PROJECT_ROOT / "experiment/outputs/cave-random-completion/mechanism_closure_main_figure_data.csv")
    panel = data.loc[data["panel"].eq("C")].copy()
    order = ["Tucker", "Tucker + PolyCal", "NTD-PL"]
    panel["method"] = pd.Categorical(panel["method"], categories=order, ordered=True)
    panel = panel.sort_values("method")
    x = np.arange(len(panel))
    colors = [COLORS["tucker"], COLORS["polycal"], COLORS["ntdpl"]]
    ax.bar(x, panel["mean"], yerr=panel["std"], color=colors, width=0.66, capsize=3, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x, ["Tucker", "+PolyCal", "NTD-PL"])
    ax.set_ylabel("Missing RMSE")
    ax.set_title("A. Joint link beats frozen-backbone beta refresh", loc="left", fontweight="bold")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    tucker = float(panel.loc[panel["method"].eq("Tucker"), "mean"].iloc[0])
    ntdpl = float(panel.loc[panel["method"].eq("NTD-PL"), "mean"].iloc[0])
    gain = 100.0 * (tucker - ntdpl) / tucker
    ax.text(2, ntdpl * 1.08, f"{gain:.1f}% lower", ha="center", va="bottom", color=COLORS["ntdpl"], fontsize=8)


def _rank_panel(ax: plt.Axes) -> None:
    sweep = pd.read_csv(PROJECT_ROOT / "neurips/tables/cave_tucker_rank_sweep.summary.csv")
    ax.plot(
        sweep["params"] / 1000.0,
        sweep["RMSE_mean"],
        marker="o",
        markersize=4,
        color=COLORS["tucker"],
        linewidth=1.6,
        label="Tucker sweep",
    )
    ntdpl_points = pd.DataFrame(
        [
            {"label": "NTD-PL (24,24,4)", "params": 27011, "rmse": 0.0256},
            {"label": "NTD-PL (33,33,4)", "params": 38279, "rmse": 0.022338953541078576},
        ]
    )
    ax.scatter(ntdpl_points["params"] / 1000.0, ntdpl_points["rmse"], s=58, color=COLORS["ntdpl"], zorder=5, label="NTD-PL")
    for row in ntdpl_points.itertuples(index=False):
        ax.annotate(row.label.replace("NTD-PL ", ""), (row.params / 1000.0, row.rmse), xytext=(5, 5), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Parameters (k)")
    ax.set_ylabel("Reconstruction RMSE")
    ax.set_title("B. Gains are not a small rank increase", loc="left", fontweight="bold")
    ax.grid(axis="both", color=COLORS["grid"], linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper right")


def _stress_panel(ax: plt.Axes) -> None:
    block = pd.read_csv(PROJECT_ROOT / "neurips/tables/cave_structured_missing_lowrank_block.summary.csv")
    kodak = pd.read_csv(PROJECT_ROOT / "neurips/tables/kodak_completion_lowrank_p3.summary.csv")
    t_block = float(block.loc[block["method"].eq("tucker"), "RMSE_missing_mean"].iloc[0])
    n_block = float(block.loc[block["method"].eq("ntdpl"), "RMSE_missing_mean"].iloc[0])
    t_kodak = float(kodak.loc[kodak["method"].eq("tucker"), "RMSE_missing_mean"].iloc[0])
    n_kodak = float(kodak.loc[kodak["method"].eq("ntdpl"), "RMSE_missing_mean"].iloc[0])
    gains = [100.0 * (t_block - n_block) / t_block, 100.0 * (t_kodak - n_kodak) / t_kodak]
    wins = ["14/15 RMSE\n15/15 SAM", "22/24 RMSE\n23/24 SSIM"]
    x = np.arange(2)
    ax.bar(x, gains, color=[COLORS["green"], COLORS["accent"]], width=0.58, edgecolor="white", linewidth=0.8)
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_xticks(x, ["CAVE block\nmissing", "Kodak RGB\ncompletion"])
    ax.set_ylabel("Relative RMSE gain (%)")
    ax.set_title("C. Low-rank stress and non-HSI transfer", loc="left", fontweight="bold")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    for idx, (gain, win) in enumerate(zip(gains, wins, strict=False)):
        ax.text(idx, gain + 0.35, f"{gain:.1f}%\n{win}", ha="center", va="bottom", fontsize=7)


def _diagnostic_panel(ax: plt.Axes) -> None:
    data = pd.read_csv(PROJECT_ROOT / "experiment/outputs/cave-random-completion/polycal_pairwise_scene_gains.csv")
    panel = data.loc[data["missing_rate"].eq(0.5)].copy()
    panel["d_beta"] = panel["polycal_gain_rmse"] / panel["RMSE_tucker"].clip(lower=1e-12)
    panel["ntdpl_gain"] = (panel["RMSE_tucker"] - panel["RMSE_ntdpl"]) / panel["RMSE_tucker"].clip(lower=1e-12)
    x = panel["d_beta"].to_numpy() * 100.0
    y = panel["ntdpl_gain"].to_numpy() * 100.0
    ax.scatter(x, y, s=34, color=COLORS["ntdpl"], alpha=0.88, edgecolor="white", linewidth=0.5)
    coef = np.polyfit(x, y, deg=1)
    grid = np.linspace(float(np.min(x)), float(np.max(x)), 100)
    ax.plot(grid, coef[0] * grid + coef[1], color=COLORS["accent"], linewidth=1.5)
    ax.set_xlabel(r"Fixed-backbone score $D_{\beta}$ (%)")
    ax.set_ylabel("NTD-PL RMSE gain (%)")
    ax.set_title("D. Cheap diagnostic predicts where NTD-PL helps", loc="left", fontweight="bold")
    ax.text(0.05, 0.92, "Spearman = 0.67", transform=ax.transAxes, fontsize=8, color="#333333")
    ax.grid(axis="both", color=COLORS["grid"], linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.95), constrained_layout=True)
    _mechanism_panel(axs[0, 0])
    _rank_panel(axs[0, 1])
    _stress_panel(axs[1, 0])
    _diagnostic_panel(axs[1, 1])
    _save(fig, "experiment_overview")
    print(f"Wrote {OUT_DIR / 'experiment_overview.pdf'} and .png")


if __name__ == "__main__":
    main()
