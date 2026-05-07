from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.utils.plotting import save_figure


INPUT_CSV = ROOT / "experiment" / "outputs" / "cave-random-completion" / "polycal_pairwise_scene_gains.csv"
OUT_STEM = ROOT / "neurips" / "figures" / "cave_link_yield_scatter"
TARGET_MISSING_RATES = (0.3, 0.5)


def _residual_link_score(rmse_tucker: pd.Series, rmse_polycal: pd.Series) -> pd.Series:
    ratio = (rmse_polycal.astype(float) ** 2) / (rmse_tucker.astype(float).clip(lower=1e-12) ** 2)
    ratio = ratio.clip(lower=1e-12)
    return 10.0 * np.log10(1.0 / ratio)


def _load() -> pd.DataFrame:
    frame = pd.read_csv(INPUT_CSV)
    frame = frame.loc[frame["missing_rate"].astype(float).isin(TARGET_MISSING_RATES)].copy()
    frame["link_score_db"] = _residual_link_score(frame["RMSE_tucker"], frame["RMSE_polycal"])
    frame["ntdpl_gain_pct"] = 100.0 * (
        frame["RMSE_tucker"].astype(float) - frame["RMSE_ntdpl"].astype(float)
    ) / frame["RMSE_tucker"].astype(float).clip(lower=1e-12)
    frame["scene_label"] = frame["scene_id"].map(lambda v: f"S{int(v):02d}")
    return frame.sort_values(["missing_rate", "link_score_db"]).reset_index(drop=True)


def _plot_panel(ax: plt.Axes, frame: pd.DataFrame, missing_rate: float) -> float:
    panel = frame.loc[np.isclose(frame["missing_rate"].astype(float), missing_rate)].copy()
    if panel.empty:
        raise RuntimeError(f"No rows found for missing rate {missing_rate}.")

    x = panel["link_score_db"].to_numpy(dtype=float)
    y = panel["ntdpl_gain_pct"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, deg=1)
    x_line = np.linspace(float(x.min()), float(x.max()), 200)
    y_line = slope * x_line + intercept
    corr = spearmanr(x, y)

    ax.scatter(
        x,
        y,
        color="#4C78A8",
        edgecolors="white",
        linewidths=0.35,
        s=30,
        alpha=0.96,
        zorder=5,
    )
    ax.plot(x_line, y_line, color="#4C78A8", linewidth=1.7, linestyle="--", zorder=4)
    label_offsets = [(-0.025, 0.18), (0.025, 0.18), (-0.025, -0.22), (0.025, -0.22)]
    for idx, row in enumerate(panel.itertuples(index=False)):
        dx, dy = label_offsets[idx % len(label_offsets)]
        ax.text(
            float(row.link_score_db) + dx,
            float(row.ntdpl_gain_pct) + dy,
            str(row.scene_label),
            fontsize=6.8,
            color="#333333",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=0.2),
        )

    ax.axhline(0.0, color="#888888", linewidth=0.9, linestyle="--", zorder=2)
    ax.grid(True, color="#dddddd", linewidth=0.5, alpha=0.7)
    ax.set_xlabel(r"$D_{\mathrm{link}}$ (dB)")
    ax.set_ylabel("NTD-PL RMSE$^\\ast$ gain (%)")
    ax.text(
        0.02,
        0.96,
        fr"$\rho_s$={float(corr.statistic):.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.4,
        fontweight="semibold",
        color="#111111",
        bbox={
            "boxstyle": "round,pad=0.16",
            "facecolor": "white",
            "edgecolor": "#bbbbbb",
            "linewidth": 0.4,
            "alpha": 0.88,
        },
    )
    return float(corr.statistic)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.6,
            "axes.labelsize": 10.2,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    frame = _load()
    if frame.empty:
        raise RuntimeError("No CAVE link-yield rows found.")

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 2.85), sharey=True, constrained_layout=False)
    for ax, missing_rate in zip(axes, TARGET_MISSING_RATES, strict=True):
        _plot_panel(ax, frame, missing_rate)

    axes[1].set_ylabel("")
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.20, top=0.985, wspace=0.20)
    save_figure(fig, OUT_STEM, formats=("pdf", "png"), dpi=400)
    plt.close(fig)
    print(f"Wrote {OUT_STEM.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
