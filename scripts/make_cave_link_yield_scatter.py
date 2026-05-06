from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.utils.plotting import PALETTE, apply_theme, save_figure, style_axes


INPUT_CSV = ROOT / "experiment" / "outputs" / "cave-random-completion" / "polycal_pairwise_scene_gains.csv"
OUT_STEM = ROOT / "neurips" / "figures" / "cave_link_yield_scatter"
TARGET_MISSING_RATE = 0.5


def _residual_link_score(rmse_tucker: pd.Series, rmse_polycal: pd.Series) -> pd.Series:
    ratio = (rmse_polycal.astype(float) ** 2) / (rmse_tucker.astype(float).clip(lower=1e-12) ** 2)
    ratio = ratio.clip(lower=1e-12)
    return 10.0 * np.log10(1.0 / ratio)


def _load() -> pd.DataFrame:
    frame = pd.read_csv(INPUT_CSV)
    frame = frame.loc[np.isclose(frame["missing_rate"].astype(float), TARGET_MISSING_RATE)].copy()
    frame["link_score_db"] = _residual_link_score(frame["RMSE_tucker"], frame["RMSE_polycal"])
    frame["ntdpl_gain_pct"] = 100.0 * (
        frame["RMSE_tucker"].astype(float) - frame["RMSE_ntdpl"].astype(float)
    ) / frame["RMSE_tucker"].astype(float).clip(lower=1e-12)
    frame["scene_label"] = frame["scene_id"].map(lambda v: f"S{int(v):02d}")
    return frame.sort_values("link_score_db").reset_index(drop=True)


def main() -> None:
    apply_theme()
    frame = _load()
    if frame.empty:
        raise RuntimeError("No CAVE link-yield rows found.")

    x = frame["link_score_db"].to_numpy(dtype=float)
    y = frame["ntdpl_gain_pct"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, deg=1)
    x_line = np.linspace(float(x.min()), float(x.max()), 200)
    y_line = slope * x_line + intercept

    fig, ax = plt.subplots(1, 1, figsize=(4.4, 3.0))
    ax.scatter(x, y, color=PALETTE.ntdpl, s=28, alpha=0.92, zorder=3)
    ax.plot(x_line, y_line, color=PALETTE.tucker, linewidth=1.6, linestyle="--", zorder=2)
    for row in frame.itertuples(index=False):
        ax.text(
            float(row.link_score_db) + 0.01,
            float(row.ntdpl_gain_pct) + 0.15,
            str(row.scene_label),
            fontsize=6.7,
            color=PALETTE.border,
        )

    ax.axhline(0.0, color=PALETTE.border, linewidth=0.8, linestyle="--", alpha=0.8)
    ax.set_xlabel(r"Residual-link score $S_{\mathrm{link}}$ (dB)")
    ax.set_ylabel("NTD-PL RMSE gain (%)")
    style_axes(ax, grid=True)
    fig.subplots_adjust(left=0.15, right=0.99, bottom=0.19, top=0.97)
    save_figure(fig, OUT_STEM, formats=("pdf", "png"), dpi=400)
    plt.close(fig)
    print(f"Wrote {OUT_STEM.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
