from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from viz.style import PALETTE, apply_style, style_axes

OUT_STEM = PROJECT_ROOT / "papers" / "tsp" / "figures" / "nonhsi_dlink_scatter_3domain"

SOURCES = [
    PROJECT_ROOT / "artifacts" / "results" / "nonhsi_dlink_diagnostic" / "20260507_000000_full68" / "per_unit_results.csv",
    PROJECT_ROOT / "artifacts" / "results" / "phase1_crossdomain_dlink" / "20260507_120000_kth_full240" / "per_unit_results.csv",
    PROJECT_ROOT / "artifacts" / "results" / "phase1_crossdomain_dlink" / "20260507_123000_ucf101_full240" / "per_unit_results.csv",
]

PANELS = [
    ("Natural images", "CBSD68"),
    ("Object-view", "COIL-100"),
    ("Action video", "KTH-Action"),
    ("Natural images", "CIFAR-10"),
    ("Object-view", "smallNORB"),
    ("Action video", "UCF101"),
]

MARKERS = {
    "CBSD68": "o",
    "CIFAR-10": "s",
    "COIL-100": "o",
    "smallNORB": "s",
    "KTH-Action": "o",
    "UCF101": "s",
}

COLORS = {
    "CBSD68": PALETTE.ntdpl,
    "CIFAR-10": PALETTE.ntdpl,
    "COIL-100": PALETTE.tt,
    "smallNORB": PALETTE.tt,
    "KTH-Action": PALETTE.cp,
    "UCF101": PALETTE.cp,
}


def load_frame() -> pd.DataFrame:
    frames = []
    missing = []
    for path in SOURCES:
        if path.exists():
            frames.append(pd.read_csv(path))
        else:
            missing.append(path)
    if missing:
        joined = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing source CSV files:\n{joined}")
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.loc[frame["status"].eq("ok")].copy()
    frame["d_link_db"] = pd.to_numeric(frame["d_link_db"], errors="coerce")
    frame["rmse_gain_pct"] = pd.to_numeric(frame["rmse_gain_pct"], errors="coerce")
    return frame.dropna(subset=["d_link_db", "rmse_gain_pct"])


def spearman_label(panel: pd.DataFrame) -> str:
    if panel.shape[0] < 2 or panel["d_link_db"].nunique() < 2:
        return r"$\rho_s$=--"
    rho = spearmanr(panel["d_link_db"], panel["rmse_gain_pct"], nan_policy="omit").statistic
    return rf"$\rho_s$={rho:.2f}" if np.isfinite(rho) else r"$\rho_s$=--"


def draw_fit(ax: plt.Axes, panel: pd.DataFrame, color: str) -> None:
    if panel.shape[0] < 2 or panel["d_link_db"].nunique() < 2:
        return
    x = panel["d_link_db"].to_numpy(dtype=float)
    y = panel["rmse_gain_pct"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, deg=1)
    xs = np.linspace(float(x.min()), float(x.max()), 100)
    ax.plot(xs, slope * xs + intercept, color=color, linewidth=1.45, linestyle="--", zorder=4)


def main() -> None:
    frame = load_frame()
    apply_style("double_column")
    plt.rcParams.update(
        {
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(6.95, 2.55), sharex=False, constrained_layout=False)

    for index, (domain, dataset) in enumerate(PANELS):
        row, col = divmod(index, 3)
        ax = axes[row, col]
        panel = frame.loc[frame["dataset"].eq(dataset)].copy()
        color = COLORS[dataset]

        ax.scatter(
            panel["d_link_db"],
            panel["rmse_gain_pct"],
            s=14,
            marker=MARKERS[dataset],
            color=color,
            edgecolors="white",
            linewidths=0.35,
            zorder=5,
            label=dataset,
        )
        draw_fit(ax, panel, color)
        ax.axhline(0.0, color=PALETTE.border, linewidth=0.75, linestyle="--", alpha=0.65, zorder=2)
        style_axes(ax, grid=True)
        ax.xaxis.grid(True, color=PALETTE.grid, linewidth=0.55, alpha=0.50)
        ax.set_xlabel(r"$D_{\mathrm{link}}$ (dB)" if row == 1 else "")
        ax.set_ylabel("RMSE gain (%)" if col == 0 else "")
        if row == 0:
            ax.set_title(domain, fontweight="bold", pad=4)
        ax.text(
            0.035,
            0.95,
            f"{dataset}, {spearman_label(panel)}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.9,
            fontweight="semibold",
            color=PALETTE.text,
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": PALETTE.white,
                "edgecolor": PALETTE.grid,
                "linewidth": 0.4,
                "alpha": 0.88,
            },
            zorder=20,
        )

    OUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.070, right=0.995, bottom=0.17, top=0.88, wspace=0.25, hspace=0.32)
    fig.savefig(OUT_STEM.with_suffix(".pdf"))
    fig.savefig(OUT_STEM.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(OUT_STEM.with_suffix(".pdf"))
    print(OUT_STEM.with_suffix(".png"))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise
