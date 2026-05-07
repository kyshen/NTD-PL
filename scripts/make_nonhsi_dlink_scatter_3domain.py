from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_STEM = PROJECT_ROOT / "neurips" / "figures" / "nonhsi_dlink_scatter_3domain"

SOURCES = [
    PROJECT_ROOT / "results" / "nonhsi_dlink_diagnostic" / "20260507_000000_full68" / "per_unit_results.csv",
    PROJECT_ROOT / "results" / "phase1_crossdomain_dlink" / "20260507_120000_kth_full240" / "per_unit_results.csv",
    PROJECT_ROOT / "results" / "phase1_crossdomain_dlink" / "20260507_123000_ucf101_full240" / "per_unit_results.csv",
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
    "CBSD68": "#4C78A8",
    "CIFAR-10": "#4C78A8",
    "COIL-100": "#59A14F",
    "smallNORB": "#59A14F",
    "KTH-Action": "#F28E2B",
    "UCF101": "#F28E2B",
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
    ax.plot(xs, slope * xs + intercept, color=color, linewidth=1.5, linestyle="--", zorder=4)


def main() -> None:
    frame = load_frame()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 17,
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.2), constrained_layout=True)

    for index, (domain, dataset) in enumerate(PANELS):
        row, col = divmod(index, 3)
        ax = axes[row, col]
        panel = frame.loc[frame["dataset"].eq(dataset)].copy()
        color = COLORS[dataset]

        ax.scatter(
            panel["d_link_db"],
            panel["rmse_gain_pct"],
            s=28,
            marker=MARKERS[dataset],
            color=color,
            edgecolors="white",
            linewidths=0.35,
            zorder=5,
            label=dataset,
        )
        draw_fit(ax, panel, color)
        ax.axhline(0.0, color="#888888", linewidth=0.9, linestyle="--", zorder=2)
        ax.grid(True, color="#dddddd", linewidth=0.5, alpha=0.7)
        ax.set_xlabel(r"$D_{\mathrm{link}}$ (dB)" if row == 1 else "")
        ax.set_ylabel("RMSE gain (%)" if col == 0 else "")
        if row == 0:
            ax.set_title(domain, fontweight="bold", pad=10)
        ax.text(
            0.02,
            0.96,
            f"{dataset}, {spearman_label(panel)}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            color="#333333",
        )

    OUT_STEM.parent.mkdir(parents=True, exist_ok=True)
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
