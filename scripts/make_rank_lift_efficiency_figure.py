from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from viz.style import PALETTE, apply_style, style_axes


TUCKER_SWEEP = ROOT / "papers/neurips/tables/cave_tucker_rank_sweep.summary.csv"
CAVE_RECON = ROOT / "artifacts/paper-outputs/cave-representation/recon_summary.csv"
RANK_LIFT_COST = ROOT / "papers/neurips/tables/rank_lift_cost.csv"
OUT_DIR = ROOT / "papers/neurips/figures"
OUT_STEM = "rank_lift_efficiency"


def _parse_mean(value: str) -> float:
    return float(str(value).split("+-", 1)[0].strip())


def _normalize_rank_label(value: object) -> str:
    text = str(value).strip()
    if not text:
        return text
    text = text.strip('"')
    text = text.replace(" ", "")
    return text


def _load_ntdpl_points() -> pd.DataFrame:
    recon = pd.read_csv(CAVE_RECON)
    recon = recon.loc[recon["Method"].eq("NTD-PL")].copy()
    recon["Rank"] = recon["Rank"].map(_normalize_rank_label)
    recon["RMSE_mean"] = recon["RMSE"].map(_parse_mean)
    recon["SAM_mean"] = recon["SAM(deg)"].map(_parse_mean)
    recon["params_k"] = recon["Params"].astype(float) / 1000.0
    recon = recon.loc[:, ["Rank", "params_k", "RMSE_mean", "SAM_mean"]]

    rank_lift = pd.read_csv(RANK_LIFT_COST)
    rank_lift = rank_lift.rename(
        columns={
            "ntdpl_rank": "Rank",
            "ntdpl_params": "params",
            "ntdpl_rmse": "RMSE_mean",
            "ntdpl_sam": "SAM_mean",
        }
    )
    rank_lift["Rank"] = rank_lift["Rank"].map(_normalize_rank_label)
    rank_lift["params_k"] = rank_lift["params"].astype(float) / 1000.0
    rank_lift = rank_lift.loc[:, ["Rank", "params_k", "RMSE_mean", "SAM_mean"]]

    frame = pd.concat([recon, rank_lift], ignore_index=True)
    frame = frame.drop_duplicates(subset=["Rank"], keep="first")
    return frame.sort_values("params_k").reset_index(drop=True)


def _annotation_offset(rank: str, metric: str) -> tuple[float, float]:
    offsets = {
        ("(18,18,3)", "RMSE_mean"): (6, -10),
        ("(24,24,4)", "RMSE_mean"): (6, 6),
        ("(33,33,4)", "RMSE_mean"): (6, -10),
        ("(40,40,5)", "RMSE_mean"): (-36, 6),
        ("(18,18,3)", "SAM_mean"): (6, -10),
        ("(24,24,4)", "SAM_mean"): (6, 6),
        ("(33,33,4)", "SAM_mean"): (6, -10),
        ("(40,40,5)", "SAM_mean"): (-36, -10),
    }
    return offsets.get((rank, metric), (6, 6))


def _plot_metric(ax: plt.Axes, tucker: pd.DataFrame, ntdpl: pd.DataFrame, metric: str, ylabel: str) -> None:
    style_by_channel = {
        3: {"color": PALETTE.highlight, "marker": "o", "label": r"Tucker $r_3=3$"},
        4: {"color": PALETTE.tucker, "marker": "s", "label": r"Tucker $r_3=4$"},
        5: {"color": PALETTE.neutral, "marker": "^", "label": r"Tucker $r_3=5$"},
    }
    for channel_rank, style in style_by_channel.items():
        sub = tucker.loc[tucker["rank_r3"].eq(channel_rank)].sort_values("params")
        if sub.empty:
            continue
        ax.plot(
            sub["params"] / 1000.0,
            sub[metric],
            color=style["color"],
            marker=style["marker"],
            markersize=3.0,
            linewidth=1.2,
            alpha=0.92,
            label=style["label"],
        )

    ax.plot(
        ntdpl["params_k"],
        ntdpl[metric],
        color=PALETTE.ntdpl,
        linewidth=1.2,
        linestyle=(0, (2.2, 2.2)),
        alpha=0.9,
        zorder=4,
    )
    ax.scatter(
        ntdpl["params_k"],
        ntdpl[metric],
        color=PALETTE.ntdpl,
        marker="D",
        s=34,
        zorder=6,
        label="NTD-PL",
        edgecolor="white",
        linewidth=0.6,
    )
    for row in ntdpl.itertuples(index=False):
        label = str(row.Rank).replace(" ", "")
        dx, dy = _annotation_offset(label, metric)
        ax.annotate(
            label,
            (row.params_k, getattr(row, metric)),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=6.6,
            color=PALETTE.border,
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.85,
            },
        )

    ax.set_xlabel("Parameters (k)")
    ax.set_ylabel(ylabel)
    style_axes(ax, grid=True)
    ax.xaxis.grid(False)
    ax.set_xlim(18.0, 62.0)


def main() -> None:
    apply_style("single_column")
    tucker = pd.read_csv(TUCKER_SWEEP)
    tucker = tucker.loc[tucker["rank_r3"].isin([3, 4, 5])].copy()
    ntdpl = _load_ntdpl_points()

    fig, axes = plt.subplots(1, 2, figsize=(6.95, 2.35))
    _plot_metric(axes[0], tucker, ntdpl, "RMSE_mean", "RMSE")
    _plot_metric(axes[1], tucker, ntdpl, "SAM_mean", "SAM")
    axes[0].set_ylim(0.018, 0.038)
    axes[1].set_ylim(11.0, 23.0)
    axes[0].legend(loc="upper right", frameon=False, handlelength=1.7, ncol=1)
    axes[1].text(
        0.98,
        0.08,
        "diamonds + dashed line: NTD-PL backbone ranks",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=7.0,
        color=PALETTE.border,
    )
    fig.subplots_adjust(left=0.078, right=0.99, bottom=0.22, top=0.96, wspace=0.24)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{ext}", dpi=600)
    plt.close(fig)
    print(f"Wrote {OUT_DIR / (OUT_STEM + '.pdf')} and .png")


if __name__ == "__main__":
    main()
