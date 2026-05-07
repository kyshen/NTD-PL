from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from viz.style import PALETTE, apply_style, style_axes


MAIN_RECON = ROOT / "experiment/outputs/cave-representation/recon_summary.csv"
LOWRANK_RECON = ROOT / "neurips/tables/cave_reconstruction_lowrank.summary.csv"
OUT_DIR = ROOT / "neurips/figures"
OUT_STEM = "cave_reconstruction_capacity"


def _parse_mean(value: object) -> float:
    return float(str(value).split("+-", 1)[0].strip())


def _normalize_rank(value: object) -> str:
    return str(value).strip().strip('"').replace(" ", "")


def _load_points() -> pd.DataFrame:
    main = pd.read_csv(MAIN_RECON)
    main = pd.DataFrame(
        {
            "rank": main["Rank"].map(_normalize_rank),
            "method": main["Method"],
            "params": main["Params"].astype(float),
            "rmse": main["RMSE"].map(_parse_mean),
            "sam": main["SAM(deg)"].map(_parse_mean),
        }
    )

    low = pd.read_csv(LOWRANK_RECON)
    low = pd.DataFrame(
        {
            "rank": low["rank"].map(_normalize_rank),
            "method": low["method"],
            "params": low["params"].astype(float),
            "rmse": low["RMSE_mean"].astype(float),
            "sam": low["SAM_mean"].astype(float),
        }
    )

    frame = pd.concat([low, main], ignore_index=True)
    frame["params_k"] = frame["params"] / 1000.0
    method_order = {"Tucker": 0, "NTD-PL": 1}
    frame["method_order"] = frame["method"].map(method_order)
    return frame.sort_values(["method_order", "params_k"]).reset_index(drop=True)


def _paired_points(frame: pd.DataFrame) -> pd.DataFrame:
    wide = frame.pivot(index="rank", columns="method", values=["params_k", "rmse", "sam"])
    rows = []
    for rank in frame.sort_values("params_k")["rank"].drop_duplicates():
        tucker = wide.loc[rank, ("params_k", "Tucker")]
        ntdpl = wide.loc[rank, ("params_k", "NTD-PL")]
        rows.append(
            {
                "rank": rank,
                "x": 0.5 * (float(tucker) + float(ntdpl)),
                "rmse_tucker": float(wide.loc[rank, ("rmse", "Tucker")]),
                "rmse_ntdpl": float(wide.loc[rank, ("rmse", "NTD-PL")]),
                "sam_tucker": float(wide.loc[rank, ("sam", "Tucker")]),
                "sam_ntdpl": float(wide.loc[rank, ("sam", "NTD-PL")]),
            }
        )
    return pd.DataFrame(rows)


def _annotate_gains(ax: plt.Axes, pairs: pd.DataFrame, metric: str) -> None:
    for _, row in pairs.iterrows():
        tucker = float(row[f"{metric}_tucker"])
        ntdpl = float(row[f"{metric}_ntdpl"])
        gain = 100.0 * (tucker - ntdpl) / tucker
        y = 0.5 * (tucker + ntdpl)
        ax.annotate(
            rf"$\downarrow${gain:.1f}%",
            xy=(float(row["x"]), y),
            xytext=(4.0, 0.0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=7.2,
            color=PALETTE.ntdpl,
            bbox={"boxstyle": "round,pad=0.12", "facecolor": PALETTE.white, "edgecolor": "none", "alpha": 0.88},
            zorder=5,
        )


def _plot_metric(ax: plt.Axes, frame: pd.DataFrame, metric: str, ylabel: str) -> None:
    styles = {
        "Tucker": {
            "color": PALETTE.tucker,
            "marker": "s",
            "linestyle": "--",
            "linewidth": 1.45,
            "markersize": 3.9,
            "label": "Tucker",
            "alpha": 0.82,
        },
        "NTD-PL": {
            "color": PALETTE.ntdpl,
            "marker": "o",
            "linestyle": "-",
            "linewidth": 2.15,
            "markersize": 5.0,
            "label": "NTD-PL",
            "alpha": 0.98,
        },
    }
    for method, style in styles.items():
        sub = frame.loc[frame["method"].eq(method)].sort_values("params_k")
        ax.plot(
            sub["params_k"],
            sub[metric],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            markersize=style["markersize"],
            label=style["label"],
            alpha=style["alpha"],
        )

    ax.set_xlabel("Parameters (k)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(17.0, 52.0)
    style_axes(ax, grid=True)


def main() -> None:
    apply_style("double_column")
    frame = _load_points()

    fig, axes = plt.subplots(1, 2, figsize=(6.95, 2.25))
    _plot_metric(axes[0], frame, "rmse", "RMSE")
    _plot_metric(axes[1], frame, "sam", "SAM")
    axes[0].set_ylim(0.018, 0.039)
    axes[1].set_ylim(11.0, 23.5)
    pairs = _paired_points(frame)
    _annotate_gains(axes[0], pairs, "rmse")
    _annotate_gains(axes[1], pairs, "sam")
    axes[0].legend(loc="upper right", frameon=False, handlelength=1.8)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.23, top=0.96, wspace=0.24)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{ext}", dpi=600)
    plt.close(fig)
    print(f"Wrote {OUT_DIR / (OUT_STEM + '.pdf')} and .png")


if __name__ == "__main__":
    main()
