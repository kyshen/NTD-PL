from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from viz.style import apply_style, method_style, style_axes


MAIN_RECON = ROOT / "papers/supplementary/processed_results/experiment_outputs/cave-representation/recon_summary.csv"
LOWRANK_RECON = ROOT / "papers/supplementary/processed_results/tables/cave_reconstruction_lowrank.summary.csv"
OUT_DIR = ROOT / "papers/tsp/figures"
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


def _plot_metric(ax: plt.Axes, frame: pd.DataFrame, metric: str, ylabel: str) -> None:
    for method in ("Tucker", "NTD-PL"):
        style = method_style(method)
        style.update({"linewidth": 1.45, "markersize": 3.1})
        if method == "NTD-PL":
            style.update({"linewidth": 1.85, "markersize": 3.6, "zorder": 4})
        else:
            style.update({"zorder": 3})
        sub = frame.loc[frame["method"].eq(method)].sort_values("params_k")
        ax.plot(
            sub["params_k"],
            sub[metric],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            markersize=style["markersize"],
            label=method,
            zorder=style["zorder"],
        )

    ax.set_xlabel("Parameters (k)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(17.0, 52.0)
    ax.set_xticks([20, 30, 40, 50])
    ax.tick_params(axis="both", pad=1.0)
    style_axes(ax, grid=True)


def main() -> None:
    apply_style("single_column")
    frame = _load_points()

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 1.58))
    _plot_metric(axes[0], frame, "rmse", "RMSE")
    _plot_metric(axes[1], frame, "sam", "SAM")
    axes[0].set_ylim(0.018, 0.039)
    axes[1].set_ylim(11.0, 23.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.015),
        handlelength=1.5,
        columnspacing=0.85,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.25, top=0.76, wspace=0.18)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{ext}", dpi=600)
    plt.close(fig)
    print(f"Wrote {OUT_DIR / (OUT_STEM + '.pdf')} and .png")


if __name__ == "__main__":
    main()
