from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from run_cave_latent_rank_budget_probe import (
    PROJECT_ROOT,
    _plot,
    _spatial_budget,
    _summarize_fit,
    _summarize_spectra,
)


DEFAULT_INPUTS = (
    PROJECT_ROOT / "artifacts" / "results" / "cave_latent_rank_budget_probe_pilot",
    PROJECT_ROOT / "artifacts" / "results" / "cave_latent_rank_budget_probe_remaining",
)
DEFAULT_OUTDIR = PROJECT_ROOT / "artifacts" / "results" / "cave_latent_rank_budget_probe_full"


def _load_inputs(paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    spectra = [pd.read_csv(path / "spectrum_metrics.csv") for path in paths]
    fits = [pd.read_csv(path / "fit_metrics.csv") for path in paths]
    spectrum = pd.concat(spectra, ignore_index=True).drop_duplicates(
        subset=["scene_id", "rank", "mode", "tensor_kind"], keep="last"
    )
    fit = pd.concat(fits, ignore_index=True).drop_duplicates(
        subset=["scene_id", "rank", "method"], keep="last"
    )
    return spectrum, fit


def _scene_trajectories(spectrum: pd.DataFrame) -> pd.DataFrame:
    frame = spectrum[
        spectrum["tensor_kind"].eq("ntdpl_signal") & spectrum["mode"].isin([1, 2])
    ].copy()
    frame["spatial_rank_budget"] = frame["rank"].map(_spatial_budget)
    return (
        frame.groupby(["scene_id", "scene_name", "spatial_rank_budget"], as_index=False)
        .agg(
            rank95=("rank_energy_950", "mean"),
            rank99=("rank_energy_990", "mean"),
            rank995=("rank_energy_995", "mean"),
            rank999=("rank_energy_999", "mean"),
        )
        .sort_values(["scene_id", "spatial_rank_budget"])
    )


def _scene_gains(fit: pd.DataFrame) -> pd.DataFrame:
    frame = fit.copy()
    frame["spatial_rank_budget"] = frame["rank"].map(_spatial_budget)
    wide = frame.pivot(
        index=["scene_id", "scene_name", "spatial_rank_budget"], columns="method", values="RMSE"
    ).reset_index()
    wide["NTDPL_gain_pct"] = 100.0 * (wide["Tucker"] - wide["NTD-PL"]) / wide["Tucker"]
    return wide


def _convergence_summary(trajectories: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    budgets = set(trajectories["spatial_rank_budget"].unique())
    intervals = [(24, 40)]
    if {40, 64}.issubset(budgets):
        intervals.append((40, 64))
    for metric in ["rank95", "rank99", "rank995", "rank999"]:
        wide = trajectories.pivot(index=["scene_id", "scene_name"], columns="spatial_rank_budget", values=metric)
        for low, high in intervals:
            delta = wide[high] - wide[low]
            rows.append(
                {
                    "metric": metric,
                    "lower_budget": low,
                    "upper_budget": high,
                    "median_rank_at_upper": float(wide[high].median()),
                    "median_growth": float(delta.median()),
                    "q25_growth": float(delta.quantile(0.25)),
                    "q75_growth": float(delta.quantile(0.75)),
                    "scenes_growth_le_1": int((delta <= 1.0).sum()),
                    "scenes_growth_le_2": int((delta <= 2.0).sum()),
                    "scenes": int(delta.size),
                }
            )
    return pd.DataFrame(rows)


def _correlations(trajectories: pd.DataFrame, gains: pd.DataFrame) -> pd.DataFrame:
    high_budget = int(trajectories["spatial_rank_budget"].max())
    lower_budget = 40 if high_budget >= 64 else 24
    rank_high = trajectories[trajectories["spatial_rank_budget"].eq(high_budget)].copy()
    rank_low = trajectories[trajectories["spatial_rank_budget"].eq(lower_budget)].copy()
    growth = rank_high[["scene_id", "rank95", "rank99", "rank995", "rank999"]].merge(
        rank_low[["scene_id", "rank95", "rank99", "rank995", "rank999"]],
        on="scene_id",
        suffixes=(f"_{high_budget}", f"_{lower_budget}"),
    )
    for metric in ["rank95", "rank99", "rank995", "rank999"]:
        growth[f"{metric}_growth"] = growth[f"{metric}_{high_budget}"] - growth[f"{metric}_{lower_budget}"]
    gain12 = gains[gains["spatial_rank_budget"].eq(12)][["scene_id", "NTDPL_gain_pct"]].rename(
        columns={"NTDPL_gain_pct": "gain12"}
    )
    gain_high = gains[gains["spatial_rank_budget"].eq(high_budget)][["scene_id", "NTDPL_gain_pct"]].rename(
        columns={"NTDPL_gain_pct": f"gain{high_budget}"}
    )
    merged = growth.merge(gain12, on="scene_id").merge(gain_high, on="scene_id")
    rows: list[dict] = []
    features = [
        f"rank95_{high_budget}",
        f"rank99_{high_budget}",
        f"rank999_{high_budget}",
        "rank95_growth",
        "rank99_growth",
        "rank999_growth",
    ]
    for target in ["gain12", f"gain{high_budget}"]:
        for feature in features:
            corr = spearmanr(merged[feature], merged[target])
            rows.append(
                {
                    "target": target,
                    "feature": feature,
                    "spearman_r": float(corr.statistic),
                    "p_value": float(corr.pvalue),
                }
            )
    return pd.DataFrame(rows).sort_values(["target", "spearman_r"], ascending=[True, False])


def _plot_scene_trajectories(trajectories: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), sharex=True)
    for ax, metric, title in zip(axes, ["rank99", "rank999"], ["99% energy rank", "99.9% energy rank"]):
        for _, group in trajectories.groupby("scene_id"):
            ax.plot(group["spatial_rank_budget"], group[metric], color="#7A8796", alpha=0.28, linewidth=0.9)
        median = trajectories.groupby("spatial_rank_budget", as_index=False)[metric].median()
        ax.plot(median["spatial_rank_budget"], median[metric], color="#2F6FBB", marker="o", linewidth=2.1, label="median")
        budgets = sorted(trajectories["spatial_rank_budget"].unique())
        ax.plot(budgets, budgets, color="#999999", linestyle="--", linewidth=1.0, label="rank budget")
        ax.set_title(title)
        ax.set_xlabel("spatial Tucker rank budget")
        ax.set_ylabel("scene-mean spatial energy rank")
        ax.set_xticks(budgets)
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "latent_rank_scene_trajectories.pdf", bbox_inches="tight")
    fig.savefig(outdir / "latent_rank_scene_trajectories.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_rank_components(summary: pd.DataFrame, outdir: Path) -> None:
    labels = {
        "ntdpl_signal": "NTD-PL signal",
        "ntdpl_nonlinear_response_component": "nonlinear response",
        "ntdpl_prediction": "NTD-PL prediction",
        "tucker_fit": "Tucker fit",
        "measured": "measured tensor",
    }
    colors = {
        "ntdpl_signal": "#2F6FBB",
        "ntdpl_nonlinear_response_component": "#C44E52",
        "ntdpl_prediction": "#D9822B",
        "tucker_fit": "#2A9D8F",
        "measured": "#333333",
    }
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.25))
    for ax, threshold in zip(axes, ["99%", "99.9%"]):
        block = summary[summary["energy_threshold"].eq(threshold)]
        for kind in labels:
            group = block[block["tensor_kind"].eq(kind)].sort_values("spatial_rank_budget")
            style = "--" if kind == "measured" else "-"
            marker = None if kind == "measured" else "o"
            ax.plot(
                group["spatial_rank_budget"],
                group["median_rank"],
                color=colors[kind],
                linestyle=style,
                marker=marker,
                linewidth=1.8,
                label=labels[kind],
            )
        ax.set_title(f"{threshold} spatial energy rank")
        ax.set_xlabel("spatial Tucker rank budget")
        ax.set_ylabel("median energy rank")
        ax.set_xticks(sorted(block["spatial_rank_budget"].unique()))
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(outdir / "response_and_measured_rank.pdf", bbox_inches="tight")
    fig.savefig(outdir / "response_and_measured_rank.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge and summarize CAVE latent-rank budget probe batches.")
    parser.add_argument("--input-dir", action="append", type=Path, dest="input_dirs")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()
    paths = args.input_dirs or list(DEFAULT_INPUTS)
    paths = [path if path.is_absolute() else PROJECT_ROOT / path for path in paths]
    outdir = args.outdir if args.outdir.is_absolute() else PROJECT_ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    spectrum, fit = _load_inputs(paths)
    summary = _summarize_spectra(spectrum)
    fit_summary = _summarize_fit(fit)
    trajectories = _scene_trajectories(spectrum)
    gains = _scene_gains(fit)
    convergence = _convergence_summary(trajectories)
    correlations = _correlations(trajectories, gains)

    spectrum.to_csv(outdir / "spectrum_metrics.csv", index=False)
    fit.to_csv(outdir / "fit_metrics.csv", index=False)
    summary.to_csv(outdir / "energy_rank_summary.csv", index=False)
    fit_summary.to_csv(outdir / "fit_summary.csv", index=False)
    trajectories.to_csv(outdir / "scene_trajectories.csv", index=False)
    gains.to_csv(outdir / "scene_gains.csv", index=False)
    convergence.to_csv(outdir / "convergence_summary.csv", index=False)
    correlations.to_csv(outdir / "correlations.csv", index=False)
    _plot(summary, outdir)
    _plot_scene_trajectories(trajectories, outdir)
    _plot_rank_components(summary, outdir)

    print(summary[summary["tensor_kind"].eq("ntdpl_signal")].to_string(index=False))
    print("\nConvergence summary:")
    print(convergence.to_string(index=False))
    print("\nFit summary:")
    print(fit_summary.to_string(index=False))
    print("\nTop correlations:")
    print(correlations.groupby("target", as_index=False).head(4).to_string(index=False))
    print(f"\nOutput: {outdir}")


if __name__ == "__main__":
    main()
