from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "artifacts" / "results"


def _result_dir(results_root: Path, rank_slug: str, p_max: int, target_slug: str, suffix: str) -> Path:
    return results_root / f"rank_inflation_spectrum_{rank_slug}_p{p_max}_{target_slug}{suffix}"


def _fit_gain_frame(root: Path, p_max: int) -> pd.DataFrame:
    fit = pd.read_csv(root / "fit_metrics.csv")
    wide = fit.pivot(index=["scene_id", "scene_name"], columns="method", values=["RMSE", "SAM"]).reset_index()
    wide.columns = ["_".join(str(item) for item in col if item) for col in wide.columns]
    wide["p_max"] = int(p_max)
    wide["RMSE_gain_pct"] = 100.0 * (wide["RMSE_Tucker"] - wide["RMSE_NTD-PL"]) / wide["RMSE_Tucker"]
    wide["SAM_gain_pct"] = 100.0 * (wide["SAM_Tucker"] - wide["SAM_NTD-PL"]) / wide["SAM_Tucker"]
    return wide


def _scene_mechanism_frame(root: Path, p_max: int) -> pd.DataFrame:
    metrics = pd.read_csv(root / "spectrum_metrics.csv")
    response = metrics[
        metrics["tensor_kind"].eq("ntdpl_nonlinear_response_component")
        & metrics["mode"].isin([1, 2])
    ]
    response_scene = response.groupby(["scene_id", "scene_name"], as_index=False).agg(
        nonlinear_rank99_spatial=("rank_energy_990", "mean"),
        nonlinear_rank999_spatial=("rank_energy_999", "mean"),
        nonlinear_tail_spatial=("tail_energy_after_base_rank", "mean"),
        nonlinear_entropy_spatial=("entropy_rank", "mean"),
    )

    paired = pd.read_csv(root / "paired_rank_inflation.csv")
    paired = paired[paired["mode"].isin([1, 2])].copy()
    paired["pred_rank999_lift"] = (
        paired["rank_energy_999_ntdpl_prediction"]
        - paired["rank_energy_999_ntdpl_signal"]
    )
    paired_scene = paired.groupby(["scene_id", "scene_name"], as_index=False).agg(
        pred_rank999_lift_spatial=("pred_rank999_lift", "mean"),
        pred_tail_capture_spatial=("tail_capture_ratio_pred_vs_measured", "mean"),
    )

    gains = _fit_gain_frame(root, p_max)
    out = gains.merge(response_scene, on=["scene_id", "scene_name"], how="inner")
    out = out.merge(paired_scene, on=["scene_id", "scene_name"], how="inner")
    return out


def _degree_summary(root: Path, p_max: int) -> dict[str, float | int]:
    metrics = pd.read_csv(root / "spectrum_metrics.csv")
    fit = _fit_gain_frame(root, p_max)
    response = metrics[
        metrics["tensor_kind"].eq("ntdpl_nonlinear_response_component")
        & metrics["mode"].isin([1, 2])
    ]
    pred = metrics[
        metrics["tensor_kind"].eq("ntdpl_prediction")
        & metrics["mode"].isin([1, 2])
    ]
    signal = metrics[
        metrics["tensor_kind"].eq("ntdpl_signal")
        & metrics["mode"].isin([1, 2])
    ]
    measured = metrics[
        metrics["tensor_kind"].eq("measured")
        & metrics["mode"].isin([1, 2])
    ]
    paired = pd.read_csv(root / "paired_rank_inflation.csv")
    paired = paired[paired["mode"].isin([1, 2])].copy()
    return {
        "p_max": int(p_max),
        "rmse_gain_pct_mean": float(fit["RMSE_gain_pct"].mean()),
        "rmse_gain_pct_median": float(fit["RMSE_gain_pct"].median()),
        "rmse_wins": int((fit["RMSE_gain_pct"] > 0.0).sum()),
        "sam_gain_pct_mean": float(fit["SAM_gain_pct"].mean()),
        "sam_gain_pct_median": float(fit["SAM_gain_pct"].median()),
        "sam_wins": int((fit["SAM_gain_pct"] > 0.0).sum()),
        "nonlinear_rank99_mean_spatial": float(response["rank_energy_990"].mean()),
        "nonlinear_rank99_median_spatial": float(response["rank_energy_990"].median()),
        "nonlinear_rank999_mean_spatial": float(response["rank_energy_999"].mean()),
        "nonlinear_rank999_median_spatial": float(response["rank_energy_999"].median()),
        "nonlinear_tail_after_base_mean_spatial": float(response["tail_energy_after_base_rank"].mean()),
        "nonlinear_tail_after_base_median_spatial": float(response["tail_energy_after_base_rank"].median()),
        "prediction_rank999_lift_mean_spatial": float(
            (
                paired["rank_energy_999_ntdpl_prediction"]
                - paired["rank_energy_999_ntdpl_signal"]
            ).mean()
        ),
        "prediction_rank999_lift_median_spatial": float(
            (
                paired["rank_energy_999_ntdpl_prediction"]
                - paired["rank_energy_999_ntdpl_signal"]
            ).median()
        ),
        "prediction_tail_capture_median_spatial": float(paired["tail_capture_ratio_pred_vs_measured"].median()),
        "signal_rank99_median_spatial": float(signal["rank_energy_990"].median()),
        "prediction_rank99_median_spatial": float(pred["rank_energy_990"].median()),
        "measured_rank99_median_spatial": float(measured["rank_energy_990"].median()),
        "signal_rank999_median_spatial": float(signal["rank_energy_999"].median()),
        "prediction_rank999_median_spatial": float(pred["rank_energy_999"].median()),
        "measured_rank999_median_spatial": float(measured["rank_energy_999"].median()),
    }


def _correlation_summary(scene: pd.DataFrame) -> pd.DataFrame:
    features = [
        "nonlinear_rank99_spatial",
        "nonlinear_rank999_spatial",
        "nonlinear_tail_spatial",
        "nonlinear_entropy_spatial",
        "pred_rank999_lift_spatial",
        "pred_tail_capture_spatial",
    ]
    rows = []
    for target in ("RMSE_gain_pct", "SAM_gain_pct"):
        for feature in features:
            spearman = spearmanr(scene[feature], scene[target])
            pearson = pearsonr(scene[feature], scene[target])
            rows.append(
                {
                    "target": target,
                    "feature": feature,
                    "spearman_r": float(spearman.statistic),
                    "spearman_p": float(spearman.pvalue),
                    "pearson_r": float(pearson.statistic),
                    "pearson_p": float(pearson.pvalue),
                }
            )
    return pd.DataFrame(rows)


def _write_plots(summary: pd.DataFrame, scene: pd.DataFrame, outdir: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
    ax2 = ax1.twinx()
    ax1.plot(summary["p_max"], summary["rmse_gain_pct_mean"], marker="o", color="#1f77b4", label="RMSE gain")
    ax1.plot(summary["p_max"], summary["sam_gain_pct_mean"], marker="s", color="#4c78a8", linestyle="--", label="SAM gain")
    ax2.plot(summary["p_max"], summary["nonlinear_rank99_median_spatial"], marker="^", color="#d62728", label="nonlinear rank99")
    ax2.plot(summary["p_max"], summary["nonlinear_rank999_median_spatial"], marker="v", color="#ff7f0e", label="nonlinear rank99.9")
    ax1.set_xlabel("p_max")
    ax1.set_ylabel("mean gain (%)")
    ax2.set_ylabel("median spatial energy rank")
    ax1.set_title("Degree sweep: gains and nonlinear-response rank")
    ax1.grid(True, alpha=0.25)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    fig.tight_layout()
    fig.savefig(outdir / "degree_sweep_trend.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    scatter = ax.scatter(
        scene["pred_tail_capture_spatial"],
        scene["RMSE_gain_pct"],
        c=scene["p_max"],
        cmap="viridis",
        s=46,
        edgecolor="white",
        linewidth=0.6,
    )
    ax.set_xlabel("prediction tail capture ratio (spatial modes)")
    ax.set_ylabel("RMSE gain (%)")
    ax.set_title("Tail capture predicts RMSE gain")
    ax.grid(True, alpha=0.25)
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("p_max")
    fig.tight_layout()
    fig.savefig(outdir / "tail_capture_vs_gain.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize CAVE rank-inflation degree sweep outputs.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--rank-slug", default="r4")
    parser.add_argument("--target-slug", default="128")
    parser.add_argument("--suffix", default="_v2")
    parser.add_argument("--p-max-list", default="1,2,4,6")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_RESULTS_ROOT / "rank_inflation_spectrum_degree_sweep_r4_128")
    args = parser.parse_args()

    results_root = args.results_root if args.results_root.is_absolute() else PROJECT_ROOT / args.results_root
    outdir = args.outdir if args.outdir.is_absolute() else PROJECT_ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    p_values = [int(item.strip()) for item in args.p_max_list.split(",") if item.strip()]

    summaries = []
    scene_frames = []
    for p_max in p_values:
        root = _result_dir(results_root, args.rank_slug, p_max, args.target_slug, args.suffix)
        if not root.exists():
            raise FileNotFoundError(f"Missing result directory: {root}")
        summaries.append(_degree_summary(root, p_max))
        scene_frames.append(_scene_mechanism_frame(root, p_max))

    summary = pd.DataFrame(summaries).sort_values("p_max").reset_index(drop=True)
    scene = pd.concat(scene_frames, ignore_index=True).sort_values(["p_max", "scene_id"]).reset_index(drop=True)
    corr = _correlation_summary(scene)

    summary.to_csv(outdir / "degree_sweep_summary.csv", index=False)
    scene.to_csv(outdir / "degree_sweep_scene_mechanism.csv", index=False)
    corr.to_csv(outdir / "degree_sweep_correlations.csv", index=False)
    _write_plots(summary, scene, outdir)

    print(summary.to_string(index=False))
    print("\nTop correlations:")
    print(corr.sort_values("spearman_r", ascending=False).head(8).to_string(index=False))
    print(f"\nOutput: {outdir}")


if __name__ == "__main__":
    main()
