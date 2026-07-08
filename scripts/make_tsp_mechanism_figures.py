from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "artifacts" / "results"
TSP_FIGURES = PROJECT_ROOT / "papers" / "tsp" / "figures"
TSP_TABLES = PROJECT_ROOT / "papers" / "tsp" / "tables"

BLUE = "#2F6FBB"
GREEN = "#2A9D8F"
ORANGE = "#D9822B"
RED = "#C44E52"
DARK = "#222222"
GRAY = "#777777"
GRID = "#D8DEE6"


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.7,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.6,
            "figure.titlesize": 9.5,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.45,
            "patch.linewidth": 0.6,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.015,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, stem: str) -> None:
    TSP_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(TSP_FIGURES / f"{stem}.pdf")
    fig.savefig(TSP_FIGURES / f"{stem}.png", dpi=240)
    plt.close(fig)


def _fmt(value: float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def _latex_table(
    frame: pd.DataFrame,
    columns: list[tuple[str, str, int]],
    caption: str,
    label: str,
    path: Path,
) -> None:
    lines = [
        r"\begin{table}[!t]",
        r"  \centering",
        rf"  \caption{{{caption}}}",
        rf"  \label{{{label}}}",
        r"  \scriptsize",
        r"  \begin{tabular}{@{}" + "l" + "r" * (len(columns) - 1) + r"@{}}",
        r"    \toprule",
        "    " + " & ".join(header for _, header, _ in columns) + r" \\",
        r"    \midrule",
    ]
    for _, row in frame.iterrows():
        cells: list[str] = []
        for key, _, digits in columns:
            value = row[key]
            if isinstance(value, str):
                cells.append(value)
            elif float(value).is_integer() and digits == 0:
                cells.append(str(int(value)))
            else:
                cells.append(_fmt(float(value), digits))
        lines.append("    " + " & ".join(cells) + r" \\")
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def make_rank_inflation() -> None:
    root = RESULTS / "rank_inflation_spectrum_degree_sweep_r12_512"
    summary = pd.read_csv(root / "degree_sweep_summary.csv")
    scene = pd.read_csv(root / "degree_sweep_scene_mechanism.csv")
    corr = pd.read_csv(root / "degree_sweep_correlations.csv")

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.65))
    ax = axes[0]
    ax2 = ax.twinx()
    ax.plot(summary["p_max"], summary["rmse_gain_pct_mean"], marker="o", color=BLUE, label="RMSE gain")
    ax.plot(summary["p_max"], summary["sam_gain_pct_mean"], marker="s", color=GREEN, label="SAM gain")
    ax2.plot(summary["p_max"], summary["response_rank999_median_spatial"], marker="^", color=ORANGE, label="response rank")
    ax.set_xlabel("maximum degree $P$")
    ax.set_ylabel("mean gain (%)")
    ax2.set_ylabel("median response rank")
    ax.set_title("Degree increases rank inflation")
    ax.grid(axis="y", color=GRID, linewidth=0.55)
    ax.set_xticks(summary["p_max"])
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=False)

    ax = axes[1]
    scatter = ax.scatter(
        scene["pred_tail_capture_spatial"],
        scene["RMSE_gain_pct"],
        c=scene["p_max"],
        cmap="viridis",
        s=28,
        edgecolor="white",
        linewidth=0.45,
    )
    ax.set_xlabel("prediction tail capture")
    ax.set_ylabel("RMSE gain (%)")
    ax.set_title("Tail capture tracks gain")
    ax.grid(color=GRID, linewidth=0.55)
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.05, pad=0.02)
    colorbar.set_label("$P$")
    fig.tight_layout(w_pad=1.0)
    _save(fig, "mechanism_rank_inflation")

    table = summary[
        [
            "p_max",
            "rmse_gain_pct_mean",
            "response_rank999_median_spatial",
            "prediction_rank999_lift_median_spatial",
            "prediction_tail_capture_median_spatial",
        ]
    ].copy()
    table = table.rename(columns={"p_max": "P"})
    _latex_table(
        table,
        [
            ("P", "$P$", 0),
            ("rmse_gain_pct_mean", "RMSE gain (\\%)", 2),
            ("response_rank999_median_spatial", "Resp. rank", 1),
            ("prediction_rank999_lift_median_spatial", "Pred. lift", 1),
            ("prediction_tail_capture_median_spatial", "Tail cap.", 3),
        ],
        "Rank-inflation summary on full-size CAVE scenes at rank $(12,12,4)$. Gains are relative to matched-rank Tucker.",
        "tab:rank-inflation-mechanism",
        TSP_TABLES / "mechanism_rank_inflation.tex",
    )
    corr_top = corr[corr["target"].eq("RMSE_gain_pct")].sort_values("spearman_r", ascending=False).head(4)
    corr_top.to_csv(TSP_TABLES / "mechanism_rank_inflation_correlations.csv", index=False)


def make_response_curve() -> None:
    roots = {p: RESULTS / f"learned_response_curve_cave_r12_p{p}_512" for p in [1, 2, 4, 6]}
    summaries = []
    curves = []
    for p, root in roots.items():
        s = pd.read_csv(root / "cave_response_summary.csv")
        c = pd.read_csv(root / "cave_response_curves.csv")
        s["p_max"] = p
        c["p_max"] = p
        summaries.append(s)
        curves.append(c)
    summary = pd.concat(summaries, ignore_index=True)
    curve = pd.concat(curves, ignore_index=True)

    degree = (
        summary.groupby("p_max", as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            nonlinear_mean=("nonlinear_deviation", "mean"),
            nonlinear_median=("nonlinear_deviation", "median"),
            monotone_mean=("monotone_fraction", "mean"),
            curvature_mean=("normalized_mean_abs_curvature", "mean"),
        )
        .sort_values("p_max")
    )
    rmse_p1 = float(degree.loc[degree["p_max"].eq(1), "rmse_mean"].iloc[0])
    degree["rmse_gain_vs_p1_pct"] = 100.0 * (rmse_p1 - degree["rmse_mean"]) / rmse_p1
    degree.to_csv(TSP_TABLES / "mechanism_response_curve_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75))
    ax = axes[0]
    high = curve[curve["p_max"].eq(6)]
    for _, group in high.groupby("scene_id"):
        ax.plot(group["s_norm"], group["f_norm"], color=BLUE, alpha=0.20, linewidth=0.75)
    mean_curve = high.groupby("grid_index", as_index=False)[["s_norm", "f_norm"]].mean()
    ax.plot(mean_curve["s_norm"], mean_curve["f_norm"], color=DARK, linewidth=1.9, label="mean")
    ax.plot([0, 1], [0, 1], color=GRAY, linestyle="--", linewidth=1.0, label="linear")
    ax.set_xlabel("latent percentile scale")
    ax.set_ylabel("response percentile scale")
    ax.set_title("Learned response curves")
    ax.grid(color=GRID, linewidth=0.55)
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1]
    ax.plot(degree["p_max"], degree["nonlinear_mean"], marker="o", color=ORANGE, label="nonlinear deviation")
    ax2 = ax.twinx()
    ax2.plot(degree["p_max"], degree["rmse_gain_vs_p1_pct"], marker="s", color=BLUE, label="RMSE gain")
    ax.set_xlabel("maximum degree $P$")
    ax.set_ylabel("nonlinear deviation")
    ax2.set_ylabel("gain over $P=1$ (%)")
    ax.set_title("Curve shape stabilizes")
    ax.set_xticks(degree["p_max"])
    ax.grid(axis="y", color=GRID, linewidth=0.55)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=False)
    fig.tight_layout(w_pad=1.0)
    _save(fig, "mechanism_learned_response")

    tex = degree.rename(columns={"p_max": "P"})[
        ["P", "rmse_gain_vs_p1_pct", "nonlinear_mean", "monotone_mean", "curvature_mean"]
    ]
    _latex_table(
        tex,
        [
            ("P", "$P$", 0),
            ("rmse_gain_vs_p1_pct", "Gain vs. $P=1$ (\\%)", 2),
            ("nonlinear_mean", "Nonlin.", 3),
            ("monotone_mean", "Mono.", 3),
            ("curvature_mean", "Curv.", 2),
        ],
        "Learned-response summary on full-size CAVE scenes at rank $(12,12,4)$.",
        "tab:learned-response-mechanism",
        TSP_TABLES / "mechanism_learned_response.tex",
    )


def make_error_regime() -> None:
    root = RESULTS / "error_regime_analysis_cave_r12_p6_512"
    overall = pd.read_csv(root / "overall.csv")
    quantile = pd.read_csv(root / "summary_intensity_quantile.csv")
    band = pd.read_csv(root / "summary_spectral_band.csv")
    regime = pd.read_csv(root / "summary_spectral_regime.csv")
    inter = pd.read_csv(root / "summary_intensity_spectral.csv")

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.55), gridspec_kw={"width_ratios": [1.0, 1.0, 1.08]})
    ax = axes[0]
    ax.plot(quantile["bin_index"], quantile["pooled_rmse_gain_pct"], marker="o", color=BLUE)
    ax.axhline(0.0, color=GRAY, linewidth=0.8)
    ax.set_xlabel("intensity decile")
    ax.set_ylabel("RMSE gain (%)")
    ax.set_title("Gain by intensity")
    ax.grid(axis="y", color=GRID, linewidth=0.55)

    ax = axes[1]
    ax.plot(band["band_index"], band["pooled_rmse_gain_pct"], color=GREEN, marker="o", markersize=2.8)
    ax.axhline(0.0, color=GRAY, linewidth=0.8)
    ax.set_xlabel("spectral band")
    ax.set_title("Gain by wavelength")
    ax.grid(axis="y", color=GRID, linewidth=0.55)

    ax = axes[2]
    pivot = inter.pivot(index="quantile_bin_index", columns="regime", values="pooled_rmse_gain_pct")
    finite = pivot.values[np.isfinite(pivot.values)]
    vmax = max(abs(float(np.nanpercentile(finite, 5))), abs(float(np.nanpercentile(finite, 95))), 1e-6) if finite.size else 1.0
    im = ax.imshow(pivot.values, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    labels = [col.replace("_", " ") for col in pivot.columns]
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=28, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
    ax.set_xlabel("spectral regime")
    ax.set_title("Joint regimes")
    colorbar = fig.colorbar(im, ax=ax, fraction=0.052, pad=0.02)
    colorbar.set_label("gain (%)")
    fig.tight_layout(w_pad=0.8)
    _save(fig, "mechanism_error_regime")

    o = {
        "RMSE gain": float(overall["rmse_gain_pct"].mean()),
        "Intensity peak": int(quantile.loc[quantile["pooled_rmse_gain_pct"].idxmax(), "bin_index"]),
        "Peak gain": float(quantile["pooled_rmse_gain_pct"].max()),
        "High-intensity contribution": float(quantile.loc[quantile["bin_index"].eq(9), "mse_reduction_contribution_pct"].iloc[0]),
        "Red contribution": float(regime.loc[regime["regime"].str.contains("red"), "mse_reduction_contribution_pct"].iloc[0]),
    }
    pd.DataFrame([o]).to_csv(TSP_TABLES / "mechanism_error_regime_overview.csv", index=False)

    regime_tex = regime.copy()
    regime_tex["regime_label"] = regime_tex["regime"].map(
        {
            "blue_400_500": "400--500 nm",
            "green_510_600": "510--600 nm",
            "red_610_700": "610--700 nm",
        }
    )
    _latex_table(
        regime_tex[["regime_label", "pooled_rmse_gain_pct", "scene_median_gain_pct", "mse_reduction_contribution_pct"]],
        [
            ("regime_label", "Regime", 0),
            ("pooled_rmse_gain_pct", "Pooled gain (\\%)", 2),
            ("scene_median_gain_pct", "Median gain (\\%)", 2),
            ("mse_reduction_contribution_pct", "MSE contrib. (\\%)", 1),
        ],
        "Spectral-regime error analysis on full-size CAVE scenes at rank $(12,12,4)$.",
        "tab:error-regime-mechanism",
        TSP_TABLES / "mechanism_error_regime.tex",
    )


def main() -> None:
    _set_style()
    TSP_TABLES.mkdir(parents=True, exist_ok=True)
    make_rank_inflation()
    make_response_curve()
    make_error_regime()
    print(f"Wrote figures to {TSP_FIGURES}")
    print(f"Wrote tables to {TSP_TABLES}")


if __name__ == "__main__":
    main()
