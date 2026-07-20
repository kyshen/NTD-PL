from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "artifacts" / "results"
TSP_FIGURES = PROJECT_ROOT / "papers" / "tsp" / "figures"
SUPP_TABLES = PROJECT_ROOT / "papers" / "tsp-supplementary" / "tables"

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
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
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


def _save(fig: plt.Figure, stem: str, out_dir: Path = TSP_FIGURES) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.png", dpi=240)
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

    make_rank_inflation_bar(summary, "mechanism_rank_inflation")

    table = summary[
        [
            "p_max",
            "rmse_gain_pct_mean",
            "nonlinear_rank999_median_spatial",
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
            ("nonlinear_rank999_median_spatial", "Nonlin. rank", 1),
            ("prediction_rank999_lift_median_spatial", "Rank inc.", 1),
            ("prediction_tail_capture_median_spatial", "Tail cap.", 3),
        ],
        "Rank-inflation summary on CAVE scenes at rank $(12,12,4)$. Gains are relative to matched-rank Tucker.",
        "tab:rank-inflation-mechanism",
        SUPP_TABLES / "mechanism_rank_inflation.tex",
    )
    corr_top = corr[corr["target"].eq("RMSE_gain_pct")].sort_values("spearman_r", ascending=False).head(4)
    corr_top.to_csv(SUPP_TABLES / "mechanism_rank_inflation_correlations.csv", index=False)


def make_rank_inflation_bar(summary: pd.DataFrame, stem: str) -> None:
    frame = summary.sort_values("p_max").copy()
    p_values = frame["p_max"].to_numpy(dtype=int)
    gains = frame["rmse_gain_pct_mean"].to_numpy(dtype=float)
    ranks = frame["nonlinear_rank999_median_spatial"].to_numpy(dtype=float)
    lifts = frame["prediction_rank999_lift_median_spatial"].to_numpy(dtype=float)

    y = np.arange(len(frame))
    fig, ax = plt.subplots(1, 1, figsize=(3.42, 1.75))
    colors = ["#AEB5BD" if p == 1 else BLUE for p in p_values]
    ax.barh(y, gains, color=colors, height=0.48, alpha=0.94, edgecolor="none", zorder=3)
    ax.scatter(gains, y, s=17, color=DARK, zorder=4)

    for idx, (gain, rank, lift) in enumerate(zip(gains, ranks, lifts)):
        label = rf"$r_{{\mathrm{{nl}}}}$ {rank:.1f}"
        if lift > 0:
            label += rf", rank $+{lift:.1f}$"
        x_text = max(gain + 0.18, 0.33)
        ax.text(
            x_text,
            idx,
            label,
            va="center",
            ha="left",
            fontsize=7.0,
            color=DARK,
        )

    ax.set_yticks(y, [rf"$P={p}$" for p in p_values])
    ax.invert_yaxis()
    ax.set_xlabel("RMSE gain over Tucker (%)")
    ax.set_xlim(0, max(gains) * 1.44)
    ax.set_xticks([0, 2, 4, 6, 8, 10, 12])
    ax.grid(axis="x", color=GRID, linewidth=0.55)
    ax.yaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.tick_params(length=2.5, width=0.6)
    fig.tight_layout(pad=0.25)
    _save(fig, stem)


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
    degree.to_csv(SUPP_TABLES / "mechanism_response_curve_summary.csv", index=False)

    mean_curve = (
        curve.groupby(["p_max", "grid_index"], as_index=False)[["s_norm", "f_norm"]]
        .mean()
        .sort_values(["p_max", "grid_index"])
    )
    colors = {1: GRAY, 2: GREEN, 4: ORANGE, 6: BLUE}
    fig, axes = plt.subplots(1, 4, figsize=(7.05, 1.46), sharex=True, sharey=True)
    for ax, p in zip(axes, [1, 2, 4, 6]):
        scene_curves = curve[curve["p_max"].eq(p)]
        for _, group in scene_curves.groupby("scene_id"):
            ax.plot(
                group["s_norm"],
                group["f_norm"],
                color=colors[p],
                alpha=0.13 if p > 1 else 0.10,
                linewidth=0.52,
            )
        m = mean_curve[mean_curve["p_max"].eq(p)]
        ax.plot(m["s_norm"], m["f_norm"], color=DARK, linewidth=1.65, solid_capstyle="round")
        ax.plot([0, 1], [0, 1], color=GRAY, linestyle="--", linewidth=0.72)
        row = degree[degree["p_max"].eq(p)].iloc[0]
        ax.set_title(rf"$P={p}$, gain {row['rmse_gain_vs_p1_pct']:.1f}%", fontsize=8.0)
        ax.text(
            0.04,
            0.94,
            rf"Curv. {row['curvature_mean']:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.0,
            color=DARK,
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": GRID,
                "linewidth": 0.35,
                "alpha": 0.86,
            },
            zorder=10,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.035, 1.035)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_yticks([0, 0.5, 1.0])
        ax.grid(color=GRID, linewidth=0.42)
        ax.tick_params(length=2.5, width=0.6)
        if ax is not axes[0]:
            ax.tick_params(labelleft=False)
    axes[0].set_ylabel(r"$f(s)$", labelpad=1.5)
    fig.supxlabel(r"$s$", y=0.01, fontsize=8.2)
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.25, top=0.82, wspace=0.18)
    _save(fig, "mechanism_learned_response")


def make_error_regime() -> None:
    root = RESULTS / "error_regime_analysis_cave_r12_p6_512"
    overall = pd.read_csv(root / "overall.csv")
    quantile = pd.read_csv(root / "summary_intensity_quantile.csv")
    band = pd.read_csv(root / "summary_spectral_band.csv")
    regime = pd.read_csv(root / "summary_spectral_regime.csv")
    inter = pd.read_csv(root / "summary_intensity_spectral.csv")

    fig = plt.figure(figsize=(3.42, 2.48))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[0.58, 0.92],
        hspace=0.52,
        wspace=0.34,
    )
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1], sharey=fig.axes[0]),
        fig.add_subplot(grid[1, :]),
    ]

    ax = axes[0]
    ax.plot(
        quantile["bin_index"],
        quantile["pooled_rmse_gain_pct"],
        marker="o",
        markersize=3.4,
        color=BLUE,
    )
    ax.axhline(0.0, color=GRAY, linewidth=0.75)
    ax.set_xlabel("intensity decile")
    ax.set_ylabel("gain (%)")
    ax.set_xticks([0, 4, 8])
    ax.grid(axis="y", color=GRID, linewidth=0.5)

    ax = axes[1]
    wavelength = 400 + 10 * band["band_index"]
    ax.plot(
        wavelength,
        band["pooled_rmse_gain_pct"],
        marker="o",
        markersize=2.5,
        color=GREEN,
    )
    ax.axhline(0.0, color=GRAY, linewidth=0.75)
    ax.set_xlabel("wavelength (nm)")
    ax.set_xticks([400, 550, 700])
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.tick_params(labelleft=False)

    ax = axes[2]
    pivot = inter.pivot(index="quantile_bin_index", columns="regime", values="pooled_rmse_gain_pct")
    pivot = pivot[[col for col in ["blue_400_500", "green_510_600", "red_610_700"] if col in pivot.columns]]
    finite = pivot.values[np.isfinite(pivot.values)]
    vmax = max(abs(float(np.nanpercentile(finite, 5))), abs(float(np.nanpercentile(finite, 95))), 1e-6) if finite.size else 1.0
    im = ax.imshow(pivot.values, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    labels = ["450", "555", "655"][: len(pivot.columns)]
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("intensity decile")
    colorbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.025)
    colorbar.set_label("gain (%)")
    fig.subplots_adjust(left=0.16, right=0.91, top=0.985, bottom=0.12)
    _save(fig, "mechanism_error_regime", TSP_FIGURES)

    o = {
        "RMSE gain": float(overall["rmse_gain_pct"].mean()),
        "Intensity peak": int(quantile.loc[quantile["pooled_rmse_gain_pct"].idxmax(), "bin_index"]),
        "Peak gain": float(quantile["pooled_rmse_gain_pct"].max()),
        "High-intensity contribution": float(quantile.loc[quantile["bin_index"].eq(9), "mse_reduction_contribution_pct"].iloc[0]),
        "Red contribution": float(regime.loc[regime["regime"].str.contains("red"), "mse_reduction_contribution_pct"].iloc[0]),
    }
    pd.DataFrame([o]).to_csv(SUPP_TABLES / "mechanism_error_regime_overview.csv", index=False)

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
        "Spectral-regime error analysis on CAVE scenes at rank $(12,12,4)$.",
        "tab:error-regime-mechanism",
        SUPP_TABLES / "mechanism_error_regime.tex",
    )


def main() -> None:
    _set_style()
    SUPP_TABLES.mkdir(parents=True, exist_ok=True)
    make_rank_inflation()
    make_response_curve()
    make_error_regime()
    print(f"Wrote figures to {TSP_FIGURES}")
    print(f"Wrote supplementary tables to {SUPP_TABLES}")


if __name__ == "__main__":
    main()
