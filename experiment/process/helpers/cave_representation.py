from __future__ import annotations

import json
import sys
import importlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import binomtest, rankdata, wilcoxon
from sklearn.manifold import trustworthiness
from sklearn.preprocessing import StandardScaler

from ...config import get_env
from ...utils.io import load_run_parquets, maybe_numeric
from ...utils.paper import sync_artifact_to_latex, write_csv_artifact, write_text_artifact
from ...utils.plotting import PALETTE, apply_theme, legend_style, method_style, save_figure, style_axes, style_colorbar


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.hsi import CAVEHSIData
from src.methods.cp import CPDecomposition
from src.methods.ntdpl import NTDPLDecomposition
from src.methods.tr import TRDecomposition
from src.methods.tt import TTDecomposition
from src.methods.tucker import TuckerDecomposition
from src.types import Result
from tensorly.tucker_tensor import tucker_to_tensor


METHOD_ORDER = ["cp", "tt", "tr", "tucker", "ntdpl"]
METHOD_FOCUS_ORDER = ["tucker", "ntdpl"]
METHOD_CLASSES = {
    "cp": CPDecomposition,
    "tt": TTDecomposition,
    "tr": TRDecomposition,
    "tucker": TuckerDecomposition,
    "ntdpl": NTDPLDecomposition,
}
RANK_ORDER = [(8, 8, 4), (12, 12, 6), (16, 16, 8)]
TARGET_RANK = (8, 8, 4)
OVERVIEW_RANK = (12, 12, 6)
FIXED_SPECTRAL_POINTS = [(29, 50), (29, 49), (30, 50), (30, 49)]
FIXED_VISUAL_RANK = (16, 16, 8)
FIXED_VISUAL_SCENES = [2,11]
VISUAL_MAIN_RANK = OVERVIEW_RANK
VISUAL_MAIN_SCENES = [2, 8, 9]
VISUAL_APPENDIX_SCENES = [2, 11, 14]
VISUAL_ROI_SIZE = 22
SPECTRAL_V2_RANK = (12, 12, 6)
PMAX_SCAN_RANK = (12, 12, 6)
PMAX_SCAN_SCENE_ID = 2
PMAX_VALUES = (1, 2, 3, 4, 5, 6)
MAIN_NTDPL_PMAX = 6
MANIFOLD_K = 10
MANIFOLD_MAX_POINTS = 1500


def _load_runs() -> tuple[pd.DataFrame, object]:
    env = get_env("cave-representation")
    runs = load_run_parquets(env.results_dir)["runs"].copy()
    if runs.empty:
        raise RuntimeError("No runs found for cave representation. Run `python -m experiment cave-representation run` first.")
    return runs, env


def _jsonish(value: Any) -> Any:
    out = value
    while isinstance(out, str):
        text = out.strip()
        if not text:
            return text
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError:
            return out
        if loaded == out:
            return loaded
        out = loaded
    return out


def _parse_rank(value: Any) -> tuple[int, int, int]:
    parsed = _jsonish(value)
    if isinstance(parsed, str):
        parsed = parsed.strip("[]()")
        items = [part.strip() for part in parsed.split(",") if part.strip()]
        return tuple(int(item) for item in items)  # type: ignore[return-value]
    if isinstance(parsed, (list, tuple)):
        return tuple(int(item) for item in parsed)  # type: ignore[return-value]
    raise ValueError(f"Cannot parse rank from value: {value}")


def _rank_text(rank: tuple[int, int, int]) -> str:
    return f"({rank[0]},{rank[1]},{rank[2]})"


def _base_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out = out.loc[out["ovr.data"].astype(str) == "cave_hsi"].copy()
    out["data_name"] = "cave_hsi"
    out["method_name"] = out["ovr.method"].astype(str)
    out["rank"] = out["ovr.method.rank"].map(_parse_rank)
    out["rank_text"] = out["rank"].map(_rank_text)
    out["scene_id"] = maybe_numeric(out["ovr.data.id"]).astype(int)
    out["RMSE"] = maybe_numeric(out["RMSE"]).astype(float)
    out["NMSE"] = maybe_numeric(out["NMSE"]).astype(float)
    out["NMSE_dB"] = maybe_numeric(out["NMSE_dB"]).astype(float)
    out["SAM"] = maybe_numeric(out["SAM"]).astype(float)
    out["CR"] = maybe_numeric(out["CR"]).astype(float)
    if "fit_time_sec" in out.columns:
        out["fit_time_sec"] = maybe_numeric(out["fit_time_sec"]).astype(float)
    else:
        out["fit_time_sec"] = np.nan
    dedup_keys = ["scene_id", "method_name", "rank_text"]
    if "ovr.method.p_max" in out.columns:
        out["p_max"] = maybe_numeric(out["ovr.method.p_max"])
        dedup_keys.append("p_max")
    out = out.sort_values("run_dir").drop_duplicates(subset=dedup_keys, keep="last")
    return out


def _main_experiment_rows(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "p_max" not in out.columns:
        return out
    ntdpl_main = out["method_name"].eq("ntdpl") & out["p_max"].eq(float(MAIN_NTDPL_PMAX))
    return out.loc[~out["method_name"].eq("ntdpl") | ntdpl_main].copy()


def _metric_text(series: pd.Series, digits: int = 4) -> str:
    values = maybe_numeric(series).dropna().to_numpy(dtype=float)
    if values.size == 0:
        return "---"
    mean = values.mean()
    if values.size == 1:
        return f"{mean:.{digits}f}"
    std = values.std(ddof=0)
    return f"{mean:.{digits}f} +- {std:.{digits}f}"


def _pm_text(mean: float, std: float, digits: int) -> str:
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def _latex_number(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _rank_pub_text(rank_text: str) -> str:
    return f"${rank_text}$"


def _latex_pvalue(value: float) -> str:
    if value < 1e-3:
        base, exp = f"{value:.2e}".split("e")
        exponent = int(exp)
        return f"{float(base):.2f}$\\times 10^{{{exponent}}}$"
    return f"{value:.3f}"


def _rank_biserial(diffs: np.ndarray) -> float:
    nonzero = diffs[~np.isclose(diffs, 0.0)]
    if nonzero.size == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero), method="average")
    positive_rank_sum = float(ranks[nonzero > 0].sum())
    negative_rank_sum = float(ranks[nonzero < 0].sum())
    denom = nonzero.size * (nonzero.size + 1) / 2.0
    if denom <= 0:
        return 0.0
    return (positive_rank_sum - negative_rank_sum) / denom


def _save_and_sync_figure(
    fig: plt.Figure,
    env: object,
    stem: str,
    *,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 400,
) -> None:
    save_figure(fig, env.artifacts_dir / stem, formats=formats, dpi=dpi)
    for fmt in formats:
        sync_artifact_to_latex(env, env.artifacts_dir / f"{stem}.{fmt}")


def _scene_pair_table(
    tbl: pd.DataFrame,
    *,
    rank: tuple[int, int, int],
    methods: tuple[str, str] = ("tucker", "ntdpl"),
    metrics: tuple[str, ...] = ("RMSE", "SAM"),
) -> pd.DataFrame:
    left_method, right_method = methods
    subset = tbl.loc[(tbl["rank"] == rank) & (tbl["method_name"].isin(methods))].copy()
    if subset.empty:
        raise RuntimeError(f"No scene-level rows found for paired comparison at rank {rank}.")

    pivot = subset.pivot_table(index="scene_id", columns="method_name", values=list(metrics), aggfunc="mean")
    rows: list[dict[str, Any]] = []
    for scene_id in sorted(subset["scene_id"].unique().tolist()):
        row: dict[str, Any] = {
            "Rank": _rank_text(rank),
            "scene_id": int(scene_id),
            "Scene": f"Scene {int(scene_id)}",
        }
        valid = True
        for metric in metrics:
            try:
                left_value = float(pivot.loc[scene_id, (metric, left_method)])
                right_value = float(pivot.loc[scene_id, (metric, right_method)])
            except KeyError:
                valid = False
                break
            delta = left_value - right_value
            if delta > 1e-12:
                winner = right_method
            elif delta < -1e-12:
                winner = left_method
            else:
                winner = "tie"
            row[f"{metric}_{left_method}"] = left_value
            row[f"{metric}_{right_method}"] = right_value
            row[f"{metric}_delta_{left_method}_minus_{right_method}"] = delta
            row[f"{metric}_winner"] = winner
        if valid:
            rows.append(row)

    paired = pd.DataFrame(rows).sort_values("scene_id").reset_index(drop=True)
    if paired.empty:
        raise RuntimeError(f"No complete Tucker/NTD-PL scene pairs found at rank {rank}.")
    return paired


def _scene_improvement_table(paired: pd.DataFrame) -> pd.DataFrame:
    out = paired.copy()
    out["RMSE_improvement"] = out["RMSE_delta_tucker_minus_ntdpl"].astype(float)
    out["SAM_improvement"] = out["SAM_delta_tucker_minus_ntdpl"].astype(float)
    return out


def _sorted_improvement_frame(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    improvement_col = f"{metric}_improvement"
    sorted_frame = frame.sort_values(improvement_col, ascending=False).reset_index(drop=True).copy()
    sorted_frame["sorted_label"] = [f"S{idx}" for idx in range(1, len(sorted_frame) + 1)]
    return sorted_frame


def _improvement_summary(values: pd.Series) -> dict[str, float]:
    numeric = values.astype(float)
    return {
        "wins": int(np.sum(numeric > 0.0)),
        "total": int(numeric.size),
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
    }


def _minimal_axes_style(ax: plt.Axes) -> None:
    style_axes(ax, grid=False)
    ax.yaxis.grid(True, color="#D8DDE3", alpha=0.55, linewidth=0.6)
    ax.xaxis.grid(False)
    ax.spines["left"].set_linewidth(0.75)
    ax.spines["bottom"].set_linewidth(0.75)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")


def _improvement_note(values: pd.Series, digits: int) -> str:
    summary = _improvement_summary(values)
    return "\n".join(
        [
            f"wins: {summary['wins']}/{summary['total']}",
            f"mean Δ = {summary['mean']:.{digits}f}",
            f"median Δ = {summary['median']:.{digits}f}",
        ]
    )


def _improvement_note_text(values: pd.Series, digits: int) -> str:
    summary = _improvement_summary(values)
    return "\n".join(
        [
            f"wins: {summary['wins']}/{summary['total']}",
            rf"mean $\Delta$ = {summary['mean']:.{digits}f}",
            rf"median $\Delta$ = {summary['median']:.{digits}f}",
        ]
    )


def _improvement_bar_panel(
    ax: plt.Axes,
    *,
    frame: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    digits: int,
    positive_color: str,
    negative_color: str,
) -> None:
    ordered = _sorted_improvement_frame(frame, metric)
    values = ordered[f"{metric}_improvement"].to_numpy(dtype=float)
    labels = ordered["sorted_label"].tolist()
    colors = [positive_color if value > 0.0 else negative_color for value in values]
    x_positions = np.arange(len(values), dtype=float)

    ax.bar(
        x_positions,
        values,
        width=0.74,
        color=colors,
        edgecolor="none",
        zorder=3,
    )
    ax.axhline(0.0, color="#5F6670", linewidth=1.0, zorder=4)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("Sorted scenes")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.6, len(values) - 0.4)
    _minimal_axes_style(ax)
    ax.text(
        0.98,
        0.97,
        _improvement_note_text(ordered[f"{metric}_improvement"], digits),
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=8.6,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#C9CDD3", "alpha": 0.97},
    )


def _distribution_panel(
    ax: plt.Axes,
    *,
    frame: pd.DataFrame,
    metric: str,
    ylabel: str,
    positive_color: str,
    baseline_color: str,
    digits: int,
) -> None:
    left = frame[f"{metric}_tucker"].to_numpy(dtype=float)
    right = frame[f"{metric}_ntdpl"].to_numpy(dtype=float)
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.055, 0.055, size=left.size)
    left_x = np.full(left.shape, 0.0, dtype=float) + jitter
    right_x = np.full(right.shape, 1.0, dtype=float) + jitter

    for x0, y0, x1, y1 in zip(left_x, left, right_x, right, strict=False):
        ax.plot([x0, x1], [y0, y1], color="#CCD2D9", linewidth=0.75, alpha=0.7, zorder=1)

    box = ax.boxplot(
        [left, right],
        positions=[0.0, 1.0],
        widths=0.42,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": PALETTE.black, "linewidth": 1.0},
        whiskerprops={"color": "#7A7F87", "linewidth": 0.9},
        capprops={"color": "#7A7F87", "linewidth": 0.9},
        boxprops={"linewidth": 0.9, "edgecolor": "#7A7F87"},
    )
    box["boxes"][0].set_facecolor("#F3F4F6")
    box["boxes"][1].set_facecolor("#DCEAF8")

    ax.scatter(left_x, left, s=18, color=baseline_color, alpha=0.9, linewidths=0.0, zorder=2)
    ax.scatter(right_x, right, s=20, color=positive_color, alpha=0.95, linewidths=0.0, zorder=3)
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels(["Tucker", "NTD-PL"])
    ax.set_ylabel(ylabel)
    _minimal_axes_style(ax)
    ax.text(
        0.98,
        0.97,
        _improvement_note_text(frame[f"{metric}_improvement"], digits),
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=8.4,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#C9CDD3", "alpha": 0.97},
    )


def scene_improvement_overview_plot() -> None:
    runs, env = _load_runs()
    tbl = _main_experiment_rows(_base_columns(runs))
    paired = _scene_pair_table(tbl, rank=OVERVIEW_RANK)
    improvement = _scene_improvement_table(paired)
    rmse_order = _sorted_improvement_frame(improvement, "RMSE").loc[:, ["scene_id", "sorted_label"]].rename(
        columns={"sorted_label": "RMSE_sorted_label"}
    )
    sam_order = _sorted_improvement_frame(improvement, "SAM").loc[:, ["scene_id", "sorted_label"]].rename(
        columns={"sorted_label": "SAM_sorted_label"}
    )
    improvement_export = improvement.merge(rmse_order, on="scene_id", how="left").merge(sam_order, on="scene_id", how="left")

    raw_columns = [
        "Rank",
        "scene_id",
        "Scene",
        "RMSE_tucker",
        "RMSE_ntdpl",
        "RMSE_delta_tucker_minus_ntdpl",
        "RMSE_winner",
        "SAM_tucker",
        "SAM_ntdpl",
        "SAM_delta_tucker_minus_ntdpl",
        "SAM_winner",
    ]
    improvement_columns = [
        "Rank",
        "scene_id",
        "Scene",
        "RMSE_improvement",
        "SAM_improvement",
        "RMSE_sorted_label",
        "SAM_sorted_label",
    ]
    improvement_csv_path, improvement_latex_csv_path = write_csv_artifact(
        env,
        improvement_export.loc[:, improvement_columns],
        "scene_improvement_overview.csv",
    )

    positive_color = "#2D76B7"
    negative_color = "#C9CDD3"

    apply_theme()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), sharex=False)
    _improvement_bar_panel(
        axes[0],
        frame=improvement,
        metric="RMSE",
        title="RMSE improvement",
        ylabel=r"$\Delta$RMSE",
        digits=4,
        positive_color=positive_color,
        negative_color=negative_color,
    )
    _improvement_bar_panel(
        axes[1],
        frame=improvement,
        metric="SAM",
        title="SAM improvement",
        ylabel=r"$\Delta$SAM (deg)",
        digits=2,
        positive_color=positive_color,
        negative_color=negative_color,
    )
    fig.tight_layout()
    _save_and_sync_figure(fig, env, "scene_improvement_overview", formats=("pdf", "png"), dpi=600)
    plt.close(fig)

    print(improvement_export.loc[:, improvement_columns].to_string(index=False))
    print(f"\nSaved: {improvement_csv_path}")
    print(f"Synced: {improvement_latex_csv_path}")


def scene_distribution_summary_plot() -> None:
    runs, env = _load_runs()
    tbl = _main_experiment_rows(_base_columns(runs))
    paired = _scene_pair_table(tbl, rank=OVERVIEW_RANK)
    improvement = _scene_improvement_table(paired)

    positive_color = "#2D76B7"
    baseline_color = "#7C838C"

    apply_theme()
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.1))
    axes[0].set_title("RMSE distribution")
    _distribution_panel(
        axes[0],
        frame=improvement,
        metric="RMSE",
        ylabel="RMSE",
        positive_color=positive_color,
        baseline_color=baseline_color,
        digits=4,
    )
    axes[1].set_title("SAM distribution")
    _distribution_panel(
        axes[1],
        frame=improvement,
        metric="SAM",
        ylabel="SAM (deg)",
        positive_color=positive_color,
        baseline_color=baseline_color,
        digits=2,
    )
    fig.tight_layout()
    _save_and_sync_figure(fig, env, "scene_distribution_summary", formats=("pdf", "png"), dpi=600)
    plt.close(fig)


def reconstruction_table() -> None:
    runs, env = _load_runs()
    tbl = _main_experiment_rows(_base_columns(runs))

    dataset = CAVEHSIData(path="data/CAVE", id=1, target_shape=(128, 128))
    tensor_size = int(dataset.get_size())
    rows: list[dict[str, Any]] = []
    for rank in RANK_ORDER:
        rank_tbl = tbl.loc[tbl["rank"] == rank].copy()
        for method in METHOD_ORDER:
            subset = rank_tbl.loc[rank_tbl["method_name"] == method].copy()
            if subset.empty:
                continue
            params = tensor_size / subset["CR"].to_numpy(dtype=float)
            cr_mean = float(subset["CR"].mean())
            rmse_mean = float(subset["RMSE"].mean())
            rmse_std = float(subset["RMSE"].std(ddof=0))
            nmse_mean = float(subset["NMSE_dB"].mean())
            nmse_std = float(subset["NMSE_dB"].std(ddof=0))
            sam_mean = float(subset["SAM"].mean())
            sam_std = float(subset["SAM"].std(ddof=0))
            rows.append(
                {
                    "Rank": _rank_text(rank),
                    "Method": env.label_for_method(method),
                    "Params": int(round(params.mean())),
                    "Scenes": str(int(subset["scene_id"].nunique())),
                    "CR": _metric_text(subset["CR"], digits=2),
                    "CR_mean": cr_mean,
                    "RMSE": _metric_text(subset["RMSE"], digits=5),
                    "RMSE_mean": rmse_mean,
                    "RMSE_std": rmse_std,
                    "NMSE(dB)": _metric_text(subset["NMSE_dB"], digits=4),
                    "NMSE_mean": nmse_mean,
                    "NMSE_std": nmse_std,
                    "SAM(deg)": _metric_text(subset["SAM"], digits=4),
                    "SAM_mean": sam_mean,
                    "SAM_std": sam_std,
                }
            )

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise RuntimeError("No rows available for CAVE reconstruction table.")

    csv_summary = summary.loc[:, ["Rank", "Method", "Params", "Scenes", "CR", "RMSE", "NMSE(dB)", "SAM(deg)"]]
    csv_path, latex_csv_path = write_csv_artifact(env, csv_summary, "recon_summary.csv")

    tex_lines = [
        r"\begin{tabular}{c|c|c|c|c|c|c|c}",
        r"    \hline",
        r"    Rank & Method & Params & Scenes & CR & RMSE & NMSE(dB) & SAM(deg) \\",
        r"    \hline",
    ]
    for rank in (_rank_text(rank) for rank in RANK_ORDER):
        rank_rows = [row for row in rows if row["Rank"] == rank]
        for idx, row in enumerate(rank_rows):
            rank_cell = row["Rank"] if idx == 0 else ""
            tex_lines.append(
                f"    {rank_cell} & {row['Method']} & {row['Params']} & {row['Scenes']} & {row['CR']} & {row['RMSE']} & {row['NMSE(dB)']} & {row['SAM(deg)']} \\\\"
            )
        tex_lines.append(r"    \hline")
    tex_lines.append(r"\end{tabular}")
    tex_path, latex_tex_path = write_text_artifact(env, "\n".join(tex_lines) + "\n", "recon_summary.tex")

    pub_lines = [
        r"\begin{tabular}{c l r c c c}",
        r"\toprule",
        r"Rank & Method & Params & RMSE$\downarrow$ & NMSE(dB)$\downarrow$ & SAM(deg)$\downarrow$\\",
        r"\midrule",
    ]
    for rank in (_rank_text(rank) for rank in RANK_ORDER):
        rank_rows = [row for row in rows if row["Rank"] == rank]
        best_rmse = min(row["RMSE_mean"] for row in rank_rows)
        best_nmse = min(row["NMSE_mean"] for row in rank_rows)
        best_sam = min(row["SAM_mean"] for row in rank_rows)
        for idx, row in enumerate(rank_rows):
            rank_cell = _rank_pub_text(row["Rank"]) if idx == 0 else ""
            rmse_text = _pm_text(row["RMSE_mean"], row["RMSE_std"], digits=4)
            nmse_text = _pm_text(row["NMSE_mean"], row["NMSE_std"], digits=3)
            sam_text = _pm_text(row["SAM_mean"], row["SAM_std"], digits=2)
            if np.isclose(row["RMSE_mean"], best_rmse):
                rmse_text = rf"\textbf{{{rmse_text}}}"
            if np.isclose(row["NMSE_mean"], best_nmse):
                nmse_text = rf"\textbf{{{nmse_text}}}"
            if np.isclose(row["SAM_mean"], best_sam):
                sam_text = rf"\textbf{{{sam_text}}}"
            pub_lines.append(
                f"{rank_cell} & {row['Method']:<7} & {row['Params']} & "
                f"{rmse_text} & {nmse_text} & {sam_text} \\\\"
            )
        pub_lines.append(r"\midrule")
    pub_lines[-1] = r"\bottomrule"
    pub_lines.append(r"\end{tabular}")
    pub_path, latex_pub_path = write_text_artifact(env, "\n".join(pub_lines) + "\n", "recon_summary_pub.tex")

    print(csv_summary.to_string(index=False))
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {tex_path}")
    print(f"Saved: {pub_path}")
    print(f"Synced: {latex_csv_path}")
    print(f"Synced: {latex_tex_path}")
    print(f"Synced: {latex_pub_path}")


def cr_curve_plot() -> None:
    runs, env = _load_runs()
    tbl = _main_experiment_rows(_base_columns(runs))
    grouped = (
        tbl.groupby(["method_name", "rank_text"], as_index=False)[["CR", "RMSE", "SAM"]]
        .mean()
        .sort_values(["method_name", "CR"])
    )

    apply_theme()
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), sharex=False)
    for metric, ax in zip(("RMSE", "SAM"), axes):
        for method in METHOD_ORDER:
            subset = grouped.loc[grouped["method_name"] == method].copy()
            if subset.empty:
                continue
            label = env.label_for_method(method)
            ax.plot(
                subset["CR"].to_numpy(dtype=float),
                subset[metric].to_numpy(dtype=float),
                label=label,
                **method_style(label),
            )
        ax.set_title(f"CAVE: {metric} vs CR")
        ax.set_xlabel("CR")
        ax.set_ylabel("SAM (deg)" if metric == "SAM" else metric)
        style_axes(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(**legend_style(handles, labels, loc="upper center", ncols=len(labels)))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save_and_sync_figure(fig, env, "cr_curve")
    plt.close(fig)


def _recon_tradeoff_frame() -> tuple[pd.DataFrame, object]:
    runs, env = _load_runs()
    tbl = _main_experiment_rows(_base_columns(runs))
    if tbl.empty:
        raise RuntimeError("No runs available for reconstruction trade-off plot.")
    tensor_size = int(np.prod(_dataset_from_row(tbl.iloc[0]).get("eval").shape))

    rows: list[dict[str, Any]] = []
    for rank in RANK_ORDER:
        rank_rows = tbl.loc[tbl["rank"] == rank].copy()
        for method in METHOD_ORDER:
            subset = rank_rows.loc[rank_rows["method_name"] == method].copy()
            if subset.empty:
                continue
            params = tensor_size / subset["CR"].to_numpy(dtype=float)
            rows.append(
                {
                    "rank": rank,
                    "rank_text": _rank_text(rank),
                    "method_name": method,
                    "Method": env.label_for_method(method),
                    "Params": float(params.mean()),
                    "RMSE": float(subset["RMSE"].mean()),
                    "SAM": float(subset["SAM"].mean()),
                    "Time": float(subset["fit_time_sec"].mean()),
                    "RMSE_std": float(subset["RMSE"].std(ddof=0)),
                    "SAM_std": float(subset["SAM"].std(ddof=0)),
                    "Time_std": float(subset["fit_time_sec"].std(ddof=0)),
                    "Scenes": int(subset["scene_id"].nunique()),
                }
            )
    frame = pd.DataFrame(rows).sort_values(["rank_text", "method_name"]).reset_index(drop=True)
    if frame.empty:
        raise RuntimeError("No rows available for reconstruction trade-off plot.")
    csv_cols = ["rank_text", "Method", "Params", "Time", "RMSE", "SAM", "Scenes"]
    csv_path, latex_csv_path = write_csv_artifact(env, frame.loc[:, csv_cols], "tradeoff_points.csv")
    print(f"Saved: {csv_path}")
    print(f"Synced: {latex_csv_path}")
    return frame, env


def _rank_marker(rank: tuple[int, int, int]) -> str:
    return {
        (8, 8, 4): "o",
        (12, 12, 6): "s",
        (16, 16, 8): "^",
    }[rank]


def _tradeoff_frontier(frame: pd.DataFrame, *, x_col: str, y_col: str) -> pd.DataFrame:
    points = frame.loc[:, [x_col, y_col]].to_numpy(dtype=float)
    keep: list[int] = []
    for i, point in enumerate(points):
        dominated = False
        for j, other in enumerate(points):
            if i == j:
                continue
            if (other[0] <= point[0] and other[1] <= point[1]) and (other[0] < point[0] or other[1] < point[1]):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return frame.iloc[keep].sort_values(x_col)


def _tradeoff_method_style(method_label: str) -> dict[str, Any]:
    if method_label == "NTD-PL":
        return {"color": PALETTE.ntdpl, "alpha": 0.98, "size": 92, "edgecolor": PALETTE.black, "linewidth": 0.8, "zorder": 5}
    if method_label == "Tucker":
        return {"color": PALETTE.tucker, "alpha": 0.95, "size": 82, "edgecolor": PALETTE.border, "linewidth": 0.7, "zorder": 4}
    style = method_style(method_label)
    return {
        "color": style["color"],
        "alpha": 0.68,
        "size": 62,
        "edgecolor": "white",
        "linewidth": 0.5,
        "zorder": 3,
    }


def _annotate_tradeoff_pair(ax: plt.Axes, frame: pd.DataFrame, *, rank: tuple[int, int, int], x_col: str, y_col: str, text: str, dx: float, dy: float) -> None:
    rank_frame = frame.loc[frame["rank"] == rank].copy()
    tucker = rank_frame.loc[rank_frame["Method"] == "Tucker"].iloc[0]
    ntdpl = rank_frame.loc[rank_frame["Method"] == "NTD-PL"].iloc[0]
    ax.annotate(
        "",
        xy=(float(ntdpl[x_col]), float(ntdpl[y_col])),
        xytext=(float(tucker[x_col]), float(tucker[y_col])),
        arrowprops={"arrowstyle": "-|>", "color": PALETTE.ntdpl, "lw": 1.0, "alpha": 0.9},
        zorder=2,
    )
    xm = 0.5 * (float(tucker[x_col]) + float(ntdpl[x_col]))
    ym = 0.5 * (float(tucker[y_col]) + float(ntdpl[y_col]))
    ax.text(
        xm + dx,
        ym + dy,
        text,
        fontsize=7.8,
        color=PALETTE.ntdpl,
        ha="left",
        va="center",
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": PALETTE.grid, "alpha": 0.95},
    )


def _plot_tradeoff_panel(
    ax: plt.Axes,
    *,
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    xlabel: str,
    ylabel: str,
    title: str,
    log_x: bool = False,
    show_frontier: bool = False,
) -> None:
    if show_frontier:
        frontier = _tradeoff_frontier(frame, x_col=x_col, y_col=y_col)
        ax.plot(
            frontier[x_col].to_numpy(dtype=float),
            frontier[y_col].to_numpy(dtype=float),
            color=PALETTE.grid,
            linewidth=1.1,
            linestyle="-",
            zorder=1,
        )
        ax.text(
            float(frontier[x_col].iloc[-1]),
            float(frontier[y_col].iloc[-1]),
            " Pareto frontier",
            fontsize=7.9,
            color=PALETTE.border,
            va="center",
        )

    for _, row in frame.iterrows():
        point_style = _tradeoff_method_style(str(row["Method"]))
        ax.scatter(
            float(row[x_col]),
            float(row[y_col]),
            marker=_rank_marker(tuple(row["rank"])),
            s=float(point_style["size"]),
            color=point_style["color"],
            alpha=float(point_style["alpha"]),
            edgecolors=str(point_style["edgecolor"]),
            linewidths=float(point_style["linewidth"]),
            zorder=int(point_style["zorder"]),
        )
        if str(row["Method"]) in {"Tucker", "NTD-PL"}:
            x_offset = 0.012 * float(frame[x_col].max()) if not log_x else 0.03 * float(row[x_col])
            y_offset = -0.004 if y_col == "RMSE" else -0.22
            ax.text(
                float(row[x_col]) + x_offset,
                float(row[y_col]) + y_offset,
                f"{row['Method']} {row['rank_text']}",
                fontsize=7.7,
                color=PALETTE.border,
                ha="left",
                va="center",
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=5)
    if log_x:
        ax.set_xscale("log")
    style_axes(ax, grid=True)


def _tradeoff_legend(fig: plt.Figure) -> None:
    method_handles = []
    for label in ("Tucker", "NTD-PL", "CP", "TT", "TR"):
        point_style = _tradeoff_method_style(label)
        method_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=7.4 if label == "NTD-PL" else 6.6,
                markerfacecolor=point_style["color"],
                markeredgecolor=point_style["edgecolor"],
                markeredgewidth=point_style["linewidth"],
                alpha=point_style["alpha"],
                label=label,
            )
        )
    rank_handles = [
        Line2D([0], [0], marker=_rank_marker(rank), linestyle="None", color=PALETTE.border, markersize=6.4, label=_rank_text(rank))
        for rank in RANK_ORDER
    ]
    fig.legend(
        **legend_style(
            method_handles,
            [h.get_label() for h in method_handles],
            loc="upper center",
            ncols=len(method_handles),
            bbox_to_anchor=(0.37, 1.02),
        )
    )
    fig.legend(
        **legend_style(
            rank_handles,
            [h.get_label() for h in rank_handles],
            loc="upper center",
            ncols=len(rank_handles),
            bbox_to_anchor=(0.83, 1.02),
        )
    )


def tradeoff_pareto_plot() -> None:
    frame, env = _recon_tradeoff_frame()

    apply_theme()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), constrained_layout=True)
    _plot_tradeoff_panel(
        axes[0],
        frame=frame,
        x_col="Params",
        y_col="RMSE",
        xlabel="Parameters",
        ylabel="RMSE",
        title="Performance vs parameter budget",
        show_frontier=True,
    )
    _plot_tradeoff_panel(
        axes[1],
        frame=frame,
        x_col="Time",
        y_col="RMSE",
        xlabel="Training time (s)",
        ylabel="RMSE",
        title="Performance vs optimization cost",
        log_x=True,
    )

    _annotate_tradeoff_pair(axes[0], frame, rank=(8, 8, 4), x_col="Params", y_col="RMSE", text="+7 params,\nbetter RMSE", dx=65.0, dy=-0.006)
    _annotate_tradeoff_pair(axes[0], frame, rank=(12, 12, 6), x_col="Params", y_col="RMSE", text="+7 params,\nlarge gain", dx=70.0, dy=-0.008)
    _annotate_tradeoff_pair(axes[1], frame, rank=(12, 12, 6), x_col="Time", y_col="RMSE", text="accuracy gain\ncosts extra training", dx=0.38, dy=-0.006)

    axes[0].text(
        0.03,
        0.97,
        "NTD-PL stays near Tucker in parameter count\nwhile moving to a better accuracy region.",
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        fontsize=8.0,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": PALETTE.grid, "alpha": 0.96},
    )
    axes[1].text(
        0.03,
        0.97,
        "Time is the main extra cost:\nnonlinear gain comes from optimization, not from many more parameters.",
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        fontsize=8.0,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": PALETTE.grid, "alpha": 0.96},
    )

    _tradeoff_legend(fig)
    _save_and_sync_figure(fig, env, "tradeoff_pareto_main", formats=("pdf", "png"), dpi=600)
    plt.close(fig)

    apply_theme()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), constrained_layout=True)
    _plot_tradeoff_panel(
        axes[0],
        frame=frame,
        x_col="Params",
        y_col="SAM",
        xlabel="Parameters",
        ylabel="SAM (deg)",
        title="SAM vs parameter budget",
        show_frontier=True,
    )
    _plot_tradeoff_panel(
        axes[1],
        frame=frame,
        x_col="Time",
        y_col="SAM",
        xlabel="Training time (s)",
        ylabel="SAM (deg)",
        title="SAM vs optimization cost",
        log_x=True,
    )
    _tradeoff_legend(fig)
    _save_and_sync_figure(fig, env, "tradeoff_pareto_sam", formats=("pdf", "png"), dpi=600)
    plt.close(fig)


def _resolve_target(target_path: str) -> type[Any]:
    module_name, _, attr = target_path.rpartition(".")
    if not module_name:
        raise ValueError(f"Invalid target path: {target_path}")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def _cfg_from_row(row: pd.Series, prefix: str) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    for key, value in row.items():
        if not key.startswith(prefix):
            continue
        field = key.removeprefix(prefix)
        parsed = _jsonish(value)
        if isinstance(parsed, list):
            parsed = tuple(parsed)
        cfg[field] = parsed
    return cfg


def _dataset_from_row(row: pd.Series) -> CAVEHSIData:
    data_cfg = _cfg_from_row(row, "data.")
    if "path" in data_cfg:
        data_cfg["path"] = str((PROJECT_ROOT / str(data_cfg["path"])).resolve())
    data_target = str(data_cfg.pop("_target_", "src.data.hsi.CAVEHSIData"))
    data_cls = _resolve_target(data_target)
    dataset = data_cls(**data_cfg)

    filter_cfg = _cfg_from_row(row, "filter.")
    filter_target = str(filter_cfg.pop("_target_", "src.filters.BiasFilter"))
    filter_cls = _resolve_target(filter_target)
    data_filter = filter_cls(**filter_cfg)
    data_filter(dataset)
    return dataset


def _load_reconstruction(row: pd.Series) -> np.ndarray:
    method_name = str(row["method_name"])
    method_cls = METHOD_CLASSES[method_name]
    method = method_cls()
    run_dir = Path(str(row["run_dir"]))
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    result = Result.load(run_dir)
    method.load_state_dict(result.state_dict)
    return np.asarray(method.reconstruct().dense, dtype=np.float32)


def _load_method_state(row: pd.Series) -> Any:
    method_name = str(row["method_name"])
    method_cls = METHOD_CLASSES[method_name]
    method = method_cls()
    run_dir = Path(str(row["run_dir"]))
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    result = Result.load(run_dir)
    method.load_state_dict(result.state_dict)
    return method


def _latent_q_from_method(method: Any) -> np.ndarray:
    core = np.asarray(method.core, dtype=np.float32)
    u1 = np.asarray(method.factors[0], dtype=np.float32)
    u2 = np.asarray(method.factors[1], dtype=np.float32)
    return np.asarray(np.einsum("abc,ia,jb->ijc", core, u1, u2), dtype=np.float32)


def _prelink_spectral_from_method(method: Any) -> np.ndarray:
    core = np.asarray(method.core, dtype=np.float32)
    factors = [np.asarray(f, dtype=np.float32) for f in method.factors]
    return np.asarray(tucker_to_tensor((core, factors)), dtype=np.float32)


def _try_load_reconstruction(row: pd.Series) -> np.ndarray | None:
    try:
        return _load_reconstruction(row)
    except Exception:
        return None


def _pseudo_rgb(cube: np.ndarray) -> np.ndarray:
    band_count = cube.shape[-1]
    indices = [int(round((band_count - 1) * frac)) for frac in (0.75, 0.5, 0.2)]
    rgb = np.stack([cube[..., idx] for idx in indices], axis=-1)
    rgb = np.clip(rgb, 0.0, None)
    scale = float(np.max(rgb))
    if scale > 1e-12:
        rgb = rgb / scale
    return rgb


def _sample_features(x_ref: np.ndarray, z: np.ndarray, max_points: int = MANIFOLD_MAX_POINTS) -> tuple[np.ndarray, np.ndarray]:
    total = x_ref.shape[0] * x_ref.shape[1]
    indices = np.arange(total)
    if total > max_points:
        rng = np.random.default_rng(0)
        indices = np.sort(rng.choice(indices, size=max_points, replace=False))
    x = x_ref.reshape(-1, x_ref.shape[-1])[indices].astype(np.float32)
    y = z.reshape(-1, z.shape[-1])[indices].astype(np.float32)
    return x, y


def _knn_order(x: np.ndarray) -> np.ndarray:
    dist = cdist(x, x, metric="euclidean")
    np.fill_diagonal(dist, np.inf)
    return np.argsort(dist, axis=1)


def _continuity(x_ref: np.ndarray, z: np.ndarray, k: int) -> float:
    order_ref = _knn_order(x_ref)
    order_z = _knn_order(z)
    n = x_ref.shape[0]
    rank_z = np.empty((n, n), dtype=np.int32)
    full_ranks = np.broadcast_to(np.arange(1, n + 1, dtype=np.int32), (n, n))
    rank_z[np.arange(n)[:, None], order_z] = full_ranks
    penalty = 0.0
    for i in range(n):
        z_neighbors = set(order_z[i, :k].tolist())
        for j in order_ref[i, :k]:
            if j not in z_neighbors:
                penalty += float(rank_z[i, j] - k)
    norm = 2.0 / (n * k * (2.0 * n - 3.0 * k - 1.0))
    return float(1.0 - norm * penalty)


def _knn_overlap(x_ref: np.ndarray, z: np.ndarray, k: int) -> float:
    order_ref = _knn_order(x_ref)[:, :k]
    order_z = _knn_order(z)[:, :k]
    scores = []
    for i in range(x_ref.shape[0]):
        overlap = len(set(order_ref[i].tolist()) & set(order_z[i].tolist()))
        scores.append(overlap / float(k))
    return float(np.mean(scores))


def _manifold_metrics(x_ref: np.ndarray, z: np.ndarray, k: int = MANIFOLD_K) -> dict[str, float]:
    x_std = StandardScaler().fit_transform(x_ref)
    z_std = StandardScaler().fit_transform(z)
    return {
        "Trustworthiness": float(trustworthiness(x_std, z_std, n_neighbors=k)),
        "Continuity": _continuity(x_std, z_std, k=k),
        "kNNOverlap": _knn_overlap(x_std, z_std, k=k),
    }


def _representative_scene(tbl: pd.DataFrame, rank: tuple[int, int, int]) -> int:
    subset = tbl.loc[(tbl["rank"] == rank) & (tbl["method_name"].isin(METHOD_FOCUS_ORDER))].copy()
    pivot = subset.pivot_table(index="scene_id", columns="method_name", values="SAM", aggfunc="mean")
    pivot["gain"] = pivot["tucker"] - pivot["ntdpl"]
    return int(pivot["gain"].idxmax())


def _representative_scenes(tbl: pd.DataFrame, rank: tuple[int, int, int], num_scenes: int = 3) -> list[int]:
    subset = tbl.loc[(tbl["rank"] == rank) & (tbl["method_name"].isin(METHOD_FOCUS_ORDER))].copy()
    pivot = subset.pivot_table(index="scene_id", columns="method_name", values="SAM", aggfunc="mean")
    pivot["gain"] = pivot["tucker"] - pivot["ntdpl"]
    ordered = pivot.sort_values("gain", ascending=False).index.tolist()
    return [int(scene_id) for scene_id in ordered[:num_scenes]]


def manifold_rank_scan_table() -> None:
    runs, env = _load_runs()
    tbl = _main_experiment_rows(_base_columns(runs))
    records: list[dict[str, Any]] = []

    for rank in RANK_ORDER:
        rank_rows = tbl.loc[(tbl["rank"] == rank) & (tbl["method_name"].isin(METHOD_FOCUS_ORDER))].copy()
        for scene_id in sorted(rank_rows["scene_id"].unique().tolist()):
            scene_rows = rank_rows.loc[rank_rows["scene_id"] == scene_id].copy()
            if scene_rows["method_name"].nunique() < 2:
                continue
            dataset = _dataset_from_row(scene_rows.iloc[0])
            x_ref = np.asarray(dataset.get("eval").dense, dtype=np.float32)
            for method_name in METHOD_FOCUS_ORDER:
                row = scene_rows.loc[scene_rows["method_name"] == method_name]
                if row.empty:
                    continue
                method = _load_method_state(row.iloc[0])
                reps = {
                    "q": _latent_q_from_method(method),
                    "s": _prelink_spectral_from_method(method),
                }
                for rep_name, rep_value in reps.items():
                    x_sample, z_sample = _sample_features(x_ref, rep_value)
                    metrics = _manifold_metrics(x_sample, z_sample, k=MANIFOLD_K)
                    records.append(
                        {
                            "Rank": _rank_text(rank),
                            "Scene": int(scene_id),
                            "Method": env.label_for_method(method_name),
                            "Representation": rep_name,
                            **metrics,
                        }
                    )

    summary = pd.DataFrame(records)
    if summary.empty:
        raise RuntimeError("No rows available for CAVE manifold rank scan.")

    aggregated = (
        summary.groupby(["Rank", "Method", "Representation"], as_index=False)[["Trustworthiness", "Continuity", "kNNOverlap"]]
        .mean()
        .sort_values(["Rank", "Method", "Representation"])
    )

    csv_path, _ = write_csv_artifact(env, aggregated, "manifold_rank_scan.csv")

    tex_lines = [
        r"\begin{tabular}{lllccc}",
        r"\toprule",
        r"Rank & Method & Rep. & Trustworthiness$\uparrow$ & Continuity$\uparrow$ & kNN overlap$\uparrow$ \\",
        r"\midrule",
    ]
    for _, row in aggregated.iterrows():
        tex_lines.append(
            f"{row['Rank']} & {row['Method']} & {row['Representation']} & "
            f"{row['Trustworthiness']:.4f} & {row['Continuity']:.4f} & {row['kNNOverlap']:.4f} \\\\"
        )
    tex_lines.extend([r"\bottomrule", r"\end{tabular}"])
    tex_path, _ = write_text_artifact(env, "\n".join(tex_lines) + "\n", "manifold_rank_scan.tex")

    print(aggregated.to_string(index=False))
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {tex_path}")


def significance_table() -> None:
    runs, env = _load_runs()
    tbl = _main_experiment_rows(_base_columns(runs))
    rows: list[dict[str, Any]] = []

    for rank in RANK_ORDER:
        subset = tbl.loc[(tbl["rank"] == rank) & (tbl["method_name"].isin(METHOD_FOCUS_ORDER))].copy()
        if subset.empty:
            continue
        pivot = subset.pivot_table(index="scene_id", columns="method_name", values=["RMSE", "SAM"], aggfunc="mean")
        for metric in ("RMSE", "SAM"):
            diffs = (
                pivot[(metric, "tucker")] - pivot[(metric, "ntdpl")]
            ).dropna().to_numpy(dtype=float)
            wins = int(np.sum(diffs > 0))
            ties = int(np.sum(np.isclose(diffs, 0.0)))
            losses = int(np.sum(diffs < 0))
            nonzero = diffs[~np.isclose(diffs, 0.0)]
            sign_p = 1.0 if nonzero.size == 0 else float(binomtest(int(np.sum(nonzero > 0)), nonzero.size, 0.5).pvalue)
            try:
                wilcoxon_p = 1.0 if nonzero.size == 0 else float(wilcoxon(nonzero, zero_method="wilcox").pvalue)
            except ValueError:
                wilcoxon_p = 1.0
            rows.append(
                {
                    "Rank": _rank_text(rank),
                    "Metric": metric,
                    "Wins / Ties / Losses": f"{wins} / {ties} / {losses}",
                    "Mean Delta": float(diffs.mean()),
                    "Median Delta": float(np.median(diffs)),
                    "Sign Test P": sign_p,
                    "Wilcoxon P": wilcoxon_p,
                }
            )

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise RuntimeError("No rows available for CAVE significance table.")

    csv_path, latex_csv_path = write_csv_artifact(env, summary, "significance.csv")
    tex_lines = [
        r"\begin{tabular}{c|c|c|c|c|c|c}",
        r"\toprule",
        r"Rank & Metric & Wins / Ties / Losses & Mean $\Delta$ & Median $\Delta$ & Sign test $p$ & Wilcoxon $p$ \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        tex_lines.append(
            f"{_rank_pub_text(row['Rank'])} & {row['Metric']} & {row['Wins / Ties / Losses']} & "
            f"{_latex_number(row['Mean Delta'], 6)} & {_latex_number(row['Median Delta'], 6)} & "
            f"{_latex_pvalue(row['Sign Test P'])} & {_latex_pvalue(row['Wilcoxon P'])} \\\\"
        )
    tex_lines.extend([r"\bottomrule", r"\end{tabular}"])
    tex_text = "\n".join(tex_lines) + "\n"
    tex_path, latex_tex_path = write_text_artifact(env, tex_text, "significance_pub.tex")

    print(summary.to_string(index=False))
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {tex_path}")
    print(f"Synced: {latex_csv_path}")
    print(f"Synced: {latex_tex_path}")


def main_rank_significance_summary() -> None:
    runs, env = _load_runs()
    tbl = _main_experiment_rows(_base_columns(runs))
    subset = tbl.loc[(tbl["rank"] == OVERVIEW_RANK) & (tbl["method_name"].isin(METHOD_FOCUS_ORDER))].copy()
    if subset.empty:
        raise RuntimeError("No rows available for the main-rank CAVE significance summary.")

    pivot = subset.pivot_table(index="scene_id", columns="method_name", values=["RMSE", "NMSE_dB", "SAM"], aggfunc="mean")
    metric_display = {"RMSE": "RMSE", "NMSE_dB": "NMSE(dB)", "SAM": "SAM"}
    rows: list[dict[str, Any]] = []
    for metric in ("RMSE", "NMSE_dB", "SAM"):
        diffs = (pivot[(metric, "tucker")] - pivot[(metric, "ntdpl")]).dropna().to_numpy(dtype=float)
        wins = int(np.sum(diffs > 0.0))
        ties = int(np.sum(np.isclose(diffs, 0.0)))
        losses = int(np.sum(diffs < 0.0))
        nonzero = diffs[~np.isclose(diffs, 0.0)]
        sign_p = 1.0 if nonzero.size == 0 else float(binomtest(int(np.sum(nonzero > 0.0)), nonzero.size, 0.5).pvalue)
        try:
            wilcoxon_p = 1.0 if nonzero.size == 0 else float(wilcoxon(nonzero, zero_method="wilcox").pvalue)
        except ValueError:
            wilcoxon_p = 1.0
        rows.append(
            {
                "Task": "Full observation",
                "Metric": metric_display[metric],
                "Win/Loss/Tie": f"{wins}/{losses}/{ties}",
                "Mean gain": float(diffs.mean()),
                "Median gain": float(np.median(diffs)),
                "Sign test p": sign_p,
                "Wilcoxon p": wilcoxon_p,
                "Rank-biserial": float(_rank_biserial(diffs)),
            }
        )

    summary = pd.DataFrame(rows)
    csv_path, latex_csv_path = write_csv_artifact(env, summary, "recon_significance_summary.csv")
    tex_lines = [
        r"\begin{tabular}{c c c c c}",
        r"\toprule",
        r"Metric & Win/Loss/Tie & Median gain & Sign test $p$ & Wilcoxon $p$ \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        tex_lines.append(
            f"{row['Metric']} & {row['Win/Loss/Tie']} & "
            f"{float(row['Median gain']):.4f} & "
            f"{_latex_pvalue(float(row['Sign test p']))} & {_latex_pvalue(float(row['Wilcoxon p']))} \\\\"
        )
    tex_lines.extend([r"\bottomrule", r"\end{tabular}"])
    tex_path, latex_tex_path = write_text_artifact(env, "\n".join(tex_lines) + "\n", "recon_significance_summary.tex")

    print(summary.to_string(index=False))
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {tex_path}")
    print(f"Synced: {latex_csv_path}")
    print(f"Synced: {latex_tex_path}")


@dataclass(frozen=True)
class _SpectralScenePayload:
    scene_id: int
    scene_name: str
    original: np.ndarray
    recon_tucker: np.ndarray
    recon_ntdpl: np.ndarray
    intensity: np.ndarray
    gradient: np.ndarray
    spectral_peak: np.ndarray
    rmse_tucker: np.ndarray
    rmse_ntdpl: np.ndarray
    sam_tucker: np.ndarray
    sam_ntdpl: np.ndarray


@dataclass(frozen=True)
class _PixelCandidate:
    category: str
    scene_id: int
    scene_name: str
    row: int
    col: int
    score: float
    rmse_tucker: float
    rmse_ntdpl: float
    sam_tucker: float
    sam_ntdpl: float
    gradient: float
    mean_intensity: float
    peak_value: float


def _spectral_sam_deg(reference: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    numerator = np.sum(reference * estimate, axis=-1)
    denominator = np.linalg.norm(reference, axis=-1) * np.linalg.norm(estimate, axis=-1)
    cosine = np.clip(numerator / np.maximum(denominator, 1e-12), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def _spatial_gradient_map(image: np.ndarray) -> np.ndarray:
    grad_y, grad_x = np.gradient(image.astype(np.float32), edge_order=1)
    return np.sqrt(grad_x**2 + grad_y**2)


@lru_cache(maxsize=64)
def _spectral_scene_payload(rank: tuple[int, int, int], scene_id: int) -> _SpectralScenePayload:
    runs, _ = _load_runs()
    tbl = _main_experiment_rows(_base_columns(runs))
    subset = tbl.loc[
        (tbl["rank"] == rank)
        & (tbl["scene_id"] == scene_id)
        & (tbl["method_name"].isin(METHOD_FOCUS_ORDER))
    ].copy()
    if subset.empty or subset["method_name"].nunique() < 2:
        raise RuntimeError(f"Missing Tucker/NTD-PL runs for scene {scene_id} at rank {rank}.")

    dataset = _dataset_from_row(subset.iloc[0])
    original = np.asarray(dataset.get("eval").dense, dtype=np.float32)
    recon_tucker = _load_reconstruction(subset.loc[subset["method_name"] == "tucker"].iloc[0])
    recon_ntdpl = _load_reconstruction(subset.loc[subset["method_name"] == "ntdpl"].iloc[0])

    intensity = np.mean(original, axis=-1)
    gradient = _spatial_gradient_map(intensity)
    spectral_peak = np.max(original, axis=-1)
    rmse_tucker = np.sqrt(np.mean((original - recon_tucker) ** 2, axis=-1))
    rmse_ntdpl = np.sqrt(np.mean((original - recon_ntdpl) ** 2, axis=-1))
    sam_tucker = _spectral_sam_deg(original, recon_tucker)
    sam_ntdpl = _spectral_sam_deg(original, recon_ntdpl)

    return _SpectralScenePayload(
        scene_id=scene_id,
        scene_name=dataset.scene_name,
        original=original,
        recon_tucker=recon_tucker,
        recon_ntdpl=recon_ntdpl,
        intensity=intensity,
        gradient=gradient,
        spectral_peak=spectral_peak,
        rmse_tucker=rmse_tucker,
        rmse_ntdpl=rmse_ntdpl,
        sam_tucker=sam_tucker,
        sam_ntdpl=sam_ntdpl,
    )


def _best_scene_candidates_for_category(payload: _SpectralScenePayload, category: str) -> list[_PixelCandidate]:
    def _scaled_positive(values: np.ndarray, q: float = 0.9, eps: float = 1e-12) -> np.ndarray:
        scale = float(np.quantile(values, q))
        return values / max(scale, eps)

    def _scaled_gain(values: np.ndarray, q: float = 0.9, eps: float = 1e-12) -> np.ndarray:
        positive = values[values > 0]
        if positive.size == 0:
            scale = 1.0
        else:
            scale = float(np.quantile(positive, q))
        return values / max(scale, eps)

    intensity = payload.intensity
    gradient = payload.gradient
    peak = payload.spectral_peak
    rmse_t = payload.rmse_tucker
    rmse_n = payload.rmse_ntdpl
    sam_t = payload.sam_tucker
    sam_n = payload.sam_ntdpl
    rmse_gain = rmse_t - rmse_n
    sam_gain = sam_t - sam_n
    grad_scaled = _scaled_positive(gradient, q=0.9)
    rmse_t_scaled = _scaled_positive(rmse_t, q=0.85)
    rmse_n_scaled = _scaled_positive(rmse_n, q=0.85)
    sam_t_scaled = _scaled_positive(sam_t, q=0.85)
    sam_n_scaled = _scaled_positive(sam_n, q=0.85)
    rmse_gain_scaled = _scaled_gain(rmse_gain, q=0.9)
    sam_gain_scaled = _scaled_gain(sam_gain, q=0.9)

    valid_signal = (
        (intensity >= float(np.quantile(intensity, 0.15)))
        & (intensity <= np.quantile(intensity, 0.93))
        & (peak >= float(np.quantile(peak, 0.35)))
    )

    if category == "smooth":
        mask = (
            valid_signal
            & (gradient <= np.quantile(gradient, 0.22))
            & (rmse_t <= np.quantile(rmse_t, 0.40))
            & (rmse_n <= np.quantile(rmse_n, 0.40))
            & (sam_t <= np.quantile(sam_t, 0.55))
            & (sam_n <= np.quantile(sam_n, 0.55))
            & (np.maximum(sam_t, sam_n) <= np.quantile(np.maximum(sam_t, sam_n), 0.60))
        )
        score = -(0.95 * (rmse_t_scaled + rmse_n_scaled) + 0.55 * (sam_t_scaled + sam_n_scaled) + 0.70 * grad_scaled)
    elif category == "boundary":
        mask = (
            valid_signal
            & (gradient >= np.quantile(gradient, 0.90))
            & ((0.5 * (rmse_t + rmse_n)) >= np.quantile(0.5 * (rmse_t + rmse_n), 0.45))
            & (np.maximum(rmse_t, rmse_n) <= np.quantile(np.maximum(rmse_t, rmse_n), 0.80))
            & (np.maximum(sam_t, sam_n) <= np.quantile(np.maximum(sam_t, sam_n), 0.72))
        )
        score = grad_scaled + 0.40 * (rmse_t_scaled + rmse_n_scaled) + 0.25 * (sam_t_scaled + sam_n_scaled) - 0.35 * np.abs(rmse_gain_scaled)
    elif category == "ntdpl_better":
        mask = (
            valid_signal
            & (rmse_gain >= np.quantile(rmse_gain, 0.992))
            & (sam_gain >= np.quantile(sam_gain, 0.85))
            & (rmse_n <= np.quantile(rmse_n, 0.72))
            & (sam_n <= np.quantile(sam_n, 0.72))
        )
        score = rmse_gain_scaled + 0.60 * sam_gain_scaled - 0.30 * rmse_n_scaled - 0.22 * sam_n_scaled
    else:
        raise ValueError(f"Unsupported category: {category}")

    points = np.argwhere(mask)
    if len(points) == 0:
        return []

    candidates: list[_PixelCandidate] = []
    for row, col in points:
        r = int(row)
        c = int(col)
        candidates.append(
            _PixelCandidate(
                category=category,
                scene_id=payload.scene_id,
                scene_name=payload.scene_name,
                row=r,
                col=c,
                score=float(score[r, c]),
                rmse_tucker=float(rmse_t[r, c]),
                rmse_ntdpl=float(rmse_n[r, c]),
                sam_tucker=float(sam_t[r, c]),
                sam_ntdpl=float(sam_n[r, c]),
                gradient=float(gradient[r, c]),
                mean_intensity=float(intensity[r, c]),
                peak_value=float(peak[r, c]),
            )
        )
    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates[:8]


def _pick_distinct_candidates(
    candidates: list[_PixelCandidate],
    count: int,
    used_scenes: set[int] | None = None,
) -> list[_PixelCandidate]:
    chosen: list[_PixelCandidate] = []
    occupied = set() if used_scenes is None else set(used_scenes)
    for candidate in candidates:
        if len(chosen) >= count:
            break
        if candidate.scene_id in occupied:
            continue
        chosen.append(candidate)
        occupied.add(candidate.scene_id)
    if len(chosen) < count:
        for candidate in candidates:
            if len(chosen) >= count:
                break
            if candidate in chosen:
                continue
            if used_scenes is not None and candidate.scene_id in occupied:
                continue
            chosen.append(candidate)
    return chosen


def _select_spectral_v2_pixels(rank: tuple[int, int, int]) -> list[_PixelCandidate]:
    runs, _ = _load_runs()
    tbl = _main_experiment_rows(_base_columns(runs))
    scene_ids = sorted(tbl.loc[tbl["rank"] == rank, "scene_id"].unique().tolist())

    grouped: dict[str, list[_PixelCandidate]] = {"ntdpl_better": []}
    for scene_id in scene_ids:
        payload = _spectral_scene_payload(rank, int(scene_id))
        grouped["ntdpl_better"].extend(_best_scene_candidates_for_category(payload, "ntdpl_better"))

    grouped["ntdpl_better"].sort(key=lambda item: item.score, reverse=True)
    if not grouped["ntdpl_better"]:
        raise RuntimeError(f"No candidate pixels found for category `ntdpl_better` at rank {rank}.")

    used_scenes: set[int] = set()
    selected = _pick_distinct_candidates(grouped["ntdpl_better"], count=4, used_scenes=used_scenes)
    return selected


def _curve_ylim(curves: list[np.ndarray]) -> tuple[float, float]:
    values = np.concatenate([np.asarray(curve, dtype=float) for curve in curves], axis=0)
    lo = float(np.quantile(values, 0.01))
    hi = float(np.quantile(values, 0.99))
    span = max(hi - lo, 0.04)
    y_min = lo - 0.12 * span
    y_max = hi + 0.15 * span
    if y_max - y_min < 0.05:
        center = 0.5 * (y_min + y_max)
        half = 0.03
        y_min = center - half
        y_max = center + half
    return y_min, y_max


def _pixel_candidate_from_payload(
    payload: _SpectralScenePayload,
    row: int,
    col: int,
    *,
    category: str,
    score: float,
) -> _PixelCandidate:
    return _PixelCandidate(
        category=category,
        scene_id=payload.scene_id,
        scene_name=payload.scene_name,
        row=int(row),
        col=int(col),
        score=float(score),
        rmse_tucker=float(payload.rmse_tucker[row, col]),
        rmse_ntdpl=float(payload.rmse_ntdpl[row, col]),
        sam_tucker=float(payload.sam_tucker[row, col]),
        sam_ntdpl=float(payload.sam_ntdpl[row, col]),
        gradient=float(payload.gradient[row, col]),
        mean_intensity=float(payload.intensity[row, col]),
        peak_value=float(payload.spectral_peak[row, col]),
    )


def _select_scene_spectral_points(payload: _SpectralScenePayload) -> list[_PixelCandidate]:
    intensity = payload.intensity
    gradient = payload.gradient
    rmse_gain = payload.rmse_tucker - payload.rmse_ntdpl
    sam_gain = payload.sam_tucker - payload.sam_ntdpl
    valid_signal = (
        (intensity >= np.quantile(intensity, 0.20))
        & (intensity <= np.quantile(intensity, 0.93))
        & (payload.spectral_peak >= np.quantile(payload.spectral_peak, 0.25))
    )

    high_mask = (
        valid_signal
        & (rmse_gain >= np.quantile(rmse_gain, 0.94))
        & (sam_gain >= np.quantile(sam_gain, 0.72))
        & (gradient >= np.quantile(gradient, 0.65))
    )
    if not np.any(high_mask):
        high_mask = valid_signal & (rmse_gain >= np.quantile(rmse_gain, 0.90))
    high_score = rmse_gain + 0.30 * sam_gain + 0.15 * gradient
    high_points = np.argwhere(high_mask)
    high_points = sorted(high_points.tolist(), key=lambda rc: float(high_score[rc[0], rc[1]]), reverse=True)
    if not high_points:
        raise RuntimeError(f"No spectral highlight pixel found for scene {payload.scene_id}.")
    r_hi, c_hi = high_points[0]
    highlight = _pixel_candidate_from_payload(payload, r_hi, c_hi, category="ntdpl_better", score=high_score[r_hi, c_hi])

    medium_mask = (
        valid_signal
        & (rmse_gain >= np.quantile(rmse_gain, 0.68))
        & (rmse_gain <= np.quantile(rmse_gain, 0.90))
        & (sam_gain >= np.quantile(sam_gain, 0.45))
        & (gradient >= np.quantile(gradient, 0.45))
        & (gradient <= np.quantile(gradient, 0.88))
    )
    if not np.any(medium_mask):
        medium_mask = valid_signal & (rmse_gain >= np.quantile(rmse_gain, 0.60))
    rr, cc = np.indices(intensity.shape)
    distance_penalty = np.sqrt((rr - highlight.row) ** 2 + (cc - highlight.col) ** 2)
    typical_score = rmse_gain + 0.20 * sam_gain + 0.10 * gradient + 0.015 * distance_penalty
    medium_points = np.argwhere(medium_mask)
    medium_points = sorted(medium_points.tolist(), key=lambda rc: float(typical_score[rc[0], rc[1]]), reverse=True)
    typical = None
    for r_md, c_md in medium_points:
        if float(distance_penalty[r_md, c_md]) >= 12.0:
            typical = _pixel_candidate_from_payload(payload, r_md, c_md, category="typical", score=typical_score[r_md, c_md])
            break
    if typical is None:
        typical = _pixel_candidate_from_payload(payload, medium_points[0][0], medium_points[0][1], category="typical", score=typical_score[medium_points[0][0], medium_points[0][1]])
    return [highlight, typical]


def _spectral_main_selection(rank: tuple[int, int, int]) -> list[_PixelCandidate]:
    selected: list[_PixelCandidate] = []
    for scene_id in VISUAL_MAIN_SCENES:
        payload = _spectral_scene_payload(rank, int(scene_id))
        selected.extend(_select_scene_spectral_points(payload))
    return selected


def _spectral_appendix_selection(rank: tuple[int, int, int]) -> list[_PixelCandidate]:
    selected: list[_PixelCandidate] = []
    for scene_id in VISUAL_APPENDIX_SCENES:
        payload = _spectral_scene_payload(rank, int(scene_id))
        selected.extend(_select_scene_spectral_points(payload)[:1])
    return selected


def _spectral_panel_label(index: int) -> str:
    return f"P{index + 1}"


def _spectral_candidate_note(candidate: _PixelCandidate) -> str:
    return (
        f"$\\Delta$RMSE={candidate.rmse_tucker - candidate.rmse_ntdpl:.4f}\n"
        f"$\\Delta$SAM={candidate.sam_tucker - candidate.sam_ntdpl:.2f}$^\\circ$"
    )


def _annotate_spectral_pixels(ax: plt.Axes, candidates: list[_PixelCandidate]) -> None:
    for idx, candidate in enumerate(candidates):
        label = _spectral_panel_label(idx)
        ax.scatter(
            candidate.col,
            candidate.row,
            s=48,
            facecolor="white",
            edgecolor=PALETTE.highlight,
            linewidth=1.2,
            zorder=5,
        )
        ax.text(
            candidate.col + 2.0,
            candidate.row - 2.0,
            label,
            fontsize=8.4,
            weight="bold",
            color=PALETTE.black,
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": PALETTE.highlight, "alpha": 0.95},
            zorder=6,
        )


def _render_spectral_main_figure(candidates: list[_PixelCandidate], env: object, stem: str) -> None:
    seen_scene_ids = list(dict.fromkeys(candidate.scene_id for candidate in candidates))
    priority_scene_ids = [scene_id for scene_id in VISUAL_MAIN_SCENES if scene_id in seen_scene_ids]
    remaining_scene_ids = [scene_id for scene_id in seen_scene_ids if scene_id not in priority_scene_ids]
    scene_ids = priority_scene_ids + remaining_scene_ids
    payloads = {scene_id: _spectral_scene_payload(SPECTRAL_V2_RANK, scene_id) for scene_id in scene_ids}
    grouped: dict[int, list[_PixelCandidate]] = {scene_id: [] for scene_id in scene_ids}
    for candidate in candidates:
        grouped[candidate.scene_id].append(candidate)
    band_axis = np.arange(1, payloads[scene_ids[0]].original.shape[-1] + 1)

    apply_theme()
    fig = plt.figure(figsize=(10.6, 5.8))
    grid = fig.add_gridspec(nrows=len(scene_ids), ncols=3, width_ratios=[1.0, 1.18, 1.18], hspace=0.26, wspace=0.18)

    for row_idx, scene_id in enumerate(scene_ids):
        payload = payloads[scene_id]
        scene_candidates = grouped[scene_id]
        map_ax = fig.add_subplot(grid[row_idx, 0])
        map_image = np.clip(_pseudo_rgb(payload.original), 0.0, 1.0)
        map_ax.imshow(map_image)
        _setup_image_axis(map_ax, map_image)
        _annotate_spectral_pixels(map_ax, scene_candidates)
        map_ax.set_title("", pad=0)
        map_ax.text(
            -0.10,
            0.50,
            payload.scene_name.replace("_", " "),
            transform=map_ax.transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=9.8,
            fontweight="semibold",
        )

        for col_offset, candidate in enumerate(scene_candidates, start=1):
            ax = fig.add_subplot(grid[row_idx, col_offset])
            gt_curve = payload.original[candidate.row, candidate.col]
            t_curve = payload.recon_tucker[candidate.row, candidate.col]
            n_curve = payload.recon_ntdpl[candidate.row, candidate.col]
            ax.plot(band_axis, gt_curve, color=PALETTE.ground_truth, linewidth=2.2, label="Ground truth", zorder=4)
            ax.plot(band_axis, t_curve, label="Tucker", zorder=2, **method_style("Tucker"))
            ax.plot(band_axis, n_curve, label="NTD-PL", zorder=3, **method_style("NTD-PL"))
            ax.set_ylim(*_curve_ylim([gt_curve, t_curve, n_curve]))
            ax.set_xlim(1, len(band_axis))
            ax.set_xlabel("Band")
            if col_offset == 1:
                ax.set_ylabel("Reflectance")
            else:
                ax.set_ylabel("")
            style_axes(ax, grid=True)
            title_suffix = "high-gain pixel" if candidate.category == "ntdpl_better" else "typical improved pixel"
            if row_idx == 0:
                ax.set_title(f"{_spectral_panel_label((row_idx * 2) + (col_offset - 1))}: {title_suffix}", pad=5)
            ax.text(
                0.03,
                0.96,
                _spectral_candidate_note(candidate),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8.0,
                bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "edgecolor": PALETTE.grid, "alpha": 0.96},
            )

    handles = [
        Line2D([0], [0], color=PALETTE.ground_truth, linewidth=2.2, label="Ground truth"),
        Line2D([0], [0], label="Tucker", **method_style("Tucker")),
        Line2D([0], [0], label="NTD-PL", **method_style("NTD-PL")),
    ]
    fig.legend(**legend_style(handles, [h.get_label() for h in handles], loc="upper center", ncols=3, bbox_to_anchor=(0.58, 1.01)))
    fig.subplots_adjust(top=0.92, bottom=0.06, left=0.06, right=0.99)
    _save_and_sync_figure(fig, env, stem, formats=("pdf", "png"), dpi=600)
    plt.close(fig)


def _render_spectral_appendix_figure(candidates: list[_PixelCandidate], env: object, stem: str) -> None:
    payloads = {_candidate.scene_id: _spectral_scene_payload(SPECTRAL_V2_RANK, _candidate.scene_id) for _candidate in candidates}
    band_axis = np.arange(1, next(iter(payloads.values())).original.shape[-1] + 1)
    error_curves_t = [np.abs(payloads[c.scene_id].original[c.row, c.col] - payloads[c.scene_id].recon_tucker[c.row, c.col]) for c in candidates]
    error_curves_n = [np.abs(payloads[c.scene_id].original[c.row, c.col] - payloads[c.scene_id].recon_ntdpl[c.row, c.col]) for c in candidates]
    diff_curves = [err_t - err_n for err_t, err_n in zip(error_curves_t, error_curves_n, strict=False)]
    error_ylim = _curve_ylim(error_curves_t + error_curves_n)
    diff_limit = float(np.quantile(np.abs(np.concatenate(diff_curves, axis=0)), 0.98))
    diff_limit = max(diff_limit, 1e-3)

    apply_theme()
    fig, axes = plt.subplots(2, len(candidates), figsize=(10.6, 4.8), sharex=True)
    for idx, candidate in enumerate(candidates):
        payload = payloads[candidate.scene_id]
        top_ax = axes[0, idx]
        bot_ax = axes[1, idx]
        err_t = np.abs(payload.original[candidate.row, candidate.col] - payload.recon_tucker[candidate.row, candidate.col])
        err_n = np.abs(payload.original[candidate.row, candidate.col] - payload.recon_ntdpl[candidate.row, candidate.col])
        diff = err_t - err_n

        top_ax.plot(band_axis, err_t, label="Tucker |error|", color=PALETTE.tucker, linewidth=1.8)
        top_ax.plot(band_axis, err_n, label="NTD-PL |error|", color=PALETTE.ntdpl, linewidth=2.0)
        top_ax.set_ylim(*error_ylim)
        top_ax.set_title(f"{_spectral_panel_label(idx)}: Scene {candidate.scene_id}", pad=5)
        if idx == 0:
            top_ax.set_ylabel("|Error|")
        style_axes(top_ax, grid=True)

        bot_ax.plot(band_axis, diff, color=PALETTE.ntdpl, linewidth=1.9)
        bot_ax.axhline(0.0, color=PALETTE.border, linewidth=0.9)
        bot_ax.fill_between(band_axis, 0.0, diff, where=diff >= 0.0, color=PALETTE.ntdpl, alpha=0.14, linewidth=0)
        bot_ax.fill_between(band_axis, 0.0, diff, where=diff < 0.0, color=PALETTE.tucker, alpha=0.10, linewidth=0)
        bot_ax.set_ylim(-1.08 * diff_limit, 1.08 * diff_limit)
        bot_ax.set_xlabel("Band")
        if idx == 0:
            bot_ax.set_ylabel(r"$\Delta |e|$")
        style_axes(bot_ax, grid=True)

    top_handles = [
        Line2D([0], [0], color=PALETTE.tucker, linewidth=1.8, label="Tucker |error|"),
        Line2D([0], [0], color=PALETTE.ntdpl, linewidth=2.0, label="NTD-PL |error|"),
    ]
    fig.legend(**legend_style(top_handles, [h.get_label() for h in top_handles], loc="upper center", ncols=2, bbox_to_anchor=(0.5, 1.02)))
    fig.text(0.50, 0.50, r"$\Delta |e| = |e_{\mathrm{Tucker}}| - |e_{\mathrm{NTD-PL}}|$", ha="center", va="center", fontsize=8.4, color=PALETTE.border)
    fig.subplots_adjust(top=0.86, bottom=0.10, left=0.07, right=0.99, hspace=0.22, wspace=0.16)
    _save_and_sync_figure(fig, env, stem, formats=("pdf", "png"), dpi=600)
    plt.close(fig)


def spectral_curve_plot() -> None:
    _, env = _load_runs()
    selected_main = _spectral_main_selection(rank=SPECTRAL_V2_RANK)
    selected_appendix = _spectral_appendix_selection(rank=SPECTRAL_V2_RANK)
    _render_spectral_main_figure(selected_main, env, "spectral_curve_main_v3")
    _render_spectral_appendix_figure(selected_appendix, env, "spectral_curve_appendix")

    note_lines = [
        "# CAVE Spectral Curves",
        "",
        f"- Rank used for selection and plotting: `{_rank_text(SPECTRAL_V2_RANK)}`",
        "- Main figure uses the same representative scenes as the spatial visualization and selects two pixels per scene: one high-gain pixel and one typical improved pixel.",
        "- Appendix figure shows spectral absolute-error curves and band-wise error difference on additional representative pixels.",
        "",
        "## Selected Pixels",
        "",
    ]
    for idx, candidate in enumerate(selected_main):
        note_lines.append(
            f"- `{_spectral_panel_label(idx)}`: scene {candidate.scene_id} (`{candidate.scene_name}`), pixel ({candidate.row}, {candidate.col}), "
            f"category={candidate.category}, gradient={candidate.gradient:.4f}, mean={candidate.mean_intensity:.4f}, peak={candidate.peak_value:.4f}, "
            f"Tucker RMSE/SAM={candidate.rmse_tucker:.4f}/{candidate.sam_tucker:.2f} deg, "
            f"NTD-PL RMSE/SAM={candidate.rmse_ntdpl:.4f}/{candidate.sam_ntdpl:.2f} deg, "
            f"delta={candidate.rmse_tucker - candidate.rmse_ntdpl:.4f}/{candidate.sam_tucker - candidate.sam_ntdpl:.2f} deg."
        )
    note_lines.extend(["", "## Appendix Pixels", ""])
    for idx, candidate in enumerate(selected_appendix):
        note_lines.append(
            f"- `A{idx + 1}`: scene {candidate.scene_id} (`{candidate.scene_name}`), pixel ({candidate.row}, {candidate.col}), "
            f"gradient={candidate.gradient:.4f}, mean={candidate.mean_intensity:.4f}, peak={candidate.peak_value:.4f}, "
            f"Tucker RMSE/SAM={candidate.rmse_tucker:.4f}/{candidate.sam_tucker:.2f} deg, "
            f"NTD-PL RMSE/SAM={candidate.rmse_ntdpl:.4f}/{candidate.sam_ntdpl:.2f} deg."
        )

    note_path, _ = write_text_artifact(env, "\n".join(note_lines) + "\n", "spectral_curves_notes.md")


def _scene_metric_lookup(rank: tuple[int, int, int]) -> pd.DataFrame:
    runs, _ = _load_runs()
    tbl = _main_experiment_rows(_base_columns(runs))
    return _scene_pair_table(tbl, rank=rank, metrics=("RMSE", "NMSE_dB", "SAM"))


def _scene_visual_payload(rank: tuple[int, int, int], scene_id: int) -> dict[str, Any]:
    payload = _spectral_scene_payload(rank, scene_id)
    metrics = _scene_metric_lookup(rank).loc[lambda frame: frame["scene_id"] == int(scene_id)].iloc[0]
    err_tucker = np.mean(np.abs(payload.original - payload.recon_tucker), axis=-1)
    err_ntdpl = np.mean(np.abs(payload.original - payload.recon_ntdpl), axis=-1)
    diff_map = err_tucker - err_ntdpl
    return {
        "scene_id": int(scene_id),
        "scene_name": payload.scene_name,
        "original": payload.original,
        "rgb_original": _pseudo_rgb(payload.original),
        "rgb_tucker": _pseudo_rgb(payload.recon_tucker),
        "rgb_ntdpl": _pseudo_rgb(payload.recon_ntdpl),
        "err_tucker": err_tucker,
        "err_ntdpl": err_ntdpl,
        "diff_map": diff_map,
        "nmse_db_tucker": float(metrics["NMSE_dB_tucker"]),
        "nmse_db_ntdpl": float(metrics["NMSE_dB_ntdpl"]),
        "sam_tucker": float(metrics["SAM_tucker"]),
        "sam_ntdpl": float(metrics["SAM_ntdpl"]),
    }


def _scene_selection() -> tuple[list[int], list[int]]:
    main_scenes = list(VISUAL_MAIN_SCENES)
    appendix = [scene_id for scene_id in VISUAL_APPENDIX_SCENES if scene_id not in main_scenes]
    return main_scenes, appendix


def _pick_visual_roi(
    diff_map: np.ndarray,
    reference_rgb: np.ndarray,
    *,
    crop_size: int = VISUAL_ROI_SIZE,
    min_signal_quantile: float = 0.30,
) -> tuple[int, int, int, int]:
    half = crop_size // 2
    intensity = np.mean(reference_rgb, axis=-1)
    signal_floor = float(np.quantile(intensity, min_signal_quantile))
    best_score = -np.inf
    best_box = (0, crop_size, 0, crop_size)

    for row in range(half, diff_map.shape[0] - half):
        r0 = row - half
        r1 = r0 + crop_size
        for col in range(half, diff_map.shape[1] - half):
            c0 = col - half
            c1 = c0 + crop_size
            patch_signal = intensity[r0:r1, c0:c1]
            if float(np.mean(patch_signal)) < signal_floor:
                continue
            patch = diff_map[r0:r1, c0:c1]
            positive = np.clip(patch, 0.0, None)
            score = float(np.mean(positive) + 0.35 * np.quantile(positive, 0.90))
            if score > best_score:
                best_score = score
                best_box = (r0, r1, c0, c1)
    return best_box


def _crop_panel(image: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    r0, r1, c0, c1 = roi
    if image.ndim == 2:
        return image[r0:r1, c0:c1]
    return image[r0:r1, c0:c1, :]


def _setup_image_axis(ax: plt.Axes, image: np.ndarray | None = None) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    if image is not None and image.ndim >= 2 and image.shape[1] > 0:
        ax.set_box_aspect(float(image.shape[0]) / float(image.shape[1]))
    ax.set_anchor("C")
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_roi_box(ax: plt.Axes, roi: tuple[int, int, int, int], *, edgecolor: str = "#F2C14E") -> None:
    r0, r1, c0, c1 = roi
    ax.add_patch(
        Rectangle(
            (c0, r0),
            c1 - c0,
            r1 - r0,
            fill=False,
            linewidth=1.1,
            linestyle="--",
            edgecolor=edgecolor,
        )
    )


def _draw_roi_inset(
    ax: plt.Axes,
    image: np.ndarray,
    roi: tuple[int, int, int, int],
    *,
    kind: str,
    diff_vmax: float,
) -> None:
    crop = _crop_panel(image, roi)
    inset = ax.inset_axes([0.56, 0.56, 0.41, 0.41])
    if kind == "rgb":
        inset.imshow(np.clip(crop, 0.0, 1.0))
    else:
        inset.imshow(crop, cmap="RdBu_r", vmin=-diff_vmax, vmax=diff_vmax)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.4)
        spine.set_edgecolor("white")


def _visual_column_titles(*, include_error: bool = True) -> list[str]:
    if include_error:
        return ["Original", "Tucker", "NTD-PL", "Error", "Error", "Difference"]
    return ["Original", "Tucker", "NTD-PL", "Difference"]


def _visual_scene_note(payload: dict[str, Any]) -> tuple[str, str]:
    tucker_note = f"NMSE(dB) {payload['nmse_db_tucker']:.2f}\nSAM {payload['sam_tucker']:.2f}"
    ntdpl_note = f"NMSE(dB) {payload['nmse_db_ntdpl']:.2f}\nSAM {payload['sam_ntdpl']:.2f}"
    return tucker_note, ntdpl_note


def _render_visual_main_figure(scene_ids: list[int], env: object, stem: str) -> None:
    payloads = [_scene_visual_payload(VISUAL_MAIN_RANK, scene_id) for scene_id in scene_ids]
    panel_titles = ["Original", "Tucker", "NTD-PL", "Difference"]
    diff_vmax = float(np.quantile(np.abs(np.concatenate([np.ravel(item["diff_map"]) for item in payloads])), 0.995))

    apply_theme()
    fig, axes = plt.subplots(len(payloads), 4, figsize=(11.7, 2.18 * len(payloads) + 0.08), constrained_layout=False)
    if len(payloads) == 1:
        axes = np.asarray([axes])

    for scene_idx, payload in enumerate(payloads):
        scene_name = payload["scene_name"].replace("_", " ")
        images = [
            (payload["rgb_original"], "rgb"),
            (payload["rgb_tucker"], "rgb"),
            (payload["rgb_ntdpl"], "rgb"),
            (payload["diff_map"], "diff"),
        ]
        for col_idx, image in enumerate(images):
            ax = axes[scene_idx, col_idx]
            image_data, kind = image
            if kind == "diff":
                ax.imshow(image_data, cmap="RdBu_r", vmin=-diff_vmax, vmax=diff_vmax)
            else:
                ax.imshow(np.clip(image_data, 0.0, 1.0))
            _setup_image_axis(ax, image_data)
            if scene_idx == 0:
                ax.set_title(panel_titles[col_idx], pad=5)
            if col_idx == 0:
                ax.text(
                    -0.035,
                    0.55,
                    scene_name,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=9.8,
                    fontweight="semibold",
                )

    diff_sm = plt.cm.ScalarMappable(cmap="RdBu_r")
    diff_sm.set_clim(-diff_vmax, diff_vmax)
    diff_cax = fig.add_axes([0.955, 0.11, 0.012, 0.80])
    diff_cb = fig.colorbar(diff_sm, cax=diff_cax)
    style_colorbar(diff_cb, label="positive: NTD-PL better")
    diff_cb.ax.set_title("Diff.", fontsize=8, pad=5)

    fig.subplots_adjust(left=0.078, right=0.945, top=0.972, bottom=0.032, wspace=0.035, hspace=0.04)
    _save_and_sync_figure(fig, env, stem, formats=("pdf", "png"), dpi=600)
    plt.close(fig)


def _render_visual_appendix_figure(scene_ids: list[int], env: object, stem: str) -> None:
    payloads = [_scene_visual_payload(VISUAL_MAIN_RANK, scene_id) for scene_id in scene_ids]
    error_vmax = float(
        np.quantile(
            np.concatenate(
                [np.ravel(item["err_tucker"]) for item in payloads] + [np.ravel(item["err_ntdpl"]) for item in payloads]
            ),
            0.995,
        )
    )
    diff_vmax = float(np.quantile(np.abs(np.concatenate([np.ravel(item["diff_map"]) for item in payloads])), 0.995))
    titles = _visual_column_titles()

    apply_theme()
    fig = plt.figure(figsize=(10.8, 1.95 * len(payloads)))
    grid = fig.add_gridspec(nrows=len(payloads), ncols=6, hspace=0.08, wspace=0.04)

    for scene_idx, payload in enumerate(payloads):
        roi = _pick_visual_roi(payload["diff_map"], payload["rgb_original"])
        tucker_note, ntdpl_note = _visual_scene_note(payload)
        scene_label = f"Scene {payload['scene_id']} ({payload['scene_name'].replace('_', ' ')})"
        full_images: list[tuple[np.ndarray, dict[str, Any]]] = [
            (payload["rgb_original"], {"kind": "rgb"}),
            (payload["rgb_tucker"], {"kind": "rgb", "note": tucker_note}),
            (payload["rgb_ntdpl"], {"kind": "rgb", "note": ntdpl_note}),
            (payload["err_tucker"], {"kind": "error"}),
            (payload["err_ntdpl"], {"kind": "error"}),
            (payload["diff_map"], {"kind": "diff"}),
        ]
        for col_idx, (image, meta) in enumerate(full_images):
            ax = fig.add_subplot(grid[scene_idx, col_idx])
            if meta["kind"] == "rgb":
                ax.imshow(np.clip(image, 0.0, 1.0))
            elif meta["kind"] == "error":
                ax.imshow(image, cmap="magma", vmin=0.0, vmax=error_vmax)
            else:
                ax.imshow(image, cmap="RdBu_r", vmin=-diff_vmax, vmax=diff_vmax)
            _draw_roi_box(ax, roi)
            _setup_image_axis(ax, image)
            if scene_idx == 0:
                ax.set_title(titles[col_idx], pad=6)
            if col_idx == 0:
                ax.text(
                    -0.03,
                    0.50,
                    scene_label,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=9.6,
                )
            note = meta.get("note")
            if note:
                ax.text(
                    0.03,
                    0.03,
                    str(note),
                    transform=ax.transAxes,
                    va="bottom",
                    ha="left",
                    fontsize=7.4,
                    color="white",
                    bbox={"boxstyle": "round,pad=0.18", "facecolor": (0.0, 0.0, 0.0, 0.45), "edgecolor": "none"},
                )

    error_sm = plt.cm.ScalarMappable(cmap="magma")
    error_sm.set_clim(0.0, error_vmax)
    diff_sm = plt.cm.ScalarMappable(cmap="RdBu_r")
    diff_sm.set_clim(-diff_vmax, diff_vmax)
    error_cax = fig.add_axes([0.92, 0.19, 0.012, 0.64])
    diff_cax = fig.add_axes([0.95, 0.19, 0.012, 0.64])
    error_cb = fig.colorbar(error_sm, cax=error_cax)
    diff_cb = fig.colorbar(diff_sm, cax=diff_cax)
    style_colorbar(error_cb)
    style_colorbar(diff_cb, label="positive: NTD-PL better")
    error_cb.ax.set_title("Error", fontsize=8, pad=5)
    diff_cb.ax.set_title("Diff.", fontsize=8, pad=5)

    fig.subplots_adjust(left=0.10, right=0.90, top=0.95, bottom=0.06)
    _save_and_sync_figure(fig, env, stem, formats=("pdf", "png"), dpi=600)
    plt.close(fig)


def visual_compare_plot() -> None:
    _, env = _load_runs()
    main_scenes, appendix_scenes = _scene_selection()
    _render_visual_main_figure(main_scenes, env, "visual_compare_spatial_main_v5")
    _render_visual_appendix_figure(appendix_scenes, env, "visual_compare_appendix")
    _render_visual_main_figure(main_scenes, env, "visual_compare")


def bandwise_error_plot() -> None:
    runs, env = _load_runs()
    tbl = _main_experiment_rows(_base_columns(runs))
    methods_to_plot = ["tucker", "ntdpl"]
    rank_panels = [(8, 8, 4), (12, 12, 6), (16, 16, 8)]
    rank_linestyles = {
        (8, 8, 4): "-",
        (12, 12, 6): "--",
        (16, 16, 8): ":",
    }

    apply_theme()
    fig, ax = plt.subplots(1, 1, figsize=(10.2, 4.2))
    for target_rank in rank_panels:
        subset = tbl.loc[tbl["rank"] == target_rank].copy()
        for method in methods_to_plot:
            curves: list[np.ndarray] = []
            method_rows = subset.loc[subset["method_name"] == method].copy()
            for _, row in method_rows.iterrows():
                dataset = _dataset_from_row(row)
                original = np.asarray(dataset.get("eval").dense, dtype=np.float32)
                recon = _try_load_reconstruction(row)
                if recon is None:
                    continue
                curves.append(np.sqrt(np.mean((original - recon) ** 2, axis=(0, 1))))

            if not curves:
                continue
            mean_curve = np.mean(np.stack(curves, axis=0), axis=0)
            label = env.label_for_method(method)
            style = method_style(label)
            style["linestyle"] = rank_linestyles[target_rank]
            style["marker"] = None
            ax.plot(
                np.arange(1, len(mean_curve) + 1),
                mean_curve,
                label=f"{label} {_rank_text(target_rank)}",
                **style,
            )

    ax.set_title("CAVE: Band-wise RMSE")
    ax.set_xlabel("Band")
    ax.set_ylabel("RMSE")
    style_axes(ax)
    method_handles = []
    for method in methods_to_plot:
        label = env.label_for_method(method)
        style = method_style(label)
        method_handles.append(
            Line2D([0], [0], color=style["color"], linestyle="-", linewidth=style["linewidth"], label=label)
        )

    rank_handles = []
    for rank in rank_panels:
        rank_handles.append(
            Line2D([0], [0], color=PALETTE.border, linestyle=rank_linestyles[rank], linewidth=2.0, label=_rank_text(rank))
        )

    fig.legend(
        **legend_style(
            method_handles,
            [h.get_label() for h in method_handles],
            loc="upper center",
            ncols=len(method_handles),
            bbox_to_anchor=(0.34, 1.02),
        )
    )
    fig.legend(
        **legend_style(
            rank_handles,
            [h.get_label() for h in rank_handles],
            loc="upper center",
            ncols=len(rank_handles),
            bbox_to_anchor=(0.78, 1.02),
        )
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save_and_sync_figure(fig, env, "bandwise_error")
    plt.close(fig)


def bandwise_nmse_db_plot() -> None:
    runs, env = _load_runs()
    tbl = _main_experiment_rows(_base_columns(runs))
    subset = tbl.loc[(tbl["rank"] == OVERVIEW_RANK) & (tbl["method_name"].isin(METHOD_FOCUS_ORDER))].copy()
    scene_records: list[dict[str, Any]] = []
    for scene_id in sorted(subset["scene_id"].unique().tolist()):
        scene_rows = subset.loc[subset["scene_id"] == scene_id].copy()
        if scene_rows["method_name"].nunique() < 2:
            continue
        curves_by_method: dict[str, np.ndarray] = {}
        for method_name in METHOD_FOCUS_ORDER:
            row = scene_rows.loc[scene_rows["method_name"] == method_name]
            if row.empty:
                continue
            dataset = _dataset_from_row(row.iloc[0])
            original = np.asarray(dataset.get("eval").dense, dtype=np.float32)
            recon = _try_load_reconstruction(row.iloc[0])
            if recon is None:
                continue
            curves_by_method[method_name] = np.sqrt(np.mean((original - recon) ** 2, axis=(0, 1)))
        if len(curves_by_method) < 2:
            continue
        scene_records.append(
            {
                "scene_id": int(scene_id),
                "scene_name": _dataset_from_row(scene_rows.iloc[0]).scene_name,
                "tucker": curves_by_method["tucker"],
                "ntdpl": curves_by_method["ntdpl"],
                "improvement": curves_by_method["tucker"] - curves_by_method["ntdpl"],
            }
        )

    if not scene_records:
        raise RuntimeError("No valid band-wise spectral curves available.")

    tucker_curves = np.stack([record["tucker"] for record in scene_records], axis=0)
    ntdpl_curves = np.stack([record["ntdpl"] for record in scene_records], axis=0)
    improvement_curves = np.stack([record["improvement"] for record in scene_records], axis=0)
    band_axis = np.arange(1, tucker_curves.shape[1] + 1)

    def _summary(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return np.mean(curves, axis=0), np.quantile(curves, 0.25, axis=0), np.quantile(curves, 0.75, axis=0)

    t_mean, t_lo, t_hi = _summary(tucker_curves)
    n_mean, n_lo, n_hi = _summary(ntdpl_curves)
    d_mean, d_lo, d_hi = _summary(improvement_curves)
    wins = int(np.sum(d_mean > 0.0))
    mean_delta = float(np.mean(d_mean))
    median_delta = float(np.median(d_mean))

    apply_theme()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), constrained_layout=True)
    ax_curve, ax_gain = axes

    ax_curve.fill_between(band_axis, t_lo, t_hi, color=PALETTE.tucker, alpha=0.14, linewidth=0)
    ax_curve.fill_between(band_axis, n_lo, n_hi, color=PALETTE.ntdpl, alpha=0.14, linewidth=0)
    ax_curve.plot(band_axis, t_mean, color=PALETTE.tucker, linewidth=1.9, linestyle="--", label="Tucker")
    ax_curve.plot(band_axis, n_mean, color=PALETTE.ntdpl, linewidth=2.2, linestyle="-", label="NTD-PL")
    ax_curve.set_title("Band-wise spectral RMSE", pad=5)
    ax_curve.set_xlabel("Band")
    ax_curve.set_ylabel("RMSE")
    style_axes(ax_curve, grid=True)
    ax_curve.text(
        0.03,
        0.96,
        f"rank = {_rank_text(OVERVIEW_RANK)}\nscenes = {len(scene_records)}\nshaded: IQR across scenes",
        transform=ax_curve.transAxes,
        va="top",
        ha="left",
        fontsize=8.0,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": PALETTE.grid, "alpha": 0.96},
    )

    ax_gain.fill_between(band_axis, d_lo, d_hi, color=PALETTE.ntdpl, alpha=0.12, linewidth=0)
    ax_gain.plot(band_axis, d_mean, color=PALETTE.ntdpl, linewidth=2.1)
    ax_gain.axhline(0.0, color=PALETTE.border, linewidth=0.95, zorder=3)
    ax_gain.fill_between(band_axis, 0.0, d_mean, where=d_mean >= 0.0, color=PALETTE.ntdpl, alpha=0.14, linewidth=0)
    ax_gain.fill_between(band_axis, 0.0, d_mean, where=d_mean < 0.0, color=PALETTE.tucker, alpha=0.10, linewidth=0)
    ax_gain.set_title("Band-wise improvement", pad=5)
    ax_gain.set_xlabel("Band")
    ax_gain.set_ylabel(r"$\Delta$RMSE")
    style_axes(ax_gain, grid=True)
    ax_gain.text(
        0.03,
        0.96,
        f"wins: {wins}/{len(band_axis)}\nmean $\\Delta$ = {mean_delta:.4f}\nmedian $\\Delta$ = {median_delta:.4f}",
        transform=ax_gain.transAxes,
        va="top",
        ha="left",
        fontsize=8.0,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": PALETTE.grid, "alpha": 0.96},
    )

    handles = [
        Line2D([0], [0], color=PALETTE.tucker, linewidth=1.9, linestyle="--", label="Tucker"),
        Line2D([0], [0], color=PALETTE.ntdpl, linewidth=2.2, linestyle="-", label="NTD-PL"),
        Line2D([0], [0], color=PALETTE.ntdpl, linewidth=2.1, linestyle="-", label=r"$\Delta$RMSE"),
    ]
    fig.legend(**legend_style(handles, [h.get_label() for h in handles], loc="upper center", ncols=3, bbox_to_anchor=(0.5, 1.03)))
    _save_and_sync_figure(fig, env, "bandwise_spectral_overview", formats=("pdf", "png"), dpi=600)
    plt.close(fig)

    heatmap = improvement_curves[np.argsort(-np.mean(improvement_curves, axis=1))]
    apply_theme()
    fig, ax = plt.subplots(1, 1, figsize=(10.4, 3.9))
    vmax = float(np.quantile(np.abs(heatmap), 0.98))
    vmax = max(vmax, 1e-3)
    im = ax.imshow(heatmap, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_title("Scene x band improvement heatmap", pad=5)
    ax.set_xlabel("Band")
    ax.set_ylabel("Scene (sorted by mean improvement)")
    style_axes(ax, grid=False)
    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.035)
    style_colorbar(cbar, label=r"$\Delta$RMSE")
    _save_and_sync_figure(fig, env, "bandwise_spectral_heatmap_appendix", formats=("pdf", "png"), dpi=600)
    plt.close(fig)


def pmax_scan_plot() -> None:
    runs, env = _load_runs()
    scan_mask = (
        runs["ovr.data"].astype(str).eq("cave_hsi")
        & maybe_numeric(runs["ovr.data.id"]).eq(float(PMAX_SCAN_SCENE_ID))
        & runs["ovr.filter"].astype(str).eq("bias-filter")
        & runs["ovr.filter.normalize_method"].astype(str).eq("max")
        & runs["ovr.method"].astype(str).isin(METHOD_FOCUS_ORDER)
    )
    scan_tbl = _base_columns(runs.loc[scan_mask].copy())
    scan_tbl = scan_tbl.loc[scan_tbl["rank"].isin(RANK_ORDER)].copy()
    ntdpl = scan_tbl.loc[scan_tbl["method_name"] == "ntdpl"].copy()
    tucker = scan_tbl.loc[scan_tbl["method_name"] == "tucker"].copy()
    if ntdpl.empty or tucker.empty:
        print("Skip `pmax_rank_heatmap`: missing filtered Tucker or NTD-PL scan rows.")
        return

    ntdpl["p_max"] = maybe_numeric(ntdpl["p_max"]).astype(int)
    ntdpl = ntdpl.loc[ntdpl["p_max"].isin(PMAX_VALUES)].copy()
    rank_labels = [_rank_text(rank) for rank in RANK_ORDER]
    available_pairs = {(row["rank_text"], int(row["p_max"])) for _, row in ntdpl.iterrows()}
    missing_pairs = [
        (rank_text, p_value)
        for rank_text in rank_labels
        for p_value in PMAX_VALUES
        if (rank_text, p_value) not in available_pairs
    ]
    if missing_pairs:
        missing_text = ", ".join(f"{rank}:{p}" for rank, p in missing_pairs)
        print(f"Skip `pmax_rank_heatmap`: incomplete scan grid ({missing_text}).")
        return

    tucker = (
        tucker.sort_values("run_dir")
        .drop_duplicates(subset=["scene_id", "rank_text", "method_name"], keep="last")
        .copy()
    )
    if tucker["rank_text"].nunique() != len(rank_labels):
        print("Skip `pmax_rank_heatmap`: missing Tucker scan rows for one or more ranks.")
        return

    tucker_lookup = tucker.set_index("rank_text")[["RMSE", "SAM"]]
    records: list[dict[str, Any]] = []
    for rank in RANK_ORDER:
        rank_text = _rank_text(rank)
        tucker_row = tucker_lookup.loc[rank_text]
        rank_rows = ntdpl.loc[ntdpl["rank_text"] == rank_text].sort_values("p_max")
        for _, row in rank_rows.iterrows():
            records.append(
                {
                    "scene_id": PMAX_SCAN_SCENE_ID,
                    "rank_text": rank_text,
                    "p_max": int(row["p_max"]),
                    "RMSE": float(row["RMSE"]),
                    "SAM": float(row["SAM"]),
                    "tucker_RMSE": float(tucker_row["RMSE"]),
                    "tucker_SAM": float(tucker_row["SAM"]),
                    "delta_RMSE": float(tucker_row["RMSE"] - row["RMSE"]),
                    "delta_SAM": float(tucker_row["SAM"] - row["SAM"]),
                }
            )

    summary = pd.DataFrame(records).sort_values(["rank_text", "p_max"]).reset_index(drop=True)
    write_csv_artifact(env, summary, "pmax_rank_heatmap.csv")

    p_values = list(PMAX_VALUES)
    rmse_grid = (
        summary.pivot(index="rank_text", columns="p_max", values="RMSE")
        .loc[rank_labels, p_values]
        .to_numpy(dtype=float)
    )
    sam_grid = (
        summary.pivot(index="rank_text", columns="p_max", values="SAM")
        .loc[rank_labels, p_values]
        .to_numpy(dtype=float)
    )
    delta_rmse = (
        summary.pivot(index="rank_text", columns="p_max", values="delta_RMSE")
        .loc[rank_labels, p_values]
        .to_numpy(dtype=float)
    )
    delta_sam = (
        summary.pivot(index="rank_text", columns="p_max", values="delta_SAM")
        .loc[rank_labels, p_values]
        .to_numpy(dtype=float)
    )

    def _heatmap_panel(
        ax: plt.Axes,
        values: np.ndarray,
        *,
        title: str,
        cbar_label: str,
        row_best_text: str,
    ) -> None:
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-6
        im = ax.imshow(values, cmap="Blues_r", vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
        ax.set_title(title, pad=5)
        ax.set_xlabel(r"$p_{\max}$")
        ax.set_ylabel("Rank")
        ax.set_xticks(np.arange(len(p_values)))
        ax.set_xticklabels([str(p) for p in p_values])
        ax.set_yticks(np.arange(len(rank_labels)))
        ax.set_yticklabels(rank_labels)
        style_axes(ax, grid=False)
        ax.tick_params(length=0.0)

        # Outline the row-best cells and highlight the global best more strongly.
        row_best = np.argmin(values, axis=1)
        for row_idx, col_idx in enumerate(row_best):
            ax.add_patch(
                Rectangle(
                    (col_idx - 0.5, row_idx - 0.5),
                    1.0,
                    1.0,
                    fill=False,
                    edgecolor=PALETTE.border,
                    linewidth=1.0,
                )
            )
        best_row, best_col = np.unravel_index(int(np.argmin(values)), values.shape)
        ax.add_patch(
            Rectangle(
                (best_col - 0.5, best_row - 0.5),
                1.0,
                1.0,
                fill=False,
                edgecolor=PALETTE.highlight,
                linewidth=2.0,
            )
        )

        span = vmax - vmin
        for row_idx in range(values.shape[0]):
            for col_idx in range(values.shape[1]):
                value = float(values[row_idx, col_idx])
                text_color = "white" if value <= vmin + 0.38 * span else PALETTE.black
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=7.8,
                    color=text_color,
                )

        cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.045)
        style_colorbar(cbar, label=cbar_label)
        ax.text(
            0.02,
            1.03,
            row_best_text,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.0,
            color=PALETTE.border,
        )

    rmse_best = summary.loc[summary.groupby("rank_text")["RMSE"].idxmin()].sort_values("rank_text")
    sam_best = summary.loc[summary.groupby("rank_text")["SAM"].idxmin()].sort_values("rank_text")

    apply_theme()
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.15))
    _heatmap_panel(
        axes[0],
        rmse_grid,
        title="RMSE over degree and rank",
        cbar_label="RMSE",
        row_best_text="row-best $p_{\\max}$: " + ", ".join(f"{row['rank_text']}→{int(row['p_max'])}" for _, row in rmse_best.iterrows()),
    )
    _heatmap_panel(
        axes[1],
        sam_grid,
        title="SAM over degree and rank",
        cbar_label="SAM (deg)",
        row_best_text="row-best $p_{\\max}$: " + ", ".join(f"{row['rank_text']}→{int(row['p_max'])}" for _, row in sam_best.iterrows()),
    )
    fig.suptitle(
        rf"CAVE scene {PMAX_SCAN_SCENE_ID}: degree--rank sensitivity under the same normalized scan",
        y=0.985,
        fontsize=11.0,
    )
    fig.text(
        0.5,
        0.01,
        "outline: row best   gold: global best   scan config: bias-filter, normalize=max",
        ha="center",
        va="bottom",
        fontsize=8.2,
        color=PALETTE.border,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    _save_and_sync_figure(fig, env, "pmax_rank_heatmap_main", formats=("pdf", "png"), dpi=600)
    plt.close(fig)

    rank_linestyles = {
        RANK_ORDER[0]: "--",
        RANK_ORDER[1]: "-.",
        RANK_ORDER[2]: "-",
    }
    apply_theme()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6), sharex=True)
    for ax, delta_values, ylabel, title in [
        (axes[0], delta_rmse, r"$\Delta$RMSE", "RMSE improvement over Tucker"),
        (axes[1], delta_sam, r"$\Delta$SAM", "SAM improvement over Tucker"),
    ]:
        for row_idx, rank in enumerate(RANK_ORDER):
            rank_text = _rank_text(rank)
            ax.plot(
                p_values,
                delta_values[row_idx],
                color=PALETTE.ntdpl,
                linewidth=2.0,
                linestyle=rank_linestyles[rank],
                marker=_rank_marker(rank),
                markersize=5.0,
                markerfacecolor="white",
                label=rank_text,
            )
        ax.axhline(0.0, color=PALETTE.tucker, linewidth=1.4, linestyle="--")
        ax.set_title(title, pad=5)
        ax.set_xlabel(r"$p_{\max}$")
        ax.set_ylabel(ylabel)
        style_axes(ax, grid=True)
    handles = [
        Line2D([0], [0], color=PALETTE.ntdpl, linestyle=rank_linestyles[rank], marker=_rank_marker(rank), markersize=5.2, linewidth=2.0, markerfacecolor="white", label=_rank_text(rank))
        for rank in RANK_ORDER
    ]
    handles.append(Line2D([0], [0], color=PALETTE.tucker, linestyle="--", linewidth=1.4, label="Tucker baseline"))
    fig.legend(**legend_style(handles, [h.get_label() for h in handles], loc="upper center", ncols=4, bbox_to_anchor=(0.5, 1.03)))
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save_and_sync_figure(fig, env, "pmax_rank_slices_appendix", formats=("pdf", "png"), dpi=600)
    plt.close(fig)


