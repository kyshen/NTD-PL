from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from ...config import get_env
from ...hsi_defaults import CAVE_RECON_MAIN_RANK
from ...utils.io import load_run_parquets, load_state_mat, maybe_numeric

from .cave_random_completion import (
    _cave_dataset_kwargs_from_row,
    _parse_rank,
    _resolve_state_path,
)
from .cave_random_completion_polycal import load_scene_original


TARGET_MISSING_RATE = 0.5
MIN_DEGREE = 1
MAX_DEGREE = 4


def _series_str(frame: pd.DataFrame, *names: str) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return frame[name].astype(str)
    raise KeyError(f"Missing string columns: {names!r}")


def _series_num(frame: pd.DataFrame, *names: str) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return maybe_numeric(frame[name])
    raise KeyError(f"Missing numeric columns: {names!r}")


def _parse_bool_array(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == np.bool_:
        return array
    return array.astype(bool)


def _extract_factors(state: dict[str, Any]) -> list[np.ndarray]:
    factors_raw = state["factors"]
    if isinstance(factors_raw, np.ndarray) and factors_raw.dtype == object:
        return [np.asarray(item, dtype=np.float32) for item in factors_raw.reshape(-1)]
    if isinstance(factors_raw, list):
        return [np.asarray(item, dtype=np.float32) for item in factors_raw]
    raise TypeError(f"Unsupported factors payload type: {type(factors_raw)}")


def _latent_from_state(state: dict[str, Any]) -> np.ndarray:
    try:
        from tensorly.tucker_tensor import tucker_to_tensor
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Missing dependency 'tensorly' for cave-parameter mechanism postprocess.") from exc
    core = np.asarray(state["core"], dtype=np.float32)
    factors = _extract_factors(state)
    return np.asarray(tucker_to_tensor((core, factors)), dtype=np.float32)


def _missing_metrics(original: np.ndarray, reconstruction: np.ndarray, observed_mask: np.ndarray) -> dict[str, float]:
    missing_mask = ~np.asarray(observed_mask, dtype=bool)
    if not np.any(missing_mask):
        return {"RMSE*": 0.0, "SAM*": 0.0, "NMSE(dB)*": -np.inf}
    diff = (np.asarray(original, dtype=np.float32) - np.asarray(reconstruction, dtype=np.float32)).astype(np.float32)
    diff_missing = diff[missing_mask]
    original_missing = np.asarray(original, dtype=np.float32)[missing_mask]

    mse = float(np.mean(diff_missing * diff_missing))
    rmse = float(np.sqrt(max(mse, 0.0)))
    signal = float(np.sum(original_missing * original_missing))
    noise = float(np.sum(diff_missing * diff_missing))
    nmse_db = float(10.0 * np.log10(max(noise, 1e-30) / max(signal, 1e-30)))

    miss_pixels = np.any(missing_mask, axis=-1)
    o = np.asarray(original, dtype=np.float32)[miss_pixels]
    r = np.asarray(reconstruction, dtype=np.float32)[miss_pixels]
    dot = np.sum(o * r, axis=1)
    norm_o = np.linalg.norm(o, axis=1)
    norm_r = np.linalg.norm(r, axis=1)
    denom = np.maximum(norm_o * norm_r, 1e-12)
    cos = np.clip(dot / denom, -1.0, 1.0)
    sam = float(np.degrees(np.mean(np.arccos(cos)))) if cos.size else 0.0
    return {
        "RMSE*": rmse,
        "SAM*": sam,
        "NMSE(dB)*": nmse_db,
    }


@lru_cache(maxsize=None)
def _load_scene(scene_id: int, path: str, target_shape: tuple[int, int], crop_shape: tuple[int, int] | None) -> np.ndarray:
    _, original = load_scene_original(scene_id, path=path, target_shape=target_shape, crop_shape=crop_shape)
    return np.asarray(original, dtype=np.float32)


def load_ntdpl_main_runs() -> tuple[pd.DataFrame, Any]:
    env = get_env("cave-random-completion")
    runs = load_run_parquets(env.results_dir)["runs"].copy()
    if runs.empty:
        raise RuntimeError("No runs found for cave-random-completion.")

    frame = runs.copy()
    frame = frame.loc[_series_str(frame, "ovr.data", "data._name").eq("cave_hsi")].copy()
    frame["method_name"] = _series_str(frame, "ovr.method", "method._name")
    frame["rank"] = _series_str(frame, "ovr.method.rank", "method.rank").map(_parse_rank)
    frame["scene_id"] = _series_num(frame, "ovr.data.id", "data.id").astype(int)
    frame["mask_seed"] = _series_num(frame, "ovr.task.seed", "task.seed").astype(int)
    frame["missing_rate"] = _series_num(frame, "ovr.task.missing_rate", "task.missing_rate").astype(float)
    frame["p_max"] = _series_num(frame, "ovr.method.p_max", "method.p_max").astype(float)
    if "RMSE_missing" in frame.columns:
        frame["RMSE_missing"] = _series_num(frame, "RMSE_missing").astype(float)
    else:
        frame["RMSE_missing"] = np.nan

    frame = frame.loc[
        frame["method_name"].eq("ntdpl")
        & frame["rank"].map(lambda value: tuple(value) == CAVE_RECON_MAIN_RANK)
        & np.isclose(frame["missing_rate"], TARGET_MISSING_RATE, atol=1e-12)
    ].copy()
    if frame.empty:
        raise RuntimeError("No NTD-PL runs available for cave-parameter mechanism analysis.")
    max_p = int(frame["p_max"].dropna().max())
    frame = frame.loc[np.isclose(frame["p_max"], float(max_p), atol=1e-12)].copy()
    frame = frame.sort_values("run_dir").drop_duplicates(
        subset=["scene_id", "mask_seed", "missing_rate", "method_name"],
        keep="last",
    )
    # Use one representative seed per scene (closest to scene-wise median RMSE_missing)
    representative_rows: list[pd.Series] = []
    for scene_id, panel in frame.groupby("scene_id"):
        if panel["RMSE_missing"].notna().any():
            median = float(panel["RMSE_missing"].median())
            chosen = panel.iloc[(panel["RMSE_missing"] - median).abs().argmin()]
        else:
            chosen = panel.sort_values("run_dir").iloc[-1]
        representative_rows.append(chosen)
    frame = pd.DataFrame(representative_rows).reset_index(drop=True)
    return frame.reset_index(drop=True), env


def _evaluate_one_run(row: pd.Series, *, max_degree: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    state = load_state_mat(_resolve_state_path(str(row["state_path"])))
    if "beta" not in state or "core" not in state or "factors" not in state:
        raise KeyError("NTD-PL state must contain beta/core/factors for degree-mechanism analysis.")

    beta = np.asarray(state["beta"], dtype=np.float32).reshape(-1)
    observed_mask = _parse_bool_array(state["observed_mask"])
    latent = _latent_from_state(state)

    data_kwargs = _cave_dataset_kwargs_from_row(row)
    original = _load_scene(
        int(row["scene_id"]),
        str(data_kwargs["path"]),
        tuple(data_kwargs["target_shape"]),
        data_kwargs["crop_shape"],
    )
    if original.shape != latent.shape:
        raise RuntimeError(
            f"Shape mismatch for scene {int(row['scene_id'])}: original={original.shape}, latent={latent.shape}."
        )

    beta_used = np.zeros(max_degree + 1, dtype=np.float32)
    if beta.size > 1:
        upto = min(max_degree, beta.size - 1)
        beta_used[: upto + 1] = beta[: upto + 1]
    term_values: list[np.ndarray] = []
    power = np.ones_like(latent, dtype=np.float32)
    for degree in range(max_degree + 1):
        term = float(beta_used[degree]) * power
        term_values.append(term.astype(np.float32))
        power = power * latent

    missing_mask = ~observed_mask
    term_magnitudes = np.array(
        [float(np.mean(np.abs(term[missing_mask]))) for term in term_values],
        dtype=np.float64,
    )

    run_metric_rows: list[dict[str, Any]] = []
    run_contrib_rows: list[dict[str, Any]] = []
    reconstruction = np.zeros_like(latent, dtype=np.float32)
    for degree in range(max_degree + 1):
        reconstruction = reconstruction + term_values[degree]
        if degree < MIN_DEGREE:
            continue
        metrics = _missing_metrics(original, reconstruction, observed_mask)
        run_metric_rows.append(
            {
                "scene_id": int(row["scene_id"]),
                "mask_seed": int(row["mask_seed"]),
                "missing_rate": float(row["missing_rate"]),
                "P": int(degree),
                **metrics,
            }
        )
        denom = float(np.sum(term_magnitudes[: degree + 1]))
        denom = max(denom, 1e-12)
        for term_degree in range(max_degree + 1):
            if term_degree > degree:
                value = np.nan
            else:
                value = float(term_magnitudes[term_degree] / denom)
            run_contrib_rows.append(
                {
                    "scene_id": int(row["scene_id"]),
                    "mask_seed": int(row["mask_seed"]),
                    "P": int(degree),
                    "term": int(term_degree),
                    "effective_contribution": value,
                }
            )
    return run_metric_rows, run_contrib_rows


def _aggregate_metric(summary: pd.DataFrame, col: str) -> pd.DataFrame:
    grouped = summary.groupby("P", as_index=False).agg(
        **{
            f"{col}_mean": (col, "mean"),
            f"{col}_std": (col, "std"),
        }
    )
    grouped[f"{col}_std"] = grouped[f"{col}_std"].fillna(0.0)
    return grouped


def parameter_mechanism_summary_latex(summary: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{c c c c c}",
        r"\toprule",
        r"$P$ & RMSE* & NMSE(dB)* & SAM* & Top contribution terms \\",
        r"\midrule",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            "    "
            + " & ".join(
                [
                    str(int(row["P"])),
                    f"{float(row['RMSE*']):.5f} $\\pm$ {float(row['RMSE*_std']):.5f}",
                    f"{float(row['NMSE(dB)*']):.3f} $\\pm$ {float(row['NMSE(dB)*_std']):.3f}",
                    f"{float(row['SAM*']):.4f} $\\pm$ {float(row['SAM*_std']):.4f}",
                    str(row["top_terms"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def build_cave_parameter_mechanism_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    runs, _ = load_ntdpl_main_runs()
    max_degree = int(min(MAX_DEGREE, int(runs["p_max"].dropna().max())))
    if max_degree < MIN_DEGREE:
        raise RuntimeError("Invalid degree range for cave-parameter mechanism.")

    metric_rows: list[dict[str, Any]] = []
    contrib_rows: list[dict[str, Any]] = []
    skipped = 0
    for row in runs.to_dict("records"):
        try:
            run_metrics, run_contrib = _evaluate_one_run(pd.Series(row), max_degree=max_degree)
        except Exception:
            skipped += 1
            continue
        metric_rows.extend(run_metrics)
        contrib_rows.extend(run_contrib)

    run_metrics = pd.DataFrame(metric_rows).sort_values(["P", "scene_id", "mask_seed"]).reset_index(drop=True)
    contribution_long = pd.DataFrame(contrib_rows).sort_values(["P", "term", "scene_id", "mask_seed"]).reset_index(drop=True)
    if run_metrics.empty or contribution_long.empty:
        raise RuntimeError("Cave parameter mechanism outputs are empty.")

    rmse = _aggregate_metric(run_metrics, "RMSE*")
    nmse = _aggregate_metric(run_metrics, "NMSE(dB)*")
    sam = _aggregate_metric(run_metrics, "SAM*")
    summary = rmse.merge(nmse, on="P", how="inner").merge(sam, on="P", how="inner")
    summary = summary.rename(
        columns={
            "RMSE*_mean": "RMSE*",
            "NMSE(dB)*_mean": "NMSE(dB)*",
            "SAM*_mean": "SAM*",
        }
    ).sort_values("P")

    contribution_summary = (
        contribution_long.groupby(["P", "term"], as_index=False)["effective_contribution"]
        .mean()
        .sort_values(["P", "term"])
        .reset_index(drop=True)
    )

    top_rows: list[dict[str, Any]] = []
    for row in summary.to_dict("records"):
        p_value = int(row["P"])
        panel = contribution_summary.loc[
            contribution_summary["P"].eq(p_value) & contribution_summary["term"].le(p_value)
        ].copy()
        panel = panel.sort_values("effective_contribution", ascending=False).reset_index(drop=True)
        top1_term = int(panel.iloc[0]["term"])
        top1_value = float(panel.iloc[0]["effective_contribution"])
        top2_term = int(panel.iloc[1]["term"]) if len(panel) > 1 else top1_term
        top2_value = float(panel.iloc[1]["effective_contribution"]) if len(panel) > 1 else 0.0
        top_rows.append(
            {
                "P": p_value,
                "top1_term": top1_term,
                "top1_contribution": top1_value,
                "top2_term": top2_term,
                "top2_contribution": top2_value,
                "top_terms": f"{top1_term} ({top1_value:.3f}), {top2_term} ({top2_value:.3f})",
            }
        )
    summary = summary.merge(pd.DataFrame(top_rows), on="P", how="left")

    figure_curve = summary.loc[:, ["P", "RMSE*", "RMSE*_std"]].copy().rename(
        columns={"P": "p_max", "RMSE*": "mean", "RMSE*_std": "std"}
    )
    figure_curve["table"] = "curve"
    figure_curve["band_lower"] = figure_curve["mean"] - figure_curve["std"]
    figure_curve["band_upper"] = figure_curve["mean"] + figure_curve["std"]

    figure_heat = contribution_summary.copy().rename(columns={"P": "p_max", "term": "degree", "effective_contribution": "value"})
    figure_heat["table"] = "heatmap"
    figure_data = pd.concat([figure_curve, figure_heat], ignore_index=True, sort=False)
    figure_data = figure_data.loc[:, ["table", "p_max", "mean", "std", "band_lower", "band_upper", "degree", "value"]]
    figure_data = figure_data.sort_values(["table", "p_max", "degree"], na_position="last").reset_index(drop=True)

    summary = summary.sort_values("P").reset_index(drop=True)
    contribution_summary = contribution_summary.sort_values(["P", "term"]).reset_index(drop=True)
    if skipped:
        print(f"[cave_parameter_mechanism] skipped {skipped} runs due to unreadable/incomplete states.")
    return figure_data, summary, contribution_summary, run_metrics
