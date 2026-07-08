from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import io
from experiment.hsi_defaults import CAVE_RECON_MAIN_RANK
from experiment.process.nonlinear_approx import dedup_nonlinear_runs
from experiment.process.helpers.cave_random_completion import (
    _cave_dataset_kwargs_from_row,
    _resolve_state_path,
    load_main_runs as load_cave_completion_runs,
    load_scene_payload as load_cave_completion_scene_payload,
    select_representative_scene as select_cave_completion_scene,
    build_scene_mean_table as build_cave_completion_scene_mean_table,
    build_scene_gain_table as build_cave_completion_scene_gain_table,
    build_scene_rate_consistency_table as build_cave_completion_scene_rate_consistency_table,
    pseudo_rgb as cave_completion_pseudo_rgb,
)
from experiment.process.helpers.cave_parameter_mechanism import build_cave_parameter_mechanism_outputs
from experiment.process.helpers.cave_random_completion_polycal import load_scene_original
from experiment.process.helpers.real_hsi_robustness import (
    build_overview_figure_data as build_real_hsi_overview_figure_data,
    build_summary as build_real_hsi_summary,
    load_main_runs as load_real_hsi_main_runs,
)
from experiment.utils.io import load_state_mat


NONLINEAR_ORDER = ("poly2", "poly3", "tanh", "exp")
NONLINEAR_ALPHA_REF = 0.25
NONLINEAR_STEP_PMAX = 5
CAVE_REPR_MAIN_RANK = CAVE_RECON_MAIN_RANK
CAVE_REPR_MAIN_PMAX = 6
CAVE_VISUAL_SCENES = (14, 8, 2)
GEOMETRY_REFERENCE_ALPHA = 0.3
GEOMETRY_REFERENCE_PMAX = 4
GEOMETRY_ORDER_VALUES = (1, 2, 3, 4)
METHOD_LABELS = {"tucker": "Tucker", "cp": "CP", "tt": "TT", "tr": "TR", "ntdpl": "NTD-PL"}
CAVE_COMPLETION_MAIN_MISSING_RATE = 0.5
CAVE_COMPLETION_VISUAL_SCENES = 3
CAVE_COMPLETION_FOCUS_SCENES = (2, 3, 8)  # beads, cd, feathers
ADVANTAGE_SPATIAL_CASE_SCENE_ID = 3  # cd
ADVANTAGE_QUARTILE_COUNT = 4


def _first_present(frame: pd.DataFrame, *names: str) -> str:
    for name in names:
        if name in frame.columns:
            return name
    raise KeyError(f"Missing required columns: {names!r}")


def _series_str(frame: pd.DataFrame, *names: str) -> pd.Series:
    return frame[_first_present(frame, *names)].astype(str).str.strip().str.strip('"')


def _series_num(frame: pd.DataFrame, *names: str) -> pd.Series:
    series = frame[_first_present(frame, *names)]
    if series.dtype == object:
        series = series.astype(str).str.strip().str.strip('"')
    return io.maybe_numeric(series)


def _with_rank(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["rank"] = out[_first_present(out, "ovr.method.rank", "method.rank")].map(io.parse_rank)
    out["rank_text"] = out["rank"].map(io.rank_text)
    return out


def aggregate_linear_paired_gap() -> pd.DataFrame:
    runs = io.load_runs("linear-consistency").copy()
    runs["method_name"] = _series_str(runs, "ovr.method", "method._name")
    runs["bias"] = _series_num(runs, "ovr.filter.bias", "filter.bias").astype(float)
    runs["seed"] = _series_num(runs, "ovr.data.seed", "data.seed").astype(int)
    runs["rmse"] = _series_num(runs, "RMSE").astype(float)
    if "ovr.method.allow_constant_term" in runs.columns:
        runs["allow_constant_term"] = _series_str(runs, "ovr.method.allow_constant_term").str.lower().eq("true")
    else:
        runs["allow_constant_term"] = _series_str(runs, "method.allow_constant_term").str.lower().eq("true")

    rows: list[dict[str, Any]] = []
    panel_specs = [(0.0, "strict_linear", "bias=0"), (0.5, "affine_shift", "bias=0.5")]
    for bias, panel_key, panel_title in panel_specs:
        tucker = runs.loc[(runs["method_name"] == "tucker") & np.isclose(runs["bias"], bias)].copy()
        for variant_key, variant_title, allow_constant in (
            ("strict", r"Restricted $\beta_0=0$", False),
            ("affine", r"Free $\beta_0$", True),
        ):
            ntdpl = runs.loc[
                (runs["method_name"] == "ntdpl")
                & np.isclose(runs["bias"], bias)
                & runs["allow_constant_term"].eq(allow_constant)
            ].copy()
            merged = (
                tucker.loc[:, ["seed", "rmse"]]
                .rename(columns={"rmse": "rmse_tucker"})
                .merge(ntdpl.loc[:, ["seed", "rmse"]], on="seed", how="inner")
                .rename(columns={"rmse": "rmse_ntdpl"})
                .sort_values("seed")
            )
            for row in merged.itertuples(index=False):
                rows.append(
                    {
                        "panel_key": panel_key,
                        "panel_title": panel_title,
                        "variant_key": variant_key,
                        "variant_title": variant_title,
                        "seed": int(row.seed),
                        "gap": float(row.rmse_tucker - row.rmse_ntdpl),
                    }
                )
    return pd.DataFrame(rows)


def _nonlinear_runs() -> pd.DataFrame:
    runs = dedup_nonlinear_runs(io.load_runs("nonlinear-approx").copy())
    runs["method_name"] = _series_str(runs, "ovr.method", "method._name")
    runs["nonlinear"] = _series_str(runs, "ovr.filter.nonlinear", "filter.nonlinear")
    runs["alpha"] = _series_num(runs, "ovr.filter.alpha", "filter.alpha").astype(float)
    runs["seed"] = _series_num(runs, "ovr.data.seed", "data.seed").astype(int)
    runs["rmse"] = _series_num(runs, "RMSE").astype(float)
    runs["p_max"] = _series_num(runs, "ovr.method.p_max", "method.p_max")
    return runs


def _nonlinear_ntdpl_alpha_pmax(frame: pd.DataFrame) -> int:
    panel = frame.loc[frame["method_name"] == "ntdpl", "p_max"].dropna()
    return int(float(panel.max()))


def aggregate_nonlinear_alpha_grid() -> pd.DataFrame:
    runs = _nonlinear_runs()
    ntdpl_pmax = _nonlinear_ntdpl_alpha_pmax(runs)
    rows: list[dict[str, Any]] = []
    for nonlinear in NONLINEAR_ORDER:
        panel = runs.loc[runs["nonlinear"] == nonlinear].copy()
        for method in ("tucker", "cp", "tt", "tr", "ntdpl"):
            sub = panel.loc[panel["method_name"] == method].copy()
            if method == "ntdpl":
                sub = sub.loc[np.isclose(sub["p_max"], float(ntdpl_pmax), atol=1e-12)].copy()
            if sub.empty:
                continue
            grouped = sub.groupby(["seed", "alpha"], as_index=False)["rmse"].mean().rename(columns={"rmse": "seed_value"})
            summary = grouped.groupby("alpha", as_index=False)["seed_value"].agg(["mean", "std"]).reset_index()
            summary["std"] = summary["std"].fillna(0.0)
            for row in summary.itertuples(index=False):
                rows.append(
                    {
                        "panel": nonlinear,
                        "method": METHOD_LABELS[method],
                        "x": float(row.alpha),
                        "mean": float(row.mean),
                        "std": float(row.std),
                        "band_lower": float(row.mean - row.std),
                        "band_upper": float(row.mean + row.std),
                        "annotation": rf"$P={ntdpl_pmax}$" if method == "ntdpl" else "",
                    }
                )
    return pd.DataFrame(rows)


def aggregate_nonlinear_pmax_grid() -> pd.DataFrame:
    runs = _nonlinear_runs()
    rows: list[dict[str, Any]] = []
    for nonlinear in NONLINEAR_ORDER:
        panel = runs.loc[(runs["nonlinear"] == nonlinear) & np.isclose(runs["alpha"], NONLINEAR_ALPHA_REF)].copy()
        ntdpl = panel.loc[panel["method_name"] == "ntdpl"].copy()
        for method in ("tucker", "ntdpl"):
            sub = panel.loc[panel["method_name"] == method].copy()
            if sub.empty:
                continue
            if method == "tucker":
                p_values = sorted(int(v) for v in ntdpl["p_max"].dropna().unique().tolist())
                replicated = []
                for p_value in p_values:
                    item = sub.copy()
                    item["p_max"] = p_value
                    replicated.append(item)
                sub = pd.concat(replicated, ignore_index=True) if replicated else sub
            grouped = sub.groupby(["seed", "p_max"], as_index=False)["rmse"].mean().rename(columns={"rmse": "seed_value"})
            summary = grouped.groupby("p_max", as_index=False)["seed_value"].agg(["mean", "std"]).reset_index()
            summary["std"] = summary["std"].fillna(0.0)
            for row in summary.itertuples(index=False):
                rows.append(
                    {
                        "panel": nonlinear,
                        "method": METHOD_LABELS[method],
                        "x": int(row.p_max),
                        "mean": float(row.mean),
                        "std": float(row.std),
                        "band_lower": float(row.mean - row.std),
                        "band_upper": float(row.mean + row.std),
                        "annotation": rf"$\alpha={NONLINEAR_ALPHA_REF:g}$",
                    }
                )
    return pd.DataFrame(rows)


def aggregate_nonlinear_step_grid() -> pd.DataFrame:
    curves = io.load_curves("nonlinear-approx").copy()
    curves["method_name"] = _series_str(curves, "ovr.method")
    curves["nonlinear"] = _series_str(curves, "ovr.filter.nonlinear")
    curves["alpha"] = _series_num(curves, "ovr.filter.alpha").astype(float)
    curves["p_max"] = _series_num(curves, "ovr.method.p_max").astype(float)
    curves["step"] = _series_num(curves, "step").astype(int)
    curves["value"] = _series_num(curves, "value").astype(float)
    metric_col = _series_str(curves, "metric")
    rmse = curves.loc[
        metric_col.eq("RMSE")
        & curves["method_name"].eq("ntdpl")
        & np.isclose(curves["alpha"], NONLINEAR_ALPHA_REF)
        & np.isclose(curves["p_max"], float(NONLINEAR_STEP_PMAX))
    ].copy()
    p_rows = curves.loc[
        metric_col.eq("p")
        & curves["method_name"].eq("ntdpl")
        & np.isclose(curves["alpha"], NONLINEAR_ALPHA_REF)
        & np.isclose(curves["p_max"], float(NONLINEAR_STEP_PMAX))
    ].copy()
    rows: list[dict[str, Any]] = []
    for nonlinear in NONLINEAR_ORDER:
        panel = rmse.loc[rmse["nonlinear"] == nonlinear].copy()
        if panel.empty:
            continue
        grouped = panel.groupby(["run_id", "step"], as_index=False)["value"].mean().rename(columns={"value": "seed_value"})
        summary = grouped.groupby("step", as_index=False)["seed_value"].agg(["mean", "std"]).reset_index()
        summary["std"] = summary["std"].fillna(0.0)
        summary["kind"] = "curve"
        summary["panel"] = nonlinear
        summary["method"] = "NTD-PL"
        summary["annotation"] = rf"$\alpha={NONLINEAR_ALPHA_REF:g},\ P={NONLINEAR_STEP_PMAX}$"
        rows.extend(summary.rename(columns={"step": "x"}).to_dict("records"))

        transition_panel = p_rows.loc[p_rows["nonlinear"] == nonlinear].copy()
        if not transition_panel.empty:
            run_id = str(sorted(transition_panel["run_id"].astype(str).unique())[0])
            transition_panel = transition_panel.loc[transition_panel["run_id"].astype(str) == run_id].copy()
            transition_panel = transition_panel.sort_values("step")
            previous = transition_panel["value"].shift(1)
            increased = transition_panel.loc[(previous.notna()) & (transition_panel["value"] > previous)]
            for idx, item in enumerate(increased.itertuples(index=False), start=2):
                rows.append(
                    {
                        "panel": nonlinear,
                        "kind": "transition",
                        "method": "NTD-PL",
                        "x": int(item.step),
                        "mean": np.nan,
                        "std": np.nan,
                        "band_lower": np.nan,
                        "band_upper": np.nan,
                        "degree": idx,
                        "annotation": rf"$\alpha={NONLINEAR_ALPHA_REF:g},\ P={NONLINEAR_STEP_PMAX}$",
                    }
                )
    return pd.DataFrame(rows)


def aggregate_cave_representation_scene_improvement() -> pd.DataFrame:
    runs = _load_cave_representation_rows()
    panel = runs.loc[runs["rank"] == CAVE_REPR_MAIN_RANK].copy()
    panel["rmse"] = _series_num(panel, "RMSE").astype(float)
    grouped = (
        panel.groupby(["scene_id", "method_name"], as_index=False)["rmse"]
        .mean()
        .pivot(index="scene_id", columns="method_name", values="rmse")
        .reset_index()
    )
    grouped = grouped.dropna(subset=["tucker", "ntdpl"]).copy()
    grouped["panel"] = "cave_reconstruction"
    grouped["gain"] = grouped["tucker"] - grouped["ntdpl"]
    grouped["scene_name"] = grouped["scene_id"].map(lambda scene_id: io.load_cave_scene(int(scene_id))[1])
    return grouped.sort_values("gain", ascending=False).reset_index(drop=True)


def _load_cave_representation_rows() -> pd.DataFrame:
    runs = _with_rank(io.load_runs("cave-representation").copy())
    runs["method_name"] = _series_str(runs, "ovr.method", "method._name")
    runs["scene_id"] = _series_num(runs, "ovr.data.id", "data.id").astype(int)
    runs["p_max"] = _series_num(runs, "ovr.method.p_max", "method.p_max")
    return runs


def _pick_representation_row(frame: pd.DataFrame, *, scene_id: int, method_name: str) -> pd.Series:
    panel = frame.loc[(frame["scene_id"] == int(scene_id)) & (frame["method_name"] == method_name)].copy()
    panel = panel.loc[panel["rank"] == CAVE_REPR_MAIN_RANK].copy()
    if method_name == "ntdpl":
        panel = panel.loc[np.isclose(panel["p_max"], float(CAVE_REPR_MAIN_PMAX), atol=1e-12)].copy()
    panel = panel.sort_values("RMSE")
    if panel.empty:
        raise RuntimeError(f"Missing cave-representation row for scene={scene_id}, method={method_name}.")
    return panel.iloc[0]


def _cave_representation_payload(scene_id: int) -> dict[str, Any]:
    runs = _load_cave_representation_rows()
    tucker_state = io.load_state(_pick_representation_row(runs, scene_id=scene_id, method_name="tucker")["state_path"])
    ntdpl_state = io.load_state(_pick_representation_row(runs, scene_id=scene_id, method_name="ntdpl")["state_path"])
    tucker_recon = io.reconstruct_observation(tucker_state)
    ntdpl_recon = io.reconstruct_observation(ntdpl_state)
    target_shape = (int(tucker_recon.shape[0]), int(tucker_recon.shape[1]))
    original, scene_name = io.load_cave_scene(scene_id, target_shape=target_shape)
    return {
        "scene_id": int(scene_id),
        "scene_name": scene_name,
        "original": original,
        "tucker_recon": tucker_recon,
        "ntdpl_recon": ntdpl_recon,
    }


def aggregate_cave_representation_image_panels() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scene_id in CAVE_VISUAL_SCENES:
        payload = _cave_representation_payload(scene_id)
        original = payload["original"]
        scene_name = str(payload["scene_name"])
        tucker_recon = payload["tucker_recon"]
        ntdpl_recon = payload["ntdpl_recon"]
        tucker_error = io.rmse_map(original, tucker_recon)
        ntdpl_error = io.rmse_map(original, ntdpl_recon)
        diff_map = tucker_error - ntdpl_error
        panel_map = {
            "original": io.pseudo_rgb(original),
            "tucker": io.pseudo_rgb(tucker_recon),
            "ntdpl": io.pseudo_rgb(ntdpl_recon),
            "tucker_error": tucker_error,
            "ntdpl_error": ntdpl_error,
            "error_reduction": diff_map,
        }
        for panel_key, image in panel_map.items():
            rows.append(
                {
                    "scene_id": int(scene_id),
                    "scene_name": scene_name,
                    "panel": panel_key,
                    "image": image,
                    "panel_type": "rgb" if panel_key in {"original", "tucker", "ntdpl"} else ("improvement" if panel_key == "error_reduction" else "error"),
                    "is_difference": panel_key == "error_reduction",
                }
            )
    return pd.DataFrame(rows)


def _spectral_sam_deg(reference: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    numerator = np.sum(reference * estimate, axis=-1)
    denominator = np.linalg.norm(reference, axis=-1) * np.linalg.norm(estimate, axis=-1)
    cosine = np.clip(numerator / np.maximum(denominator, 1e-12), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def _spatial_gradient_map(image: np.ndarray) -> np.ndarray:
    grad_y, grad_x = np.gradient(image.astype(np.float32), edge_order=1)
    return np.sqrt(grad_x**2 + grad_y**2)


def _select_scene_spectral_points(original: np.ndarray, tucker_recon: np.ndarray, ntdpl_recon: np.ndarray) -> list[tuple[int, int, str]]:
    intensity = np.mean(original, axis=-1)
    gradient = _spatial_gradient_map(intensity)
    peak = np.max(original, axis=-1)
    rmse_tucker = io.rmse_map(original, tucker_recon)
    rmse_ntdpl = io.rmse_map(original, ntdpl_recon)
    sam_tucker = _spectral_sam_deg(original, tucker_recon)
    sam_ntdpl = _spectral_sam_deg(original, ntdpl_recon)
    rmse_gain = rmse_tucker - rmse_ntdpl
    sam_gain = sam_tucker - sam_ntdpl

    valid_signal = (
        (intensity >= np.quantile(intensity, 0.20))
        & (intensity <= np.quantile(intensity, 0.93))
        & (peak >= np.quantile(peak, 0.25))
    )

    highlight_mask = (
        valid_signal
        & (rmse_gain >= np.quantile(rmse_gain, 0.94))
        & (sam_gain >= np.quantile(sam_gain, 0.72))
        & (gradient >= np.quantile(gradient, 0.65))
    )
    if not np.any(highlight_mask):
        highlight_mask = valid_signal & (rmse_gain >= np.quantile(rmse_gain, 0.90))
    highlight_score = rmse_gain + 0.30 * sam_gain + 0.15 * gradient
    highlight_points = np.argwhere(highlight_mask)
    if len(highlight_points) == 0:
        raise RuntimeError("Failed to select a spectral highlight pixel.")
    highlight_points = sorted(highlight_points.tolist(), key=lambda rc: float(highlight_score[rc[0], rc[1]]), reverse=True)
    highlight_row, highlight_col = (int(highlight_points[0][0]), int(highlight_points[0][1]))

    typical_mask = (
        valid_signal
        & (rmse_gain >= np.quantile(rmse_gain, 0.68))
        & (rmse_gain <= np.quantile(rmse_gain, 0.90))
        & (sam_gain >= np.quantile(sam_gain, 0.45))
        & (gradient >= np.quantile(gradient, 0.45))
        & (gradient <= np.quantile(gradient, 0.88))
    )
    if not np.any(typical_mask):
        typical_mask = valid_signal & (rmse_gain >= np.quantile(rmse_gain, 0.60))
    rr, cc = np.indices(intensity.shape)
    distance_penalty = np.sqrt((rr - highlight_row) ** 2 + (cc - highlight_col) ** 2)
    typical_score = rmse_gain + 0.20 * sam_gain + 0.10 * gradient + 0.015 * distance_penalty
    typical_points = np.argwhere(typical_mask)
    typical_points = sorted(typical_points.tolist(), key=lambda rc: float(typical_score[rc[0], rc[1]]), reverse=True)
    if not typical_points:
        raise RuntimeError("Failed to select a typical spectral pixel.")
    typical_row = typical_col = 0
    for row, col in typical_points:
        if float(distance_penalty[row, col]) >= 12.0:
            typical_row, typical_col = int(row), int(col)
            break
    else:
        typical_row, typical_col = (int(typical_points[0][0]), int(typical_points[0][1]))
    return [(highlight_row, highlight_col, "high-gain"), (typical_row, typical_col, "typical")]


def aggregate_cave_representation_spectra() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    panel_index = 1
    for scene_id in CAVE_VISUAL_SCENES:
        payload = _cave_representation_payload(scene_id)
        original = payload["original"]
        tucker_recon = payload["tucker_recon"]
        ntdpl_recon = payload["ntdpl_recon"]
        scene_name = str(payload["scene_name"])
        map_rgb = io.pseudo_rgb(original)
        for pixel_row, pixel_col, _category in _select_scene_spectral_points(original, tucker_recon, ntdpl_recon):
            pixel_label = f"P{panel_index}"
            rows.append(
                {
                    "kind": "map",
                    "scene_id": int(scene_id),
                    "scene_name": scene_name,
                    "pixel_label": pixel_label,
                    "row": int(pixel_row),
                    "col": int(pixel_col),
                    "image": map_rgb,
                    "panel_index": panel_index,
                }
            )
            band_axis = np.arange(1, original.shape[-1] + 1)
            for method, curve in (
                ("Ground truth", original[pixel_row, pixel_col]),
                ("Tucker", tucker_recon[pixel_row, pixel_col]),
                ("NTD-PL", ntdpl_recon[pixel_row, pixel_col]),
            ):
                for band, value in zip(band_axis, curve, strict=False):
                    rows.append(
                        {
                            "kind": "curve",
                            "scene_id": int(scene_id),
                            "scene_name": scene_name,
                            "pixel_label": pixel_label,
                            "row": int(pixel_row),
                            "col": int(pixel_col),
                            "method": method,
                            "band": int(band),
                            "value": float(value),
                            "panel_index": panel_index,
                        }
                    )
            panel_index += 1
    return pd.DataFrame(rows)


def aggregate_cave_random_completion_scene_improvement() -> pd.DataFrame:
    frame = io.load_output_csv("cave-random-completion", "random_completion_scene_gains.csv").copy()
    frame["panel"] = frame["missing_rate"].map(lambda value: f"rho_{float(value):.1f}")
    frame["gain"] = io.maybe_numeric(frame["RMSE_gain"]).astype(float)
    return frame


def _load_completion_scene_rate_consistency() -> pd.DataFrame:
    try:
        frame = io.load_output_csv("cave-random-completion", "random_completion_scene_rate_consistency.csv").copy()
    except FileNotFoundError:
        raw, _ = load_cave_completion_runs()
        scene_mean = build_cave_completion_scene_mean_table(raw)
        scene_gain = build_cave_completion_scene_gain_table(scene_mean)
        frame = build_cave_completion_scene_rate_consistency_table(scene_gain)
    frame["scene_id"] = io.maybe_numeric(frame["scene_id"]).astype(int)
    frame["missing_rate"] = io.maybe_numeric(frame["missing_rate"]).astype(float)
    frame["delta_rmse_missing"] = io.maybe_numeric(frame["RMSE_gain"]).astype(float)
    frame["delta_sam_missing"] = io.maybe_numeric(frame["SAM_gain"]).astype(float)
    frame["scene_sort_index"] = io.maybe_numeric(frame["scene_sort_index"]).astype(int)
    frame["missing_rate_label"] = frame["missing_rate"].map(lambda value: f"{float(value):.1f}")
    frame["scene_label"] = frame["scene_id"].map(lambda value: f"S{int(value):02d}")
    return frame.sort_values(["scene_sort_index", "missing_rate"]).reset_index(drop=True)


def aggregate_cave_random_completion_scene_rate_consistency() -> pd.DataFrame:
    return _load_completion_scene_rate_consistency()


def _missing_only_rmse_map(reference: np.ndarray, estimate: np.ndarray, missing_mask: np.ndarray) -> np.ndarray:
    diff2 = (np.asarray(reference, dtype=float) - np.asarray(estimate, dtype=float)) ** 2
    mask = np.asarray(missing_mask, dtype=float)
    missing_count = np.sum(mask, axis=-1)
    numer = np.sum(diff2 * mask, axis=-1)
    output = np.full_like(missing_count, np.nan, dtype=float)
    valid = missing_count > 0
    output[valid] = np.sqrt(numer[valid] / missing_count[valid])
    return output


def _observed_input_rgb(original: np.ndarray, observed_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(observed_mask, dtype=bool)
    base = cave_completion_pseudo_rgb(np.asarray(original, dtype=float))
    observed_frac = np.mean(mask.astype(float), axis=-1)
    observed_frac = np.clip(observed_frac, 0.0, 1.0)
    # Keep observed pixels vivid while fading missing regions to a clean neutral gray.
    gray = np.full_like(base, 0.80)
    alpha = np.power(observed_frac, 1.15)[..., None]
    return np.clip(alpha * base + (1.0 - alpha) * gray, 0.0, 1.0)


def _select_completion_visual_scenes(frame: pd.DataFrame, scene_gain: pd.DataFrame, *, missing_rate: float, count: int) -> list[int]:
    panel = scene_gain.loc[np.isclose(scene_gain["missing_rate"], float(missing_rate), atol=1e-12)].copy()
    if panel.empty:
        return [int(frame["scene_id"].iloc[0])]
    panel = panel.sort_values("RMSE_gain", ascending=False).reset_index(drop=True)
    positive = panel.loc[panel["RMSE_gain"] > 0.0].copy()
    if positive.empty:
        return [int(panel.iloc[0]["scene_id"])]
    chosen: list[int] = []
    quantiles = np.linspace(0.2, 0.8, num=max(count, 1))
    for q in quantiles:
        idx = int(round(float(q) * (len(positive) - 1)))
        candidate = int(positive.iloc[idx]["scene_id"])
        if candidate not in chosen:
            chosen.append(candidate)
    for row in positive.itertuples(index=False):
        if len(chosen) >= count:
            break
        scene_id = int(row.scene_id)
        if scene_id not in chosen:
            chosen.append(scene_id)
    return chosen[:count]


def aggregate_cave_random_completion_visual_grid() -> pd.DataFrame:
    frame, _ = load_cave_completion_runs()
    scene_mean = build_cave_completion_scene_mean_table(frame)
    scene_gain = build_cave_completion_scene_gain_table(scene_mean)
    available_scene_ids = set(frame["scene_id"].astype(int).unique().tolist())
    scene_ids = [scene_id for scene_id in CAVE_COMPLETION_FOCUS_SCENES if int(scene_id) in available_scene_ids]
    if len(scene_ids) < CAVE_COMPLETION_VISUAL_SCENES:
        default_scene = select_cave_completion_scene(frame, scene_gain, missing_rate=CAVE_COMPLETION_MAIN_MISSING_RATE)
        if int(default_scene) not in scene_ids:
            scene_ids.append(int(default_scene))
    if len(scene_ids) < CAVE_COMPLETION_VISUAL_SCENES:
        fallback_scene_ids = _select_completion_visual_scenes(
            frame,
            scene_gain,
            missing_rate=CAVE_COMPLETION_MAIN_MISSING_RATE,
            count=CAVE_COMPLETION_VISUAL_SCENES,
        )
        for scene_id in fallback_scene_ids:
            sid = int(scene_id)
            if sid not in scene_ids:
                scene_ids.append(sid)
            if len(scene_ids) >= CAVE_COMPLETION_VISUAL_SCENES:
                break
    scene_ids = scene_ids[:CAVE_COMPLETION_VISUAL_SCENES]

    gain_panel = scene_gain.loc[np.isclose(scene_gain["missing_rate"], float(CAVE_COMPLETION_MAIN_MISSING_RATE), atol=1e-12)].copy()
    rmse_gain_lookup = {
        int(row.scene_id): float(row.RMSE_gain)
        for row in gain_panel.itertuples(index=False)
    }

    rows: list[dict[str, Any]] = []
    for scene_id in scene_ids:
        payload = load_cave_completion_scene_payload(
            frame,
            scene_id=int(scene_id),
            missing_rate=CAVE_COMPLETION_MAIN_MISSING_RATE,
        )
        original = np.asarray(payload.original, dtype=float)
        observed_mask = np.asarray(payload.observed_mask, dtype=bool)
        missing_mask = ~observed_mask
        tucker_recon = np.asarray(payload.recon_tucker, dtype=float)
        ntdpl_recon = np.asarray(payload.recon_ntdpl, dtype=float)
        tucker_error = _missing_only_rmse_map(original, tucker_recon, missing_mask)
        ntdpl_error = _missing_only_rmse_map(original, ntdpl_recon, missing_mask)
        gain_map = tucker_error - ntdpl_error
        scene_name = str(payload.scene_name)
        panel_map = {
            "original": (cave_completion_pseudo_rgb(original), "rgb"),
            "observed_input": (_observed_input_rgb(original, observed_mask), "rgb"),
            "tucker_completion": (cave_completion_pseudo_rgb(tucker_recon), "rgb"),
            "ntdpl_completion": (cave_completion_pseudo_rgb(ntdpl_recon), "rgb"),
            "tucker_missing_error": (tucker_error, "error"),
            "ntdpl_missing_error": (ntdpl_error, "error"),
            "missing_error_reduction": (gain_map, "improvement"),
        }
        for panel_key, (image, panel_type) in panel_map.items():
            rows.append(
                {
                    "scene_id": int(scene_id),
                    "scene_name": scene_name,
                    "panel": panel_key,
                    "image": image,
                    "panel_type": panel_type,
                    "missing_rate": float(CAVE_COMPLETION_MAIN_MISSING_RATE),
                    "rmse_gain": float(rmse_gain_lookup.get(int(scene_id), np.nan)),
                }
            )
    return pd.DataFrame(rows)


def _missing_only_sam_map(reference: np.ndarray, estimate: np.ndarray, missing_mask: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference, dtype=np.float64)
    est = np.asarray(estimate, dtype=np.float64)
    mask = np.asarray(missing_mask, dtype=bool)
    masked_ref = np.where(mask, ref, 0.0)
    masked_est = np.where(mask, est, 0.0)
    numerator = np.sum(masked_ref * masked_est, axis=-1)
    ref_norm = np.linalg.norm(masked_ref, axis=-1)
    est_norm = np.linalg.norm(masked_est, axis=-1)
    denom = np.maximum(ref_norm * est_norm, 1e-12)
    cosine = np.clip(numerator / denom, -1.0, 1.0)
    angles = np.degrees(np.arccos(cosine))
    invalid = (np.sum(mask, axis=-1) <= 0) | (ref_norm <= 1e-12) | (est_norm <= 1e-12)
    angles = angles.astype(float)
    angles[invalid] = np.nan
    return angles


def _boundary_score_from_cube(cube: np.ndarray) -> np.ndarray:
    spectral_cube = np.asarray(cube, dtype=np.float32)
    grad_row, grad_col = np.gradient(spectral_cube, axis=(0, 1), edge_order=1)
    grad_mag = np.sqrt(grad_row * grad_row + grad_col * grad_col)
    return np.mean(grad_mag, axis=-1)


def _rank_quartile_map(score_map: np.ndarray, n_bins: int = ADVANTAGE_QUARTILE_COUNT) -> np.ndarray:
    flat = np.asarray(score_map, dtype=np.float64).reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    bins = np.empty_like(order, dtype=np.int32)
    total = flat.size
    for rank, flat_idx in enumerate(order):
        bins[flat_idx] = min((rank * n_bins) // max(total, 1), n_bins - 1)
    return bins.reshape(score_map.shape)


def _advantage_scene_maps_from_states(frame: pd.DataFrame, *, scene_id: int, missing_rate: float) -> dict[str, Any]:
    panel = frame.loc[
        np.isclose(frame["missing_rate"], float(missing_rate), atol=1e-12)
        & frame["scene_id"].eq(int(scene_id))
    ].copy()
    if panel.empty:
        raise RuntimeError(f"Missing scene rows for scene={scene_id}, missing_rate={missing_rate}.")
    paired = (
        panel.groupby(["mask_seed", "method_name"], as_index=False)
        .size()
        .pivot(index="mask_seed", columns="method_name", values="size")
        .fillna(0)
    )
    valid_seeds = [
        int(seed)
        for seed, row in paired.iterrows()
        if row.get("tucker", 0) > 0 and row.get("ntdpl", 0) > 0
    ]
    if not valid_seeds:
        raise RuntimeError(f"No paired tucker/ntdpl seeds for scene={scene_id}.")

    ref_row = panel.iloc[0]
    scene_name, original = load_scene_original(int(scene_id), **_cave_dataset_kwargs_from_row(ref_row))
    tucker_rmse_maps: list[np.ndarray] = []
    ntdpl_rmse_maps: list[np.ndarray] = []
    tucker_sam_maps: list[np.ndarray] = []
    ntdpl_sam_maps: list[np.ndarray] = []
    tucker_recons: list[np.ndarray] = []
    ntdpl_recons: list[np.ndarray] = []
    missing_masks: list[np.ndarray] = []

    for seed in sorted(valid_seeds):
        row_t = panel.loc[panel["mask_seed"].eq(seed) & panel["method_name"].eq("tucker")].iloc[0]
        row_n = panel.loc[panel["mask_seed"].eq(seed) & panel["method_name"].eq("ntdpl")].iloc[0]
        state_t = load_state_mat(_resolve_state_path(row_t["state_path"]))
        state_n = load_state_mat(_resolve_state_path(row_n["state_path"]))
        observed_mask = np.asarray(state_t["observed_mask"], dtype=bool)
        missing_mask = ~observed_mask
        recon_tucker = np.asarray(state_t["reconstruction"], dtype=np.float32)
        recon_ntdpl = np.asarray(state_n["reconstruction"], dtype=np.float32)
        tucker_recons.append(recon_tucker)
        ntdpl_recons.append(recon_ntdpl)
        missing_masks.append(missing_mask)
        tucker_rmse_maps.append(_missing_only_rmse_map(original, recon_tucker, missing_mask))
        ntdpl_rmse_maps.append(_missing_only_rmse_map(original, recon_ntdpl, missing_mask))
        tucker_sam_maps.append(_missing_only_sam_map(original, recon_tucker, missing_mask))
        ntdpl_sam_maps.append(_missing_only_sam_map(original, recon_ntdpl, missing_mask))

    tucker_rmse = np.nanmean(np.stack(tucker_rmse_maps, axis=0), axis=0)
    ntdpl_rmse = np.nanmean(np.stack(ntdpl_rmse_maps, axis=0), axis=0)
    tucker_sam = np.nanmean(np.stack(tucker_sam_maps, axis=0), axis=0)
    ntdpl_sam = np.nanmean(np.stack(ntdpl_sam_maps, axis=0), axis=0)
    gain_rmse = tucker_rmse - ntdpl_rmse
    gain_sam = tucker_sam - ntdpl_sam
    difficulty_score = tucker_rmse
    boundary_score = _boundary_score_from_cube(original)
    difficulty_bins = _rank_quartile_map(difficulty_score, ADVANTAGE_QUARTILE_COUNT)
    boundary_bins = _rank_quartile_map(boundary_score, ADVANTAGE_QUARTILE_COUNT)

    return {
        "scene_id": int(scene_id),
        "scene_name": str(scene_name),
        "original": np.asarray(original, dtype=np.float32),
        "recon_tucker_mean": np.mean(np.stack(tucker_recons, axis=0), axis=0).astype(np.float32),
        "recon_ntdpl_mean": np.mean(np.stack(ntdpl_recons, axis=0), axis=0).astype(np.float32),
        "missing_mask_ratio": np.mean(np.stack(missing_masks, axis=0).astype(float), axis=0),
        "difficulty_score": difficulty_score,
        "boundary_score": boundary_score,
        "difficulty_bins": difficulty_bins,
        "boundary_bins": boundary_bins,
        "delta_rmse_missing": gain_rmse,
        "delta_sam_missing": gain_sam,
    }


def _build_advantage_quartile_tables(
    *,
    missing_rate: float = CAVE_COMPLETION_MAIN_MISSING_RATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, dict[str, Any]]]:
    frame, _ = load_cave_completion_runs()
    scene_ids = sorted(frame["scene_id"].astype(int).unique().tolist())
    per_scene_maps: dict[int, dict[str, Any]] = {}
    scene_rows: list[dict[str, Any]] = []
    for scene_id in scene_ids:
        try:
            maps = _advantage_scene_maps_from_states(frame, scene_id=int(scene_id), missing_rate=missing_rate)
        except Exception:
            continue
        per_scene_maps[int(scene_id)] = maps
        for diff_bin in range(ADVANTAGE_QUARTILE_COUNT):
            for bnd_bin in range(ADVANTAGE_QUARTILE_COUNT):
                mask = (maps["difficulty_bins"] == diff_bin) & (maps["boundary_bins"] == bnd_bin)
                rmse_values = maps["delta_rmse_missing"][mask]
                sam_values = maps["delta_sam_missing"][mask]
                rmse_valid = np.isfinite(rmse_values)
                sam_valid = np.isfinite(sam_values)
                scene_rows.append(
                    {
                        "scene_id": int(scene_id),
                        "scene_name": str(maps["scene_name"]),
                        "difficulty_bin": int(diff_bin),
                        "boundary_bin": int(bnd_bin),
                        "difficulty_quartile": f"Q{diff_bin + 1}",
                        "boundary_quartile": f"Q{bnd_bin + 1}",
                        "delta_rmse_missing": float(np.nanmean(rmse_values)) if np.any(rmse_valid) else np.nan,
                        "delta_sam_missing": float(np.nanmean(sam_values)) if np.any(sam_valid) else np.nan,
                        "pixel_count_rmse": int(np.sum(rmse_valid)),
                        "pixel_count_sam": int(np.sum(sam_valid)),
                    }
                )
    scene_df = pd.DataFrame(scene_rows)
    if scene_df.empty:
        raise RuntimeError("No valid scene quartile statistics for cave completion advantage analysis.")

    summary_df = (
        scene_df.groupby(["difficulty_bin", "boundary_bin"], as_index=False)
        .agg(
            difficulty_quartile=("difficulty_quartile", "first"),
            boundary_quartile=("boundary_quartile", "first"),
            delta_rmse_missing=("delta_rmse_missing", "mean"),
            delta_sam_missing=("delta_sam_missing", "mean"),
            scene_count=("scene_id", "nunique"),
            pixel_count_rmse=("pixel_count_rmse", "sum"),
            pixel_count_sam=("pixel_count_sam", "sum"),
        )
        .sort_values(["boundary_bin", "difficulty_bin"])
        .reset_index(drop=True)
    )
    difficulty_marginal = (
        scene_df.groupby(["difficulty_bin", "difficulty_quartile"], as_index=False)
        .agg(
            delta_rmse_missing=("delta_rmse_missing", "mean"),
            delta_sam_missing=("delta_sam_missing", "mean"),
            scene_count=("scene_id", "nunique"),
            pixel_count_rmse=("pixel_count_rmse", "sum"),
            pixel_count_sam=("pixel_count_sam", "sum"),
        )
        .sort_values("difficulty_bin")
        .reset_index(drop=True)
    )
    boundary_marginal = (
        scene_df.groupby(["boundary_bin", "boundary_quartile"], as_index=False)
        .agg(
            delta_rmse_missing=("delta_rmse_missing", "mean"),
            delta_sam_missing=("delta_sam_missing", "mean"),
            scene_count=("scene_id", "nunique"),
            pixel_count_rmse=("pixel_count_rmse", "sum"),
            pixel_count_sam=("pixel_count_sam", "sum"),
        )
        .sort_values("boundary_bin")
        .reset_index(drop=True)
    )
    output_dir = io.PAPER_OUTPUT_ROOT / "cave-random-completion"
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_df.to_csv(output_dir / "advantage_heatmap_scene_stats_missing_only.csv", index=False)
    summary_df.to_csv(output_dir / "advantage_heatmap_summary_missing_only.csv", index=False)
    difficulty_marginal.to_csv(output_dir / "advantage_gain_by_difficulty_quartile.csv", index=False)
    boundary_marginal.to_csv(output_dir / "advantage_gain_by_boundary_quartile.csv", index=False)
    return scene_df, summary_df, difficulty_marginal, boundary_marginal, per_scene_maps


def _select_representative_advantage_scene(scene_df: pd.DataFrame) -> int:
    hard_boundary = scene_df.loc[
        scene_df["difficulty_bin"].eq(ADVANTAGE_QUARTILE_COUNT - 1)
        & scene_df["boundary_bin"].eq(ADVANTAGE_QUARTILE_COUNT - 1)
    ].copy()
    hard_boundary = hard_boundary.sort_values("delta_rmse_missing")
    positive = hard_boundary.loc[hard_boundary["delta_rmse_missing"] > 0.0].copy()
    target = positive if not positive.empty else hard_boundary
    median = float(target["delta_rmse_missing"].median())
    target["dist"] = np.abs(target["delta_rmse_missing"] - median)
    return int(target.sort_values("dist").iloc[0]["scene_id"])


def aggregate_cave_random_completion_advantage_heatmap() -> pd.DataFrame:
    _, summary_df, _, _, _ = _build_advantage_quartile_tables(missing_rate=CAVE_COMPLETION_MAIN_MISSING_RATE)
    return summary_df


def aggregate_cave_random_completion_advantage_spatial_case() -> pd.DataFrame:
    scene_df, summary_df, _, _, maps_by_scene = _build_advantage_quartile_tables(
        missing_rate=CAVE_COMPLETION_MAIN_MISSING_RATE
    )
    if ADVANTAGE_SPATIAL_CASE_SCENE_ID in maps_by_scene:
        representative_scene = int(ADVANTAGE_SPATIAL_CASE_SCENE_ID)
    else:
        representative_scene = _select_representative_advantage_scene(scene_df)
    maps = maps_by_scene[int(representative_scene)]
    top_difficulty = maps["difficulty_bins"] == (ADVANTAGE_QUARTILE_COUNT - 1)
    top_boundary = maps["boundary_bins"] == (ADVANTAGE_QUARTILE_COUNT - 1)
    overlap = top_difficulty & top_boundary

    output_dir = io.PAPER_OUTPUT_ROOT / "cave-random-completion"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "scene_id": int(maps["scene_id"]),
                "scene_name": str(maps["scene_name"]),
                "missing_rate": float(CAVE_COMPLETION_MAIN_MISSING_RATE),
                "selection_rule": "median-positive hard+boundary intersection gain",
                "q4q4_delta_rmse_missing": float(
                    summary_df.loc[
                        summary_df["difficulty_bin"].eq(ADVANTAGE_QUARTILE_COUNT - 1)
                        & summary_df["boundary_bin"].eq(ADVANTAGE_QUARTILE_COUNT - 1),
                        "delta_rmse_missing",
                    ].iloc[0]
                ),
            }
        ]
    ).to_csv(output_dir / "advantage_spatial_case_selection.csv", index=False)

    rows = [
        {
            "scene_id": int(maps["scene_id"]),
            "scene_name": str(maps["scene_name"]),
            "panel": "original",
            "image": cave_completion_pseudo_rgb(maps["original"]),
            "panel_type": "rgb",
            "top_difficulty_mask": top_difficulty,
            "top_boundary_mask": top_boundary,
            "overlap_mask": overlap,
        },
        {
            "scene_id": int(maps["scene_id"]),
            "scene_name": str(maps["scene_name"]),
            "panel": "difficulty",
            "image": np.asarray(maps["difficulty_score"], dtype=float),
            "panel_type": "difficulty",
            "top_difficulty_mask": top_difficulty,
            "top_boundary_mask": top_boundary,
            "overlap_mask": overlap,
        },
        {
            "scene_id": int(maps["scene_id"]),
            "scene_name": str(maps["scene_name"]),
            "panel": "boundary",
            "image": np.asarray(maps["boundary_score"], dtype=float),
            "panel_type": "boundary",
            "top_difficulty_mask": top_difficulty,
            "top_boundary_mask": top_boundary,
            "overlap_mask": overlap,
        },
        {
            "scene_id": int(maps["scene_id"]),
            "scene_name": str(maps["scene_name"]),
            "panel": "gain",
            "image": np.asarray(maps["delta_rmse_missing"], dtype=float),
            "panel_type": "gain",
            "top_difficulty_mask": top_difficulty,
            "top_boundary_mask": top_boundary,
            "overlap_mask": overlap,
        },
    ]
    return pd.DataFrame(rows)


def aggregate_mechanism_closure_main_figure() -> pd.DataFrame:
    try:
        frame = io.load_output_csv("cave-random-completion", "mechanism_closure_main_figure_data.csv").copy()
    except FileNotFoundError:
        from experiment.process.helpers.mechanism_closure import build_mechanism_closure_tables

        _, _, frame = build_mechanism_closure_tables()
    frame["x"] = io.maybe_numeric(frame["x"]).astype(float) + 1.0
    frame["mean"] = io.maybe_numeric(frame["mean"]).astype(float)
    frame["std"] = io.maybe_numeric(frame["std"]).astype(float)
    frame["band_lower"] = io.maybe_numeric(frame["band_lower"]).astype(float)
    frame["band_upper"] = io.maybe_numeric(frame["band_upper"]).astype(float)
    frame["panel"] = frame["panel"].astype(str)
    frame["annotation"] = ""
    return frame.loc[:, ["panel", "method", "x", "mean", "std", "band_lower", "band_upper", "annotation"]].copy()


def aggregate_cave_parameter_mechanism_main_figure() -> pd.DataFrame:
    try:
        frame = io.load_output_csv("cave-random-completion", "cave_parameter_mechanism_main_figure_data.csv").copy()
    except FileNotFoundError:
        frame, _, _, _ = build_cave_parameter_mechanism_outputs()
    frame["p_max"] = io.maybe_numeric(frame["p_max"]).astype(float)
    if "mean" in frame.columns:
        frame["mean"] = io.maybe_numeric(frame["mean"]).astype(float)
    if "std" in frame.columns:
        frame["std"] = io.maybe_numeric(frame["std"]).astype(float)
    if "band_lower" in frame.columns:
        frame["band_lower"] = io.maybe_numeric(frame["band_lower"]).astype(float)
    if "band_upper" in frame.columns:
        frame["band_upper"] = io.maybe_numeric(frame["band_upper"]).astype(float)
    if "degree" in frame.columns:
        frame["degree"] = io.maybe_numeric(frame["degree"]).astype(float)
    if "value" in frame.columns:
        frame["value"] = io.maybe_numeric(frame["value"]).astype(float)
    return frame


def aggregate_real_hsi_robustness_overview() -> pd.DataFrame:
    try:
        frame = io.load_output_csv("real-hsi-robustness", "real_hsi_robustness_overview_figure_data.csv").copy()
    except FileNotFoundError:
        runs, _, _ = load_real_hsi_main_runs()
        summary = build_real_hsi_summary(runs)
        frame = build_real_hsi_overview_figure_data(summary)
    frame["x"] = io.maybe_numeric(frame["x"]).astype(float)
    frame["mean"] = io.maybe_numeric(frame["mean"]).astype(float)
    frame["std"] = io.maybe_numeric(frame["std"]).astype(float)
    frame["band_lower"] = io.maybe_numeric(frame["band_lower"]).astype(float)
    frame["band_upper"] = io.maybe_numeric(frame["band_upper"]).astype(float)
    return frame


def _geometry_runs() -> pd.DataFrame:
    runs = io.load_runs("geometry-visualization").copy()
    runs["method_name"] = _series_str(runs, "ovr.method", "method._name")
    runs["nonlinear"] = _series_str(runs, "ovr.filter.nonlinear", "filter.nonlinear")
    runs["alpha"] = _series_num(runs, "ovr.filter.alpha", "filter.alpha").astype(float)
    runs["p_max"] = _series_num(runs, "ovr.method.p_max", "method.p_max").astype(int)
    return runs


def _select_geometry_row(frame: pd.DataFrame, *, alpha: float, p_max: int) -> pd.Series:
    panel = frame.loc[
        frame["method_name"].eq("ntdpl")
        & frame["nonlinear"].eq("poly3")
        & np.isclose(frame["alpha"], alpha, atol=1e-12)
        & frame["p_max"].eq(int(p_max))
    ].copy()
    if panel.empty:
        raise RuntimeError(f"Missing geometry-visualization state for alpha={alpha}, p_max={p_max}.")
    return panel.sort_values("run_dir").iloc[0]


def _geometry_schedule(*, alpha: float = GEOMETRY_REFERENCE_ALPHA) -> dict[int, dict[str, Any]]:
    runs = _geometry_runs()
    schedule: dict[int, dict[str, Any]] = {}
    for order in GEOMETRY_ORDER_VALUES:
        row = _select_geometry_row(runs, alpha=alpha, p_max=order)
        state = io.load_state(row["state_path"])
        beta = np.asarray(state["beta"], dtype=float).reshape(-1)
        schedule[int(order)] = {"row": row, "state": state, "beta": beta}
    return schedule


def _geometry_s_grid(schedule: dict[int, dict[str, Any]]) -> tuple[np.ndarray, float]:
    ref_state = schedule[max(schedule)]["state"]
    latent = io.reconstruct_tucker(ref_state)
    q_low, q_high = np.quantile(latent, [0.05, 0.95])
    s_limit = max(abs(float(q_low)), abs(float(q_high)))
    s_limit = max(1.5, float(np.ceil(s_limit * 2.0) / 2.0))
    return np.linspace(-s_limit, s_limit, 241), s_limit


def _poly_second_derivative(beta: np.ndarray, s_values: np.ndarray) -> np.ndarray:
    coeffs = np.asarray(beta, dtype=float).reshape(-1)
    if coeffs.size <= 2:
        return np.zeros_like(s_values, dtype=float)
    second = np.zeros_like(s_values, dtype=float)
    for degree, coefficient in enumerate(coeffs[2:], start=2):
        second = second + degree * (degree - 1) * float(coefficient) * np.power(s_values, degree - 2)
    return second


def _geometry_patch(schedule: dict[int, dict[str, Any]], *, s_limit: float, grid_size: int = 61) -> dict[str, Any]:
    ref_state = schedule[max(schedule)]["state"]
    core = np.asarray(ref_state["core"], dtype=float)
    flat_core = np.abs(core).ravel()
    top_indices = np.argsort(flat_core)[-2:]
    basis_1 = np.zeros_like(core)
    basis_2 = np.zeros_like(core)
    basis_1[np.unravel_index(top_indices[-1], core.shape)] = 1.0
    basis_2[np.unravel_index(top_indices[-2], core.shape)] = 1.0

    base = io.reconstruct_tucker(ref_state)
    z1 = io.reconstruct_tucker({"core": basis_1, "factors": ref_state["factors"]})
    z2 = io.reconstruct_tucker({"core": basis_2, "factors": ref_state["factors"]})

    influence = np.sqrt(z1.reshape(-1) ** 2 + z2.reshape(-1) ** 2)
    top_count = min(512, influence.size)
    top_idx = np.argsort(influence)[-top_count:]
    weights = influence[top_idx]
    weights = weights / np.maximum(np.sum(weights), 1e-12)

    base_vec = base.reshape(-1)[top_idx]
    z1_vec = z1.reshape(-1)[top_idx]
    z2_vec = z2.reshape(-1)[top_idx]
    direction_scale = float(np.percentile(np.sqrt(z1_vec**2 + z2_vec**2), 90))
    coord_limit = 0.70 * s_limit / max(direction_scale, 1e-12)
    coord_limit = max(coord_limit, 0.35)
    coords = np.linspace(-coord_limit, coord_limit, grid_size)
    u_grid, v_grid = np.meshgrid(coords, coords)
    latent_patch = (
        base_vec[:, None, None]
        + z1_vec[:, None, None] * u_grid[None, :, :]
        + z2_vec[:, None, None] * v_grid[None, :, :]
    )
    return {
        "weights": weights,
        "latent_patch": latent_patch,
        "u_grid": u_grid,
        "v_grid": v_grid,
        "coords": coords,
    }


def _geometry_response_map(patch: dict[str, Any], beta: np.ndarray) -> np.ndarray:
    transformed = io.apply_polynomial(np.asarray(patch["latent_patch"], dtype=float), np.asarray(beta, dtype=float))
    return np.tensordot(np.asarray(patch["weights"], dtype=float), transformed, axes=(0, 0))


def aggregate_geometry_link_evolution() -> pd.DataFrame:
    schedule = _geometry_schedule(alpha=GEOMETRY_REFERENCE_ALPHA)
    s_grid, s_limit = _geometry_s_grid(schedule)
    patch = _geometry_patch(schedule, s_limit=s_limit)

    baseline_beta = np.asarray(schedule[1]["beta"], dtype=float)
    baseline_curve = io.apply_polynomial(s_grid, baseline_beta)
    baseline_response = _geometry_response_map(patch, baseline_beta)
    rows: list[dict[str, Any]] = []

    for s_value in s_grid:
        rows.append(
            {
                "table": "identity",
                "order": 0,
                "label": r"$y=s$",
                "s": float(s_value),
                "value": float(s_value),
            }
        )

    for order in GEOMETRY_ORDER_VALUES:
        beta = np.asarray(schedule[order]["beta"], dtype=float)
        curve = io.apply_polynomial(s_grid, beta)
        delta = curve - baseline_curve
        response = _geometry_response_map(patch, beta)
        response_delta = response - baseline_response
        mean_abs_delta = float(np.mean(np.abs(delta)))
        mean_abs_response_delta = float(np.mean(np.abs(response_delta)))

        for s_value, value, delta_value in zip(s_grid, curve, delta, strict=False):
            rows.append(
                {
                    "table": "curve",
                    "order": int(order),
                    "label": f"p={order}",
                    "s": float(s_value),
                    "value": float(value),
                }
            )
            if order >= 2:
                rows.append(
                    {
                        "table": "delta",
                        "order": int(order),
                        "label": f"p={order}",
                        "s": float(s_value),
                        "value": float(delta_value),
                    }
                )

        for metric, metric_value in (
            ("mean_abs_link_deviation", mean_abs_delta),
            ("mean_abs_response_deviation", mean_abs_response_delta),
        ):
            rows.append(
                {
                    "table": "metric",
                    "metric": metric,
                    "order": int(order),
                    "label": f"p={order}",
                    "value": float(metric_value),
                    "alpha": float(GEOMETRY_REFERENCE_ALPHA),
                }
            )

    return pd.DataFrame(rows)


def aggregate_geometry_response_maps() -> pd.DataFrame:
    schedule = _geometry_schedule(alpha=GEOMETRY_REFERENCE_ALPHA)
    s_grid, s_limit = _geometry_s_grid(schedule)
    patch = _geometry_patch(schedule, s_limit=s_limit)
    baseline = _geometry_response_map(patch, np.asarray(schedule[1]["beta"], dtype=float))
    u_grid = np.asarray(patch["u_grid"], dtype=float)
    v_grid = np.asarray(patch["v_grid"], dtype=float)
    rows: list[dict[str, Any]] = []
    for order in GEOMETRY_ORDER_VALUES:
        response = _geometry_response_map(patch, np.asarray(schedule[order]["beta"], dtype=float))
        deviation = response - baseline
        for row_idx in range(response.shape[0]):
            for col_idx in range(response.shape[1]):
                rows.append(
                    {
                        "order": int(order),
                        "u": float(u_grid[row_idx, col_idx]),
                        "v": float(v_grid[row_idx, col_idx]),
                        "response": float(response[row_idx, col_idx]),
                        "deviation": float(deviation[row_idx, col_idx]),
                    }
                )
    return pd.DataFrame(rows)


AGGREGATORS = {
    "linear_paired_gap": aggregate_linear_paired_gap,
    "nonlinear_alpha_grid": aggregate_nonlinear_alpha_grid,
    "nonlinear_pmax_grid": aggregate_nonlinear_pmax_grid,
    "nonlinear_step_grid": aggregate_nonlinear_step_grid,
    "cave_representation_scene_improvement": aggregate_cave_representation_scene_improvement,
    "cave_representation_image_panels": aggregate_cave_representation_image_panels,
    "cave_representation_spectra": aggregate_cave_representation_spectra,
    "cave_random_completion_scene_improvement": aggregate_cave_random_completion_scene_improvement,
    "cave_random_completion_scene_rate_consistency": aggregate_cave_random_completion_scene_rate_consistency,
    "cave_random_completion_visual_grid": aggregate_cave_random_completion_visual_grid,
    "cave_random_completion_advantage_heatmap": aggregate_cave_random_completion_advantage_heatmap,
    "cave_random_completion_advantage_spatial_case": aggregate_cave_random_completion_advantage_spatial_case,
    "mechanism_closure_main_figure": aggregate_mechanism_closure_main_figure,
    "cave_parameter_mechanism_main_figure": aggregate_cave_parameter_mechanism_main_figure,
    "real_hsi_robustness_overview": aggregate_real_hsi_robustness_overview,
    "geometry_link_evolution": aggregate_geometry_link_evolution,
    "geometry_response_maps": aggregate_geometry_response_maps,
}


def build(aggregate_key: str) -> pd.DataFrame:
    return AGGREGATORS[aggregate_key]()
