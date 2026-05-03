from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...utils.io import load_state_mat
from src.data.hsi import CAVEHSIData
from src.filters.bias import BiasFilter
from src.methods.mlpcal import ScalarMLPCalibration
from src.methods.polycal import PolynomialCalibration
from src.metrics import val_NMSE_dB, val_RMSE, val_SAM
from src.types import Tensor

from .cave_random_completion import (
    _cave_dataset_kwargs_from_row,
    _resolve_state_path,
    load_main_runs,
    observed_fraction_map,
    pseudo_rgb,
    rmse_map,
)


POLYCAL_DEGREES = (2, 3, 4)
POLYCAL_LAMBDA = 1e-6
MLPCAL_HIDDEN_UNITS = 16
MLPCAL_LAMBDA = 1e-5
MLPCAL_LR = 1e-3
MLPCAL_MAX_ITER = 1500
MLPCAL_BATCH_SIZE = 8192
MLPCAL_MAX_TRAIN_SAMPLES = 200_000
MLPCAL_TARGET_MISSING_RATES = (0.5,)
TARGET_MISSING_RATES = (0.3, 0.5)
MAIN_MISSING_RATE = 0.5
MAIN_POLYCAL_DEGREE = 4
MAIN_MLPCAL_NAME = f"tucker_mlpcal_h{MLPCAL_HIDDEN_UNITS}"

METHOD_LABELS = {
    "tucker": "Tucker",
    "tucker_polycal_p2": "Tucker + PolyCal(P=2)",
    "tucker_polycal_p3": "Tucker + PolyCal(P=3)",
    "tucker_polycal_p4": "Tucker + PolyCal(P=4)",
    MAIN_MLPCAL_NAME: f"Tucker + MLPCal(H={MLPCAL_HIDDEN_UNITS})",
    "ntdpl": "NTD-PL",
}


@dataclass(frozen=True)
class PolyCalRepresentativePayload:
    scene_id: int
    scene_name: str
    mask_seed: int
    missing_rate: float
    original: np.ndarray
    observed_mask: np.ndarray
    recon_tucker: np.ndarray
    recon_polycal: np.ndarray
    recon_ntdpl: np.ndarray


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


@lru_cache(maxsize=None)
def load_scene_original(
    scene_id: int,
    *,
    path: str = "data/CAVE",
    target_shape: tuple[int, int] = (512, 512),
    crop_shape: tuple[int, int] | None = None,
) -> tuple[str, np.ndarray]:
    dataset = CAVEHSIData(path=path, id=int(scene_id), target_shape=target_shape, crop_shape=crop_shape)
    BiasFilter(seed=0, normalize_method="max", snr_db=None, bias=None)(dataset)
    scene_name = str(getattr(dataset, "scene_name", f"scene-{scene_id}"))
    original = np.asarray(dataset.get(split="eval").dense, dtype=np.float32)
    return scene_name, original


def _completion_metrics(original: np.ndarray, reconstruction: np.ndarray, observed_mask: np.ndarray) -> dict[str, float]:
    missing_mask = ~np.asarray(observed_mask, dtype=bool)
    original_tensor = Tensor(shape=original.shape, dense=original)
    missing_tensor = Tensor(shape=original.shape, dense=original, mask=missing_mask)
    recon_tensor = Tensor(shape=reconstruction.shape, dense=reconstruction)
    return {
        "RMSE_all": val_RMSE(original_tensor, recon_tensor),
        "RMSE_missing": val_RMSE(missing_tensor, recon_tensor),
        "SAM_all": val_SAM(original_tensor, recon_tensor),
        "SAM_missing": val_SAM(missing_tensor, recon_tensor),
        "NMSE_dB_all": val_NMSE_dB(original_tensor, recon_tensor),
    }


def load_target_runs() -> tuple[pd.DataFrame, object]:
    frame, env = load_main_runs()
    selected = frame.loc[frame["missing_rate"].isin(TARGET_MISSING_RATES)].copy()
    selected["scene_name"] = selected.apply(
        lambda row: load_scene_original(int(row["scene_id"]), **_cave_dataset_kwargs_from_row(row))[0],
        axis=1,
    )
    return selected, env


def collect_polycal_results(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tucker_rows = frame.loc[frame["method_name"] == "tucker"].copy()
    metric_rows: list[dict[str, Any]] = []
    coeff_rows: list[dict[str, Any]] = []

    for row in tucker_rows.to_dict("records"):
        scene_id = int(row["scene_id"])
        mask_seed = int(row["mask_seed"])
        missing_rate = float(row["missing_rate"])
        state = load_state_mat(_resolve_state_path(row["state_path"]))
        recon_tucker = np.asarray(_jsonish(state["reconstruction"]), dtype=np.float32)
        observed_mask = np.asarray(_jsonish(state["observed_mask"]), dtype=bool)
        scene_name, original = load_scene_original(scene_id, **_cave_dataset_kwargs_from_row(row))

        for degree in POLYCAL_DEGREES:
            model = PolynomialCalibration(degree=degree, lambda_reg=POLYCAL_LAMBDA).fit(
                recon_tucker,
                original,
                observed_mask,
            )
            recon_polycal = model.apply(recon_tucker)
            metrics = _completion_metrics(original, recon_polycal, observed_mask)
            diagnostics = model.diagnostics
            assert diagnostics is not None
            method_name = f"tucker_polycal_p{degree}"
            metric_row = {
                "scene_id": scene_id,
                "scene_name": scene_name,
                "mask_seed": mask_seed,
                "missing_rate": missing_rate,
                "method_name": method_name,
                "degree": degree,
                "lambda_reg": float(model.lambda_reg),
                "polycal_fit_time_sec": float(diagnostics.fit_time_sec),
                "fit_time_sec": float(row["fit_time_sec"]) + float(diagnostics.fit_time_sec),
                **metrics,
            }
            metric_rows.append(metric_row)

            coeff_row = {
                "scene_id": scene_id,
                "scene_name": scene_name,
                "mask_seed": mask_seed,
                "missing_rate": missing_rate,
                "method_name": method_name,
                "degree": degree,
                "lambda_reg": float(model.lambda_reg),
                "observed_count": int(diagnostics.observed_count),
                "x_min": float(diagnostics.x_min),
                "x_max": float(diagnostics.x_max),
                "x_mean": float(diagnostics.x_mean),
                "design_cond_raw": float(diagnostics.design_cond_raw),
                "design_cond_scaled": float(diagnostics.design_cond_scaled),
                "polycal_fit_time_sec": float(diagnostics.fit_time_sec),
            }
            coeff = model.coefficients
            assert coeff is not None
            for idx in range(MAIN_POLYCAL_DEGREE + 1):
                coeff_row[f"a_{idx}"] = float(coeff[idx]) if idx < coeff.size else np.nan
            coeff_rows.append(coeff_row)

    metrics_frame = pd.DataFrame(metric_rows).sort_values(
        ["missing_rate", "scene_id", "mask_seed", "degree"]
    ).reset_index(drop=True)
    coeff_frame = pd.DataFrame(coeff_rows).sort_values(
        ["missing_rate", "scene_id", "mask_seed", "degree"]
    ).reset_index(drop=True)
    return metrics_frame, coeff_frame


def collect_mlpcal_results(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tucker_rows = frame.loc[
        frame["method_name"].eq("tucker")
        & frame["missing_rate"].isin(MLPCAL_TARGET_MISSING_RATES)
    ].copy()
    metric_rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []

    for row in tucker_rows.to_dict("records"):
        scene_id = int(row["scene_id"])
        mask_seed = int(row["mask_seed"])
        missing_rate = float(row["missing_rate"])
        state = load_state_mat(_resolve_state_path(row["state_path"]))
        recon_tucker = np.asarray(_jsonish(state["reconstruction"]), dtype=np.float32)
        observed_mask = np.asarray(_jsonish(state["observed_mask"]), dtype=bool)
        scene_name, original = load_scene_original(scene_id, **_cave_dataset_kwargs_from_row(row))

        model = ScalarMLPCalibration(
            hidden_units=MLPCAL_HIDDEN_UNITS,
            lambda_reg=MLPCAL_LAMBDA,
            lr=MLPCAL_LR,
            max_iter=MLPCAL_MAX_ITER,
            batch_size=MLPCAL_BATCH_SIZE,
            max_train_samples=MLPCAL_MAX_TRAIN_SAMPLES,
            random_state=mask_seed,
        ).fit(recon_tucker, original, observed_mask)
        recon_mlpcal = model.apply(recon_tucker)
        metrics = _completion_metrics(original, recon_mlpcal, observed_mask)
        diagnostics = model.diagnostics
        assert diagnostics is not None

        metric_rows.append(
            {
                "scene_id": scene_id,
                "scene_name": scene_name,
                "mask_seed": mask_seed,
                "missing_rate": missing_rate,
                "method_name": MAIN_MLPCAL_NAME,
                "hidden_units": MLPCAL_HIDDEN_UNITS,
                "lambda_reg": MLPCAL_LAMBDA,
                "mlpcal_fit_time_sec": float(diagnostics.fit_time_sec),
                "fit_time_sec": float(row["fit_time_sec"]) + float(diagnostics.fit_time_sec),
                **metrics,
            }
        )
        diag_rows.append(
            {
                "scene_id": scene_id,
                "scene_name": scene_name,
                "mask_seed": mask_seed,
                "missing_rate": missing_rate,
                "method_name": MAIN_MLPCAL_NAME,
                **diagnostics.__dict__,
            }
        )

    metrics_frame = pd.DataFrame(metric_rows).sort_values(
        ["missing_rate", "scene_id", "mask_seed"]
    ).reset_index(drop=True)
    diag_frame = pd.DataFrame(diag_rows).sort_values(
        ["missing_rate", "scene_id", "mask_seed"]
    ).reset_index(drop=True)
    return metrics_frame, diag_frame


def merge_mechanism_runs(
    frame: pd.DataFrame,
    polycal_metrics: pd.DataFrame,
    mlpcal_metrics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    base_cols = [
        "scene_id",
        "scene_name",
        "mask_seed",
        "missing_rate",
        "method_name",
        "RMSE_all",
        "RMSE_missing",
        "SAM_all",
        "SAM_missing",
        "NMSE_dB_all",
        "fit_time_sec",
        "state_path",
    ]
    existing = frame.loc[frame["method_name"].isin(["tucker", "ntdpl"]), base_cols].copy()
    polycal_with_state = polycal_metrics.copy()
    polycal_with_state["state_path"] = None
    frames = [existing, polycal_with_state[base_cols]]
    if mlpcal_metrics is not None and not mlpcal_metrics.empty:
        mlpcal_with_state = mlpcal_metrics.copy()
        mlpcal_with_state["state_path"] = None
        frames.append(mlpcal_with_state[base_cols])
    merged = pd.concat(frames, ignore_index=True)
    return merged.sort_values(["missing_rate", "scene_id", "mask_seed", "method_name"]).reset_index(drop=True)


def build_scene_mean(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = ["RMSE_all", "RMSE_missing", "SAM_all", "SAM_missing", "NMSE_dB_all", "fit_time_sec"]
    return (
        frame.groupby(["missing_rate", "method_name", "scene_id"], as_index=False)[metrics]
        .mean()
        .sort_values(["missing_rate", "method_name", "scene_id"])
        .reset_index(drop=True)
    )


def build_summary(scene_mean: pd.DataFrame) -> pd.DataFrame:
    summary = (
        scene_mean.groupby(["missing_rate", "method_name"], as_index=False)
        .agg(
            RMSE_missing_mean=("RMSE_missing", "mean"),
            RMSE_missing_std=("RMSE_missing", "std"),
            SAM_missing_mean=("SAM_missing", "mean"),
            SAM_missing_std=("SAM_missing", "std"),
            RMSE_all_mean=("RMSE_all", "mean"),
            RMSE_all_std=("RMSE_all", "std"),
            SAM_all_mean=("SAM_all", "mean"),
            SAM_all_std=("SAM_all", "std"),
            NMSE_dB_all_mean=("NMSE_dB_all", "mean"),
            NMSE_dB_all_std=("NMSE_dB_all", "std"),
            Time_mean=("fit_time_sec", "mean"),
            Time_std=("fit_time_sec", "std"),
            n_scenes=("scene_id", "nunique"),
        )
        .sort_values(["missing_rate", "method_name"])
        .reset_index(drop=True)
    )
    for col in summary.columns:
        if col.endswith("_std"):
            summary[col] = summary[col].fillna(0.0)
    return summary


def build_main_table(summary: pd.DataFrame) -> pd.DataFrame:
    methods = ["tucker", f"tucker_polycal_p{MAIN_POLYCAL_DEGREE}", MAIN_MLPCAL_NAME, "ntdpl"]
    panel = summary.loc[
        np.isclose(summary["missing_rate"], MAIN_MISSING_RATE, atol=1e-12)
        & summary["method_name"].isin(methods)
    ].copy()
    order = {method: idx for idx, method in enumerate(methods)}
    panel["order"] = panel["method_name"].map(order).fillna(99)
    panel = panel.sort_values("order").drop(columns="order")
    rows: list[dict[str, str]] = []
    for row in panel.to_dict("records"):
        rows.append(
            {
                "method": METHOD_LABELS[str(row["method_name"])],
                "RMSE*": pm_text(float(row["RMSE_missing_mean"]), float(row["RMSE_missing_std"]), 5),
                "SAM*": pm_text(float(row["SAM_missing_mean"]), float(row["SAM_missing_std"]), 4),
                "RMSE(all)": pm_text(float(row["RMSE_all_mean"]), float(row["RMSE_all_std"]), 5),
                "SAM(all)": pm_text(float(row["SAM_all_mean"]), float(row["SAM_all_std"]), 4),
                "NMSE(dB)(all)": pm_text(float(row["NMSE_dB_all_mean"]), float(row["NMSE_dB_all_std"]), 3),
                "missing_rate": f"{MAIN_MISSING_RATE:.1f}",
            }
        )
    return pd.DataFrame(rows)


def build_degree_ablation_table(summary: pd.DataFrame) -> pd.DataFrame:
    methods = [f"tucker_polycal_p{degree}" for degree in POLYCAL_DEGREES]
    panel = summary.loc[summary["method_name"].isin(methods)].copy()
    panel["degree"] = panel["method_name"].str.extract(r"p(\d+)").astype(int)
    panel = panel.sort_values(["missing_rate", "degree"])
    rows: list[dict[str, str]] = []
    for row in panel.to_dict("records"):
        rows.append(
            {
                "missing_rate": f"{float(row['missing_rate']):.1f}",
                "method": METHOD_LABELS[str(row["method_name"])],
                "RMSE*": pm_text(float(row["RMSE_missing_mean"]), float(row["RMSE_missing_std"]), 5),
                "SAM*": pm_text(float(row["SAM_missing_mean"]), float(row["SAM_missing_std"]), 4),
            }
        )
    return pd.DataFrame(rows)


def pm_text(mean: float, std: float, digits: int) -> str:
    return f"{mean:.{digits}f} +- {std:.{digits}f}"


def pm_latex(mean: float, std: float, digits: int) -> str:
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def latex_main_table(summary: pd.DataFrame) -> str:
    panel = build_main_table(summary)
    lines = [
        r"\begin{tabular}{c|c|c|c|c|c}",
        r"    \hline",
        r"    Method & RMSE* & SAM* & RMSE(all) & SAM(all) & NMSE(dB)(all) \\",
        r"    \hline",
    ]
    for row in panel.to_dict("records"):
        lines.append(
            "    "
            + " & ".join(
                [
                    str(row["method"]),
                    str(row["RMSE*"]).replace("+-", r"$\pm$"),
                    str(row["SAM*"]).replace("+-", r"$\pm$"),
                    str(row["RMSE(all)"]).replace("+-", r"$\pm$"),
                    str(row["SAM(all)"]).replace("+-", r"$\pm$"),
                    str(row["NMSE(dB)(all)"]).replace("+-", r"$\pm$"),
                ]
            )
            + r" \\"
        )
    lines.append(r"    \hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def latex_degree_table(summary: pd.DataFrame) -> str:
    panel = build_degree_ablation_table(summary)
    lines = [
        r"\begin{tabular}{c|c|c|c}",
        r"    \hline",
        r"    Missing rate & Method & RMSE* & SAM* \\",
        r"    \hline",
    ]
    last_rate: str | None = None
    for row in panel.to_dict("records"):
        current_rate = str(row["missing_rate"])
        rate_text = current_rate if current_rate != last_rate else ""
        lines.append(
            "    "
            + " & ".join(
                [
                    rate_text,
                    str(row["method"]),
                    str(row["RMSE*"]).replace("+-", r"$\pm$"),
                    str(row["SAM*"]).replace("+-", r"$\pm$"),
                ]
            )
            + r" \\"
        )
        if current_rate != last_rate and last_rate is not None:
            pass
        last_rate = current_rate
    lines.append(r"    \hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def build_pairwise_scene_gains(scene_mean: pd.DataFrame) -> pd.DataFrame:
    methods = ["tucker", f"tucker_polycal_p{MAIN_POLYCAL_DEGREE}", "ntdpl"]
    pivot = scene_mean.loc[scene_mean["method_name"].isin(methods)].pivot_table(
        index=["missing_rate", "scene_id"],
        columns="method_name",
        values=["RMSE_missing", "SAM_missing"],
        aggfunc="mean",
    )
    rows: list[dict[str, Any]] = []
    for (missing_rate, scene_id), payload in pivot.iterrows():
        row = {
            "missing_rate": float(missing_rate),
            "scene_id": int(scene_id),
        }
        t = float(payload[("RMSE_missing", "tucker")])
        p = float(payload[("RMSE_missing", f"tucker_polycal_p{MAIN_POLYCAL_DEGREE}")])
        n = float(payload[("RMSE_missing", "ntdpl")])
        ts = float(payload[("SAM_missing", "tucker")])
        ps = float(payload[("SAM_missing", f"tucker_polycal_p{MAIN_POLYCAL_DEGREE}")])
        ns = float(payload[("SAM_missing", "ntdpl")])
        row.update(
            {
                "RMSE_tucker": t,
                "RMSE_polycal": p,
                "RMSE_ntdpl": n,
                "SAM_tucker": ts,
                "SAM_polycal": ps,
                "SAM_ntdpl": ns,
                "polycal_gain_rmse": t - p,
                "joint_extra_gain_rmse": p - n,
                "polycal_gain_sam": ts - ps,
                "joint_extra_gain_sam": ps - ns,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["missing_rate", "scene_id"]).reset_index(drop=True)


def mechanism_notes(pairwise: pd.DataFrame) -> str:
    lines = ["# PolyCal Mechanism Notes", ""]
    for missing_rate in TARGET_MISSING_RATES:
        panel = pairwise.loc[np.isclose(pairwise["missing_rate"], missing_rate, atol=1e-12)].copy()
        if panel.empty:
            continue
        polycal_wins = int((panel["polycal_gain_rmse"] > 0).sum())
        ntdpl_wins = int((panel["joint_extra_gain_rmse"] > 0).sum())
        lines.append(f"## missing_rate = {missing_rate:.1f}")
        lines.append(
            f"- PolyCal(P={MAIN_POLYCAL_DEGREE}) improves RMSE* over Tucker on "
            f"{polycal_wins}/{len(panel)} scenes after seed averaging."
        )
        lines.append(
            f"- NTD-PL further improves RMSE* over PolyCal(P={MAIN_POLYCAL_DEGREE}) on "
            f"{ntdpl_wins}/{len(panel)} scenes after seed averaging."
        )
        hard_cases = panel.loc[panel["joint_extra_gain_rmse"] < 0].sort_values("joint_extra_gain_rmse")
        if hard_cases.empty:
            lines.append("- No scene shows worse RMSE* than PolyCal after switching to NTD-PL.")
        else:
            for _, row in hard_cases.head(3).iterrows():
                lines.append(
                    f"- Scene {int(row['scene_id'])}: joint_extra_gain_rmse = {float(row['joint_extra_gain_rmse']):.5f}, "
                    f"joint_extra_gain_sam = {float(row['joint_extra_gain_sam']):.4f}"
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def select_representative_run(mechanism_runs: pd.DataFrame, pairwise_scene: pd.DataFrame) -> tuple[int, int]:
    panel = pairwise_scene.loc[np.isclose(pairwise_scene["missing_rate"], MAIN_MISSING_RATE, atol=1e-12)].copy()
    positive = panel.loc[
        (panel["polycal_gain_rmse"] > 0.0)
        & (panel["joint_extra_gain_rmse"] > 0.0)
    ].copy()
    if positive.empty:
        candidate_scene = int(panel.iloc[0]["scene_id"])
    else:
        target_poly = float(positive["polycal_gain_rmse"].median())
        target_joint = float(positive["joint_extra_gain_rmse"].median())
        positive["score"] = (
            (positive["polycal_gain_rmse"] - target_poly) ** 2
            + (positive["joint_extra_gain_rmse"] - target_joint) ** 2
        )
        candidate_scene = int(positive.sort_values("score").iloc[0]["scene_id"])

    scene_runs = mechanism_runs.loc[
        np.isclose(mechanism_runs["missing_rate"], MAIN_MISSING_RATE, atol=1e-12)
        & mechanism_runs["scene_id"].eq(candidate_scene)
        & mechanism_runs["method_name"].eq(f"tucker_polycal_p{MAIN_POLYCAL_DEGREE}")
    ].copy()
    scene_mean = float(scene_runs["RMSE_missing"].mean())
    scene_runs["delta"] = np.abs(scene_runs["RMSE_missing"] - scene_mean)
    chosen_seed = int(scene_runs.sort_values("delta").iloc[0]["mask_seed"])
    return candidate_scene, chosen_seed


def load_representative_payload(mechanism_runs: pd.DataFrame, *, scene_id: int, mask_seed: int) -> PolyCalRepresentativePayload:
    row_t = mechanism_runs.loc[
        np.isclose(mechanism_runs["missing_rate"], MAIN_MISSING_RATE, atol=1e-12)
        & mechanism_runs["scene_id"].eq(scene_id)
        & mechanism_runs["mask_seed"].eq(mask_seed)
        & mechanism_runs["method_name"].eq("tucker")
    ].iloc[0]
    row_n = mechanism_runs.loc[
        np.isclose(mechanism_runs["missing_rate"], MAIN_MISSING_RATE, atol=1e-12)
        & mechanism_runs["scene_id"].eq(scene_id)
        & mechanism_runs["mask_seed"].eq(mask_seed)
        & mechanism_runs["method_name"].eq("ntdpl")
    ].iloc[0]
    scene_name, original = load_scene_original(scene_id, **_cave_dataset_kwargs_from_row(row_t))

    state_t = load_state_mat(_resolve_state_path(row_t["state_path"]))
    state_n = load_state_mat(_resolve_state_path(row_n["state_path"]))
    recon_tucker = np.asarray(_jsonish(state_t["reconstruction"]), dtype=np.float32)
    observed_mask = np.asarray(_jsonish(state_t["observed_mask"]), dtype=bool)
    recon_ntdpl = np.asarray(_jsonish(state_n["reconstruction"]), dtype=np.float32)

    model = PolynomialCalibration(degree=MAIN_POLYCAL_DEGREE, lambda_reg=POLYCAL_LAMBDA).fit(
        recon_tucker,
        original,
        observed_mask,
    )
    recon_polycal = model.apply(recon_tucker)
    return PolyCalRepresentativePayload(
        scene_id=scene_id,
        scene_name=scene_name,
        mask_seed=mask_seed,
        missing_rate=MAIN_MISSING_RATE,
        original=original,
        observed_mask=observed_mask,
        recon_tucker=recon_tucker,
        recon_polycal=recon_polycal,
        recon_ntdpl=recon_ntdpl,
    )


def select_spectral_pixels(payload: PolyCalRepresentativePayload, *, count: int = 3) -> list[dict[str, Any]]:
    err_t = rmse_map(payload.original, payload.recon_tucker)
    err_p = rmse_map(payload.original, payload.recon_polycal)
    err_n = rmse_map(payload.original, payload.recon_ntdpl)
    h, w = err_t.shape

    candidates = [
        ("Typical", np.abs(err_t - np.median(err_t))),
        ("PolyCal gain", -(err_t - err_p)),
        ("NTD-PL extra gain", -(err_p - err_n)),
    ]

    used: set[tuple[int, int]] = set()
    picks: list[dict[str, Any]] = []
    for label, score in candidates:
        flat = np.argsort(score, axis=None)
        for idx in flat:
            r, c = np.unravel_index(int(idx), (h, w))
            coord = (int(r), int(c))
            if coord in used:
                continue
            used.add(coord)
            picks.append(
                {
                    "label": label,
                    "row": int(r),
                    "col": int(c),
                    "tucker_rmse": float(err_t[r, c]),
                    "polycal_rmse": float(err_p[r, c]),
                    "ntdpl_rmse": float(err_n[r, c]),
                }
            )
            break
        if len(picks) >= count:
            break
    return picks


def spectral_curves(payload: PolyCalRepresentativePayload, pixels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    band_axis = np.arange(1, payload.original.shape[-1] + 1, dtype=int)
    curves: list[dict[str, Any]] = []
    for item in pixels:
        r = int(item["row"])
        c = int(item["col"])
        curves.append(
            {
                "label": str(item["label"]),
                "row": r,
                "col": c,
                "band": band_axis,
                "ground_truth": np.asarray(payload.original[r, c, :], dtype=np.float32),
                "tucker": np.asarray(payload.recon_tucker[r, c, :], dtype=np.float32),
                "polycal": np.asarray(payload.recon_polycal[r, c, :], dtype=np.float32),
                "ntdpl": np.asarray(payload.recon_ntdpl[r, c, :], dtype=np.float32),
            }
        )
    return curves


def representative_metadata(payload: PolyCalRepresentativePayload, pixels: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scene_id": payload.scene_id,
                "scene_name": payload.scene_name,
                "mask_seed": payload.mask_seed,
                "missing_rate": payload.missing_rate,
                **pixel,
            }
            for pixel in pixels
        ]
    )


def polycal_error_map(payload: PolyCalRepresentativePayload) -> dict[str, np.ndarray]:
    return {
        "rgb_ground_truth": pseudo_rgb(payload.original),
        "rgb_tucker": pseudo_rgb(payload.recon_tucker),
        "rgb_polycal": pseudo_rgb(payload.recon_polycal),
        "rgb_ntdpl": pseudo_rgb(payload.recon_ntdpl),
        "observed_fraction": observed_fraction_map(payload.observed_mask),
        "err_tucker": rmse_map(payload.original, payload.recon_tucker),
        "err_polycal": rmse_map(payload.original, payload.recon_polycal),
        "err_ntdpl": rmse_map(payload.original, payload.recon_ntdpl),
    }
