from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd

from ...config import get_env
from ...utils.io import load_state_mat
from .cave_random_completion import _cave_dataset_kwargs_from_row
from .cave_random_completion_polycal import (
    MAIN_MISSING_RATE,
    MAIN_POLYCAL_DEGREE,
    METHOD_LABELS,
    POLYCAL_LAMBDA,
    load_scene_original,
    load_target_runs,
)
from .cave_random_completion import _resolve_state_path


DIFFICULTY_BIN_COUNT = 4
BOUNDARY_BIN_COUNT = 4
METHOD_ORDER = ["tucker", f"tucker_polycal_p{MAIN_POLYCAL_DEGREE}", "ntdpl"]


@dataclass(frozen=True)
class SceneAdvantageMaps:
    scene_id: int
    scene_name: str
    original: np.ndarray
    recon_tucker: np.ndarray
    recon_polycal: np.ndarray
    recon_ntdpl: np.ndarray
    pixel_rmse_tucker: np.ndarray
    pixel_rmse_polycal: np.ndarray
    pixel_rmse_ntdpl: np.ndarray
    pixel_sam_tucker: np.ndarray
    pixel_sam_polycal: np.ndarray
    pixel_sam_ntdpl: np.ndarray
    difficulty_score: np.ndarray
    boundary_score: np.ndarray
    difficulty_bins: np.ndarray
    boundary_bins: np.ndarray
    gain_polycal_rmse: np.ndarray
    gain_ntdpl_rmse: np.ndarray
    extra_gain_rmse: np.ndarray
    gain_polycal_sam: np.ndarray
    gain_ntdpl_sam: np.ndarray
    extra_gain_sam: np.ndarray


def _poly_eval(x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float64)
    for coeff in coeffs[::-1]:
        out = out * x + float(coeff)
    return out.astype(np.float32)


def _per_pixel_rmse(original: np.ndarray, reconstruction: np.ndarray) -> np.ndarray:
    diff = np.asarray(original, dtype=np.float32) - np.asarray(reconstruction, dtype=np.float32)
    return np.sqrt(np.mean(diff * diff, axis=-1))


def _per_pixel_sam(original: np.ndarray, reconstruction: np.ndarray) -> np.ndarray:
    x = np.asarray(original, dtype=np.float64)
    y = np.asarray(reconstruction, dtype=np.float64)
    x_norm = np.linalg.norm(x, axis=-1)
    y_norm = np.linalg.norm(y, axis=-1)
    denom = np.maximum(x_norm * y_norm, 1e-12)
    cosine = np.sum(x * y, axis=-1) / denom
    cosine = np.clip(cosine, -1.0, 1.0)
    angles = np.degrees(np.arccos(cosine))
    invalid = (x_norm <= 1e-12) | (y_norm <= 1e-12)
    angles[invalid] = 0.0
    return angles.astype(np.float32)


def _boundary_score(cube: np.ndarray) -> np.ndarray:
    x = np.asarray(cube, dtype=np.float32)
    gx, gy = np.gradient(x, axis=(0, 1))
    grad_mag = np.sqrt(gx * gx + gy * gy)
    return np.mean(grad_mag, axis=-1)


def _rank_bin_map(score_map: np.ndarray, n_bins: int) -> np.ndarray:
    flat = np.asarray(score_map, dtype=np.float64).reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    bins = np.empty_like(order, dtype=np.int32)
    n = flat.size
    for idx, flat_index in enumerate(order):
        bin_id = min((idx * n_bins) // max(n, 1), n_bins - 1)
        bins[flat_index] = bin_id
    return bins.reshape(score_map.shape)


def _bin_label(prefix: str, bin_id: int, n_bins: int) -> str:
    return f"{prefix}{bin_id + 1}"


def _artifact_dir() -> Path:
    env = get_env("cave-random-completion")
    path = env.artifacts_dir / "advantage_maps"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _polycal_coefficients(frame: pd.DataFrame, env: object) -> pd.DataFrame:
    coeff_path = env.artifacts_dir / "polycal_run_coefficients.csv"
    if not coeff_path.exists():
        raise FileNotFoundError(
            "polycal_run_coefficients.csv not found. Run `python -m experiment postprocess cave-random-completion` first."
        )
    coeff = pd.read_csv(coeff_path)
    coeff = coeff.loc[coeff["degree"].astype(int) == MAIN_POLYCAL_DEGREE].copy()
    coeff["missing_rate"] = coeff["missing_rate"].astype(float)
    coeff["scene_id"] = coeff["scene_id"].astype(int)
    coeff["mask_seed"] = coeff["mask_seed"].astype(int)
    return coeff


def build_scene_advantage_maps(
    *,
    missing_rate: float = MAIN_MISSING_RATE,
    save_npz: bool = True,
) -> tuple[dict[int, SceneAdvantageMaps], pd.DataFrame]:
    frame, env = load_target_runs()
    frame = frame.loc[np.isclose(frame["missing_rate"], float(missing_rate), atol=1e-12)].copy()
    coeff = _polycal_coefficients(frame, env)
    scenes = sorted(frame["scene_id"].unique().tolist())
    out: dict[int, SceneAdvantageMaps] = {}
    scene_rows: list[dict[str, Any]] = []
    out_dir = _artifact_dir()

    for scene_id in scenes:
        panel = frame.loc[frame["scene_id"].eq(scene_id)].copy()
        scene_name, original = load_scene_original(int(scene_id), **_cave_dataset_kwargs_from_row(panel.iloc[0]))
        available_pairs = (
            panel.groupby(["mask_seed", "method_name"], as_index=False)
            .size()
            .pivot(index="mask_seed", columns="method_name", values="size")
            .fillna(0)
        )
        valid_seeds = [
            int(seed)
            for seed, row in available_pairs.iterrows()
            if row.get("tucker", 0) > 0 and row.get("ntdpl", 0) > 0
        ]
        if not valid_seeds:
            continue

        tucker_recons: list[np.ndarray] = []
        polycal_recons: list[np.ndarray] = []
        ntdpl_recons: list[np.ndarray] = []
        for seed in sorted(valid_seeds):
            row_t = panel.loc[panel["mask_seed"].eq(seed) & panel["method_name"].eq("tucker")].iloc[0]
            row_n = panel.loc[panel["mask_seed"].eq(seed) & panel["method_name"].eq("ntdpl")].iloc[0]
            try:
                state_t = load_state_mat(_resolve_state_path(row_t["state_path"]))
                state_n = load_state_mat(_resolve_state_path(row_n["state_path"]))
            except Exception as exc:
                warnings.warn(
                    f"Skipping scene {scene_id} seed {seed} in advantage analysis due to unreadable state: {exc}",
                    RuntimeWarning,
                )
                continue
            recon_t = np.asarray(state_t["reconstruction"], dtype=np.float32)
            recon_n = np.asarray(state_n["reconstruction"], dtype=np.float32)
            coeff_row = coeff.loc[
                coeff["scene_id"].eq(int(scene_id))
                & coeff["mask_seed"].eq(int(seed))
                & np.isclose(coeff["missing_rate"], float(missing_rate), atol=1e-12)
            ]
            if coeff_row.empty:
                warnings.warn(
                    f"Skipping scene {scene_id} seed {seed} in advantage analysis due to missing polycal coefficients.",
                    RuntimeWarning,
                )
                continue
            coeff_row = coeff_row.iloc[0]
            coef = coeff_row[[f"a_{k}" for k in range(MAIN_POLYCAL_DEGREE + 1)]].to_numpy(dtype=float)
            recon_p = _poly_eval(recon_t, coef)
            tucker_recons.append(recon_t)
            polycal_recons.append(recon_p)
            ntdpl_recons.append(recon_n)

        if not tucker_recons or not polycal_recons or not ntdpl_recons:
            continue
        recon_tucker = np.mean(np.stack(tucker_recons, axis=0), axis=0).astype(np.float32)
        recon_polycal = np.mean(np.stack(polycal_recons, axis=0), axis=0).astype(np.float32)
        recon_ntdpl = np.mean(np.stack(ntdpl_recons, axis=0), axis=0).astype(np.float32)

        pixel_rmse_tucker = _per_pixel_rmse(original, recon_tucker)
        pixel_rmse_polycal = _per_pixel_rmse(original, recon_polycal)
        pixel_rmse_ntdpl = _per_pixel_rmse(original, recon_ntdpl)
        pixel_sam_tucker = _per_pixel_sam(original, recon_tucker)
        pixel_sam_polycal = _per_pixel_sam(original, recon_polycal)
        pixel_sam_ntdpl = _per_pixel_sam(original, recon_ntdpl)
        difficulty_score = pixel_rmse_tucker.astype(np.float32)
        boundary_score = _boundary_score(original).astype(np.float32)
        difficulty_bins = _rank_bin_map(difficulty_score, DIFFICULTY_BIN_COUNT)
        boundary_bins = _rank_bin_map(boundary_score, BOUNDARY_BIN_COUNT)

        maps = SceneAdvantageMaps(
            scene_id=int(scene_id),
            scene_name=scene_name,
            original=original,
            recon_tucker=recon_tucker,
            recon_polycal=recon_polycal,
            recon_ntdpl=recon_ntdpl,
            pixel_rmse_tucker=pixel_rmse_tucker,
            pixel_rmse_polycal=pixel_rmse_polycal,
            pixel_rmse_ntdpl=pixel_rmse_ntdpl,
            pixel_sam_tucker=pixel_sam_tucker,
            pixel_sam_polycal=pixel_sam_polycal,
            pixel_sam_ntdpl=pixel_sam_ntdpl,
            difficulty_score=difficulty_score,
            boundary_score=boundary_score,
            difficulty_bins=difficulty_bins,
            boundary_bins=boundary_bins,
            gain_polycal_rmse=(pixel_rmse_tucker - pixel_rmse_polycal).astype(np.float32),
            gain_ntdpl_rmse=(pixel_rmse_tucker - pixel_rmse_ntdpl).astype(np.float32),
            extra_gain_rmse=(pixel_rmse_polycal - pixel_rmse_ntdpl).astype(np.float32),
            gain_polycal_sam=(pixel_sam_tucker - pixel_sam_polycal).astype(np.float32),
            gain_ntdpl_sam=(pixel_sam_tucker - pixel_sam_ntdpl).astype(np.float32),
            extra_gain_sam=(pixel_sam_polycal - pixel_sam_ntdpl).astype(np.float32),
        )
        out[int(scene_id)] = maps
        scene_rows.append(
            {
                "scene_id": int(scene_id),
                "scene_name": scene_name,
                "missing_rate": float(missing_rate),
                "rmse_mean_tucker": float(np.mean(pixel_rmse_tucker)),
                "rmse_mean_polycal": float(np.mean(pixel_rmse_polycal)),
                "rmse_mean_ntdpl": float(np.mean(pixel_rmse_ntdpl)),
                "extra_gain_rmse_mean": float(np.mean(maps.extra_gain_rmse)),
                "extra_gain_rmse_hardest": float(np.mean(maps.extra_gain_rmse[difficulty_bins == DIFFICULTY_BIN_COUNT - 1])),
                "extra_gain_rmse_boundary_high": float(np.mean(maps.extra_gain_rmse[boundary_bins == BOUNDARY_BIN_COUNT - 1])),
            }
        )

        if save_npz:
            np.savez_compressed(
                out_dir / f"scene_{int(scene_id):02d}_mr_{int(round(missing_rate * 10)):02d}.npz",
                scene_id=int(scene_id),
                scene_name=scene_name,
                original=original,
                recon_tucker=recon_tucker,
                recon_polycal=recon_polycal,
                recon_ntdpl=recon_ntdpl,
                pixel_rmse_tucker=pixel_rmse_tucker,
                pixel_rmse_polycal=pixel_rmse_polycal,
                pixel_rmse_ntdpl=pixel_rmse_ntdpl,
                pixel_sam_tucker=pixel_sam_tucker,
                pixel_sam_polycal=pixel_sam_polycal,
                pixel_sam_ntdpl=pixel_sam_ntdpl,
                difficulty_score=difficulty_score,
                boundary_score=boundary_score,
                difficulty_bins=difficulty_bins,
                boundary_bins=boundary_bins,
                gain_polycal_rmse=maps.gain_polycal_rmse,
                gain_ntdpl_rmse=maps.gain_ntdpl_rmse,
                extra_gain_rmse=maps.extra_gain_rmse,
                gain_polycal_sam=maps.gain_polycal_sam,
                gain_ntdpl_sam=maps.gain_ntdpl_sam,
                extra_gain_sam=maps.extra_gain_sam,
            )

    scene_df = pd.DataFrame(scene_rows)
    if scene_df.empty:
        raise RuntimeError("No valid Tucker/NTD-PL paired seeds available for advantage analysis.")
    return out, scene_df.sort_values("scene_id").reset_index(drop=True)


def _group_stats(maps_by_scene: dict[int, SceneAdvantageMaps], *, group_type: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if group_type == "difficulty":
        n_bins = DIFFICULTY_BIN_COUNT
    elif group_type == "boundary":
        n_bins = BOUNDARY_BIN_COUNT
    else:
        raise ValueError(f"Unsupported group_type: {group_type}")

    for scene_id, maps in maps_by_scene.items():
        bin_map = maps.difficulty_bins if group_type == "difficulty" else maps.boundary_bins
        for bin_id in range(n_bins):
            mask = bin_map == bin_id
            rows.append(
                {
                    "scene_id": scene_id,
                    "scene_name": maps.scene_name,
                    "bin_id": int(bin_id),
                    "bin_label": _bin_label("Q", bin_id, n_bins),
                    "group_type": group_type,
                    "pixel_count": int(np.sum(mask)),
                    "Tucker_RMSE": float(np.mean(maps.pixel_rmse_tucker[mask])),
                    "PolyCal_RMSE": float(np.mean(maps.pixel_rmse_polycal[mask])),
                    "NTDPL_RMSE": float(np.mean(maps.pixel_rmse_ntdpl[mask])),
                    "Tucker_SAM": float(np.mean(maps.pixel_sam_tucker[mask])),
                    "PolyCal_SAM": float(np.mean(maps.pixel_sam_polycal[mask])),
                    "NTDPL_SAM": float(np.mean(maps.pixel_sam_ntdpl[mask])),
                    "gain_polycal_rmse": float(np.mean(maps.gain_polycal_rmse[mask])),
                    "gain_ntdpl_rmse": float(np.mean(maps.gain_ntdpl_rmse[mask])),
                    "extra_gain_rmse": float(np.mean(maps.extra_gain_rmse[mask])),
                    "gain_polycal_sam": float(np.mean(maps.gain_polycal_sam[mask])),
                    "gain_ntdpl_sam": float(np.mean(maps.gain_ntdpl_sam[mask])),
                    "extra_gain_sam": float(np.mean(maps.extra_gain_sam[mask])),
                }
            )
    return pd.DataFrame(rows)


def build_difficulty_stats(maps_by_scene: dict[int, SceneAdvantageMaps]) -> tuple[pd.DataFrame, pd.DataFrame]:
    scene_stats = _group_stats(maps_by_scene, group_type="difficulty")
    summary = (
        scene_stats.groupby(["bin_id", "bin_label"], as_index=False)
        .agg(
            Tucker_RMSE_mean=("Tucker_RMSE", "mean"),
            Tucker_RMSE_std=("Tucker_RMSE", "std"),
            PolyCal_RMSE_mean=("PolyCal_RMSE", "mean"),
            PolyCal_RMSE_std=("PolyCal_RMSE", "std"),
            NTDPL_RMSE_mean=("NTDPL_RMSE", "mean"),
            NTDPL_RMSE_std=("NTDPL_RMSE", "std"),
            Tucker_SAM_mean=("Tucker_SAM", "mean"),
            Tucker_SAM_std=("Tucker_SAM", "std"),
            PolyCal_SAM_mean=("PolyCal_SAM", "mean"),
            PolyCal_SAM_std=("PolyCal_SAM", "std"),
            NTDPL_SAM_mean=("NTDPL_SAM", "mean"),
            NTDPL_SAM_std=("NTDPL_SAM", "std"),
            gain_polycal_rmse_mean=("gain_polycal_rmse", "mean"),
            gain_polycal_rmse_std=("gain_polycal_rmse", "std"),
            gain_ntdpl_rmse_mean=("gain_ntdpl_rmse", "mean"),
            gain_ntdpl_rmse_std=("gain_ntdpl_rmse", "std"),
            extra_gain_rmse_mean=("extra_gain_rmse", "mean"),
            extra_gain_rmse_std=("extra_gain_rmse", "std"),
            gain_polycal_sam_mean=("gain_polycal_sam", "mean"),
            gain_polycal_sam_std=("gain_polycal_sam", "std"),
            gain_ntdpl_sam_mean=("gain_ntdpl_sam", "mean"),
            gain_ntdpl_sam_std=("gain_ntdpl_sam", "std"),
            extra_gain_sam_mean=("extra_gain_sam", "mean"),
            extra_gain_sam_std=("extra_gain_sam", "std"),
        )
        .sort_values("bin_id")
        .reset_index(drop=True)
    )
    return scene_stats, summary


def build_boundary_stats(maps_by_scene: dict[int, SceneAdvantageMaps]) -> tuple[pd.DataFrame, pd.DataFrame]:
    scene_stats = _group_stats(maps_by_scene, group_type="boundary")
    summary = (
        scene_stats.groupby(["bin_id", "bin_label"], as_index=False)
        .agg(
            Tucker_RMSE_mean=("Tucker_RMSE", "mean"),
            Tucker_RMSE_std=("Tucker_RMSE", "std"),
            PolyCal_RMSE_mean=("PolyCal_RMSE", "mean"),
            PolyCal_RMSE_std=("PolyCal_RMSE", "std"),
            NTDPL_RMSE_mean=("NTDPL_RMSE", "mean"),
            NTDPL_RMSE_std=("NTDPL_RMSE", "std"),
            Tucker_SAM_mean=("Tucker_SAM", "mean"),
            Tucker_SAM_std=("Tucker_SAM", "std"),
            PolyCal_SAM_mean=("PolyCal_SAM", "mean"),
            PolyCal_SAM_std=("PolyCal_SAM", "std"),
            NTDPL_SAM_mean=("NTDPL_SAM", "mean"),
            NTDPL_SAM_std=("NTDPL_SAM", "std"),
            gain_polycal_rmse_mean=("gain_polycal_rmse", "mean"),
            gain_polycal_rmse_std=("gain_polycal_rmse", "std"),
            gain_ntdpl_rmse_mean=("gain_ntdpl_rmse", "mean"),
            gain_ntdpl_rmse_std=("gain_ntdpl_rmse", "std"),
            extra_gain_rmse_mean=("extra_gain_rmse", "mean"),
            extra_gain_rmse_std=("extra_gain_rmse", "std"),
            gain_polycal_sam_mean=("gain_polycal_sam", "mean"),
            gain_polycal_sam_std=("gain_polycal_sam", "std"),
            gain_ntdpl_sam_mean=("gain_ntdpl_sam", "mean"),
            gain_ntdpl_sam_std=("gain_ntdpl_sam", "std"),
            extra_gain_sam_mean=("extra_gain_sam", "mean"),
            extra_gain_sam_std=("extra_gain_sam", "std"),
        )
        .sort_values("bin_id")
        .reset_index(drop=True)
    )
    return scene_stats, summary


def build_heatmap_stats(maps_by_scene: dict[int, SceneAdvantageMaps]) -> tuple[pd.DataFrame, pd.DataFrame]:
    scene_rows: list[dict[str, Any]] = []
    for scene_id, maps in maps_by_scene.items():
        for difficulty_bin in range(DIFFICULTY_BIN_COUNT):
            for boundary_bin in range(BOUNDARY_BIN_COUNT):
                mask = (maps.difficulty_bins == difficulty_bin) & (maps.boundary_bins == boundary_bin)
                scene_rows.append(
                    {
                        "scene_id": scene_id,
                        "scene_name": maps.scene_name,
                        "difficulty_bin": int(difficulty_bin),
                        "boundary_bin": int(boundary_bin),
                        "difficulty_label": _bin_label("Q", difficulty_bin, DIFFICULTY_BIN_COUNT),
                        "boundary_label": _bin_label("Q", boundary_bin, BOUNDARY_BIN_COUNT),
                        "pixel_count": int(np.sum(mask)),
                        "gain_ntdpl_rmse": float(np.mean(maps.gain_ntdpl_rmse[mask])),
                        "extra_gain_rmse": float(np.mean(maps.extra_gain_rmse[mask])),
                        "gain_ntdpl_sam": float(np.mean(maps.gain_ntdpl_sam[mask])),
                        "extra_gain_sam": float(np.mean(maps.extra_gain_sam[mask])),
                    }
                )
    scene_df = pd.DataFrame(scene_rows)
    summary = (
        scene_df.groupby(["difficulty_bin", "boundary_bin"], as_index=False)
        .agg(
            difficulty_label=("difficulty_label", "first"),
            boundary_label=("boundary_label", "first"),
            gain_ntdpl_rmse_mean=("gain_ntdpl_rmse", "mean"),
            gain_ntdpl_rmse_std=("gain_ntdpl_rmse", "std"),
            extra_gain_rmse_mean=("extra_gain_rmse", "mean"),
            extra_gain_rmse_std=("extra_gain_rmse", "std"),
            gain_ntdpl_sam_mean=("gain_ntdpl_sam", "mean"),
            gain_ntdpl_sam_std=("gain_ntdpl_sam", "std"),
            extra_gain_sam_mean=("extra_gain_sam", "mean"),
            extra_gain_sam_std=("extra_gain_sam", "std"),
        )
        .sort_values(["boundary_bin", "difficulty_bin"])
        .reset_index(drop=True)
    )
    return scene_df, summary


def build_contrast_summary(difficulty_summary: pd.DataFrame, boundary_summary: pd.DataFrame, difficulty_scene: pd.DataFrame, boundary_scene: pd.DataFrame) -> pd.DataFrame:
    easy = difficulty_summary.loc[difficulty_summary["bin_id"] == 0].iloc[0]
    hard = difficulty_summary.loc[difficulty_summary["bin_id"] == DIFFICULTY_BIN_COUNT - 1].iloc[0]
    low = boundary_summary.loc[boundary_summary["bin_id"] == 0].iloc[0]
    high = boundary_summary.loc[boundary_summary["bin_id"] == BOUNDARY_BIN_COUNT - 1].iloc[0]

    hard_vs_easy_scene = (
        difficulty_scene.pivot_table(index="scene_id", columns="bin_id", values="gain_ntdpl_rmse", aggfunc="mean")
        .reset_index(drop=True)
    )
    hard_vs_easy_extra_scene = (
        difficulty_scene.pivot_table(index="scene_id", columns="bin_id", values="extra_gain_rmse", aggfunc="mean")
        .reset_index(drop=True)
    )
    boundary_scene_cmp = (
        boundary_scene.pivot_table(index="scene_id", columns="bin_id", values="gain_ntdpl_rmse", aggfunc="mean")
        .reset_index(drop=True)
    )
    boundary_extra_scene_cmp = (
        boundary_scene.pivot_table(index="scene_id", columns="bin_id", values="extra_gain_rmse", aggfunc="mean")
        .reset_index(drop=True)
    )
    rows = [
        {
            "comparison": "hardest_vs_easiest_gain_ntdpl_rmse",
            "left_group": "Q4",
            "right_group": "Q1",
            "left_mean": float(hard["gain_ntdpl_rmse_mean"]),
            "right_mean": float(easy["gain_ntdpl_rmse_mean"]),
            "delta": float(hard["gain_ntdpl_rmse_mean"] - easy["gain_ntdpl_rmse_mean"]),
            "scene_consistency": int((hard_vs_easy_scene[3] > hard_vs_easy_scene[0]).sum()),
            "n_scenes": int(len(hard_vs_easy_scene)),
        },
        {
            "comparison": "high_boundary_vs_low_boundary_gain_ntdpl_rmse",
            "left_group": "Q4",
            "right_group": "Q1",
            "left_mean": float(high["gain_ntdpl_rmse_mean"]),
            "right_mean": float(low["gain_ntdpl_rmse_mean"]),
            "delta": float(high["gain_ntdpl_rmse_mean"] - low["gain_ntdpl_rmse_mean"]),
            "scene_consistency": int((boundary_scene_cmp[3] > boundary_scene_cmp[0]).sum()),
            "n_scenes": int(len(boundary_scene_cmp)),
        },
        {
            "comparison": "hardest_vs_easiest_extra_gain_rmse",
            "left_group": "Q4",
            "right_group": "Q1",
            "left_mean": float(hard["extra_gain_rmse_mean"]),
            "right_mean": float(easy["extra_gain_rmse_mean"]),
            "delta": float(hard["extra_gain_rmse_mean"] - easy["extra_gain_rmse_mean"]),
            "scene_consistency": int((hard_vs_easy_extra_scene[3] > hard_vs_easy_extra_scene[0]).sum()),
            "n_scenes": int(len(hard_vs_easy_extra_scene)),
        },
        {
            "comparison": "high_boundary_vs_low_boundary_extra_gain_rmse",
            "left_group": "Q4",
            "right_group": "Q1",
            "left_mean": float(high["extra_gain_rmse_mean"]),
            "right_mean": float(low["extra_gain_rmse_mean"]),
            "delta": float(high["extra_gain_rmse_mean"] - low["extra_gain_rmse_mean"]),
            "scene_consistency": int((boundary_extra_scene_cmp[3] > boundary_extra_scene_cmp[0]).sum()),
            "n_scenes": int(len(boundary_extra_scene_cmp)),
        },
    ]
    return pd.DataFrame(rows)


def select_representative_scene(maps_by_scene: dict[int, SceneAdvantageMaps], heatmap_scene: pd.DataFrame) -> int:
    target = heatmap_scene.loc[
        (heatmap_scene["difficulty_bin"] == DIFFICULTY_BIN_COUNT - 1)
        & (heatmap_scene["boundary_bin"] == BOUNDARY_BIN_COUNT - 1)
    ].copy()
    target = target.sort_values("extra_gain_rmse")
    positive = target.loc[target["extra_gain_rmse"] > 0.0].copy()
    panel = positive if not positive.empty else target
    median_value = float(panel["extra_gain_rmse"].median())
    panel["delta"] = np.abs(panel["extra_gain_rmse"] - median_value)
    return int(panel.sort_values("delta").iloc[0]["scene_id"])


def summary_notes(
    difficulty_summary: pd.DataFrame,
    boundary_summary: pd.DataFrame,
    heatmap_summary: pd.DataFrame,
    contrast_summary: pd.DataFrame,
) -> str:
    hard = difficulty_summary.loc[difficulty_summary["bin_id"] == DIFFICULTY_BIN_COUNT - 1].iloc[0]
    easy = difficulty_summary.loc[difficulty_summary["bin_id"] == 0].iloc[0]
    high = boundary_summary.loc[boundary_summary["bin_id"] == BOUNDARY_BIN_COUNT - 1].iloc[0]
    low = boundary_summary.loc[boundary_summary["bin_id"] == 0].iloc[0]
    hard_high = heatmap_summary.loc[
        (heatmap_summary["difficulty_bin"] == DIFFICULTY_BIN_COUNT - 1)
        & (heatmap_summary["boundary_bin"] == BOUNDARY_BIN_COUNT - 1)
    ].iloc[0]

    lines = [
        "# Advantage Region Notes",
        "",
        f"- Hardest difficulty quartile extra_gain(RMSE) = {float(hard['extra_gain_rmse_mean']):.5f}; "
        f"easiest quartile = {float(easy['extra_gain_rmse_mean']):.5f}.",
        f"- Highest boundary quartile extra_gain(RMSE) = {float(high['extra_gain_rmse_mean']):.5f}; "
        f"lowest quartile = {float(low['extra_gain_rmse_mean']):.5f}.",
        f"- Hard+boundary intersection extra_gain(RMSE) = {float(hard_high['extra_gain_rmse_mean']):.5f}.",
        "",
    ]
    for row in contrast_summary.to_dict("records"):
        lines.append(
            f"- {row['comparison']}: delta = {float(row['delta']):.5f}, "
            f"scene consistency = {int(row['scene_consistency'])}/{int(row['n_scenes'])}."
        )
    return "\n".join(lines) + "\n"
