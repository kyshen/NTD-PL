from __future__ import annotations

import pandas as pd

from ...config import get_env
from ..helpers.cave_random_completion_advantage import (
    build_boundary_stats,
    build_contrast_summary,
    build_difficulty_stats,
    build_heatmap_stats,
    build_scene_advantage_maps,
    summary_notes,
)
from ..registry import register_postprocessor
from ...utils.paper import write_csv_artifact, write_text_artifact


def _pm(mean: float, std: float, digits: int) -> str:
    return f"{mean:.{digits}f} +- {std:.{digits}f}"


def _summary_display(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for row in frame.to_dict("records"):
        rows.append(
            {
                "bin": str(row["bin_label"]),
                "Tucker_RMSE": _pm(float(row["Tucker_RMSE_mean"]), float(row["Tucker_RMSE_std"]), 5),
                "PolyCal_RMSE": _pm(float(row["PolyCal_RMSE_mean"]), float(row["PolyCal_RMSE_std"]), 5),
                "NTDPL_RMSE": _pm(float(row["NTDPL_RMSE_mean"]), float(row["NTDPL_RMSE_std"]), 5),
                "gain_polycal_rmse": _pm(float(row["gain_polycal_rmse_mean"]), float(row["gain_polycal_rmse_std"]), 5),
                "gain_ntdpl_rmse": _pm(float(row["gain_ntdpl_rmse_mean"]), float(row["gain_ntdpl_rmse_std"]), 5),
                "extra_gain_rmse": _pm(float(row["extra_gain_rmse_mean"]), float(row["extra_gain_rmse_std"]), 5),
                "Tucker_SAM": _pm(float(row["Tucker_SAM_mean"]), float(row["Tucker_SAM_std"]), 4),
                "PolyCal_SAM": _pm(float(row["PolyCal_SAM_mean"]), float(row["PolyCal_SAM_std"]), 4),
                "NTDPL_SAM": _pm(float(row["NTDPL_SAM_mean"]), float(row["NTDPL_SAM_std"]), 4),
            }
        )
    return pd.DataFrame(rows)


@register_postprocessor(exp_name="cave-random-completion", order=60)
def advantage_tables() -> None:
    maps_by_scene, scene_summary = build_scene_advantage_maps(missing_rate=0.5, save_npz=True)
    diff_scene, diff_summary = build_difficulty_stats(maps_by_scene)
    bound_scene, bound_summary = build_boundary_stats(maps_by_scene)
    heat_scene, heat_summary = build_heatmap_stats(maps_by_scene)
    contrast = build_contrast_summary(diff_summary, bound_summary, diff_scene, bound_scene)
    env = get_env("cave-random-completion")

    write_csv_artifact(env, scene_summary, "advantage_scene_summary.csv")
    write_csv_artifact(env, diff_scene, "advantage_difficulty_scene_stats.csv")
    write_csv_artifact(env, diff_summary, "advantage_difficulty_summary.csv")
    write_csv_artifact(env, _summary_display(diff_summary), "advantage_difficulty_summary_display.csv")
    write_csv_artifact(env, bound_scene, "advantage_boundary_scene_stats.csv")
    write_csv_artifact(env, bound_summary, "advantage_boundary_summary.csv")
    write_csv_artifact(env, _summary_display(bound_summary), "advantage_boundary_summary_display.csv")
    write_csv_artifact(env, heat_scene, "advantage_heatmap_scene_stats.csv")
    write_csv_artifact(env, heat_summary, "advantage_heatmap_summary.csv")
    write_csv_artifact(env, contrast, "advantage_contrast_summary.csv")
    write_text_artifact(env, summary_notes(diff_summary, bound_summary, heat_summary, contrast), "advantage_notes.md")
