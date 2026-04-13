from __future__ import annotations

from ..helpers.cave_random_completion_polycal import (
    build_degree_ablation_table,
    build_main_table,
    build_pairwise_scene_gains,
    build_scene_mean,
    build_summary,
    collect_polycal_results,
    latex_degree_table,
    latex_main_table,
    load_target_runs,
    merge_mechanism_runs,
    mechanism_notes,
)
from ..registry import register_postprocessor
from ...utils.paper import write_csv_artifact, write_text_artifact


@register_postprocessor(exp_name="cave-random-completion", order=40)
def polycal_tables() -> None:
    frame, env = load_target_runs()
    polycal_metrics, polycal_coeffs = collect_polycal_results(frame)
    mechanism_runs = merge_mechanism_runs(frame, polycal_metrics)
    scene_mean = build_scene_mean(mechanism_runs)
    summary = build_summary(scene_mean)
    pairwise = build_pairwise_scene_gains(scene_mean)

    write_csv_artifact(env, polycal_metrics, "polycal_run_metrics.csv")
    write_csv_artifact(env, polycal_coeffs, "polycal_run_coefficients.csv")
    write_csv_artifact(env, scene_mean, "polycal_scene_mean.csv")
    write_csv_artifact(env, summary, "polycal_summary.csv")
    write_csv_artifact(env, build_main_table(summary), "polycal_main_table.csv")
    write_text_artifact(env, latex_main_table(summary), "polycal_main_table.tex")
    write_csv_artifact(env, build_degree_ablation_table(summary), "polycal_degree_ablation.csv")
    write_text_artifact(env, latex_degree_table(summary), "polycal_degree_ablation.tex")
    write_csv_artifact(env, pairwise, "polycal_pairwise_scene_gains.csv")
    write_text_artifact(env, mechanism_notes(pairwise), "polycal_mechanism_notes.md")
