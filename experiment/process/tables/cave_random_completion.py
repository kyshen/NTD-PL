from __future__ import annotations

from ..helpers.cave_random_completion import (
    anomaly_notes,
    build_full_table,
    build_main_summary,
    build_paper_table,
    build_scene_gain_table,
    build_scene_rate_consistency_table,
    build_scene_mean_table,
    build_significance_summary,
    latex_full_table,
    latex_table,
    load_main_runs,
    significance_summary_latex,
)
from ..registry import register_postprocessor
from ...utils.paper import write_csv_artifact, write_text_artifact


@register_postprocessor(order=10)
def main_table() -> None:
    frame, env = load_main_runs()
    scene_mean = build_scene_mean_table(frame)
    summary = build_main_summary(scene_mean)
    paper_table = build_paper_table(summary)
    full_table = build_full_table(summary)
    scene_gain = build_scene_gain_table(scene_mean)
    scene_rate_consistency = build_scene_rate_consistency_table(scene_gain)

    csv_path, _ = write_csv_artifact(env, paper_table, "random_completion_main_table.csv")
    tex_path, _ = write_text_artifact(env, latex_table(summary), "random_completion_main_table.tex")
    full_csv_path, _ = write_csv_artifact(env, full_table, "random_completion_full_table.csv")
    full_tex_path, _ = write_text_artifact(env, latex_full_table(summary), "random_completion_full_table.tex")
    scene_csv_path, _ = write_csv_artifact(env, scene_mean, "random_completion_scene_mean.csv")
    gain_csv_path, _ = write_csv_artifact(env, scene_gain, "random_completion_scene_gains.csv")
    consistency_csv_path, _ = write_csv_artifact(
        env,
        scene_rate_consistency,
        "random_completion_scene_rate_consistency.csv",
    )
    note_path, _ = write_text_artifact(env, anomaly_notes(scene_gain), "random_completion_notes.md")

    print(f"Saved: {csv_path}")
    print(f"Saved: {tex_path}")
    print(f"Saved: {full_csv_path}")
    print(f"Saved: {full_tex_path}")
    print(f"Saved: {scene_csv_path}")
    print(f"Saved: {gain_csv_path}")
    print(f"Saved: {consistency_csv_path}")
    print(f"Saved: {note_path}")


@register_postprocessor(order=15)
def significance_summary() -> None:
    frame, env = load_main_runs()
    scene_mean = build_scene_mean_table(frame)
    summary = build_significance_summary(scene_mean)

    csv_path, _ = write_csv_artifact(env, summary, "random_completion_significance_summary.csv")
    tex_path, _ = write_text_artifact(env, significance_summary_latex(summary), "random_completion_significance_summary.tex")

    print(summary.to_string(index=False))
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {tex_path}")
