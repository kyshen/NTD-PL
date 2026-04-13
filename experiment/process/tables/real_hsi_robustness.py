from __future__ import annotations

from ...utils.paper import write_csv_artifact, write_text_artifact
from ..helpers.real_hsi_robustness import (
    appendix_table_latex,
    build_appendix_table,
    build_consistency_summary,
    build_main_table_display,
    build_main_table,
    build_overview_figure_data,
    build_summary,
    load_main_runs,
    main_table_latex,
    protocol_table_latex,
)
from ..registry import register_postprocessor


@register_postprocessor(order=10)
def main_tables() -> None:
    frame, metadata, env = load_main_runs()
    summary = build_summary(frame)
    main_table_numeric = build_main_table(summary)
    main_table = build_main_table_display(main_table_numeric)
    appendix_table = build_appendix_table(summary)
    consistency = build_consistency_summary(main_table_numeric)
    overview_figure_data = build_overview_figure_data(summary)

    csv_main_path, _ = write_csv_artifact(env, main_table, "real_hsi_robustness_main_table.csv")
    csv_main_numeric_path, _ = write_csv_artifact(env, main_table_numeric, "real_hsi_robustness_main_table_numeric.csv")
    tex_main_path, _ = write_text_artifact(env, main_table_latex(main_table_numeric), "real_hsi_robustness_main_table.tex")
    csv_appendix_path, _ = write_csv_artifact(env, appendix_table, "real_hsi_robustness_appendix_table.csv")
    tex_appendix_path, _ = write_text_artifact(
        env,
        appendix_table_latex(summary),
        "real_hsi_robustness_appendix_table.tex",
    )
    tex_appendix_alias_path, _ = write_text_artifact(
        env,
        appendix_table_latex(summary),
        "real_hsi_robustness_appendix.tex",
    )
    protocol_csv_path, _ = write_csv_artifact(env, metadata, "real_hsi_robustness_protocol.csv")
    protocol_tex_path, _ = write_text_artifact(
        env,
        protocol_table_latex(metadata),
        "real_hsi_robustness_protocol.tex",
    )
    summary_csv_path, _ = write_csv_artifact(env, summary, "real_hsi_robustness_summary.csv")
    consistency_csv_path, _ = write_csv_artifact(env, consistency, "real_hsi_robustness_consistency.csv")
    runs_csv_path, _ = write_csv_artifact(env, frame, "real_hsi_robustness_runs.csv")
    overview_csv_path, _ = write_csv_artifact(env, overview_figure_data, "real_hsi_robustness_overview_figure_data.csv")

    print(f"Saved: {csv_main_path}")
    print(f"Saved: {csv_main_numeric_path}")
    print(f"Saved: {tex_main_path}")
    print(f"Saved: {csv_appendix_path}")
    print(f"Saved: {tex_appendix_path}")
    print(f"Saved: {tex_appendix_alias_path}")
    print(f"Saved: {protocol_csv_path}")
    print(f"Saved: {protocol_tex_path}")
    print(f"Saved: {summary_csv_path}")
    print(f"Saved: {consistency_csv_path}")
    print(f"Saved: {runs_csv_path}")
    print(f"Saved: {overview_csv_path}")
