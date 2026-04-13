from __future__ import annotations

from ..helpers.cave_parameter_mechanism import (
    build_cave_parameter_mechanism_outputs,
    parameter_mechanism_summary_latex,
)
from ..registry import register_postprocessor
from ...config import get_env
from ...utils.paper import write_csv_artifact, write_text_artifact


@register_postprocessor(exp_name="cave-random-completion", order=46)
def cave_parameter_mechanism_tables() -> None:
    env = get_env("cave-random-completion")
    figure_data, summary, contribution_summary, run_metrics = build_cave_parameter_mechanism_outputs()

    write_csv_artifact(env, figure_data, "cave_parameter_mechanism_main_figure_data.csv")
    write_csv_artifact(env, summary, "cave_parameter_mechanism_summary.csv")
    write_csv_artifact(env, contribution_summary, "cave_parameter_mechanism_contribution_summary.csv")
    write_csv_artifact(env, run_metrics, "cave_parameter_mechanism_run_metrics.csv")
    write_text_artifact(env, parameter_mechanism_summary_latex(summary), "parameter_mechanism_summary_table.tex")
