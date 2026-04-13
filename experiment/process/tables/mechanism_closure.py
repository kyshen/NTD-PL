from __future__ import annotations

from ..helpers.mechanism_closure import (
    build_mechanism_closure_tables,
    mechanism_closure_main_table_latex,
    mechanism_closure_main_table_numeric_csv,
)
from ..registry import register_postprocessor
from ...config import get_env
from ...utils.paper import write_csv_artifact, write_text_artifact


@register_postprocessor(exp_name="cave-random-completion", order=45)
def mechanism_closure_tables() -> None:
    env = get_env("cave-random-completion")
    numeric, display, figure_data = build_mechanism_closure_tables()

    write_csv_artifact(env, mechanism_closure_main_table_numeric_csv(numeric), "mechanism_closure_main_table.csv")
    write_csv_artifact(
        env,
        display.loc[
            :,
            ["setting", "method", "params", "rmse_metric", "rmse", "sam_metric", "sam"],
        ],
        "mechanism_closure_main_table_display.csv",
    )
    write_text_artifact(env, mechanism_closure_main_table_latex(display), "mechanism_closure_main_table.tex")
    write_csv_artifact(env, figure_data, "mechanism_closure_main_figure_data.csv")
