from __future__ import annotations

from ..registry import register_postprocessor
from ...utils.paper import write_csv_artifact, write_text_artifact
from ..nonlinear_approx import METHOD_ORDER, POLY_GROUP, load_nonlinear_group, select_ntdpl_for_nonlinear_approx
from .common import latex_for_nonlinear_method_grid, nonlinear_method_grid_table


@register_postprocessor(exp_name="nonlinear-approx", order=20)
def main_results_table() -> None:
    tbl, env = load_nonlinear_group(POLY_GROUP)
    summary, alpha_levels = nonlinear_method_grid_table(
        tbl,
        env=env,
        nonlinears=list(POLY_GROUP.nonlinears),
        method_order=METHOD_ORDER,
        ntdpl_selector=select_ntdpl_for_nonlinear_approx,
        digits=4,
    )

    destination, _ = write_csv_artifact(env, summary, artifact_name=POLY_GROUP.results_artifact)

    tex_path, latex_path = write_text_artifact(
        env,
        latex_for_nonlinear_method_grid(
            summary,
            env=env,
            nonlinears=list(POLY_GROUP.nonlinears),
            alpha_levels=alpha_levels,
            method_order=METHOD_ORDER,
        ),
        artifact_name=POLY_GROUP.table_artifact,
    )

    print(summary.to_string(index=False))
    print(f"\nSaved: {destination}")
    print(f"Saved: {tex_path}")
    print(f"Synced: {latex_path}")
