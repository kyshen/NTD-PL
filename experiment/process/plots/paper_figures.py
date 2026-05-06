from __future__ import annotations

from pathlib import Path

from viz.pipeline import build_figures

from ..registry import register_postprocessor


EXPERIMENT_FIGURES: dict[str, tuple[str, ...]] = {
    "linear-consistency": ("linear_consistency_paired_gap",),
    "nonlinear-approx": (
        "nonlinear_pmax_grid",
        "nonlinear_alpha_grid",
        "nonlinear_step_grid",
    ),
    "geometry-visualization": (
        "geometry_link_evolution",
        "geometry_response_maps",
    ),
    "cave-representation": (
        "cave_reconstruction_scene_gain",
        "cave_reconstruction_visual_grid",
        "cave_reconstruction_spectra",
    ),
    "cave-random-completion": (
        "cave_completion_scene_gain",
        "cave_completion_visual_grid",
        "cave_completion_advantage_heatmap",
        "cave_completion_advantage_spatial_case",
        "mechanism_closure_main_figure",
        "cave_parameter_mechanism",
        "cave_completion_scene_gain_sorted_debug",
    ),
    "real-hsi-robustness": (
        "real_hsi_robustness_overview",
        "gain_cr_curves",
    ),
}


def _render_experiment_figures(exp_name: str) -> None:
    figure_ids = EXPERIMENT_FIGURES.get(exp_name, ())
    if not figure_ids:
        return
    for figure_id, paths in build_figures(figure_ids=figure_ids):
        print(f"[figure] {figure_id}")
        for path in paths:
            print(f"  -> {Path(path)}")


@register_postprocessor(exp_name="linear-consistency", order=100)
def render_linear_consistency_figures() -> None:
    _render_experiment_figures("linear-consistency")


@register_postprocessor(exp_name="nonlinear-approx", order=100)
def render_nonlinear_figures() -> None:
    _render_experiment_figures("nonlinear-approx")


@register_postprocessor(exp_name="geometry-visualization", order=100)
def render_geometry_figures() -> None:
    _render_experiment_figures("geometry-visualization")


@register_postprocessor(exp_name="cave-representation", order=100)
def render_cave_representation_figures() -> None:
    _render_experiment_figures("cave-representation")


@register_postprocessor(exp_name="cave-random-completion", order=100)
def render_cave_completion_figures() -> None:
    _render_experiment_figures("cave-random-completion")


@register_postprocessor(exp_name="real-hsi-robustness", order=100)
def render_real_hsi_robustness_figures() -> None:
    _render_experiment_figures("real-hsi-robustness")
