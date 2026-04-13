from __future__ import annotations

from ..helpers.cave_representation_original_space import (
    scene_improvement_overview_plot,
    spectral_curve_plot,
    visual_compare_plot,
)
from ..registry import register_postprocessor


@register_postprocessor(exp_name="cave-representation", order=10)
def scene_gain_figure() -> None:
    scene_improvement_overview_plot()


@register_postprocessor(exp_name="cave-representation", order=20)
def visual_grid_figure() -> None:
    visual_compare_plot()


@register_postprocessor(exp_name="cave-representation", order=30)
def spectral_figure() -> None:
    spectral_curve_plot()
