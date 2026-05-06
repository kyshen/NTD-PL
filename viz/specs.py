from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence


FigureRole = Literal["single_column", "double_column", "appendix_wide", "compact"]
FigureSection = Literal["main", "appendix"]
FigureFamily = Literal[
    "paired_boxplot",
    "boxplot_summary",
    "line_grid",
    "step_line_grid",
    "sorted_gain_grid",
    "heatmap",
    "line_plus_heatmap",
    "image_comparison_grid",
    "spectra_panel",
    "geometry_evolution",
    "geometry_response_maps",
    "gain_cr_curves",
    "heatmap_pair",
    "completion_visual_grid",
    "spatial_case_grid",
]


@dataclass(frozen=True)
class ExportTarget:
    section: FigureSection
    stem: str
    role: FigureRole
    legacy_paths: tuple[str, ...] = ()
    formats: tuple[str, ...] = ("pdf",)


@dataclass(frozen=True)
class BaseSpec:
    figure_id: str
    family: FigureFamily
    title: str
    subtitle: str | None
    export: ExportTarget
    latex_path: str
    latex_width: str
    height: float | None = None
    shared_legend: bool = False
    recommended_pairing: str | None = None
    appendix_only: bool = False
    note: str | None = None


@dataclass(frozen=True)
class PanelSpec:
    key: str
    title: str
    xlabel: str
    ylabel: str
    x_order: tuple[Any, ...] | None = None
    x_tick_labels: tuple[str, ...] | None = None
    annotation: str | None = None


@dataclass(frozen=True)
class PairedBoxplotSpec(BaseSpec):
    variants: tuple[str, ...] = ("strict", "affine")


@dataclass(frozen=True)
class BoxplotGridSpec(BaseSpec):
    panels: tuple[PanelSpec, ...] = ()
    x_label: str = "Degree"
    y_label: str = "Value"


@dataclass(frozen=True)
class LineGridSpec(BaseSpec):
    panels: tuple[PanelSpec, ...] = ()
    band_columns: tuple[str, str] = ("mean", "std")
    method_order: tuple[str, ...] = ()
    plot_kind: Literal["line", "bar"] = "line"
    emphasize_diff: bool = False


@dataclass(frozen=True)
class SortedGainGridSpec(BaseSpec):
    panels: tuple[PanelSpec, ...] = ()
    gain_column: str = "gain"
    panel_column: str = "panel"
    sort_descending: bool = True
    zero_baseline: bool = True


@dataclass(frozen=True)
class HeatmapSpec(BaseSpec):
    x_label: str = ""
    y_label: str = ""
    value_label: str = ""
    cmap: str = "viridis"
    center_zero: bool = False
    annotate_cells: bool = False


@dataclass(frozen=True)
class HeatmapPairSpec(BaseSpec):
    x_label: str = ""
    y_label: str = ""
    left_value_col: str = "value_left"
    right_value_col: str = "value_right"
    left_title: str = ""
    right_title: str = ""
    left_value_label: str = ""
    right_value_label: str = ""
    cmap: str = "RdBu_r"
    center_zero: bool = True
    x_col: str = "missing_rate"
    y_col: str = "scene_label"
    annotate_cells: bool = False
    show_colorbar: bool = True
    square_axes: bool = False
    square_cells: bool = False
    shared_color_scale: bool = True


@dataclass(frozen=True)
class CompletionVisualGridSpec(BaseSpec):
    panel_order: tuple[str, ...] = ()
    panel_titles: tuple[str, ...] = ()
    show_colorbars: bool = True


@dataclass(frozen=True)
class SpatialCaseGridSpec(BaseSpec):
    panel_order: tuple[str, ...] = ("original", "difficulty", "boundary", "gain")
    panel_titles: tuple[str, ...] = (
        "Original / pseudo-RGB",
        "Tucker difficulty map",
        "Boundary map",
        "Missing-only error reduction",
    )
    show_colorbars: bool = True
    show_notes: bool = True


@dataclass(frozen=True)
class LinePlusHeatmapSpec(BaseSpec):
    left_panel: PanelSpec = field(default_factory=lambda: PanelSpec("left", "", "", ""))
    right_panel: PanelSpec = field(default_factory=lambda: PanelSpec("right", "", "", ""))
    cmap: str = "viridis"
    value_label: str = ""
    show_colorbar: bool = True


@dataclass(frozen=True)
class ImageComparisonGridSpec(BaseSpec):
    scene_order: tuple[int, ...] = ()
    panel_order: tuple[str, ...] = ()
    diff_panel: str | None = None
    colorbar_label: str | None = None


@dataclass(frozen=True)
class SpectraPanelSpec(BaseSpec):
    scene_order: tuple[int, ...] = ()
    columns_per_scene: int = 2


@dataclass(frozen=True)
class GeometryEvolutionSpec(BaseSpec):
    alpha: float = 0.3
    order_values: tuple[int, ...] = ()


@dataclass(frozen=True)
class GeometryResponseMapsSpec(BaseSpec):
    alpha: float = 0.3
    order_values: tuple[int, ...] = ()


@dataclass(frozen=True)
class FigureJob:
    spec: BaseSpec
    aggregate_key: str

    @property
    def figure_id(self) -> str:
        return self.spec.figure_id
