from __future__ import annotations

import math
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec

from .specs import (
    BaseSpec,
    BoxplotGridSpec,
    CompletionVisualGridSpec,
    GeometryEvolutionSpec,
    GeometryResponseMapsSpec,
    HeatmapSpec,
    HeatmapPairSpec,
    ImageComparisonGridSpec,
    LineGridSpec,
    LinePlusHeatmapSpec,
    PairedBoxplotSpec,
    SpatialCaseGridSpec,
    SortedGainGridSpec,
    SpectraPanelSpec,
)
from .style import PALETTE, apply_style, method_style, role_preset, style_axes, style_colorbar


def _figure_size(spec: BaseSpec, default_height: float) -> tuple[float, float]:
    preset = role_preset(spec.export.role)
    return preset.width, spec.height or default_height


def _grid_shape(panel_count: int) -> tuple[int, int]:
    if panel_count <= 1:
        return 1, 1
    if panel_count == 2:
        return 1, 2
    if panel_count <= 4:
        return 2, 2
    return math.ceil(panel_count / 3), 3


def _row_spacing(n_rows: int, *, base: float, multirow: float) -> float:
    return multirow if n_rows > 1 else base


def _order_style(order: int) -> dict[str, object]:
    style_map: dict[int, dict[str, object]] = {
        1: {"color": PALETTE.tucker, "linestyle": "--", "marker": "s", "linewidth": 1.9, "markersize": 4.2},
        2: {"color": PALETTE.neutral, "linestyle": ":", "marker": "D", "linewidth": 1.8, "markersize": 4.0},
        3: {"color": PALETTE.highlight, "linestyle": "-.", "marker": "^", "linewidth": 1.9, "markersize": 4.2},
        4: {"color": PALETTE.ntdpl, "linestyle": "-", "marker": "o", "linewidth": 2.2, "markersize": 4.6},
    }
    return style_map.get(order, {"color": PALETTE.text, "linestyle": "-", "marker": None, "linewidth": 1.6, "markersize": 0.0})


def render_paired_boxplot(data, spec: PairedBoxplotSpec):
    apply_style(spec.export.role)
    panel_order = list(dict.fromkeys(data["panel_key"].tolist()))
    fig, axes = plt.subplots(1, len(panel_order), figsize=_figure_size(spec, 2.2), sharey=True, constrained_layout=False)
    axes = np.atleast_1d(axes)
    palette = {"strict": PALETTE.neutral, "affine": PALETTE.ntdpl}
    y_values = data["gap"].to_numpy(dtype=float)
    y_span = max(float(y_values.max() - y_values.min()), 1e-6)
    pad = 0.10 * y_span

    for ax, panel_key in zip(axes, panel_order, strict=False):
        panel = data.loc[data["panel_key"] == panel_key].copy()
        variant_order = ["strict", "affine"]
        series_list = [panel.loc[panel["variant_key"] == key, "gap"].to_numpy(dtype=float) for key in variant_order]
        box = ax.boxplot(
            series_list,
            positions=np.arange(len(variant_order)),
            widths=0.58,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": PALETTE.text, "linewidth": 1.0},
            whiskerprops={"color": PALETTE.border, "linewidth": 0.8},
            capprops={"color": PALETTE.border, "linewidth": 0.8},
        )
        for patch, key in zip(box["boxes"], variant_order, strict=False):
            patch.set_facecolor(palette[key])
            patch.set_alpha(0.72)
            patch.set_edgecolor(PALETTE.border)
            patch.set_linewidth(0.8)

        rng = np.random.default_rng(20260328 + len(panel))
        for xpos, values, key in zip(np.arange(len(variant_order)), series_list, variant_order, strict=False):
            jitter = rng.uniform(-0.07, 0.07, size=len(values))
            ax.scatter(xpos + jitter, values, s=16, color=palette[key], alpha=0.85, edgecolors="none", zorder=3)
        ax.axhline(0.0, color=PALETTE.border, linestyle="--", linewidth=0.9, alpha=0.8)
        ax.set_xticks(np.arange(len(variant_order)))
        ax.set_xticklabels([r"Restricted" "\n" r"$\beta_0=0$", r"Free" "\n" r"$\beta_0$"])
        ax.set_title(str(panel["panel_title"].iloc[0]))
        ax.set_ylabel(r"RMSE gap (Tucker - NTD-PL)" if ax is axes[0] else "")
        ax.set_ylim(float(y_values.min()) - pad, float(y_values.max()) + pad)
        style_axes(ax, grid=True)

    fig.subplots_adjust(top=0.84, bottom=0.24, left=0.08, right=0.995, wspace=0.14)
    return fig


def render_boxplot_grid(data, spec: BoxplotGridSpec):
    apply_style(spec.export.role)
    panel_order = [panel.key for panel in spec.panels]
    n_rows, n_cols = _grid_shape(len(panel_order))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=_figure_size(spec, 4.8), sharey=True, constrained_layout=False)
    axes = np.atleast_1d(axes).reshape(n_rows, n_cols)
    for idx, panel_key in enumerate(panel_order):
        ax = axes.ravel()[idx]
        panel_data = data.loc[data["panel"] == panel_key].copy()
        degrees = sorted(int(value) for value in panel_data["degree"].unique().tolist())
        series = [panel_data.loc[panel_data["degree"] == degree, "beta"].to_numpy(dtype=float) for degree in degrees]
        box = ax.boxplot(
            series,
            positions=np.arange(len(degrees)),
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": PALETTE.text, "linewidth": 1.0},
            whiskerprops={"color": PALETTE.border, "linewidth": 0.8},
            capprops={"color": PALETTE.border, "linewidth": 0.8},
        )
        for degree, patch in zip(degrees, box["boxes"], strict=False):
            patch.set_facecolor(PALETTE.tucker if degree < 2 else PALETTE.ntdpl)
            patch.set_alpha(0.72)
            patch.set_edgecolor(PALETTE.border)
            patch.set_linewidth(0.8)
        ax.set_xticks(np.arange(len(degrees)))
        ax.set_xticklabels([str(degree) for degree in degrees])
        panel_meta = next((panel for panel in spec.panels if panel.key == panel_key), None)
        ax.set_title(panel_meta.title if panel_meta else panel_key, pad=2.5)
        ax.set_xlabel(spec.x_label, labelpad=1.5)
        ax.set_ylabel(spec.y_label if idx % n_cols == 0 else "")
        style_axes(ax, grid=True)
    for extra_ax in axes.ravel()[len(panel_order):]:
        extra_ax.set_axis_off()
    fig.subplots_adjust(
        top=0.90,
        bottom=0.14,
        left=0.08,
        right=0.99,
        wspace=0.18,
        hspace=_row_spacing(n_rows, base=0.24, multirow=0.38),
    )
    return fig


def render_line_grid(data, spec: LineGridSpec, *, step_mode: bool = False):
    apply_style(spec.export.role)
    bar_mode = (not step_mode) and spec.plot_kind == "bar"
    panel_order = [panel.key for panel in spec.panels] if spec.panels else list(dict.fromkeys(data["panel"].tolist()))
    n_rows, n_cols = _grid_shape(len(panel_order))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=_figure_size(spec, 5.0 if len(panel_order) > 2 else 3.0),
        sharey=not bar_mode,
        constrained_layout=False,
    )
    axes = np.atleast_1d(axes).reshape(n_rows, n_cols)
    legend_handles = []
    lower_all: list[float] = []
    upper_all: list[float] = []

    for idx, panel_key in enumerate(panel_order):
        ax = axes.ravel()[idx]
        panel_data = data.loc[data["panel"] == panel_key].copy()
        panel_meta = next((panel for panel in spec.panels if panel.key == panel_key), None)
        if bar_mode:
            methods = list(spec.method_order) if spec.method_order else list(dict.fromkeys(panel_data["method"].astype(str).tolist()))
            x_values: list[float] = []
            y_values: list[float] = []
            method_values: list[str] = []
            bar_colors: list[str] = []
            for method in methods:
                sub = panel_data.loc[panel_data["method"] == method].copy()
                if sub.empty:
                    continue
                row = sub.sort_values("x").iloc[0]
                x_values.append(float(row["x"]))
                y_values.append(float(row["mean"]))
                method_values.append(method)
                bar_colors.append(str(method_style(method)["color"]))

            if x_values:
                ax.bar(x_values, y_values, width=0.62, color=bar_colors, alpha=0.86, edgecolor=PALETTE.border, linewidth=0.7, zorder=3)
                vmin = min(y_values)
                vmax = max(y_values)
                span = max(vmax - vmin, 1e-9)
                lower_pad = 0.04 * span
                upper_pad = 0.34 * span
                ax.set_ylim(vmin - lower_pad, vmax + upper_pad)

                base_method = "Tucker" if "Tucker" in method_values else method_values[0]
                base_idx = method_values.index(base_method)
                base_value = y_values[base_idx]
                if spec.emphasize_diff:
                    for m_name, xpos, yval in zip(method_values, x_values, y_values, strict=False):
                        if m_name == base_method:
                            continue
                        delta = float(yval - base_value)
                        sign = "+" if delta >= 0 else ""
                        ax.text(
                            float(xpos),
                            float(yval) + 0.09 * span,
                            f"{sign}{delta:.4f}",
                            ha="center",
                            va="bottom",
                            fontsize=7.0,
                            color="#D62828",
                            fontweight="semibold",
                        )

            if panel_meta is not None:
                ax.set_title(panel_meta.title, pad=2.5)
                ax.set_xlabel(panel_meta.xlabel, labelpad=1.5)
                ax.set_ylabel(panel_meta.ylabel if idx % n_cols == 0 else "")
                if panel_meta.x_order is not None:
                    xticks = np.asarray(list(panel_meta.x_order), dtype=float)
                    ax.set_xticks(xticks)
                    if panel_meta.x_tick_labels is not None:
                        ax.set_xticklabels(list(panel_meta.x_tick_labels))
            else:
                ax.set_title(panel_key, pad=2.5)
            style_axes(ax, grid=True)
            ax.yaxis.grid(True, color=PALETTE.grid, alpha=0.72, linewidth=0.6)
            ax.xaxis.grid(False)
            continue

        for method in spec.method_order:
            sub = panel_data.loc[panel_data["method"] == method].copy()
            if step_mode and "kind" in sub.columns:
                sub = sub.loc[sub["kind"] == "curve"].copy()
            if sub.empty:
                continue
            style = method_style(method)
            if spec.figure_id == "nonlinear_pmax_grid" and method == "Tucker":
                style["marker"] = None
                style["markersize"] = 0.0
            if step_mode:
                style["marker"] = None
                style["markersize"] = 0.0
            x = sub["x"].to_numpy(dtype=float)
            y = sub["mean"].to_numpy(dtype=float)
            lower = sub["band_lower"].to_numpy(dtype=float)
            upper = sub["band_upper"].to_numpy(dtype=float)
            if not np.allclose(lower, upper):
                ax.fill_between(x, lower, upper, color=style["color"], alpha=0.14, linewidth=0.0)
            line, = ax.plot(
                x,
                y,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style.get("marker"),
                linewidth=style["linewidth"],
                markersize=style.get("markersize", 4.0),
                label=method,
            )
            if all(handle.get_label() != line.get_label() for handle in legend_handles):
                legend_handles.append(line)
            lower_all.extend([float(value) for value in lower if np.isfinite(value)])
            upper_all.extend([float(value) for value in upper if np.isfinite(value)])

        if step_mode:
            transitions = panel_data.loc[panel_data["kind"] == "transition"].copy()
            curve = panel_data.loc[panel_data["kind"] == "curve"].copy()
            if not transitions.empty and not curve.empty:
                curve_x = curve["x"].to_numpy(dtype=float)
                curve_y = curve["mean"].to_numpy(dtype=float)
                for item in transitions.itertuples(index=False):
                    nearest = int(np.argmin(np.abs(curve_x - float(item.x))))
                    ax.axvline(float(item.x), color=PALETTE.highlight, linestyle="--", linewidth=0.9, alpha=0.85)
                    ax.plot(curve_x[nearest], curve_y[nearest], marker="o", color=PALETTE.highlight, markersize=4.5)
                    ax.annotate(
                        rf"$p={int(item.degree)}$",
                        (curve_x[nearest], curve_y[nearest]),
                        xytext=(0, 7),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=7.1,
                        color=PALETTE.highlight,
                    )

        if panel_meta is not None:
            ax.set_title(panel_meta.title, pad=2.5)
            ax.set_xlabel(panel_meta.xlabel, labelpad=1.5)
            ax.set_ylabel(panel_meta.ylabel if idx % n_cols == 0 else "")
            if panel_meta.x_order is not None:
                ax.set_xticks(np.asarray(list(panel_meta.x_order), dtype=float))
                if panel_meta.x_tick_labels is not None:
                    ax.set_xticklabels(list(panel_meta.x_tick_labels))
                    if any(len(str(item)) > 6 for item in panel_meta.x_tick_labels):
                        ax.tick_params(axis="x", labelrotation=16)
        else:
            ax.set_title(panel_key, pad=2.5)
        annotation = panel_data["annotation"].dropna().astype(str)
        if not annotation.empty and annotation.iloc[0]:
            ax.text(0.98, 0.04, annotation.iloc[0], transform=ax.transAxes, ha="right", va="bottom", fontsize=7.4, color=PALETTE.border)
        style_axes(ax, grid=True)

    for extra_ax in axes.ravel()[len(panel_order):]:
        extra_ax.set_axis_off()
    if bar_mode:
        fig.subplots_adjust(
            top=0.90,
            bottom=0.12,
            left=0.08,
            right=0.99,
            wspace=0.22,
            hspace=_row_spacing(n_rows, base=0.26, multirow=0.40),
        )
        return fig
    if lower_all and upper_all:
        y_min = float(min(lower_all))
        y_max = float(max(upper_all))
        pad = 0.06 * max(y_max - y_min, 1e-6)
        for ax in axes.ravel()[: len(panel_order)]:
            ax.set_ylim(y_min - pad, y_max + pad)
    if spec.shared_legend and legend_handles:
        fig.legend(
            legend_handles,
            [handle.get_label() for handle in legend_handles],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncols=min(len(legend_handles), 5),
        )
        fig.subplots_adjust(
            top=0.84,
            bottom=0.12,
            left=0.08,
            right=0.99,
            wspace=0.20,
            hspace=_row_spacing(n_rows, base=0.26, multirow=0.42),
        )
    else:
        fig.subplots_adjust(
            top=0.90,
            bottom=0.12,
            left=0.08,
            right=0.99,
            wspace=0.20,
            hspace=_row_spacing(n_rows, base=0.26, multirow=0.40),
        )
    return fig


def render_sorted_gain_grid(data, spec: SortedGainGridSpec):
    apply_style(spec.export.role)
    panel_order = [panel.key for panel in spec.panels] if spec.panels else list(dict.fromkeys(data["panel"].tolist()))
    n_rows, n_cols = _grid_shape(len(panel_order))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=_figure_size(spec, 5.0 if len(panel_order) > 2 else 3.0), sharey=True, constrained_layout=False)
    axes = np.atleast_1d(axes).reshape(n_rows, n_cols)
    y_values = data[spec.gain_column].to_numpy(dtype=float)
    pad = 0.08 * max(float(y_values.max() - y_values.min()), 1e-6)

    for idx, panel_key in enumerate(panel_order):
        ax = axes.ravel()[idx]
        panel_data = data.loc[data["panel"] == panel_key].copy()
        panel_data = panel_data.sort_values(spec.gain_column, ascending=not spec.sort_descending).reset_index(drop=True)
        gains = panel_data[spec.gain_column].to_numpy(dtype=float)
        labels = [f"S{i}" for i in range(1, len(panel_data) + 1)]
        colors = [PALETTE.ntdpl if value >= 0.0 else PALETTE.neutral for value in gains]
        ax.bar(np.arange(len(panel_data)), gains, color=colors, width=0.82, edgecolor="none")
        if spec.zero_baseline:
            ax.axhline(0.0, color=PALETTE.border, linestyle="-", linewidth=0.8, alpha=0.8)
        ax.set_xticks(np.arange(len(panel_data)))
        ax.set_xticklabels(labels)
        ax.tick_params(axis="x", labelsize=7.3)
        panel_meta = next((panel for panel in spec.panels if panel.key == panel_key), None)
        if panel_meta is not None:
            ax.set_title(panel_meta.title, pad=2.5)
            ax.set_xlabel(panel_meta.xlabel, labelpad=1.5)
            ax.set_ylabel(panel_meta.ylabel if idx % n_cols == 0 else "")
        style_axes(ax, grid=True)
        ax.set_ylim(float(y_values.min()) - pad, float(y_values.max()) + pad)

    for extra_ax in axes.ravel()[len(panel_order):]:
        extra_ax.set_axis_off()
    fig.subplots_adjust(
        top=0.90,
        bottom=0.14,
        left=0.08,
        right=0.99,
        wspace=0.18,
        hspace=_row_spacing(n_rows, base=0.24, multirow=0.38),
    )
    return fig


def render_heatmap(data, spec: HeatmapSpec):
    apply_style(spec.export.role)
    fig, ax = plt.subplots(1, 1, figsize=_figure_size(spec, 2.9), constrained_layout=False)
    x_col = "difficulty_bin" if "difficulty_bin" in data.columns else "degree"
    y_col = "boundary_bin" if "boundary_bin" in data.columns else "p_max"
    value_col = "value" if "value" in data.columns else "contribution_mean"
    matrix = data.pivot(index=y_col, columns=x_col, values=value_col).sort_index().sort_index(axis=1)
    values = matrix.to_numpy(dtype=float)
    cmap = plt.get_cmap(spec.cmap).copy()
    if spec.center_zero:
        bound = float(np.nanmax(np.abs(values)))
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-bound, vmax=bound)
        image = ax.imshow(values, origin="lower", aspect="auto", cmap=cmap, norm=norm)
    else:
        image = ax.imshow(values, origin="lower", aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels([str(item) for item in matrix.columns.tolist()])
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels([str(item) for item in matrix.index.tolist()])
    ax.set_xlabel(spec.x_label)
    ax.set_ylabel(spec.y_label)
    ax.set_title(spec.title)
    if spec.annotate_cells:
        vmax = float(np.nanmax(np.abs(values))) if values.size else 1.0
        for row_idx in range(values.shape[0]):
            for col_idx in range(values.shape[1]):
                value = values[row_idx, col_idx]
                if np.isnan(value):
                    continue
                text_color = PALETTE.white if abs(value) > 0.45 * vmax else PALETTE.text
                ax.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center", fontsize=7.0, color=text_color)
    style_axes(ax, grid=False)
    fig.subplots_adjust(top=0.90, bottom=0.16, left=0.12, right=0.84)
    cax = fig.add_axes([0.865, ax.get_position().y0, 0.022, ax.get_position().height])
    colorbar = fig.colorbar(image, cax=cax)
    style_colorbar(colorbar, label=spec.value_label)
    return fig


def render_line_plus_heatmap(data, spec: LinePlusHeatmapSpec):
    apply_style(spec.export.role)
    fig = plt.figure(figsize=_figure_size(spec, 2.9))
    grid = GridSpec(1, 2, figure=fig, width_ratios=[1.08, 1.05], wspace=0.18)
    ax_left = fig.add_subplot(grid[0, 0])
    ax_right = fig.add_subplot(grid[0, 1])

    curve = data.loc[data["table"] == "curve"].copy().sort_values("p_max")
    ax_left.fill_between(curve["p_max"], curve["band_lower"], curve["band_upper"], color=PALETTE.ntdpl, alpha=0.16, linewidth=0.0)
    ax_left.plot(curve["p_max"], curve["mean"], color=PALETTE.ntdpl, marker="o", linewidth=2.1, markersize=4.8)
    if np.any(np.isclose(curve["p_max"].to_numpy(dtype=float), 1.0, atol=1e-12)):
        ax_left.axvline(1.0, linestyle="--", linewidth=0.9, color=PALETTE.border, alpha=0.9)
    ax_left.set_xlabel(spec.left_panel.xlabel)
    ax_left.set_ylabel(spec.left_panel.ylabel)
    ax_left.set_title(spec.left_panel.title)
    ax_left.set_xticks(curve["p_max"].to_numpy(dtype=float))
    ax_left.set_xticklabels([str(int(v)) for v in curve["p_max"].to_numpy(dtype=float)])
    style_axes(ax_left, grid=True)

    heat = data.loc[data["table"] == "heatmap"].copy()
    matrix = heat.pivot(index="p_max", columns="degree", values="value").sort_index().sort_index(axis=1)
    values = matrix.to_numpy(dtype=float)
    cmap = plt.get_cmap(spec.cmap).copy()
    cmap.set_bad(color="#eceff4")
    image = ax_right.imshow(values, origin="lower", aspect="auto", cmap=cmap)
    ax_right.set_xticks(np.arange(matrix.shape[1]))
    ax_right.set_xticklabels([str(item) for item in matrix.columns.tolist()])
    ax_right.set_yticks(np.arange(matrix.shape[0]))
    ax_right.set_yticklabels([str(item) for item in matrix.index.tolist()])
    ax_right.set_xlabel(spec.right_panel.xlabel)
    ax_right.set_ylabel(spec.right_panel.ylabel)
    ax_right.set_title(spec.right_panel.title)
    vmax = float(np.nanmax(values)) if values.size else 1.0
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            if np.isnan(value):
                continue
            text_color = PALETTE.white if value > 0.45 * vmax else PALETTE.text
            ax_right.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=7.0, color=text_color)
    style_axes(ax_right, grid=False)

    if spec.show_colorbar:
        fig.subplots_adjust(top=0.90, bottom=0.14, left=0.07, right=0.90)
        cax = fig.add_axes([0.915, ax_right.get_position().y0, 0.026, ax_right.get_position().height])
        colorbar = fig.colorbar(image, cax=cax)
        style_colorbar(colorbar, label=spec.value_label)
    else:
        fig.subplots_adjust(top=0.90, bottom=0.14, left=0.07, right=0.98)
    return fig


def render_heatmap_pair(data, spec: HeatmapPairSpec):
    apply_style(spec.export.role)
    fig, axes = plt.subplots(1, 2, figsize=_figure_size(spec, 4.6), constrained_layout=False)
    axes = np.atleast_1d(axes)
    y_values = list(dict.fromkeys(data[spec.y_col].astype(str).tolist()))
    x_values = list(dict.fromkeys(data[spec.x_col].astype(str).tolist()))
    left_matrix = (
        data.pivot(index=spec.y_col, columns=spec.x_col, values=spec.left_value_col)
        .reindex(index=y_values, columns=x_values)
    )
    right_matrix = (
        data.pivot(index=spec.y_col, columns=spec.x_col, values=spec.right_value_col)
        .reindex(index=y_values, columns=x_values)
    )
    matrices = [left_matrix.to_numpy(dtype=float), right_matrix.to_numpy(dtype=float)]
    titles = [spec.left_title, spec.right_title]
    labels = [spec.left_value_label, spec.right_value_label]
    images = []

    def _abs_bound(values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return 1.0
        return max(float(np.nanmax(np.abs(finite))), 1e-6)

    panel_bounds = [_abs_bound(item) for item in matrices]
    shared_bound = max(panel_bounds) if panel_bounds else 1.0

    for idx, (ax, values, title) in enumerate(zip(axes, matrices, titles, strict=False)):
        cmap = plt.get_cmap(spec.cmap).copy()
        cmap.set_bad(color="#f3f5f7")
        if spec.center_zero:
            bound = shared_bound if spec.shared_color_scale else panel_bounds[idx]
            norm = TwoSlopeNorm(vcenter=0.0, vmin=-bound, vmax=bound)
            image = ax.imshow(values, origin="upper", aspect="equal" if spec.square_cells else "auto", cmap=cmap, norm=norm)
        else:
            image = ax.imshow(values, origin="upper", aspect="equal" if spec.square_cells else "auto", cmap=cmap)
        images.append(image)
        ax.set_title(title, fontsize=8.8, pad=4.0)
        ax.set_xticks(np.arange(len(x_values)))
        ax.set_xticklabels(x_values)
        ax.set_yticks(np.arange(len(y_values)))
        if idx == 0:
            ax.set_yticklabels(y_values)
            ax.set_ylabel(spec.y_label)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel(spec.x_label)
        if spec.square_axes:
            ax.set_box_aspect(1.0)
        if spec.annotate_cells:
            bound = panel_bounds[idx] if spec.center_zero else _abs_bound(values)
            for row_idx in range(values.shape[0]):
                for col_idx in range(values.shape[1]):
                    value = float(values[row_idx, col_idx])
                    if not np.isfinite(value):
                        continue
                    text_color = PALETTE.white if abs(value) > 0.50 * bound else PALETTE.text
                    ax.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center", fontsize=7.0, color=text_color)
        style_axes(ax, grid=False)

    if spec.show_colorbar:
        fig.subplots_adjust(top=0.90, bottom=0.15, left=0.08, right=0.93, wspace=0.18)
        for idx, (ax, image, label) in enumerate(zip(axes, images, labels, strict=False)):
            position = ax.get_position()
            cax = fig.add_axes([position.x1 + 0.008, position.y0, 0.016, position.height])
            colorbar = fig.colorbar(image, cax=cax)
            style_colorbar(colorbar, label=label)
    else:
        fig.subplots_adjust(top=0.90, bottom=0.15, left=0.08, right=0.99, wspace=0.06)
    return fig


def render_image_comparison_grid(data, spec: ImageComparisonGridSpec):
    apply_style(spec.export.role)
    scene_order = list(spec.scene_order)
    panel_order = list(spec.panel_order)
    fig, axes = plt.subplots(len(scene_order), len(panel_order), figsize=_figure_size(spec, 5.4), constrained_layout=False)
    axes = np.atleast_2d(axes)
    title_map = {
        "original": "GT",
        "tucker": "Tucker",
        "ntdpl": "NTD-PL",
        "tucker_error": "Tucker err",
        "ntdpl_error": "NTD-PL err",
        "error_reduction": "Err. reduction",
    }
    error_values = np.concatenate(
        [np.ravel(np.nan_to_num(row["image"], nan=0.0)) for _, row in data.loc[data["panel_type"] == "error"].iterrows()]
    )
    improve_values = np.concatenate(
        [np.ravel(np.nan_to_num(row["image"], nan=0.0)) for _, row in data.loc[data["panel_type"] == "improvement"].iterrows()]
    )
    error_vmax = max(float(np.nanmax(error_values)), 1e-6)
    improve_bound = max(float(np.nanmax(np.abs(improve_values))), 1e-6)
    improve_norm = TwoSlopeNorm(vcenter=0.0, vmin=-improve_bound, vmax=improve_bound)
    err_cmap = plt.get_cmap("magma").copy()
    err_cmap.set_bad(color="#f3f5f7")
    gain_cmap = plt.get_cmap("RdBu_r").copy()
    gain_cmap.set_bad(color="#f3f5f7")
    err_im = None
    improve_im = None

    for row_idx, scene_id in enumerate(scene_order):
        scene_data = data.loc[data["scene_id"] == scene_id].copy()
        scene_name = str(scene_data["scene_name"].iloc[0]).replace("_", " ")
        for col_idx, panel_key in enumerate(panel_order):
            ax = axes[row_idx, col_idx]
            item = scene_data.loc[scene_data["panel"] == panel_key].iloc[0]
            image = item["image"]
            panel_type = str(item["panel_type"]) if "panel_type" in item else ("improvement" if bool(item.get("is_difference", False)) else "rgb")
            if panel_type == "rgb":
                im = ax.imshow(np.clip(image, 0.0, 1.0))
            elif panel_type == "error":
                err_im = ax.imshow(np.asarray(image, dtype=float), cmap=err_cmap, vmin=0.0, vmax=error_vmax)
            else:
                improve_im = ax.imshow(np.asarray(image, dtype=float), cmap=gain_cmap, norm=improve_norm)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(title_map.get(panel_key, panel_key.replace("_", " ").title()), fontsize=8.6, pad=3.0)
            if col_idx == 0:
                ax.text(-0.08, 0.5, scene_name, transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=8.6, fontweight="semibold")
    fig.subplots_adjust(top=0.91, bottom=0.05, left=0.05, right=0.92, wspace=0.025, hspace=0.05)
    y0 = min(axes[row, -1].get_position().y0 for row in range(axes.shape[0]))
    y1 = max(axes[row, -1].get_position().y1 for row in range(axes.shape[0]))
    span = y1 - y0
    if err_im is not None:
        cax_err = fig.add_axes([0.928, y0 + 0.52 * span, 0.016, 0.46 * span])
        err_cbar = fig.colorbar(err_im, cax=cax_err)
        style_colorbar(err_cbar, label="RMSE")
    if improve_im is not None:
        cax_gain = fig.add_axes([0.928, y0, 0.016, 0.46 * span])
        gain_cbar = fig.colorbar(improve_im, cax=cax_gain)
        style_colorbar(gain_cbar, label=spec.colorbar_label)
    return fig


def render_completion_visual_grid(data, spec: CompletionVisualGridSpec):
    apply_style(spec.export.role)
    panel_order = list(spec.panel_order)
    panel_titles = list(spec.panel_titles) if spec.panel_titles else [item.replace("_", " ").title() for item in panel_order]
    scene_order = data.loc[:, ["scene_id", "scene_name", "rmse_gain"]].drop_duplicates().sort_values("scene_id").to_dict("records")
    fig, axes = plt.subplots(len(scene_order), len(panel_order), figsize=_figure_size(spec, 5.6), constrained_layout=False)
    axes = np.atleast_2d(axes)

    error_values = np.concatenate(
        [
            np.ravel(np.nan_to_num(row["image"], nan=0.0))
            for _, row in data.loc[data["panel_type"] == "error"].iterrows()
        ]
    )
    improve_values = np.concatenate(
        [
            np.ravel(np.nan_to_num(row["image"], nan=0.0))
            for _, row in data.loc[data["panel_type"] == "improvement"].iterrows()
        ]
    )
    error_vmax = max(float(np.nanmax(error_values)), 1e-6)
    improve_bound = max(float(np.nanmax(np.abs(improve_values))), 1e-6)
    improve_norm = TwoSlopeNorm(vcenter=0.0, vmin=-improve_bound, vmax=improve_bound)
    err_cmap = plt.get_cmap("magma").copy()
    err_cmap.set_bad(color="#f3f5f7")
    gain_cmap = plt.get_cmap("RdBu_r").copy()
    gain_cmap.set_bad(color="#f3f5f7")
    err_im = None
    improve_im = None

    for row_idx, scene in enumerate(scene_order):
        scene_id = int(scene["scene_id"])
        scene_name = str(scene["scene_name"]).replace("_", " ")
        rmse_gain = float(scene["rmse_gain"]) if np.isfinite(float(scene["rmse_gain"])) else np.nan
        scene_data = data.loc[data["scene_id"] == scene_id].copy()
        for col_idx, panel_key in enumerate(panel_order):
            ax = axes[row_idx, col_idx]
            panel_row = scene_data.loc[scene_data["panel"] == panel_key].iloc[0]
            image = panel_row["image"]
            panel_type = str(panel_row["panel_type"])
            if panel_type == "rgb":
                ax.imshow(np.clip(np.asarray(image, dtype=float), 0.0, 1.0))
            elif panel_type == "error":
                err_im = ax.imshow(np.asarray(image, dtype=float), cmap=err_cmap, vmin=0.0, vmax=error_vmax)
            else:
                improve_im = ax.imshow(np.asarray(image, dtype=float), cmap=gain_cmap, norm=improve_norm)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0 and col_idx < len(panel_titles):
                ax.set_title(panel_titles[col_idx], fontsize=8.6, pad=3.0)
            if col_idx == 0:
                label = scene_name if not np.isfinite(rmse_gain) else f"{scene_name}\nΔRMSE={rmse_gain:.4f}"
                ax.set_ylabel(label, rotation=90, labelpad=11, fontsize=8.4, fontweight="normal")

    if spec.show_colorbars:
        fig.subplots_adjust(top=0.91, bottom=0.05, left=0.05, right=0.92, wspace=0.025, hspace=0.05)
        y0 = min(axes[row, -1].get_position().y0 for row in range(axes.shape[0]))
        y1 = max(axes[row, -1].get_position().y1 for row in range(axes.shape[0]))
        if err_im is not None:
            cax_err = fig.add_axes([0.928, y0 + 0.52 * (y1 - y0), 0.016, 0.46 * (y1 - y0)])
            err_cbar = fig.colorbar(err_im, cax=cax_err)
            style_colorbar(err_cbar, label="RMSE*")
        if improve_im is not None:
            cax_gain = fig.add_axes([0.928, y0, 0.016, 0.46 * (y1 - y0)])
            gain_cbar = fig.colorbar(improve_im, cax=cax_gain)
            style_colorbar(gain_cbar, label="Tucker - NTD-PL (>0 better)")
    else:
        fig.subplots_adjust(top=0.91, bottom=0.05, left=0.05, right=0.99, wspace=0.025, hspace=0.05)
    return fig


def render_spatial_case_grid(data, spec: SpatialCaseGridSpec):
    apply_style(spec.export.role)
    panel_order = list(spec.panel_order)
    panel_titles = list(spec.panel_titles)
    row = data.iloc[0]
    scene_name = str(row["scene_name"]).replace("_", " ")
    fig, axes = plt.subplots(1, len(panel_order), figsize=_figure_size(spec, 3.0), constrained_layout=False)
    axes = np.atleast_1d(axes)

    difficulty_values = np.asarray(data.loc[data["panel"] == "difficulty", "image"].iloc[0], dtype=float)
    boundary_values = np.asarray(data.loc[data["panel"] == "boundary", "image"].iloc[0], dtype=float)
    gain_values = np.asarray(data.loc[data["panel"] == "gain", "image"].iloc[0], dtype=float)
    difficulty_vmin = float(np.nanquantile(difficulty_values, 0.02))
    difficulty_vmax = float(np.nanquantile(difficulty_values, 0.98))
    if not np.isfinite(difficulty_vmin) or not np.isfinite(difficulty_vmax) or np.isclose(difficulty_vmin, difficulty_vmax):
        difficulty_vmin = float(np.nanmin(difficulty_values))
        difficulty_vmax = float(np.nanmax(difficulty_values))

    boundary_vmin = float(np.nanquantile(boundary_values, 0.02))
    boundary_vmax = float(np.nanquantile(boundary_values, 0.98))
    if not np.isfinite(boundary_vmin) or not np.isfinite(boundary_vmax) or np.isclose(boundary_vmin, boundary_vmax):
        boundary_vmin = float(np.nanmin(boundary_values))
        boundary_vmax = float(np.nanmax(boundary_values))

    gain_bound = max(float(np.nanquantile(np.abs(gain_values), 0.98)), 1e-6)
    gain_norm = TwoSlopeNorm(vcenter=0.0, vmin=-gain_bound, vmax=gain_bound)
    difficulty_im = None
    boundary_im = None
    gain_im = None

    for idx, panel_key in enumerate(panel_order):
        ax = axes[idx]
        panel = data.loc[data["panel"] == panel_key].iloc[0]
        image = panel["image"]
        panel_type = str(panel["panel_type"])
        if panel_type == "rgb":
            ax.imshow(np.clip(np.asarray(image, dtype=float), 0.0, 1.0))
        elif panel_type == "difficulty":
            difficulty_im = ax.imshow(np.asarray(image, dtype=float), cmap="Oranges", vmin=difficulty_vmin, vmax=difficulty_vmax)
        elif panel_type == "boundary":
            boundary_im = ax.imshow(np.asarray(image, dtype=float), cmap="Blues", vmin=boundary_vmin, vmax=boundary_vmax)
        else:
            gain_im = ax.imshow(np.asarray(image, dtype=float), cmap="RdBu_r", norm=gain_norm)

        overlap = np.asarray(panel["overlap_mask"], dtype=float)
        # Use a single overlap contour to avoid clashing with the heatmap colormap.
        ax.contour(overlap, levels=[0.5], colors=[PALETTE.border], linewidths=1.9, alpha=0.62)
        ax.contour(overlap, levels=[0.5], colors=[PALETTE.white], linewidths=1.1, alpha=0.97)

        ax.set_title(panel_titles[idx] if idx < len(panel_titles) else panel_key, fontsize=8.2, pad=3.0)
        ax.set_xticks([])
        ax.set_yticks([])
        if panel_type == "gain":
            ax.text(
                0.03,
                0.06,
                "Blue: Tucker better",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=6.6,
                color="#2166AC",
                bbox={"boxstyle": "round,pad=0.12", "facecolor": PALETTE.white, "edgecolor": PALETTE.grid, "alpha": 0.92},
            )
            ax.text(
                0.97,
                0.06,
                "Red: NTD-PL better",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=6.6,
                color="#B2182B",
                bbox={"boxstyle": "round,pad=0.12", "facecolor": PALETTE.white, "edgecolor": PALETTE.grid, "alpha": 0.92},
            )
        style_axes(ax, grid=False)

    axes[0].text(
        -0.07,
        0.5,
        scene_name,
        transform=axes[0].transAxes,
        rotation=90,
        va="center",
        ha="right",
        fontsize=8.4,
        fontweight="semibold",
    )
    bottom = 0.14 if spec.show_notes else 0.06
    right = 0.955 if spec.show_colorbars else 0.99
    fig.subplots_adjust(top=0.86, bottom=bottom, left=0.06, right=right, wspace=0.04)

    if spec.show_colorbars:
        base_y = axes[-1].get_position().y0
        height = axes[-1].get_position().height
        cbar_width = 0.008
        x0 = 0.962
        if difficulty_im is not None:
            cax = fig.add_axes([x0, base_y + 0.68 * height, cbar_width, 0.30 * height])
            style_colorbar(fig.colorbar(difficulty_im, cax=cax), label="Difficulty (RMSE*)")
        if boundary_im is not None:
            cax = fig.add_axes([x0, base_y + 0.34 * height, cbar_width, 0.30 * height])
            style_colorbar(fig.colorbar(boundary_im, cax=cax), label="Boundary score")
        if gain_im is not None:
            cax = fig.add_axes([x0, base_y, cbar_width, 0.30 * height])
            style_colorbar(fig.colorbar(gain_im, cax=cax), label=r"$\Delta$RMSE*")

    if spec.show_notes:
        fig.text(0.50, 0.05, "Contours: top difficulty (yellow), top boundary (cyan), intersection (white).", ha="center", va="center", fontsize=7.5, color=PALETTE.border)
        fig.text(0.50, 0.015, "Positive error reduction indicates NTD-PL better on missing-only recovery.", ha="center", va="bottom", fontsize=7.5, color=PALETTE.border)
    return fig


def render_spectra_panel(data, spec: SpectraPanelSpec):
    apply_style(spec.export.role)
    scene_order = list(spec.scene_order)
    fig = plt.figure(figsize=_figure_size(spec, 5.6))
    grid = GridSpec(
        len(scene_order),
        1 + spec.columns_per_scene,
        figure=fig,
        width_ratios=[1.0, 1.18, 1.18],
        hspace=_row_spacing(len(scene_order), base=0.26, multirow=0.36),
        wspace=0.18,
    )
    legend_handles = []
    legend_labels = []
    map_rows = data.loc[data["kind"] == "map"].copy()
    curve_rows = data.loc[data["kind"] == "curve"].copy()

    for row_idx, scene_id in enumerate(scene_order):
        scene_map = map_rows.loc[map_rows["scene_id"] == scene_id].copy()
        scene_name = str(scene_map["scene_name"].iloc[0]).replace("_", " ")
        map_ax = fig.add_subplot(grid[row_idx, 0])
        map_image = scene_map["image"].iloc[0]
        map_ax.imshow(np.clip(map_image, 0.0, 1.0))
        map_ax.set_xticks([])
        map_ax.set_yticks([])
        map_ax.text(-0.08, 0.5, scene_name, transform=map_ax.transAxes, rotation=90, va="center", ha="right", fontsize=8.6, fontweight="semibold")
        for item in scene_map.itertuples(index=False):
            map_ax.scatter(item.col, item.row, s=28, facecolor=PALETTE.white, edgecolor=PALETTE.highlight, linewidth=1.0, zorder=5)
            map_ax.text(item.col + 2.0, item.row - 1.5, item.pixel_label, fontsize=7.1, color=PALETTE.text, bbox={"boxstyle": "round,pad=0.16", "facecolor": PALETTE.white, "edgecolor": PALETTE.highlight, "alpha": 0.95})

        pixel_order = sorted(scene_map["pixel_label"].astype(str).unique().tolist())
        for col_offset, pixel_label in enumerate(pixel_order, start=1):
            ax = fig.add_subplot(grid[row_idx, col_offset])
            panel = curve_rows.loc[(curve_rows["scene_id"] == scene_id) & (curve_rows["pixel_label"] == pixel_label)].copy()
            for method in ("Ground truth", "Tucker", "NTD-PL"):
                sub = panel.loc[panel["method"] == method].copy().sort_values("band")
                style = method_style(method)
                line, = ax.plot(
                    sub["band"],
                    sub["value"],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    marker=style.get("marker"),
                    linewidth=style["linewidth"],
                    markersize=3.8,
                    label=method,
                )
                if method not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(method)
            ax.set_title(pixel_label, pad=2.0)
            ax.set_xlabel("Band", labelpad=1.2)
            ax.set_ylabel("Reflectance" if col_offset == 1 else "")
            style_axes(ax, grid=True)
    fig.legend(legend_handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncols=3)
    fig.subplots_adjust(top=0.88, bottom=0.10, left=0.08, right=0.99)
    return fig


def render_geometry_evolution(data, spec: GeometryEvolutionSpec):
    apply_style(spec.export.role)
    fig = plt.figure(figsize=_figure_size(spec, 5.1))
    grid = GridSpec(2, 2, figure=fig, height_ratios=[1.38, 0.92], hspace=0.40, wspace=0.26)
    ax_link = fig.add_subplot(grid[0, 0])
    ax_delta = fig.add_subplot(grid[0, 1])
    ax_metric_link = fig.add_subplot(grid[1, 0])
    ax_metric_response = fig.add_subplot(grid[1, 1])

    identity = data.loc[data["table"] == "identity"].copy().sort_values("s")
    curves = data.loc[data["table"] == "curve"].copy()
    deltas = data.loc[data["table"] == "delta"].copy()
    metrics = data.loc[data["table"] == "metric"].copy()

    handles = []
    identity_line, = ax_link.plot(
        identity["s"],
        identity["value"],
        color=PALETTE.text,
        linestyle=(0, (3, 2)),
        linewidth=1.2,
        alpha=0.7,
        label=r"$y=s$",
    )
    handles.append(identity_line)

    for order in spec.order_values:
        style = _order_style(int(order))
        label = f"p={order}" + (" (linear)" if int(order) == 1 else "")
        panel = curves.loc[curves["order"] == int(order)].copy().sort_values("s")
        line, = ax_link.plot(
            panel["s"],
            panel["value"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=float(style["linewidth"]),
            marker=style["marker"],
            markersize=float(style["markersize"]),
            markevery=24,
            label=label,
        )
        handles.append(line)

    ax_link.set_title("Learned link functions")
    ax_link.set_xlabel(r"Latent scalar $s$")
    ax_link.set_ylabel(r"$f_{\beta^{(p)}}(s)$")
    ax_link.text(
        0.98,
        0.03,
        rf"real learned $\beta$, $\alpha={spec.alpha:g}$",
        transform=ax_link.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.4,
        color=PALETTE.border,
    )
    style_axes(ax_link, grid=True)

    ax_delta.axhline(0.0, color=PALETTE.border, linestyle=(0, (3, 2)), linewidth=0.9, alpha=0.85)
    for order in spec.order_values:
        if int(order) < 2:
            continue
        style = _order_style(int(order))
        panel = deltas.loc[deltas["order"] == int(order)].copy().sort_values("s")
        ax_delta.plot(
            panel["s"],
            panel["value"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=float(style["linewidth"]),
            marker=style["marker"],
            markersize=float(style["markersize"]),
            markevery=24,
            label=f"p={order}",
        )
    ax_delta.set_title(r"Deviation from the $p=1$ link")
    ax_delta.set_xlabel(r"Latent scalar $s$")
    ax_delta.set_ylabel(r"$\Delta_p(s)=f_{\beta^{(p)}}(s)-f_{\beta^{(1)}}(s)$")
    style_axes(ax_delta, grid=True)

    metric_specs = [
        ("mean_abs_link_deviation", ax_metric_link, r"Mean $|\Delta_p(s)|$ over the shared $s$ range"),
        ("mean_abs_response_deviation", ax_metric_response, r"Mean $|y_p(u,v)-y_1(u,v)|$ on the fixed local patch"),
    ]
    for metric_key, ax, title in metric_specs:
        panel = metrics.loc[metrics["metric"] == metric_key].copy().sort_values("order")
        x_values = panel["order"].to_numpy(dtype=float)
        y_values = panel["value"].to_numpy(dtype=float)
        ax.axhline(0.0, color=PALETTE.grid, linewidth=0.7, alpha=0.7)
        for order, x_value, y_value in zip(panel["order"], x_values, y_values, strict=False):
            style = _order_style(int(order))
            ax.plot(
                [x_value],
                [y_value],
                color=style["color"],
                marker=style["marker"],
                markersize=float(style["markersize"]) + 0.4,
                linestyle="None",
            )
        ax.plot(x_values, y_values, color=PALETTE.border, linewidth=1.2, alpha=0.75)
        ax.set_title(title, fontsize=8.6)
        ax.set_xlabel(r"$p_{\max}$")
        ax.set_xticks(np.asarray(spec.order_values, dtype=float))
        ax.set_ylabel("Magnitude")
        style_axes(ax, grid=True)

    fig.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncols=min(len(handles), 5),
    )
    fig.subplots_adjust(top=0.84, bottom=0.11, left=0.09, right=0.99)
    return fig


def render_geometry_response_maps(data, spec: GeometryResponseMapsSpec):
    apply_style(spec.export.role)
    order_values = [int(order) for order in spec.order_values]
    fig = plt.figure(figsize=_figure_size(spec, 4.95))
    grid = GridSpec(2, len(order_values), figure=fig, hspace=0.08, wspace=0.04)
    response_rows = data.loc[:, ["order", "u", "v", "response"]].copy()
    deviation_rows = data.loc[:, ["order", "u", "v", "deviation"]].copy()
    response_values = response_rows["response"].to_numpy(dtype=float)
    deviation_values = deviation_rows["deviation"].to_numpy(dtype=float)
    response_vmin = float(np.nanmin(response_values))
    response_vmax = float(np.nanmax(response_values))
    deviation_bound = float(np.nanmax(np.abs(deviation_values)))

    baseline = response_rows.loc[response_rows["order"] == min(order_values)].copy()
    baseline_matrix = baseline.pivot(index="v", columns="u", values="response").sort_index().sort_index(axis=1)
    baseline_u = baseline_matrix.columns.to_numpy(dtype=float)
    baseline_v = baseline_matrix.index.to_numpy(dtype=float)
    baseline_u_grid, baseline_v_grid = np.meshgrid(baseline_u, baseline_v)
    baseline_levels = np.linspace(float(np.nanmin(baseline_matrix.to_numpy(dtype=float))), float(np.nanmax(baseline_matrix.to_numpy(dtype=float))), 7)
    response_levels = np.linspace(response_vmin, response_vmax, 11)
    response_artist = None
    deviation_artist = None

    for col_idx, order in enumerate(order_values):
        response_ax = fig.add_subplot(grid[0, col_idx])
        deviation_ax = fig.add_subplot(grid[1, col_idx])

        response_panel = response_rows.loc[response_rows["order"] == order].copy()
        response_matrix = response_panel.pivot(index="v", columns="u", values="response").sort_index().sort_index(axis=1)
        u_values = response_matrix.columns.to_numpy(dtype=float)
        v_values = response_matrix.index.to_numpy(dtype=float)
        u_grid, v_grid = np.meshgrid(u_values, v_values)
        response_matrix_values = response_matrix.to_numpy(dtype=float)
        response_artist = response_ax.contourf(
            u_grid,
            v_grid,
            response_matrix_values,
            levels=response_levels,
            cmap="cividis",
            extend="both",
        )
        response_ax.contour(
            baseline_u_grid,
            baseline_v_grid,
            baseline_matrix.to_numpy(dtype=float),
            levels=baseline_levels,
            colors=PALETTE.border,
            linewidths=0.55,
            linestyles="--",
            alpha=0.58,
        )
        response_ax.set_aspect("equal")
        response_ax.set_xticks([])
        if col_idx == 0:
            response_ax.set_ylabel("Response\n" + r"local coord. $v$")
        else:
            response_ax.set_yticks([])
        style_axes(response_ax, grid=False)

        deviation_panel = deviation_rows.loc[deviation_rows["order"] == order].copy()
        deviation_matrix = deviation_panel.pivot(index="v", columns="u", values="deviation").sort_index().sort_index(axis=1)
        deviation_matrix_values = deviation_matrix.to_numpy(dtype=float)
        deviation_artist = deviation_ax.contourf(
            u_grid,
            v_grid,
            deviation_matrix_values,
            levels=np.linspace(-deviation_bound, deviation_bound, 13),
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vcenter=0.0, vmin=-deviation_bound, vmax=deviation_bound),
            extend="both",
        )
        deviation_ax.contour(
            u_grid,
            v_grid,
            deviation_matrix_values,
            levels=[0.0],
            colors=PALETTE.border,
            linewidths=0.7,
            alpha=0.70,
        )
        deviation_ax.set_aspect("equal")
        deviation_ax.set_xlabel(r"local coord. $u$")
        if col_idx == 0:
            deviation_ax.set_ylabel("Deviation\n" + r"local coord. $v$")
        else:
            deviation_ax.set_yticks([])
        style_axes(deviation_ax, grid=False)

    fig.subplots_adjust(top=0.99, bottom=0.10, left=0.07, right=0.995)
    return fig


def render_figure(data, spec: BaseSpec):
    if spec.family == "paired_boxplot":
        return render_paired_boxplot(data, spec)  # type: ignore[arg-type]
    if spec.family == "boxplot_summary":
        return render_boxplot_grid(data, spec)  # type: ignore[arg-type]
    if spec.family == "line_grid":
        return render_line_grid(data, spec)  # type: ignore[arg-type]
    if spec.family == "step_line_grid":
        return render_line_grid(data, spec, step_mode=True)  # type: ignore[arg-type]
    if spec.family == "sorted_gain_grid":
        return render_sorted_gain_grid(data, spec)  # type: ignore[arg-type]
    if spec.family == "heatmap":
        return render_heatmap(data, spec)  # type: ignore[arg-type]
    if spec.family == "line_plus_heatmap":
        return render_line_plus_heatmap(data, spec)  # type: ignore[arg-type]
    if spec.family == "heatmap_pair":
        return render_heatmap_pair(data, spec)  # type: ignore[arg-type]
    if spec.family == "image_comparison_grid":
        return render_image_comparison_grid(data, spec)  # type: ignore[arg-type]
    if spec.family == "completion_visual_grid":
        return render_completion_visual_grid(data, spec)  # type: ignore[arg-type]
    if spec.family == "spatial_case_grid":
        return render_spatial_case_grid(data, spec)  # type: ignore[arg-type]
    if spec.family == "spectra_panel":
        return render_spectra_panel(data, spec)  # type: ignore[arg-type]
    if spec.family == "geometry_evolution":
        return render_geometry_evolution(data, spec)  # type: ignore[arg-type]
    if spec.family == "geometry_response_maps":
        return render_geometry_response_maps(data, spec)  # type: ignore[arg-type]
    raise ValueError(f"Unsupported figure family: {spec.family}")
