from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from ..config import get_env
from .io import maybe_numeric
from .nmse_common import (
    DEFAULT_METHOD_ORDER,
    curve_band_summary,
    filter_levels,
    load_runs,
    mean_curve,
    select_ntdpl,
    sorted_unique,
)
from .plotting import apply_theme, legend_style, method_style, rounded_bounds, save_figure, style_axes


@dataclass(frozen=True)
class NmseAlphaSpec:
    exp_name: str
    ntdpl_p_max: int = 8
    method_order: Sequence[str] = tuple(DEFAULT_METHOD_ORDER)
    alpha_values: Sequence[float] | None = None
    alpha_range: tuple[float, float] | None = None


def save_nmse_vs_alpha(exp_name: str, nonlinear: str, spec: NmseAlphaSpec) -> Path:
    apply_theme()
    env = get_env(exp_name)
    runs = load_runs(env)

    tbl = runs.loc[runs["ovr.filter.nonlinear"].astype(str) == str(nonlinear)].copy()
    if tbl.empty:
        raise RuntimeError(f"No runs found for exp={exp_name}, nonlinear={nonlinear}.")

    alpha_levels = sorted_unique(tbl["ovr.filter.alpha"])
    alpha_levels = filter_levels(alpha_levels, spec.alpha_values, spec.alpha_range)

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    all_y: list[float] = []

    for method in spec.method_order:
        sub = tbl.loc[tbl["ovr.method"].astype(str) == str(method)].copy()
        if method == "ntdpl":
            sub = select_ntdpl(sub, spec.ntdpl_p_max)
        if sub.empty:
            continue

        label = env.label_for_method(method)
        sub["ovr.filter.alpha"] = maybe_numeric(sub["ovr.filter.alpha"])
        sub["NMSE_dB"] = maybe_numeric(sub["NMSE_dB"])
        style = method_style(label)

        if method == "ntdpl":
            summary = curve_band_summary(sub, "ovr.filter.alpha", "NMSE_dB")
            x_vals = summary["ovr.filter.alpha"].to_numpy(dtype=float)
            y_vals = summary["mean"].to_numpy(dtype=float)
            ax.fill_between(
                x_vals,
                summary["lower"].to_numpy(dtype=float),
                summary["upper"].to_numpy(dtype=float),
                color=style["color"],
                alpha=0.18,
                linewidth=0,
            )
        else:
            summary = mean_curve(sub, "ovr.filter.alpha", "NMSE_dB")
            x_vals = summary["ovr.filter.alpha"].to_numpy(dtype=float)
            y_vals = summary["NMSE_dB"].to_numpy(dtype=float)

        ax.plot(x_vals, y_vals, label=label, **style)
        all_y.extend([float(v) for v in y_vals.tolist()])

    ax.set_title(f"{nonlinear}: NMSE-alpha")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("NMSE (dB)")
    if alpha_levels:
        ax.set_xticks(alpha_levels)
        ax.set_xticklabels([f"{v:g}" for v in alpha_levels])
    if all_y:
        ax.set_ylim(*rounded_bounds(all_y, 5.0))

    style_axes(ax, grid=True)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            **legend_style(
                handles,
                labels,
                ncols=max(1, min(len(labels), 3)),
                loc="upper center",
                bbox_to_anchor=(0.5, 1.22),
            )
        )
        fig.subplots_adjust(top=0.78)

    stem = f"nmse_alpha_{nonlinear}"
    destination = env.artifacts_dir / stem
    save_figure(fig, destination)
    return destination.with_suffix(".pdf")
