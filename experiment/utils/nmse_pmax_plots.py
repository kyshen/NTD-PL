from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from ..config import get_env
from .io import maybe_numeric
from .nmse_common import curve_band_summary, filter_levels, load_runs, sorted_unique
from .plotting import apply_theme, method_style, rounded_bounds, save_figure, style_axes


@dataclass(frozen=True)
class NmsePmaxSpec:
    exp_name: str
    alpha_ref: float = 0.25
    pmax_values: Sequence[int] | None = None
    pmax_range: tuple[int, int] | None = None


def save_nmse_vs_pmax(exp_name: str, nonlinear: str, spec: NmsePmaxSpec) -> Path:
    apply_theme()
    env = get_env(exp_name)
    runs = load_runs(env)

    ntdpl_all = runs.loc[
        (runs["ovr.method"].astype(str) == "ntdpl")
        & (runs["ovr.filter.nonlinear"].astype(str) == str(nonlinear))
    ].copy()
    if ntdpl_all.empty:
        raise RuntimeError(
            f"No NTDPL runs found for exp={exp_name}, nonlinear={nonlinear}."
        )

    alpha_series = maybe_numeric(ntdpl_all["ovr.filter.alpha"]).to_numpy(dtype=float)
    panel = ntdpl_all.loc[
        np.isclose(alpha_series, float(spec.alpha_ref), equal_nan=False)
    ].copy()
    panel["ovr.method.p_max"] = maybe_numeric(panel["ovr.method.p_max"])
    panel["NMSE_dB"] = maybe_numeric(panel["NMSE_dB"])

    pmax_levels = sorted_unique(panel["ovr.method.p_max"]) if not panel.empty else []
    pmax_levels = filter_levels(pmax_levels, spec.pmax_values, spec.pmax_range)

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    if not panel.empty:
        if pmax_levels:
            panel = panel.loc[
                panel["ovr.method.p_max"].isin([float(v) for v in pmax_levels])
            ]
        label = env.label_for_method("ntdpl")
        style = method_style(label)
        summary = curve_band_summary(panel, "ovr.method.p_max", "NMSE_dB")
        x_vals = summary["ovr.method.p_max"].to_numpy(dtype=float)
        y_vals = summary["mean"].to_numpy(dtype=float)
        ax.fill_between(
            x_vals,
            summary["lower"].to_numpy(dtype=float),
            summary["upper"].to_numpy(dtype=float),
            color=style["color"],
            alpha=0.18,
            linewidth=0,
        )
        ax.plot(x_vals, y_vals, **style)

        ax.set_xticks(pmax_levels if pmax_levels else np.unique(x_vals))
        ax.set_xlim(float(x_vals.min()) - 0.2, float(x_vals.max()) + 0.2)
        ax.set_ylim(*rounded_bounds(y_vals, 5.0))

    ax.set_title(f"{nonlinear}: NMSE-pmax")
    ax.set_xlabel(r"$p_{max}$")
    ax.set_ylabel("NMSE (dB)")
    ax.text(
        0.98,
        0.05,
        rf"$\alpha={spec.alpha_ref:g}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
    )
    style_axes(ax, grid=True)

    stem = f"nmse_pmax_{nonlinear}"
    destination = env.artifacts_dir / stem
    save_figure(fig, destination)
    return destination.with_suffix(".pdf")
