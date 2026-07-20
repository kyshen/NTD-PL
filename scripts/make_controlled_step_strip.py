from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.tucker import TuckerData
from src.methods.ntdpl import NTDPLDecomposition
from src.types import LogCallback, Tensor
from viz.style import PALETTE, apply_style, method_style, style_axes
from scripts.make_controlled_linear_noise_preview import SEEDS, _add_noise, _response


OUT_DIR = ROOT / "papers" / "tsp-supplementary" / "figures"
OUT_STEM = "controlled_nonlinear_step_strip"
ALPHA_REF = 0.25
P_MAX = 6

PANEL_LABELS = {
    "linear": r"$s$",
    "poly3": r"$s^2+s^3$",
    "tanh": r"$\tanh(\kappa s)$",
    "exp": r"$(e^{\kappa s}-1)/\kappa$",
}


def _target_for(seed: int, panel: str) -> np.ndarray:
    data = TuckerData(shape=(10, 10, 10), rank=(4, 4, 4), seed=seed)
    signal = np.asarray(data.get("fit").dense, dtype=np.float32)
    signal = signal / (np.linalg.norm(signal) + 1e-8) * np.sqrt(signal.size)
    clean = _response(signal, panel, ALPHA_REF)
    panel_index = list(PANEL_LABELS).index(panel)
    noise_seed = 10_000 + 1_000 * seed + 37 * panel_index
    if panel != "linear":
        noise_seed += int(round(ALPHA_REF * 1000))
    return _add_noise(clean, seed=noise_seed)


def _fit_trace(target: np.ndarray, seed: int) -> list[dict[str, float | int]]:
    tensor = Tensor(shape=target.shape, dense=target.astype(np.float32))
    log = LogCallback(log_level=1)
    model = NTDPLDecomposition(
        rank=(4, 4, 4),
        init_n_iter_max=50,
        init="tucker",
        stable_beta_update=True,
        beta_update_stage="before_grad",
        random_state=seed,
        p_max=P_MAX,
        link_kind="power",
        allow_constant_term=True,
        use_continuation=True,
        factor_normalize=True,
        lr_core=1e-4,
        lr_factors=3e-4,
        lambda_core=1e-6,
        lambda_factors=1e-6,
        lambda_beta=1e-6,
        beta_update_method="moments_normal_eq",
        beta_update_interval=5,
        n_iter_max=1000,
    )
    model.fit(tensor, None, log)
    if log.logs is None:
        return []
    rows: list[dict[str, float | int]] = []
    for step, item in enumerate(log.logs):
        rows.append(
            {
                "step": int(step),
                "RMSE": float(item["RMSE"]),
                "p": int(item["p"]),
            }
        )
    return rows


def _collect() -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for panel in PANEL_LABELS:
        for seed in SEEDS:
            target = _target_for(seed, panel)
            for item in _fit_trace(target, seed):
                rows.append({"panel": panel, "seed": int(seed), **item})
    return pd.DataFrame(rows)


def _aggregate_trace(frame: pd.DataFrame) -> pd.DataFrame:
    curves = (
        frame.groupby(["panel", "step"], as_index=False)["RMSE"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"step": "x"})
    )
    curves["std"] = curves["std"].fillna(0.0)
    curves["kind"] = "curve"
    curves["method"] = "NTD-PL"

    rows = curves.to_dict("records")
    for panel in PANEL_LABELS:
        first_seed = int(frame.loc[frame["panel"].eq(panel), "seed"].min())
        path = frame.loc[frame["panel"].eq(panel) & frame["seed"].eq(first_seed)].sort_values("step")
        previous = path["p"].shift(1)
        increased = path.loc[(previous.notna()) & (path["p"] > previous)]
        for idx, item in enumerate(increased.itertuples(index=False), start=2):
            rows.append(
                {
                    "panel": panel,
                    "x": int(item.step),
                    "mean": np.nan,
                    "std": np.nan,
                    "kind": "transition",
                    "method": "NTD-PL",
                    "degree": idx,
                }
            )
    return pd.DataFrame(rows)


def _compact_ntd_style() -> dict:
    style = method_style("NTD-PL")
    style.update({"linewidth": 1.85, "markersize": 3.5, "zorder": 4})
    return style


def _plot_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    panel_key: str,
    *,
    show_ylabel: bool,
    show_xlabel: bool,
    y_limits: tuple[float, float],
    y_ticks: list[float],
) -> None:
    panel = data.loc[data["panel"].eq(panel_key)].copy()
    curve = panel.loc[panel["kind"].eq("curve")].sort_values("x")
    transitions = panel.loc[panel["kind"].eq("transition")].sort_values("x")

    x = curve["x"].to_numpy(dtype=float)
    y = curve["mean"].to_numpy(dtype=float)
    std = curve["std"].fillna(0).to_numpy(dtype=float)

    style = _compact_ntd_style()
    ax.fill_between(x, y - std, y + std, color=style["color"], alpha=0.10, linewidth=0, zorder=1)
    ax.plot(
        x,
        y,
        color=style["color"],
        linestyle=style["linestyle"],
        marker=None,
        linewidth=style["linewidth"],
        zorder=style["zorder"],
    )

    for item in transitions.itertuples(index=False):
        xpos = float(item.x)
        nearest = int(np.argmin(np.abs(x - xpos)))
        ax.axvline(
            xpos,
            color=PALETTE.highlight,
            linestyle="--",
            linewidth=0.95,
            alpha=0.88,
            zorder=2,
        )
        ax.plot(
            x[nearest],
            y[nearest],
            marker="o",
            color=PALETTE.highlight,
            markersize=3.4,
            zorder=5,
        )

    ax.set_title(PANEL_LABELS.get(panel_key, panel_key), pad=2.0)
    ax.set_xlim(-35, 1035)
    ax.set_ylim(*y_limits)
    ax.set_yticks(y_ticks)
    ax.set_xticks([0, 400, 800])
    ax.set_xlabel("iteration" if show_xlabel else "", labelpad=1.5)
    ax.set_ylabel("RMSE" if show_ylabel else "", labelpad=1.5)
    ax.tick_params(axis="both", pad=1.0)
    style_axes(ax, grid=True)


def main() -> None:
    apply_style("single_column")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = _collect()
    raw.to_csv(OUT_DIR / f"{OUT_STEM}_raw.csv", index=False)
    data = _aggregate_trace(raw)
    data.to_csv(OUT_DIR / f"{OUT_STEM}.csv", index=False)
    panel_order = list(PANEL_LABELS)

    curve = data.loc[data["kind"].eq("curve")].copy()
    y_min = float((curve["mean"] - curve["std"].fillna(0)).min())
    y_max = float((curve["mean"] + curve["std"].fillna(0)).max())
    y_min = max(0.0, y_min - 0.01)
    y_max = y_max + 0.01
    tick_step = 0.1
    y_ticks = np.arange(0.0, np.ceil(y_max / tick_step) * tick_step + 1e-9, tick_step).tolist()
    y_limits = (0.0, y_ticks[-1] if y_ticks else y_max)

    fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.58), sharex=True, sharey=True)
    flat_axes = axes.ravel()
    for idx, (ax, panel_key) in enumerate(zip(flat_axes, panel_order, strict=True)):
        _plot_panel(
            ax,
            data,
            panel_key,
            show_ylabel=idx == 0,
            show_xlabel=True,
            y_limits=y_limits,
            y_ticks=y_ticks,
        )

    ntd_style = _compact_ntd_style()
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=ntd_style["color"],
            linestyle=ntd_style["linestyle"],
            linewidth=ntd_style["linewidth"],
            label="NTD-PL",
        ),
        Line2D(
            [0],
            [0],
            color=PALETTE.highlight,
            linestyle="--",
            marker="o",
            linewidth=0.95,
            markersize=3.4,
            label="degree activation",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.015),
        handlelength=1.7,
        columnspacing=0.85,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.25, top=0.76, wspace=0.14)

    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{ext}", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
