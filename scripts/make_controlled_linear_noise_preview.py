from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.tucker import TuckerData
from src.filters.nonlinear import apply_exp_response, apply_poly3, apply_tanh
from src.methods.cp import CPDecomposition
from src.methods.ntdpl import NTDPLDecomposition
from src.methods.tt import TTDecomposition
from src.methods.tucker import TuckerDecomposition
from src.metrics import val_RMSE
from src.types import LogCallback, Tensor
from viz.style import apply_style, method_style, style_axes


OUT_DIR = ROOT / "papers" / "tsp" / "figures"
OUT_STEM = "controlled_nonlinear_alpha_grid"
METHOD_ORDER = ("Tucker", "CP", "TT", "NTD-PL")
ALPHA_LEVELS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
SEEDS = tuple(range(10))
SNR_DB = 30.0

PANEL_LABELS = {
    "linear": r"$s$",
    "poly3": r"$s^2+s^3$",
    "tanh": r"$\tanh(\kappa s)$",
    "exp": r"$(e^{\kappa s}-1)/\kappa$",
}


def _fit_rmse(
    method: str,
    target: np.ndarray,
    rank: tuple[int, int, int],
    seed: int,
    *,
    p_max: int = 6,
) -> float:
    tensor = Tensor(shape=target.shape, dense=target.astype(np.float32))
    log = LogCallback(log_level=0)
    if method == "Tucker":
        model = TuckerDecomposition(rank=rank, n_iter_max=1000, init="svd", tol=1e-4)
    elif method == "CP":
        model = CPDecomposition(
            rank=rank,
            n_iter_max=1000,
            cp_rank=None,
            init_method="random",
            tol=1e-8,
            random_state=seed,
            normalize_factors=False,
        )
    elif method == "TT":
        model = TTDecomposition(rank=rank, n_iter_max=1000, tt_rank=None, svd="truncated_svd")
    elif method == "NTD-PL":
        model = NTDPLDecomposition(
            rank=rank,
            init_n_iter_max=50,
            init="tucker",
            stable_beta_update=True,
            beta_update_stage="before_grad",
            random_state=seed,
            p_max=int(p_max),
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
    else:
        raise ValueError(method)
    model.fit(tensor, None, log)
    return val_RMSE(tensor, model.reconstruct())


def _add_noise(target: np.ndarray, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    signal_power = float(np.mean(target**2))
    noise_power = signal_power / (10.0 ** (SNR_DB / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=target.shape).astype(np.float32)
    return np.asarray(target, dtype=np.float32) + noise


def _response(signal: np.ndarray, panel: str, alpha: float) -> np.ndarray:
    if panel == "linear":
        return signal
    if panel == "poly3":
        return apply_poly3(signal, alpha)
    if panel == "tanh":
        return apply_tanh(signal, alpha)
    if panel == "exp":
        return apply_exp_response(signal, alpha)
    raise ValueError(panel)


def _noisy_response_runs() -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    shape = (10, 10, 10)
    rank = (4, 4, 4)
    panel_order = ("linear", "poly3", "tanh", "exp")
    for seed in SEEDS:
        data = TuckerData(shape=shape, rank=rank, seed=seed)
        signal = np.asarray(data.get("fit").dense, dtype=np.float32)
        signal = signal / (np.linalg.norm(signal) + 1e-8) * np.sqrt(signal.size)
        for panel in panel_order:
            for alpha in ALPHA_LEVELS:
                clean = _response(signal, panel, float(alpha))
                noise_seed = 10_000 + 1_000 * seed + 37 * panel_order.index(panel)
                if panel != "linear":
                    noise_seed += int(round(float(alpha) * 1000))
                target = _add_noise(clean, seed=noise_seed)
                for method in METHOD_ORDER:
                    rmse = _fit_rmse(method, target, rank, seed)
                    rows.append(
                        {
                            "panel": panel,
                            "method": method,
                            "x": float(alpha),
                            "seed": int(seed),
                            "rmse": float(rmse),
                        }
                    )
    return pd.DataFrame(rows)


def _summarize(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(["panel", "method", "x"], as_index=False)["rmse"].agg(["mean", "std"]).reset_index()
    grouped["std"] = grouped["std"].fillna(0.0)
    grouped["band_lower"] = grouped["mean"] - grouped["std"]
    grouped["band_upper"] = grouped["mean"] + grouped["std"]
    return grouped


def _compact_method_style(method: str) -> dict:
    style = method_style(method)
    style.update({"linewidth": 1.25, "markersize": 3.0})
    if method == "NTD-PL":
        style.update({"linewidth": 1.85, "markersize": 3.5, "zorder": 4})
    elif method == "Tucker":
        style.update({"linewidth": 1.45, "zorder": 3})
    else:
        style.update({"alpha": 0.88, "zorder": 2})
    return style


def _plot_panel(
    ax: plt.Axes,
    panel_data: pd.DataFrame,
    panel_key: str,
    *,
    show_ylabel: bool,
    y_limits: tuple[float, float],
    y_ticks: list[float],
) -> None:
    for method in METHOD_ORDER:
        sub = panel_data.loc[panel_data["method"].eq(method)].sort_values("x")
        if sub.empty:
            continue
        style = _compact_method_style(method)
        x = sub["x"].to_numpy(dtype=float)
        y = sub["mean"].to_numpy(dtype=float)
        if method == "NTD-PL":
            ax.fill_between(
                x,
                sub["band_lower"].to_numpy(dtype=float),
                sub["band_upper"].to_numpy(dtype=float),
                color=style["color"],
                alpha=0.10,
                linewidth=0,
                zorder=1,
            )
        ax.plot(
            x,
            y,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style.get("marker"),
            linewidth=style["linewidth"],
            markersize=style.get("markersize", 2.5),
            alpha=style.get("alpha", 1.0),
            zorder=style.get("zorder", 2),
            label=method,
        )

    ax.set_title(PANEL_LABELS.get(panel_key, panel_key), pad=2.0)
    ax.set_xticks([0.10, 0.20, 0.30, 0.40])
    ax.set_xticklabels([".10", ".20", ".30", ".40"])
    ax.set_ylim(*y_limits)
    ax.set_yticks(y_ticks)
    ax.set_xlabel(r"residual energy $\alpha$", labelpad=1.5)
    ax.set_ylabel("RMSE" if show_ylabel else "", labelpad=1.5)
    ax.tick_params(axis="both", pad=1.0)
    style_axes(ax, grid=True)

def main() -> None:
    apply_style("single_column")
    data = _summarize(_noisy_response_runs())

    y_min = max(0.0, float(data["band_lower"].min()) - 0.01)
    y_max = float(data["band_upper"].max()) + 0.01
    tick_step = 0.1
    y_ticks = np.arange(0.0, np.ceil(y_max / tick_step) * tick_step + 1e-9, tick_step).tolist()
    y_limits = (y_min, y_ticks[-1] if y_ticks else y_max)

    panel_order = ["linear", "poly3", "tanh", "exp"]
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.58), sharex=True, sharey=True)
    flat_axes = axes.ravel()
    for idx, (ax, panel_key) in enumerate(zip(flat_axes, panel_order, strict=True)):
        _plot_panel(
            ax,
            data.loc[data["panel"].eq(panel_key)].copy(),
            panel_key,
            show_ylabel=idx == 0,
            y_limits=y_limits,
            y_ticks=y_ticks,
        )

    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.015),
        handlelength=1.5,
        columnspacing=0.65,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.25, top=0.76, wspace=0.14)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT_DIR / f"{OUT_STEM}.csv", index=False)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{ext}", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
