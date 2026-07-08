from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

for _thread_key in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "TBB_NUM_THREADS",
):
    os.environ.setdefault(_thread_key, "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorly.tucker_tensor import tucker_to_tensor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.hsi import CAVEHSIData
from src.filters import BiasFilter
from src.methods.ntdpl import NTDPLDecomposition
from src.ntdpl.links import PowerLink
from src.types import LogCallback, Tensor


DEFAULT_OUTDIR = PROJECT_ROOT / "artifacts" / "results" / "learned_response_curve_probe"


def _parse_int_set(text: str) -> list[int]:
    out: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(v.strip()) for v in part.split("-", 1)]
            out.update(range(start, end + 1))
        else:
            out.add(int(part))
    return sorted(out)


def _parse_rank(text: str) -> tuple[int, int, int]:
    values = [int(part.strip()) for part in text.replace("x", ",").split(",") if part.strip()]
    if len(values) != 3:
        raise ValueError(f"Expected three rank entries, got {text!r}.")
    return values[0], values[1], values[2]


def _parse_shape(text: str) -> tuple[int, int]:
    values = [int(part.strip()) for part in text.replace("x", ",").split(",") if part.strip()]
    if len(values) != 2:
        raise ValueError(f"Expected two shape entries, got {text!r}.")
    return values[0], values[1]


def _load_cave_scene(scene_id: int, target_shape: tuple[int, int]) -> tuple[str, np.ndarray]:
    dataset = CAVEHSIData(path="data/CAVE", id=int(scene_id), target_shape=target_shape, crop_shape=None)
    BiasFilter(seed=0, normalize_method="max", snr_db=None, bias=None)(dataset)
    scene_name = str(getattr(dataset, "scene_name", f"scene_{scene_id:02d}"))
    cube = np.asarray(dataset.get("eval").dense, dtype=np.float32)
    return scene_name, cube


def _fit_ntdpl(
    x: np.ndarray,
    *,
    rank: tuple[int, int, int],
    p_max: int,
    n_iter_max: int,
    init_n_iter_max: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    model = NTDPLDecomposition(
        rank=rank,
        init_n_iter_max=init_n_iter_max,
        p_max=p_max,
        allow_constant_term=True,
        n_iter_max=n_iter_max,
        use_continuation=True,
        factor_normalize=True,
        lr_core=1e-4,
        lr_factors=3e-4,
        lambda_core=1e-6,
        lambda_factors=1e-6,
        lambda_beta=1e-6,
        beta_update_method="ridge_lstsq",
        init="tucker",
        random_state=0,
        beta_update_interval=5,
        solver_variant="optimized",
        stable_beta_update=True,
        beta_update_stage="before_grad",
        link_kind="power",
    )
    tensor = Tensor(shape=x.shape, dense=np.asarray(x, dtype=np.float32))
    start = perf_counter()
    model.fit(tensor, mask=None, logcallback=LogCallback(log_level=0))
    elapsed = perf_counter() - start
    signal = np.asarray(tucker_to_tensor((model.core, model.factors)), dtype=np.float32)
    prediction = np.asarray(model.reconstruct().dense, dtype=np.float32)
    beta = np.asarray(model.beta, dtype=np.float32)
    return signal, prediction, beta, float(elapsed)


def _poly_value(grid: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return PowerLink().value(np.asarray(grid, dtype=np.float32), np.asarray(beta, dtype=np.float32))


def _poly_derivative(grid: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return PowerLink().derivative(np.asarray(grid, dtype=np.float32), np.asarray(beta, dtype=np.float32))


def _curve_diagnostics(grid: np.ndarray, values: np.ndarray, derivative: np.ndarray) -> dict[str, float]:
    x = np.asarray(grid, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    dy = np.asarray(derivative, dtype=np.float64)
    affine = np.polyval(np.polyfit(x, y, deg=1), x)
    denom = float(np.sqrt(np.mean((y - np.mean(y)) ** 2)))
    nonlin = float(np.sqrt(np.mean((y - affine) ** 2)) / max(denom, 1e-12))
    orientation = 1.0 if float(y[-1] - y[0]) >= 0.0 else -1.0
    oriented_dy = orientation * dy
    monotone_frac = float(np.mean(oriented_dy > 0.0))
    slope_p05 = float(np.percentile(oriented_dy, 5))
    slope_p50 = float(np.percentile(oriented_dy, 50))
    slope_p95 = float(np.percentile(oriented_dy, 95))
    endpoint_slope_ratio = (
        float(oriented_dy[-1] / oriented_dy[0]) if abs(float(oriented_dy[0])) > 1e-12 else float("nan")
    )
    dx = np.diff(x)
    dy_grid = np.diff(y)
    local_slope = dy_grid / np.maximum(dx, 1e-12)
    curvature = np.diff(local_slope) / np.maximum(0.5 * (dx[1:] + dx[:-1]), 1e-12)
    slope_scale = max(float(np.percentile(np.abs(local_slope), 95)), 1e-12)
    return {
        "nonlinear_deviation": nonlin,
        "monotone_fraction": monotone_frac,
        "slope_p05": slope_p05,
        "slope_p50": slope_p50,
        "slope_p95": slope_p95,
        "slope_dynamic_range": float((slope_p95 - slope_p05) / max(abs(slope_p50), 1e-12)),
        "endpoint_slope_ratio": endpoint_slope_ratio,
        "normalized_mean_abs_curvature": float(np.mean(np.abs(curvature)) / slope_scale) if curvature.size else 0.0,
    }


def _normalized_curve(grid: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(grid, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    x_norm = (x - x[0]) / max(float(x[-1] - x[0]), 1e-12)
    y_norm = (y - y[0]) / max(float(y[-1] - y[0]), 1e-12)
    return x_norm, y_norm


def _cave_one(
    scene_id: int,
    *,
    target_shape: tuple[int, int],
    rank: tuple[int, int, int],
    p_max: int,
    n_iter_max: int,
    init_n_iter_max: int,
    grid_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scene_name, y = _load_cave_scene(scene_id, target_shape)
    signal, prediction, beta, fit_time = _fit_ntdpl(
        y,
        rank=rank,
        p_max=p_max,
        n_iter_max=n_iter_max,
        init_n_iter_max=init_n_iter_max,
    )
    s = signal.reshape(-1).astype(np.float64)
    grid = np.quantile(s, np.linspace(0.01, 0.99, grid_size)).astype(np.float32)
    values = _poly_value(grid, beta)
    derivative = _poly_derivative(grid, beta)
    x_norm, y_norm = _normalized_curve(grid, values)
    diag = _curve_diagnostics(grid, values, derivative)
    rmse = float(np.sqrt(np.mean((prediction - y) ** 2)))

    curve_rows = [
        {
            "experiment": "cave",
            "scene_id": int(scene_id),
            "scene_name": scene_name,
            "grid_index": int(i),
            "s": float(grid[i]),
            "f": float(values[i]),
            "dfds": float(derivative[i]),
            "s_norm": float(x_norm[i]),
            "f_norm": float(y_norm[i]),
            "identity_norm": float(x_norm[i]),
        }
        for i in range(grid_size)
    ]
    summary = {
        "experiment": "cave",
        "scene_id": int(scene_id),
        "scene_name": scene_name,
        "rank": str(rank),
        "p_max": int(p_max),
        "fit_time_sec": fit_time,
        "rmse": rmse,
        "s_min_p01": float(grid[0]),
        "s_max_p99": float(grid[-1]),
        "beta": json.dumps([float(v) for v in beta.reshape(-1)]),
        **diag,
    }
    return summary, curve_rows


def _random_tucker_signal(shape: tuple[int, int, int], rank: tuple[int, int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    factors = []
    for dim, r in zip(shape, rank):
        q, _ = np.linalg.qr(rng.normal(size=(dim, r)))
        factors.append(q[:, :r].astype(np.float32))
    core = rng.normal(size=rank).astype(np.float32)
    signal = np.asarray(tucker_to_tensor((core, factors)), dtype=np.float32)
    signal = (signal - signal.mean()) / max(float(signal.std()), 1e-8)
    return signal.astype(np.float32)


def _response_function(name: str) -> Callable[[np.ndarray], np.ndarray]:
    if name == "square":
        return lambda x: x**2
    if name == "poly23":
        return lambda x: x**2 + 0.5 * x**3
    if name == "tanh":
        return lambda x: np.tanh(1.4 * x)
    if name == "exp":
        return lambda x: np.expm1(0.7 * x) / 0.7
    raise ValueError(f"Unknown response {name!r}.")


def _affine_align(source: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    x = np.asarray(source, dtype=np.float64).reshape(-1)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    a, b = np.polyfit(x, y, deg=1)
    return float(a), float(b)


def _controlled_one(
    response_name: str,
    seed: int,
    *,
    shape: tuple[int, int, int],
    rank: tuple[int, int, int],
    p_max: int,
    n_iter_max: int,
    init_n_iter_max: int,
    grid_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    true_signal = _random_tucker_signal(shape, rank, seed)
    response = _response_function(response_name)
    y = response(true_signal).astype(np.float32)
    y = (y - y.min()) / max(float(y.max() - y.min()), 1e-8)
    signal, prediction, beta, fit_time = _fit_ntdpl(
        y,
        rank=rank,
        p_max=p_max,
        n_iter_max=n_iter_max,
        init_n_iter_max=init_n_iter_max,
    )
    a, b = _affine_align(signal, true_signal)
    grid = np.quantile(signal.reshape(-1), np.linspace(0.01, 0.99, grid_size)).astype(np.float32)
    learned = _poly_value(grid, beta)
    derivative = _poly_derivative(grid, beta)
    true_on_learned = response(a * grid + b)
    true_on_learned = (true_on_learned - true_on_learned.min()) / max(float(true_on_learned.max() - true_on_learned.min()), 1e-8)
    learned_norm = (learned - learned.min()) / max(float(learned.max() - learned.min()), 1e-8)
    curve_rmse = float(np.sqrt(np.mean((learned_norm - true_on_learned) ** 2)))
    latent_corr = float(np.corrcoef(signal.reshape(-1), true_signal.reshape(-1))[0, 1])
    pred_rmse = float(np.sqrt(np.mean((prediction - y) ** 2)))
    x_norm, y_norm = _normalized_curve(grid, learned)
    diag = _curve_diagnostics(grid, learned, derivative)
    curve_rows = [
        {
            "experiment": "controlled",
            "response": response_name,
            "seed": int(seed),
            "grid_index": int(i),
            "s": float(grid[i]),
            "f": float(learned[i]),
            "dfds": float(derivative[i]),
            "s_norm": float(x_norm[i]),
            "f_norm": float(y_norm[i]),
            "identity_norm": float(x_norm[i]),
            "true_f_aligned_norm": float(true_on_learned[i]),
            "learned_f_norm_minmax": float(learned_norm[i]),
        }
        for i in range(grid_size)
    ]
    summary = {
        "experiment": "controlled",
        "response": response_name,
        "seed": int(seed),
        "rank": str(rank),
        "p_max": int(p_max),
        "fit_time_sec": fit_time,
        "prediction_rmse": pred_rmse,
        "curve_rmse_after_affine_latent_align": curve_rmse,
        "latent_corr": latent_corr,
        "abs_latent_corr": abs(latent_corr),
        "beta": json.dumps([float(v) for v in beta.reshape(-1)]),
        **diag,
    }
    return summary, curve_rows


def run_cave(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    scene_ids = _parse_int_set(args.scene_ids)
    rank = _parse_rank(args.rank)
    target_shape = _parse_shape(args.target_shape)
    summaries: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.max_workers))) as executor:
        futures = [
            executor.submit(
                _cave_one,
                scene_id,
                target_shape=target_shape,
                rank=rank,
                p_max=args.p_max,
                n_iter_max=args.n_iter_max,
                init_n_iter_max=args.init_n_iter_max,
                grid_size=args.grid_size,
            )
            for scene_id in scene_ids
        ]
        for future in as_completed(futures):
            summary, curve_rows = future.result()
            summaries.append(summary)
            curves.extend(curve_rows)
    return pd.DataFrame(summaries).sort_values("scene_id"), pd.DataFrame(curves).sort_values(["scene_id", "grid_index"])


def run_controlled(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    rank = _parse_rank(args.controlled_rank)
    shape_values = [int(part.strip()) for part in args.controlled_shape.replace("x", ",").split(",") if part.strip()]
    if len(shape_values) != 3:
        raise ValueError("Expected --controlled-shape with three entries.")
    shape = (shape_values[0], shape_values[1], shape_values[2])
    responses = [item.strip() for item in args.responses.split(",") if item.strip()]
    seeds = _parse_int_set(args.seeds)
    summaries: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    jobs = [(response, seed) for response in responses for seed in seeds]
    with ProcessPoolExecutor(max_workers=max(1, int(args.max_workers))) as executor:
        futures = [
            executor.submit(
                _controlled_one,
                response,
                seed,
                shape=shape,
                rank=rank,
                p_max=args.p_max,
                n_iter_max=args.n_iter_max,
                init_n_iter_max=args.init_n_iter_max,
                grid_size=args.grid_size,
            )
            for response, seed in jobs
        ]
        for future in as_completed(futures):
            summary, curve_rows = future.result()
            summaries.append(summary)
            curves.extend(curve_rows)
    return (
        pd.DataFrame(summaries).sort_values(["response", "seed"]),
        pd.DataFrame(curves).sort_values(["response", "seed", "grid_index"]),
    )


def _plot_cave(summary: pd.DataFrame, curves: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    for _, scene_curve in curves.groupby("scene_id"):
        ax.plot(scene_curve["s_norm"], scene_curve["f_norm"], color="#2d6cdf", alpha=0.25, linewidth=1.0)
    mean_curve = curves.groupby("grid_index", as_index=False)[["s_norm", "f_norm"]].mean()
    ax.plot(mean_curve["s_norm"], mean_curve["f_norm"], color="#111111", linewidth=2.2, label="mean learned curve")
    ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1.2, label="identity")
    ax.set_xlabel("latent signal percentile scale")
    ax.set_ylabel("response percentile scale")
    ax.set_title("CAVE learned response curves")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "cave_learned_response_curves.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5))
    axes[0].hist(summary["nonlinear_deviation"], bins=8, color="#2d6cdf", alpha=0.85)
    axes[0].set_title("Nonlinear deviation")
    axes[0].set_xlabel("RMS deviation from best affine")
    axes[0].set_ylabel("scenes")
    axes[1].hist(summary["monotone_fraction"], bins=8, color="#db7c26", alpha=0.85)
    axes[1].set_title("Monotonicity")
    axes[1].set_xlabel("fraction with positive derivative")
    fig.tight_layout()
    fig.savefig(outdir / "cave_response_diagnostics.png", dpi=200)
    plt.close(fig)


def _plot_controlled(summary: pd.DataFrame, curves: pd.DataFrame, outdir: Path) -> None:
    responses = sorted(curves["response"].unique().tolist())
    fig, axes = plt.subplots(1, len(responses), figsize=(4.2 * len(responses), 3.7), squeeze=False)
    for ax, response in zip(axes[0], responses):
        subset = curves[curves["response"].eq(response)]
        for _, run_curve in subset.groupby("seed"):
            ax.plot(run_curve["s_norm"], run_curve["learned_f_norm_minmax"], color="#2d6cdf", alpha=0.35, linewidth=1.0)
        mean_curve = subset.groupby("grid_index", as_index=False)[["s_norm", "learned_f_norm_minmax", "true_f_aligned_norm"]].mean()
        ax.plot(mean_curve["s_norm"], mean_curve["learned_f_norm_minmax"], color="#111111", linewidth=2.0, label="learned")
        ax.plot(mean_curve["s_norm"], mean_curve["true_f_aligned_norm"], color="#db3a34", linestyle="--", linewidth=1.8, label="true aligned")
        ax.set_title(response)
        ax.set_xlabel("learned latent scale")
        ax.set_ylabel("normalized response")
        ax.grid(True, alpha=0.25)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(outdir / "controlled_learned_response_curves.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe learned response curves for NTD-PL.")
    parser.add_argument("--mode", choices=["cave", "controlled", "both"], default="both")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--grid-size", type=int, default=101)
    parser.add_argument("--rank", default="4,4,2")
    parser.add_argument("--target-shape", default="128,128")
    parser.add_argument("--scene-ids", default="1-15")
    parser.add_argument("--controlled-rank", default="4,4,3")
    parser.add_argument("--controlled-shape", default="48,48,16")
    parser.add_argument("--responses", default="square,poly23,tanh,exp")
    parser.add_argument("--seeds", default="0-4")
    parser.add_argument("--p-max", type=int, default=6)
    parser.add_argument("--n-iter-max", type=int, default=180)
    parser.add_argument("--init-n-iter-max", type=int, default=50)
    args = parser.parse_args()

    outdir = args.outdir if args.outdir.is_absolute() else PROJECT_ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    if args.mode in {"cave", "both"}:
        cave_summary, cave_curves = run_cave(args)
        cave_summary.to_csv(outdir / "cave_response_summary.csv", index=False)
        cave_curves.to_csv(outdir / "cave_response_curves.csv", index=False)
        _plot_cave(cave_summary, cave_curves, outdir)
        print("CAVE summary:")
        print(cave_summary.describe(include="all").to_string())

    if args.mode in {"controlled", "both"}:
        controlled_summary, controlled_curves = run_controlled(args)
        controlled_summary.to_csv(outdir / "controlled_response_summary.csv", index=False)
        controlled_curves.to_csv(outdir / "controlled_response_curves.csv", index=False)
        _plot_controlled(controlled_summary, controlled_curves, outdir)
        print("\nControlled summary by response:")
        print(
            controlled_summary.groupby("response")[
                [
                    "prediction_rmse",
                    "curve_rmse_after_affine_latent_align",
                    "latent_corr",
                    "abs_latent_corr",
                    "nonlinear_deviation",
                    "monotone_fraction",
                ]
            ].mean().to_string()
        )

    print(f"\nOutput: {outdir}")


if __name__ == "__main__":
    main()
