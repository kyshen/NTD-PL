from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any

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
import tensorly as tl
from tensorly.tenalg.core_tenalg import multi_mode_dot
from tensorly.tucker_tensor import tucker_to_tensor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.hsi import CAVEHSIData
from src.filters import BiasFilter
from src.methods.ntdpl import NTDPLDecomposition
from src.methods.tucker import TuckerDecomposition
from src.ntdpl.core import _adam_init, _adam_step, _init_ntdpl_factors, normalize_tucker
from src.ntdpl.links import PowerLink
from src.types import LogCallback, Tensor


DEFAULT_OUTDIR = PROJECT_ROOT / "artifacts" / "results" / "transfer_learned_response_probe"


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


def _rmse(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(x, dtype=np.float32) - np.asarray(y, dtype=np.float32)) ** 2)))


def _fit_tucker(
    x: np.ndarray,
    *,
    rank: tuple[int, int, int],
    n_iter_max: int,
) -> tuple[np.ndarray, float]:
    model = TuckerDecomposition(rank=rank, n_iter_max=n_iter_max, init="svd", tol=1e-4)
    tensor = Tensor(shape=x.shape, dense=np.asarray(x, dtype=np.float32))
    start = perf_counter()
    model.fit(tensor, mask=None, logcallback=LogCallback(log_level=0))
    elapsed = perf_counter() - start
    return np.asarray(model.reconstruct().dense, dtype=np.float32), float(elapsed)


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


def _fit_fixed_beta_ntdpl(
    x: np.ndarray,
    *,
    rank: tuple[int, int, int],
    beta: np.ndarray,
    n_iter_max: int,
    init_n_iter_max: int,
    lr_core: float = 1e-4,
    lr_factors: float = 3e-4,
    lambda_core: float = 1e-6,
    lambda_factors: float = 1e-6,
    calibrate_output_affine: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float], float]:
    x = np.asarray(x, dtype=np.float32)
    beta = np.asarray(beta, dtype=np.float32)
    link = PowerLink()
    core, factors = _init_ntdpl_factors(
        X=x,
        rank=rank,
        init="tucker",
        init_n_iter_max=init_n_iter_max,
        mask_float=None,
        random_state=0,
    )
    core, factors = normalize_tucker(core, factors)
    calibration = {"output_scale": 1.0, "output_offset": 0.0}
    if calibrate_output_affine:
        signal_init = tucker_to_tensor((core, factors))
        z = np.asarray(link.value(signal_init, beta), dtype=np.float64).reshape(-1)
        y = x.astype(np.float64).reshape(-1)
        design = np.column_stack([np.ones_like(z), z])
        coeff, *_ = np.linalg.lstsq(design, y, rcond=None)
        offset, scale = float(coeff[0]), float(coeff[1])
        beta = (scale * beta).astype(np.float32)
        beta[0] = np.float32(beta[0] + offset)
        calibration = {"output_scale": scale, "output_offset": offset}
    st_core = _adam_init(core.shape)
    st_factors = [_adam_init(f.shape) for f in factors]
    fit_scale = np.float32(1.0 / x.size)
    modes_all = list(range(x.ndim))

    start = perf_counter()
    for _ in range(1, n_iter_max + 1):
        signal = tucker_to_tensor((core, factors))
        prediction, dfd_s = link.value_and_derivative(signal, beta)
        residual = (prediction - x) * fit_scale
        t = residual * dfd_s

        grad_core = multi_mode_dot(t, [f.T for f in factors], modes=modes_all)
        grad_core = grad_core.astype(np.float32) + lambda_core * core
        _adam_step(core, grad_core, st_core, b1=0.9, b2=0.999, lr=lr_core, eps=1e-8)

        for mode in range(x.ndim):
            other_modes = [k for k in range(x.ndim) if k != mode]
            m = multi_mode_dot(core, [factors[k] for k in other_modes], modes=other_modes)
            z = tl.unfold(m, mode=mode)
            tn = tl.unfold(t, mode=mode)
            grad_factor = np.dot(tn, z.T)
            grad_factor = grad_factor.astype(np.float32) + lambda_factors * factors[mode]
            _adam_step(factors[mode], grad_factor, st_factors[mode], b1=0.9, b2=0.999, lr=lr_factors, eps=1e-8)

        core, factors = normalize_tucker(core, factors)

    elapsed = perf_counter() - start
    signal = np.asarray(tucker_to_tensor((core, factors)), dtype=np.float32)
    prediction = np.asarray(link.value(signal, beta), dtype=np.float32)
    return signal, prediction, beta, calibration, float(elapsed)


def _fit_baseline_one(
    scene_id: int,
    *,
    target_shape: tuple[int, int],
    rank: tuple[int, int, int],
    p_max: int,
    n_iter_max: int,
    init_n_iter_max: int,
    tucker_iter_max: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    scene_name, cube = _load_cave_scene(scene_id, target_shape)
    tucker_pred, tucker_time = _fit_tucker(cube, rank=rank, n_iter_max=tucker_iter_max)
    signal, ntdpl_pred, beta, ntdpl_time = _fit_ntdpl(
        cube,
        rank=rank,
        p_max=p_max,
        n_iter_max=n_iter_max,
        init_n_iter_max=init_n_iter_max,
    )
    baseline = {
        "scene_id": int(scene_id),
        "scene_name": scene_name,
        "rank": str(rank),
        "p_max": int(p_max),
        "tucker_rmse": _rmse(cube, tucker_pred),
        "ntdpl_self_rmse": _rmse(cube, ntdpl_pred),
        "self_gain_vs_tucker_pct": 100.0 * (_rmse(cube, tucker_pred) - _rmse(cube, ntdpl_pred)) / max(_rmse(cube, tucker_pred), 1e-12),
        "tucker_fit_time_sec": tucker_time,
        "ntdpl_fit_time_sec": ntdpl_time,
    }
    source = {
        "source_scene_id": int(scene_id),
        "source_scene_name": scene_name,
        "beta": json.dumps([float(v) for v in beta.reshape(-1)]),
        "source_self_rmse": _rmse(cube, ntdpl_pred),
        "signal_p01": float(np.percentile(signal, 1.0)),
        "signal_p99": float(np.percentile(signal, 99.0)),
    }
    return baseline, source


def _transfer_one(
    source: dict[str, Any],
    target_scene_id: int,
    *,
    target_shape: tuple[int, int],
    rank: tuple[int, int, int],
    n_iter_max: int,
    init_n_iter_max: int,
    calibrate_output_affine: bool,
) -> dict[str, Any]:
    target_name, cube = _load_cave_scene(target_scene_id, target_shape)
    beta = np.asarray(json.loads(str(source["beta"])), dtype=np.float32)
    signal, pred, beta_used, calibration, fit_time = _fit_fixed_beta_ntdpl(
        cube,
        rank=rank,
        beta=beta,
        n_iter_max=n_iter_max,
        init_n_iter_max=init_n_iter_max,
        calibrate_output_affine=calibrate_output_affine,
    )
    return {
        "source_scene_id": int(source["source_scene_id"]),
        "source_scene_name": str(source["source_scene_name"]),
        "target_scene_id": int(target_scene_id),
        "target_scene_name": target_name,
        "transfer_rmse": _rmse(cube, pred),
        "transfer_fit_time_sec": fit_time,
        "calibrate_output_affine": bool(calibrate_output_affine),
        "output_scale": float(calibration["output_scale"]),
        "output_offset": float(calibration["output_offset"]),
        "source_beta": str(source["beta"]),
        "fixed_beta_used": json.dumps([float(v) for v in beta_used.reshape(-1)]),
        "target_signal_p01": float(np.percentile(signal, 1.0)),
        "target_signal_p99": float(np.percentile(signal, 99.0)),
    }


def _aggregate_transfer(baselines: pd.DataFrame, transfers: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    joined = transfers.merge(
        baselines[
            [
                "scene_id",
                "scene_name",
                "tucker_rmse",
                "ntdpl_self_rmse",
            ]
        ],
        left_on="target_scene_id",
        right_on="scene_id",
        how="left",
    )
    joined["transfer_gain_vs_tucker_pct"] = 100.0 * (joined["tucker_rmse"] - joined["transfer_rmse"]) / joined["tucker_rmse"].clip(lower=1e-12)
    joined["transfer_gap_to_self_pct_of_tucker_gain"] = (
        (joined["transfer_rmse"] - joined["ntdpl_self_rmse"])
        / (joined["tucker_rmse"] - joined["ntdpl_self_rmse"]).clip(lower=1e-12)
        * 100.0
    )
    summary = (
        joined.groupby(["target_scene_id", "target_scene_name"], as_index=False)
        .agg(
            tucker_rmse=("tucker_rmse", "first"),
            ntdpl_self_rmse=("ntdpl_self_rmse", "first"),
            transfer_rmse_mean=("transfer_rmse", "mean"),
            transfer_rmse_median=("transfer_rmse", "median"),
            transfer_rmse_min=("transfer_rmse", "min"),
            transfer_rmse_std=("transfer_rmse", "std"),
            transfer_gain_vs_tucker_mean_pct=("transfer_gain_vs_tucker_pct", "mean"),
            transfer_gain_vs_tucker_median_pct=("transfer_gain_vs_tucker_pct", "median"),
            transfer_gain_vs_tucker_best_pct=("transfer_gain_vs_tucker_pct", "max"),
        )
        .sort_values("target_scene_id")
    )
    best_idx = joined.groupby("target_scene_id")["transfer_rmse"].idxmin()
    best = joined.loc[best_idx, ["target_scene_id", "source_scene_id", "source_scene_name", "transfer_rmse"]]
    summary = summary.merge(best, on="target_scene_id", how="left").rename(
        columns={
            "source_scene_id": "best_source_scene_id",
            "source_scene_name": "best_source_scene_name",
            "transfer_rmse": "best_source_transfer_rmse",
        }
    )
    return joined.sort_values(["target_scene_id", "source_scene_id"]), summary


def _plot_outputs(joined: pd.DataFrame, summary: pd.DataFrame, outdir: Path) -> None:
    pivot = joined.pivot(index="target_scene_id", columns="source_scene_id", values="transfer_gain_vs_tucker_pct")
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    finite = pivot.values[np.isfinite(pivot.values)]
    if finite.size:
        lo, hi = np.percentile(finite, [5.0, 95.0])
        vmax = max(abs(float(lo)), abs(float(hi)), 1e-6)
    else:
        vmax = 1.0
    im = ax.imshow(pivot.values, cmap="RdBu", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns)
    ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
    ax.set_xlabel("source scene")
    ax.set_ylabel("target scene")
    ax.set_title("Transfer response gain vs Tucker (%)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outdir / "transfer_gain_heatmap.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = np.arange(len(summary))
    ax.bar(x - 0.2, summary["transfer_gain_vs_tucker_median_pct"], width=0.4, label="median source")
    ax.bar(x + 0.2, summary["transfer_gain_vs_tucker_best_pct"], width=0.4, label="best source")
    ax.axhline(0.0, color="#777777", linewidth=1.0)
    ax.set_xticks(x, labels=summary["target_scene_id"].astype(str))
    ax.set_xlabel("target scene")
    ax.set_ylabel("gain vs Tucker (%)")
    ax.set_title("Transferred response by target")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "transfer_gain_by_target.png", dpi=200)
    plt.close(fig)


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scene_ids = _parse_int_set(args.scene_ids)
    source_ids = _parse_int_set(args.source_ids) if args.source_ids else scene_ids
    rank = _parse_rank(args.rank)
    target_shape = _parse_shape(args.target_shape)

    baselines: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.max_workers))) as executor:
        futures = [
            executor.submit(
                _fit_baseline_one,
                scene_id,
                target_shape=target_shape,
                rank=rank,
                p_max=args.p_max,
                n_iter_max=args.n_iter_max,
                init_n_iter_max=args.init_n_iter_max,
                tucker_iter_max=args.tucker_iter_max,
            )
            for scene_id in sorted(set(scene_ids) | set(source_ids))
        ]
        for future in as_completed(futures):
            baseline, source = future.result()
            baselines.append(baseline)
            sources.append(source)

    baseline_df = pd.DataFrame(baselines).sort_values("scene_id")
    source_df = pd.DataFrame(sources).sort_values("source_scene_id")
    source_records = [row for row in source_df.to_dict("records") if int(row["source_scene_id"]) in set(source_ids)]

    transfer_rows: list[dict[str, Any]] = []
    jobs = [
        (source, target_id)
        for source in source_records
        for target_id in scene_ids
        if bool(args.include_self_transfer) or int(source["source_scene_id"]) != int(target_id)
    ]
    with ProcessPoolExecutor(max_workers=max(1, int(args.max_workers))) as executor:
        futures = [
            executor.submit(
                _transfer_one,
                source,
                target_id,
                target_shape=target_shape,
                rank=rank,
                n_iter_max=args.transfer_iter_max,
                init_n_iter_max=args.init_n_iter_max,
                calibrate_output_affine=bool(args.calibrate_output_affine),
            )
            for source, target_id in jobs
        ]
        for future in as_completed(futures):
            transfer_rows.append(future.result())

    transfer_df = pd.DataFrame(transfer_rows)
    joined_df, summary_df = _aggregate_transfer(baseline_df, transfer_df)
    return baseline_df, joined_df, summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe whether learned NTD-PL response curves transfer across CAVE scenes.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--scene-ids", default="1-15")
    parser.add_argument("--source-ids", default="", help="Defaults to --scene-ids.")
    parser.add_argument("--rank", default="4,4,2")
    parser.add_argument("--target-shape", default="128,128")
    parser.add_argument("--p-max", type=int, default=6)
    parser.add_argument("--n-iter-max", type=int, default=150)
    parser.add_argument("--transfer-iter-max", type=int, default=150)
    parser.add_argument("--tucker-iter-max", type=int, default=150)
    parser.add_argument("--init-n-iter-max", type=int, default=50)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--include-self-transfer", action="store_true")
    parser.add_argument("--calibrate-output-affine", action="store_true")
    args = parser.parse_args()

    outdir = args.outdir if args.outdir.is_absolute() else PROJECT_ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    baseline_df, transfer_df, summary_df = run(args)
    baseline_df.to_csv(outdir / "target_baselines.csv", index=False)
    transfer_df.to_csv(outdir / "transfer_pairs.csv", index=False)
    summary_df.to_csv(outdir / "transfer_summary_by_target.csv", index=False)
    _plot_outputs(transfer_df, summary_df, outdir)

    print("Baselines:")
    print(baseline_df[["scene_id", "scene_name", "tucker_rmse", "ntdpl_self_rmse", "self_gain_vs_tucker_pct"]].to_string(index=False))
    print("\nTransfer summary:")
    print(
        summary_df[
            [
                "target_scene_id",
                "target_scene_name",
                "transfer_gain_vs_tucker_median_pct",
                "transfer_gain_vs_tucker_best_pct",
                "best_source_scene_id",
            ]
        ].to_string(index=False)
    )
    print("\nOverall:")
    print(
        transfer_df[
            [
                "transfer_rmse",
                "transfer_gain_vs_tucker_pct",
                "transfer_gap_to_self_pct_of_tucker_gain",
            ]
        ].describe().to_string()
    )
    print(f"\nOutput: {outdir}")


if __name__ == "__main__":
    main()
