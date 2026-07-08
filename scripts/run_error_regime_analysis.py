from __future__ import annotations

import argparse
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.hsi import CAVEHSIData
from src.filters import BiasFilter
from src.methods.ntdpl import NTDPLDecomposition
from src.methods.tucker import TuckerDecomposition
from src.types import LogCallback, Tensor


DEFAULT_OUTDIR = PROJECT_ROOT / "artifacts" / "results" / "error_regime_analysis"
FIXED_INTENSITY_EDGES = (0.0, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.000001)


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


def _fit_tucker(
    x: np.ndarray,
    *,
    rank: tuple[int, int, int],
    n_iter_max: int,
) -> tuple[np.ndarray, dict[str, float]]:
    model = TuckerDecomposition(rank=rank, n_iter_max=n_iter_max, init="svd", tol=1e-4)
    tensor = Tensor(shape=x.shape, dense=np.asarray(x, dtype=np.float32))
    start = perf_counter()
    model.fit(tensor, mask=None, logcallback=LogCallback(log_level=0))
    elapsed = perf_counter() - start
    return np.asarray(model.reconstruct().dense, dtype=np.float32), {"fit_time_sec": float(elapsed)}


def _fit_ntdpl(
    x: np.ndarray,
    *,
    rank: tuple[int, int, int],
    p_max: int,
    n_iter_max: int,
    init_n_iter_max: int,
) -> tuple[np.ndarray, dict[str, float]]:
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
    return np.asarray(model.reconstruct().dense, dtype=np.float32), {"fit_time_sec": float(elapsed)}


def _rmse_from_residual(residual: np.ndarray) -> float:
    if residual.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.asarray(residual, dtype=np.float64) ** 2)))


def _subset_metrics(
    y: np.ndarray,
    tucker_pred: np.ndarray,
    ntdpl_pred: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int]:
    m = np.asarray(mask, dtype=bool)
    count = int(np.sum(m))
    if count == 0:
        return {
            "count": 0,
            "mean_intensity": float("nan"),
            "tucker_mse": float("nan"),
            "ntdpl_mse": float("nan"),
            "mse_reduction": float("nan"),
            "tucker_rmse": float("nan"),
            "ntdpl_rmse": float("nan"),
            "rmse_gain_pct": float("nan"),
            "tucker_bias": float("nan"),
            "ntdpl_bias": float("nan"),
            "abs_error_gain_pct": float("nan"),
        }
    t_res = np.asarray(tucker_pred - y, dtype=np.float64)[m]
    n_res = np.asarray(ntdpl_pred - y, dtype=np.float64)[m]
    t_mse = float(np.mean(t_res**2))
    n_mse = float(np.mean(n_res**2))
    t_rmse = _rmse_from_residual(t_res)
    n_rmse = _rmse_from_residual(n_res)
    t_abs = float(np.mean(np.abs(t_res)))
    n_abs = float(np.mean(np.abs(n_res)))
    return {
        "count": count,
        "mean_intensity": float(np.mean(y[m])),
        "tucker_mse": t_mse,
        "ntdpl_mse": n_mse,
        "mse_reduction": float(t_mse - n_mse),
        "tucker_rmse": t_rmse,
        "ntdpl_rmse": n_rmse,
        "rmse_gain_pct": float(100.0 * (t_rmse - n_rmse) / max(t_rmse, 1e-12)),
        "tucker_bias": float(np.mean(t_res)),
        "ntdpl_bias": float(np.mean(n_res)),
        "abs_error_gain_pct": float(100.0 * (t_abs - n_abs) / max(t_abs, 1e-12)),
    }


def _band_wavelengths(num_bands: int) -> np.ndarray:
    if num_bands == 31:
        return np.linspace(400.0, 700.0, num_bands)
    return np.arange(num_bands, dtype=float)


def _spectral_regime(band_index0: int, num_bands: int) -> tuple[str, int]:
    if num_bands == 31:
        wavelength = float(_band_wavelengths(num_bands)[band_index0])
        if wavelength <= 500.0:
            return "blue_400_500", 0
        if wavelength <= 600.0:
            return "green_510_600", 1
        return "red_610_700", 2
    split = np.array_split(np.arange(num_bands), 3)
    labels = ("low_bands", "mid_bands", "high_bands")
    for idx, values in enumerate(split):
        if band_index0 in set(int(v) for v in values):
            return labels[idx], idx
    return labels[-1], 2


def _bin_label(left: float, right: float, index: int) -> str:
    return f"{index:02d}:{left:.3g}-{right:.3g}"


def _quantile_edges(values: np.ndarray, num_bins: int) -> np.ndarray:
    edges = np.quantile(values.reshape(-1), np.linspace(0.0, 1.0, num_bins + 1))
    edges = np.asarray(edges, dtype=np.float64)
    edges[0] -= 1e-12
    edges[-1] += 1e-12
    return edges


def _intensity_rows(
    y: np.ndarray,
    tucker_pred: np.ndarray,
    ntdpl_pred: np.ndarray,
    *,
    scene_id: int,
    scene_name: str,
    bin_kind: str,
    edges: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(len(edges) - 1):
        left = float(edges[i])
        right = float(edges[i + 1])
        if i == len(edges) - 2:
            mask = (y >= left) & (y <= right)
        else:
            mask = (y >= left) & (y < right)
        rows.append(
            {
                "scene_id": int(scene_id),
                "scene_name": scene_name,
                "bin_kind": bin_kind,
                "bin_index": int(i),
                "bin_label": _bin_label(left, right, i),
                "bin_left": left,
                "bin_right": right,
                **_subset_metrics(y, tucker_pred, ntdpl_pred, mask),
            }
        )
    return rows


def _spectral_rows(
    y: np.ndarray,
    tucker_pred: np.ndarray,
    ntdpl_pred: np.ndarray,
    *,
    scene_id: int,
    scene_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    band_rows: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []
    num_bands = int(y.shape[-1])
    wavelengths = _band_wavelengths(num_bands)
    for b in range(num_bands):
        regime, regime_index = _spectral_regime(b, num_bands)
        mask = np.zeros_like(y, dtype=bool)
        mask[..., b] = True
        band_rows.append(
            {
                "scene_id": int(scene_id),
                "scene_name": scene_name,
                "band_index": int(b + 1),
                "wavelength_nm": float(wavelengths[b]),
                "regime": regime,
                "regime_index": int(regime_index),
                **_subset_metrics(y, tucker_pred, ntdpl_pred, mask),
            }
        )
    for regime_index in range(3):
        band_mask = np.array([_spectral_regime(b, num_bands)[1] == regime_index for b in range(num_bands)])
        regime = _spectral_regime(int(np.flatnonzero(band_mask)[0]), num_bands)[0]
        mask = np.broadcast_to(band_mask.reshape((1, 1, num_bands)), y.shape)
        regime_rows.append(
            {
                "scene_id": int(scene_id),
                "scene_name": scene_name,
                "regime": regime,
                "regime_index": int(regime_index),
                "band_start": int(np.flatnonzero(band_mask)[0] + 1),
                "band_end": int(np.flatnonzero(band_mask)[-1] + 1),
                **_subset_metrics(y, tucker_pred, ntdpl_pred, mask),
            }
        )
    return band_rows, regime_rows


def _interaction_rows(
    y: np.ndarray,
    tucker_pred: np.ndarray,
    ntdpl_pred: np.ndarray,
    *,
    scene_id: int,
    scene_name: str,
    quantile_edges: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    num_bands = int(y.shape[-1])
    for i in range(len(quantile_edges) - 1):
        left = float(quantile_edges[i])
        right = float(quantile_edges[i + 1])
        intensity_mask = (y >= left) & (y <= right) if i == len(quantile_edges) - 2 else (y >= left) & (y < right)
        for regime_index in range(3):
            band_mask = np.array([_spectral_regime(b, num_bands)[1] == regime_index for b in range(num_bands)])
            regime = _spectral_regime(int(np.flatnonzero(band_mask)[0]), num_bands)[0]
            spectral_mask = np.broadcast_to(band_mask.reshape((1, 1, num_bands)), y.shape)
            rows.append(
                {
                    "scene_id": int(scene_id),
                    "scene_name": scene_name,
                    "quantile_bin_index": int(i),
                    "quantile_bin_label": _bin_label(left, right, i),
                    "regime": regime,
                    "regime_index": int(regime_index),
                    **_subset_metrics(y, tucker_pred, ntdpl_pred, intensity_mask & spectral_mask),
                }
            )
    return rows


def _scene_analysis(
    scene_id: int,
    *,
    target_shape: tuple[int, int],
    rank: tuple[int, int, int],
    p_max: int,
    n_iter_max: int,
    init_n_iter_max: int,
    tucker_iter_max: int,
    num_quantile_bins: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    scene_name, y = _load_cave_scene(scene_id, target_shape)
    tucker_pred, tucker_info = _fit_tucker(y, rank=rank, n_iter_max=tucker_iter_max)
    ntdpl_pred, ntdpl_info = _fit_ntdpl(y, rank=rank, p_max=p_max, n_iter_max=n_iter_max, init_n_iter_max=init_n_iter_max)
    overall_metrics = _subset_metrics(y, tucker_pred, ntdpl_pred, np.ones_like(y, dtype=bool))
    overall = {
        "scene_id": int(scene_id),
        "scene_name": scene_name,
        "rank": str(rank),
        "p_max": int(p_max),
        "target_shape": str(target_shape),
        "tucker_fit_time_sec": float(tucker_info["fit_time_sec"]),
        "ntdpl_fit_time_sec": float(ntdpl_info["fit_time_sec"]),
        **overall_metrics,
    }
    fixed_rows = _intensity_rows(
        y,
        tucker_pred,
        ntdpl_pred,
        scene_id=scene_id,
        scene_name=scene_name,
        bin_kind="fixed_value",
        edges=np.asarray(FIXED_INTENSITY_EDGES, dtype=np.float64),
    )
    q_edges = _quantile_edges(y, num_quantile_bins)
    quantile_rows = _intensity_rows(
        y,
        tucker_pred,
        ntdpl_pred,
        scene_id=scene_id,
        scene_name=scene_name,
        bin_kind="scene_quantile",
        edges=q_edges,
    )
    band_rows, regime_rows = _spectral_rows(y, tucker_pred, ntdpl_pred, scene_id=scene_id, scene_name=scene_name)
    interaction_rows = _interaction_rows(
        y,
        tucker_pred,
        ntdpl_pred,
        scene_id=scene_id,
        scene_name=scene_name,
        quantile_edges=q_edges,
    )
    return overall, fixed_rows, quantile_rows, band_rows, regime_rows, interaction_rows


def _weighted_summary(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(group_cols, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        count = group["count"].to_numpy(dtype=float)
        total = float(np.sum(count))
        row = {col: value for col, value in zip(group_cols, key_tuple)}
        if total <= 0:
            rows.append(row)
            continue
        for col in ["tucker_rmse", "ntdpl_rmse"]:
            row[col] = float(np.sqrt(np.sum((group[col].to_numpy(dtype=float) ** 2) * count) / total))
        for col in ["tucker_mse", "ntdpl_mse", "mse_reduction"]:
            row[col] = float(np.sum(group[col].to_numpy(dtype=float) * count) / total)
        for col in ["mean_intensity", "tucker_bias", "ntdpl_bias", "rmse_gain_pct", "abs_error_gain_pct"]:
            row[col] = float(np.sum(group[col].to_numpy(dtype=float) * count) / total)
        row["count"] = int(total)
        row["scene_mean_gain_pct"] = float(group["rmse_gain_pct"].mean())
        row["scene_median_gain_pct"] = float(group["rmse_gain_pct"].median())
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty and {"tucker_rmse", "ntdpl_rmse"}.issubset(out.columns):
        out["pooled_rmse_gain_pct"] = 100.0 * (out["tucker_rmse"] - out["ntdpl_rmse"]) / out["tucker_rmse"].clip(lower=1e-12)
    if not out.empty and {"mse_reduction", "count"}.issubset(out.columns):
        contribution = out["mse_reduction"].to_numpy(dtype=float) * out["count"].to_numpy(dtype=float)
        total_reduction = float(np.nansum(contribution))
        out["mse_reduction_contribution_pct"] = 100.0 * contribution / max(abs(total_reduction), 1e-12)
    return out


def _plot_outputs(
    fixed_summary: pd.DataFrame,
    quantile_summary: pd.DataFrame,
    band_summary: pd.DataFrame,
    regime_summary: pd.DataFrame,
    interaction_summary: pd.DataFrame,
    outdir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    axes[0].plot(fixed_summary["bin_index"], fixed_summary["pooled_rmse_gain_pct"], marker="o", color="#2d6cdf")
    axes[0].axhline(0.0, color="#777777", linewidth=1.0)
    axes[0].set_xticks(fixed_summary["bin_index"], fixed_summary["bin_label"], rotation=35, ha="right")
    axes[0].set_ylabel("pooled RMSE gain vs Tucker (%)")
    axes[0].set_title("Fixed intensity bins")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].plot(quantile_summary["bin_index"], quantile_summary["pooled_rmse_gain_pct"], marker="o", color="#111111")
    axes[1].axhline(0.0, color="#777777", linewidth=1.0)
    axes[1].set_xlabel("scene intensity quantile bin")
    axes[1].set_title("Equal-count intensity bins")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "intensity_bin_gain.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(band_summary["band_index"], band_summary["pooled_rmse_gain_pct"], marker="o", linewidth=1.6, color="#2d6cdf")
    ax.axhline(0.0, color="#777777", linewidth=1.0)
    for _, row in regime_summary.iterrows():
        ax.axvspan(float(row["band_start"]) - 0.5, float(row["band_end"]) + 0.5, alpha=0.08)
    ax.set_xlabel("spectral band")
    ax.set_ylabel("pooled RMSE gain vs Tucker (%)")
    ax.set_title("Spectral-band error gain")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "spectral_band_gain.png", dpi=200)
    plt.close(fig)

    pivot = interaction_summary.pivot(index="quantile_bin_index", columns="regime", values="pooled_rmse_gain_pct")
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    finite = pivot.values[np.isfinite(pivot.values)]
    vmax = max(abs(float(np.nanpercentile(finite, 5))), abs(float(np.nanpercentile(finite, 95))), 1e-6) if finite.size else 1.0
    im = ax.imshow(pivot.values, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
    ax.set_xlabel("spectral regime")
    ax.set_ylabel("intensity quantile bin")
    ax.set_title("Gain by intensity and spectral regime")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outdir / "intensity_spectral_gain_heatmap.png", dpi=200)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    scene_ids = _parse_int_set(args.scene_ids)
    rank = _parse_rank(args.rank)
    target_shape = _parse_shape(args.target_shape)
    overall_rows: list[dict[str, Any]] = []
    fixed_rows: list[dict[str, Any]] = []
    quantile_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []
    interaction_rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.max_workers))) as executor:
        futures = [
            executor.submit(
                _scene_analysis,
                scene_id,
                target_shape=target_shape,
                rank=rank,
                p_max=args.p_max,
                n_iter_max=args.n_iter_max,
                init_n_iter_max=args.init_n_iter_max,
                tucker_iter_max=args.tucker_iter_max,
                num_quantile_bins=args.num_quantile_bins,
            )
            for scene_id in scene_ids
        ]
        for future in as_completed(futures):
            overall, fixed, quantile, band, regime, interaction = future.result()
            overall_rows.append(overall)
            fixed_rows.extend(fixed)
            quantile_rows.extend(quantile)
            band_rows.extend(band)
            regime_rows.extend(regime)
            interaction_rows.extend(interaction)

    frames = {
        "overall": pd.DataFrame(overall_rows).sort_values("scene_id"),
        "intensity_fixed": pd.DataFrame(fixed_rows).sort_values(["scene_id", "bin_index"]),
        "intensity_quantile": pd.DataFrame(quantile_rows).sort_values(["scene_id", "bin_index"]),
        "spectral_band": pd.DataFrame(band_rows).sort_values(["scene_id", "band_index"]),
        "spectral_regime": pd.DataFrame(regime_rows).sort_values(["scene_id", "regime_index"]),
        "intensity_spectral": pd.DataFrame(interaction_rows).sort_values(["scene_id", "quantile_bin_index", "regime_index"]),
    }
    frames["summary_intensity_fixed"] = _weighted_summary(frames["intensity_fixed"], ["bin_index", "bin_label"])
    frames["summary_intensity_quantile"] = _weighted_summary(frames["intensity_quantile"], ["bin_index"])
    frames["summary_spectral_band"] = _weighted_summary(frames["spectral_band"], ["band_index", "wavelength_nm", "regime", "regime_index"])
    frames["summary_spectral_regime"] = _weighted_summary(frames["spectral_regime"], ["regime_index", "regime", "band_start", "band_end"])
    frames["summary_intensity_spectral"] = _weighted_summary(frames["intensity_spectral"], ["quantile_bin_index", "regime_index", "regime"])
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Tucker vs NTD-PL error by intensity bins and spectral regimes.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--scene-ids", default="1-15")
    parser.add_argument("--rank", default="4,4,2")
    parser.add_argument("--target-shape", default="128,128")
    parser.add_argument("--p-max", type=int, default=6)
    parser.add_argument("--n-iter-max", type=int, default=150)
    parser.add_argument("--tucker-iter-max", type=int, default=150)
    parser.add_argument("--init-n-iter-max", type=int, default=50)
    parser.add_argument("--num-quantile-bins", type=int, default=10)
    parser.add_argument("--max-workers", type=int, default=6)
    args = parser.parse_args()

    outdir = args.outdir if args.outdir.is_absolute() else PROJECT_ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    frames = run(args)
    for name, frame in frames.items():
        frame.to_csv(outdir / f"{name}.csv", index=False)
    _plot_outputs(
        frames["summary_intensity_fixed"],
        frames["summary_intensity_quantile"],
        frames["summary_spectral_band"],
        frames["summary_spectral_regime"],
        frames["summary_intensity_spectral"],
        outdir,
    )

    print("Overall:")
    print(frames["overall"][["scene_id", "scene_name", "tucker_rmse", "ntdpl_rmse", "rmse_gain_pct"]].to_string(index=False))
    print("\nIntensity quantile summary:")
    print(
        frames["summary_intensity_quantile"][
            ["bin_index", "mean_intensity", "pooled_rmse_gain_pct", "scene_median_gain_pct", "mse_reduction_contribution_pct"]
        ].to_string(index=False)
    )
    print("\nSpectral regime summary:")
    print(
        frames["summary_spectral_regime"][
            ["regime", "band_start", "band_end", "pooled_rmse_gain_pct", "scene_median_gain_pct", "mse_reduction_contribution_pct"]
        ].to_string(index=False)
    )
    print(f"\nOutput: {outdir}")


if __name__ == "__main__":
    main()
