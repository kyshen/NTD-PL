from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "TBB_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.hsi import CAVEHSIData
from src.filters.bias import BiasFilter
from src.methods.tucker import TuckerDecomposition
from src.metrics import val_RMSE
from src.types import LogCallback, Tensor


DEFAULT_PER_SCENE = PROJECT_ROOT / "artifacts/results/cave_main_baselines_r24_full/per_scene.csv"
DEFAULT_OUTDIR = PROJECT_ROOT / "papers/supplementary/processed_results/diagnostics/cave_full_reconstruction"
DEFAULT_RANK = (24, 24, 4)
SCENE_IDS = tuple(range(1, 16))


def _worker_env() -> None:
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
        "TBB_NUM_THREADS",
    ):
        os.environ.setdefault(key, "1")


def _load_scene(scene_id: int) -> tuple[str, Tensor, Tensor]:
    dataset = CAVEHSIData(
        path="data/CAVE",
        id=int(scene_id),
        target_shape=(512, 512),
        crop_shape=None,
    )
    BiasFilter(seed=0, normalize_method="max", snr_db=None, bias=None)(dataset)
    return dataset.scene_name, dataset.get("fit"), dataset.get("eval")


def _poly_design(x: np.ndarray, degree: int) -> np.ndarray:
    return np.vander(np.asarray(x, dtype=np.float64), N=int(degree) + 1, increasing=True)


def _fit_scalar_poly_predict(
    x_pred: np.ndarray,
    x_target: np.ndarray,
    *,
    degree: int,
    lambda_reg: float,
    sample_size: int,
    seed: int,
) -> np.ndarray:
    x = np.asarray(x_pred, dtype=np.float64).reshape(-1)
    y = np.asarray(x_target, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}.")
    if 0 < int(sample_size) < x.size:
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(x.size, size=int(sample_size), replace=False)
        x_fit = x[idx]
        y_fit = y[idx]
    else:
        x_fit = x
        y_fit = y

    phi = _poly_design(x_fit, int(degree))
    scales = np.maximum(np.linalg.norm(phi, axis=0), 1e-12)
    phi_scaled = phi / scales
    gram = phi_scaled.T @ phi_scaled
    rhs = phi_scaled.T @ y_fit
    coeff_scaled = np.linalg.solve(gram + float(lambda_reg) * np.eye(gram.shape[0]), rhs)
    coeff = coeff_scaled / scales

    pred = np.zeros_like(x, dtype=np.float64)
    for c in coeff[::-1]:
        pred = pred * x + float(c)
    return pred.reshape(x_pred.shape).astype(np.float32)


def _d_link_db(tucker_rmse: float, link_rmse: float) -> float:
    ratio = (float(link_rmse) ** 2) / max(float(tucker_rmse) ** 2, 1e-12)
    return float(10.0 * math.log10(1.0 / max(ratio, 1e-12)))


def _load_gain_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.loc[frame["method_name"].isin(["tucker", "ntdpl"])].copy()
    wide = frame.pivot_table(
        index=["scene_id", "scene_name"],
        columns="method_name",
        values=["RMSE", "SAM", "params"],
        aggfunc="mean",
    )
    rows = []
    for (scene_id, scene_name), row in wide.iterrows():
        tucker_rmse = float(row[("RMSE", "tucker")])
        ntdpl_rmse = float(row[("RMSE", "ntdpl")])
        tucker_sam = float(row[("SAM", "tucker")])
        ntdpl_sam = float(row[("SAM", "ntdpl")])
        rows.append(
            {
                "scene_id": int(scene_id),
                "scene_name": scene_name,
                "tucker_rmse": tucker_rmse,
                "ntdpl_rmse": ntdpl_rmse,
                "rmse_gain_pct": 100.0 * (tucker_rmse - ntdpl_rmse) / max(tucker_rmse, 1e-12),
                "tucker_sam": tucker_sam,
                "ntdpl_sam": ntdpl_sam,
                "sam_gain_pct": 100.0 * (tucker_sam - ntdpl_sam) / max(tucker_sam, 1e-12),
                "tucker_params": int(row[("params", "tucker")]),
                "ntdpl_params": int(row[("params", "ntdpl")]),
            }
        )
    return pd.DataFrame(rows)


def _run_one(scene_id: int, args: argparse.Namespace) -> dict[str, Any]:
    _worker_env()
    start = perf_counter()
    scene_name, fit_tensor, eval_tensor = _load_scene(scene_id)
    tucker = TuckerDecomposition(rank=DEFAULT_RANK, n_iter_max=int(args.tucker_n_iter_max), init="svd", tol=1e-4)
    tucker.fit(fit_tensor, None, LogCallback(0))
    tucker_recon = np.asarray(tucker.reconstruct().dense, dtype=np.float32)
    link_recon = _fit_scalar_poly_predict(
        tucker_recon,
        np.asarray(eval_tensor.dense, dtype=np.float32),
        degree=int(args.p_max),
        lambda_reg=float(args.lambda_beta),
        sample_size=int(args.link_sample_size),
        seed=int(args.seed) + int(scene_id),
    )
    tucker_rmse_refit = float(val_RMSE(eval_tensor, Tensor(shape=tucker_recon.shape, dense=tucker_recon)))
    link_rmse = float(val_RMSE(eval_tensor, Tensor(shape=link_recon.shape, dense=link_recon)))
    return {
        "scene_id": int(scene_id),
        "scene_name": scene_name,
        "rank": "(" + ",".join(str(v) for v in DEFAULT_RANK) + ")",
        "d_link_db": _d_link_db(tucker_rmse_refit, link_rmse),
        "tucker_rmse_refit": tucker_rmse_refit,
        "link_refresh_rmse": link_rmse,
        "runtime_link": float(perf_counter() - start),
        "status": "ok",
        "notes": f"full reconstruction; p_max={int(args.p_max)}; link_sample_size={int(args.link_sample_size)}",
    }


def _assign_labels(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["diagnostic_label"] = ""
    values = pd.to_numeric(out["d_link_db"], errors="coerce")
    low = float(values.quantile(1.0 / 3.0))
    high = float(values.quantile(2.0 / 3.0))
    out.loc[values <= low, "diagnostic_label"] = "boundary"
    out.loc[(values > low) & (values < high), "diagnostic_label"] = "moderate"
    out.loc[values >= high, "diagnostic_label"] = "effective"
    return out


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    ok = frame.loc[frame["status"].eq("ok")].copy()
    ordered = ok.sort_values("d_link_db").reset_index(drop=True)
    bins = np.array_split(ordered, 3)
    rho = spearmanr(ok["d_link_db"], ok["rmse_gain_pct"], nan_policy="omit").statistic
    return pd.DataFrame(
        [
            {
                "domain": "Hyperspectral",
                "dataset": "CAVE",
                "num_units": int(ok.shape[0]),
                "median_d_link": float(ok["d_link_db"].median()),
                "mean_d_link": float(ok["d_link_db"].mean()),
                "spearman_dlink_gain": float(rho),
                "mean_gain": float(ok["rmse_gain_pct"].mean()),
                "median_gain": float(ok["rmse_gain_pct"].median()),
                "low_tercile_gain": float(bins[0]["rmse_gain_pct"].mean()),
                "mid_tercile_gain": float(bins[1]["rmse_gain_pct"].mean()),
                "high_tercile_gain": float(bins[2]["rmse_gain_pct"].mean()),
                "num_effective": int(ok["diagnostic_label"].eq("effective").sum()),
                "num_boundary": int(ok["diagnostic_label"].eq("boundary").sum()),
            }
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CAVE full-reconstruction D_link diagnostics.")
    parser.add_argument("--per-scene", type=Path, default=DEFAULT_PER_SCENE)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--jobs", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--p-max", dest="p_max", type=int, default=6)
    parser.add_argument("--lambda-beta", type=float, default=1e-6)
    parser.add_argument("--link-sample-size", type=int, default=400_000)
    parser.add_argument("--tucker-n-iter-max", type=int, default=300)
    args = parser.parse_args()

    gains = _load_gain_rows(PROJECT_ROOT / args.per_scene if not args.per_scene.is_absolute() else args.per_scene)
    jobs = max(1, min(int(args.jobs), len(SCENE_IDS)))
    if jobs == 1:
        rows = [_run_one(scene_id, args) for scene_id in SCENE_IDS]
    else:
        rows = []
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = {executor.submit(_run_one, scene_id, args): scene_id for scene_id in SCENE_IDS}
            for index, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                print(f"[{index}/{len(futures)}] scene {row['scene_id']:02d} {row['scene_name']}: ok", flush=True)
                rows.append(row)

    link = pd.DataFrame(rows)
    frame = gains.merge(link, on=["scene_id", "scene_name"], how="inner")
    frame = _assign_labels(frame)
    frame.insert(0, "domain", "Hyperspectral")
    frame.insert(1, "dataset", "CAVE")
    frame["unit_id"] = frame["scene_id"].map(lambda value: f"cave_{int(value):02d}")
    frame["unit_label"] = frame["scene_name"]
    frame["shape"] = "512x512x31"
    frame["seed"] = int(args.seed)
    frame = frame[
        [
            "domain",
            "dataset",
            "unit_id",
            "unit_label",
            "shape",
            "rank",
            "seed",
            "tucker_rmse",
            "ntdpl_rmse",
            "rmse_gain_pct",
            "tucker_sam",
            "ntdpl_sam",
            "sam_gain_pct",
            "d_link_db",
            "diagnostic_label",
            "tucker_params",
            "ntdpl_params",
            "runtime_link",
            "status",
            "notes",
            "tucker_rmse_refit",
            "link_refresh_rmse",
        ]
    ].sort_values("unit_id")
    summary = _summary(frame)

    outdir = PROJECT_ROOT / args.outdir if not args.outdir.is_absolute() else args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(outdir / "per_unit_results.csv", index=False)
    summary.to_csv(outdir / "summary_by_dataset.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    print(f"Wrote {outdir}", flush=True)


if __name__ == "__main__":
    main()
