from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_real_tensor_reconstruction_benchmark as bench
from src.methods.ntdpl import NTDPLDecomposition
from src.methods.tucker import TuckerDecomposition
from src.metrics import val_NMSE_dB, val_PSNR, val_RMSE, val_SAM, val_SSIM
from src.types import LogCallback, Tensor


OUT_PATH = PROJECT_ROOT / "papers" / "neurips" / "tables" / "real_tensor_table6_sweep.csv"


RANK_GRIDS: dict[str, list[tuple[int, ...]]] = {
    "cbsd": [
        (4, 12, 12, 2),
        (6, 12, 12, 2),
        (8, 12, 12, 2),
        (8, 16, 16, 2),
        (10, 16, 16, 2),
        (12, 16, 16, 2),
        (8, 20, 20, 2),
        (12, 20, 20, 2),
        (12, 24, 24, 2),
        (16, 24, 24, 2),
        (16, 24, 24, 3),
        (20, 24, 24, 3),
        (16, 32, 32, 3),
        (20, 32, 32, 3),
    ],
    "cifar": [
        (8, 8, 8, 2),
        (12, 8, 8, 2),
        (16, 8, 8, 2),
        (16, 12, 12, 2),
        (24, 12, 12, 2),
        (32, 12, 12, 2),
        (24, 16, 16, 2),
        (32, 16, 16, 2),
        (48, 16, 16, 2),
        (48, 16, 16, 3),
        (64, 16, 16, 3),
        (64, 20, 20, 3),
        (96, 20, 20, 3),
    ],
    "norb": [
        (4, 6, 12, 12, 2),
        (6, 6, 12, 12, 2),
        (6, 8, 16, 16, 2),
        (8, 8, 16, 16, 2),
        (10, 8, 16, 16, 2),
        (8, 10, 20, 20, 2),
        (10, 10, 20, 20, 2),
        (12, 10, 20, 20, 2),
        (12, 12, 24, 24, 2),
        (16, 12, 24, 24, 2),
        (18, 12, 24, 24, 2),
    ],
}


NTDPL_SETTINGS: list[dict[str, Any]] = [
    {"p_max": 2, "n_iter_max": 160, "lr_core": 1e-4, "lr_factors": 3e-4, "lambda_beta": 1e-6},
    {"p_max": 3, "n_iter_max": 160, "lr_core": 1e-4, "lr_factors": 3e-4, "lambda_beta": 1e-6},
    {"p_max": 4, "n_iter_max": 180, "lr_core": 7e-5, "lr_factors": 2e-4, "lambda_beta": 1e-6},
    {"p_max": 3, "n_iter_max": 220, "lr_core": 7e-5, "lr_factors": 2e-4, "lambda_beta": 1e-7},
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="cbsd,cifar,norb")
    parser.add_argument("--max-ranks", type=int, default=0, help="Debug limit per dataset; 0 means all ranks.")
    parser.add_argument("--settings", type=int, default=0, help="Debug limit for NTDPL settings; 0 means all.")
    parser.add_argument("--rank-indices", default="", help="Comma-separated zero-based rank indices per dataset.")
    parser.add_argument("--setting-indices", default="", help="Comma-separated zero-based setting indices.")
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    selected = [x.strip() for x in args.datasets.split(",") if x.strip()]
    rows: list[dict[str, Any]] = []
    if args.append and OUT_PATH.exists():
        rows.extend(pd.read_csv(OUT_PATH).to_dict("records"))

    done = {
        (
            str(row["dataset_key"]),
            str(row["rank"]),
            int(row["p_max"]),
            int(row["n_iter_max"]),
            float(row["lr_core"]),
            float(row["lr_factors"]),
            float(row["lambda_beta"]),
        )
        for row in rows
    }

    for dataset_key in selected:
        dense = _load_dataset(dataset_key)
        tensor = Tensor(shape=dense.shape, dense=dense)
        tucker_cache: dict[str, tuple[dict[str, float], float]] = {}
        ranks = RANK_GRIDS[dataset_key]
        if args.max_ranks > 0:
            ranks = ranks[: args.max_ranks]
        rank_indices = _parse_indices(args.rank_indices)
        if rank_indices:
            ranks = [rank for idx, rank in enumerate(ranks) if idx in rank_indices]
        settings = NTDPL_SETTINGS
        if args.settings > 0:
            settings = settings[: args.settings]
        setting_indices = _parse_indices(args.setting_indices)
        if setting_indices:
            settings = [setting for idx, setting in enumerate(settings) if idx in setting_indices]
        for rank, setting in itertools.product(ranks, settings):
            key = (
                dataset_key,
                _rank_text(rank),
                int(setting["p_max"]),
                int(setting["n_iter_max"]),
                float(setting["lr_core"]),
                float(setting["lr_factors"]),
                float(setting["lambda_beta"]),
            )
            if key in done:
                continue
            print(f"\n=== {dataset_key} rank={rank} setting={setting} ===", flush=True)
            row = _run_one(dataset_key, tensor, rank, setting, tucker_cache)
            rows.append(row)
            done.add(key)
            pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
            print(_short(row), flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")
    if not frame.empty:
        for dataset_key, panel in frame.groupby("dataset_key"):
            best = panel.sort_values(["gain_rmse_pct", "delta_psnr"], ascending=[False, False]).head(8)
            print(f"\nBest RMSE gains for {dataset_key}:")
            print(best[["rank", "p_max", "n_iter_max", "tucker_rmse", "ntdpl_rmse", "gain_rmse_pct", "delta_psnr", "delta_ssim", "delta_sam"]].to_string(index=False))


def _load_dataset(dataset_key: str) -> np.ndarray:
    cfg = bench.BENCHMARKS[dataset_key]
    loader = getattr(bench, str(cfg["loader"]))
    dense = np.asarray(loader(), dtype=np.float32)
    return bench._normalize_max(dense)


def _run_one(
    dataset_key: str,
    tensor: Tensor,
    rank: tuple[int, ...],
    setting: dict[str, Any],
    tucker_cache: dict[str, tuple[dict[str, float], float]],
) -> dict[str, Any]:
    rank_key = _rank_text(rank)
    if rank_key not in tucker_cache:
        tucker_cache[rank_key] = _fit_tucker(tensor, rank)
    tucker_metrics, tucker_time = tucker_cache[rank_key]
    ntdpl_metrics, ntdpl_time = _fit_ntdpl(tensor, rank, setting)
    return {
        "dataset_key": dataset_key,
        "dataset": bench.BENCHMARKS[dataset_key]["dataset"],
        "shape": "x".join(str(v) for v in tensor.shape),
        "rank": _rank_text(rank),
        "p_max": int(setting["p_max"]),
        "n_iter_max": int(setting["n_iter_max"]),
        "lr_core": float(setting["lr_core"]),
        "lr_factors": float(setting["lr_factors"]),
        "lambda_beta": float(setting["lambda_beta"]),
        "tucker_time_sec": tucker_time,
        "ntdpl_time_sec": ntdpl_time,
        **{f"tucker_{k}": v for k, v in tucker_metrics.items()},
        **{f"ntdpl_{k}": v for k, v in ntdpl_metrics.items()},
        "gain_rmse_pct": 100.0 * (tucker_metrics["rmse"] - ntdpl_metrics["rmse"]) / max(tucker_metrics["rmse"], 1e-12),
        "delta_nmse_db": tucker_metrics["nmse_db"] - ntdpl_metrics["nmse_db"],
        "delta_psnr": ntdpl_metrics["psnr"] - tucker_metrics["psnr"],
        "delta_ssim": ntdpl_metrics["ssim"] - tucker_metrics["ssim"],
        "delta_sam": tucker_metrics["sam"] - ntdpl_metrics["sam"],
    }


def _fit_tucker(tensor: Tensor, rank: tuple[int, ...]) -> tuple[dict[str, float], float]:
    start = time.perf_counter()
    method = TuckerDecomposition(rank=rank, n_iter_max=180, init="svd", tol=1e-5)
    method.fit(tensor, None, LogCallback(log_level=0))
    elapsed = float(time.perf_counter() - start)
    return _metrics(tensor, method.reconstruct()), elapsed


def _fit_ntdpl(tensor: Tensor, rank: tuple[int, ...], setting: dict[str, Any]) -> tuple[dict[str, float], float]:
    start = time.perf_counter()
    method = NTDPLDecomposition(
        rank=rank,
        init_n_iter_max=50,
        init="tucker",
        solver_variant="optimized",
        stable_beta_update=True,
        beta_update_stage="before_grad",
        random_state=0,
        p_max=int(setting["p_max"]),
        allow_constant_term=True,
        use_continuation=True,
        factor_normalize=True,
        lr_core=float(setting["lr_core"]),
        lr_factors=float(setting["lr_factors"]),
        lambda_core=1e-6,
        lambda_factors=1e-6,
        lambda_beta=float(setting["lambda_beta"]),
        beta_update_method="ridge_lstsq",
        beta_update_interval=5,
        n_iter_max=int(setting["n_iter_max"]),
    )
    method.fit(tensor, None, LogCallback(log_level=0))
    elapsed = float(time.perf_counter() - start)
    return _metrics(tensor, method.reconstruct()), elapsed


def _metrics(tensor: Tensor, recon: Tensor) -> dict[str, float]:
    return {
        "rmse": val_RMSE(tensor, recon),
        "nmse_db": val_NMSE_dB(tensor, recon),
        "psnr": val_PSNR(tensor, recon),
        "ssim": val_SSIM(tensor, recon),
        "sam": val_SAM(tensor, recon),
    }


def _rank_text(rank: tuple[int, ...]) -> str:
    return "(" + ",".join(str(v) for v in rank) + ")"


def _parse_indices(text: str) -> set[int]:
    if not text.strip():
        return set()
    return {int(item.strip()) for item in text.split(",") if item.strip()}


def _short(row: dict[str, Any]) -> str:
    return (
        f"RMSE {row['tucker_rmse']:.5f}->{row['ntdpl_rmse']:.5f} "
        f"gain={row['gain_rmse_pct']:.2f}% PSNR+={row['delta_psnr']:.2f} "
        f"SSIM+={row['delta_ssim']:.4f} SAMdelta={row['delta_sam']:.2f}"
    )


if __name__ == "__main__":
    main()
