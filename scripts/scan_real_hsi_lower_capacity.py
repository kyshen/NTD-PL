from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment.hsi_defaults import REAL_HSI_DATA_PATHS
from src.data.hsi import _load_hsi_from_file
from src.methods.ntdpl import NTDPLDecomposition
from src.methods.tucker import TuckerDecomposition
from src.metrics import val_RMSE
from src.types import LogCallback, Tensor

DATASETS = [
    ("jasper_ridge_hsi", "Jasper Ridge"),
    ("samson_hsi", "Samson"),
    ("urban_hsi", "Urban"),
    ("cuprite_hsi", "Cuprite"),
]
REFERENCE_SHAPE = (128, 128, 31)
REFERENCE_RANK = (12, 12, 6)
BASE_CR = math.prod(REFERENCE_SHAPE) / (
    math.prod(REFERENCE_RANK) + sum(dim * rank for dim, rank in zip(REFERENCE_SHAPE, REFERENCE_RANK))
)
TARGET_CRS = [
    ("current", BASE_CR),
    ("lowercap_1", BASE_CR * 1.25),
    ("lowercap_2", BASE_CR * 1.50),
]


def _cpu_worker_env() -> None:
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


def _auto_rank_from_shape(shape: tuple[int, int, int], target_cr: float) -> tuple[int, int, int]:
    target_params = math.prod(shape) / target_cr
    fractions = [rank / dim for rank, dim in zip(REFERENCE_RANK, REFERENCE_SHAPE)]
    best: tuple[float, tuple[int, int, int]] | None = None
    for step in range(6, 121):
        scale = step / 40.0
        rank = tuple(
            max(2, min(dim - 1, int(round(scale * frac * dim))))
            for frac, dim in zip(fractions, shape)
        )
        params = math.prod(rank) + sum(dim * rk for dim, rk in zip(shape, rank))
        error = abs(params - target_params)
        if best is None or error < best[0]:
            best = (error, rank)
    if best is None:
        raise RuntimeError(f"Failed to find rank for shape={shape}, target_cr={target_cr}")
    return best[1]


def _compression_ratio(shape: tuple[int, int, int], rank: tuple[int, int, int]) -> float:
    params = math.prod(rank) + sum(int(dim) * int(rk) for dim, rk in zip(shape, rank))
    return math.prod(shape) / params


def _run_one(dataset_name: str, label: str, target_name: str, target_cr: float) -> dict[str, object]:
    _cpu_worker_env()
    cube = np.asarray(_load_hsi_from_file(PROJECT_ROOT / REAL_HSI_DATA_PATHS[dataset_name]), dtype=np.float32)
    cube = cube / float(np.max(cube))
    shape = tuple(int(v) for v in cube.shape)
    rank = _auto_rank_from_shape(shape, target_cr)
    tensor = Tensor(shape=cube.shape, dense=cube)

    tucker = TuckerDecomposition(rank=rank, n_iter_max=300, init="svd", tol=1e-4)
    tucker.fit(tensor, None, LogCallback(log_level=0))
    recon_tucker = tucker.reconstruct()
    rmse_tucker = val_RMSE(tensor, recon_tucker)

    ntdpl = NTDPLDecomposition(
        rank=rank,
        init_n_iter_max=50,
        init="tucker",
        solver_variant="optimized",
        stable_beta_update=True,
        beta_update_stage="before_grad",
        random_state=0,
        p_max=4,
        allow_constant_term=True,
        use_continuation=True,
        factor_normalize=True,
        lr_core=1e-4,
        lr_factors=3e-4,
        lambda_core=1e-6,
        lambda_factors=1e-6,
        lambda_beta=1e-6,
        beta_update_method="ridge_lstsq",
        beta_update_interval=5,
        n_iter_max=300,
    )
    ntdpl.fit(tensor, None, LogCallback(log_level=0))
    recon_ntdpl = ntdpl.reconstruct()
    rmse_ntdpl = val_RMSE(tensor, recon_ntdpl)

    return {
        "dataset": label,
        "target": target_name,
        "rank": rank,
        "cr": _compression_ratio(shape, rank),
        "tucker_rmse": float(rmse_tucker),
        "ntdpl_rmse": float(rmse_ntdpl),
        "gain_pct": 100.0 * (float(rmse_tucker) - float(rmse_ntdpl)) / max(float(rmse_tucker), 1e-12),
    }


def main() -> None:
    jobs = []
    for target_name, target_cr in TARGET_CRS:
        for dataset_name, label in DATASETS:
            jobs.append((dataset_name, label, target_name, target_cr))

    results: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_run_one, *job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            print("DONE", row)
            results.append(row)

    target_order = {"current": 0, "lowercap_1": 1, "lowercap_2": 2}
    dataset_order = {label: idx for idx, (_, label) in enumerate(DATASETS)}
    results.sort(key=lambda row: (target_order[str(row["target"])], dataset_order[str(row["dataset"])]))

    print("\nSUMMARY")
    for target_name, _ in TARGET_CRS:
        rows = [row for row in results if row["target"] == target_name]
        avg_gain = sum(float(row["gain_pct"]) for row in rows) / len(rows)
        avg_tucker = sum(float(row["tucker_rmse"]) for row in rows) / len(rows)
        print(target_name, "avg_gain", round(avg_gain, 3), "avg_tucker", round(avg_tucker, 5))
        for row in rows:
            print(row)


if __name__ == "__main__":
    main()
