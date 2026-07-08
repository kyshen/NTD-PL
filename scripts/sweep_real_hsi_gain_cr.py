from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment.hsi_defaults import REAL_HSI_DATA_PATHS, reference_target_cr
from experiment.runner import _project_python
from src.data.hsi import _load_hsi_from_file


DATASETS = (
    ("jasper_ridge_hsi", "Jasper Ridge"),
    ("samson_hsi", "Samson"),
    ("urban_hsi", "Urban"),
    ("cuprite_hsi", "Cuprite"),
)

TARGET_CR_MULTIPLIERS = (1.0, 1.25, 1.50, 1.75)


def _shape(dataset_name: str) -> tuple[int, int, int]:
    cube = _load_hsi_from_file(PROJECT_ROOT / REAL_HSI_DATA_PATHS[dataset_name])
    return tuple(int(v) for v in cube.shape)


def _auto_rank_from_shape(shape: tuple[int, int, int], target_cr: float) -> tuple[int, int, int]:
    reference_shape = (128, 128, 31)
    reference_rank = (12, 12, 6)
    target_params = math.prod(shape) / target_cr
    fractions = [rank / dim for rank, dim in zip(reference_rank, reference_shape)]
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
        raise RuntimeError(f"Failed to derive rank for shape={shape}, target_cr={target_cr}")
    return best[1]


def _rank_text(rank: tuple[int, int, int]) -> str:
    return f"({rank[0]},{rank[1]},{rank[2]})"


def _compression_ratio(shape: tuple[int, int, int], rank: tuple[int, int, int]) -> float:
    params = math.prod(rank) + sum(int(dim) * int(rk) for dim, rk in zip(shape, rank))
    return math.prod(shape) / params


def _cpu_child_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
            "TBB_NUM_THREADS": "1",
        }
    )
    return env


def _hydra_command(dataset_name: str, rank: tuple[int, int, int], method: str) -> list[str]:
    rank_slug = f"r{rank[0]}_{rank[1]}_{rank[2]}"
    dataset_shape = _shape(dataset_name)
    hydra_root = f"artifacts/multirun/real-hsi-robustness/gain_cr/{dataset_name}_{rank_slug}_{method}"
    command = [
        _project_python(PROJECT_ROOT),
        "run.py",
        "-m",
        "exp=real-hsi-robustness",
        "exp_mode=benchmark",
        f"data={dataset_name}",
        f"data.target_shape=[{dataset_shape[0]},{dataset_shape[1]}]",
        "data.crop_shape=null",
        "task=decompose",
        "task.log_level=0",
        "filter=bias-filter",
        "filter.normalize_method=max",
        f"method={method}",
        f"method.rank=[{rank[0]},{rank[1]},{rank[2]}]",
        "method.n_iter_max=300",
        f"hydra.sweep.dir={hydra_root}",
        "hydra.sweep.subdir=.",
    ]
    if method == "ntdpl":
        command.extend(
            [
                "method.init_n_iter_max=50",
                "method.init=tucker",
                "method.solver_variant=optimized",
                "method.stable_beta_update=true",
                "method.beta_update_stage=before_grad",
                "method.random_state=0",
                "method.p_max=4",
                "method.allow_constant_term=true",
                "method.use_continuation=true",
                "method.factor_normalize=true",
                "method.lr_core=1e-4",
                "method.lr_factors=3e-4",
                "method.lambda_core=1e-6",
                "method.lambda_factors=1e-6",
                "method.lambda_beta=1e-6",
                "method.beta_update_method=ridge_lstsq",
                "method.beta_update_interval=5",
            ]
        )
    elif method == "tucker":
        command.extend(["method.init=svd", "method.tol=1e-4"])
    else:
        raise ValueError(f"Unsupported method: {method}")
    return command


def _run_command(command: list[str]) -> None:
    subprocess.run(command, cwd=PROJECT_ROOT, env=_cpu_child_env(), check=True)


def _run_commands(commands: list[list[str]], jobs: int) -> None:
    jobs = max(1, jobs)
    if jobs == 1 or len(commands) == 1:
        for command in commands:
            _run_command(command)
        return
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [executor.submit(_run_command, command) for command in commands]
        for future in as_completed(futures):
            future.result()


def _build_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    base_cr = reference_target_cr()
    for dataset_name, label in DATASETS:
        shape = _shape(dataset_name)
        for multiplier in TARGET_CR_MULTIPLIERS:
            target_cr = base_cr * multiplier
            rank = _auto_rank_from_shape(shape, target_cr)
            cr = _compression_ratio(shape, rank)
            rows.append(
                {
                    "dataset": dataset_name,
                    "dataset_label": label,
                    "target_cr": target_cr,
                    "rank": _rank_text(rank),
                    "cr": cr,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run external HSI gain-CR sweeps with Hydra multirun.")
    parser.add_argument("--jobs", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 2)))
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()

    if args.collect_only:
        out = _build_table()
        print(out.to_string(index=False))
        return

    commands = []
    for dataset_name, _ in DATASETS:
        shape = _shape(dataset_name)
        for multiplier in TARGET_CR_MULTIPLIERS:
            rank = _auto_rank_from_shape(shape, reference_target_cr() * multiplier)
            commands.append(_hydra_command(dataset_name, rank, "tucker"))
            commands.append(_hydra_command(dataset_name, rank, "ntdpl"))

    _run_commands(commands, args.jobs)


if __name__ == "__main__":
    main()
