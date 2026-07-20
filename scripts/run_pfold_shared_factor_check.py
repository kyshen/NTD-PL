from __future__ import annotations

"""Small controlled check of the general and shared-factor p-fold models."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import sys

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

import numpy as np
from tensorly.decomposition import tucker
from tensorly.tucker_tensor import tucker_to_tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ntdpl.core import ntdpl


def _fold_core(core: np.ndarray, p: int) -> np.ndarray:
    """Arrange the p-fold outer product of the core as a Tucker core."""
    product = core
    for _ in range(p - 1):
        product = np.multiply.outer(product, core)
    n_modes = core.ndim
    axes = [q * n_modes + n for n in range(n_modes) for q in range(p)]
    shape = tuple(size**p for size in core.shape)
    return np.transpose(product, axes).reshape(shape)


def _shared_map(factor: np.ndarray, p: int) -> np.ndarray:
    out = np.ones((factor.shape[0],) + (factor.shape[1],) * p)
    for q in range(p):
        shape = [factor.shape[0]] + [1] * p
        shape[q + 1] = factor.shape[1]
        out *= factor.reshape(shape)
    return out.reshape(factor.shape[0], factor.shape[1] ** p)


def _normalize_columns(factor: np.ndarray) -> np.ndarray:
    scale = np.maximum(np.linalg.norm(factor, axis=0, keepdims=True), 1e-8)
    return factor / scale


def _relative_rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((target - prediction) ** 2)) / np.std(target))


def _fit_ntdpl(target: np.ndarray, rank: tuple[int, ...], iterations: int) -> float:
    core, factors, beta = ntdpl(
        target,
        rank=rank,
        init_n_iter_max=40,
        p_max=2,
        allow_constant_term=True,
        n_iter_max=iterations,
        use_continuation=True,
        factor_normalize=True,
        lr_core=3e-3,
        lr_factors=3e-3,
        lambda_core=1e-8,
        lambda_factors=1e-8,
        lambda_beta=1e-8,
        beta_update_method="ridge_lstsq",
        init="tucker",
        random_state=0,
        beta_update_interval=1,
        stable_beta_update=True,
        beta_update_stage="before_grad",
        return_history=False,
    )
    signal = tucker_to_tensor((core, factors))
    prediction = beta[0] + beta[1] * signal + beta[2] * signal**2
    return _relative_rmse(target, prediction)


def _one_seed(
    seed: int,
    shape: tuple[int, ...],
    rank: tuple[int, ...],
    iterations: int,
) -> list[dict[str, float | int | str]]:
    rng = np.random.default_rng(seed)
    core = rng.normal(size=rank)
    factors = [
        _normalize_columns(rng.normal(size=(size, rnk)))
        for size, rnk in zip(shape, rank)
    ]
    initial_signal = tucker_to_tensor((core, factors))
    core *= 0.75 / max(float(np.max(np.abs(initial_signal))), 1e-8)
    signal = tucker_to_tensor((core, factors))

    folded_core = _fold_core(core, 2)
    shared_factors = [_shared_map(factor, 2) for factor in factors]
    shared_term = tucker_to_tensor((folded_core, shared_factors))
    if not np.allclose(shared_term, signal**2, atol=1e-6):
        raise RuntimeError("The shared-factor p-fold construction is inconsistent.")

    separate_factors = [
        _normalize_columns(rng.normal(size=(size, rnk**2)))
        for size, rnk in zip(shape, rank)
    ]
    separate_term = tucker_to_tensor((folded_core, separate_factors))
    separate_term *= np.std(shared_term) / max(float(np.std(separate_term)), 1e-8)

    rows: list[dict[str, float | int | str]] = []
    for target_name, target in (
        ("shared factors", signal + 0.8 * shared_term),
        ("separate p-fold maps", signal + 0.8 * separate_term),
    ):
        tucker_core, tucker_factors = tucker(
            target,
            rank=rank,
            n_iter_max=100,
            init="svd",
            random_state=seed,
        )
        rows.append(
            {
                "seed": seed,
                "target": target_name,
                "tucker_rmse": _relative_rmse(
                    target, tucker_to_tensor((tucker_core, tucker_factors))
                ),
                "ntdpl_rmse": _fit_ntdpl(target, rank, iterations),
                "general_pfold_rmse": 0.0,
            }
        )
    return rows


def _mean_std(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=1))


def _format_error(mean: float, std: float) -> str:
    if 0.0 < mean < 1e-3:
        mean_coeff, mean_exp = f"{mean:.1e}".split("e")
        std_coeff, std_exp = f"{std:.1e}".split("e")
        return (
            f"${mean_coeff}\\times10^{{{int(mean_exp)}}}"
            f" \\pm {std_coeff}\\times10^{{{int(std_exp)}}}$"
        )
    return f"{mean:.3f} $\\pm$ {std:.3f}"


def _write_outputs(
    rows: list[dict[str, float | int | str]],
    out_prefix: Path,
    shape: tuple[int, ...],
    rank: tuple[int, ...],
) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: (str(row["target"]), int(row["seed"])))
    out_prefix.with_suffix(".csv").write_text(
        "seed,target,tucker_rmse,ntdpl_rmse,general_pfold_rmse\n"
        + "\n".join(
            f"{row['seed']},{row['target']},{float(row['tucker_rmse']):.8g},"
            f"{float(row['ntdpl_rmse']):.8g},{float(row['general_pfold_rmse']):.8g}"
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )

    tex = [
        r"\begin{tabular}{@{}l c c c@{}}",
        r"\toprule",
        r"Target structure & Tucker & NTD-PL & General $p$-fold \\",
        r"\midrule",
    ]
    for target in ("shared factors", "separate p-fold maps"):
        subset = [row for row in rows if row["target"] == target]
        values = {
            key: _mean_std([float(row[key]) for row in subset])
            for key in ("tucker_rmse", "ntdpl_rmse", "general_pfold_rmse")
        }
        label = "Shared factors" if target == "shared factors" else "Separate maps"
        tex.append(
            f"{label} & {_format_error(*values['tucker_rmse'])} & "
            f"{_format_error(*values['ntdpl_rmse'])} & "
            f"{values['general_pfold_rmse'][0]:.3f}\\\\"
        )
    tex.extend([r"\bottomrule", r"\end{tabular}"])
    out_prefix.with_suffix(".tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0-7")
    parser.add_argument("--shape", default="12,11,10")
    parser.add_argument("--rank", default="2,2,2")
    parser.add_argument("--iterations", type=int, default=450)
    parser.add_argument("--jobs", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument(
        "--out-prefix",
        default="papers/tsp-supplementary/tables/pfold_shared_factor_check",
    )
    args = parser.parse_args()
    if "-" in args.seeds:
        lo, hi = (int(value) for value in args.seeds.split("-", 1))
        seeds = list(range(lo, hi + 1))
    else:
        seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    shape = tuple(int(value) for value in args.shape.split(","))
    rank = tuple(int(value) for value in args.rank.split(","))
    jobs = max(1, min(args.jobs, len(seeds)))
    if jobs == 1:
        nested = [_one_seed(seed, shape, rank, args.iterations) for seed in seeds]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            nested = list(
                executor.map(
                    _one_seed,
                    seeds,
                    [shape] * len(seeds),
                    [rank] * len(seeds),
                    [args.iterations] * len(seeds),
                )
            )
    rows = [row for group in nested for row in group]
    _write_outputs(rows, PROJECT_ROOT / args.out_prefix, shape, rank)
    print(f"Wrote {PROJECT_ROOT / args.out_prefix}.csv and .tex")


if __name__ == "__main__":
    main()
