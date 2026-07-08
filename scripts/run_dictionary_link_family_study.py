from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from tensorly import tucker_to_tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.methods.ntdpl import NTDPLDecomposition
from src.methods.tucker import TuckerDecomposition
from src.metrics import val_RMSE
from src.types import LogCallback, Tensor
from src.utils.filter_ops import mix_with_exact_energy_ratio, orthogonal_nonlinear_part


LINK_SPECS = {
    "power": {"label": "Power", "capacity": "$d=5$", "p_max": 4},
    "chebyshev": {"label": "Chebyshev", "capacity": "$d=5$", "p_max": 4},
    "rbf": {"label": "RBF", "capacity": "$d=5$", "p_max": 4},
    "spline": {"label": "Spline", "capacity": "$d=5$", "p_max": 4},
}

RESPONSE_SPECS = {
    "square": {"label": r"$s^2$"},
    "square_cubic": {"label": r"$s^2+s^3$"},
    "tanh": {"label": r"$\tanh(\kappa s)$"},
    "exp": {"label": r"$(e^{\kappa s}-1)/\kappa$"},
}


def _rank_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def _pm(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def _fit_tucker(x: np.ndarray, rank: tuple[int, ...], n_iter: int, mask: np.ndarray | None = None) -> np.ndarray:
    model = TuckerDecomposition(rank=rank, n_iter_max=n_iter, init="svd", tol=1e-4)
    model.fit(Tensor(shape=x.shape, dense=x), mask=mask, logcallback=LogCallback(log_level=0))
    return np.asarray(model.reconstruct().dense, dtype=np.float32)


def _fit_ntdpl(
    x: np.ndarray,
    rank: tuple[int, ...],
    n_iter: int,
    *,
    link_kind: str,
    p_max: int,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    model = NTDPLDecomposition(
        rank=rank,
        init_n_iter_max=min(50, max(5, n_iter // 3)),
        p_max=p_max,
        allow_constant_term=True,
        n_iter_max=n_iter,
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
        stable_beta_update=True,
        beta_update_stage="before_grad",
        link_kind=link_kind,
    )
    model.fit(Tensor(shape=x.shape, dense=x), mask=mask, logcallback=LogCallback(log_level=0))
    return np.asarray(model.reconstruct().dense, dtype=np.float32)


def _controlled_latent(seed: int, shape: tuple[int, int, int], rank: tuple[int, int, int]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    factors = []
    for dim, rk in zip(shape, rank):
        q, _ = np.linalg.qr(rng.normal(size=(dim, rk)))
        factors.append(q[:, :rk].astype(np.float32))
    core = rng.normal(size=rank).astype(np.float32)
    latent = np.asarray(tucker_to_tensor((core, factors)), dtype=np.float32)
    latent = latent - float(np.mean(latent))
    scale = float(np.std(latent))
    if scale > 1e-12:
        latent = latent / scale
    return latent.astype(np.float32)


def _controlled_response(latent: np.ndarray, response: str, alpha: float) -> np.ndarray:
    if response == "square":
        raw = latent**2
    elif response == "square_cubic":
        raw = latent**2 + latent**3
    elif response == "tanh":
        raw = np.tanh(1.5 * latent)
    elif response == "exp":
        raw = np.expm1(0.6 * latent) / 0.6
    else:
        raise ValueError(f"Unsupported response: {response}")
    residual = orthogonal_nonlinear_part(latent, raw)
    y = mix_with_exact_energy_ratio(latent, residual, alpha)
    y = y - float(np.min(y))
    y = y / max(float(np.max(y)), 1e-12)
    return y.astype(np.float32)


def run_controlled(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    shape = _rank_tuple(args.controlled_shape)
    rank = _rank_tuple(args.controlled_rank)
    seeds = list(range(int(args.controlled_seeds)))

    for response in RESPONSE_SPECS:
        for seed in seeds:
            latent = _controlled_latent(seed, shape, rank)
            target = _controlled_response(latent, response, float(args.alpha))
            tucker_recon = _fit_tucker(target, rank, int(args.controlled_iter))
            tucker_rmse = val_RMSE(Tensor(shape=target.shape, dense=target), Tensor(shape=target.shape, dense=tucker_recon))
            rows.append(
                {
                    "setting": "controlled",
                    "response": response,
                    "seed": seed,
                    "dictionary": "tucker",
                    "RMSE": tucker_rmse,
                }
            )
            for kind, spec in LINK_SPECS.items():
                recon = _fit_ntdpl(
                    target,
                    rank,
                    int(args.controlled_iter),
                    link_kind=kind,
                    p_max=int(spec["p_max"]),
                )
                rmse = val_RMSE(Tensor(shape=target.shape, dense=target), Tensor(shape=target.shape, dense=recon))
                rows.append(
                    {
                        "setting": "controlled",
                        "response": response,
                        "seed": seed,
                        "dictionary": kind,
                        "RMSE": rmse,
                    }
                )
    return pd.DataFrame(rows)


def summarize(controlled: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for kind, spec in LINK_SPECS.items():
        row: dict[str, Any] = {
            "dictionary": kind,
            "label": spec["label"],
            "capacity": spec["capacity"],
        }
        for response in RESPONSE_SPECS:
            panel = controlled.loc[
                (controlled["dictionary"] == kind) & (controlled["response"] == response)
            ]
            row[f"{response}_rmse"] = float(panel["RMSE"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def latex_table(summary: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{@{}lccccc@{}}",
        r"\toprule",
        r"Dictionary & Budget & \multicolumn{4}{c}{Target link (RMSE$\downarrow$)} \\",
        r"\cmidrule(l){3-6}",
        r" & & $s^2$ & $s^2+s^3$ & $\tanh_\kappa$ & $\exp_\kappa$ \\",
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"{row.label} & {row.capacity} & "
            f"{_pm(float(row.square_rmse), 4)} & "
            f"{_pm(float(row.square_cubic_rmse), 4)} & "
            f"{_pm(float(row.tanh_rmse), 4)} & "
            f"{_pm(float(row.exp_rmse), 4)} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Representative dictionary-link family study for the TSP paper.")
    parser.add_argument("--controlled-shape", default="24,24,24")
    parser.add_argument("--controlled-rank", default="4,4,4")
    parser.add_argument("--controlled-seeds", type=int, default=5)
    parser.add_argument("--controlled-iter", type=int, default=120)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--outdir", default="artifacts/results/dictionary_link_family")
    parser.add_argument("--table-path", default="papers/tsp/tables/dictionary_link_family.tex")
    args = parser.parse_args()

    controlled = run_controlled(args)
    summary = summarize(controlled)

    outdir = PROJECT_ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    controlled.to_csv(outdir / "controlled_runs.csv", index=False)
    summary.to_csv(outdir / "summary.csv", index=False)

    table_path = PROJECT_ROOT / args.table_path
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(latex_table(summary), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
