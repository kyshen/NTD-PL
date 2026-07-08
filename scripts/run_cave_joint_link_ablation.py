from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import os
from pathlib import Path
import sys
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

import numpy as np
import pandas as pd
from tensorly.decomposition import tucker
from tensorly.tenalg.core_tenalg import multi_mode_dot
from tensorly.tucker_tensor import tucker_to_tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment.hsi_defaults import CAVE_RECON_MAIN_RANK
from src.data.hsi import CAVEHSIData
from src.filters.bias import BiasFilter
from src.methods.ntdpl import NTDPLDecomposition
from src.methods.tucker import TuckerDecomposition
from src.metrics import val_NMSE_dB, val_RMSE, val_SAM
from src.types import LogCallback, Tensor


METHOD_ORDER = {
    "tucker": 0,
    "ntdpl": 1,
    "spline": 2,
}
METHOD_LABELS = {
    "tucker": "Tucker",
    "ntdpl": "NTD-PL(P=4)",
    "spline": "Joint Spline Link(K=8)",
}


@dataclass
class SplineLinkedTucker:
    rank: tuple[int, int, int]
    n_iter_max: int = 150
    init_n_iter_max: int = 50
    n_knots: int = 8
    lambda_beta: float = 1e-6
    lambda_core: float = 1e-6
    lambda_factors: float = 1e-6
    lr_core: float = 1e-4
    lr_factors: float = 3e-4
    random_state: int = 0

    def fit(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        start = perf_counter()
        x = np.asarray(x, dtype=np.float32)
        core, factors = tucker(
            x,
            rank=self.rank,
            n_iter_max=self.init_n_iter_max,
            init="svd",
            random_state=self.random_state,
        )
        core = np.asarray(core, dtype=np.float32)
        factors = [np.asarray(f, dtype=np.float32) for f in factors]
        core, factors = _normalize_tucker(core, factors)

        s0 = np.asarray(tucker_to_tensor((core, factors)), dtype=np.float32)
        knots = np.quantile(s0.reshape(-1), np.linspace(0.0, 1.0, self.n_knots)).astype(np.float32)
        knots = np.unique(knots)
        if knots.size < 4:
            lo = float(np.min(s0))
            hi = float(np.max(s0))
            if abs(hi - lo) < 1e-8:
                hi = lo + 1.0
            knots = np.linspace(lo, hi, self.n_knots, dtype=np.float32)

        theta = _fit_spline_coefficients(s0, x, knots, self.lambda_beta)
        st_core = _adam_init(core.shape)
        st_factors = [_adam_init(f.shape) for f in factors]
        fit_scale = np.float32(1.0 / x.size)
        modes_all = list(range(x.ndim))

        for _ in range(self.n_iter_max):
            s = np.asarray(tucker_to_tensor((core, factors)), dtype=np.float32)
            yhat, dyds = _spline_eval_and_deriv(s, knots, theta)
            t = (yhat - x) * dyds * fit_scale

            grad_core = multi_mode_dot(t, [f.T for f in factors], modes=modes_all)
            grad_core = grad_core.astype(np.float32) + self.lambda_core * core
            _adam_step(core, grad_core, st_core, lr=self.lr_core)

            for mode in range(x.ndim):
                other_modes = [k for k in range(x.ndim) if k != mode]
                m = multi_mode_dot(core, [factors[k] for k in other_modes], modes=other_modes)
                z = np.asarray(np.moveaxis(m, mode, 0).reshape(m.shape[mode], -1), dtype=np.float32)
                tn = np.asarray(np.moveaxis(t, mode, 0).reshape(t.shape[mode], -1), dtype=np.float32)
                grad_a = tn @ z.T
                grad_a = grad_a.astype(np.float32) + self.lambda_factors * factors[mode]
                _adam_step(factors[mode], grad_a, st_factors[mode], lr=self.lr_factors)

            core, factors = _normalize_tucker(core, factors)
            s = np.asarray(tucker_to_tensor((core, factors)), dtype=np.float32)
            theta = _fit_spline_coefficients(s, x, knots, self.lambda_beta)

        s = np.asarray(tucker_to_tensor((core, factors)), dtype=np.float32)
        yhat, _ = _spline_eval_and_deriv(s, knots, theta)
        return np.asarray(yhat, dtype=np.float32), perf_counter() - start


def _fit_spline_coefficients(s: np.ndarray, y: np.ndarray, knots: np.ndarray, lambda_beta: float) -> np.ndarray:
    basis = _linear_spline_basis(s.reshape(-1), knots)
    target = y.reshape(-1).astype(np.float64)
    gram = basis.T @ basis
    rhs = basis.T @ target
    theta = np.linalg.solve(gram + float(lambda_beta) * np.eye(gram.shape[0]), rhs)
    return theta.astype(np.float32)


def _linear_spline_basis(values: np.ndarray, knots: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    knots = np.asarray(knots, dtype=np.float32)
    k = knots.size
    idx = np.searchsorted(knots, values, side="right") - 1
    idx = np.clip(idx, 0, k - 2)
    left = knots[idx]
    right = knots[idx + 1]
    weight_right = (values - left) / np.maximum(right - left, 1e-8)
    weight_right = np.clip(weight_right, 0.0, 1.0)
    weight_left = 1.0 - weight_right

    basis = np.zeros((values.size, k), dtype=np.float32)
    rows = np.arange(values.size)
    basis[rows, idx] = weight_left
    basis[rows, idx + 1] = weight_right
    return basis.astype(np.float64)


def _spline_eval_and_deriv(s: np.ndarray, knots: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = np.asarray(s, dtype=np.float32).reshape(-1)
    knots = np.asarray(knots, dtype=np.float32)
    theta = np.asarray(theta, dtype=np.float32)
    idx = np.searchsorted(knots, flat, side="right") - 1
    idx = np.clip(idx, 0, knots.size - 2)
    left = knots[idx]
    right = knots[idx + 1]
    denom = np.maximum(right - left, 1e-8)
    wr = np.clip((flat - left) / denom, 0.0, 1.0)
    wl = 1.0 - wr
    y = wl * theta[idx] + wr * theta[idx + 1]
    dy = (theta[idx + 1] - theta[idx]) / denom
    return y.reshape(s.shape).astype(np.float32), dy.reshape(s.shape).astype(np.float32)


def _normalize_tucker(core: np.ndarray, factors: list[np.ndarray], eps: float = 1e-12) -> tuple[np.ndarray, list[np.ndarray]]:
    core_new = np.asarray(core, dtype=np.float32)
    out = []
    for mode, factor in enumerate(factors):
        scale = np.maximum(np.linalg.norm(factor, axis=0), eps).astype(np.float32)
        out.append((factor / scale[None, :]).astype(np.float32))
        moved = np.moveaxis(core_new, mode, 0)
        moved = moved * scale.reshape((-1,) + (1,) * (moved.ndim - 1))
        core_new = np.moveaxis(moved, 0, mode)
    return core_new, out


def _adam_init(shape: tuple[int, ...]) -> dict[str, Any]:
    return {"m": np.zeros(shape, dtype=np.float32), "v": np.zeros(shape, dtype=np.float32), "t": 0}


def _adam_step(param: np.ndarray, grad: np.ndarray, state: dict[str, Any], *, lr: float) -> None:
    state["t"] += 1
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    state["m"] = b1 * state["m"] + (1.0 - b1) * grad
    state["v"] = b2 * state["v"] + (1.0 - b2) * (grad * grad)
    mhat = state["m"] / (1.0 - b1 ** state["t"])
    vhat = state["v"] / (1.0 - b2 ** state["t"])
    param -= lr * mhat / (np.sqrt(vhat) + eps)


def _load_cave_scene(scene_id: int, target_shape: tuple[int, int]) -> tuple[str, np.ndarray]:
    dataset = CAVEHSIData(
        path="data/CAVE",
        id=int(scene_id),
        target_shape=target_shape,
        crop_shape=None,
    )
    BiasFilter(seed=0, normalize_method="max", snr_db=None, bias=None)(dataset)
    scene_name = str(getattr(dataset, "scene_name", f"scene-{scene_id}"))
    cube = np.asarray(dataset.get(split="eval").dense, dtype=np.float32)
    return scene_name, cube


def _metrics(original: np.ndarray, reconstruction: np.ndarray) -> dict[str, float]:
    target = Tensor(shape=original.shape, dense=original)
    recon = Tensor(shape=reconstruction.shape, dense=reconstruction)
    return {
        "RMSE": val_RMSE(target, recon),
        "SAM": val_SAM(target, recon),
        "NMSE_dB": val_NMSE_dB(target, recon),
    }


def _fit_tucker(cube: np.ndarray, rank: tuple[int, int, int], n_iter_max: int) -> tuple[np.ndarray, float, int]:
    model = TuckerDecomposition(rank=rank, n_iter_max=n_iter_max, init="svd", tol=1e-4)
    tensor = Tensor(shape=cube.shape, dense=cube)
    start = perf_counter()
    model.fit(tensor, mask=None, logcallback=LogCallback(log_level=0))
    elapsed = perf_counter() - start
    return np.asarray(model.reconstruct().dense, dtype=np.float32), elapsed, model.get_num_params()


def _fit_ntdpl(
    cube: np.ndarray,
    rank: tuple[int, int, int],
    n_iter_max: int,
    *,
    link_kind: str = "power",
    p_max: int = 4,
) -> tuple[np.ndarray, float, int]:
    model = NTDPLDecomposition(
        rank=rank,
        init_n_iter_max=50,
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
        link_kind=link_kind,
    )
    tensor = Tensor(shape=cube.shape, dense=cube)
    start = perf_counter()
    model.fit(tensor, mask=None, logcallback=LogCallback(log_level=0))
    elapsed = perf_counter() - start
    return np.asarray(model.reconstruct().dense, dtype=np.float32), elapsed, model.get_num_params()


def _run_scene(scene_id: int, target_shape: tuple[int, int], rank: tuple[int, int, int], n_iter_max: int) -> list[dict[str, Any]]:
    scene_name, cube = _load_cave_scene(scene_id, target_shape)
    rows = []

    for method_name, fit_fn in (
        ("tucker", lambda x: _fit_tucker(x, rank, n_iter_max)),
        ("ntdpl", lambda x: _fit_ntdpl(x, rank, n_iter_max)),
        ("spline", lambda x: _fit_ntdpl(x, rank, n_iter_max, link_kind="spline", p_max=7)),
    ):
        recon, fit_time, params = fit_fn(cube)
        row = {
            "scene_id": int(scene_id),
            "scene_name": scene_name,
            "method": method_name,
            "method_label": METHOD_LABELS[method_name],
            "rank": str(rank),
            "target_shape": str(target_shape),
            "params": int(params),
            "fit_time_sec": float(fit_time),
            **_metrics(cube, recon),
        }
        rows.append(row)
    print(f"Finished scene {scene_id:02d} ({scene_name})")
    return rows


def _spline_param_count(shape: tuple[int, ...], rank: tuple[int, ...], n_knots: int) -> int:
    return int(np.prod(rank) + sum(dim * rk for dim, rk in zip(shape, rank)) + n_knots)


def _parse_scene_ids(text: str) -> list[int]:
    if "-" in text:
        lo, hi = [int(part.strip()) for part in text.split("-", 1)]
        return list(range(lo, hi + 1))
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    out = (
        frame.groupby(["method", "method_label"], as_index=False)
        .agg(
            params=("params", "mean"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            SAM_mean=("SAM", "mean"),
            SAM_std=("SAM", "std"),
            NMSE_dB_mean=("NMSE_dB", "mean"),
            NMSE_dB_std=("NMSE_dB", "std"),
            fit_time_sec_mean=("fit_time_sec", "mean"),
            fit_time_sec_std=("fit_time_sec", "std"),
            n_scenes=("scene_id", "nunique"),
        )
    )
    out["order"] = out["method"].map(METHOD_ORDER)
    return out.sort_values("order").drop(columns="order").reset_index(drop=True)


def _pm(mean: float, std: float, digits: int) -> str:
    if not np.isfinite(std):
        std = 0.0
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def _to_latex(summary: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{@{}l r c c c@{}}",
        r"\toprule",
        r"Method & Params & RMSE$\downarrow$ & SAM$\downarrow$ & Time$\downarrow$ \\",
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"{row.method_label} & "
            f"{int(round(row.params / 1000.0))}k & "
            f"{_pm(float(row.RMSE_mean), float(row.RMSE_std), 4)} & "
            f"{_pm(float(row.SAM_mean), float(row.SAM_std), 2)} & "
            f"{_pm(float(row.fit_time_sec_mean), float(row.fit_time_sec_std), 1)}s\\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a lightweight CAVE joint-link family ablation.")
    parser.add_argument("--scene-ids", default="1-3")
    parser.add_argument("--target-shape", default="256,256")
    parser.add_argument("--rank", default="18,18,3")
    parser.add_argument("--n-iter-max", type=int, default=120)
    parser.add_argument("--jobs", type=int, default=max(1, min(3, (os.cpu_count() or 2) // 4)))
    parser.add_argument("--out-prefix", default="papers/neurips/tables/cave_joint_link_ablation")
    args = parser.parse_args()

    scene_ids = _parse_scene_ids(args.scene_ids)
    target_shape = tuple(int(part.strip()) for part in args.target_shape.split(","))  # type: ignore[assignment]
    rank = tuple(int(part.strip()) for part in args.rank.split(","))  # type: ignore[assignment]
    jobs = max(1, min(int(args.jobs), len(scene_ids)))

    all_rows: list[dict[str, Any]] = []
    if jobs == 1:
        for scene_id in scene_ids:
            all_rows.extend(_run_scene(scene_id, target_shape, rank, args.n_iter_max))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = [
                executor.submit(_run_scene, scene_id, target_shape, rank, args.n_iter_max)
                for scene_id in scene_ids
            ]
            for future in as_completed(futures):
                all_rows.extend(future.result())

    frame = pd.DataFrame(all_rows)
    frame["order"] = frame["method"].map(METHOD_ORDER)
    frame = frame.sort_values(["scene_id", "order"]).drop(columns="order").reset_index(drop=True)
    summary = _summary(frame)

    out_prefix = PROJECT_ROOT / args.out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_prefix.with_suffix(".per_scene.csv"), index=False)
    summary.to_csv(out_prefix.with_suffix(".summary.csv"), index=False)
    out_prefix.with_suffix(".tex").write_text(_to_latex(summary), encoding="utf-8")
    print(f"Wrote {out_prefix}.per_scene.csv, .summary.csv, and .tex")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
