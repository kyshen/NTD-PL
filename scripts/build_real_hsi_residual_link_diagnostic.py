from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
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
import pandas as pd
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment.hsi_defaults import REAL_HSI_DATA_PATHS, completion_rank_for_dataset
from src.data.hsi import _load_hsi_from_file
from src.methods.tucker import TuckerDecomposition
from src.metrics import val_RMSE, val_SAM
from src.types import LogCallback, Tensor


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    label: str


DATASETS = (
    DatasetSpec("jasper_ridge_hsi", "Jasper Ridge"),
    DatasetSpec("samson_hsi", "Samson"),
    DatasetSpec("urban_hsi", "Urban"),
    DatasetSpec("cuprite_hsi", "Cuprite"),
)


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


def _normalize_global_max(cube: np.ndarray) -> np.ndarray:
    cube = np.asarray(cube, dtype=np.float32)
    scale = float(np.max(cube))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Invalid global max scale: {scale}")
    return cube / scale


def _compression_ratio(shape: tuple[int, int, int], rank: tuple[int, int, int]) -> float:
    params = int(np.prod(rank)) + sum(int(dim) * int(rk) for dim, rk in zip(shape, rank))
    return float(np.prod(shape)) / max(float(params), 1.0)


def _residual_link_score_from_rmse(rmse_tucker: float, rmse_poly: float) -> float:
    ratio = (float(rmse_poly) ** 2) / max(float(rmse_tucker) ** 2, 1e-12)
    ratio = max(ratio, 1e-12)
    return 10.0 * np.log10(1.0 / ratio)


def _fit_tucker(cube: np.ndarray, rank: tuple[int, int, int], *, n_iter_max: int) -> np.ndarray:
    method = TuckerDecomposition(
        rank=rank,
        n_iter_max=int(n_iter_max),
        init="svd",
        tol=1e-7,
    )
    tensor = Tensor(shape=cube.shape, dense=cube)
    method.fit(tensor, mask=None, logcallback=LogCallback(log_level=0))
    return np.asarray(method.reconstruct().dense, dtype=np.float32)


def _poly_design(x: np.ndarray, degree: int) -> np.ndarray:
    return np.vander(x, N=int(degree) + 1, increasing=True)


def _fit_scalar_poly_predict(
    x_pred: np.ndarray,
    target: np.ndarray,
    *,
    degree: int,
    lambda_reg: float,
    sample_size: int,
    seed: int,
) -> np.ndarray:
    x = np.asarray(x_pred, dtype=np.float64).reshape(-1)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(seed)
    if 0 < sample_size < x.size:
        idx = rng.choice(x.size, size=int(sample_size), replace=False)
        x_fit = x[idx]
        y_fit = y[idx]
    else:
        x_fit = x
        y_fit = y

    phi = _poly_design(x_fit, degree)
    col_scales = np.maximum(np.linalg.norm(phi, axis=0), 1e-12)
    phi_scaled = phi / col_scales
    gram = phi_scaled.T @ phi_scaled
    rhs = phi_scaled.T @ y_fit
    coeff_scaled = np.linalg.solve(gram + float(lambda_reg) * np.eye(gram.shape[0]), rhs)
    coeff = coeff_scaled / col_scales

    pred = np.zeros_like(x, dtype=np.float64)
    for c in coeff[::-1]:
        pred = pred * x + float(c)
    return pred.reshape(x_pred.shape).astype(np.float32)


def _load_gain_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing gain table: {path}. Run the real-hsi-robustness postprocess first."
        )
    frame = pd.read_csv(path)
    return frame.loc[frame["task_name"].eq("decompose")].copy()


def _format_table(frame: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{@{}l c c c@{}}",
        r"\toprule",
        r"Dataset & Residual-link score & NTD-PL gain & \(\Delta\)SAM \\",
        r"\midrule",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"{row.dataset_label} & "
            f"{row.link_score:.2f} & "
            f"{row.ntdpl_gain_pct:.1f}\\% & "
            f"{row.delta_sam:.2f}\\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def merge_into_main_table(main_csv: Path, out_tex: Path, frame: pd.DataFrame) -> None:
    main = pd.read_csv(main_csv).copy()
    link = frame.loc[:, ["dataset", "link_score"]].copy()
    merged = main.merge(link, on="dataset", how="left")
    lines = [
        r"\begin{tabular}{@{}l c c c c c@{}}",
        r"\toprule",
        r"Dataset & CR & Tucker & NTD-PL & Gain & $S_{\mathrm{link}}$ \\",
        r"\midrule",
    ]
    for row in merged.itertuples(index=False):
        dataset_name = str(row.dataset)
        shape = tuple(int(v) for v in _load_hsi_from_file(PROJECT_ROOT / REAL_HSI_DATA_PATHS[dataset_name]).shape)
        rank = completion_rank_for_dataset(PROJECT_ROOT, dataset_name)
        cr = _compression_ratio(shape, rank)
        tucker = float(row.tucker_rmse)
        ntdpl = float(row.ntdpl_rmse)
        gain = float(row.gain_pct)
        link_score = float(row.link_score)
        tucker_text = f"{tucker:.4f}"
        ntdpl_text = f"{ntdpl:.4f}"
        if tucker <= ntdpl:
            tucker_text = rf"\textbf{{{tucker_text}}}"
        else:
            ntdpl_text = rf"\textbf{{{ntdpl_text}}}"
        gain_text = f"{gain:.2f}\\%"
        if gain > 0.0:
            gain_text = rf"\textbf{{{gain_text}}}"
        link_text = f"{link_score:.2f}"
        if link_score > 0.0:
            link_text = rf"\textbf{{{link_text}}}"
        lines.append(
            " & ".join(
                [
                    str(row.dataset_label),
                    f"{cr:.1f}",
                    tucker_text,
                    ntdpl_text,
                    gain_text,
                    link_text,
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_summary(frame: pd.DataFrame) -> str:
    spearman = spearmanr(frame["link_score"], frame["ntdpl_gain_pct"])
    pearson = pearsonr(frame["link_score"], frame["ntdpl_gain_pct"])
    lines = [
        r"\begin{tabular}{@{}l c c@{}}",
        r"\toprule",
        r"Diagnostic & Correlation with NTD-PL RMSE gain & \(p\)-value \\",
        r"\midrule",
        f"Spearman & {float(spearman.statistic):.3f} & {float(spearman.pvalue):.3f}\\\\",
        f"Pearson & {float(pearson.statistic):.3f} & {float(pearson.pvalue):.3f}\\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines) + "\n"


def _build_one(
    spec: DatasetSpec,
    *,
    degree: int,
    lambda_reg: float,
    n_iter_max: int,
    sample_size: int,
    seed: int,
    gains: pd.DataFrame,
) -> dict[str, float | str]:
    _cpu_worker_env()
    rel = REAL_HSI_DATA_PATHS[spec.name]
    cube = _normalize_global_max(_load_hsi_from_file(PROJECT_ROOT / rel))
    rank = completion_rank_for_dataset(PROJECT_ROOT, spec.name)
    tucker = _fit_tucker(cube, rank, n_iter_max=n_iter_max)
    poly = _fit_scalar_poly_predict(
        tucker,
        cube,
        degree=degree,
        lambda_reg=lambda_reg,
        sample_size=sample_size,
        seed=seed,
    )

    target = Tensor(shape=cube.shape, dense=cube)
    tucker_tensor = Tensor(shape=cube.shape, dense=tucker)
    poly_tensor = Tensor(shape=cube.shape, dense=poly)
    rmse_tucker = val_RMSE(target, tucker_tensor)
    rmse_poly = val_RMSE(target, poly_tensor)
    sam_tucker = val_SAM(target, tucker_tensor)
    sam_poly = val_SAM(target, poly_tensor)

    residual = cube - tucker
    corrected_residual = cube - poly
    residual_energy = float(np.mean(residual**2))
    residual_explained = 1.0 - float(np.mean(corrected_residual**2)) / max(residual_energy, 1e-12)
    link_score = _residual_link_score_from_rmse(rmse_tucker, rmse_poly)

    gain_row = gains.loc[gains["dataset"].eq(spec.name)].iloc[0]
    return {
        "dataset": spec.name,
        "dataset_label": spec.label,
        "rank": str(rank),
        "rmse_tucker_refit": rmse_tucker,
        "rmse_polycal_refit": rmse_poly,
        "sam_tucker_refit": sam_tucker,
        "sam_polycal_refit": sam_poly,
        "link_score": link_score,
        "residual_explained": residual_explained,
        "ntdpl_gain_pct": float(gain_row["gain_pct"]),
        "delta_sam": float(gain_row["delta_sam"]),
    }


def build(
    *,
    degree: int,
    lambda_reg: float,
    n_iter_max: int,
    sample_size: int,
    seed: int,
    gain_csv: Path,
    jobs: int,
) -> pd.DataFrame:
    gains = _load_gain_table(gain_csv)
    jobs = max(1, min(int(jobs), len(DATASETS)))
    if jobs == 1:
        rows = [
            _build_one(
                spec,
                degree=degree,
                lambda_reg=lambda_reg,
                n_iter_max=n_iter_max,
                sample_size=sample_size,
                seed=seed,
                gains=gains,
            )
            for spec in DATASETS
        ]
    else:
        rows = []
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = [
                executor.submit(
                    _build_one,
                    spec,
                    degree=degree,
                    lambda_reg=lambda_reg,
                    n_iter_max=n_iter_max,
                    sample_size=sample_size,
                    seed=seed,
                    gains=gains,
                )
                for spec in DATASETS
            ]
            for future in as_completed(futures):
                row = future.result()
                print(f"Finished {row['dataset_label']}")
                rows.append(row)
    order = {spec.name: idx for idx, spec in enumerate(DATASETS)}
    return (
        pd.DataFrame(rows)
        .assign(_order=lambda frame: frame["dataset"].map(order))
        .sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build real-HSI link-yield diagnostics for NTD-PL."
    )
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--lambda-reg", type=float, default=1e-6)
    parser.add_argument("--n-iter-max", type=int, default=300)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=400_000,
        help="Number of entries used to fit the scalar polynomial. Use <=0 for all entries.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--gain-csv",
        type=Path,
        default=PROJECT_ROOT
        / "experiment/outputs/real-hsi-robustness/real_hsi_robustness_main_table_numeric.csv",
    )
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=PROJECT_ROOT / "neurips/tables/real_hsi_residual_link_diagnostic",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, min(len(DATASETS), (os.cpu_count() or 2) // 2)),
        help="Number of datasets to process in parallel. BLAS threads are capped per worker.",
    )
    args = parser.parse_args()

    _cpu_worker_env()
    frame = build(
        degree=args.degree,
        lambda_reg=args.lambda_reg,
        n_iter_max=args.n_iter_max,
        sample_size=args.sample_size,
        seed=args.seed,
        gain_csv=args.gain_csv,
        jobs=args.jobs,
    )
    out_prefix = args.out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_prefix.with_suffix(".csv"), index=False)
    out_prefix.with_suffix(".tex").write_text(_format_table(frame), encoding="utf-8")
    out_prefix.with_suffix(".summary.tex").write_text(_format_summary(frame), encoding="utf-8")
    merge_into_main_table(
        PROJECT_ROOT / "experiment/outputs/real-hsi-robustness/real_hsi_robustness_main_table_numeric.csv",
        PROJECT_ROOT / "neurips/tables/real_hsi_robustness_main.tex",
        frame,
    )
    print(f"Wrote {out_prefix}.csv, .tex, and .summary.tex")


if __name__ == "__main__":
    main()
