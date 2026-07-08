from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.hsi import _load_hsi_from_file


DATASETS = [
    ("jasper_ridge_hsi", "Jasper Ridge", Path("data/hsi/jasperRidge2_R198.mat")),
    ("samson_hsi", "Samson", Path("data/hsi-similar/samson_1.img")),
    ("urban_hsi", "Urban", Path("data/hsi-similar/Urban_R162.mat")),
    ("cuprite_hsi", "Cuprite", Path("data/hsi-similar/Cuprite_S1_R188.img")),
]


def _normalize_global_max(cube: np.ndarray) -> np.ndarray:
    cube = np.asarray(cube, dtype=np.float32)
    scale = float(np.max(cube))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Invalid max scale: {scale}")
    return cube / scale


def _spectral_smoothness_metrics(cube: np.ndarray) -> dict[str, float]:
    first = np.diff(cube, axis=2)
    second = np.diff(cube, n=2, axis=2)
    mean_abs_first = float(np.mean(np.abs(first)))
    mean_abs_second = float(np.mean(np.abs(second)))
    rms_first = float(np.sqrt(np.mean(first**2)))
    rms_second = float(np.sqrt(np.mean(second**2)))
    curvature_ratio_l1 = mean_abs_second / (mean_abs_first + 1e-12)
    curvature_ratio_l2 = rms_second / (rms_first + 1e-12)
    return {
        "spectral_tv_l1": mean_abs_first,
        "spectral_curvature_l1": mean_abs_second,
        "spectral_curvature_ratio_l1": curvature_ratio_l1,
        "spectral_tv_l2": rms_first,
        "spectral_curvature_l2": rms_second,
        "spectral_curvature_ratio_l2": curvature_ratio_l2,
    }


def _spectral_subspace_metrics(cube: np.ndarray) -> dict[str, float]:
    x = cube.reshape(-1, cube.shape[2]).astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    cov = np.cov(x, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(np.sort(eigvals)[::-1], a_min=0.0, a_max=None)
    total = float(np.sum(eigvals))
    if total <= 0.0:
        raise ValueError("Degenerate spectral covariance")

    p = eigvals / total
    p = p[p > 0.0]
    effective_rank = float(np.exp(-np.sum(p * np.log(p))))
    cumulative = np.cumsum(eigvals) / total
    pc95 = int(np.searchsorted(cumulative, 0.95) + 1)
    pc99 = int(np.searchsorted(cumulative, 0.99) + 1)
    top10 = float(np.sum(eigvals[: min(10, len(eigvals))]) / total)
    top20 = float(np.sum(eigvals[: min(20, len(eigvals))]) / total)
    return {
        "spectral_effective_rank": effective_rank,
        "spectral_pc95": float(pc95),
        "spectral_pc99": float(pc99),
        "spectral_top10_energy": top10,
        "spectral_top20_energy": top20,
    }


def _boundary_metrics(cube: np.ndarray) -> dict[str, float]:
    dx = np.diff(cube, axis=0)
    dy = np.diff(cube, axis=1)
    dx_mid = dx[:, :-1, :]
    dy_mid = dy[:-1, :, :]
    boundary = np.sqrt(dx_mid**2 + dy_mid**2).mean(axis=2)
    return {
        "boundary_mean": float(np.mean(boundary)),
        "boundary_p90": float(np.percentile(boundary, 90.0)),
        "boundary_p95": float(np.percentile(boundary, 95.0)),
    }


def _neighbor_sam_metrics(cube: np.ndarray) -> dict[str, float]:
    x_right = cube[:, 1:, :]
    x_left = cube[:, :-1, :]
    x_down = cube[1:, :, :]
    x_up = cube[:-1, :, :]

    def pair_sam(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        dot = np.sum(a * b, axis=2)
        denom = np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2)
        cos = np.clip(dot / np.maximum(denom, 1e-12), -1.0, 1.0)
        return np.degrees(np.arccos(cos))

    sam_h = pair_sam(x_left, x_right).reshape(-1)
    sam_v = pair_sam(x_up, x_down).reshape(-1)
    sam = np.concatenate([sam_h, sam_v], axis=0)
    return {
        "neighbor_sam_mean_deg": float(np.mean(sam)),
        "neighbor_sam_p90_deg": float(np.percentile(sam, 90.0)),
        "neighbor_sam_p95_deg": float(np.percentile(sam, 95.0)),
    }


def _local_variability_metrics(cube: np.ndarray) -> dict[str, float]:
    center = cube[1:-1, 1:-1, :]
    neighbors = [
        cube[:-2, 1:-1, :],
        cube[2:, 1:-1, :],
        cube[1:-1, :-2, :],
        cube[1:-1, 2:, :],
    ]
    diff_sq = sum((center - n) ** 2 for n in neighbors) / float(len(neighbors))
    local_rmse = np.sqrt(np.mean(diff_sq, axis=2))
    return {
        "local_neighbor_rmse_mean": float(np.mean(local_rmse)),
        "local_neighbor_rmse_p90": float(np.percentile(local_rmse, 90.0)),
    }


def _load_model_behavior() -> pd.DataFrame:
    main = pd.read_csv("artifacts/paper-outputs/real-hsi-robustness/real_hsi_robustness_main_table.csv")
    main["Gain(%)"] = main["Gain(%)"].astype(float)
    main["Delta_NMSE(dB)"] = main["Delta_NMSE(dB)"].astype(float)
    pivot = (
        main.pivot_table(
            index="dataset",
            columns="task_label",
            values=["Gain(%)", "Delta_NMSE(dB)", "Delta_SAM"],
            aggfunc="first",
        )
        .sort_index(axis=1)
    )
    pivot.columns = [
        f"{metric.lower().replace('(%)', 'pct').replace('(db)', 'db').replace('delta_', 'delta_').replace(' ', '_')}_{task.lower().replace('.', '').replace(' ', '_')}"
        for metric, task in pivot.columns
    ]
    pivot = pivot.reset_index()
    return pivot


def analyze() -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for dataset, label, rel_path in DATASETS:
        cube = _normalize_global_max(_load_hsi_from_file(rel_path))
        row: dict[str, float | str] = {
            "dataset": dataset,
            "dataset_label": label,
            "shape": f"{cube.shape[0]}x{cube.shape[1]}x{cube.shape[2]}",
        }
        row.update(_spectral_smoothness_metrics(cube))
        row.update(_spectral_subspace_metrics(cube))
        row.update(_boundary_metrics(cube))
        row.update(_neighbor_sam_metrics(cube))
        row.update(_local_variability_metrics(cube))
        rows.append(row)

    frame = pd.DataFrame(rows)
    model = _load_model_behavior()
    frame = frame.merge(model, on="dataset", how="left")
    return frame


def main() -> None:
    out_dir = Path("artifacts/paper-outputs/real-hsi-robustness")
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = analyze()
    csv_path = out_dir / "real_hsi_scene_characteristics.csv"
    frame.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(frame)


if __name__ == "__main__":
    main()
