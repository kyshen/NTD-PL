from __future__ import annotations

import math
from pathlib import Path

from src.data.hsi import _load_hsi_from_file


CAVE_FULL_SHAPE = (512, 512, 31)
CAVE_RECON_RANKS = ((24, 24, 4), (33, 33, 4), (40, 40, 5))
CAVE_RECON_MAIN_RANK = CAVE_RECON_RANKS[1]

REFERENCE_SHAPE = (128, 128, 31)
REFERENCE_RANK = (12, 12, 6)

REAL_HSI_DATA_PATHS = {
    "jasper_ridge_hsi": "data/hsi/jasperRidge2_R198.mat",
    "samson_hsi": "data/hsi-similar/samson_1.img",
    "urban_hsi": "data/hsi-similar/Urban_R162.mat",
    "cuprite_hsi": "data/hsi-similar/Cuprite_S1_R188.img",
}


def reference_target_cr() -> float:
    size = math.prod(REFERENCE_SHAPE)
    params = math.prod(REFERENCE_RANK) + sum(dim * rank for dim, rank in zip(REFERENCE_SHAPE, REFERENCE_RANK))
    return size / params


def auto_rank_from_shape(shape: tuple[int, int, int]) -> tuple[int, int, int]:
    target_params = math.prod(shape) / reference_target_cr()
    fractions = [rank / dim for rank, dim in zip(REFERENCE_RANK, REFERENCE_SHAPE)]
    best: tuple[float, tuple[int, int, int]] | None = None
    for step in range(10, 121):
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
        raise RuntimeError(f"Failed to derive automatic rank for shape {shape}.")
    return best[1]


def load_dataset_shape(project_root: Path, dataset_name: str) -> tuple[int, int, int]:
    if dataset_name == "cave_hsi":
        return CAVE_FULL_SHAPE
    try:
        rel_path = REAL_HSI_DATA_PATHS[dataset_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported HSI dataset for shape lookup: {dataset_name}") from exc
    return tuple(int(v) for v in _load_hsi_from_file(project_root / rel_path).shape)


def completion_rank_for_dataset(project_root: Path, dataset_name: str) -> tuple[int, int, int]:
    if dataset_name == "cave_hsi":
        return CAVE_RECON_MAIN_RANK
    return auto_rank_from_shape(load_dataset_shape(project_root, dataset_name))
