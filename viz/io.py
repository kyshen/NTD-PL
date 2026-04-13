from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.hsi import CAVEHSIData
from src.filters.bias import BiasFilter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MULTIRUN_ROOT = PROJECT_ROOT / "multirun"
LEGACY_OUTPUT_ROOT = PROJECT_ROOT / "experiment" / "outputs"
FIGURE_ROOT = LEGACY_OUTPUT_ROOT / "figures"


def maybe_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series
    converted = pd.to_numeric(series, errors="coerce")
    return converted if converted.notna().any() else series


def load_runs(exp_name: str) -> pd.DataFrame:
    return pd.read_parquet(MULTIRUN_ROOT / exp_name / "runs.parquet")


def load_curves(exp_name: str) -> pd.DataFrame:
    return pd.read_parquet(MULTIRUN_ROOT / exp_name / "curves.parquet")


def load_output_csv(exp_name: str, file_name: str) -> pd.DataFrame:
    return pd.read_csv(LEGACY_OUTPUT_ROOT / exp_name / file_name)


def load_output_text(exp_name: str, file_name: str) -> str:
    return (LEGACY_OUTPUT_ROOT / exp_name / file_name).read_text(encoding="utf-8")


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {key: np.asarray(value) for key, value in payload.items()}


def load_state(path: str | Path) -> dict[str, Any]:
    from scipy.io import loadmat

    resolved = resolve_path(path)
    raw = loadmat(resolved)
    return {key: value for key, value in raw.items() if not key.startswith("__")}


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def jsonish(value: Any) -> Any:
    result = value
    while isinstance(result, str):
        stripped = result.strip()
        if not stripped:
            return stripped
        try:
            loaded = json.loads(stripped)
        except json.JSONDecodeError:
            return result
        if loaded == result:
            return loaded
        result = loaded
    return result


def parse_rank(value: Any) -> tuple[int, int, int]:
    parsed = jsonish(value)
    if isinstance(parsed, str):
        tokens = [item.strip() for item in parsed.strip("[]()").split(",") if item.strip()]
        return tuple(int(token) for token in tokens)  # type: ignore[return-value]
    if isinstance(parsed, (list, tuple)):
        return tuple(int(item) for item in parsed)  # type: ignore[return-value]
    raise ValueError(f"Cannot parse rank from value {value!r}")


def rank_text(rank: tuple[int, int, int]) -> str:
    return f"({rank[0]},{rank[1]},{rank[2]})"


def extract_factors(state: dict[str, Any]) -> list[np.ndarray]:
    factors = state["factors"]
    return [np.asarray(factors[0, idx], dtype=float) for idx in range(factors.shape[1])]


def mode_n_product(tensor: np.ndarray, matrix: np.ndarray, mode: int) -> np.ndarray:
    moved = np.moveaxis(tensor, mode, 0)
    reshaped = moved.reshape(moved.shape[0], -1)
    product = matrix @ reshaped
    output_shape = (matrix.shape[0],) + moved.shape[1:]
    return np.moveaxis(product.reshape(output_shape), 0, mode)


def reconstruct_tucker(state: dict[str, Any]) -> np.ndarray:
    tensor = np.asarray(state["core"], dtype=float)
    for mode, factor in enumerate(extract_factors(state)):
        tensor = mode_n_product(tensor, factor, mode)
    return np.asarray(tensor, dtype=float)


def apply_polynomial(values: np.ndarray, beta: np.ndarray) -> np.ndarray:
    coeffs = np.asarray(beta, dtype=float).reshape(-1)
    if coeffs.size == 0:
        return np.asarray(values, dtype=float)
    output = np.full_like(values, coeffs[-1], dtype=float)
    for coefficient in coeffs[-2::-1]:
        output = output * values + coefficient
    return output


def reconstruct_observation(state: dict[str, Any]) -> np.ndarray:
    if "reconstruction" in state:
        return np.asarray(state["reconstruction"], dtype=float)
    fitted = np.asarray(state.get("fitted", []), dtype=float)
    if fitted.ndim == 3 and fitted.size:
        return fitted
    latent = reconstruct_tucker(state)
    beta = np.asarray(state.get("beta", []), dtype=float).reshape(-1)
    if beta.size == 0:
        return latent
    return apply_polynomial(latent, beta)


def observed_mask_from_state(state: dict[str, Any]) -> np.ndarray | None:
    if "observed_mask" not in state:
        return None
    return np.asarray(state["observed_mask"], dtype=bool)


def load_cave_scene(scene_id: int, *, target_shape: tuple[int, int] = (128, 128)) -> tuple[np.ndarray, str]:
    dataset = CAVEHSIData(path="data/CAVE", id=int(scene_id), target_shape=target_shape)
    BiasFilter(seed=0, normalize_method="max", snr_db=None, bias=None)(dataset)
    cube = np.asarray(dataset.get(split="eval").dense, dtype=float)
    scene_name = str(getattr(dataset, "scene_name", f"scene-{scene_id}"))
    return cube, scene_name


def pseudo_rgb(cube: np.ndarray) -> np.ndarray:
    band_count = cube.shape[-1]
    band_indices = [int(round((band_count - 1) * frac)) for frac in (0.75, 0.50, 0.20)]
    rgb = np.stack([cube[..., idx] for idx in band_indices], axis=-1)
    rgb = np.clip(rgb, 0.0, None)
    scale = float(np.max(rgb))
    if scale > 1e-12:
        rgb = rgb / scale
    return rgb


def rmse_map(reference: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    diff = np.asarray(reference, dtype=float) - np.asarray(estimate, dtype=float)
    return np.sqrt(np.mean(diff * diff, axis=-1))
