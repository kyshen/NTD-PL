from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_run_parquets(results_dir: str | Path) -> dict[str, pd.DataFrame]:
    results_path = Path(results_dir)
    return {
        "runs": pd.read_parquet(results_path / "runs.parquet"),
        "curves": pd.read_parquet(results_path / "curves.parquet"),
    }


def maybe_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series
    converted = pd.to_numeric(series, errors="coerce")
    if converted.notna().any():
        return converted
    return series


def load_state_mat(path: str | Path) -> dict[str, Any]:
    try:
        from scipy.io import loadmat  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing optional dependency 'scipy'. Install it to load .mat files (e.g., `pip install scipy`)."
        ) from exc

    data = loadmat(path)
    return {key: value for key, value in data.items() if not key.startswith("__")}
