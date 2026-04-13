from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import re

import numpy as np
import pandas as pd

from ..config import get_env
from ..utils.io import load_run_parquets, maybe_numeric


@dataclass(frozen=True)
class LoadedResults:
    runs: pd.DataFrame
    curves: pd.DataFrame
    env: object


@dataclass(frozen=True)
class ResultSlice:
    setting: str
    key: str
    subset: pd.DataFrame
    fixed: dict[str, str]

    @property
    def key_name(self) -> str:
        return self.key.removeprefix("ovr.")

    @property
    def key_slug(self) -> str:
        return Path(self.key).name.replace(".", "_")

    @property
    def fixed_suffix(self) -> str:
        return fixed_value_suffix(self.fixed)

    def stem(self, prefix: str, *, default: str | None = None) -> str:
        stem = default or f"{prefix}_{self.key_slug}"
        if self.fixed_suffix:
            return f"{stem}_{self.fixed_suffix}"
        return stem


def load_results(exp_name: str, *, require_curves: bool = True) -> LoadedResults:
    env = get_env(exp_name)
    payload = load_run_parquets(env.results_dir)
    runs = payload["runs"].copy()
    curves = payload["curves"].copy()
    if runs.empty or (require_curves and curves.empty):
        suffix = " runs/curves" if require_curves else " runs"
        raise RuntimeError(
            f"No{suffix} found for {exp_name}. Run `python -m experiment {exp_name} run` first."
        )
    return LoadedResults(runs=runs, curves=curves, env=env)


def bool_mask(series: pd.Series, expected: bool) -> np.ndarray:
    return series.astype(str).str.lower().eq(str(expected).lower()).to_numpy()


def float_mask(series: pd.Series, expected: float, *, atol: float = 1e-12) -> np.ndarray:
    return np.isclose(
        maybe_numeric(series).to_numpy(dtype=float),
        expected,
        rtol=0.0,
        atol=atol,
        equal_nan=False,
    )


def parse_setting_sweep(setting: str) -> tuple[str, list[str]]:
    text = str(setting).strip()
    if "=" not in text:
        raise ValueError(
            "Invalid setting sweep. Expected format like 'filter.bias=0,0.5' (key=value1,value2,...)"
        )
    key, raw_values = text.split("=", 1)
    key = key.strip()
    values = [item.strip() for item in raw_values.split(",") if item.strip()]
    if not key or not values:
        raise ValueError(
            "Invalid setting sweep. Expected format like 'filter.bias=0,0.5' (key=value1,value2,...)"
        )
    return key, values


def resolve_setting_series(frame: pd.DataFrame, key: str) -> pd.Series:
    key = key.strip()
    if key in frame.columns:
        return frame[key]

    if key.startswith("ovr."):
        base_key = key.removeprefix("ovr.")
        if base_key in frame.columns:
            return frame[base_key]
    else:
        ovr_key = f"ovr.{key}"
        if ovr_key in frame.columns:
            return frame[ovr_key]

    raise KeyError(f"Setting key not found in runs table: {key}")


def setting_value_mask(frame: pd.DataFrame, key: str, value: str) -> np.ndarray:
    series = resolve_setting_series(frame, key)
    value_text = str(value).strip()
    if value_text == "":
        return np.zeros(len(frame), dtype=bool)

    try:
        numeric_value = float(value_text)
    except ValueError:
        numeric_value = None

    if numeric_value is not None:
        return np.isclose(
            maybe_numeric(series).to_numpy(dtype=float),
            numeric_value,
            rtol=0.0,
            atol=1e-12,
            equal_nan=False,
        )

    return series.astype(str).eq(value_text).to_numpy()


def series_with_fallback(
    frame: pd.DataFrame,
    override_col: str,
    base_col: str,
    default: float | str,
) -> pd.Series:
    if override_col in frame.columns:
        return frame[override_col]
    if base_col in frame.columns:
        return frame[base_col]
    return pd.Series([default] * len(frame), index=frame.index)


def select_method_runs(
    frame: pd.DataFrame,
    method: str,
    *,
    method_col: str = "ovr.method",
    p_max: int | float | None = None,
    p_max_col: str = "ovr.method.p_max",
) -> pd.DataFrame:
    selected = frame.loc[frame[method_col].astype(str) == str(method)].copy()
    if p_max is None or selected.empty or p_max_col not in selected.columns:
        return selected
    return selected.loc[float_mask(selected[p_max_col], float(p_max))].copy()


def varying_setting_keys(
    frame: pd.DataFrame,
    *,
    include_prefixes: tuple[str, ...] = ("ovr.",),
    exclude_keys: set[str] | None = None,
) -> list[str]:
    exclude_keys = exclude_keys or set()
    keys: list[str] = []
    for column in frame.columns:
        if column in exclude_keys:
            continue
        if not any(column.startswith(prefix) for prefix in include_prefixes):
            continue
        numeric = maybe_numeric(frame[column])
        if numeric.notna().any():
            values = numeric.dropna().unique()
        else:
            values = frame[column].dropna().astype(str).unique()
        if len(values) > 1:
            keys.append(column)
    return sorted(keys)


def format_setting_sweep(frame: pd.DataFrame, key: str) -> str:
    series = resolve_setting_series(frame, key)
    numeric = maybe_numeric(series)
    if numeric.notna().any():
        values = sorted({float(value) for value in numeric.dropna().to_numpy(dtype=float)})
        value_text = ",".join(f"{value:g}" for value in values)
    else:
        values = sorted({str(value) for value in series.dropna().astype(str).tolist()})
        value_text = ",".join(values)
    if not value_text:
        raise ValueError(f"No values available to build a setting sweep for '{key}'.")
    normalized_key = key.removeprefix("ovr.")
    return f"{normalized_key}={value_text}"


def infer_setting_sweep(
    frame: pd.DataFrame,
    *,
    preferred_key: str | None = None,
    exclude_keys: set[str] | None = None,
) -> str:
    exclude_keys = exclude_keys or set()
    if preferred_key is not None:
        return format_setting_sweep(frame, preferred_key)

    varying = varying_setting_keys(frame, exclude_keys=exclude_keys)
    if not varying:
        raise RuntimeError("No varying override settings found.")
    if len(varying) > 1:
        joined = ", ".join(varying)
        raise RuntimeError(f"Multiple varying override settings found; choose one explicitly: {joined}")
    return format_setting_sweep(frame, varying[0])


def infer_setting_sweeps(
    frame: pd.DataFrame,
    *,
    include_prefixes: tuple[str, ...] = ("ovr.",),
    exclude_keys: set[str] | None = None,
) -> list[str]:
    varying = varying_setting_keys(frame, include_prefixes=include_prefixes, exclude_keys=exclude_keys)
    if not varying:
        raise RuntimeError("No varying override settings found.")
    return [format_setting_sweep(frame, key) for key in varying]


def setting_values(frame: pd.DataFrame, key: str) -> list[str]:
    _, values = parse_setting_sweep(format_setting_sweep(frame, key))
    return values


def value_slug(value: object) -> str:
    text = str(value).strip()
    if not text:
        return "empty"
    try:
        numeric = float(text)
    except ValueError:
        slug = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
        return slug or "value"

    return f"{numeric:.2f}".replace("-", "m").replace(".", "p")


def fixed_value_suffix(fixed: dict[str, str]) -> str:
    parts = [
        f"{Path(name).name.replace('.', '_')}_{value_slug(value)}"
        for name, value in fixed.items()
    ]
    return "_".join(part for part in parts if part)


def cartesian_setting_slices(
    frame: pd.DataFrame,
    *,
    target_key: str,
    include_prefixes: tuple[str, ...] = ("ovr.",),
    exclude_keys: set[str] | None = None,
) -> list[tuple[pd.DataFrame, dict[str, str]]]:
    exclude_keys = exclude_keys or set()
    normalized_target_key = target_key.removeprefix("ovr.")
    other_keys = [
        key
        for key in varying_setting_keys(frame, include_prefixes=include_prefixes, exclude_keys=exclude_keys)
        if key.removeprefix("ovr.") != normalized_target_key
    ]
    if not other_keys:
        return [(frame.copy(), {})]

    value_grid = [setting_values(frame, key) for key in other_keys]
    slices: list[tuple[pd.DataFrame, dict[str, str]]] = []
    for values in product(*value_grid):
        subset = frame
        fixed: dict[str, str] = {}
        for key, value in zip(other_keys, values, strict=False):
            subset = subset[setting_value_mask(subset, key, value)].copy()
            fixed[key.removeprefix("ovr.")] = value
        if not subset.empty:
            slices.append((subset, fixed))
    return slices


def iter_result_slices(
    frame: pd.DataFrame,
    *,
    include_prefixes: tuple[str, ...] = ("ovr.",),
    exclude_keys: set[str] | None = None,
) -> list[ResultSlice]:
    slices: list[ResultSlice] = []
    for setting in infer_setting_sweeps(
        frame,
        include_prefixes=include_prefixes,
        exclude_keys=exclude_keys,
    ):
        key, _ = parse_setting_sweep(setting)
        for subset, fixed in cartesian_setting_slices(
            frame,
            target_key=key,
            include_prefixes=include_prefixes,
            exclude_keys=exclude_keys,
        ):
            slices.append(ResultSlice(setting=setting, key=key, subset=subset, fixed=fixed))
    return slices


def iter_setting_subsets(
    frame: pd.DataFrame,
    *,
    include_prefixes: tuple[str, ...] = ("ovr.",),
    exclude_keys: set[str] | None = None,
) -> list[tuple[str, str, pd.DataFrame, dict[str, str]]]:
    return [
        (item.setting, item.key, item.subset, item.fixed)
        for item in iter_result_slices(
            frame,
            include_prefixes=include_prefixes,
            exclude_keys=exclude_keys,
        )
    ]
