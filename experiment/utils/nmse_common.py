from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ..config import ExperimentEnv
from .io import load_run_parquets, maybe_numeric


DEFAULT_METHOD_ORDER: list[str] = ["tucker", "cp", "tt", "tr", "ntdpl"]


def load_runs(env: ExperimentEnv) -> pd.DataFrame:
    try:
        runs = load_run_parquets(env.results_dir)["runs"].copy()
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Missing parquet results for {env.exp_name} at {Path(env.results_dir)!s}. "
            f"Run `python -m experiment {env.exp_name} run` (and `python collect.py --exp={env.exp_name}` if needed) first."
        ) from exc
    if runs.empty:
        raise RuntimeError(
            f"No runs found for {env.exp_name}. Run `python -m experiment {env.exp_name} run` first."
        )
    return runs


def curve_band_summary(frame: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    seed_col = "ovr.data.seed" if "ovr.data.seed" in frame.columns else "run_id"
    grouped = (
        frame.groupby([seed_col, x_col], as_index=False)[y_col]
        .mean()
        .rename(columns={y_col: "seed_value"})
    )
    summary = grouped.groupby(x_col, as_index=False)["seed_value"].agg(["mean", "std"]).reset_index()
    summary["std"] = summary["std"].fillna(0.0)
    summary["lower"] = summary["mean"] - summary["std"]
    summary["upper"] = summary["mean"] + summary["std"]
    return summary.sort_values(x_col)


def mean_curve(frame: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    seed_col = "ovr.data.seed" if "ovr.data.seed" in frame.columns else "run_id"
    grouped = frame.groupby([seed_col, x_col], as_index=False)[y_col].mean()
    return grouped.groupby(x_col, as_index=False)[y_col].mean().sort_values(x_col)


def sorted_unique(values: pd.Series) -> list[float]:
    numeric = maybe_numeric(values).dropna()
    uniq = np.sort(pd.unique(numeric))
    return [float(v) for v in uniq]


def filter_levels(
    levels: list[float],
    allow: Sequence[float] | None,
    span: tuple[float, float] | None,
) -> list[float]:
    selected = levels
    if span is not None:
        lo, hi = span
        selected = [v for v in selected if lo <= float(v) <= hi]
    if allow is not None:
        allow_set = {float(v) for v in allow}
        selected = [v for v in selected if float(v) in allow_set]
    return selected


def select_ntdpl(frame: pd.DataFrame, ntdpl_p_max: int) -> pd.DataFrame:
    ntdpl = frame.loc[frame["ovr.method"] == "ntdpl"].copy()
    if ntdpl.empty:
        return ntdpl
    p_max = maybe_numeric(ntdpl["ovr.method.p_max"]).to_numpy(dtype=float)
    requested = float(ntdpl_p_max)
    mask = np.isclose(p_max, requested, equal_nan=False)
    selected = ntdpl.loc[mask].copy()
    if not selected.empty:
        return selected

    available = np.unique(p_max[~np.isnan(p_max)])
    if available.size == 0:
        print("Warning: NTDPL runs have no numeric p_max; plotting without p_max filtering.")
        return ntdpl

    fallback = float(np.max(available))
    print(
        f"Warning: No NTDPL runs found with p_max={requested:g}; using p_max={fallback:g} instead."
    )
    return ntdpl.loc[np.isclose(p_max, fallback, equal_nan=False)].copy()


def infer_nonlinears(tbl: pd.DataFrame) -> list[str]:
    if "ovr.filter.nonlinear" not in tbl.columns:
        raise KeyError("Missing column: ovr.filter.nonlinear")
    nonlinears = [str(v) for v in tbl["ovr.filter.nonlinear"].dropna().unique().tolist()]
    usable: list[str] = []
    for nonlinear in nonlinears:
        panel = tbl.loc[tbl["ovr.filter.nonlinear"].astype(str) == nonlinear]
        if panel.empty:
            continue
        if "ovr.filter.alpha" not in panel.columns or "NMSE_dB" not in panel.columns:
            continue
        alpha = maybe_numeric(panel["ovr.filter.alpha"]).dropna()
        nmse = maybe_numeric(panel["NMSE_dB"]).dropna()
        if alpha.empty or nmse.empty:
            continue
        usable.append(nonlinear)
    return sorted(set(usable))
