from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from .common import load_results, select_method_runs
from ..utils.io import maybe_numeric


METHOD_ORDER: list[str] = ["tucker", "cp", "tt", "tr", "ntdpl"]
NTDPL_P_MAX = 8
P_MAX_ALPHA_REF = 0.25


@dataclass(frozen=True)
class NonlinearApproxGroup:
    name: str
    nonlinears: tuple[str, ...]
    table_artifact: str
    results_artifact: str


POLY_GROUP = NonlinearApproxGroup(
    name="poly",
    nonlinears=("poly2", "poly3"),
    table_artifact="poly_table.tex",
    results_artifact="poly_results.csv",
)

NONPOLY_GROUP = NonlinearApproxGroup(
    name="nonpoly",
    nonlinears=("tanh", "exp"),
    table_artifact="nonpoly_table.tex",
    results_artifact="nonpoly_results.csv",
)


def dedup_nonlinear_runs(frame: pd.DataFrame) -> pd.DataFrame:
    subset_cols = [
        col
        for col in ("ovr.method", "ovr.filter.nonlinear", "ovr.filter.alpha", "ovr.method.p_max", "ovr.data.seed")
        if col in frame.columns
    ]
    if not subset_cols:
        return frame
    sort_col = "run_dir" if "run_dir" in frame.columns else None
    panel = frame.sort_values(sort_col) if sort_col is not None else frame.copy()
    return panel.drop_duplicates(subset=subset_cols, keep="last").reset_index(drop=True)


def load_nonlinear_group(group: NonlinearApproxGroup) -> tuple[pd.DataFrame, object]:
    loaded = load_results("nonlinear-approx", require_curves=False)
    runs = loaded.runs.loc[loaded.runs["ovr.filter.nonlinear"].isin(group.nonlinears)].copy()
    runs = dedup_nonlinear_runs(runs)
    if runs.empty:
        raise RuntimeError(
            f"No {group.name} nonlinear-approx runs found. "
            "Run `python -m experiment nonlinear-approx run` first."
        )
    return runs, loaded.env


def select_ntdpl_for_nonlinear_approx(
    frame: pd.DataFrame,
    *,
    requested_p_max: int = NTDPL_P_MAX,
) -> pd.DataFrame:
    ntdpl = select_method_runs(frame, "ntdpl")
    if ntdpl.empty:
        return ntdpl

    selected = select_method_runs(ntdpl, "ntdpl", p_max=requested_p_max)
    if not selected.empty:
        return selected

    pmax_series = maybe_numeric(ntdpl["ovr.method.p_max"]).dropna()
    if pmax_series.empty:
        print("Warning: NTDPL runs have no numeric p_max; using all NTDPL rows.")
        return ntdpl

    fallback = int(float(pmax_series.max()))
    print(f"Warning: No NTDPL runs found with p_max={requested_p_max}; using p_max={fallback} instead.")
    return select_method_runs(ntdpl, "ntdpl", p_max=fallback)


def alpha_levels(frame: pd.DataFrame) -> list[float]:
    numeric = maybe_numeric(frame["ovr.filter.alpha"]).dropna().to_numpy(dtype=float)
    return sorted({float(value) for value in numeric})


def pmax_levels(frame: pd.DataFrame) -> list[int]:
    numeric = maybe_numeric(frame["ovr.method.p_max"]).dropna().to_numpy(dtype=float)
    return sorted({int(value) for value in numeric})
