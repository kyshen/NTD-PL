from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from pathlib import Path

from ...utils.io import maybe_numeric
from ...utils.paper import write_csv_artifact, write_text_artifact
from ..common import (
    LoadedResults,
    bool_mask,
    float_mask,
    load_results,
    parse_setting_sweep,
    resolve_setting_series,
    setting_value_mask,
    series_with_fallback,
    varying_setting_keys,
)


MethodSelector = Callable[[pd.DataFrame, str], pd.DataFrame]
MethodLabeler = Callable[[str], str]


@dataclass(frozen=True)
class RowSpec:
    key: str
    label: str
    subset: pd.DataFrame


def metric_text(series: pd.Series, digits: int = 4, *, pm: str = " +- ") -> str:
    values = maybe_numeric(series).dropna().to_numpy(dtype=float)
    if values.size == 0:
        return "---"
    mean = values.mean()
    if values.size == 1:
        return f"{mean:.{digits}f}"
    std = values.std(ddof=0)
    return f"{mean:.{digits}f}{pm}{std:.{digits}f}"


def method_summary_for_setting(
    frame: pd.DataFrame,
    *,
    setting: str,
    metrics: list[str],
    method_order: list[str],
    env: object | None = None,
    method_col: str = "ovr.method",
    method_selector: MethodSelector | None = None,
    method_labeler: MethodLabeler | None = None,
    digits_by_metric: dict[str, int] | None = None,
) -> pd.DataFrame:
    key, values = parse_setting_sweep(setting)
    digits_by_metric = digits_by_metric or {}
    rows: list[dict[str, str]] = []

    if method_selector is None:
        def method_selector(local_frame: pd.DataFrame, method: str) -> pd.DataFrame:
            return local_frame.loc[local_frame[method_col] == method].copy()

    if method_labeler is None:
        if env is not None and hasattr(env, "label_for_method"):
            method_labeler = env.label_for_method
        else:
            method_labeler = str

    for value in values:
        panel = frame.loc[setting_value_mask(frame, key, value)].copy()
        if panel.empty:
            continue

        for method in method_order:
            subset = method_selector(panel, method)
            if subset.empty:
                continue

            row: dict[str, str] = {
                "Setting": f"{key}={value}",
                "Method": method_labeler(method),
                "n_runs": str(len(subset)),
            }
            for metric in metrics:
                if metric not in subset.columns:
                    raise KeyError(f"Metric column not found in runs table: {metric}")
                row[metric] = metric_text(subset[metric], digits=digits_by_metric.get(metric, 4))
            rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise RuntimeError(f"No rows available for summary with setting sweep: {setting}")

    ordered_cols = ["Setting", "Method", *metrics, "n_runs"]
    return summary.loc[:, [col for col in ordered_cols if col in summary.columns]]


def latex_for_method_summary(
    summary: pd.DataFrame,
    *,
    metrics: list[str],
    first_col: str = "Setting",
    method_col: str = "Method",
) -> str:
    def latex_label(text: str) -> str:
        if text == "NMSE_dB":
            return "NMSE(dB)"
        return text.replace("_", r"\_")

    tex_lines = [
        rf"\begin{{tabular}}{{c|c|{'c' * len(metrics)}}}",
        r"    \hline",
        f"    {latex_label(first_col)} & {latex_label(method_col)} & "
        + " & ".join(latex_label(metric) for metric in metrics)
        + r" \\",
        r"    \hline",
    ]
    for setting_value in summary[first_col].drop_duplicates():
        panel_rows = summary.loc[summary[first_col] == setting_value].to_dict("records")
        for idx, row in enumerate(panel_rows):
            setting_text = row[first_col] if idx == 0 else ""
            metric_cells = [str(row[metric]).replace("+-", r"$\pm$") for metric in metrics]
            tex_lines.append(f"    {setting_text} & {row[method_col]} & " + " & ".join(metric_cells) + r" \\")
        tex_lines.append(r"    \hline")
    tex_lines.append(r"\end{tabular}")
    return "\n".join(tex_lines) + "\n"


def approx_table(
    *,
    frame: pd.DataFrame,
    env: object,
    setting: str,
    stem: str,
    method_order: list[str],
    metrics: list[str],
    method_selector: MethodSelector | None = None,
    digits_by_metric: dict[str, int] | None = None,
    fixed: dict[str, str] | None = None,
    csv_artifact_name: str | None = None,
    latex_csv_name: str | None = None,
    tex_artifact_name: str | None = None,
    latex_tex_name: str | None = None,
) -> tuple[pd.DataFrame, Path, Path, Path, Path]:
    summary = method_summary_for_setting(
        frame,
        setting=setting,
        metrics=metrics,
        method_order=method_order,
        env=env,
        method_selector=method_selector,
        digits_by_metric=digits_by_metric,
    )
    fixed = fixed or {}
    if fixed:
        for fixed_key, fixed_value in fixed.items():
            summary = summary.copy()
            summary.insert(0, fixed_key, fixed_value)

    csv_path, csv_latex_path = write_csv_artifact(
        env,
        summary,
        artifact_name=csv_artifact_name or f"{stem}.csv",
        latex_name=latex_csv_name,
    )
    tex_path, latex_path = write_text_artifact(
        env,
        latex_for_method_summary(summary, metrics=metrics),
        artifact_name=tex_artifact_name or f"table_{Path(stem).name.removeprefix('summary_')}.tex",
        latex_name=latex_tex_name,
    )
    return summary, csv_path, csv_latex_path, tex_path, latex_path


def load_method_summary_for_setting(
    exp_name: str,
    *,
    setting: str,
    metrics: list[str],
    method_order: list[str],
    require_curves: bool = False,
    run_filter: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    method_selector: MethodSelector | None = None,
    digits_by_metric: dict[str, int] | None = None,
    allowed_varying_keys: set[str] | None = None,
) -> tuple[pd.DataFrame, object, pd.DataFrame]:
    loaded = load_results(exp_name, require_curves=require_curves)
    frame = loaded.runs if run_filter is None else run_filter(loaded.runs.copy())
    if frame.empty:
        raise RuntimeError(f"No runs available for summary in {exp_name}.")

    key, _ = parse_setting_sweep(setting)
    allowed = {
        "ovr.method",
        "ovr.method.p_max",
        "ovr.data.seed",
        "run_id",
        "run_dir",
        "state_path",
        key,
        f"ovr.{key}" if not key.startswith("ovr.") else key.removeprefix("ovr."),
    }
    if allowed_varying_keys:
        allowed |= allowed_varying_keys

    extra_varying = varying_setting_keys(frame, exclude_keys=allowed)
    if extra_varying:
        joined = ", ".join(extra_varying)
        raise RuntimeError(
            f"Summary for setting '{setting}' in '{exp_name}' is ambiguous because other settings also vary: {joined}"
        )

    summary = method_summary_for_setting(
        frame,
        setting=setting,
        metrics=metrics,
        method_order=method_order,
        env=loaded.env,
        method_selector=method_selector,
        digits_by_metric=digits_by_metric,
    )
    return summary, loaded.env, frame


def nonlinear_method_grid_table(
    frame: pd.DataFrame,
    *,
    env: object,
    nonlinears: list[str],
    method_order: list[str],
    value_col: str = "NMSE_dB",
    alpha_col: str = "ovr.filter.alpha",
    nonlinear_col: str = "ovr.filter.nonlinear",
    method_col: str = "ovr.method",
    ntdpl_method: str = "ntdpl",
    ntdpl_selector: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    digits: int = 4,
) -> tuple[pd.DataFrame, list[float]]:
    alpha_series = maybe_numeric(frame[alpha_col])
    alpha_levels = np.sort(alpha_series.dropna().unique())
    alpha_values = [float(value) for value in alpha_levels]
    rows: list[dict[str, str]] = []

    for method in method_order:
        panel = frame.loc[frame[method_col] == method].copy()
        if method == ntdpl_method and ntdpl_selector is not None:
            panel = ntdpl_selector(panel)

        row: dict[str, str] = {"Method": env.label_for_method(method) if hasattr(env, "label_for_method") else method}
        for nonlinear in nonlinears:
            for alpha in alpha_values:
                mask = (
                    (panel[nonlinear_col].astype(str) == nonlinear)
                    & np.isclose(alpha_series.loc[panel.index].to_numpy(dtype=float), alpha, equal_nan=False)
                )
                values = maybe_numeric(panel.loc[mask, value_col]).dropna().to_numpy(dtype=float)
                row[f"{nonlinear}@{alpha:g}"] = f"{values.mean():.{digits}f}" if values.size else "---"
        rows.append(row)

    return pd.DataFrame(rows), alpha_values


def latex_for_nonlinear_method_grid(
    summary: pd.DataFrame,
    *,
    env: object,
    nonlinears: list[str],
    alpha_levels: list[float],
    method_order: list[str],
) -> str:
    best_map: dict[str, float] = {}
    for nonlinear in nonlinears:
        for alpha in alpha_levels:
            column = f"{nonlinear}@{alpha:g}"
            numeric = pd.to_numeric(summary[column], errors="coerce")
            if numeric.notna().any():
                best_map[column] = float(numeric.min())

    def fmt_cell(column: str, value: str) -> str:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.notna(numeric) and column in best_map and np.isclose(float(numeric), best_map[column]):
            return rf"\textbf{{{value}}}"
        return value

    n_alpha = len(alpha_levels)
    tex_lines = [
        rf"\begin{{tabular}}{{c|{'c' * n_alpha}|{'c' * n_alpha}}}",
        r"    \hline",
        rf"    Method & \multicolumn{{{n_alpha}}}{{c|}}{{{nonlinears[0]}}} & \multicolumn{{{n_alpha}}}{{c}}{{{nonlinears[1]}}} \\",
        r"    \hline",
        "     & " + " & ".join([f"{alpha:g}" for alpha in alpha_levels] * len(nonlinears)) + r" \\",
        r"    \hline",
    ]

    for method in method_order[: len(method_order) - 1]:
        row = summary.loc[summary["Method"] == env.label_for_method(method)].iloc[0]
        values = [
            fmt_cell(f"{nonlinear}@{alpha:g}", row[f"{nonlinear}@{alpha:g}"])
            for nonlinear in nonlinears
            for alpha in alpha_levels
        ]
        tex_lines.append(f"    {row['Method']} & " + " & ".join(values) + r" \\")
    tex_lines.append(r"    \hline")
    final_method = method_order[-1]
    row = summary.loc[summary["Method"] == env.label_for_method(final_method)].iloc[0]
    values = [
        fmt_cell(f"{nonlinear}@{alpha:g}", row[f"{nonlinear}@{alpha:g}"])
        for nonlinear in nonlinears
        for alpha in alpha_levels
    ]
    tex_lines.append(f"    {row['Method']} & " + " & ".join(values) + r" \\")
    tex_lines.extend([r"    \hline", r"\end{tabular}"])
    return "\n".join(tex_lines) + "\n"


def single_metric_rows_by_nonlinear(
    *,
    row_specs: list[RowSpec],
    nonlinears: list[str],
    value_col: str,
    digits: int,
    first_col: str,
    pm: str = " $\\pm$ ",
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict[str, str]] = []
    best_map: dict[str, float] = {}

    for spec in row_specs:
        row: dict[str, str] = {first_col: spec.label}
        for nonlinear in nonlinears:
            raw = metric_text(
                spec.subset.loc[spec.subset["ovr.filter.nonlinear"] == nonlinear, value_col],
                digits=digits,
                pm=pm,
            )
            row[nonlinear] = raw
            if raw != "---":
                mean = float(raw.split("$\\pm$", 1)[0].strip()) if "$\\pm$" in raw else float(raw)
                best_map[nonlinear] = min(best_map.get(nonlinear, mean), mean)
        rows.append(row)

    return pd.DataFrame(rows), best_map


def latex_for_single_metric_rows(
    rows: list[dict[str, str]],
    *,
    nonlinears: list[str],
    first_col: str,
    header_label: str,
    best_map: dict[str, float],
) -> str:
    def fmt_cell(nonlinear: str, raw: str) -> str:
        if raw == "---":
            return raw
        mean = float(raw.split("$\\pm$", 1)[0].strip()) if "$\\pm$" in raw else float(raw)
        return rf"\textbf{{{raw}}}" if np.isclose(mean, best_map[nonlinear]) else raw

    tex_lines = [
        rf"\begin{{tabular}}{{c|{'c' * len(nonlinears)}}}",
        r"    \hline",
        f"    {header_label} & " + " & ".join(nonlinears) + r" \\",
        r"    \hline",
    ]
    for row in rows:
        values = [fmt_cell(nonlinear, row[nonlinear]) for nonlinear in nonlinears]
        tex_lines.append(f"    {row[first_col]} & " + " & ".join(values) + r" \\")
    tex_lines.extend([r"    \hline", r"\end{tabular}"])
    return "\n".join(tex_lines) + "\n"
