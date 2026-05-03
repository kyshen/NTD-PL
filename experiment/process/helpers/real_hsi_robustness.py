from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...config import get_env
from ...utils.io import load_run_parquets, maybe_numeric
from src.data.hsi import _load_hsi_from_file


MAIN_PMAX = 4
MAIN_METHODS = ("tucker", "ntdpl")
MAIN_TASKS = ("decompose", "random-missing-completion")
COMPLETION_MISSING_RATE = 0.5
DATASET_ORDER = ("jasper_ridge_hsi", "samson_hsi", "urban_hsi", "cuprite_hsi")
DATASET_LABELS = {
    "jasper_ridge_hsi": "Jasper Ridge",
    "samson_hsi": "Samson",
    "urban_hsi": "Urban",
    "cuprite_hsi": "Cuprite",
}
DATASET_PATHS = {
    "jasper_ridge_hsi": Path("data/hsi/jasperRidge2_R198.mat"),
    "samson_hsi": Path("data/hsi-similar/samson_1.img"),
    "urban_hsi": Path("data/hsi-similar/Urban_R162.mat"),
    "cuprite_hsi": Path("data/hsi-similar/Cuprite_S1_R188.img"),
}
TASK_LABELS = {
    "decompose": "Recon.",
    "random-missing-completion": "Compl.",
}


def _jsonish(value: Any) -> Any:
    out = value
    while isinstance(out, str):
        text = out.strip()
        if not text:
            return text
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError:
            return out
        if loaded == out:
            return loaded
        out = loaded
    return out


def _parse_rank(value: Any) -> tuple[int, int, int]:
    parsed = _jsonish(value)
    if isinstance(parsed, str):
        parsed = parsed.strip("[]()")
        items = [part.strip() for part in parsed.split(",") if part.strip()]
        return tuple(int(item) for item in items)  # type: ignore[return-value]
    if isinstance(parsed, (list, tuple)):
        return tuple(int(item) for item in parsed)  # type: ignore[return-value]
    raise ValueError(f"Cannot parse rank from value: {value}")


def _task_metric_columns(task_name: str) -> tuple[str, str, str]:
    if task_name == "decompose":
        return ("RMSE", "NMSE_dB", "SAM")
    if task_name == "random-missing-completion":
        return ("RMSE_missing", "NMSE_dB_missing", "SAM_missing")
    raise ValueError(f"Unsupported task: {task_name}")


def _rank_text(rank: tuple[int, int, int]) -> str:
    return f"({rank[0]},{rank[1]},{rank[2]})"


def _format_pm(mean: float, std: float, digits: int) -> str:
    return f"{mean:.{digits}f} +- {std:.{digits}f}"


def _latex_pm(mean: float, std: float, digits: int) -> str:
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def _dataset_metadata(project_root: Path) -> pd.DataFrame:
    rows: list[dict[str, str | int]] = []
    for dataset_name in DATASET_ORDER:
        cube = np.asarray(_load_hsi_from_file(project_root / DATASET_PATHS[dataset_name]), dtype=np.float32)
        rows.append(
            {
                "dataset": dataset_name,
                "dataset_label": DATASET_LABELS[dataset_name],
                "source_shape": f"{cube.shape[0]}x{cube.shape[1]}x{cube.shape[2]}",
                "bands_used": int(cube.shape[2]),
                "crop_shape": "full scene",
                "normalization": "global max",
                "rank_rule": "auto CR-matched",
                "completion_rate": "rho=0.5",
            }
        )
    return pd.DataFrame(rows)


def load_main_runs() -> tuple[pd.DataFrame, pd.DataFrame, object]:
    env = get_env("real-hsi-robustness")
    runs = load_run_parquets(env.results_dir)["runs"].copy()
    if runs.empty:
        raise RuntimeError(
            "No runs found for real-hsi-robustness. Run `python -m experiment real-hsi-robustness run` first."
        )

    frame = runs.copy()
    frame["dataset"] = frame["ovr.data"].astype(str)
    frame["method_name"] = frame["ovr.method"].astype(str)
    frame["task_name"] = frame["ovr.task"].astype(str)
    frame["rank"] = frame["ovr.method.rank"].map(_parse_rank)
    frame["fit_time_sec"] = maybe_numeric(frame["fit_time_sec"]).astype(float)
    if "ovr.method.p_max" in frame.columns:
        frame["p_max"] = maybe_numeric(frame["ovr.method.p_max"])
    else:
        frame["p_max"] = np.nan
    if "ovr.task.seed" in frame.columns:
        frame["mask_seed"] = maybe_numeric(frame["ovr.task.seed"])
    else:
        frame["mask_seed"] = np.nan
    if "ovr.task.missing_rate" in frame.columns:
        frame["missing_rate"] = maybe_numeric(frame["ovr.task.missing_rate"])
    else:
        frame["missing_rate"] = np.nan

    for col in ("RMSE", "NMSE_dB", "SAM", "RMSE_missing", "NMSE_dB_missing", "SAM_missing"):
        if col in frame.columns:
            frame[col] = maybe_numeric(frame[col]).astype(float)

    frame = frame.loc[frame["dataset"].isin(DATASET_ORDER)].copy()
    frame = frame.loc[frame["task_name"].isin(MAIN_TASKS)].copy()
    frame = frame.loc[frame["method_name"].isin(MAIN_METHODS)].copy()
    frame = frame.loc[
        ~frame["method_name"].eq("ntdpl") | np.isclose(frame["p_max"], float(MAIN_PMAX), atol=1e-12)
    ].copy()
    completion_mask = frame["task_name"].eq("random-missing-completion")
    frame = frame.loc[
        ~completion_mask | np.isclose(frame["missing_rate"], COMPLETION_MISSING_RATE, atol=1e-12)
    ].copy()

    dedup_keys: list[str] = ["dataset", "task_name", "method_name"]
    if completion_mask.any():
        dedup_keys.extend(["mask_seed", "missing_rate"])
    frame = frame.sort_values("run_dir").drop_duplicates(subset=dedup_keys, keep="last").reset_index(drop=True)
    metadata = _dataset_metadata(env.project_root)
    rank_summary = (
        frame.groupby("dataset", as_index=False)
        .agg(rank_rule=("rank", lambda values: _rank_text(tuple(int(v) for v in values.iloc[0]))))
    )
    metadata = metadata.drop(columns=["rank_rule"]).merge(rank_summary, on="dataset", how="left")
    return frame, metadata, env


def build_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset_name in DATASET_ORDER:
        for task_name in MAIN_TASKS:
            panel = frame.loc[
                frame["dataset"].eq(dataset_name)
                & frame["task_name"].eq(task_name)
            ].copy()
            if panel.empty:
                continue
            rmse_col, nmse_col, sam_col = _task_metric_columns(task_name)
            for method_name in MAIN_METHODS:
                method_panel = panel.loc[panel["method_name"].eq(method_name)].copy()
                if method_panel.empty:
                    continue
                rows.append(
                    {
                        "dataset": dataset_name,
                        "dataset_label": DATASET_LABELS[dataset_name],
                        "task_name": task_name,
                        "task_label": TASK_LABELS[task_name],
                        "method_name": method_name,
                        "method_label": "NTD-PL" if method_name == "ntdpl" else "Tucker",
                        "rank": _rank_text(tuple(int(v) for v in method_panel["rank"].iloc[0])),
                        "metric_label": "RMSE*" if task_name == "random-missing-completion" else "RMSE",
                        "rmse_mean": float(method_panel[rmse_col].mean()),
                        "rmse_std": float(method_panel[rmse_col].std(ddof=0)) if len(method_panel) > 1 else 0.0,
                        "nmse_mean": float(method_panel[nmse_col].mean()),
                        "nmse_std": float(method_panel[nmse_col].std(ddof=0)) if len(method_panel) > 1 else 0.0,
                        "sam_mean": float(method_panel[sam_col].mean()),
                        "sam_std": float(method_panel[sam_col].std(ddof=0)) if len(method_panel) > 1 else 0.0,
                        "fit_time_mean": float(method_panel["fit_time_sec"].mean()),
                        "fit_time_std": float(method_panel["fit_time_sec"].std(ddof=0))
                        if len(method_panel) > 1
                        else 0.0,
                        "n_runs": int(len(method_panel)),
                    }
                )
    summary = pd.DataFrame(rows)
    if summary.empty:
        raise RuntimeError("No rows available for the real HSI robustness summary.")
    return summary


def build_main_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset_name in DATASET_ORDER:
        for task_name in MAIN_TASKS:
            panel = summary.loc[
                summary["dataset"].eq(dataset_name)
                & summary["task_name"].eq(task_name)
            ].copy()
            if panel["method_name"].nunique() < 2:
                continue
            tucker = panel.loc[panel["method_name"].eq("tucker")].iloc[0]
            ntdpl = panel.loc[panel["method_name"].eq("ntdpl")].iloc[0]
            rmse_gain_pct = 100.0 * (float(tucker["rmse_mean"]) - float(ntdpl["rmse_mean"])) / float(tucker["rmse_mean"])
            nmse_gain = float(tucker["nmse_mean"]) - float(ntdpl["nmse_mean"])
            sam_gain = float(tucker["sam_mean"]) - float(ntdpl["sam_mean"])
            rows.append(
                {
                    "dataset": dataset_name,
                    "dataset_label": DATASET_LABELS[dataset_name],
                    "task_name": task_name,
                    "task_label": TASK_LABELS[task_name],
                    "rank": str(tucker["rank"]),
                    "metric_label": str(tucker["metric_label"]),
                    "tucker_rmse": float(tucker["rmse_mean"]),
                    "ntdpl_rmse": float(ntdpl["rmse_mean"]),
                    "gain_pct": float(rmse_gain_pct),
                    "delta_nmse_db": float(nmse_gain),
                    "delta_sam": float(sam_gain),
                    "all_metrics_positive": int(rmse_gain_pct > 0.0) + int(nmse_gain > 0.0) + int(sam_gain > 0.0),
                }
            )
    main = pd.DataFrame(rows)
    if main.empty:
        raise RuntimeError("No rows available for the real HSI robustness main table.")
    return main


def build_main_table_display(main_table: pd.DataFrame) -> pd.DataFrame:
    panel = main_table.copy()
    panel["Tucker"] = panel["tucker_rmse"].map(lambda v: f"{float(v):.5f}")
    panel["NTD-PL"] = panel["ntdpl_rmse"].map(lambda v: f"{float(v):.5f}")
    panel["Gain(%)"] = panel["gain_pct"].map(lambda v: f"{float(v):.2f}")
    panel["Delta_NMSE(dB)"] = panel["delta_nmse_db"].map(lambda v: f"{float(v):.3f}")
    panel["Delta_SAM"] = panel["delta_sam"].map(lambda v: f"{float(v):.3f}")
    return panel.loc[
        :,
        [
            "dataset",
            "dataset_label",
            "task_name",
            "task_label",
            "rank",
            "metric_label",
            "Tucker",
            "NTD-PL",
            "Gain(%)",
            "Delta_NMSE(dB)",
            "Delta_SAM",
            "all_metrics_positive",
        ],
    ].reset_index(drop=True)


def build_appendix_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in summary.to_dict("records"):
        rows.append(
            {
                "dataset": row["dataset_label"],
                "task": row["task_label"],
                "method": row["method_label"],
                "rank": row["rank"],
                "RMSE": _format_pm(float(row["rmse_mean"]), float(row["rmse_std"]), 5),
                "NMSE(dB)": _format_pm(float(row["nmse_mean"]), float(row["nmse_std"]), 3),
                "SAM": _format_pm(float(row["sam_mean"]), float(row["sam_std"]), 3),
                "Time(s)": _format_pm(float(row["fit_time_mean"]), float(row["fit_time_std"]), 2),
                "n_runs": int(row["n_runs"]),
            }
        )
    return pd.DataFrame(rows)


def build_consistency_summary(main_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task_name in MAIN_TASKS:
        panel = main_table.loc[main_table["task_name"].eq(task_name)].copy()
        if panel.empty:
            continue
        rows.append(
            {
                "task": TASK_LABELS[task_name],
                "datasets": int(len(panel)),
                "rmse_gain_positive": int(np.sum(panel["gain_pct"] > 0.0)),
                "nmse_gain_positive": int(np.sum(panel["delta_nmse_db"] > 0.0)),
                "sam_gain_positive": int(np.sum(panel["delta_sam"] > 0.0)),
                "all_three_positive": int(np.sum(panel["all_metrics_positive"].astype(int) == 3)),
            }
        )
    return pd.DataFrame(rows)


def build_overview_figure_data(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    x_map = {name: idx + 1 for idx, name in enumerate(DATASET_ORDER)}
    for task_name, panel_key, panel_title, y_label in (
        ("decompose", "reconstruction", "Reconstruction", "RMSE"),
        ("random-missing-completion", "completion", "Completion", "RMSE*"),
    ):
        panel = summary.loc[summary["task_name"].eq(task_name)].copy()
        for method_name in MAIN_METHODS:
            method_panel = panel.loc[panel["method_name"].eq(method_name)].copy()
            method_panel = method_panel.sort_values("dataset")
            for row in method_panel.to_dict("records"):
                x = float(x_map[str(row["dataset"])])
                mean = float(row["rmse_mean"])
                std = float(row["rmse_std"])
                rows.append(
                    {
                        "panel": panel_key,
                        "panel_title": panel_title,
                        "method": "NTD-PL" if method_name == "ntdpl" else "Tucker",
                        "x": x,
                        "mean": mean,
                        "std": std,
                        "band_lower": mean - std,
                        "band_upper": mean + std,
                        "annotation": "Lower is better.",
                        "y_label": y_label,
                    }
                )
    return pd.DataFrame(rows).sort_values(["panel", "method", "x"]).reset_index(drop=True)


def main_table_latex(main_table: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{@{}l cccc cccc@{}}",
        r"    \toprule",
        r"    \multirow{2}{*}{Dataset} & \multicolumn{4}{c}{Reconstruction} & \multicolumn{4}{c}{Completion} \\",
        r"    \cmidrule(lr){2-5}\cmidrule(lr){6-9}",
        r"    & Tucker & NTD-PL & Gain & $\Delta$SAM & Tucker & NTD-PL & Gain & $\Delta$SAM \\",
        r"    \midrule",
    ]
    for dataset_name in DATASET_ORDER:
        panel = main_table.loc[main_table["dataset"].eq(dataset_name)].copy()
        if panel.empty:
            continue
        recon = panel.loc[panel["task_name"].eq("reconstruction")].iloc[0]
        compl = panel.loc[panel["task_name"].eq("completion")].iloc[0]

        def _fmt_pair(tucker: float, ntdpl: float) -> tuple[str, str]:
            tucker_text = f"{tucker:.4f}"
            ntdpl_text = f"{ntdpl:.4f}"
            if tucker <= ntdpl:
                tucker_text = rf"\textbf{{{tucker_text}}}"
            else:
                ntdpl_text = rf"\textbf{{{ntdpl_text}}}"
            return tucker_text, ntdpl_text

        def _fmt_positive(value: float, precision: int) -> str:
            text = f"{value:.{precision}f}"
            if value > 0.0:
                return rf"\textbf{{{text}}}"
            return text

        def _fmt_gain(value: float) -> str:
            text = f"{value:.2f}\\%"
            if value > 0.0:
                return rf"\textbf{{{text}}}"
            return text

        recon_tucker, recon_ntdpl = _fmt_pair(float(recon["tucker_rmse"]), float(recon["ntdpl_rmse"]))
        compl_tucker, compl_ntdpl = _fmt_pair(float(compl["tucker_rmse"]), float(compl["ntdpl_rmse"]))
        lines.append(
            "    "
            + " & ".join(
                [
                    str(recon["dataset_label"]),
                    recon_tucker,
                    recon_ntdpl,
                    _fmt_gain(float(recon["gain_pct"])),
                    _fmt_positive(float(recon["delta_sam"]), 2),
                    compl_tucker,
                    compl_ntdpl,
                    _fmt_gain(float(compl["gain_pct"])),
                    _fmt_positive(float(compl["delta_sam"]), 2),
                ]
            )
            + r" \\"
        )
        lines.append(r"    \midrule")
    lines[-1] = r"    \bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def appendix_table_latex(summary: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{l l l l c c c c c}",
        r"    \toprule",
        r"    Dataset & Task & Method & Rank & RMSE & NMSE(dB) & SAM & Time(s) & $n$ \\",
        r"    \midrule",
    ]
    for dataset_name in DATASET_ORDER:
        dataset_panel = summary.loc[summary["dataset"].eq(dataset_name)].copy()
        if dataset_panel.empty:
            continue
        for task_name in MAIN_TASKS:
            task_panel = dataset_panel.loc[dataset_panel["task_name"].eq(task_name)].copy()
            if task_panel.empty:
                continue
            for idx, (_, row) in enumerate(task_panel.iterrows()):
                dataset_cell = DATASET_LABELS[dataset_name] if task_name == MAIN_TASKS[0] and idx == 0 else ""
                task_cell = TASK_LABELS[task_name] if idx == 0 else ""
                lines.append(
                    "    "
                    + " & ".join(
                        [
                            dataset_cell,
                            task_cell,
                            str(row["method_label"]),
                            str(row["rank"]),
                            _latex_pm(float(row["rmse_mean"]), float(row["rmse_std"]), 5),
                            _latex_pm(float(row["nmse_mean"]), float(row["nmse_std"]), 3),
                            _latex_pm(float(row["sam_mean"]), float(row["sam_std"]), 3),
                            _latex_pm(float(row["fit_time_mean"]), float(row["fit_time_std"]), 2),
                            str(int(row["n_runs"])),
                        ]
                    )
                    + r" \\"
                )
            lines.append(r"    \midrule")
    lines[-1] = r"    \bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def protocol_table_latex(metadata: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{l c c c c c c}",
        r"    \toprule",
        r"    Dataset & Source shape & Bands & Crop & Normalization & Rank & Completion \\",
        r"    \midrule",
    ]
    for _, row in metadata.iterrows():
        lines.append(
            "    "
            + " & ".join(
                [
                    str(row["dataset_label"]),
                    str(row["source_shape"]),
                    str(int(row["bands_used"])),
                    str(row["crop_shape"]),
                    str(row["normalization"]),
                    str(row["rank_rule"]),
                    str(row["completion_rate"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"    \bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"
