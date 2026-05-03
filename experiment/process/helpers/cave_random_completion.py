from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest, rankdata, wilcoxon

from ...config import get_env
from ...hsi_defaults import CAVE_RECON_MAIN_RANK
from ...utils.io import load_run_parquets, load_state_mat, maybe_numeric
from src.data.hsi import CAVEHSIData
from src.filters.bias import BiasFilter


MAIN_RANK = CAVE_RECON_MAIN_RANK
MAIN_PMAX = 4
MAIN_METHODS = ("tucker", "ntdpl")
MAIN_MISSING_RATES = (0.1, 0.3, 0.5, 0.7)


@dataclass(frozen=True)
class ScenePayload:
    scene_id: int
    scene_name: str
    original: np.ndarray
    observed_mask: np.ndarray
    recon_tucker: np.ndarray
    recon_ntdpl: np.ndarray


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


def _resolve_state_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return get_env("cave-random-completion").project_root / path


def _frame_value(row: Any, *keys: str) -> Any:
    for key in keys:
        if hasattr(row, "get"):
            value = row.get(key, None)
        else:
            value = None
        if value is None:
            continue
        if isinstance(value, float) and np.isnan(value):
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _parse_shape2(value: Any, *, default: tuple[int, int]) -> tuple[int, int]:
    parsed = _jsonish(value)
    if parsed is None:
        return default
    if isinstance(parsed, str):
        text = parsed.strip()
        if not text or text.lower() == "null":
            return default
        parsed = [part.strip() for part in text.strip("[]()").split(",") if part.strip()]
    if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
        return int(parsed[0]), int(parsed[1])
    return default


def _parse_optional_shape2(value: Any) -> tuple[int, int] | None:
    parsed = _jsonish(value)
    if parsed is None:
        return None
    if isinstance(parsed, str):
        text = parsed.strip()
        if not text or text.lower() == "null":
            return None
        parsed = [part.strip() for part in text.strip("[]()").split(",") if part.strip()]
    if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
        return int(parsed[0]), int(parsed[1])
    return None


def _cave_dataset_kwargs_from_row(row: Any) -> dict[str, Any]:
    path = _frame_value(row, "data.path", "ovr.data.path") or "data/CAVE"
    target_shape = _parse_shape2(
        _frame_value(row, "data.target_shape", "ovr.data.target_shape"),
        default=(512, 512),
    )
    crop_shape = _parse_optional_shape2(_frame_value(row, "data.crop_shape", "ovr.data.crop_shape"))
    return {
        "path": str(path),
        "target_shape": target_shape,
        "crop_shape": crop_shape,
    }


def load_main_runs() -> tuple[pd.DataFrame, object]:
    env = get_env("cave-random-completion")
    runs = load_run_parquets(env.results_dir)["runs"].copy()
    if runs.empty:
        raise RuntimeError(
            "No runs found for cave-random-completion. Run `python -m experiment cave-random-completion run` first."
        )

    frame = runs.copy()
    frame = frame.loc[frame["ovr.data"].astype(str) == "cave_hsi"].copy()
    frame["method_name"] = frame["ovr.method"].astype(str)
    frame["rank"] = frame["ovr.method.rank"].map(_parse_rank)
    frame["scene_id"] = maybe_numeric(frame["ovr.data.id"]).astype(int)
    frame["mask_seed"] = maybe_numeric(frame["ovr.task.seed"]).astype(int)
    frame["missing_rate"] = maybe_numeric(frame["ovr.task.missing_rate"]).astype(float)
    frame["RMSE_all"] = maybe_numeric(frame["RMSE_all"]).astype(float)
    frame["RMSE_missing"] = maybe_numeric(frame["RMSE_missing"]).astype(float)
    frame["SAM_all"] = maybe_numeric(frame["SAM_all"]).astype(float)
    frame["SAM_missing"] = maybe_numeric(frame["SAM_missing"]).astype(float)
    frame["NMSE_dB_all"] = maybe_numeric(frame["NMSE_dB_all"]).astype(float)
    frame["fit_time_sec"] = maybe_numeric(frame["fit_time_sec"]).astype(float)
    if "NMSE_dB_missing" in frame.columns:
        frame["NMSE_dB_missing"] = maybe_numeric(frame["NMSE_dB_missing"]).astype(float)
    else:
        frame["NMSE_dB_missing"] = np.nan
    if "ovr.method.p_max" in frame.columns:
        frame["p_max"] = maybe_numeric(frame["ovr.method.p_max"])
    else:
        frame["p_max"] = np.nan

    frame = frame.loc[frame["rank"] == MAIN_RANK].copy()
    frame = frame.loc[frame["method_name"].isin(MAIN_METHODS)].copy()
    frame = frame.loc[
        ~frame["method_name"].eq("ntdpl") | np.isclose(frame["p_max"], float(MAIN_PMAX), atol=1e-12)
    ].copy()
    frame = frame.loc[frame["missing_rate"].isin(MAIN_MISSING_RATES)].copy()
    frame = frame.sort_values("run_dir").drop_duplicates(
        subset=["scene_id", "mask_seed", "missing_rate", "method_name"],
        keep="last",
    )
    return frame, env


def build_scene_mean_table(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "RMSE_all",
        "RMSE_missing",
        "SAM_all",
        "SAM_missing",
        "NMSE_dB_all",
        "NMSE_dB_missing",
        "fit_time_sec",
    ]
    grouped = (
        frame.groupby(["missing_rate", "method_name", "scene_id"], as_index=False)[metrics]
        .mean()
        .sort_values(["missing_rate", "method_name", "scene_id"])
    )
    return grouped


def build_main_summary(scene_mean: pd.DataFrame) -> pd.DataFrame:
    summary = (
        scene_mean.groupby(["missing_rate", "method_name"], as_index=False)
        .agg(
            RMSE_all_mean=("RMSE_all", "mean"),
            RMSE_all_std=("RMSE_all", "std"),
            RMSE_missing_mean=("RMSE_missing", "mean"),
            RMSE_missing_std=("RMSE_missing", "std"),
            SAM_all_mean=("SAM_all", "mean"),
            SAM_all_std=("SAM_all", "std"),
            SAM_missing_mean=("SAM_missing", "mean"),
            SAM_missing_std=("SAM_missing", "std"),
            NMSE_dB_all_mean=("NMSE_dB_all", "mean"),
            NMSE_dB_all_std=("NMSE_dB_all", "std"),
            NMSE_dB_missing_mean=("NMSE_dB_missing", "mean"),
            NMSE_dB_missing_std=("NMSE_dB_missing", "std"),
            Time_mean=("fit_time_sec", "mean"),
            Time_std=("fit_time_sec", "std"),
            n_scenes=("scene_id", "nunique"),
        )
        .sort_values(["missing_rate", "method_name"])
        .reset_index(drop=True)
    )
    method_order = {"tucker": 0, "ntdpl": 1}
    summary["method_order"] = summary["method_name"].map(method_order).fillna(99)
    summary = summary.sort_values(["missing_rate", "method_order"]).drop(columns=["method_order"]).reset_index(drop=True)
    for col in summary.columns:
        if col.endswith("_std"):
            summary[col] = summary[col].fillna(0.0)
    return summary


def build_paper_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str | float]] = []
    for row in summary.to_dict("records"):
        rows.append(
            {
                "missing_rate": float(row["missing_rate"]),
                "method": "NTD-PL" if row["method_name"] == "ntdpl" else "Tucker",
                "RMSE*": pm_text(float(row["RMSE_missing_mean"]), float(row["RMSE_missing_std"]), 5),
                "NMSE(dB)*": pm_text(float(row["NMSE_dB_missing_mean"]), float(row["NMSE_dB_missing_std"]), 3),
                "SAM*": pm_text(float(row["SAM_missing_mean"]), float(row["SAM_missing_std"]), 4),
                "n_scenes": int(row["n_scenes"]),
            }
        )
    return pd.DataFrame(rows)


def build_full_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str | float]] = []
    for row in summary.to_dict("records"):
        rows.append(
            {
                "missing_rate": float(row["missing_rate"]),
                "method": "NTD-PL" if row["method_name"] == "ntdpl" else "Tucker",
                "RMSE(all)": pm_text(float(row["RMSE_all_mean"]), float(row["RMSE_all_std"]), 5),
                "RMSE*": pm_text(float(row["RMSE_missing_mean"]), float(row["RMSE_missing_std"]), 5),
                "SAM(all)": pm_text(float(row["SAM_all_mean"]), float(row["SAM_all_std"]), 4),
                "SAM*": pm_text(float(row["SAM_missing_mean"]), float(row["SAM_missing_std"]), 4),
                "NMSE(dB)(all)": pm_text(float(row["NMSE_dB_all_mean"]), float(row["NMSE_dB_all_std"]), 3),
                "NMSE(dB)*": pm_text(float(row["NMSE_dB_missing_mean"]), float(row["NMSE_dB_missing_std"]), 3),
                "n_scenes": int(row["n_scenes"]),
            }
        )
    return pd.DataFrame(rows)


def pm_text(mean: float, std: float, digits: int) -> str:
    return f"{mean:.{digits}f} +- {std:.{digits}f}"


def latex_table(summary: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{@{}c l c c c@{}}",
        r"    \toprule",
        r"    $\rho$ & Method & RMSE* & NMSE*(dB) & SAM* \\",
        r"    \midrule",
    ]
    for missing_rate in MAIN_MISSING_RATES:
        panel = summary.loc[np.isclose(summary["missing_rate"], missing_rate, atol=1e-12)].copy()
        best_rmse = float(panel["RMSE_missing_mean"].min())
        best_nmse = float(panel["NMSE_dB_missing_mean"].min())
        best_sam = float(panel["SAM_missing_mean"].min())
        for idx, (_, row) in enumerate(panel.iterrows()):
            mr_text = f"{missing_rate:.1f}" if idx == 0 else ""
            method = "NTD-PL" if row["method_name"] == "ntdpl" else "Tucker"
            rmse_text = pm_latex(float(row["RMSE_missing_mean"]), float(row["RMSE_missing_std"]), 4)
            nmse_text = pm_latex(float(row["NMSE_dB_missing_mean"]), float(row["NMSE_dB_missing_std"]), 3)
            sam_text = pm_latex(float(row["SAM_missing_mean"]), float(row["SAM_missing_std"]), 2)
            if np.isclose(float(row["RMSE_missing_mean"]), best_rmse):
                rmse_text = rf"\textbf{{{rmse_text}}}"
            if np.isclose(float(row["NMSE_dB_missing_mean"]), best_nmse):
                nmse_text = rf"\textbf{{{nmse_text}}}"
            if np.isclose(float(row["SAM_missing_mean"]), best_sam):
                sam_text = rf"\textbf{{{sam_text}}}"
            lines.append(
                "    "
                + " & ".join(
                    [
                        mr_text,
                        method,
                        rmse_text,
                        nmse_text,
                        sam_text,
                    ]
                )
                + r" \\"
            )
        lines.append(r"    \midrule")
    lines[-1] = r"    \bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def latex_full_table(summary: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{c l c c c c c}",
        r"    \toprule",
        r"    $\rho$ & Method & RMSE(all) & RMSE* & SAM(all) & SAM* & NMSE(dB)(all) \\",
        r"    \midrule",
    ]
    for missing_rate in MAIN_MISSING_RATES:
        panel = summary.loc[np.isclose(summary["missing_rate"], missing_rate, atol=1e-12)].copy()
        for idx, (_, row) in enumerate(panel.iterrows()):
            mr_text = f"{missing_rate:.1f}" if idx == 0 else ""
            method = "NTD-PL" if row["method_name"] == "ntdpl" else "Tucker"
            lines.append(
                "    "
                + " & ".join(
                    [
                        mr_text,
                        method,
                        pm_latex(float(row["RMSE_all_mean"]), float(row["RMSE_all_std"]), 4),
                        pm_latex(float(row["RMSE_missing_mean"]), float(row["RMSE_missing_std"]), 4),
                        pm_latex(float(row["SAM_all_mean"]), float(row["SAM_all_std"]), 2),
                        pm_latex(float(row["SAM_missing_mean"]), float(row["SAM_missing_std"]), 2),
                        pm_latex(float(row["NMSE_dB_all_mean"]), float(row["NMSE_dB_all_std"]), 3),
                    ]
                )
                + r" \\"
            )
        lines.append(r"    \midrule")
    lines[-1] = r"    \bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def pm_latex(mean: float, std: float, digits: int) -> str:
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def build_scene_gain_table(scene_mean: pd.DataFrame) -> pd.DataFrame:
    pivot = scene_mean.pivot_table(
        index=["missing_rate", "scene_id"],
        columns="method_name",
        values=["RMSE_missing", "SAM_missing"],
        aggfunc="mean",
    )
    rows: list[dict[str, float | int]] = []
    for (missing_rate, scene_id), payload in pivot.iterrows():
        if ("RMSE_missing", "tucker") not in pivot.columns or ("RMSE_missing", "ntdpl") not in pivot.columns:
            continue
        if ("SAM_missing", "tucker") not in pivot.columns or ("SAM_missing", "ntdpl") not in pivot.columns:
            continue
        rows.append(
            {
                "missing_rate": float(missing_rate),
                "scene_id": int(scene_id),
                "RMSE_missing_tucker": float(payload[("RMSE_missing", "tucker")]),
                "RMSE_missing_ntdpl": float(payload[("RMSE_missing", "ntdpl")]),
                "SAM_missing_tucker": float(payload[("SAM_missing", "tucker")]),
                "SAM_missing_ntdpl": float(payload[("SAM_missing", "ntdpl")]),
                "RMSE_gain": float(payload[("RMSE_missing", "tucker")] - payload[("RMSE_missing", "ntdpl")]),
                "SAM_gain": float(payload[("SAM_missing", "tucker")] - payload[("SAM_missing", "ntdpl")]),
            }
        )
    return pd.DataFrame(rows).sort_values(["missing_rate", "RMSE_gain"], ascending=[True, False]).reset_index(drop=True)


def build_scene_rate_consistency_table(scene_gain: pd.DataFrame) -> pd.DataFrame:
    if scene_gain.empty:
        return pd.DataFrame(
            columns=[
                "scene_id",
                "scene_label",
                "missing_rate",
                "RMSE_gain",
                "SAM_gain",
                "mean_RMSE_gain",
                "mean_SAM_gain",
                "scene_sort_index",
            ]
        )

    frame = scene_gain.copy()
    frame["scene_id"] = maybe_numeric(frame["scene_id"]).astype(int)
    frame["missing_rate"] = maybe_numeric(frame["missing_rate"]).astype(float)
    frame["RMSE_gain"] = maybe_numeric(frame["RMSE_gain"]).astype(float)
    frame["SAM_gain"] = maybe_numeric(frame["SAM_gain"]).astype(float)

    mean_gain = (
        frame.groupby("scene_id", as_index=False)
        .agg(
            mean_RMSE_gain=("RMSE_gain", "mean"),
            mean_SAM_gain=("SAM_gain", "mean"),
        )
        .sort_values(["mean_RMSE_gain", "mean_SAM_gain"], ascending=[False, False])
        .reset_index(drop=True)
    )
    mean_gain["scene_sort_index"] = np.arange(1, len(mean_gain) + 1, dtype=int)
    frame = frame.merge(mean_gain, on="scene_id", how="left")
    frame["scene_label"] = frame["scene_id"].map(lambda value: f"S{int(value):02d}")
    return (
        frame.loc[
            :,
            [
                "scene_id",
                "scene_label",
                "missing_rate",
                "RMSE_gain",
                "SAM_gain",
                "mean_RMSE_gain",
                "mean_SAM_gain",
                "scene_sort_index",
            ],
        ]
        .sort_values(["scene_sort_index", "missing_rate"])
        .reset_index(drop=True)
    )


def _latex_pvalue(value: float) -> str:
    if value < 1e-3:
        base, exp = f"{value:.2e}".split("e")
        exponent = int(exp)
        return f"{float(base):.2f}$\\times 10^{{{exponent}}}$"
    return f"{value:.3f}"


def _rank_biserial(diffs: np.ndarray) -> float:
    nonzero = diffs[~np.isclose(diffs, 0.0)]
    if nonzero.size == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero), method="average")
    positive_rank_sum = float(ranks[nonzero > 0].sum())
    negative_rank_sum = float(ranks[nonzero < 0].sum())
    denom = nonzero.size * (nonzero.size + 1) / 2.0
    if denom <= 0:
        return 0.0
    return (positive_rank_sum - negative_rank_sum) / denom


def build_significance_summary(scene_mean: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for missing_rate in MAIN_MISSING_RATES:
        panel = scene_mean.loc[np.isclose(scene_mean["missing_rate"], missing_rate, atol=1e-12)].copy()
        if panel.empty:
            continue
        pivot = panel.pivot_table(
            index="scene_id",
            columns="method_name",
            values=["RMSE_missing", "SAM_missing"],
            aggfunc="mean",
        )
        for metric, display_name in (("RMSE_missing", "RMSE"), ("SAM_missing", "SAM")):
            diffs = (pivot[(metric, "tucker")] - pivot[(metric, "ntdpl")]).dropna().to_numpy(dtype=float)
            wins = int(np.sum(diffs > 0.0))
            ties = int(np.sum(np.isclose(diffs, 0.0)))
            losses = int(np.sum(diffs < 0.0))
            nonzero = diffs[~np.isclose(diffs, 0.0)]
            sign_p = 1.0 if nonzero.size == 0 else float(binomtest(int(np.sum(nonzero > 0.0)), nonzero.size, 0.5).pvalue)
            try:
                wilcoxon_p = 1.0 if nonzero.size == 0 else float(wilcoxon(nonzero, zero_method="wilcox").pvalue)
            except ValueError:
                wilcoxon_p = 1.0
            rows.append(
                {
                    "Task": f"Completion ($\\rho={missing_rate:.1f}$)",
                    "missing_rate": float(missing_rate),
                    "Metric": display_name,
                    "Wins": wins,
                    "Losses": losses,
                    "Ties": ties,
                    "Wins/Losses": f"{wins}/{losses}",
                    "Win/Loss/Tie": f"{wins}/{losses}/{ties}",
                    "Mean gain": float(diffs.mean()),
                    "Median gain": float(np.median(diffs)),
                    "Sign test p": sign_p,
                    "Wilcoxon p": wilcoxon_p,
                    "Rank-biserial": float(_rank_biserial(diffs)),
                }
            )

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise RuntimeError("No rows available for CAVE random completion significance summary.")
    return summary.sort_values(["missing_rate", "Metric"]).reset_index(drop=True)


def significance_summary_latex(summary: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{@{}c*{4}{c}*{4}{c}@{}}",
        r"\toprule",
        r" & \multicolumn{4}{c}{RMSE} & \multicolumn{4}{c}{SAM} \\",
        r"\cmidrule(lr){2-5}\cmidrule(l){6-9}",
        r"$\rho$ & W/L & Med. gain & Sign $p$ & Wilcoxon $p$ & W/L & Med. gain & Sign $p$ & Wilcoxon $p$ \\",
        r"\midrule",
    ]
    for missing_rate in MAIN_MISSING_RATES:
        panel = summary.loc[np.isclose(summary["missing_rate"], missing_rate, atol=1e-12)].copy()
        if panel.empty:
            continue
        metric_rows = {str(row["Metric"]): row for row in panel.to_dict("records")}
        rmse = metric_rows.get("RMSE")
        sam = metric_rows.get("SAM")
        if rmse is None or sam is None:
            continue
        lines.append(
            f"{missing_rate:.1f} & "
            f"{rmse['Wins/Losses']} & {float(rmse['Median gain']):.4f} & "
            f"{_latex_pvalue(float(rmse['Sign test p']))} & {_latex_pvalue(float(rmse['Wilcoxon p']))} & "
            f"{sam['Wins/Losses']} & {float(sam['Median gain']):.4f} & "
            f"{_latex_pvalue(float(sam['Sign test p']))} & {_latex_pvalue(float(sam['Wilcoxon p']))} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def anomaly_notes(scene_gain: pd.DataFrame) -> str:
    lines = ["# Scene-Level Notes", ""]
    if scene_gain.empty:
        lines.append("No paired Tucker/NTD-PL scene summaries were found.")
        return "\n".join(lines) + "\n"

    for missing_rate in MAIN_MISSING_RATES:
        panel = scene_gain.loc[np.isclose(scene_gain["missing_rate"], missing_rate, atol=1e-12)].copy()
        if panel.empty:
            continue
        negatives = panel.loc[panel["RMSE_gain"] < 0].sort_values("RMSE_gain")
        lines.append(f"## missing_rate = {missing_rate:.1f}")
        if negatives.empty:
            lines.append("- NTD-PL improves RMSE* on all scenes after seed averaging.")
        else:
            for _, row in negatives.head(5).iterrows():
                lines.append(
                    f"- Scene {int(row['scene_id'])}: RMSE gain = {float(row['RMSE_gain']):.5f}, "
                    f"SAM gain = {float(row['SAM_gain']):.4f}"
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def select_representative_scene(frame: pd.DataFrame, scene_gain: pd.DataFrame, *, missing_rate: float = 0.5) -> int:
    panel = scene_gain.loc[np.isclose(scene_gain["missing_rate"], missing_rate, atol=1e-12)].copy()
    if panel.empty:
        return int(frame["scene_id"].iloc[0])
    panel = panel.sort_values("RMSE_gain", ascending=False).reset_index(drop=True)
    positive = panel.loc[panel["RMSE_gain"] > 0].copy()
    if positive.empty:
        return int(panel.iloc[0]["scene_id"])
    return int(positive.iloc[len(positive) // 2]["scene_id"])


def load_scene_payload(frame: pd.DataFrame, *, scene_id: int, missing_rate: float = 0.5) -> ScenePayload:
    panel = frame.loc[
        np.isclose(frame["missing_rate"], missing_rate, atol=1e-12)
        & frame["scene_id"].eq(scene_id)
    ].copy()
    if panel.empty:
        raise RuntimeError(f"No runs found for scene {scene_id} at missing_rate={missing_rate}.")

    selected_rows = []
    for method_name in MAIN_METHODS:
        method_panel = panel.loc[panel["method_name"] == method_name].copy()
        method_panel = method_panel.sort_values("RMSE_missing").reset_index(drop=True)
        if method_panel.empty:
            raise RuntimeError(f"No {method_name} rows found for scene {scene_id} at missing_rate={missing_rate}.")
        selected_rows.append(method_panel.iloc[0])

    dataset_kwargs = _cave_dataset_kwargs_from_row(selected_rows[0])
    dataset = CAVEHSIData(
        path=dataset_kwargs["path"],
        id=scene_id,
        target_shape=dataset_kwargs["target_shape"],
        crop_shape=dataset_kwargs["crop_shape"],
    )
    BiasFilter(seed=0, normalize_method="max", snr_db=None, bias=None)(dataset)
    original = np.asarray(dataset.get(split="eval").dense, dtype=np.float32)

    state_lookup = {}
    for row in selected_rows:
        state = load_state_mat(_resolve_state_path(row["state_path"]))
        state_lookup[str(row["method_name"])] = state

    observed_mask = np.asarray(state_lookup["tucker"]["observed_mask"], dtype=bool)
    tucker_recon = np.asarray(state_lookup["tucker"]["reconstruction"], dtype=np.float32)
    ntdpl_recon = np.asarray(state_lookup["ntdpl"]["reconstruction"], dtype=np.float32)
    return ScenePayload(
        scene_id=scene_id,
        scene_name=str(getattr(dataset, "scene_name", f"scene-{scene_id}")),
        original=original,
        observed_mask=observed_mask,
        recon_tucker=tucker_recon,
        recon_ntdpl=ntdpl_recon,
    )


def pseudo_rgb(cube: np.ndarray) -> np.ndarray:
    band_count = cube.shape[-1]
    indices = [int(round((band_count - 1) * frac)) for frac in (0.75, 0.5, 0.2)]
    rgb = np.stack([cube[..., idx] for idx in indices], axis=-1)
    rgb = np.clip(rgb, 0.0, None)
    scale = float(np.max(rgb))
    if scale > 1e-12:
        rgb = rgb / scale
    return rgb


def observed_fraction_map(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 3:
        raise ValueError(f"Expected 3D observed mask, got {mask.shape}.")
    return np.mean(mask.astype(np.float32), axis=-1)


def rmse_map(original: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
    diff = np.asarray(original, dtype=np.float32) - np.asarray(reconstructed, dtype=np.float32)
    return np.sqrt(np.mean(diff * diff, axis=-1))
