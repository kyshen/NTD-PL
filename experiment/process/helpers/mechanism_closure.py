from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...config import get_env
from ...hsi_defaults import CAVE_RECON_MAIN_RANK
from ...utils.io import load_run_parquets, load_state_mat, maybe_numeric
from src.data.hsi import CAVEHSIData
from src.filters.bias import BiasFilter
from src.methods.polycal import PolynomialCalibration
from src.metrics import val_RMSE, val_SAM
from src.types import Tensor

from .cave_random_completion import _cave_dataset_kwargs_from_row
from .cave_random_completion_polycal import (
    MAIN_MISSING_RATE as COMPLETION_MAIN_MISSING_RATE,
    MAIN_POLYCAL_DEGREE,
    POLYCAL_LAMBDA,
)


RECON_MAIN_RANK = CAVE_RECON_MAIN_RANK
RECON_NTDPL_PMAX = 6
COMPLETION_NTDPL_PMAX = 4
CAVE_FULL_SHAPE = (512, 512, 31)

METHOD_ORDER = ("tucker", f"tucker_polycal_p{MAIN_POLYCAL_DEGREE}", "ntdpl")
METHOD_DISPLAY = {
    "tucker": "Tucker",
    f"tucker_polycal_p{MAIN_POLYCAL_DEGREE}": "Tucker + PolyCal",
    "ntdpl": "NTD-PL",
}
SETTING_ORDER = ("full_reconstruction", "completion")
SETTING_DISPLAY = {
    "full_reconstruction": "Full reconstruction",
    "completion": rf"Completion ($\rho={COMPLETION_MAIN_MISSING_RATE:.1f}$)",
}
TABLE_SETTING_DISPLAY = {
    "full_reconstruction": "Recon.",
    "completion": "Compl.",
}


@dataclass(frozen=True)
class MetricSummary:
    rmse_mean: float
    rmse_std: float
    sam_mean: float
    sam_std: float
    n_scenes: int


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
        items = [part.strip() for part in parsed.strip("[]()").split(",") if part.strip()]
        return tuple(int(item) for item in items)  # type: ignore[return-value]
    if isinstance(parsed, (list, tuple)):
        return tuple(int(item) for item in parsed)  # type: ignore[return-value]
    raise ValueError(f"Cannot parse rank from value: {value}")


def _pm_text(mean: float, std: float, digits: int) -> str:
    return f"{mean:.{digits}f} +- {std:.{digits}f}"


def _pm_latex(mean: float, std: float, digits: int) -> str:
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def _param_count_tucker(rank: tuple[int, int, int], shape: tuple[int, int, int] = CAVE_FULL_SHAPE) -> int:
    core = int(np.prod(rank))
    factors = int(sum(dim * r for dim, r in zip(shape, rank, strict=True)))
    return core + factors


def _method_param_count(*, setting_key: str, method_name: str) -> tuple[int, str]:
    base = _param_count_tucker(RECON_MAIN_RANK)
    if method_name == "tucker":
        return base, "Backbone only"
    if method_name == f"tucker_polycal_p{MAIN_POLYCAL_DEGREE}":
        return base + (MAIN_POLYCAL_DEGREE + 1), f"Backbone + {MAIN_POLYCAL_DEGREE + 1} beta coeffs"
    if method_name == "ntdpl":
        pmax = RECON_NTDPL_PMAX if setting_key == "full_reconstruction" else COMPLETION_NTDPL_PMAX
        return base + (pmax + 1), f"Joint model, $p_{{max}}={pmax}$"
    raise KeyError(f"Unsupported method for params: {method_name}")


def _load_scene_original(scene_id: int, row: pd.Series) -> np.ndarray:
    kwargs = _cave_dataset_kwargs_from_row(row)
    dataset = CAVEHSIData(
        path=kwargs["path"],
        id=int(scene_id),
        target_shape=kwargs["target_shape"],
        crop_shape=kwargs["crop_shape"],
    )
    BiasFilter(seed=0, normalize_method="max", snr_db=None, bias=None)(dataset)
    return np.asarray(dataset.get(split="eval").dense, dtype=np.float32)


def _resolve_state_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return get_env("cave-representation").project_root / path


def _state_reconstruction(state: dict[str, Any]) -> np.ndarray:
    for key in ("reconstruction", "fitted"):
        if key in state:
            value = np.asarray(_jsonish(state[key]), dtype=np.float32)
            if value.ndim == 3:
                return value
    if "core" in state and "factors" in state:
        try:
            from tensorly.tucker_tensor import tucker_to_tensor
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Missing dependency 'tensorly' to rebuild reconstruction from core/factors.") from exc
        core = np.asarray(state["core"], dtype=np.float32)
        factors_raw = state["factors"]
        if isinstance(factors_raw, np.ndarray) and factors_raw.dtype == object:
            factors = [np.asarray(item, dtype=np.float32) for item in factors_raw.reshape(-1)]
        elif isinstance(factors_raw, list):
            factors = [np.asarray(item, dtype=np.float32) for item in factors_raw]
        else:
            raise TypeError(f"Unsupported factors payload type: {type(factors_raw)}")
        latent = np.asarray(tucker_to_tensor((core, factors)), dtype=np.float32)
        beta = np.asarray(state.get("beta", []), dtype=np.float32).reshape(-1)
        if beta.size <= 1:
            return latent
        out = np.zeros_like(latent, dtype=np.float32)
        for degree, coeff in enumerate(beta):
            out = out + float(coeff) * (latent ** degree)
        return out
    raise KeyError("State contains neither reconstruction/fitted nor core/factors payload.")


def _load_recon_runs_main_rank() -> pd.DataFrame:
    env = get_env("cave-representation")
    runs = load_run_parquets(env.results_dir)["runs"].copy()
    if runs.empty:
        raise RuntimeError("No cave-representation runs found.")
    frame = runs.copy()
    frame["method_name"] = frame.get("method._name", frame.get("ovr.method")).astype(str)
    frame["scene_id"] = maybe_numeric(frame.get("data.id", frame.get("ovr.data.id"))).astype(int)
    frame["rank"] = frame.get("method.rank", frame.get("ovr.method.rank")).map(_parse_rank)
    if "ovr.method.p_max" in frame.columns:
        frame["p_max"] = maybe_numeric(frame["ovr.method.p_max"])
    else:
        frame["p_max"] = maybe_numeric(frame.get("method.p_max", np.nan))
    frame["RMSE"] = maybe_numeric(frame["RMSE"]).astype(float)
    frame["SAM"] = maybe_numeric(frame["SAM"]).astype(float)
    frame = frame.loc[
        frame["method_name"].isin(("tucker", "ntdpl"))
        & frame["rank"].map(lambda value: tuple(value) == RECON_MAIN_RANK)
    ].copy()
    frame = frame.loc[
        ~frame["method_name"].eq("ntdpl")
        | np.isclose(frame["p_max"], float(RECON_NTDPL_PMAX), atol=1e-12)
    ].copy()
    frame = frame.sort_values("run_dir").drop_duplicates(
        subset=["scene_id", "method_name"],
        keep="last",
    )
    return frame.reset_index(drop=True)


def _polycal_recon_summary() -> tuple[MetricSummary, pd.DataFrame]:
    frame = _load_recon_runs_main_rank()
    tucker = frame.loc[frame["method_name"] == "tucker"].copy()
    if tucker.empty:
        raise RuntimeError("No Tucker rows found for main-rank reconstruction.")
    rows: list[dict[str, float | int]] = []
    for row in tucker.itertuples(index=False):
        state = load_state_mat(_resolve_state_path(str(row.state_path)))
        recon_tucker = _state_reconstruction(state)
        original = _load_scene_original(int(row.scene_id), pd.Series(row._asdict()))
        observed_mask = np.ones_like(original, dtype=bool)
        model = PolynomialCalibration(degree=MAIN_POLYCAL_DEGREE, lambda_reg=POLYCAL_LAMBDA).fit(
            recon_tucker,
            original,
            observed_mask,
        )
        recon_polycal = model.apply(recon_tucker)
        original_tensor = Tensor(shape=original.shape, dense=original)
        recon_tensor = Tensor(shape=recon_polycal.shape, dense=recon_polycal)
        rows.append(
            {
                "scene_id": int(row.scene_id),
                "RMSE": float(val_RMSE(original_tensor, recon_tensor)),
                "SAM": float(val_SAM(original_tensor, recon_tensor)),
            }
        )
    poly_scene = pd.DataFrame(rows).sort_values("scene_id").reset_index(drop=True)
    summary = MetricSummary(
        rmse_mean=float(poly_scene["RMSE"].mean()),
        rmse_std=float(poly_scene["RMSE"].std(ddof=0)),
        sam_mean=float(poly_scene["SAM"].mean()),
        sam_std=float(poly_scene["SAM"].std(ddof=0)),
        n_scenes=int(poly_scene["scene_id"].nunique()),
    )
    return summary, poly_scene


def _recon_main_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = _load_recon_runs_main_rank()
    grouped = (
        frame.groupby("method_name", as_index=False)
        .agg(
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", lambda s: float(np.std(s.to_numpy(dtype=float), ddof=0))),
            SAM_mean=("SAM", "mean"),
            SAM_std=("SAM", lambda s: float(np.std(s.to_numpy(dtype=float), ddof=0))),
            n_scenes=("scene_id", "nunique"),
        )
        .sort_values("method_name")
        .reset_index(drop=True)
    )
    poly_summary, poly_scene = _polycal_recon_summary()
    poly_row = pd.DataFrame(
        [
            {
                "method_name": f"tucker_polycal_p{MAIN_POLYCAL_DEGREE}",
                "RMSE_mean": poly_summary.rmse_mean,
                "RMSE_std": poly_summary.rmse_std,
                "SAM_mean": poly_summary.sam_mean,
                "SAM_std": poly_summary.sam_std,
                "n_scenes": poly_summary.n_scenes,
            }
        ]
    )
    merged = (
        pd.concat([grouped, poly_row], ignore_index=True)
        .loc[lambda df: df["method_name"].isin(METHOD_ORDER)]
        .reset_index(drop=True)
    )
    return merged, poly_scene


def _completion_summary() -> pd.DataFrame:
    env = get_env("cave-random-completion")
    summary_path = env.artifacts_dir / "polycal_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"{summary_path} not found. Run cave-random-completion postprocess first to generate polycal summary."
        )
    frame = pd.read_csv(summary_path)
    panel = frame.loc[
        np.isclose(frame["missing_rate"], COMPLETION_MAIN_MISSING_RATE, atol=1e-12)
        & frame["method_name"].isin(METHOD_ORDER)
    ].copy()
    panel = panel.sort_values("method_name").reset_index(drop=True)
    return panel


def build_mechanism_closure_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    recon_summary, _ = _recon_main_summary()
    completion_summary = _completion_summary()

    rows: list[dict[str, Any]] = []
    for setting_key, source in (
        ("full_reconstruction", recon_summary),
        ("completion", completion_summary),
    ):
        for method_name in METHOD_ORDER:
            panel = source.loc[source["method_name"] == method_name]
            if panel.empty:
                continue
            row = panel.iloc[0]
            params, param_note = _method_param_count(setting_key=setting_key, method_name=method_name)
            if setting_key == "full_reconstruction":
                rmse_mean = float(row["RMSE_mean"])
                rmse_std = float(row["RMSE_std"])
                sam_mean = float(row["SAM_mean"])
                sam_std = float(row["SAM_std"])
                rmse_metric = "RMSE"
                sam_metric = "SAM"
            else:
                rmse_mean = float(row["RMSE_missing_mean"])
                rmse_std = float(row["RMSE_missing_std"])
                sam_mean = float(row["SAM_missing_mean"])
                sam_std = float(row["SAM_missing_std"])
                rmse_metric = "RMSE*"
                sam_metric = "SAM*"
            rows.append(
                {
                    "setting_key": setting_key,
                    "setting": SETTING_DISPLAY[setting_key],
                    "method_name": method_name,
                    "method": METHOD_DISPLAY[method_name],
                    "params": int(params),
                    "param_note": param_note,
                    "rmse_metric": rmse_metric,
                    "sam_metric": sam_metric,
                    "rmse_mean": rmse_mean,
                    "rmse_std": rmse_std,
                    "sam_mean": sam_mean,
                    "sam_std": sam_std,
                }
            )
    numeric = pd.DataFrame(rows)
    if numeric.empty:
        raise RuntimeError("Mechanism closure table is empty.")

    numeric["setting_order"] = numeric["setting_key"].map({key: idx for idx, key in enumerate(SETTING_ORDER)})
    numeric["method_order"] = numeric["method_name"].map({key: idx for idx, key in enumerate(METHOD_ORDER)})
    numeric = numeric.sort_values(["setting_order", "method_order"]).drop(columns=["setting_order", "method_order"])

    display_rows: list[dict[str, Any]] = []
    for setting_key in SETTING_ORDER:
        panel = numeric.loc[numeric["setting_key"] == setting_key].copy()
        if panel.empty:
            continue
        best_rmse = float(panel["rmse_mean"].min())
        best_sam = float(panel["sam_mean"].min())
        for idx, row in enumerate(panel.itertuples(index=False), start=1):
            method_label = "PolyCal" if row.method_name == f"tucker_polycal_p{MAIN_POLYCAL_DEGREE}" else row.method
            display_rows.append(
                {
                    "setting": TABLE_SETTING_DISPLAY[setting_key] if idx == 1 else "",
                    "method": method_label,
                    "params": f"{int(row.params):,}",
                    "rmse_metric": row.rmse_metric,
                    "rmse": _pm_text(float(row.rmse_mean), float(row.rmse_std), 5),
                    "sam_metric": row.sam_metric,
                    "sam": _pm_text(float(row.sam_mean), float(row.sam_std), 4),
                    "param_note": row.param_note,
                    "is_best_rmse": np.isclose(float(row.rmse_mean), best_rmse, atol=1e-12),
                    "is_best_sam": np.isclose(float(row.sam_mean), best_sam, atol=1e-12),
                }
            )
    display = pd.DataFrame(display_rows)

    figure_rows: list[dict[str, Any]] = []
    panel_def = (
        ("A", "full_reconstruction", "rmse_mean", "rmse_std", "Full reconstruction, RMSE"),
        ("B", "full_reconstruction", "sam_mean", "sam_std", "Full reconstruction, SAM"),
        ("C", "completion", "rmse_mean", "rmse_std", "Completion, RMSE*"),
        ("D", "completion", "sam_mean", "sam_std", "Completion, SAM*"),
    )
    x_map = {name: idx for idx, name in enumerate(METHOD_ORDER)}
    for panel_key, setting_key, mean_col, std_col, panel_title in panel_def:
        panel = numeric.loc[numeric["setting_key"] == setting_key].copy()
        for method_name in METHOD_ORDER:
            row = panel.loc[panel["method_name"] == method_name]
            if row.empty:
                continue
            item = row.iloc[0]
            figure_rows.append(
                {
                    "panel": panel_key,
                    "panel_title": panel_title,
                    "method": METHOD_DISPLAY[method_name],
                    "x": float(x_map[method_name]),
                    "mean": float(item[mean_col]),
                    "std": float(item[std_col]),
                    "band_lower": float(item[mean_col] - item[std_col]),
                    "band_upper": float(item[mean_col] + item[std_col]),
                    "annotation": (
                        "Params are near-matched: Tucker vs NTD-PL differ by only a few coeffs; "
                        "the fixed-backbone beta refresh leaves Tucker geometry frozen."
                    ),
                }
            )
    figure_data = pd.DataFrame(figure_rows).sort_values(["panel", "x"]).reset_index(drop=True)
    if figure_data.empty:
        raise RuntimeError("Mechanism closure figure data is empty.")
    return numeric.reset_index(drop=True), display.reset_index(drop=True), figure_data


def mechanism_closure_main_table_latex(display: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{@{}c l r c c@{}}",
        r"\toprule",
        r"Setting & Method & Param. & RMSE$\downarrow$ & SAM$\downarrow$ \\",
        r"\midrule",
    ]
    seen_setting = False
    records = display.to_dict("records")
    for row_idx, row in enumerate(records):
        rmse = str(row["rmse"]).replace("+-", r"$\pm$")
        sam = str(row["sam"]).replace("+-", r"$\pm$")
        if bool(row["is_best_rmse"]):
            rmse = rf"\textbf{{{rmse}}}"
        if bool(row["is_best_sam"]):
            sam = rf"\textbf{{{sam}}}"
        setting_raw = str(row["setting"])
        if setting_raw and seen_setting:
            lines.append(r"    \midrule")
        if setting_raw:
            seen_setting = True
            span = 1
            for next_idx in range(row_idx + 1, len(records)):
                if str(records[next_idx]["setting"]):
                    break
                span += 1
            setting = rf"\multirow{{{span}}}{{*}}{{{setting_raw}}}"
        else:
            setting = ""
        lines.append(
            "    "
            + " & ".join(
                [
                    setting,
                    str(row["method"]),
                    str(row["params"]),
                    rmse,
                    sam,
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def mechanism_closure_main_table_numeric_csv(numeric: pd.DataFrame) -> pd.DataFrame:
    out = numeric.copy()
    out["rmse_pm"] = out.apply(lambda r: _pm_text(float(r["rmse_mean"]), float(r["rmse_std"]), 5), axis=1)
    out["sam_pm"] = out.apply(lambda r: _pm_text(float(r["sam_mean"]), float(r["sam_std"]), 4), axis=1)
    return out.loc[
        :,
        [
            "setting_key",
            "setting",
            "method_name",
            "method",
            "params",
            "param_note",
            "rmse_metric",
            "rmse_mean",
            "rmse_std",
            "rmse_pm",
            "sam_metric",
            "sam_mean",
            "sam_std",
            "sam_pm",
        ],
    ].reset_index(drop=True)
