from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ...utils.io import load_state_mat, maybe_numeric
from ...utils.paper import write_csv_artifact, write_text_artifact
from ..common import float_mask, load_results, select_method_runs
from ..registry import register_postprocessor


SETTING_LABELS = {
    0.0: "bias=0",
    0.5: "bias=0.5",
}

CONSTRAINT_LABELS = {
    "tucker": "Linear Tucker baseline",
    "strict": r"$P=6,\ \beta_0=0$",
    "affine": r"$P=6,\ \beta_0$ free",
}


@dataclass(frozen=True)
class RowSpec:
    setting_bias: float
    method_key: str
    method_label: str


ROW_SPECS = [
    RowSpec(0.0, "tucker", "Tucker"),
    RowSpec(0.0, "strict", r"NTD-PL ($\beta_0 = 0$)"),
    RowSpec(0.0, "affine", r"NTD-PL ($\beta_0$ free)"),
    RowSpec(0.5, "tucker", "Tucker"),
    RowSpec(0.5, "strict", r"NTD-PL ($\beta_0 = 0$)"),
    RowSpec(0.5, "affine", r"NTD-PL ($\beta_0$ free)"),
]


def _bool_series(frame: pd.DataFrame, column: str, default: bool) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([default] * len(frame), index=frame.index)
    return frame[column].astype(str).str.lower().eq("true")


def _select_rows(frame: pd.DataFrame, *, bias: float, method_key: str) -> pd.DataFrame:
    panel = frame.loc[float_mask(frame["ovr.filter.bias"], bias)].copy()
    if method_key == "tucker":
        return panel.loc[panel["ovr.method"].astype(str) == "tucker"].copy()

    rows = select_method_runs(panel, "ntdpl", p_max=6)
    allow_constant = method_key == "affine"
    mask = _bool_series(rows, "ovr.method.allow_constant_term", default=True)
    return rows.loc[mask if allow_constant else ~mask].copy()


def _format_mean_std(series: pd.Series, digits: int) -> str:
    values = maybe_numeric(series).dropna().to_numpy(dtype=float)
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    return f"{mean:.{digits}f} +- {std:.{digits}f}"


def _higher_order_contribution_text(state_paths: pd.Series) -> str:
    contributions: list[float] = []
    try:
        from tensorly.tucker_tensor import tucker_to_tensor
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing optional dependency 'tensorly'. Install it to compute higher-order contribution diagnostics."
        ) from exc

    for state_path in state_paths.dropna():
        state = load_state_mat(Path(str(state_path)))
        beta = np.asarray(state["beta"], dtype=float).reshape(-1)
        core = np.asarray(state["core"], dtype=float)
        factors_obj = state["factors"]
        if isinstance(factors_obj, np.ndarray) and factors_obj.dtype == object:
            factors = [np.asarray(factors_obj[0, idx], dtype=float) for idx in range(factors_obj.shape[1])]
        elif isinstance(factors_obj, list):
            factors = [np.asarray(item, dtype=float) for item in factors_obj]
        else:
            factors = [np.asarray(factors_obj, dtype=float)]

        latent = np.asarray(tucker_to_tensor((core, factors)), dtype=float)
        term_norms = np.asarray(
            [float(np.linalg.norm(float(coeff) * np.power(latent, degree))) for degree, coeff in enumerate(beta)],
            dtype=float,
        )
        denom = max(float(term_norms.sum()), 1e-12)
        contributions.append(100.0 * float(term_norms[2:].sum()) / denom)

    return _format_mean_std(pd.Series(contributions, dtype=float), digits=2)


def _paired_rmse_gap(frame: pd.DataFrame, *, bias: float, method_key: str) -> tuple[str, pd.DataFrame]:
    if method_key == "tucker":
        return "---", pd.DataFrame()

    tucker = _select_rows(frame, bias=bias, method_key="tucker")
    other = _select_rows(frame, bias=bias, method_key=method_key)
    merged = (
        tucker.loc[:, ["ovr.data.seed", "RMSE"]]
        .rename(columns={"RMSE": "RMSE_tucker"})
        .merge(
            other.loc[:, ["ovr.data.seed", "RMSE"]],
            on="ovr.data.seed",
            how="inner",
        )
        .rename(columns={"RMSE": "RMSE_ntdpl"})
        .sort_values("ovr.data.seed")
    )
    merged["RMSE_gap"] = merged["RMSE_tucker"] - merged["RMSE_ntdpl"]
    return _format_mean_std(merged["RMSE_gap"], digits=6), merged


def _beta_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in ROW_SPECS:
        if spec.method_key == "tucker":
            continue
        subset = _select_rows(frame, bias=spec.setting_bias, method_key=spec.method_key)
        betas: list[np.ndarray] = []
        for state_path in subset["state_path"].dropna():
            mat = load_state_mat(Path(str(state_path)))
            betas.append(np.asarray(mat["beta"], dtype=float).reshape(-1))
        if not betas:
            continue
        beta_stack = np.vstack(betas)
        rows.append(
            {
                "Setting": SETTING_LABELS[spec.setting_bias],
                "Method": spec.method_label,
                "beta_0_mean": float(beta_stack[:, 0].mean()),
                "beta_0_std": float(beta_stack[:, 0].std(ddof=0)),
                "beta_1_mean": float(beta_stack[:, 1].mean()) if beta_stack.shape[1] > 1 else np.nan,
                "beta_1_std": float(beta_stack[:, 1].std(ddof=0)) if beta_stack.shape[1] > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _latex_table(summary: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{c|c|c|c|c}",
        r"    \hline",
        r"    Setting & Method & RMSE & NMSE(dB) & $p>1$ contrib. (\%) \\",
        r"    \hline",
    ]
    for setting in summary["Setting"].drop_duplicates():
        panel = summary.loc[summary["Setting"] == setting].to_dict("records")
        for idx, row in enumerate(panel):
            setting_text = rf"\multirow{{{len(panel)}}}{{*}}{{{row['Setting']}}}" if idx == 0 else ""
            cells = [
                setting_text,
                row["Method"],
                str(row["RMSE"]).replace("+-", r"$\pm$"),
                str(row["NMSE(dB)"]).replace("+-", r"$\pm$"),
                str(row["HO contrib.(%)"]).replace("+-", r"$\pm$"),
            ]
            lines.append("    " + " & ".join(cells) + r" \\")
        lines.append(r"    \hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


@register_postprocessor(order=20)
def linear_consistency_table() -> None:
    loaded = load_results("linear-consistency", require_curves=False)
    frame = loaded.runs.copy()

    rows: list[dict[str, object]] = []
    for spec in ROW_SPECS:
        subset = _select_rows(frame, bias=spec.setting_bias, method_key=spec.method_key)
        if subset.empty:
            raise RuntimeError(
                f"Missing rows for bias={spec.setting_bias:g}, method_key={spec.method_key} in linear-consistency."
            )
        rows.append(
            {
                "Setting": SETTING_LABELS[spec.setting_bias],
                "Method": spec.method_label,
                "RMSE": _format_mean_std(subset["RMSE"], digits=6),
                "NMSE(dB)": _format_mean_std(subset["NMSE_dB"], digits=4),
                "HO contrib.(%)": "---"
                if spec.method_key == "tucker"
                else _higher_order_contribution_text(subset["state_path"]),
            }
        )

    summary = pd.DataFrame(rows)
    csv_path, latex_csv_path = write_csv_artifact(
        loaded.env,
        summary,
        artifact_name="table.csv",
        latex_name="table.csv",
    )
    tex_path, latex_tex_path = write_text_artifact(
        loaded.env,
        _latex_table(summary),
        artifact_name="table.tex",
        latex_name="table.tex",
    )

    beta_summary = _beta_summary(frame)
    beta_csv_path, beta_latex_path = write_csv_artifact(
        loaded.env,
        beta_summary,
        artifact_name="beta_summary.csv",
        latex_name="beta_summary.csv",
    )

    print(summary.to_string(index=False))
    print(f"\nSaved: {csv_path}")
    print(f"Synced: {latex_csv_path}")
    print(f"Saved: {tex_path}")
    print(f"Synced: {latex_tex_path}")
    print(f"Saved: {beta_csv_path}")
    print(f"Synced: {beta_latex_path}")
