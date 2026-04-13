from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tensorly.tucker_tensor import tucker_to_tensor

from ...config import get_env
from ...utils.io import load_state_mat, maybe_numeric
from ...utils.paper import write_csv_artifact, write_text_artifact
from ..common import float_mask, load_results, select_method_runs
from ..nonlinear_approx import dedup_nonlinear_runs
from ..registry import register_postprocessor


ALPHA_REF = 0.25
P_MAX_REF = 6
NONLINEAR_ORDER = ("poly2", "poly3", "sin", "tanh")


def _factors_from_state(state: dict[str, object]) -> list[np.ndarray]:
    factors_obj = state["factors"]
    if isinstance(factors_obj, np.ndarray) and factors_obj.dtype == object:
        return [np.asarray(factors_obj[0, idx], dtype=float) for idx in range(factors_obj.shape[1])]
    if isinstance(factors_obj, list):
        return [np.asarray(item, dtype=float) for item in factors_obj]
    return [np.asarray(factors_obj, dtype=float)]


def _summary_frame() -> pd.DataFrame:
    loaded = load_results("nonlinear-approx", require_curves=False)
    frame = dedup_nonlinear_runs(loaded.runs.copy())
    frame = frame.loc[float_mask(frame["ovr.filter.alpha"], ALPHA_REF)].copy()
    frame = select_method_runs(frame, "ntdpl", p_max=P_MAX_REF)
    if frame.empty:
        raise RuntimeError(
            f"No nonlinear-approx NTD-PL runs found for alpha={ALPHA_REF:g}, p_max={P_MAX_REF}."
        )

    rows: list[dict[str, float | int | str]] = []
    for nonlinear in NONLINEAR_ORDER:
        subset = frame.loc[frame["ovr.filter.nonlinear"].astype(str) == nonlinear].copy()
        if subset.empty:
            raise RuntimeError(
                f"Missing nonlinear-approx NTD-PL runs for nonlinear='{nonlinear}', alpha={ALPHA_REF:g}, p_max={P_MAX_REF}."
            )
        for state_path in subset["state_path"].dropna():
            state = load_state_mat(Path(str(state_path)))
            beta = np.asarray(state["beta"], dtype=float).reshape(-1)
            core = np.asarray(state["core"], dtype=float)
            latent = np.asarray(tucker_to_tensor((core, _factors_from_state(state))), dtype=float)
            terms = [
                float(beta[degree]) * np.power(latent, degree) if degree < len(beta) else np.zeros_like(latent)
                for degree in range(P_MAX_REF + 1)
            ]
            term_norms = np.asarray([float(np.linalg.norm(term)) for term in terms], dtype=float)
            total_term_norm = max(float(term_norms.sum()), 1e-12)
            for degree, term_norm in enumerate(term_norms):
                rows.append(
                    {
                        "nonlinear": nonlinear,
                        "degree": int(degree),
                        "effective_contribution_pct": 100.0 * float(term_norm) / total_term_norm,
                    }
                )

    summary = (
        pd.DataFrame(rows)
        .groupby(["nonlinear", "degree"], as_index=False)
        .agg(
            contribution_mean=("effective_contribution_pct", "mean"),
            contribution_std=("effective_contribution_pct", "std"),
            n_runs=("effective_contribution_pct", "size"),
        )
        .sort_values(["nonlinear", "degree"])
        .reset_index(drop=True)
    )
    summary["contribution_std"] = maybe_numeric(summary["contribution_std"]).fillna(0.0)
    return summary


def _latex_table(summary: pd.DataFrame) -> str:
    wide_mean = (
        summary.pivot(index="nonlinear", columns="degree", values="contribution_mean")
        .reindex(index=list(NONLINEAR_ORDER), columns=list(range(P_MAX_REF + 1)))
        .reset_index()
    )
    wide_std = (
        summary.pivot(index="nonlinear", columns="degree", values="contribution_std")
        .reindex(index=list(NONLINEAR_ORDER), columns=list(range(P_MAX_REF + 1)))
        .reset_index()
    )
    lines = [
        r"\begin{tabular}{c|c|c|c|c|c|c|c}",
        r"    \hline",
        rf"    nonlinear($\alpha={ALPHA_REF:g}$) & $\beta_0$ & $\beta_1$ & $\beta_2$ & $\beta_3$ & $\beta_4$ & $\beta_5$ & $\beta_6$ \\",
        r"    \hline",
    ]
    mean_rows = wide_mean.to_dict("records")
    std_rows = {str(row["nonlinear"]): row for row in wide_std.to_dict("records")}
    for row in mean_rows:
        cells = [str(row["nonlinear"])]
        std_row = std_rows[str(row["nonlinear"])]
        for degree in range(P_MAX_REF + 1):
            value = float(row[degree])
            std = float(std_row[degree])
            cells.append(f"{value:.1f} $\\pm$ {std:.1f}")
        lines.append("    " + " & ".join(cells) + r" \\")
    lines.extend([r"    \hline", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


@register_postprocessor(exp_name="nonlinear-approx", order=46)
def nonlinear_beta_contribution_table() -> None:
    env = get_env("nonlinear-approx")
    summary = _summary_frame()
    csv_path, latex_csv_path = write_csv_artifact(env, summary, "beta_contribution_summary.csv")
    tex_path, latex_tex_path = write_text_artifact(env, _latex_table(summary), "beta_contribution_table.tex")
    print(summary.to_string(index=False))
    print(f"Saved: {csv_path}")
    print(f"Synced: {latex_csv_path}")
    print(f"Saved: {tex_path}")
    print(f"Synced: {latex_tex_path}")
