from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "results" / "cave_main_baselines_r24_full"
TABLE_PATH = PROJECT_ROOT / "papers" / "tsp" / "tables" / "cave_main_baselines.tex"
SUMMARY_PATH = RESULTS_DIR / "multibudget_summary.csv"

BUDGETS = (
    {"name": "low", "label": r"Low ($\approx13$k)", "rank": "(12,12,4)"},
    {"name": "medium", "label": r"Medium ($\approx27$k)", "rank": "(24,24,4)"},
    {"name": "high", "label": r"High ($\approx42$k)", "rank": "(36,36,4)"},
)
METHOD_ORDER = ("cp", "tt", "tucker", "ntdpl_p2", "ntdpl_p4", "ntdpl")
METHOD_LABELS = {
    "cp": "CP",
    "tt": "Tensor Train",
    "tucker": "Tucker",
    "ntdpl_p2": r"NTD-PL ($P=2$)",
    "ntdpl_p4": r"NTD-PL ($P=4$)",
    "ntdpl": r"\textbf{NTD-PL ($P=6$)}",
}


def _read_rows() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    wanted_ranks = {item["rank"] for item in BUDGETS}
    for path in sorted(RESULTS_DIR.glob("*_r*_s*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("rank") in wanted_ranks:
            rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No baseline JSON files found in {RESULTS_DIR}")
    return pd.DataFrame(rows)


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    out = (
        frame.groupby(["rank", "method_name"], as_index=False)
        .agg(
            params=("params", "mean"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            NMSE_dB_mean=("NMSE_dB", "mean"),
            NMSE_dB_std=("NMSE_dB", "std"),
            n=("scene_id", "nunique"),
        )
    )
    rank_to_budget = {item["rank"]: item["name"] for item in BUDGETS}
    out["budget"] = out["rank"].map(rank_to_budget)
    return out


def _metric_text(mean: float, std: float, digits: int, bold: bool) -> str:
    text = f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"
    return rf"\textbf{{{text}}}" if bold else text


def _row(summary: pd.DataFrame, budget: str, method: str) -> pd.Series:
    rows = summary.loc[
        summary["budget"].eq(budget) & summary["method_name"].eq(method)
    ]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one row for budget={budget}, method={method}; found {len(rows)}")
    row = rows.iloc[0]
    if int(row["n"]) != 15:
        raise RuntimeError(f"Incomplete result for budget={budget}, method={method}: n={int(row['n'])}")
    return row


def _write_table(summary: pd.DataFrame) -> None:
    best: dict[tuple[str, str], float] = {}
    for budget in (item["name"] for item in BUDGETS):
        subset = summary.loc[
            summary["budget"].eq(budget) & summary["method_name"].isin(METHOD_ORDER)
        ]
        if len(subset) != len(METHOD_ORDER):
            raise RuntimeError(f"Budget {budget} is incomplete: found {len(subset)}/{len(METHOD_ORDER)} methods")
        best[(budget, "RMSE")] = float(subset["RMSE_mean"].min())
        best[(budget, "NMSE")] = float(subset["NMSE_dB_mean"].min())

    lines = [
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l c c c c c c@{}}",
        r"\toprule",
        "Method"
        + "".join(rf" & \multicolumn{{2}}{{c}}{{{item['label']}}}" for item in BUDGETS)
        + r" \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r" & RMSE$\downarrow$ & NMSE(dB)$\downarrow$"
        r" & RMSE$\downarrow$ & NMSE(dB)$\downarrow$"
        r" & RMSE$\downarrow$ & NMSE(dB)$\downarrow$ \\",
        r"\midrule",
    ]
    for method in METHOD_ORDER:
        cells = [METHOD_LABELS[method]]
        for item in BUDGETS:
            row = _row(summary, item["name"], method)
            cells.append(
                _metric_text(
                    float(row["RMSE_mean"]),
                    float(row["RMSE_std"]),
                    4,
                    abs(float(row["RMSE_mean"]) - best[(item["name"], "RMSE")]) < 5e-6,
                )
            )
            cells.append(
                _metric_text(
                    float(row["NMSE_dB_mean"]),
                    float(row["NMSE_dB_std"]),
                    2,
                    abs(float(row["NMSE_dB_mean"]) - best[(item["name"], "NMSE")]) < 5e-4,
                )
            )
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular*}"])

    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    frame = _read_rows()
    summary = _summary(frame)
    summary.to_csv(SUMMARY_PATH, index=False)
    _write_table(summary)
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {TABLE_PATH}")


if __name__ == "__main__":
    main()
