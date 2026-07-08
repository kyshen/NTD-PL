from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TUCKER_SWEEP = ROOT / "papers/neurips/tables/cave_tucker_rank_sweep.summary.csv"
CAVE_RECON_MAIN = ROOT / "artifacts/paper-outputs/cave-representation/recon_summary.csv"
CAVE_RECON_LOW = ROOT / "papers/neurips/tables/cave_reconstruction_lowrank.summary.csv"
OUT_PREFIX = ROOT / "papers/neurips/tables/rank_lift_cost"


MATCHES = [
    {"ntdpl_rank": "(18,18,3)", "match_rank": "(28,28,3)"},
    {"ntdpl_rank": "(24,24,4)", "match_rank": "(35,35,4)"},
    {"ntdpl_rank": "(33,33,4)", "match_rank": "(49,49,4)"},
]


def _norm_rank(value: object) -> str:
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        text = "(" + text[1:-1] + ")"
    return text.replace(" ", "")


def _parse_mean(value: object) -> float:
    text = str(value)
    if "+-" in text:
        return float(text.split("+-", 1)[0].strip())
    return float(text)


def _load_ntdpl() -> pd.DataFrame:
    main = pd.read_csv(CAVE_RECON_MAIN)
    main = main.loc[main["Method"].eq("NTD-PL")].copy()
    main = main.rename(columns={"Rank": "rank", "Params": "params"})
    main["rank"] = main["rank"].map(_norm_rank)
    main["RMSE_mean"] = main["RMSE"].map(_parse_mean)
    main["SAM_mean"] = main["SAM(deg)"].map(_parse_mean)
    main["fit_time_mean"] = pd.NA

    runs = pd.read_parquet(ROOT / "artifacts/multirun/cave-representation/runs.parquet")
    runs = runs.loc[runs["method._name"].eq("ntdpl")].copy()
    runs["rank"] = runs["method.rank"].map(_norm_rank)
    times = runs.groupby("rank", as_index=False)["fit_time_sec"].mean().rename(columns={"fit_time_sec": "fit_time_mean"})
    main = main.drop(columns=["fit_time_mean"]).merge(times, on="rank", how="left")

    low = pd.read_csv(CAVE_RECON_LOW)
    low = low.loc[low["method"].eq("NTD-PL")].copy()
    low["rank"] = low["rank"].map(_norm_rank)
    low = low.rename(columns={"params": "params"})
    low = low[["rank", "params", "RMSE_mean", "SAM_mean", "fit_time_mean"]]

    merged = pd.concat(
        [
            main[["rank", "params", "RMSE_mean", "SAM_mean", "fit_time_mean"]],
            low,
        ],
        ignore_index=True,
    )
    return merged.drop_duplicates("rank", keep="last")


def _load_tucker() -> pd.DataFrame:
    frame = pd.read_csv(TUCKER_SWEEP)
    frame["rank"] = frame["rank"].map(_norm_rank)
    return frame[["rank", "params", "RMSE_mean", "SAM_mean", "fit_time_mean"]]


def _format_k(value: float) -> str:
    return f"{value / 1000.0:.0f}k"


def _format_sec(value: float) -> str:
    return f"{value:.1f}s"


def build() -> pd.DataFrame:
    ntdpl = _load_ntdpl().set_index("rank")
    tucker = _load_tucker().set_index("rank")
    rows = []
    for item in MATCHES:
        n_rank = item["ntdpl_rank"]
        t_rank = item["match_rank"]
        n = ntdpl.loc[n_rank]
        t = tucker.loc[t_rank]
        rows.append(
            {
                "ntdpl_rank": n_rank,
                "ntdpl_params": int(n["params"]),
                "ntdpl_time": float(n["fit_time_mean"]),
                "ntdpl_rmse": float(n["RMSE_mean"]),
                "ntdpl_sam": float(n["SAM_mean"]),
                "tucker_rank": t_rank,
                "tucker_params": int(t["params"]),
                "tucker_time": float(t["fit_time_mean"]),
                "tucker_rmse": float(t["RMSE_mean"]),
                "tucker_sam": float(t["SAM_mean"]),
                "extra_params_pct": 100.0 * (float(t["params"]) - float(n["params"])) / float(n["params"]),
                "time_ratio": float(n["fit_time_mean"]) / max(float(t["fit_time_mean"]), 1e-12),
            }
        )
    return pd.DataFrame(rows)


def to_latex(frame: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{@{}l r r l r r c c@{}}",
        r"\toprule",
        r"\multicolumn{3}{c}{NTD-PL} & \multicolumn{3}{c}{RMSE-matching Tucker} & Extra params & Time ratio \\",
        r"\cmidrule(lr){1-3}\cmidrule(lr){4-6}\cmidrule(lr){7-8}",
        r"Rank & Params & Time & Rank & Params & Time & & \\",
        r"\midrule",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"{row.ntdpl_rank} & {_format_k(row.ntdpl_params)} & {_format_sec(row.ntdpl_time)} & "
            f"{row.tucker_rank} & {_format_k(row.tucker_params)} & {_format_sec(row.tucker_time)} & "
            f"{row.extra_params_pct:.1f}\\% & {row.time_ratio:.1f}$\\times$\\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    frame = build()
    OUT_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT_PREFIX.with_suffix(".csv"), index=False)
    OUT_PREFIX.with_suffix(".tex").write_text(to_latex(frame), encoding="utf-8")
    print(f"Wrote {OUT_PREFIX}.csv and .tex")


if __name__ == "__main__":
    main()
