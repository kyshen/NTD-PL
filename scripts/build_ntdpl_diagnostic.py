from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _fmt_pct(value: float) -> str:
    return f"{100.0 * value:.2f}"


def build_diagnostic(input_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(input_csv)
    required = {
        "missing_rate",
        "scene_id",
        "RMSE_tucker",
        "RMSE_polycal",
        "RMSE_ntdpl",
        "polycal_gain_rmse",
        "joint_extra_gain_rmse",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{input_csv} is missing required columns: {sorted(missing)}")

    rows: list[dict[str, float | int | str]] = []
    bin_rows: list[dict[str, float | int | str]] = []
    for missing_rate, panel in frame.groupby("missing_rate", sort=True):
        panel = panel.copy()
        panel["polycal_score"] = panel["polycal_gain_rmse"] / panel["RMSE_tucker"].clip(lower=1e-12)
        panel["ntdpl_gain"] = (panel["RMSE_tucker"] - panel["RMSE_ntdpl"]) / panel["RMSE_tucker"].clip(lower=1e-12)
        panel["joint_extra_gain"] = panel["joint_extra_gain_rmse"] / panel["RMSE_tucker"].clip(lower=1e-12)

        for target in ("ntdpl_gain", "joint_extra_gain"):
            pearson = pearsonr(panel["polycal_score"], panel[target])
            spearman = spearmanr(panel["polycal_score"], panel[target])
            rows.append(
                {
                    "missing_rate": float(missing_rate),
                    "target": target,
                    "pearson_r": float(pearson.statistic),
                    "pearson_p": float(pearson.pvalue),
                    "spearman_r": float(spearman.statistic),
                    "spearman_p": float(spearman.pvalue),
                    "n_scenes": int(panel["scene_id"].nunique()),
                }
            )

        panel["score_tertile"] = pd.qcut(panel["polycal_score"], 3, labels=("low", "mid", "high"))
        for tertile, subset in panel.groupby("score_tertile", observed=True):
            bin_rows.append(
                {
                    "missing_rate": float(missing_rate),
                    "score_tertile": str(tertile),
                    "polycal_score_mean": float(subset["polycal_score"].mean()),
                    "ntdpl_gain_mean": float(subset["ntdpl_gain"].mean()),
                    "joint_extra_gain_mean": float(subset["joint_extra_gain"].mean()),
                    "n_scenes": int(subset["scene_id"].nunique()),
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(bin_rows)


def write_latex(summary: pd.DataFrame, bins: pd.DataFrame, output_tex: Path) -> None:
    lines = [
        r"\begin{tabular}{@{}c c c c c c@{}}",
        r"\toprule",
        r"$\rho$ & Spearman $\rho_s$ & Low-score gain & Mid-score gain & High-score gain & Scenes\\",
        r"\midrule",
    ]
    target_summary = summary.loc[summary["target"].eq("ntdpl_gain")].copy()
    for row in target_summary.itertuples(index=False):
        panel = bins.loc[bins["missing_rate"].eq(row.missing_rate)].set_index("score_tertile")
        lines.append(
            f"{row.missing_rate:.1f} & "
            f"{row.spearman_r:.3f} & "
            f"{_fmt_pct(float(panel.loc['low', 'ntdpl_gain_mean']))}\\% & "
            f"{_fmt_pct(float(panel.loc['mid', 'ntdpl_gain_mean']))}\\% & "
            f"{_fmt_pct(float(panel.loc['high', 'ntdpl_gain_mean']))}\\% & "
            f"{int(row.n_scenes)}\\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    output_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a cheap diagnostic table for NTD-PL gains.")
    parser.add_argument(
        "--input-csv",
        default="experiment/outputs/cave-random-completion/polycal_pairwise_scene_gains.csv",
    )
    parser.add_argument("--out-prefix", default="neurips/tables/ntdpl_polycal_diagnostic")
    args = parser.parse_args()

    summary, bins = build_diagnostic(PROJECT_ROOT / args.input_csv)
    out_prefix = PROJECT_ROOT / args.out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_prefix.with_suffix(".summary.csv"), index=False)
    bins.to_csv(out_prefix.with_suffix(".tertiles.csv"), index=False)
    write_latex(summary, bins, out_prefix.with_suffix(".tex"))
    print(f"Wrote {out_prefix}.summary.csv, .tertiles.csv, and .tex")


if __name__ == "__main__":
    main()
