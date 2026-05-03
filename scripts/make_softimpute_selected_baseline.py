from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "neurips" / "tables"
FIGURE_DIR = PROJECT_ROOT / "neurips" / "figures"
INPUT = TABLE_DIR / "lowrank_core_cave_softimpute_selected.paired.csv"
OUT_PREFIX = TABLE_DIR / "softimpute_selected_baseline"

METHOD_LABELS = {
    "ntdpl": "NTD-PL",
    "softimpute": "SoftImpute",
}
COLORS = {
    "ntdpl": "#1E5A8A",
    "softimpute": "#C97C1A",
    "grid": "#D9DEE4",
    "border": "#24313C",
}


def _selected_rows(frame: pd.DataFrame) -> pd.DataFrame:
    complete = frame.loc[frame["n_pairs"].eq(45)].copy()
    selected = complete.loc[
        complete["protocol"].eq("random")
        | (complete["protocol"].eq("block") & complete["rank"].eq("(12,12,3)"))
    ].copy()
    selected["protocol_order"] = selected["protocol"].map({"random": 0, "block": 1})
    selected["rank_order"] = selected["rank"].map({"(12,12,3)": 0, "(24,24,4)": 1})
    selected["method_order"] = selected["method"].map({"ntdpl": 0, "softimpute": 1})
    return selected.sort_values(["protocol_order", "rank_order", "missing_rate", "method_order"]).drop(
        columns=["protocol_order", "rank_order", "method_order"]
    )


def _write_latex(frame: pd.DataFrame, output: Path) -> None:
    lines = [
        r"\begin{tabular}{@{}l c c l c c c@{}}",
        r"\toprule",
        r"Mask & Rank & $\rho$ & Method & RMSE gain & SAM gain & RMSE wins\\",
        r"\midrule",
    ]
    for row in frame.to_dict("records"):
        lines.append(
            " & ".join(
                [
                    str(row["protocol"]),
                    str(row["rank"]),
                    f"{float(row['missing_rate']):.1f}",
                    METHOD_LABELS.get(str(row["method"]), str(row["method"])),
                    f"{float(row['RMSE_missing_gain_median_pct']):.1f}\\%",
                    f"{float(row['SAM_missing_gain_median_pct']):.1f}\\%",
                    f"{int(row['RMSE_missing_wins'])}/{int(row['n_pairs'])}",
                ]
            )
            + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    output.write_text("\n".join(lines), encoding="utf-8")


def _plot(frame: pd.DataFrame, output_stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    panels = [
        ("random", "(12,12,3)", "A. Random, rank (12,12,3)"),
        ("random", "(24,24,4)", "B. Random, rank (24,24,4)"),
        ("block", "(12,12,3)", "C. Block, rank (12,12,3)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35), constrained_layout=True, sharey=True)
    for ax, (protocol, rank, title) in zip(axes, panels, strict=True):
        panel = frame.loc[frame["protocol"].eq(protocol) & frame["rank"].eq(rank)].copy()
        for method in ("ntdpl", "softimpute"):
            sub = panel.loc[panel["method"].eq(method)].sort_values("missing_rate")
            ax.plot(
                sub["missing_rate"],
                sub["RMSE_missing_gain_median_pct"],
                marker="o",
                linewidth=1.8,
                markersize=4.2,
                color=COLORS[method],
                label=METHOD_LABELS[method],
            )
        ax.axhline(0.0, color=COLORS["border"], linewidth=0.8)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel(r"Missing rate $\rho$")
        ax.set_ylabel("Median RMSE gain (%)" if ax is axes[0] else "")
        ax.set_xticks([0.3, 0.5, 0.7])
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=2, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.savefig(FIGURE_DIR / f"{output_stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{output_stem}.png", dpi=240, bbox_inches="tight")


def main() -> None:
    frame = pd.read_csv(INPUT)
    selected = _selected_rows(frame)
    OUT_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(OUT_PREFIX.with_suffix(".csv"), index=False)
    _write_latex(selected, OUT_PREFIX.with_suffix(".tex"))
    _plot(selected, OUT_PREFIX.name)
    print(f"Wrote {OUT_PREFIX.with_suffix('.csv')}, .tex, and figure outputs")


if __name__ == "__main__":
    main()
