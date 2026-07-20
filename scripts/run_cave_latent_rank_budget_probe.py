from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _thread_key in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "TBB_NUM_THREADS",
):
    os.environ.setdefault(_thread_key, "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_cave_rank_inflation_spectrum import _direct_scene_rows


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTDIR = PROJECT_ROOT / "artifacts" / "results" / "cave_latent_rank_budget_probe"
DEFAULT_R12 = PROJECT_ROOT / "artifacts" / "results" / "rank_inflation_spectrum_r12_p6_512"
THRESHOLDS = (
    ("rank_energy_950", "95%"),
    ("rank_energy_990", "99%"),
    ("rank_energy_995", "99.5%"),
    ("rank_energy_999", "99.9%"),
)


def _parse_ints(text: str) -> list[int]:
    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(item.strip()) for item in part.split("-", 1)]
            values.extend(range(start, end + 1))
        else:
            values.append(int(part))
    return sorted(set(values))


def _parse_shape(text: str) -> tuple[int, int]:
    values = [int(part.strip()) for part in text.replace("x", ",").split(",") if part.strip()]
    if len(values) != 2:
        raise ValueError("Expected --target-shape with two dimensions.")
    return values[0], values[1]


def _run_job(
    scene_id: int,
    spatial_rank: int,
    spectral_rank: int,
    target_shape: tuple[int, int],
    n_iter_max: int,
    p_max: int,
) -> tuple[int, int, list[dict], list[dict]]:
    rank = (spatial_rank, spatial_rank, spectral_rank)
    spectrum, fit = _direct_scene_rows(scene_id, target_shape, rank, n_iter_max, p_max)
    return scene_id, spatial_rank, spectrum, fit


def _load_reused_rank12(root: Path, scene_ids: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    spectrum = pd.read_csv(root / "spectrum_metrics.csv")
    fit = pd.read_csv(root / "fit_metrics.csv")
    spectrum = spectrum[spectrum["scene_id"].isin(scene_ids)].copy()
    fit = fit[fit["scene_id"].isin(scene_ids)].copy()
    return spectrum, fit


def _spatial_budget(rank_text: str) -> int:
    return int(str(rank_text).strip("()[]").split(",")[0].strip())


def _summarize_spectra(metrics: pd.DataFrame) -> pd.DataFrame:
    frame = metrics[
        metrics["mode"].isin([1, 2])
        & metrics["tensor_kind"].isin(
            [
                "ntdpl_signal",
                "ntdpl_nonlinear_response_component",
                "ntdpl_prediction",
                "tucker_fit",
                "measured",
            ]
        )
    ].copy()
    frame["spatial_rank_budget"] = frame["rank"].map(_spatial_budget)
    rows: list[dict] = []
    for (budget, kind), group in frame.groupby(["spatial_rank_budget", "tensor_kind"]):
        for metric, label in THRESHOLDS:
            values = group[metric].to_numpy(dtype=float)
            rows.append(
                {
                    "spatial_rank_budget": int(budget),
                    "tensor_kind": kind,
                    "energy_threshold": label,
                    "median_rank": float(np.median(values)),
                    "q25_rank": float(np.quantile(values, 0.25)),
                    "q75_rank": float(np.quantile(values, 0.75)),
                    "mean_rank": float(np.mean(values)),
                    "ceiling_fraction": float(np.mean(values >= float(budget))),
                    "samples": int(values.size),
                }
            )
    return pd.DataFrame(rows).sort_values(["tensor_kind", "energy_threshold", "spatial_rank_budget"])


def _summarize_fit(fit: pd.DataFrame) -> pd.DataFrame:
    frame = fit.copy()
    frame["spatial_rank_budget"] = frame["rank"].map(_spatial_budget)
    summary = (
        frame.groupby(["spatial_rank_budget", "method"], as_index=False)
        .agg(RMSE_mean=("RMSE", "mean"), RMSE_median=("RMSE", "median"), SAM_mean=("SAM", "mean"))
    )
    wide = summary.pivot(index="spatial_rank_budget", columns="method", values="RMSE_mean").reset_index()
    wide["NTDPL_gain_pct"] = 100.0 * (wide["Tucker"] - wide["NTD-PL"]) / wide["Tucker"]
    return summary.merge(wide[["spatial_rank_budget", "NTDPL_gain_pct"]], on="spatial_rank_budget", how="left")


def _plot(summary: pd.DataFrame, outdir: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.25))
    signal = summary[summary["tensor_kind"].eq("ntdpl_signal")]
    colors = {"95%": "#777777", "99%": "#2A9D8F", "99.5%": "#D9822B", "99.9%": "#2F6FBB"}
    for label in ["95%", "99%", "99.5%", "99.9%"]:
        block = signal[signal["energy_threshold"].eq(label)].sort_values("spatial_rank_budget")
        x = block["spatial_rank_budget"].to_numpy(dtype=float)
        y = block["median_rank"].to_numpy(dtype=float)
        axes[0].plot(x, y, marker="o", label=label, color=colors[label])
        axes[0].fill_between(
            x,
            block["q25_rank"].to_numpy(dtype=float),
            block["q75_rank"].to_numpy(dtype=float),
            color=colors[label],
            alpha=0.10,
        )
    budgets = sorted(signal["spatial_rank_budget"].unique())
    axes[0].plot(budgets, budgets, color="#999999", linestyle="--", linewidth=1.0, label="rank budget")
    axes[0].set_title("Energy rank of NTD-PL Tucker signal")
    axes[0].set_xlabel("spatial Tucker rank budget")
    axes[0].set_ylabel("median spatial energy rank")
    axes[0].set_xticks(budgets)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(ncol=2)

    labels = {
        "ntdpl_signal": "NTD-PL signal",
        "ntdpl_prediction": "NTD-PL prediction",
        "tucker_fit": "Tucker fit",
    }
    colors2 = {"ntdpl_signal": "#2F6FBB", "ntdpl_prediction": "#D9822B", "tucker_fit": "#2A9D8F"}
    r999 = summary[summary["energy_threshold"].eq("99.9%")]
    for kind in ["ntdpl_signal", "ntdpl_prediction", "tucker_fit"]:
        block = r999[r999["tensor_kind"].eq(kind)].sort_values("spatial_rank_budget")
        axes[1].plot(
            block["spatial_rank_budget"],
            block["median_rank"],
            marker="o",
            color=colors2[kind],
            label=labels[kind],
        )
    axes[1].plot(budgets, budgets, color="#999999", linestyle="--", linewidth=1.0, label="rank budget")
    axes[1].set_title("99.9% energy rank")
    axes[1].set_xlabel("spatial Tucker rank budget")
    axes[1].set_ylabel("median spatial energy rank")
    axes[1].set_xticks(budgets)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(outdir / "latent_rank_budget_probe.pdf", bbox_inches="tight")
    fig.savefig(outdir / "latent_rank_budget_probe.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe whether the NTD-PL Tucker signal energy rank saturates as its rank budget grows.")
    parser.add_argument("--scene-ids", default="1,2,5,6,10")
    parser.add_argument("--spatial-ranks", default="16,24,32,40")
    parser.add_argument("--spectral-rank", type=int, default=4)
    parser.add_argument("--target-shape", default="512,512")
    parser.add_argument("--p-max", type=int, default=6)
    parser.add_argument("--n-iter-max", type=int, default=120)
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--reuse-r12", type=Path, default=DEFAULT_R12)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    scene_ids = _parse_ints(args.scene_ids)
    spatial_ranks = _parse_ints(args.spatial_ranks)
    target_shape = _parse_shape(args.target_shape)
    outdir = args.outdir if args.outdir.is_absolute() else PROJECT_ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    spectrum_rows: list[dict] = []
    fit_rows: list[dict] = []
    jobs = [(scene_id, rank) for rank in spatial_ranks for scene_id in scene_ids]
    with ProcessPoolExecutor(max_workers=max(1, int(args.max_workers))) as executor:
        futures = {
            executor.submit(
                _run_job,
                scene_id,
                rank,
                int(args.spectral_rank),
                target_shape,
                int(args.n_iter_max),
                int(args.p_max),
            ): (scene_id, rank)
            for scene_id, rank in jobs
        }
        for future in as_completed(futures):
            scene_id, rank = futures[future]
            _, _, spectrum, fit = future.result()
            spectrum_rows.extend(spectrum)
            fit_rows.extend(fit)
            print(f"completed scene={scene_id}, spatial_rank={rank}", flush=True)

    spectrum = pd.DataFrame(spectrum_rows)
    fit = pd.DataFrame(fit_rows)
    reuse_root = args.reuse_r12 if args.reuse_r12.is_absolute() else PROJECT_ROOT / args.reuse_r12
    reused_spectrum, reused_fit = _load_reused_rank12(reuse_root, scene_ids)
    spectrum = pd.concat([reused_spectrum, spectrum], ignore_index=True)
    fit = pd.concat([reused_fit, fit], ignore_index=True)
    spectrum = spectrum.sort_values(["rank", "scene_id", "mode", "tensor_kind"]).reset_index(drop=True)
    fit = fit.sort_values(["rank", "scene_id", "method"]).reset_index(drop=True)

    summary = _summarize_spectra(spectrum)
    fit_summary = _summarize_fit(fit)
    spectrum.to_csv(outdir / "spectrum_metrics.csv", index=False)
    fit.to_csv(outdir / "fit_metrics.csv", index=False)
    summary.to_csv(outdir / "energy_rank_summary.csv", index=False)
    fit_summary.to_csv(outdir / "fit_summary.csv", index=False)
    _plot(summary, outdir)
    print(summary[summary["tensor_kind"].eq("ntdpl_signal")].to_string(index=False))
    print(fit_summary.to_string(index=False))
    print(f"Output: {outdir}")


if __name__ == "__main__":
    main()
