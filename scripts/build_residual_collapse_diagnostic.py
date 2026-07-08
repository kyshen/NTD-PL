from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment.process.helpers.cave_random_completion import load_main_runs
from experiment.process.helpers.cave_random_completion_polycal import load_scene_original
from experiment.utils.io import load_state_mat


TARGET_MISSING_RATES = (0.3, 0.5)


def _jsonish(value):
    out = value
    while isinstance(out, str):
        text = out.strip()
        if not text:
            return text
        try:
            loaded = __import__("json").loads(text)
        except __import__("json").JSONDecodeError:
            return out
        if loaded == out:
            return loaded
        out = loaded
    return out


def _resolve_state_path(path_text: object) -> Path | None:
    raw = Path(str(_jsonish(path_text)).strip().strip('"'))
    if raw.is_absolute() and raw.exists():
        return raw
    candidate = PROJECT_ROOT / raw
    if candidate.exists():
        return candidate
    return None


def _observed_sample(
    recon_tucker: np.ndarray,
    original: np.ndarray,
    observed_mask: np.ndarray,
    *,
    sample_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(recon_tucker, dtype=np.float64)[observed_mask].reshape(-1)
    residual = (
        np.asarray(original, dtype=np.float64) - np.asarray(recon_tucker, dtype=np.float64)
    )[observed_mask].reshape(-1)
    if 0 < sample_size < x.size:
        rng = np.random.default_rng(seed)
        idx = rng.choice(x.size, size=int(sample_size), replace=False)
        x = x[idx]
        residual = residual[idx]
    return x, residual


def _collapse_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    *,
    n_bins: int,
) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, int(n_bins) + 1)
    edges = np.quantile(x_train, quantiles)
    edges[0] = -np.inf
    edges[-1] = np.inf

    train_bins = np.digitize(x_train, edges[1:-1], right=False)
    val_bins = np.digitize(x_val, edges[1:-1], right=False)

    bin_means = np.full(int(n_bins), np.mean(y_train), dtype=np.float64)
    for idx in range(int(n_bins)):
        mask = train_bins == idx
        if np.any(mask):
            bin_means[idx] = float(np.mean(y_train[mask]))
    return bin_means[val_bins]


def _cv_residual_collapse_score(
    x: np.ndarray,
    residual: np.ndarray,
    *,
    n_folds: int,
    n_bins: int,
    seed: int,
) -> tuple[float, float]:
    if x.size < max(256, n_folds * n_bins * 4):
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    perm = rng.permutation(x.size)
    x = x[perm]
    residual = residual[perm]
    fold_ids = np.arange(x.size) % int(n_folds)

    scores: list[float] = []
    baseline_scores: list[float] = []
    for fold in range(int(n_folds)):
        val_mask = fold_ids == fold
        train_mask = ~val_mask
        x_train = x[train_mask]
        y_train = residual[train_mask]
        x_val = x[val_mask]
        y_val = residual[val_mask]
        if x_train.size < 64 or x_val.size < 32:
            continue

        pred = _collapse_predict(x_train, y_train, x_val, n_bins=n_bins)
        mse_model = float(np.mean((y_val - pred) ** 2))
        mse_zero = float(np.mean(y_val**2))
        var_val = float(np.var(y_val))
        denom = max(var_val, 1e-12)
        scores.append(1.0 - mse_model / denom)
        baseline_scores.append(1.0 - mse_zero / denom)

    if not scores:
        return np.nan, np.nan
    return float(np.mean(scores)), float(np.mean(baseline_scores))


def _pair_scene_gain(scene_mean: pd.DataFrame) -> pd.DataFrame:
    pivot = scene_mean.pivot_table(
        index=["missing_rate", "scene_id"],
        columns="method_name",
        values="RMSE_missing",
        aggfunc="mean",
    ).reset_index()
    pivot["ntdpl_gain"] = (pivot["tucker"] - pivot["ntdpl"]) / pivot["tucker"].clip(lower=1e-12)
    return pivot.loc[:, ["missing_rate", "scene_id", "ntdpl_gain"]]


def build(
    *,
    sample_size: int,
    n_folds: int,
    n_bins: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame, _ = load_main_runs()
    frame = frame.loc[
        frame["method_name"].isin(["tucker", "ntdpl"])
        & frame["missing_rate"].isin(TARGET_MISSING_RATES)
    ].copy()

    run_rows: list[dict[str, float | int | str]] = []
    for row in frame.loc[frame["method_name"].eq("tucker")].to_dict("records"):
        scene_id = int(row["scene_id"])
        mask_seed = int(row["mask_seed"])
        missing_rate = float(row["missing_rate"])
        state_path = _resolve_state_path(row["state_path"])
        if state_path is None:
            continue
        state = load_state_mat(state_path)
        recon_tucker = np.asarray(_jsonish(state["reconstruction"]), dtype=np.float32)
        observed_mask = np.asarray(_jsonish(state["observed_mask"]), dtype=bool)
        scene_name, original = load_scene_original(scene_id, **{
            "path": str(_jsonish(row.get("data.path") or row.get("ovr.data.path") or "data/CAVE")),
            "target_shape": tuple(_jsonish(row.get("data.target_shape") or row.get("ovr.data.target_shape") or [512, 512])),
            "crop_shape": _jsonish(row.get("data.crop_shape") or row.get("ovr.data.crop_shape")),
        })
        x, residual = _observed_sample(
            recon_tucker,
            original,
            observed_mask,
            sample_size=sample_size,
            seed=1000 * scene_id + mask_seed,
        )
        collapse_score, zero_baseline_r2 = _cv_residual_collapse_score(
            x,
            residual,
            n_folds=n_folds,
            n_bins=n_bins,
            seed=17 * scene_id + mask_seed,
        )
        run_rows.append(
            {
                "missing_rate": missing_rate,
                "scene_id": scene_id,
                "scene_name": scene_name,
                "mask_seed": mask_seed,
                "residual_collapse_score": collapse_score,
                "zero_baseline_r2": zero_baseline_r2,
                "sample_count": int(x.size),
            }
        )

    run_frame = pd.DataFrame(run_rows).sort_values(["missing_rate", "scene_id", "mask_seed"]).reset_index(drop=True)
    if run_frame.empty:
        raise RuntimeError("No valid Tucker completion states were found for residual collapse diagnostic.")

    scene_score = (
        run_frame.groupby(["missing_rate", "scene_id", "scene_name"], as_index=False)[
            ["residual_collapse_score", "zero_baseline_r2"]
        ]
        .mean()
        .sort_values(["missing_rate", "scene_id"])
        .reset_index(drop=True)
    )

    scene_mean = (
        frame.groupby(["missing_rate", "method_name", "scene_id"], as_index=False)[["RMSE_missing"]]
        .mean()
        .sort_values(["missing_rate", "method_name", "scene_id"])
    )
    gains = _pair_scene_gain(scene_mean)
    merged = scene_score.merge(gains, on=["missing_rate", "scene_id"], how="left")

    summary_rows: list[dict[str, float | int]] = []
    for missing_rate, panel in merged.groupby("missing_rate", sort=True):
        score = panel["residual_collapse_score"].to_numpy(dtype=float)
        gain = panel["ntdpl_gain"].to_numpy(dtype=float)
        pearson = pearsonr(score, gain)
        spearman = spearmanr(score, gain)
        tertiles = pd.qcut(panel["residual_collapse_score"], 3, labels=("low", "mid", "high"))
        grouped = panel.assign(score_tertile=tertiles).groupby("score_tertile", observed=True)["ntdpl_gain"].mean()
        summary_rows.append(
            {
                "missing_rate": float(missing_rate),
                "pearson_r": float(pearson.statistic),
                "pearson_p": float(pearson.pvalue),
                "spearman_r": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue),
                "low_gain_pct": 100.0 * float(grouped.get("low", np.nan)),
                "mid_gain_pct": 100.0 * float(grouped.get("mid", np.nan)),
                "high_gain_pct": 100.0 * float(grouped.get("high", np.nan)),
                "n_scenes": int(panel["scene_id"].nunique()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("missing_rate").reset_index(drop=True)
    return run_frame, merged, summary


def write_latex(summary: pd.DataFrame, output_tex: Path) -> None:
    lines = [
        r"\begin{tabular}{@{}c c c c c c@{}}",
        r"\toprule",
        r"$\rho$ & Spearman $\rho_s$ & Low-score gain & Mid-score gain & High-score gain & Scenes\\",
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"{row.missing_rate:.1f} & "
            f"{row.spearman_r:.3f} & "
            f"{row.low_gain_pct:.2f}\\% & "
            f"{row.mid_gain_pct:.2f}\\% & "
            f"{row.high_gain_pct:.2f}\\% & "
            f"{int(row.n_scenes)}\\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    output_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a residual collapse diagnostic for CAVE completion."
    )
    parser.add_argument("--sample-size", type=int, default=120_000)
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--n-bins", type=int, default=32)
    parser.add_argument("--out-prefix", default="papers/neurips/tables/residual_collapse_diagnostic")
    args = parser.parse_args()

    run_frame, merged, summary = build(
        sample_size=args.sample_size,
        n_folds=args.n_folds,
        n_bins=args.n_bins,
    )
    out_prefix = PROJECT_ROOT / args.out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    run_frame.to_csv(out_prefix.with_suffix(".runs.csv"), index=False)
    merged.to_csv(out_prefix.with_suffix(".scene.csv"), index=False)
    summary.to_csv(out_prefix.with_suffix(".summary.csv"), index=False)
    write_latex(summary, out_prefix.with_suffix(".tex"))
    print(f"Wrote {out_prefix}.runs.csv, .scene.csv, .summary.csv, and .tex")


if __name__ == "__main__":
    main()
