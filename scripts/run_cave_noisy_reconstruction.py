from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import sys
from typing import Any

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

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_cave_joint_link_ablation import _fit_ntdpl, _fit_tucker, _load_cave_scene, _metrics


def _parse_scene_ids(text: str) -> list[int]:
    if "-" in text:
        lo, hi = [int(part.strip()) for part in text.split("-", 1)]
        return list(range(lo, hi + 1))
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _parse_int_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in text.replace("x", ",").split(",") if part.strip())


def _add_noise(clean: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    signal_power = float(np.mean(np.asarray(clean, dtype=np.float32) ** 2))
    noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=clean.shape).astype(np.float32)
    return (np.asarray(clean, dtype=np.float32) + noise).astype(np.float32)


def _format_mean_std(mean: float, std: float, digits: int) -> str:
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def _snr_label(snr: float | None) -> str:
    return "Clean" if snr is None else f"{int(snr)} dB"


def _run_one(
    scene_id: int,
    clean: np.ndarray,
    scene_name: str,
    snr: float | None,
    target_shape: tuple[int, int],
    rank: tuple[int, int, int],
    n_iter_max: int,
) -> list[dict[str, Any]]:
    observed = clean if snr is None else _add_noise(clean, snr, seed=1000 + 37 * scene_id + int(snr))
    rows: list[dict[str, Any]] = []
    for method, fit_fn in (
        ("Tucker", lambda x: _fit_tucker(x, rank, n_iter_max)),
        ("NTD-PL", lambda x: _fit_ntdpl(x, rank, n_iter_max)),
    ):
        recon, fit_time, params = fit_fn(observed)
        rows.append(
            {
                "scene_id": scene_id,
                "scene_name": scene_name,
                "target_shape": str(target_shape),
                "rank": str(rank),
                "snr": _snr_label(snr),
                "snr_value": 999.0 if snr is None else float(snr),
                "method": method,
                "params": int(params),
                "fit_time_sec": float(fit_time),
                **_metrics(clean, recon),
            }
        )
    return rows


def _run_scene(
    scene_id: int,
    target_shape: tuple[int, int],
    rank: tuple[int, int, int],
    snrs: list[float | None],
    n_iter_max: int,
) -> list[dict[str, Any]]:
    scene_name, clean = _load_cave_scene(scene_id, target_shape)
    rows: list[dict[str, Any]] = []
    for snr in snrs:
        rows.extend(_run_one(scene_id, clean, scene_name, snr, target_shape, rank, n_iter_max))
    return rows


def _summarize(frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frame.groupby(["snr", "snr_value", "method"], as_index=False)
        .agg(
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            SAM_mean=("SAM", "mean"),
            SAM_std=("SAM", "std"),
            n_scenes=("scene_id", "nunique"),
        )
        .sort_values(["snr_value", "method"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return summary


def _to_latex(summary: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{@{}lcccccc@{}}",
        r"\toprule",
        r"\multirow{2}{*}{SNR} &",
        r"\multicolumn{3}{c}{Clean RMSE$\downarrow$} &",
        r"\multicolumn{3}{c}{Clean SAM$\downarrow$} \\",
        r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
        r"& Tucker & NTD-PL & Rel. gain &",
        r"Tucker & NTD-PL & Rel. gain \\",
        r"\midrule",
    ]
    for snr, group in summary.groupby("snr", sort=False):
        lookup = {row.method: row for row in group.itertuples(index=False)}
        if "Tucker" not in lookup or "NTD-PL" not in lookup:
            continue
        t = lookup["Tucker"]
        n = lookup["NTD-PL"]
        rmse_gain = 100.0 * (float(t.RMSE_mean) - float(n.RMSE_mean)) / float(t.RMSE_mean)
        sam_gain = 100.0 * (float(t.SAM_mean) - float(n.SAM_mean)) / float(t.SAM_mean)
        lines.append(
            f"{snr} & "
            f"{_format_mean_std(float(t.RMSE_mean), float(t.RMSE_std), 4)} & "
            f"\\textbf{{{_format_mean_std(float(n.RMSE_mean), float(n.RMSE_std), 4)}}} & "
            f"{rmse_gain:.2f}\\%\n& "
            f"{_format_mean_std(float(t.SAM_mean), float(t.SAM_std), 2)} & "
            f"\\textbf{{{_format_mean_std(float(n.SAM_mean), float(n.SAM_std), 2)}}} & "
            f"{sam_gain:.2f}\\% \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CAVE noisy full-reconstruction robustness.")
    parser.add_argument("--scene-ids", default="1-15")
    parser.add_argument("--target-shape", default="512,512")
    parser.add_argument("--rank", default="12,12,6")
    parser.add_argument("--snr", default="clean,40,30,20,10")
    parser.add_argument("--n-iter-max", type=int, default=100)
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--out-prefix", default="papers/tsp-supplementary/tables/cave_noise_robustness")
    args = parser.parse_args()

    scene_ids = _parse_scene_ids(args.scene_ids)
    target_shape_raw = _parse_int_tuple(args.target_shape)
    rank_raw = _parse_int_tuple(args.rank)
    if len(target_shape_raw) != 2 or len(rank_raw) != 3:
        raise ValueError("Expected --target-shape with two entries and --rank with three entries.")
    target_shape = (target_shape_raw[0], target_shape_raw[1])
    rank = (rank_raw[0], rank_raw[1], rank_raw[2])
    snrs: list[float | None] = []
    for item in args.snr.split(","):
        item = item.strip().lower()
        snrs.append(None if item == "clean" else float(item))

    out_prefix = PROJECT_ROOT / args.out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    partial_path = out_prefix.with_suffix(".partial.csv")
    rows: list[dict[str, Any]] = []
    workers = max(1, min(int(args.workers), len(scene_ids)))
    print(
        f"Running {len(scene_ids)} full-size CAVE scenes at {target_shape} "
        f"with {workers} workers.",
        flush=True,
    )
    if workers == 1:
        for scene_id in scene_ids:
            rows.extend(_run_scene(scene_id, target_shape, rank, snrs, args.n_iter_max))
            pd.DataFrame(rows).to_csv(partial_path, index=False)
            print(f"Finished scene {scene_id:02d}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_run_scene, scene_id, target_shape, rank, snrs, args.n_iter_max): scene_id
                for scene_id in scene_ids
            }
            for future in as_completed(futures):
                scene_id = futures[future]
                rows.extend(future.result())
                pd.DataFrame(rows).to_csv(partial_path, index=False)
                print(f"Finished scene {scene_id:02d} ({len(rows) // (2 * len(snrs))}/{len(scene_ids)})", flush=True)

    method_order = {"Tucker": 0, "NTD-PL": 1}
    frame = pd.DataFrame(rows)
    frame["method_order"] = frame["method"].map(method_order)
    frame = frame.sort_values(
        ["scene_id", "snr_value", "method_order"],
        ascending=[True, False, True],
    ).drop(columns="method_order").reset_index(drop=True)
    summary = _summarize(frame)
    frame.to_csv(out_prefix.with_suffix(".per_scene.csv"), index=False)
    summary.to_csv(out_prefix.with_suffix(".summary.csv"), index=False)
    out_prefix.with_suffix(".tex").write_text(_to_latex(summary), encoding="utf-8")
    partial_path.unlink(missing_ok=True)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
