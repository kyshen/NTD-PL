from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment.process.helpers.cave_random_completion import _cave_dataset_kwargs_from_row, _resolve_state_path
from experiment.process.helpers.cave_random_completion_polycal import (
    MLPCAL_BATCH_SIZE,
    MLPCAL_HIDDEN_UNITS,
    MLPCAL_LAMBDA,
    MLPCAL_LR,
    MLPCAL_MAX_ITER,
    MLPCAL_MAX_TRAIN_SAMPLES,
    POLYCAL_LAMBDA,
    _completion_metrics,
    load_scene_original,
    load_target_runs,
    pm_latex,
)
from experiment.utils.io import load_state_mat
from src.methods.mlpcal import ScalarMLPCalibration
from src.methods.polycal import PolynomialCalibration


METHOD_ORDER = ["tucker", "polycal", "mlpcal", "ntdpl"]
METHOD_LABELS = {
    "tucker": "Tucker",
    "polycal": "Tucker + PolyCal(P=4)",
    "mlpcal": f"Tucker + MLPCal(H={MLPCAL_HIDDEN_UNITS})",
    "ntdpl": "NTD-PL",
}


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


def _parse_scene_ids(text: str) -> list[int]:
    if "-" in text:
        start, end = [int(part.strip()) for part in text.split("-", 1)]
        return list(range(start, end + 1))
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect a lightweight MLPCal mechanism subset.")
    parser.add_argument("--scene-ids", default="1-15")
    parser.add_argument("--mask-seed", type=int, default=0)
    parser.add_argument("--missing-rate", type=float, default=0.5)
    parser.add_argument("--max-iter", type=int, default=MLPCAL_MAX_ITER)
    parser.add_argument("--max-train-samples", type=int, default=MLPCAL_MAX_TRAIN_SAMPLES)
    parser.add_argument("--out-prefix", default="neurips/tables/mlpcal_subset_seed0_mr05")
    args = parser.parse_args()

    scene_ids = _parse_scene_ids(args.scene_ids)
    frame, _ = load_target_runs()
    rows: list[dict[str, Any]] = []

    for scene_id in scene_ids:
        tucker_row = _select_row(frame, scene_id, args.mask_seed, args.missing_rate, "tucker")
        ntdpl_row = _select_row(frame, scene_id, args.mask_seed, args.missing_rate, "ntdpl")
        scene_name, original = load_scene_original(scene_id, **_cave_dataset_kwargs_from_row(tucker_row))

        tucker_state = load_state_mat(_resolve_state_path(tucker_row["state_path"]))
        recon_tucker = np.asarray(_jsonish(tucker_state["reconstruction"]), dtype=np.float32)
        observed_mask = np.asarray(_jsonish(tucker_state["observed_mask"]), dtype=bool)

        rows.append(_row_from_existing(scene_id, scene_name, "tucker", tucker_row))
        rows.append(_row_from_existing(scene_id, scene_name, "ntdpl", ntdpl_row))

        poly = PolynomialCalibration(degree=4, lambda_reg=POLYCAL_LAMBDA).fit(
            recon_tucker,
            original,
            observed_mask,
        )
        recon_poly = poly.apply(recon_tucker)
        rows.append(
            _row_from_metrics(
                scene_id,
                scene_name,
                "polycal",
                _completion_metrics(original, recon_poly, observed_mask),
                float(poly.diagnostics.fit_time_sec) if poly.diagnostics is not None else np.nan,
            )
        )

        mlp = ScalarMLPCalibration(
            hidden_units=MLPCAL_HIDDEN_UNITS,
            lambda_reg=MLPCAL_LAMBDA,
            lr=MLPCAL_LR,
            max_iter=args.max_iter,
            batch_size=MLPCAL_BATCH_SIZE,
            max_train_samples=args.max_train_samples,
            random_state=args.mask_seed,
        ).fit(recon_tucker, original, observed_mask)
        recon_mlp = mlp.apply(recon_tucker)
        rows.append(
            _row_from_metrics(
                scene_id,
                scene_name,
                "mlpcal",
                _completion_metrics(original, recon_mlp, observed_mask),
                float(mlp.diagnostics.fit_time_sec) if mlp.diagnostics is not None else np.nan,
            )
        )
        print(f"Finished scene {scene_id:02d} ({scene_name})")

    result = pd.DataFrame(rows)
    result["order"] = result["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    result = result.sort_values(["scene_id", "order"]).drop(columns="order").reset_index(drop=True)

    summary = (
        result.groupby("method", as_index=False)
        .agg(
            RMSE_missing_mean=("RMSE_missing", "mean"),
            RMSE_missing_std=("RMSE_missing", "std"),
            SAM_missing_mean=("SAM_missing", "mean"),
            SAM_missing_std=("SAM_missing", "std"),
            NMSE_dB_all_mean=("NMSE_dB_all", "mean"),
            NMSE_dB_all_std=("NMSE_dB_all", "std"),
            fit_time_sec_mean=("fit_time_sec", "mean"),
            fit_time_sec_std=("fit_time_sec", "std"),
            n_scenes=("scene_id", "nunique"),
        )
    )
    summary["order"] = summary["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    summary = summary.sort_values("order").drop(columns="order").reset_index(drop=True)

    out_prefix = PROJECT_ROOT / args.out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_prefix.with_suffix(".per_scene.csv"), index=False)
    summary.to_csv(out_prefix.with_suffix(".summary.csv"), index=False)
    out_prefix.with_suffix(".tex").write_text(_summary_to_latex(summary), encoding="utf-8")
    print(f"Saved {out_prefix.with_suffix('.per_scene.csv')}")
    print(f"Saved {out_prefix.with_suffix('.summary.csv')}")
    print(f"Saved {out_prefix.with_suffix('.tex')}")
    print(summary.to_string(index=False))


def _select_row(
    frame: pd.DataFrame,
    scene_id: int,
    mask_seed: int,
    missing_rate: float,
    method_name: str,
) -> pd.Series:
    panel = frame.loc[
        frame["method_name"].eq(method_name)
        & frame["scene_id"].eq(scene_id)
        & frame["mask_seed"].eq(mask_seed)
        & np.isclose(frame["missing_rate"], missing_rate, atol=1e-12)
    ].copy()
    if panel.empty:
        raise ValueError(
            f"No row for method={method_name}, scene_id={scene_id}, "
            f"mask_seed={mask_seed}, missing_rate={missing_rate}."
        )
    return panel.iloc[0]


def _row_from_existing(scene_id: int, scene_name: str, method: str, row: pd.Series) -> dict[str, Any]:
    return {
        "scene_id": scene_id,
        "scene_name": scene_name,
        "method": method,
        "method_label": METHOD_LABELS[method],
        "RMSE_missing": float(row["RMSE_missing"]),
        "SAM_missing": float(row["SAM_missing"]),
        "NMSE_dB_all": float(row["NMSE_dB_all"]),
        "fit_time_sec": float(row["fit_time_sec"]),
    }


def _row_from_metrics(
    scene_id: int,
    scene_name: str,
    method: str,
    metrics: dict[str, float],
    fit_time_sec: float,
) -> dict[str, Any]:
    return {
        "scene_id": scene_id,
        "scene_name": scene_name,
        "method": method,
        "method_label": METHOD_LABELS[method],
        "RMSE_missing": float(metrics["RMSE_missing"]),
        "SAM_missing": float(metrics["SAM_missing"]),
        "NMSE_dB_all": float(metrics["NMSE_dB_all"]),
        "fit_time_sec": float(fit_time_sec),
    }


def _summary_to_latex(summary: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{@{}l c c c c@{}}",
        r"\toprule",
        r"Method & RMSE*$\downarrow$ & SAM*$\downarrow$ & NMSE(dB)$\downarrow$ & Fit time (s)$\downarrow$\\",
        r"\midrule",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            " & ".join(
                [
                    METHOD_LABELS[str(row["method"])],
                    pm_latex(float(row["RMSE_missing_mean"]), float(row["RMSE_missing_std"]), 5),
                    pm_latex(float(row["SAM_missing_mean"]), float(row["SAM_missing_std"]), 3),
                    pm_latex(float(row["NMSE_dB_all_mean"]), float(row["NMSE_dB_all_std"]), 3),
                    pm_latex(float(row["fit_time_sec_mean"]), float(row["fit_time_sec_std"]), 2),
                ]
            )
            + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
