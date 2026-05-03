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
    _completion_metrics,
    load_scene_original,
    load_target_runs,
)
from experiment.utils.io import load_state_mat
from src.methods.mlpcal import ScalarMLPCalibration
from src.methods.polycal import PolynomialCalibration


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test Tucker + MLPCal on a small CAVE completion subset.")
    parser.add_argument("--scene-id", type=int, default=1)
    parser.add_argument("--mask-seed", type=int, default=0)
    parser.add_argument("--missing-rate", type=float, default=0.5)
    parser.add_argument("--max-iter", type=int, default=MLPCAL_MAX_ITER)
    parser.add_argument("--max-train-samples", type=int, default=MLPCAL_MAX_TRAIN_SAMPLES)
    args = parser.parse_args()

    frame, _ = load_target_runs()
    panel = frame.loc[
        frame["method_name"].eq("tucker")
        & frame["scene_id"].eq(args.scene_id)
        & frame["mask_seed"].eq(args.mask_seed)
        & np.isclose(frame["missing_rate"], args.missing_rate, atol=1e-12)
    ].copy()
    if panel.empty:
        raise SystemExit(
            "No matching Tucker run found for "
            f"scene_id={args.scene_id}, mask_seed={args.mask_seed}, missing_rate={args.missing_rate}."
        )
    row = panel.iloc[0]

    state = load_state_mat(_resolve_state_path(row["state_path"]))
    recon_tucker = np.asarray(_jsonish(state["reconstruction"]), dtype=np.float32)
    observed_mask = np.asarray(_jsonish(state["observed_mask"]), dtype=bool)
    scene_name, original = load_scene_original(args.scene_id, **_cave_dataset_kwargs_from_row(row))

    poly = PolynomialCalibration(degree=4, lambda_reg=1e-6).fit(recon_tucker, original, observed_mask)
    recon_poly = poly.apply(recon_tucker)
    poly_metrics = _completion_metrics(original, recon_poly, observed_mask)

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
    mlp_metrics = _completion_metrics(original, recon_mlp, observed_mask)

    tucker_metrics = {
        "RMSE_missing": float(row["RMSE_missing"]),
        "SAM_missing": float(row["SAM_missing"]),
        "NMSE_dB_all": float(row["NMSE_dB_all"]),
    }

    print(f"Scene: {args.scene_id} ({scene_name}), seed={args.mask_seed}, missing_rate={args.missing_rate}")
    print("Tucker:", _fmt_metrics(tucker_metrics))
    print("PolyCal(P=4):", _fmt_metrics(poly_metrics))
    print("MLPCal:", _fmt_metrics(mlp_metrics))
    assert mlp.diagnostics is not None
    print(
        "MLPCal diagnostics:",
        {
            "fit_time_sec": round(mlp.diagnostics.fit_time_sec, 3),
            "train_count": mlp.diagnostics.train_count,
            "final_loss": round(mlp.diagnostics.final_loss, 6),
            "hidden_units": mlp.diagnostics.hidden_units,
            "max_iter": mlp.diagnostics.max_iter,
        },
    )


def _fmt_metrics(metrics: dict[str, float]) -> dict[str, float]:
    keys = ["RMSE_missing", "SAM_missing", "NMSE_dB_all"]
    return {key: round(float(metrics[key]), 6) for key in keys if key in metrics}


if __name__ == "__main__":
    main()
