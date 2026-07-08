from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment.process.helpers.cave_random_completion_polycal import (
    MLPCAL_BATCH_SIZE,
    MLPCAL_HIDDEN_UNITS,
    MLPCAL_LAMBDA,
    MLPCAL_LR,
    MLPCAL_MAX_ITER,
    MLPCAL_MAX_TRAIN_SAMPLES,
    POLYCAL_LAMBDA,
    pm_latex,
)
from experiment.process.helpers.mechanism_closure import (
    MAIN_POLYCAL_DEGREE,
    RECON_MAIN_RANK,
    _load_recon_runs_main_rank,
    _load_scene_original,
    _resolve_state_path,
    _state_reconstruction,
)
from experiment.utils.io import load_state_mat
from src.methods.mlpcal import ScalarMLPCalibration
from src.methods.polycal import PolynomialCalibration
from src.metrics import val_NMSE_dB, val_RMSE, val_SAM
from src.types import Tensor


METHOD_ORDER = ["tucker", "polycal", "mlpcal", "ntdpl"]
METHOD_LABELS = {
    "tucker": "Tucker",
    "polycal": f"Tucker + PolyCal(P={MAIN_POLYCAL_DEGREE})",
    "mlpcal": f"Tucker + MLPCal(H={MLPCAL_HIDDEN_UNITS})",
    "ntdpl": "NTD-PL",
}
LATENT_UPDATE = {
    "tucker": "Joint",
    "polycal": "Fixed Tucker",
    "mlpcal": "Fixed Tucker",
    "ntdpl": "Joint",
}
SCALAR_MAP = {
    "tucker": "Identity",
    "polycal": "Polynomial",
    "mlpcal": "Scalar MLP",
    "ntdpl": "Polynomial",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect fixed-backbone scalar-link refresh baselines on CAVE reconstruction."
    )
    parser.add_argument("--scene-ids", default="1-15")
    parser.add_argument("--hidden-units", type=int, default=MLPCAL_HIDDEN_UNITS)
    parser.add_argument("--lambda-reg", type=float, default=MLPCAL_LAMBDA)
    parser.add_argument("--lr", type=float, default=MLPCAL_LR)
    parser.add_argument("--batch-size", type=int, default=MLPCAL_BATCH_SIZE)
    parser.add_argument("--max-iter", type=int, default=MLPCAL_MAX_ITER)
    parser.add_argument("--max-train-samples", type=int, default=MLPCAL_MAX_TRAIN_SAMPLES)
    parser.add_argument("--out-prefix", default="papers/neurips/tables/recon_calibration_baselines")
    args = parser.parse_args()

    frame = _load_recon_runs_main_rank()
    rows: list[dict[str, Any]] = []
    scene_ids = _parse_scene_ids(args.scene_ids)
    method_labels = dict(METHOD_LABELS)
    method_labels["mlpcal"] = f"Tucker + MLPCal(H={args.hidden_units})"

    for scene_id in scene_ids:
        tucker_row = _select_row(frame, scene_id, "tucker")
        ntdpl_row = _select_row(frame, scene_id, "ntdpl")
        original = _load_scene_original(int(scene_id), tucker_row)
        mask = np.ones_like(original, dtype=bool)

        tucker_state = load_state_mat(_resolve_state_path(str(tucker_row["state_path"])))
        ntdpl_state = load_state_mat(_resolve_state_path(str(ntdpl_row["state_path"])))
        recon_tucker = _state_reconstruction(tucker_state)
        recon_ntdpl = _state_reconstruction(ntdpl_state)

        rows.append(_row(scene_id, "tucker", original, recon_tucker, float(tucker_row["fit_time_sec"]), method_labels))
        rows.append(_row(scene_id, "ntdpl", original, recon_ntdpl, float(ntdpl_row["fit_time_sec"]), method_labels))

        poly = PolynomialCalibration(degree=MAIN_POLYCAL_DEGREE, lambda_reg=POLYCAL_LAMBDA).fit(
            recon_tucker,
            original,
            mask,
        )
        rows.append(
            _row(
                scene_id,
                "polycal",
                original,
                poly.apply(recon_tucker),
                float(poly.diagnostics.fit_time_sec) if poly.diagnostics is not None else np.nan,
                method_labels,
            )
        )

        mlp = ScalarMLPCalibration(
            hidden_units=args.hidden_units,
            lambda_reg=args.lambda_reg,
            lr=args.lr,
            max_iter=args.max_iter,
            batch_size=args.batch_size,
            max_train_samples=args.max_train_samples,
            random_state=int(scene_id),
        ).fit(recon_tucker, original, mask)
        rows.append(
            _row(
                scene_id,
                "mlpcal",
                original,
                mlp.apply(recon_tucker),
                float(mlp.diagnostics.fit_time_sec) if mlp.diagnostics is not None else np.nan,
                method_labels,
            )
        )
        print(f"Finished CAVE scene {scene_id:02d}")

    per_scene = pd.DataFrame(rows)
    per_scene["order"] = per_scene["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    per_scene = per_scene.sort_values(["scene_id", "order"]).drop(columns="order").reset_index(drop=True)

    summary = _summarize(per_scene)
    out_prefix = PROJECT_ROOT / args.out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    per_scene.to_csv(out_prefix.with_suffix(".per_scene.csv"), index=False)
    summary.to_csv(out_prefix.with_suffix(".summary.csv"), index=False)
    out_prefix.with_suffix(".tex").write_text(_summary_to_latex(summary), encoding="utf-8")
    print(f"Saved {out_prefix.with_suffix('.per_scene.csv')}")
    print(f"Saved {out_prefix.with_suffix('.summary.csv')}")
    print(f"Saved {out_prefix.with_suffix('.tex')}")
    print(summary.to_string(index=False))


def _parse_scene_ids(text: str) -> list[int]:
    if "-" in text:
        start, end = [int(part.strip()) for part in text.split("-", 1)]
        return list(range(start, end + 1))
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _select_row(frame: pd.DataFrame, scene_id: int, method_name: str) -> pd.Series:
    panel = frame.loc[
        frame["scene_id"].eq(int(scene_id)) & frame["method_name"].eq(method_name)
    ].copy()
    if panel.empty:
        raise ValueError(f"No reconstruction row for scene_id={scene_id}, method={method_name}.")
    return panel.iloc[0]


def _row(
    scene_id: int,
    method: str,
    original: np.ndarray,
    reconstruction: np.ndarray,
    fit_time_sec: float,
    method_labels: dict[str, str],
) -> dict[str, Any]:
    original_tensor = Tensor(shape=original.shape, dense=original)
    recon_tensor = Tensor(shape=reconstruction.shape, dense=reconstruction)
    return {
        "scene_id": int(scene_id),
        "method": method,
        "method_label": method_labels[method],
        "latent_update": LATENT_UPDATE[method],
        "scalar_map": SCALAR_MAP[method],
        "rank": str(tuple(RECON_MAIN_RANK)),
        "RMSE": val_RMSE(original_tensor, recon_tensor),
        "NMSE_dB": val_NMSE_dB(original_tensor, recon_tensor),
        "SAM": val_SAM(original_tensor, recon_tensor),
        "fit_time_sec": float(fit_time_sec),
    }


def _summarize(per_scene: pd.DataFrame) -> pd.DataFrame:
    summary = (
        per_scene.groupby("method", as_index=False)
        .agg(
            method_label=("method_label", "first"),
            latent_update=("latent_update", "first"),
            scalar_map=("scalar_map", "first"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            NMSE_dB_mean=("NMSE_dB", "mean"),
            NMSE_dB_std=("NMSE_dB", "std"),
            SAM_mean=("SAM", "mean"),
            SAM_std=("SAM", "std"),
            fit_time_sec_mean=("fit_time_sec", "mean"),
            fit_time_sec_std=("fit_time_sec", "std"),
            n_scenes=("scene_id", "nunique"),
        )
    )
    summary["order"] = summary["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    summary = summary.sort_values("order").drop(columns="order").reset_index(drop=True)
    tucker = summary.loc[summary["method"].eq("tucker")].iloc[0]
    summary["RMSE_gain_pct"] = 100.0 * (float(tucker["RMSE_mean"]) - summary["RMSE_mean"]) / float(tucker["RMSE_mean"])
    summary["SAM_gain_pct"] = 100.0 * (float(tucker["SAM_mean"]) - summary["SAM_mean"]) / float(tucker["SAM_mean"])
    return summary


def _gain_text(value: float, method: str) -> str:
    if method == "tucker":
        return "--"
    return f"{value:.1f}\\%"


def _summary_to_latex(summary: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{@{}l c c c c c c@{}}",
        r"\toprule",
        r"Method & Latent & Map & RMSE$\downarrow$ & SAM$\downarrow$ & RMSE gain & SAM gain \\",
        r"\midrule",
    ]
    for row in summary.to_dict("records"):
        rmse = pm_latex(float(row["RMSE_mean"]), float(row["RMSE_std"]), 4)
        sam = pm_latex(float(row["SAM_mean"]), float(row["SAM_std"]), 2)
        if row["method"] == "ntdpl":
            rmse = rf"\textbf{{{rmse}}}"
            sam = rf"\textbf{{{sam}}}"
        lines.append(
            " & ".join(
                [
                    str(row["method_label"]),
                    str(row["latent_update"]),
                    str(row["scalar_map"]),
                    rmse,
                    sam,
                    _gain_text(float(row["RMSE_gain_pct"]), str(row["method"])),
                    _gain_text(float(row["SAM_gain_pct"]), str(row["method"])),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
