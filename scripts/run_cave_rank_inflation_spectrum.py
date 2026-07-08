from __future__ import annotations

import argparse
import os
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
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
from scipy.io import loadmat
from tensorly.tucker_tensor import tucker_to_tensor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.hsi import CAVEHSIData
from src.filters import BiasFilter, NonlinearFilter
from src.methods.ntdpl import NTDPLDecomposition
from src.methods.tucker import TuckerDecomposition
from src.metrics import val_NMSE_dB, val_RMSE, val_SAM
from src.types import LogCallback, Tensor


DEFAULT_RUNS = PROJECT_ROOT / "artifacts" / "multirun" / "cave-representation" / "runs.parquet"
DEFAULT_OUTDIR = PROJECT_ROOT / "artifacts" / "results" / "rank_inflation_spectrum"
TENSOR_KINDS = (
    "measured",
    "tucker_fit",
    "ntdpl_signal",
    "ntdpl_prediction",
    "ntdpl_response_component",
    "tucker_residual",
    "ntdpl_residual",
)
ENERGY_THRESHOLDS = (0.95, 0.99, 0.995, 0.999)


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


def _parse_rank(value: Any) -> tuple[int, int, int]:
    parsed = _jsonish(value)
    if isinstance(parsed, str):
        parsed = parsed.strip("[]()")
        return tuple(int(part.strip()) for part in parsed.split(",") if part.strip())  # type: ignore[return-value]
    if isinstance(parsed, (list, tuple)):
        return tuple(int(item) for item in parsed)  # type: ignore[return-value]
    raise ValueError(f"Cannot parse rank from {value!r}.")


def _rank_text(rank: tuple[int, int, int]) -> str:
    return f"({rank[0]},{rank[1]},{rank[2]})"


def _base_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out = out.loc[out["data._name"].astype(str).eq("cave_hsi")].copy()
    out["method_name"] = out["method._name"].astype(str)
    out["rank"] = out["method.rank"].map(_parse_rank)
    out["rank_text"] = out["rank"].map(_rank_text)
    out["scene_id"] = out["data.id"].astype(int)
    if "method.p_max" in out.columns:
        out["p_max"] = pd.to_numeric(out["method.p_max"], errors="coerce")
        dedup_keys = ["scene_id", "method_name", "rank_text", "p_max"]
    else:
        dedup_keys = ["scene_id", "method_name", "rank_text"]
    return out.sort_values("run_dir").drop_duplicates(subset=dedup_keys, keep="last").reset_index(drop=True)


def _main_experiment_rows(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "p_max" not in out.columns:
        return out
    ntdpl_main = out["method_name"].eq("ntdpl") & out["p_max"].eq(6.0)
    return out.loc[~out["method_name"].eq("ntdpl") | ntdpl_main].copy()


def _dataset_from_row(row: pd.Series) -> CAVEHSIData:
    target_shape = _jsonish(row.get("data.target_shape", [512, 512]))
    crop_shape = _jsonish(row.get("data.crop_shape", None))
    if crop_shape == "null":
        crop_shape = None
    dataset = CAVEHSIData(
        path=str(row.get("data.path", "data/CAVE")),
        id=int(row["scene_id"]),
        target_shape=tuple(int(v) for v in target_shape),
        crop_shape=None if crop_shape is None else tuple(int(v) for v in crop_shape),
    )
    filter_name = str(row.get("filter._name", "bias-filter"))
    filter_cls = NonlinearFilter if filter_name == "nonlinear-filter" else BiasFilter
    snr_db = _jsonish(row.get("filter.snr_db", None))
    bias = _jsonish(row.get("filter.bias", None))
    data_filter = filter_cls(
        seed=int(row.get("filter.seed", 0)),
        normalize_method=_jsonish(row.get("filter.normalize_method", "max")),
        snr_db=None if snr_db in {"null", ""} else snr_db,
        bias=None if bias in {"null", ""} else bias,
        nonlinear=_jsonish(row.get("filter.nonlinear", "none")),
        alpha=float(row.get("filter.alpha", 0.0)) if "filter.alpha" in row else 0.0,
    )
    data_filter(dataset)
    return dataset


def _resolve_artifact_path(path_text: Any) -> Path:
    raw = Path(str(_jsonish(path_text)).strip().strip('"'))
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend([PROJECT_ROOT / raw, PROJECT_ROOT / "artifacts" / raw])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve artifact path {path_text!r}; tried {candidates}.")


def _load_state(row: dict[str, Any]) -> dict[str, Any]:
    path_value = row.get("state_path") or row.get("run_dir")
    path = _resolve_artifact_path(path_value)
    if path.is_dir():
        path = path / "state.mat"
    mat = loadmat(path, squeeze_me=True)
    return {k: v for k, v in mat.items() if not k.startswith("__")}


def _state_reconstruction(state: dict[str, Any]) -> np.ndarray:
    if "reconstruction" in state:
        return np.asarray(state["reconstruction"], dtype=np.float32)
    core = np.asarray(state["core"], dtype=np.float32)
    factors = [np.asarray(factor, dtype=np.float32) for factor in np.asarray(state["factors"], dtype=object).reshape(-1)]
    return np.asarray(tucker_to_tensor((core, factors)), dtype=np.float32)


def _state_tucker_signal(state: dict[str, Any]) -> np.ndarray:
    core = np.asarray(state["core"], dtype=np.float32)
    factors = [np.asarray(factor, dtype=np.float32) for factor in np.asarray(state["factors"], dtype=object).reshape(-1)]
    return np.asarray(tucker_to_tensor((core, factors)), dtype=np.float32)


def _unfold(tensor: np.ndarray, mode: int) -> np.ndarray:
    moved = np.moveaxis(np.asarray(tensor, dtype=np.float64), mode, 0)
    return moved.reshape(moved.shape[0], -1)


def _singular_values(tensor: np.ndarray, mode: int) -> np.ndarray:
    mat = _unfold(tensor, mode)
    gram = mat @ mat.T
    eigvals = np.linalg.eigvalsh(gram)
    eigvals = np.maximum(eigvals, 0.0)
    return np.sqrt(eigvals[::-1])


def _spectrum_metrics(svals: np.ndarray, base_rank: int) -> dict[str, float | int]:
    values = np.asarray(svals, dtype=np.float64)
    energy = values**2
    total = float(np.sum(energy))
    if total <= 0.0:
        out: dict[str, float | int] = {
            "stable_rank": 0.0,
            "entropy_rank": 0.0,
            "tail_energy_at_base_rank": 0.0,
            "tail_energy_at_2x_base_rank": 0.0,
            "tail_energy_after_base_rank": 0.0,
            "tail_energy_after_2x_base_rank": 0.0,
        }
        for threshold in ENERGY_THRESHOLDS:
            out[f"rank_energy_{int(round(1000 * threshold))}"] = 0
        return out

    cumulative = np.cumsum(energy) / total
    probs = energy / total
    positive = probs[probs > 0.0]
    base_rank = int(max(base_rank, 0))
    double_rank = int(min(len(values), max(2 * base_rank, 0)))
    out = {
        "stable_rank": float(total / max(values[0] ** 2, 1e-30)),
        "entropy_rank": float(np.exp(-np.sum(positive * np.log(positive)))),
        "tail_energy_at_base_rank": float(1.0 - cumulative[min(base_rank, len(cumulative)) - 1]) if base_rank > 0 else 1.0,
        "tail_energy_at_2x_base_rank": float(1.0 - cumulative[double_rank - 1]) if double_rank > 0 else 1.0,
        "tail_energy_after_base_rank": float(np.sum(energy[base_rank:]) / total),
        "tail_energy_after_2x_base_rank": float(np.sum(energy[double_rank:]) / total),
    }
    for threshold in ENERGY_THRESHOLDS:
        out[f"rank_energy_{int(round(1000 * threshold))}"] = int(np.searchsorted(cumulative, threshold) + 1)
    return out


def _scene_rows(scene_frame: pd.DataFrame, rank: tuple[int, int, int]) -> list[dict[str, Any]]:
    scene_id = int(scene_frame["scene_id"].iloc[0])
    scene_name = str(scene_frame["scene_name"].iloc[0]) if "scene_name" in scene_frame else f"scene_{scene_id:02d}"
    rank_text = _rank_text(rank)
    tucker_row = scene_frame.loc[scene_frame["method_name"].eq("tucker")].iloc[0].to_dict()
    ntdpl_row = scene_frame.loc[scene_frame["method_name"].eq("ntdpl")].iloc[0].to_dict()
    dataset = _dataset_from_row(scene_frame.iloc[0])
    measured = np.asarray(dataset.get("eval").dense, dtype=np.float32)
    tucker_state = _load_state(tucker_row)
    ntdpl_state = _load_state(ntdpl_row)
    tensors = {
        "measured": measured,
        "tucker_fit": _state_reconstruction(tucker_state),
        "ntdpl_signal": _state_tucker_signal(ntdpl_state),
        "ntdpl_prediction": _state_reconstruction(ntdpl_state),
    }
    tensors["ntdpl_response_component"] = tensors["ntdpl_prediction"] - tensors["ntdpl_signal"]
    tensors["tucker_residual"] = tensors["measured"] - tensors["tucker_fit"]
    tensors["ntdpl_residual"] = tensors["measured"] - tensors["ntdpl_prediction"]

    rows: list[dict[str, Any]] = []
    for tensor_kind, tensor in tensors.items():
        for mode, base_rank in enumerate(rank):
            svals = _singular_values(tensor, mode)
            metrics = _spectrum_metrics(svals, base_rank)
            rows.append(
                {
                    "scene_id": scene_id,
                    "scene_name": scene_name,
                    "rank": rank_text,
                    "mode": mode + 1,
                    "base_rank": int(base_rank),
                    "tensor_kind": tensor_kind,
                    "num_singular_values": int(len(svals)),
                    "top_singular_values": json.dumps([float(v) for v in svals[: min(24, len(svals))]]),
                    **metrics,
                }
            )
    return rows


def _metric_rows(original: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = Tensor(shape=original.shape, dense=original)
    pred = Tensor(shape=prediction.shape, dense=prediction)
    return {
        "RMSE": float(val_RMSE(target, pred)),
        "SAM": float(val_SAM(target, pred)),
        "NMSE_dB": float(val_NMSE_dB(target, pred)),
    }


def _load_cave_scene(scene_id: int, target_shape: tuple[int, int]) -> tuple[str, np.ndarray]:
    dataset = CAVEHSIData(path="data/CAVE", id=int(scene_id), target_shape=target_shape, crop_shape=None)
    BiasFilter(seed=0, normalize_method="max", snr_db=None, bias=None)(dataset)
    scene_name = str(getattr(dataset, "scene_name", f"scene_{scene_id:02d}"))
    return scene_name, np.asarray(dataset.get("eval").dense, dtype=np.float32)


def _fit_tucker_direct(cube: np.ndarray, rank: tuple[int, int, int], n_iter_max: int) -> tuple[np.ndarray, dict[str, float]]:
    from time import perf_counter

    model = TuckerDecomposition(rank=rank, n_iter_max=n_iter_max, init="svd", tol=1e-4)
    tensor = Tensor(shape=cube.shape, dense=cube)
    start = perf_counter()
    model.fit(tensor, mask=None, logcallback=LogCallback(log_level=0))
    elapsed = perf_counter() - start
    return np.asarray(model.reconstruct().dense, dtype=np.float32), {
        "fit_time_sec": float(elapsed),
        "params": float(model.get_num_params()),
    }


def _fit_ntdpl_direct(
    cube: np.ndarray,
    rank: tuple[int, int, int],
    n_iter_max: int,
    p_max: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    from time import perf_counter

    model = NTDPLDecomposition(
        rank=rank,
        init_n_iter_max=50,
        p_max=int(p_max),
        allow_constant_term=True,
        n_iter_max=n_iter_max,
        use_continuation=True,
        factor_normalize=True,
        lr_core=1e-4,
        lr_factors=3e-4,
        lambda_core=1e-6,
        lambda_factors=1e-6,
        lambda_beta=1e-6,
        beta_update_method="ridge_lstsq",
        init="tucker",
        random_state=0,
        beta_update_interval=5,
        solver_variant="optimized",
        stable_beta_update=True,
        beta_update_stage="before_grad",
        link_kind="power",
    )
    tensor = Tensor(shape=cube.shape, dense=cube)
    start = perf_counter()
    model.fit(tensor, mask=None, logcallback=LogCallback(log_level=0))
    elapsed = perf_counter() - start
    signal = np.asarray(tucker_to_tensor((model.core, model.factors)), dtype=np.float32)
    prediction = np.asarray(model.reconstruct().dense, dtype=np.float32)
    beta = np.asarray(model.beta, dtype=np.float32)
    return signal, prediction, beta, {
        "fit_time_sec": float(elapsed),
        "params": float(model.get_num_params()),
    }


def _direct_scene_rows(
    scene_id: int,
    target_shape: tuple[int, int],
    rank: tuple[int, int, int],
    n_iter_max: int,
    p_max: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scene_name, measured = _load_cave_scene(scene_id, target_shape)
    tucker_fit, tucker_info = _fit_tucker_direct(measured, rank, n_iter_max)
    ntdpl_signal, ntdpl_prediction, beta, ntdpl_info = _fit_ntdpl_direct(measured, rank, n_iter_max, p_max)
    tensors = {
        "measured": measured,
        "tucker_fit": tucker_fit,
        "ntdpl_signal": ntdpl_signal,
        "ntdpl_prediction": ntdpl_prediction,
        "ntdpl_response_component": ntdpl_prediction - ntdpl_signal,
        "tucker_residual": measured - tucker_fit,
        "ntdpl_residual": measured - ntdpl_prediction,
    }

    spectrum_rows: list[dict[str, Any]] = []
    for tensor_kind, tensor in tensors.items():
        for mode, base_rank in enumerate(rank):
            svals = _singular_values(tensor, mode)
            spectrum_rows.append(
                {
                    "scene_id": int(scene_id),
                    "scene_name": scene_name,
                    "rank": _rank_text(rank),
                    "target_shape": str(target_shape),
                    "p_max": int(p_max),
                    "n_iter_max": int(n_iter_max),
                    "mode": mode + 1,
                    "base_rank": int(base_rank),
                    "tensor_kind": tensor_kind,
                    "num_singular_values": int(len(svals)),
                    "top_singular_values": json.dumps([float(v) for v in svals[: min(24, len(svals))]]),
                    **_spectrum_metrics(svals, base_rank),
                }
            )

    fit_rows = [
        {
            "scene_id": int(scene_id),
            "scene_name": scene_name,
            "rank": _rank_text(rank),
            "target_shape": str(target_shape),
            "method": "Tucker",
            **tucker_info,
            **_metric_rows(measured, tucker_fit),
        },
        {
            "scene_id": int(scene_id),
            "scene_name": scene_name,
            "rank": _rank_text(rank),
            "target_shape": str(target_shape),
            "method": "NTD-PL",
            "p_max": int(p_max),
            "beta": json.dumps([float(v) for v in beta.reshape(-1)]),
            **ntdpl_info,
            **_metric_rows(measured, ntdpl_prediction),
        },
    ]
    return spectrum_rows, fit_rows


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if abs(denominator) > 1e-12 else float("nan")


def _paired_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    idx = ["scene_id", "scene_name", "rank", "mode", "base_rank"]
    wide = metrics.pivot_table(index=idx, columns="tensor_kind", values=["rank_energy_990", "rank_energy_995", "rank_energy_999", "tail_energy_after_base_rank", "entropy_rank", "stable_rank"], aggfunc="first")
    wide.columns = [f"{metric}_{kind}" for metric, kind in wide.columns]
    wide = wide.reset_index()
    rows = []
    for row in wide.to_dict("records"):
        out = dict(row)
        for metric in ("rank_energy_990", "rank_energy_995", "rank_energy_999", "tail_energy_after_base_rank", "entropy_rank", "stable_rank"):
            out[f"{metric}_ntdpl_pred_over_signal"] = _ratio(float(row[f"{metric}_ntdpl_prediction"]), float(row[f"{metric}_ntdpl_signal"]))
            out[f"{metric}_measured_over_signal"] = _ratio(float(row[f"{metric}_measured"]), float(row[f"{metric}_ntdpl_signal"]))
        out["tail_capture_ratio_pred_vs_measured"] = _ratio(
            float(row["tail_energy_after_base_rank_ntdpl_prediction"]),
            float(row["tail_energy_after_base_rank_measured"]),
        )
        rows.append(out)
    return pd.DataFrame(rows)


def _aggregate(paired: pd.DataFrame) -> pd.DataFrame:
    agg_cols = [
        "rank_energy_990_ntdpl_prediction",
        "rank_energy_990_ntdpl_signal",
        "rank_energy_990_measured",
        "rank_energy_990_tucker_fit",
        "tail_energy_after_base_rank_ntdpl_prediction",
        "tail_energy_after_base_rank_ntdpl_signal",
        "tail_energy_after_base_rank_measured",
        "tail_energy_after_base_rank_tucker_fit",
        "tail_capture_ratio_pred_vs_measured",
        "entropy_rank_ntdpl_prediction",
        "entropy_rank_ntdpl_signal",
        "entropy_rank_measured",
    ]
    summary = (
        paired.groupby(["rank", "mode", "base_rank"], as_index=False)[agg_cols]
        .agg(["mean", "median", "std"])
        .reset_index()
    )
    summary.columns = ["_".join(str(part) for part in col if part != "") for col in summary.columns]
    return summary


def build(*, runs_path: Path, ranks: set[str] | None, scene_ids: set[int] | None, max_workers: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    runs = pd.read_parquet(runs_path)
    table = _main_experiment_rows(_base_columns(runs))
    table = table.loc[table["method_name"].isin(["tucker", "ntdpl"])].copy()
    if ranks:
        table = table.loc[table["rank_text"].isin(ranks)].copy()
    if scene_ids:
        table = table.loc[table["scene_id"].isin(scene_ids)].copy()

    groups: list[tuple[int, tuple[int, int, int], pd.DataFrame]] = []
    for (scene_id, rank_text), group in table.groupby(["scene_id", "rank_text"], sort=True):
        if set(group["method_name"]) >= {"tucker", "ntdpl"}:
            groups.append((int(scene_id), _parse_rank(rank_text), group.copy()))
    if not groups:
        raise RuntimeError("No complete Tucker/NTD-PL scene-rank pairs found.")

    rows: list[dict[str, Any]] = []
    if max_workers <= 1:
        for _, rank, group in groups:
            rows.extend(_scene_rows(group, rank))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_scene_rows, group, rank) for _, rank, group in groups]
            for future in as_completed(futures):
                rows.extend(future.result())

    metrics = pd.DataFrame(rows).sort_values(["rank", "scene_id", "mode", "tensor_kind"]).reset_index(drop=True)
    paired = _paired_summary(metrics).sort_values(["rank", "scene_id", "mode"]).reset_index(drop=True)
    aggregate = _aggregate(paired)
    return metrics, paired, aggregate


def build_direct(
    *,
    scene_ids: set[int] | None,
    rank: tuple[int, int, int],
    target_shape: tuple[int, int],
    n_iter_max: int,
    p_max: int,
    max_workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scenes = sorted(scene_ids or set(range(1, 16)))
    spectrum_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    if max_workers <= 1:
        for scene_id in scenes:
            scene_spectrum, scene_fit = _direct_scene_rows(scene_id, target_shape, rank, n_iter_max, p_max)
            spectrum_rows.extend(scene_spectrum)
            fit_rows.extend(scene_fit)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_direct_scene_rows, scene_id, target_shape, rank, n_iter_max, p_max)
                for scene_id in scenes
            ]
            for future in as_completed(futures):
                scene_spectrum, scene_fit = future.result()
                spectrum_rows.extend(scene_spectrum)
                fit_rows.extend(scene_fit)

    metrics = pd.DataFrame(spectrum_rows).sort_values(["rank", "scene_id", "mode", "tensor_kind"]).reset_index(drop=True)
    paired = _paired_summary(metrics).sort_values(["rank", "scene_id", "mode"]).reset_index(drop=True)
    aggregate = _aggregate(paired)
    fit_metrics = pd.DataFrame(fit_rows).sort_values(["scene_id", "method"]).reset_index(drop=True)
    return metrics, paired, aggregate, fit_metrics


def _parse_int_set(text: str | None) -> set[int] | None:
    if not text:
        return None
    out: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(item.strip()) for item in part.split("-", 1)]
            out.update(range(start, end + 1))
        else:
            out.add(int(part))
    return out


def _parse_rank_filter(text: str) -> set[str] | None:
    stripped = text.strip()
    if not stripped:
        return None
    if ";" in stripped:
        return {item.strip() for item in stripped.split(";") if item.strip()}
    return {_rank_text(_parse_rank(stripped))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute CAVE rank-inflation spectrum diagnostics from saved reconstruction states.")
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--ranks", default="", help="Comma-separated rank texts, e.g. '(40,40,5),(33,33,4)'. Empty means all.")
    parser.add_argument("--scene-ids", default="", help="Comma/range scene ids, e.g. '1-15'. Empty means all.")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--direct-fit", action="store_true", help="Fit Tucker and NTD-PL directly instead of reading saved states.")
    parser.add_argument("--rank", default="12,12,6", help="Rank for --direct-fit.")
    parser.add_argument("--target-shape", default="128,128", help="Spatial target shape for --direct-fit.")
    parser.add_argument("--n-iter-max", type=int, default=80)
    parser.add_argument("--p-max", type=int, default=4)
    args = parser.parse_args()

    rank_filter = _parse_rank_filter(args.ranks)
    scene_filter = _parse_int_set(args.scene_ids)

    outdir = args.outdir if args.outdir.is_absolute() else PROJECT_ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    if args.direct_fit:
        rank = _parse_rank(args.rank)
        target_shape_raw = tuple(int(part.strip()) for part in args.target_shape.replace("x", ",").split(",") if part.strip())
        if len(target_shape_raw) != 2:
            raise ValueError("Expected --target-shape with two entries.")
        metrics, paired, aggregate, fit_metrics = build_direct(
            scene_ids=scene_filter,
            rank=rank,
            target_shape=(target_shape_raw[0], target_shape_raw[1]),
            n_iter_max=int(args.n_iter_max),
            p_max=int(args.p_max),
            max_workers=max(1, int(args.max_workers)),
        )
        metrics.to_csv(outdir / "spectrum_metrics.csv", index=False)
        paired.to_csv(outdir / "paired_rank_inflation.csv", index=False)
        aggregate.to_csv(outdir / "summary_by_rank_mode.csv", index=False)
        fit_metrics.to_csv(outdir / "fit_metrics.csv", index=False)
        print(f"Wrote {len(metrics)} spectrum rows, {len(paired)} paired rows, {len(fit_metrics)} fit rows.")
        print(f"Output: {outdir}")
        print(aggregate.to_string(index=False))
        print(fit_metrics.groupby('method')[['RMSE', 'SAM', 'fit_time_sec']].mean().to_string())
        return

    metrics, paired, aggregate = build(
        runs_path=args.runs if args.runs.is_absolute() else PROJECT_ROOT / args.runs,
        ranks=rank_filter,
        scene_ids=scene_filter,
        max_workers=max(1, int(args.max_workers)),
    )
    metrics.to_csv(outdir / "spectrum_metrics.csv", index=False)
    paired.to_csv(outdir / "paired_rank_inflation.csv", index=False)
    aggregate.to_csv(outdir / "summary_by_rank_mode.csv", index=False)
    print(f"Wrote {len(metrics)} spectrum rows, {len(paired)} paired rows.")
    print(f"Output: {outdir}")
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
