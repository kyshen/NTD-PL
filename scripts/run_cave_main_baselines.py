from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "TBB_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.hsi import CAVEHSIData
from src.filters.bias import BiasFilter
from src.methods.cp import CPDecomposition
from src.methods.ntdpl import NTDPLDecomposition
from src.methods.tt import TTDecomposition
from src.methods.tucker import TuckerDecomposition
from src.metrics import val_NMSE_dB, val_RMSE, val_SAM
from src.types import LogCallback, Tensor
from src.utils.tensor_ranks import (
    cp_rank_from_tucker,
    tt_rank_from_tucker,
)


DEFAULT_OUTDIR = PROJECT_ROOT / "artifacts" / "results" / "cave_main_baselines_r24_full"
DEFAULT_TABLE = PROJECT_ROOT / "papers" / "tsp" / "tables" / "cave_main_baselines.tex"
DEFAULT_RANK = (24, 24, 4)
DEFAULT_SHAPE = (512, 512, 31)
METHOD_ORDER = ("cp", "tt", "tucker", "ntdpl_p2", "ntdpl_p4", "ntdpl")
METHOD_LABELS = {
    "cp": "CP",
    "tt": "TT",
    "tucker": "Tucker",
    "ntdpl_p2": "NTD-PL ($P=2$)",
    "ntdpl_p4": "NTD-PL ($P=4$)",
    "ntdpl": "NTD-PL",
}


def _parse_rank(text: str) -> tuple[int, int, int]:
    parts = [int(part.strip()) for part in text.replace("x", ",").split(",") if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"Expected three rank entries, got {text!r}.")
    return (parts[0], parts[1], parts[2])


def _parse_int_set(text: str) -> list[int]:
    values: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(v.strip()) for v in part.split("-", 1)]
            values.update(range(start, end + 1))
        else:
            values.add(int(part))
    return sorted(values)


def _rank_text(rank: tuple[int, int, int]) -> str:
    return f"({rank[0]},{rank[1]},{rank[2]})"


def _method_tag(method_name: str, rank: tuple[int, int, int], scene_id: int) -> str:
    return f"{method_name}_r{rank[0]}_{rank[1]}_{rank[2]}_s{scene_id:02d}"


def _load_scene(scene_id: int) -> tuple[str, Tensor, Tensor]:
    dataset = CAVEHSIData(
        path="data/CAVE",
        id=scene_id,
        target_shape=(512, 512),
        crop_shape=None,
    )
    BiasFilter(seed=0, normalize_method="max", snr_db=None, bias=None)(dataset)
    return dataset.scene_name, dataset.get("fit"), dataset.get("eval")


def _evaluate(eval_tensor: Tensor, dense: np.ndarray, params: int, fit_time_sec: float) -> dict[str, Any]:
    rec = Tensor(shape=dense.shape, dense=np.asarray(dense, dtype=np.float32))
    return {
        "params": int(params),
        "fit_time_sec": float(fit_time_sec),
        "RMSE": float(val_RMSE(eval_tensor, rec)),
        "SAM": float(val_SAM(eval_tensor, rec)),
        "NMSE_dB": float(val_NMSE_dB(eval_tensor, rec)),
    }


def _fit_method(method_name: str, fit_tensor: Tensor, eval_tensor: Tensor, rank: tuple[int, int, int]) -> dict[str, Any]:
    start = perf_counter()
    log = LogCallback(0)

    if method_name == "tucker":
        method = TuckerDecomposition(rank=rank, n_iter_max=300, init="svd", tol=1e-4)
        method.fit(fit_tensor, None, log)
        dense = method.reconstruct().dense
        return _evaluate(eval_tensor, dense, method.get_num_params(), perf_counter() - start)

    if method_name == "cp":
        cp_rank = cp_rank_from_tucker(DEFAULT_SHAPE, rank, include_weights=True)
        method = CPDecomposition(
            rank=rank,
            cp_rank=cp_rank,
            n_iter_max=300,
            init_method="random",
            tol=1e-6,
            random_state=0,
            normalize_factors=False,
        )
        method.fit(fit_tensor, None, log)
        dense = method.reconstruct().dense
        out = _evaluate(eval_tensor, dense, method.get_num_params(), perf_counter() - start)
        out["effective_rank"] = int(cp_rank)
        return out

    if method_name == "tt":
        tt_rank = tt_rank_from_tucker(DEFAULT_SHAPE, rank)
        method = TTDecomposition(rank=rank, tt_rank=tt_rank, svd="truncated_svd")
        method.fit(fit_tensor, None, log)
        dense = method.reconstruct().dense
        out = _evaluate(eval_tensor, dense, method.get_num_params(), perf_counter() - start)
        out["effective_rank"] = str(tuple(int(v) for v in tt_rank))
        return out

    if method_name == "ntdpl" or method_name.startswith("ntdpl_p"):
        p_max = 6 if method_name == "ntdpl" else int(method_name.removeprefix("ntdpl_p"))
        method = NTDPLDecomposition(
            rank=rank,
            init_n_iter_max=50,
            init="tucker",
            stable_beta_update=True,
            beta_update_stage="before_grad",
            random_state=0,
            p_max=p_max,
            link_kind="power",
            allow_constant_term=True,
            n_iter_max=300,
            use_continuation=True,
            factor_normalize=True,
            lr_core=1e-4,
            lr_factors=3e-4,
            lambda_core=1e-6,
            lambda_factors=1e-6,
            lambda_beta=1e-6,
            beta_update_method="moments_normal_eq",
            beta_update_interval=5,
        )
        method.fit(fit_tensor, None, log)
        dense = method.reconstruct().dense
        out = _evaluate(eval_tensor, dense, method.get_num_params(), perf_counter() - start)
        out["p_max"] = int(p_max)
        return out

    raise ValueError(f"Unknown method {method_name!r}.")


def _run_one(
    scene_id: int,
    method_name: str,
    rank: tuple[int, int, int],
    outdir: str,
    force: bool,
) -> dict[str, Any]:
    out_path = Path(outdir) / f"{_method_tag(method_name, rank, scene_id)}.json"
    if out_path.exists() and not force:
        return json.loads(out_path.read_text(encoding="utf-8"))

    scene_name, fit_tensor, eval_tensor = _load_scene(scene_id)
    result = _fit_method(method_name, fit_tensor, eval_tensor, rank)
    row = {
        "scene_id": int(scene_id),
        "scene_name": scene_name,
        "rank": _rank_text(rank),
        "method_name": method_name,
        "method": METHOD_LABELS[method_name],
        **result,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(out_path)
    return row


def _collect(outdir: Path, rank: tuple[int, int, int], scene_ids: list[int], methods: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    wanted = {
        (scene_id, method_name)
        for scene_id in scene_ids
        for method_name in methods
    }
    for path in sorted(outdir.glob(f"*_r{rank[0]}_{rank[1]}_{rank[2]}_s*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        key = (int(row["scene_id"]), str(row["method_name"]))
        if key in wanted:
            rows.append(row)
    return pd.DataFrame(rows)


def _summarize(frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frame.groupby(["method_name", "method", "rank"], as_index=False)
        .agg(
            params=("params", "mean"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            SAM_mean=("SAM", "mean"),
            SAM_std=("SAM", "std"),
            NMSE_dB_mean=("NMSE_dB", "mean"),
            NMSE_dB_std=("NMSE_dB", "std"),
            fit_time_mean=("fit_time_sec", "mean"),
            n=("scene_id", "nunique"),
        )
    )
    order = {name: idx for idx, name in enumerate(METHOD_ORDER)}
    summary["order"] = summary["method_name"].map(order)
    return summary.sort_values(["order"]).drop(columns=["order"]).reset_index(drop=True)


def _paired_wins(frame: pd.DataFrame, metric: str) -> dict[str, int]:
    pivot = frame.pivot_table(index="scene_id", columns="method_name", values=metric, aggfunc="mean")
    if "ntdpl" not in pivot:
        return {}
    out: dict[str, int] = {}
    for method_name in pivot.columns:
        if method_name == "ntdpl":
            continue
        paired = pivot[[method_name, "ntdpl"]].dropna()
        out[method_name] = int((paired["ntdpl"] < paired[method_name]).sum())
    return out


def _fmt_mean_std(mean: float, std: float, digits: int) -> str:
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def _write_latex(summary: pd.DataFrame, frame: pd.DataFrame, path: Path) -> None:
    best_rmse = float(summary["RMSE_mean"].min())
    best_nmse = float(summary["NMSE_dB_mean"].min())

    lines = [
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l c r c c@{}}",
        r"\toprule",
        r"Method & Rank/budget & Params & RMSE$\downarrow$ & NMSE(dB)$\downarrow$ \\",
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        params = f"{int(round(float(row.params) / 1000.0))}k"
        rmse = _fmt_mean_std(float(row.RMSE_mean), float(row.RMSE_std), 4)
        nmse = _fmt_mean_std(float(row.NMSE_dB_mean), float(row.NMSE_dB_std), 2)
        if abs(float(row.RMSE_mean) - best_rmse) < 5e-6:
            rmse = rf"\textbf{{{rmse}}}"
        if abs(float(row.NMSE_dB_mean) - best_nmse) < 5e-4:
            nmse = rf"\textbf{{{nmse}}}"
        method = str(row.method)
        rank_budget = rf"\({row.rank}\)"
        if row.method_name == "cp":
            eff = frame.loc[frame["method_name"].eq("cp"), "effective_rank"].dropna()
            if not eff.empty:
                rank_budget = rf"\(R={int(float(eff.iloc[0]))}\)"
        elif row.method_name == "tt":
            method = "Tensor Train"
            rank_budget = r"\((1,7,7,1)\)"
        elif row.method_name == "ntdpl_p2":
            method = r"NTD-PL ($P=2$)"
        elif row.method_name == "ntdpl_p4":
            method = r"NTD-PL ($P=4$)"
        elif row.method_name == "ntdpl":
            method = r"\textbf{NTD-PL ($P=6$)}"
        lines.append(f"{method} & {rank_budget} & {params} & {rmse} & {nmse} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular*}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_outputs(frame: pd.DataFrame, outdir: Path, table_path: Path) -> None:
    frame = frame.sort_values(["scene_id", "method_name"]).reset_index(drop=True)
    summary = _summarize(frame)
    outdir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(outdir / "per_scene.csv", index=False)
    summary.to_csv(outdir / "summary.csv", index=False)
    _write_latex(summary, frame, table_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full-size CAVE main baseline comparison.")
    parser.add_argument("--scene-ids", default="1-15")
    parser.add_argument("--rank", type=_parse_rank, default=DEFAULT_RANK)
    parser.add_argument("--methods", default=",".join(METHOD_ORDER))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--table-path", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()

    scene_ids = _parse_int_set(args.scene_ids)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = sorted(set(methods) - set(METHOD_ORDER))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for scene_id in scene_ids:
        for method_name in methods:
            path = args.outdir / f"{_method_tag(method_name, args.rank, scene_id)}.json"
            if args.force or not path.exists():
                jobs.append((scene_id, method_name))

    if args.collect_only:
        print(f"Collect only: {len(jobs)} missing jobs.", flush=True)
    elif jobs:
        print(f"Running {len(jobs)} jobs with {args.workers} workers.", flush=True)
        with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
            futures = [
                executor.submit(_run_one, scene_id, method_name, args.rank, str(args.outdir), bool(args.force))
                for scene_id, method_name in jobs
            ]
            for idx, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                print(
                    f"[{idx:03d}/{len(futures):03d}] "
                    f"s{int(row['scene_id']):02d} {row['method_name']} "
                    f"RMSE={float(row['RMSE']):.5f} SAM={float(row['SAM']):.2f} "
                    f"time={float(row['fit_time_sec']):.1f}s",
                    flush=True,
                )
    else:
        print("All jobs already cached.", flush=True)

    frame = _collect(args.outdir, args.rank, scene_ids, methods)
    expected = len(scene_ids) * len(methods)
    if len(frame) < expected:
        print(f"Only collected {len(frame)}/{expected} rows; outputs reflect completed jobs.", flush=True)
    if not frame.empty:
        _write_outputs(frame, args.outdir, args.table_path)
        print(f"Wrote {args.outdir / 'per_scene.csv'}", flush=True)
        print(f"Wrote {args.outdir / 'summary.csv'}", flush=True)
        print(f"Wrote {args.table_path}", flush=True)


if __name__ == "__main__":
    main()
