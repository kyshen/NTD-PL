from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_EXP = "neurips-lowrank-core"
DEFAULT_METHODS = ("tucker", "ntdpl", "softimpute")
METHOD_LABELS = {
    "tucker": "Tucker",
    "ntdpl": "NTD-PL",
    "softimpute": "SoftImpute",
    "cp": "CP",
}
LOWER_IS_BETTER = {
    "RMSE_missing": True,
    "SAM_missing": True,
    "NMSE_dB_missing": True,
    "ERGAS_missing": True,
}
HIGHER_IS_BETTER = {
    "PSNR_missing": True,
    "SSIM_missing": True,
}


def _project_python() -> str:
    venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
            "TBB_NUM_THREADS": "1",
        }
    )
    return env


def _parse_int_set(text: str) -> list[int]:
    items: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(v.strip()) for v in part.split("-", 1)]
            items.update(range(start, end + 1))
        else:
            items.add(int(part))
    return sorted(items)


def _parse_float_set(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def _parse_rank(text: str) -> tuple[int, int, int]:
    values = [int(part.strip()) for part in text.replace("x", ",").split(",") if part.strip()]
    if len(values) != 3:
        raise argparse.ArgumentTypeError(f"Expected three rank entries, got {text!r}.")
    return tuple(values)  # type: ignore[return-value]


def _parse_shape2(text: str) -> tuple[int, int]:
    values = [int(part.strip()) for part in text.replace("x", ",").split(",") if part.strip()]
    if len(values) != 2:
        raise argparse.ArgumentTypeError(f"Expected two spatial shape entries, got {text!r}.")
    return tuple(values)  # type: ignore[return-value]


def _rank_text(rank: tuple[int, int, int]) -> str:
    return f"({rank[0]},{rank[1]},{rank[2]})"


def _rank_override(rank: tuple[int, int, int]) -> str:
    return f"method.rank=[{rank[0]},{rank[1]},{rank[2]}]"


def _run_dirs(exp: str) -> list[Path]:
    root = PROJECT_ROOT / "multirun" / exp
    if not root.exists():
        return []
    return [path.parent.parent for path in root.rglob(".hydra/config.yaml")]


def _cfg_value(cfg: dict[str, Any], key: str) -> object | None:
    value: object = cfg
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    if isinstance(value, dict) and "_name" in value:
        return value["_name"]
    return value


def _normalize(value: object) -> object:
    if isinstance(value, list):
        return tuple(_normalize(item) for item in value)
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _completed_keys(exp: str) -> set[tuple[Any, ...]]:
    keys: set[tuple[Any, ...]] = set()
    for run_dir in _run_dirs(exp):
        if not (run_dir / "eval.json").exists():
            continue
        cfg = OmegaConf.to_container(OmegaConf.load(run_dir / ".hydra" / "config.yaml"), resolve=True)
        if not isinstance(cfg, dict):
            continue
        data_name = str(_cfg_value(cfg, "data._name"))
        task_name = str(_cfg_value(cfg, "task._name"))
        method = str(_cfg_value(cfg, "method._name"))
        item_id = int(_cfg_value(cfg, "data.id"))
        seed = int(_cfg_value(cfg, "task.seed"))
        missing_rate = float(_cfg_value(cfg, "task.missing_rate"))
        rank = tuple(int(v) for v in _normalize(_cfg_value(cfg, "method.rank")))  # type: ignore[arg-type]
        target_shape = tuple(int(v) for v in _normalize(_cfg_value(cfg, "data.target_shape")))  # type: ignore[arg-type]
        pattern = str(_cfg_value(cfg, "task.pattern")) if task_name == "structured-missing-completion" else "random"
        keys.add((data_name, task_name, item_id, seed, missing_rate, pattern, method, rank, target_shape))
    return keys


def _data_overrides(dataset: str, item_id: int, target_shape: tuple[int, int]) -> list[str]:
    if dataset == "cave_hsi":
        return [
            "data=cave_hsi",
            f"data.id={item_id}",
            f"data.target_shape=[{target_shape[0]},{target_shape[1]}]",
            "data.crop_shape=null",
        ]
    if dataset == "kodak":
        return [
            "data=kodak",
            f"data.id={item_id}",
            f"data.target_shape=[{target_shape[0]},{target_shape[1]}]",
        ]
    if dataset == "cbsd":
        return [
            "data=cbsd",
            f"data.id={item_id}",
            f"data.target_shape=[{target_shape[0]},{target_shape[1]}]",
        ]
    raise ValueError(f"Unsupported dataset: {dataset}")


def _command(
    exp: str,
    *,
    dataset: str,
    item_id: int,
    seed: int,
    missing_rate: float,
    protocol: str,
    method: str,
    rank: tuple[int, int, int],
    target_shape: tuple[int, int],
    n_iter_max: int,
    p_max: int,
) -> list[str]:
    task_name = "structured-missing-completion" if protocol == "block" else "random-missing-completion"
    exp_mode = "run" if method == "ntdpl" else "benchmark"
    rank_slug = f"r{rank[0]}_{rank[1]}_{rank[2]}"
    run_slug = f"{dataset}_{protocol}_id{item_id:02d}_seed{seed}_mr{missing_rate:g}_{method}_{rank_slug}"
    command = [
        _project_python(),
        "run.py",
        "-m",
        f"exp={exp}",
        f"exp_mode={exp_mode}",
        *_data_overrides(dataset, item_id, target_shape),
        f"task={task_name}",
        "task.log_level=1",
        f"task.seed={seed}",
        f"task.missing_rate={missing_rate}",
        "filter=bias-filter",
        "filter.normalize_method=max",
        f"method={method}",
        _rank_override(rank),
        f"method.n_iter_max={n_iter_max}",
        f"hydra.sweep.dir=multirun/{exp}/{exp_mode}/{run_slug}",
        "hydra.sweep.subdir=.",
    ]
    if protocol == "block":
        command.append("task.pattern=block")
    if method == "ntdpl":
        command.extend(
            [
                f"method.p_max={p_max}",
                "method.init=tucker",
                "method.use_continuation=true",
                "method.factor_normalize=true",
                "method.beta_update_method=ridge_lstsq",
            ]
        )
    if method == "softimpute":
        command.extend(["method.outer_n_iter_max=8", "method.init_fill=mean"])
    if method == "cp":
        command.extend(["method.init_method=svd", "method.normalize_factors=true"])
    return command


def _run_commands(commands: list[list[str]], max_parallel: int) -> None:
    if not commands:
        return
    if max_parallel <= 1:
        for command in commands:
            subprocess.run(command, cwd=PROJECT_ROOT, env=_child_env(), check=True)
        return
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = [
            executor.submit(subprocess.run, command, cwd=PROJECT_ROOT, env=_child_env(), check=True)
            for command in commands
        ]
        for future in as_completed(futures):
            future.result()


def _collect_rows(exp: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_dir in _run_dirs(exp):
        eval_path = run_dir / "eval.json"
        if not eval_path.exists():
            continue
        cfg = OmegaConf.to_container(OmegaConf.load(run_dir / ".hydra" / "config.yaml"), resolve=True)
        if not isinstance(cfg, dict):
            continue
        metrics = _read_json(eval_path)
        task_name = str(_cfg_value(cfg, "task._name"))
        rank = tuple(int(v) for v in _normalize(_cfg_value(cfg, "method.rank")))  # type: ignore[arg-type]
        target_shape = tuple(int(v) for v in _normalize(_cfg_value(cfg, "data.target_shape")))  # type: ignore[arg-type]
        row = {
            "dataset": str(_cfg_value(cfg, "data._name")),
            "item_id": int(_cfg_value(cfg, "data.id")),
            "task": task_name,
            "protocol": str(_cfg_value(cfg, "task.pattern")) if task_name == "structured-missing-completion" else "random",
            "seed": int(_cfg_value(cfg, "task.seed")),
            "missing_rate": float(_cfg_value(cfg, "task.missing_rate")),
            "method": str(_cfg_value(cfg, "method._name")),
            "rank": _rank_text(rank),  # type: ignore[arg-type]
            "target_shape": f"({target_shape[0]},{target_shape[1]})",
            "fit_time_sec": float(metrics.get("fit_time_sec", float("nan"))),
            "run_dir": str(run_dir),
        }
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                row[key] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_win_count(frame: pd.DataFrame, method: str, metric: str, *, higher_is_better: bool) -> int:
    index_cols = ["dataset", "protocol", "rank", "missing_rate", "seed", "item_id"]
    pivot = frame.pivot_table(index=index_cols, columns="method", values=metric, aggfunc="first")
    if method not in pivot.columns or "tucker" not in pivot.columns:
        return 0
    if higher_is_better:
        return int((pivot[method] > pivot["tucker"]).sum())
    return int((pivot[method] < pivot["tucker"]).sum())


def _group_win_count(frame: pd.DataFrame, row: pd.Series, metric: str, *, higher_is_better: bool) -> int:
    group = frame.loc[
        frame["dataset"].eq(row["dataset"])
        & frame["protocol"].eq(row["protocol"])
        & frame["rank"].eq(row["rank"])
        & frame["target_shape"].eq(row["target_shape"])
        & frame["missing_rate"].eq(float(row["missing_rate"]))
    ].copy()
    return _paired_win_count(group, str(row["method"]), metric, higher_is_better=higher_is_better)


def _write_outputs(frame: pd.DataFrame, out_prefix: Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    frame = frame.sort_values(["dataset", "protocol", "rank", "missing_rate", "seed", "item_id", "method"])
    frame.to_csv(out_prefix.with_suffix(".per_run.csv"), index=False)

    agg_spec: dict[str, tuple[str, str]] = {
        "fit_time_mean": ("fit_time_sec", "mean"),
        "n_runs": ("item_id", "count"),
        "n_items": ("item_id", "nunique"),
    }
    for metric in [*LOWER_IS_BETTER.keys(), *HIGHER_IS_BETTER.keys()]:
        if metric in frame.columns:
            agg_spec[f"{metric}_mean"] = (metric, "mean")
            agg_spec[f"{metric}_std"] = (metric, "std")

    summary = (
        frame.groupby(["dataset", "protocol", "rank", "target_shape", "missing_rate", "method"], as_index=False)
        .agg(**agg_spec)
        .reset_index(drop=True)
    )
    for column in summary.columns:
        if column.endswith("_std"):
            summary[column] = summary[column].fillna(0.0)
    if "RMSE_missing" in frame.columns:
        summary["rmse_wins_vs_tucker"] = summary.apply(
            lambda row: _group_win_count(frame, row, "RMSE_missing", higher_is_better=False),
            axis=1,
        )
    if "SAM_missing" in frame.columns:
        summary["sam_wins_vs_tucker"] = summary.apply(
            lambda row: _group_win_count(frame, row, "SAM_missing", higher_is_better=False),
            axis=1,
        )
    method_order = {name: idx for idx, name in enumerate(DEFAULT_METHODS)}
    summary["method_order"] = summary["method"].map(method_order).fillna(99)
    summary = summary.sort_values(["dataset", "protocol", "rank", "missing_rate", "method_order"]).drop(
        columns=["method_order"]
    )
    summary.to_csv(out_prefix.with_suffix(".summary.csv"), index=False)
    out_prefix.with_suffix(".tex").write_text(_summary_to_latex(summary), encoding="utf-8")
    paired = _paired_gain_summary(frame)
    if not paired.empty:
        paired.to_csv(out_prefix.with_suffix(".paired.csv"), index=False)
        out_prefix.with_suffix(".paired.tex").write_text(_paired_to_latex(paired), encoding="utf-8")


def _fmt_pm(row: pd.Series, metric: str, digits: int) -> str:
    mean = float(row.get(f"{metric}_mean", float("nan")))
    std = float(row.get(f"{metric}_std", float("nan")))
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def _summary_to_latex(summary: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{@{}l l c c c c c@{}}",
        r"\toprule",
        r"Protocol & Method & Rank & $\rho$ & RMSE*$\downarrow$ & SAM*$\downarrow$ & Runs\\",
        r"\midrule",
    ]
    for row in summary.to_dict("records"):
        series = pd.Series(row)
        lines.append(
            " & ".join(
                [
                    str(row["protocol"]),
                    METHOD_LABELS.get(str(row["method"]), str(row["method"])),
                    str(row["rank"]),
                    f"{float(row['missing_rate']):.1f}",
                    _fmt_pm(series, "RMSE_missing", 4),
                    _fmt_pm(series, "SAM_missing", 2),
                    str(int(row["n_runs"])),
                ]
            )
            + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def _paired_gain_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["dataset", "protocol", "rank", "target_shape", "missing_rate"]
    for keys, group in frame.groupby(group_cols, sort=True):
        pivot = group.pivot_table(
            index=["item_id", "seed"],
            columns="method",
            values=["RMSE_missing", "SAM_missing", "PSNR_missing", "SSIM_missing"],
            aggfunc="first",
        )
        if ("RMSE_missing", "tucker") not in pivot.columns:
            continue
        for method in sorted(group["method"].unique().tolist()):
            if method == "tucker" or ("RMSE_missing", method) not in pivot.columns:
                continue
            row: dict[str, Any] = dict(zip(group_cols, keys, strict=True))
            row["method"] = method
            row["n_pairs"] = int(pivot[[("RMSE_missing", "tucker"), ("RMSE_missing", method)]].dropna().shape[0])
            for metric in ("RMSE_missing", "SAM_missing"):
                if (metric, method) not in pivot.columns or (metric, "tucker") not in pivot.columns:
                    continue
                base = pivot[(metric, "tucker")].astype(float)
                value = pivot[(metric, method)].astype(float)
                gain = (base - value) / base.clip(lower=1e-12)
                row[f"{metric}_gain_mean_pct"] = float(100.0 * gain.mean())
                row[f"{metric}_gain_median_pct"] = float(100.0 * gain.median())
                row[f"{metric}_wins"] = int((gain > 0.0).sum())
                row[f"{metric}_losses"] = int((gain < 0.0).sum())
                row[f"{metric}_min_gain_pct"] = float(100.0 * gain.min())
                row[f"{metric}_max_gain_pct"] = float(100.0 * gain.max())
            for metric in ("PSNR_missing", "SSIM_missing"):
                if (metric, method) not in pivot.columns or (metric, "tucker") not in pivot.columns:
                    continue
                base = pivot[(metric, "tucker")].astype(float)
                value = pivot[(metric, method)].astype(float)
                delta = value - base
                row[f"{metric}_delta_mean"] = float(delta.mean())
                row[f"{metric}_delta_median"] = float(delta.median())
                row[f"{metric}_wins"] = int((delta > 0.0).sum())
            rows.append(row)
    return pd.DataFrame(rows)


def _paired_to_latex(paired: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{@{}l l c c c c c@{}}",
        r"\toprule",
        r"Protocol & Method & Rank & $\rho$ & RMSE gain & SAM gain & Wins\\",
        r"\midrule",
    ]
    for row in paired.to_dict("records"):
        n_pairs = int(row["n_pairs"])
        lines.append(
            " & ".join(
                [
                    str(row["protocol"]),
                    METHOD_LABELS.get(str(row["method"]), str(row["method"])),
                    str(row["rank"]),
                    f"{float(row['missing_rate']):.1f}",
                    f"{float(row['RMSE_missing_gain_median_pct']):.1f}\\%",
                    f"{float(row['SAM_missing_gain_median_pct']):.1f}\\%",
                    f"{int(row['RMSE_missing_wins'])}/{n_pairs} RMSE, "
                    f"{int(row['SAM_missing_wins'])}/{n_pairs} SAM",
                ]
            )
            + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a low-rank NeurIPS core completion sweep.")
    parser.add_argument("--exp", default=DEFAULT_EXP)
    parser.add_argument("--dataset", choices=("cave_hsi", "kodak", "cbsd"), default="cave_hsi")
    parser.add_argument("--item-ids", default="1-15")
    parser.add_argument("--protocols", default="random,block")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--missing-rates", default="0.3,0.5,0.7")
    parser.add_argument("--rank", action="append", type=_parse_rank, dest="ranks")
    parser.add_argument("--target-shape", type=_parse_shape2, default=(128, 128))
    parser.add_argument("--n-iter-max", type=int, default=180)
    parser.add_argument("--p-max", type=int, default=4)
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--out-prefix", default="neurips/tables/lowrank_core_sweep")
    args = parser.parse_args()

    default_ranks = ((12, 12, 3), (16, 16, 3), (20, 20, 4), (24, 24, 4))
    ranks = tuple(args.ranks or default_ranks)
    item_ids = _parse_int_set(args.item_ids)
    seeds = _parse_int_set(args.seeds)
    missing_rates = _parse_float_set(args.missing_rates)
    protocols = [item.strip() for item in args.protocols.split(",") if item.strip()]
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]

    completed = _completed_keys(args.exp)
    commands = []
    for protocol in protocols:
        if protocol not in {"random", "block"}:
            raise ValueError(f"Unsupported protocol {protocol!r}; use random or block.")
        task_name = "structured-missing-completion" if protocol == "block" else "random-missing-completion"
        for rank in ranks:
            for missing_rate in missing_rates:
                for seed in seeds:
                    for item_id in item_ids:
                        for method in methods:
                            key = (
                                args.dataset,
                                task_name,
                                item_id,
                                seed,
                                float(missing_rate),
                                "block" if protocol == "block" else "random",
                                method,
                                rank,
                                args.target_shape,
                            )
                            if key in completed:
                                continue
                            commands.append(
                                _command(
                                    args.exp,
                                    dataset=args.dataset,
                                    item_id=item_id,
                                    seed=seed,
                                    missing_rate=missing_rate,
                                    protocol=protocol,
                                    method=method,
                                    rank=rank,
                                    target_shape=args.target_shape,
                                    n_iter_max=args.n_iter_max,
                                    p_max=args.p_max,
                                )
                            )

    if args.collect_only:
        print(f"Collect-only mode: {len(completed)} completed jobs available; {len(commands)} jobs missing.")
    else:
        print(f"Running {len(commands)} missing jobs.")
        _run_commands(commands, max(1, args.max_parallel))

    frame = _collect_rows(args.exp)
    if frame.empty:
        print(f"No completed runs found for exp={args.exp!r}.")
        return

    rank_filter = {_rank_text(rank) for rank in ranks}
    shape_filter = f"({args.target_shape[0]},{args.target_shape[1]})"
    frame = frame.loc[
        frame["dataset"].eq(args.dataset)
        & frame["item_id"].isin(item_ids)
        & frame["protocol"].isin(protocols)
        & frame["method"].isin(methods)
        & frame["seed"].isin(seeds)
        & frame["missing_rate"].isin(missing_rates)
        & frame["rank"].isin(rank_filter)
        & frame["target_shape"].eq(shape_filter)
    ].copy()
    if frame.empty:
        print("No completed runs matched the requested filter.")
        return
    _write_outputs(frame, PROJECT_ROOT / args.out_prefix)
    print(f"Wrote {args.out_prefix}.per_run.csv, .summary.csv, and .tex")


if __name__ == "__main__":
    main()
