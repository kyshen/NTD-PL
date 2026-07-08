from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_EXP = "neurips-kodak-completion"
DEFAULT_METHODS = ("tucker", "ntdpl")
DEFAULT_RANK = (24, 24, 2)
METHOD_LABELS = {
    "tucker": "Tucker",
    "ntdpl": "NTD-PL",
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


def _run_dirs(exp: str) -> list[Path]:
    root = PROJECT_ROOT / "artifacts" / "multirun" / exp
    if not root.exists():
        return []
    return [path.parent.parent for path in root.rglob(".hydra/config.yaml")]


def _cfg_value(cfg: dict, key: str) -> object | None:
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


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _completed_keys(exp: str) -> set[tuple[int, int, float, str, tuple[int, int, int], tuple[int, int]]]:
    keys: set[tuple[int, int, float, str, tuple[int, int, int], tuple[int, int]]] = set()
    for run_dir in _run_dirs(exp):
        if not (run_dir / "eval.json").exists():
            continue
        cfg = OmegaConf.to_container(OmegaConf.load(run_dir / ".hydra" / "config.yaml"), resolve=True)
        if not isinstance(cfg, dict):
            continue
        if _cfg_value(cfg, "data._name") != "kodak":
            continue
        if _cfg_value(cfg, "task._name") != "random-missing-completion":
            continue
        image_id = int(_cfg_value(cfg, "data.id"))
        seed = int(_cfg_value(cfg, "task.seed"))
        missing_rate = float(_cfg_value(cfg, "task.missing_rate"))
        method = str(_cfg_value(cfg, "method._name"))
        rank = tuple(int(v) for v in _normalize(_cfg_value(cfg, "method.rank")))  # type: ignore[arg-type]
        target_shape = tuple(int(v) for v in _normalize(_cfg_value(cfg, "data.target_shape")))  # type: ignore[arg-type]
        keys.add((image_id, seed, missing_rate, method, rank, target_shape))  # type: ignore[arg-type]
    return keys


def _rank_override(rank: tuple[int, int, int]) -> str:
    return f"method.rank=[{rank[0]},{rank[1]},{rank[2]}]"


def _command(
    exp: str,
    *,
    image_id: int,
    seed: int,
    missing_rate: float,
    method: str,
    rank: tuple[int, int, int],
    target_shape: tuple[int, int],
    n_iter_max: int,
    p_max: int,
) -> list[str]:
    exp_mode = "benchmark" if method == "tucker" else "run"
    run_slug = f"kodim{image_id:02d}_seed{seed}_{method}"
    command = [
        _project_python(),
        "run.py",
        "-m",
        f"exp={exp}",
        f"exp_mode={exp_mode}",
        "data=kodak",
        f"data.id={image_id}",
        f"data.target_shape=[{target_shape[0]},{target_shape[1]}]",
        "task=random-missing-completion",
        "task.log_level=1",
        f"task.seed={seed}",
        f"task.missing_rate={missing_rate}",
        "filter=bias-filter",
        "filter.normalize_method=max",
        f"method={method}",
        _rank_override(rank),
        f"method.n_iter_max={n_iter_max}",
        f"hydra.sweep.dir=artifacts/multirun/{exp}/{exp_mode}/{run_slug}",
        "hydra.sweep.subdir=.",
    ]
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
    rows: list[dict] = []
    for run_dir in _run_dirs(exp):
        eval_path = run_dir / "eval.json"
        if not eval_path.exists():
            continue
        cfg = OmegaConf.to_container(OmegaConf.load(run_dir / ".hydra" / "config.yaml"), resolve=True)
        if not isinstance(cfg, dict):
            continue
        if _cfg_value(cfg, "data._name") != "kodak":
            continue
        if _cfg_value(cfg, "task._name") != "random-missing-completion":
            continue
        rank = tuple(int(v) for v in _normalize(_cfg_value(cfg, "method.rank")))  # type: ignore[arg-type]
        target_shape = tuple(int(v) for v in _normalize(_cfg_value(cfg, "data.target_shape")))  # type: ignore[arg-type]
        metrics = _read_json(eval_path)
        rows.append(
            {
                "image_id": int(_cfg_value(cfg, "data.id")),
                "seed": int(_cfg_value(cfg, "task.seed")),
                "missing_rate": float(_cfg_value(cfg, "task.missing_rate")),
                "method": str(_cfg_value(cfg, "method._name")),
                "rank": f"({rank[0]},{rank[1]},{rank[2]})",
                "target_shape": f"({target_shape[0]},{target_shape[1]})",
                "observed_rate": float(metrics["observed_rate"]),
                "RMSE_missing": float(metrics["RMSE_missing"]),
                "PSNR_missing": float(metrics.get("PSNR_missing", float("nan"))),
                "SSIM_missing": float(metrics.get("SSIM_missing", float("nan"))),
                "SAM_missing": float(metrics["SAM_missing"]),
                "NMSE_dB_missing": float(metrics.get("NMSE_dB_missing", float("nan"))),
                "RMSE_all": float(metrics["RMSE_all"]),
                "PSNR_all": float(metrics.get("PSNR_all", float("nan"))),
                "SSIM_all": float(metrics.get("SSIM_all", float("nan"))),
                "fit_time_sec": float(metrics.get("fit_time_sec", float("nan"))),
                "run_dir": str(run_dir),
            }
        )
    return pd.DataFrame(rows)


def _paired_win_count(frame: pd.DataFrame, metric: str, *, higher_is_better: bool) -> int:
    pivot = frame.pivot_table(index=["image_id", "seed"], columns="method", values=metric, aggfunc="first")
    if not {"tucker", "ntdpl"}.issubset(pivot.columns):
        return 0
    if higher_is_better:
        return int((pivot["ntdpl"] > pivot["tucker"]).sum())
    return int((pivot["ntdpl"] < pivot["tucker"]).sum())


def _write_outputs(frame: pd.DataFrame, out_prefix: Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    frame = frame.sort_values(["image_id", "seed", "method"]).reset_index(drop=True)
    frame.to_csv(out_prefix.with_suffix(".per_run.csv"), index=False)

    summary = (
        frame.groupby(["missing_rate", "method"], as_index=False)
        .agg(
            RMSE_missing_mean=("RMSE_missing", "mean"),
            RMSE_missing_std=("RMSE_missing", "std"),
            PSNR_missing_mean=("PSNR_missing", "mean"),
            PSNR_missing_std=("PSNR_missing", "std"),
            SSIM_missing_mean=("SSIM_missing", "mean"),
            SSIM_missing_std=("SSIM_missing", "std"),
            SAM_missing_mean=("SAM_missing", "mean"),
            SAM_missing_std=("SAM_missing", "std"),
            fit_time_mean=("fit_time_sec", "mean"),
            n_runs=("image_id", "count"),
            n_images=("image_id", "nunique"),
        )
        .reset_index(drop=True)
    )
    for column in summary.columns:
        if column.endswith("_std"):
            summary[column] = summary[column].fillna(0.0)
    method_order = {name: idx for idx, name in enumerate(DEFAULT_METHODS)}
    summary["method_order"] = summary["method"].map(method_order).fillna(99)
    summary = summary.sort_values(["missing_rate", "method_order"]).drop(columns=["method_order"])
    summary["rmse_wins"] = _paired_win_count(frame, "RMSE_missing", higher_is_better=False)
    summary["psnr_wins"] = _paired_win_count(frame, "PSNR_missing", higher_is_better=True)
    summary["ssim_wins"] = _paired_win_count(frame, "SSIM_missing", higher_is_better=True)
    summary.to_csv(out_prefix.with_suffix(".summary.csv"), index=False)

    lines = [
        r"\begin{tabular}{@{}l c c c c c@{}}",
        r"\toprule",
        r"Method & RMSE*$\downarrow$ & PSNR*$\uparrow$ & SSIM*$\uparrow$ & SAM*$\downarrow$ & Images\\",
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"{METHOD_LABELS.get(row.method, row.method)} & "
            f"{row.RMSE_missing_mean:.4f} $\\pm$ {row.RMSE_missing_std:.4f} & "
            f"{row.PSNR_missing_mean:.2f} $\\pm$ {row.PSNR_missing_std:.2f} & "
            f"{row.SSIM_missing_mean:.3f} $\\pm$ {row.SSIM_missing_std:.3f} & "
            f"{row.SAM_missing_mean:.2f} $\\pm$ {row.SAM_missing_std:.2f} & "
            f"{int(row.n_images)}\\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    out_prefix.with_suffix(".tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Kodak random-missing image completion experiments.")
    parser.add_argument("--exp", default=DEFAULT_EXP)
    parser.add_argument("--image-ids", default="1-24")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--missing-rate", type=float, default=0.5)
    parser.add_argument("--rank", type=_parse_rank, default=DEFAULT_RANK)
    parser.add_argument("--target-shape", type=_parse_shape2, default=(128, 128))
    parser.add_argument("--n-iter-max", type=int, default=120)
    parser.add_argument("--p-max", type=int, default=4)
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--out-prefix", default="papers/neurips/tables/kodak_completion")
    args = parser.parse_args()

    image_ids = _parse_int_set(args.image_ids)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    completed = _completed_keys(args.exp)
    commands = [
        _command(
            args.exp,
            image_id=image_id,
            seed=args.seed,
            missing_rate=args.missing_rate,
            method=method,
            rank=args.rank,
            target_shape=args.target_shape,
            n_iter_max=args.n_iter_max,
            p_max=args.p_max,
        )
        for image_id in image_ids
        for method in methods
        if (image_id, args.seed, float(args.missing_rate), method, args.rank, args.target_shape) not in completed
    ]

    if args.collect_only:
        print(f"Collect-only mode: {len(completed)} completed jobs available; {len(commands)} jobs missing.")
    else:
        print(f"Running {len(commands)} missing jobs for {len(image_ids)} images x {len(methods)} methods.")
        _run_commands(commands, max(1, args.max_parallel))

    frame = _collect_rows(args.exp)
    if frame.empty:
        print(f"No completed runs found for exp={args.exp!r}.")
        return
    rank_text = f"({args.rank[0]},{args.rank[1]},{args.rank[2]})"
    shape_text = f"({args.target_shape[0]},{args.target_shape[1]})"
    frame = frame.loc[
        frame["image_id"].isin(image_ids)
        & frame["method"].isin(methods)
        & frame["seed"].eq(args.seed)
        & frame["missing_rate"].eq(float(args.missing_rate))
        & frame["rank"].eq(rank_text)
        & frame["target_shape"].eq(shape_text)
    ].copy()
    if frame.empty:
        print("No completed runs matched the requested filter.")
        return
    _write_outputs(frame, PROJECT_ROOT / args.out_prefix)
    print(f"Wrote {args.out_prefix}.per_run.csv, .summary.csv, and .tex")


if __name__ == "__main__":
    main()
