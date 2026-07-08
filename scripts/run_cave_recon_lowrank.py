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

from src.utils.tensor_ranks import tucker_param_count


DEFAULT_EXP = "neurips-cave-recon-lowrank"
DEFAULT_RANKS = ((18, 18, 3),)
DEFAULT_SHAPE = (512, 512, 31)
METRIC_COLUMNS = ("RMSE", "SAM", "NMSE_dB")


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


def _parse_rank(text: str) -> tuple[int, int, int]:
    values = [int(part.strip()) for part in text.replace("x", ",").split(",") if part.strip()]
    if len(values) != 3:
        raise argparse.ArgumentTypeError(f"Expected three rank entries, got {text!r}.")
    return tuple(values)  # type: ignore[return-value]


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


def _rank_text(rank: tuple[int, int, int]) -> str:
    return f"({rank[0]},{rank[1]},{rank[2]})"


def _rank_override(rank: tuple[int, int, int]) -> str:
    return f"method.rank=[{rank[0]},{rank[1]},{rank[2]}]"


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


def _completed_keys(exp: str) -> set[tuple[int, tuple[int, int, int], str]]:
    keys: set[tuple[int, tuple[int, int, int], str]] = set()
    for run_dir in _run_dirs(exp):
        if not (run_dir / "eval.json").exists():
            continue
        cfg = OmegaConf.to_container(OmegaConf.load(run_dir / ".hydra" / "config.yaml"), resolve=True)
        if not isinstance(cfg, dict):
            continue
        if _cfg_value(cfg, "data._name") != "cave_hsi":
            continue
        method = str(_cfg_value(cfg, "method._name"))
        if method not in {"tucker", "ntdpl"}:
            continue
        rank = tuple(int(v) for v in _cfg_value(cfg, "method.rank"))  # type: ignore[arg-type]
        keys.add((int(_cfg_value(cfg, "data.id")), rank, method))
    return keys


def _command(exp: str, scene_id: int, rank: tuple[int, int, int], method: str, n_iter_max: int) -> list[str]:
    rank_slug = f"r{rank[0]}_{rank[1]}_{rank[2]}"
    scene_slug = f"s{scene_id:02d}"
    method_args = ["method=tucker"] if method == "tucker" else [
        "method=ntdpl",
        "method.p_max=6",
        "method.init=tucker",
        "method.use_continuation=true",
        "method.factor_normalize=true",
        "method.beta_update_method=moments_normal_eq",
    ]
    return [
        _project_python(),
        "run.py",
        "-m",
        f"exp={exp}",
        "exp_mode=benchmark",
        "data=cave_hsi",
        f"data.id={scene_id}",
        "data.target_shape=[512,512]",
        "data.crop_shape=null",
        "task=decompose",
        "task.log_level=0",
        "filter=bias-filter",
        "filter.normalize_method=max",
        *method_args,
        _rank_override(rank),
        f"method.n_iter_max={n_iter_max}",
        f"hydra.sweep.dir=artifacts/multirun/{exp}/benchmark/{method}_{rank_slug}_{scene_slug}",
        "hydra.sweep.subdir=.",
    ]


def _run_commands(commands: list[list[str]], max_parallel: int) -> None:
    if max_parallel <= 1:
        for command in commands:
            subprocess.run(command, cwd=PROJECT_ROOT, env=_child_env(), check=True)
        return
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = [executor.submit(subprocess.run, command, cwd=PROJECT_ROOT, env=_child_env(), check=True) for command in commands]
        for future in as_completed(futures):
            future.result()


def _collect_rows(exp: str) -> pd.DataFrame:
    rows: list[dict] = []
    for run_dir in _run_dirs(exp):
        eval_path = run_dir / "eval.json"
        cfg_path = run_dir / ".hydra" / "config.yaml"
        if not eval_path.exists():
            continue
        cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
        if not isinstance(cfg, dict):
            continue
        method = str(_cfg_value(cfg, "method._name"))
        if _cfg_value(cfg, "data._name") != "cave_hsi" or method not in {"tucker", "ntdpl"}:
            continue
        rank = tuple(int(v) for v in _cfg_value(cfg, "method.rank"))  # type: ignore[arg-type]
        metrics = json.loads(eval_path.read_text(encoding="utf-8"))
        params = tucker_param_count(DEFAULT_SHAPE, rank) + (7 if method == "ntdpl" else 0)
        rows.append(
            {
                "scene_id": int(_cfg_value(cfg, "data.id")),
                "rank": _rank_text(rank),
                "method": "Tucker" if method == "tucker" else "NTD-PL",
                "params": params,
                "run_dir": str(run_dir),
                **{key: float(metrics[key]) for key in METRIC_COLUMNS if key in metrics},
                "fit_time_sec": float(metrics.get("fit_time_sec", float("nan"))),
            }
        )
    return pd.DataFrame(rows)


def _write_outputs(frame: pd.DataFrame, out_prefix: Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    frame = frame.sort_values(["rank", "method", "scene_id"]).reset_index(drop=True)
    frame.to_csv(out_prefix.with_suffix(".per_scene.csv"), index=False)

    summary = (
        frame.groupby(["rank", "method", "params"], as_index=False)
        .agg(
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            SAM_mean=("SAM", "mean"),
            SAM_std=("SAM", "std"),
            NMSE_dB_mean=("NMSE_dB", "mean"),
            NMSE_dB_std=("NMSE_dB", "std"),
            fit_time_mean=("fit_time_sec", "mean"),
            n=("scene_id", "nunique"),
        )
        .sort_values(["rank", "method"])
        .reset_index(drop=True)
    )
    summary.to_csv(out_prefix.with_suffix(".summary.csv"), index=False)

    lines = [
        r"\begin{tabular}{@{}l r cc cc@{}}",
        r"\toprule",
        r"Rank & Params & \multicolumn{2}{c}{RMSE$\downarrow$} & \multicolumn{2}{c}{SAM$\downarrow$} \\",
        r"\cmidrule(lr){3-4}\cmidrule(l){5-6}",
        r" & & Tucker & NTD-PL & Tucker & NTD-PL \\",
        r"\midrule",
    ]
    for rank, group in summary.groupby("rank", sort=False):
        lookup = {row.method: row for row in group.itertuples(index=False)}
        if "Tucker" not in lookup or "NTD-PL" not in lookup:
            continue
        tucker = lookup["Tucker"]
        ntdpl = lookup["NTD-PL"]
        lines.append(
            f"${rank}$ & {int(round(ntdpl.params / 1000))}k & "
            f"{tucker.RMSE_mean:.4f} & \\textbf{{{ntdpl.RMSE_mean:.4f}}} & "
            f"{tucker.SAM_mean:.2f} & \\textbf{{{ntdpl.SAM_mean:.2f}}} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    out_prefix.with_suffix(".tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and summarize low-rank CAVE reconstruction.")
    parser.add_argument("--exp", default=DEFAULT_EXP)
    parser.add_argument("--scene-ids", default="1-15")
    parser.add_argument("--rank", action="append", type=_parse_rank, dest="ranks")
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--n-iter-max", type=int, default=300)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--out-prefix", default="papers/neurips/tables/cave_reconstruction_lowrank")
    args = parser.parse_args()

    scene_ids = _parse_int_set(args.scene_ids)
    ranks = tuple(args.ranks or DEFAULT_RANKS)
    methods = ("tucker", "ntdpl")
    completed = _completed_keys(args.exp)
    commands = [
        _command(args.exp, scene_id, rank, method, args.n_iter_max)
        for rank in ranks
        for scene_id in scene_ids
        for method in methods
        if (scene_id, rank, method) not in completed
    ]
    if args.collect_only:
        print(f"Collect-only mode: {len(completed)} completed jobs available; {len(commands)} jobs missing.")
    else:
        print(f"Running {len(commands)} missing jobs for {len(ranks)} ranks x {len(scene_ids)} scenes x {len(methods)} methods.")
        _run_commands(commands, max(1, args.max_parallel))

    frame = _collect_rows(args.exp)
    if frame.empty:
        print(f"No completed runs found for exp={args.exp!r}.")
        return
    _write_outputs(frame, PROJECT_ROOT / args.out_prefix)
    print(f"Wrote {args.out_prefix}.per_scene.csv, .summary.csv, and .tex")


if __name__ == "__main__":
    main()
