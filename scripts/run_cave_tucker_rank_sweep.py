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


DEFAULT_EXP = "neurips-cave-tucker-rank-sweep"
DEFAULT_RANKS = (
    (19, 19, 3),
    (20, 20, 3),
    (21, 21, 3),
    (22, 22, 3),
    (23, 23, 3),
    (24, 24, 3),
    (25, 25, 3),
    (26, 26, 3),
    (27, 27, 3),
    (28, 28, 3),
    (29, 29, 3),
    (30, 30, 3),
    (24, 24, 4),
    (25, 25, 4),
    (26, 26, 4),
    (27, 27, 4),
    (28, 28, 4),
    (29, 29, 4),
    (30, 30, 4),
    (31, 31, 4),
    (32, 32, 4),
    (33, 33, 4),
    (34, 34, 4),
    (35, 35, 4),
    (36, 36, 4),
    (37, 37, 4),
    (38, 38, 4),
    (39, 39, 4),
    (40, 40, 4),
    (41, 41, 4),
    (42, 42, 4),
    (43, 43, 4),
    (44, 44, 4),
    (45, 45, 4),
    (46, 46, 4),
    (47, 47, 4),
    (48, 48, 4),
    (49, 49, 4),
    (50, 50, 4),
)
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


def _rank_override(rank: tuple[int, int, int]) -> str:
    return f"method.rank=[{rank[0]},{rank[1]},{rank[2]}]"


def _rank_text(rank: tuple[int, int, int]) -> str:
    return f"({rank[0]},{rank[1]},{rank[2]})"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _completed_keys(exp: str) -> set[tuple[int, tuple[int, int, int]]]:
    keys: set[tuple[int, tuple[int, int, int]]] = set()
    for run_dir in _run_dirs(exp):
        if not (run_dir / "eval.json").exists():
            continue
        cfg = OmegaConf.to_container(OmegaConf.load(run_dir / ".hydra" / "config.yaml"), resolve=True)
        if not isinstance(cfg, dict):
            continue
        if _cfg_value(cfg, "data._name") != "cave_hsi":
            continue
        if _cfg_value(cfg, "method._name") != "tucker":
            continue
        scene_id = int(_cfg_value(cfg, "data.id"))
        rank = tuple(int(v) for v in _normalize(_cfg_value(cfg, "method.rank")))  # type: ignore[arg-type]
        keys.add((scene_id, rank))  # type: ignore[arg-type]
    return keys


def _command(exp: str, scene_id: int, rank: tuple[int, int, int], n_iter_max: int) -> list[str]:
    rank_slug = f"r{rank[0]}_{rank[1]}_{rank[2]}"
    scene_slug = f"s{scene_id:02d}"
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
        "method=tucker",
        _rank_override(rank),
        f"method.n_iter_max={n_iter_max}",
        f"hydra.sweep.dir=artifacts/multirun/{exp}/benchmark/{rank_slug}_{scene_slug}",
        "hydra.sweep.subdir=.",
    ]


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
        cfg_path = run_dir / ".hydra" / "config.yaml"
        if not eval_path.exists():
            continue
        cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
        if not isinstance(cfg, dict):
            continue
        if _cfg_value(cfg, "data._name") != "cave_hsi" or _cfg_value(cfg, "method._name") != "tucker":
            continue
        rank = tuple(int(v) for v in _normalize(_cfg_value(cfg, "method.rank")))  # type: ignore[arg-type]
        metrics = _read_json(eval_path)
        rows.append(
            {
                "scene_id": int(_cfg_value(cfg, "data.id")),
                "rank": _rank_text(rank),  # type: ignore[arg-type]
                "rank_r1": rank[0],
                "rank_r2": rank[1],
                "rank_r3": rank[2],
                "params": tucker_param_count(DEFAULT_SHAPE, rank),  # type: ignore[arg-type]
                "run_dir": str(run_dir),
                **{key: float(metrics[key]) for key in METRIC_COLUMNS if key in metrics},
                "fit_time_sec": float(metrics.get("fit_time_sec", float("nan"))),
            }
        )
    return pd.DataFrame(rows)


def _write_outputs(frame: pd.DataFrame, out_prefix: Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    frame = frame.sort_values(["rank_r1", "rank_r2", "rank_r3", "scene_id"]).reset_index(drop=True)
    frame.to_csv(out_prefix.with_suffix(".per_scene.csv"), index=False)

    summary = (
        frame.groupby(["rank", "rank_r1", "rank_r2", "rank_r3", "params"], as_index=False)
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
        .sort_values(["rank_r1", "rank_r2", "rank_r3"])
        .reset_index(drop=True)
    )
    for column in ("RMSE_std", "SAM_std", "NMSE_dB_std"):
        summary[column] = summary[column].fillna(0.0)
    summary.to_csv(out_prefix.with_suffix(".summary.csv"), index=False)

    lines = [
        r"\begin{tabular}{@{}l r c c c r@{}}",
        r"\toprule",
        r"Rank & Params & RMSE$\downarrow$ & SAM$\downarrow$ & NMSE(dB)$\downarrow$ & Scenes\\",
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"{row.rank} & {int(row.params):,} & "
            f"{row.RMSE_mean:.4f} $\\pm$ {row.RMSE_std:.4f} & "
            f"{row.SAM_mean:.2f} $\\pm$ {row.SAM_std:.2f} & "
            f"{row.NMSE_dB_mean:.2f} $\\pm$ {row.NMSE_dB_std:.2f} & "
            f"{int(row.n)}\\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    out_prefix.with_suffix(".tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and summarize a NeurIPS CAVE Tucker rank sweep.")
    parser.add_argument("--exp", default=DEFAULT_EXP)
    parser.add_argument("--scene-ids", default="1-15")
    parser.add_argument("--rank", action="append", type=_parse_rank, dest="ranks")
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--n-iter-max", type=int, default=300)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--out-prefix", default="papers/neurips/tables/cave_tucker_rank_sweep")
    args = parser.parse_args()

    scene_ids = _parse_int_set(args.scene_ids)
    ranks = tuple(args.ranks or DEFAULT_RANKS)

    completed = _completed_keys(args.exp)
    commands = [
        _command(args.exp, scene_id, rank, args.n_iter_max)
        for rank in ranks
        for scene_id in scene_ids
        if (scene_id, rank) not in completed
    ]
    if args.collect_only:
        print(f"Collect-only mode: {len(completed)} completed jobs available; {len(commands)} jobs missing.")
    else:
        print(f"Running {len(commands)} missing jobs for {len(ranks)} ranks x {len(scene_ids)} scenes.")
        _run_commands(commands, max(1, args.max_parallel))

    frame = _collect_rows(args.exp)
    if frame.empty:
        print(f"No completed runs found for exp={args.exp!r}.")
        return
    _write_outputs(frame, PROJECT_ROOT / args.out_prefix)
    print(f"Wrote {args.out_prefix}.per_scene.csv, .summary.csv, and .tex")


if __name__ == "__main__":
    main()
