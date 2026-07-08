from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment.hsi_defaults import CAVE_RECON_MAIN_RANK


EXP = "cave-random-completion"


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


def _rank_override(rank: tuple[int, int, int]) -> str:
    return f"method.rank=[{rank[0]},{rank[1]},{rank[2]}]"


def _command(
    *,
    scene_id: int,
    seed: int,
    missing_rate: float,
    method: str,
    target_shape: tuple[int, int],
    n_iter_max: int,
) -> list[str]:
    exp_mode = "run" if method == "ntdpl" else "benchmark"
    run_slug = f"diag_s{scene_id:02d}_seed{seed}_mr{missing_rate:g}_{method}"
    command = [
        _project_python(),
        "run.py",
        "-m",
        f"exp={EXP}",
        f"exp_mode={exp_mode}",
        "data=cave_hsi",
        f"data.id={scene_id}",
        f"data.target_shape=[{target_shape[0]},{target_shape[1]}]",
        "data.crop_shape=null",
        "task=random-missing-completion",
        "task.log_level=1",
        f"task.seed={seed}",
        f"task.missing_rate={missing_rate}",
        "filter=bias-filter",
        "filter.normalize_method=max",
        f"method={method}",
        _rank_override(CAVE_RECON_MAIN_RANK),
        f"method.n_iter_max={n_iter_max}",
        f"hydra.sweep.dir=artifacts/multirun/{EXP}/{exp_mode}/{run_slug}",
        "hydra.sweep.subdir=.",
    ]
    if method == "ntdpl":
        command.extend(
            [
                "method.p_max=4",
                "method.init=tucker",
                "method.use_continuation=true",
                "method.factor_normalize=true",
                "method.beta_update_method=ridge_lstsq",
            ]
        )
    return command


def _run_commands(commands: list[list[str]], max_parallel: int) -> None:
    env = _child_env()
    if max_parallel <= 1:
        for command in commands:
            subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)
        return
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = [executor.submit(subprocess.run, command, cwd=PROJECT_ROOT, env=env, check=True) for command in commands]
        for future in as_completed(futures):
            future.result()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rerun a minimal CAVE completion batch for diagnostics."
    )
    parser.add_argument("--scene-ids", default="1-15")
    parser.add_argument("--missing-rates", default="0.3,0.5")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--methods", default="tucker,ntdpl")
    parser.add_argument("--target-shape", default="512,512")
    parser.add_argument("--n-iter-max", type=int, default=300)
    parser.add_argument("--max-parallel", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--postprocess", action="store_true")
    args = parser.parse_args()

    scene_ids = []
    for token in args.scene_ids.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = [int(v.strip()) for v in token.split("-", 1)]
            scene_ids.extend(range(start, end + 1))
        else:
            scene_ids.append(int(token))
    missing_rates = [float(x.strip()) for x in args.missing_rates.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    shape_vals = [int(x.strip()) for x in args.target_shape.split(",") if x.strip()]
    if len(shape_vals) != 2:
        raise ValueError("target-shape must have two integers, e.g. 512,512")
    target_shape = (shape_vals[0], shape_vals[1])

    commands: list[list[str]] = []
    for scene_id in scene_ids:
        for missing_rate in missing_rates:
            for seed in seeds:
                for method in methods:
                    commands.append(
                        _command(
                            scene_id=scene_id,
                            seed=seed,
                            missing_rate=missing_rate,
                            method=method,
                            target_shape=target_shape,
                            n_iter_max=args.n_iter_max,
                        )
                    )

    print(f"Launching {len(commands)} jobs with max_parallel={args.max_parallel}")
    _run_commands(commands, max_parallel=max(1, int(args.max_parallel)))

    if args.collect:
        subprocess.run(
            [_project_python(), "collect.py", f"--exp={EXP}"],
            cwd=PROJECT_ROOT,
            env=_child_env(),
            check=True,
        )
    if args.postprocess:
        subprocess.run(
            [_project_python(), "-m", "experiment", "postprocess", EXP],
            cwd=PROJECT_ROOT,
            env=_child_env(),
            check=True,
        )


if __name__ == "__main__":
    main()
