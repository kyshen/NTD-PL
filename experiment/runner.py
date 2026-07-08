from __future__ import annotations

from datetime import datetime
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from omegaconf import OmegaConf

from .config import get_env
from .hsi_defaults import (
    CAVE_RECON_MAIN_RANK,
    CAVE_RECON_RANKS,
    completion_rank_for_dataset,
    load_dataset_shape,
)
from .process import postprocess_experiment


def _project_python(project_root: Path) -> str:
    venv_python = project_root / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def _cpu_intensive_child_env() -> dict[str, str]:
    env = os.environ.copy()
    # Prefer process-level parallelism over oversubscribed BLAS/OpenMP threads.
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


def _default_parallel_jobs(*, reserve: int = 2, cap: int = 12) -> int:
    logical = os.cpu_count() or 2
    return max(1, min(cap, logical - reserve))


def _run_command(command: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    subprocess.run(
        command,
        check=True,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
    )


def _run_commands_parallel(
    commands: list[list[str]],
    cwd: Path,
    max_parallel: int,
    env: dict[str, str] | None = None,
) -> None:
    if not commands:
        return
    if max_parallel <= 1 or len(commands) == 1:
        for command in commands:
            _run_command(command, cwd=cwd, env=env)
        return
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = [executor.submit(_run_command, command, cwd, env) for command in commands]
        for future in as_completed(futures):
            future.result()


def seed_override(seed_count: int) -> str:
    if seed_count < 1:
        raise ValueError(f"seed_count must be >= 1, got {seed_count}")
    return "data.seed=" + ",".join(str(i) for i in range(seed_count))


def _remove_path(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _remove_file(path: Path) -> None:
    if path.exists():
        path.unlink()


def _remove_results_dir(exp_name: str, exp_mode: str | None = None) -> None:
    env = get_env(exp_name)
    results_dir_base = env.project_root / "artifacts" / "multirun" / exp_name
    if exp_mode:
        results_dir = results_dir_base / exp_mode
    else:
        results_dir = results_dir_base
    _remove_path(results_dir)


def _is_ntdpl_group(overrides: list[str]) -> bool:
    """Check if a group contains ntdpl method.

    Note: A single override may sweep multiple methods, e.g. "method=tucker,ntdpl".
    """

    def _parse_method_values(value: str) -> set[str]:
        return {v.strip() for v in value.split(",") if v.strip()}

    methods: set[str] = set()
    for override in overrides:
        if not override.startswith("method="):
            continue
        _, value = override.split("=", 1)
        methods |= _parse_method_values(value)

    if "ntdpl" in methods and (methods - {"ntdpl"}):
        raise ValueError(
            "A single group cannot mix 'ntdpl' with other methods. "
            "Split this group into benchmark_groups and ntdpl_groups. "
            f"Got method={sorted(methods)}"
        )
    return "ntdpl" in methods


def _cleanup_experiment_artifacts(exp: str) -> None:
    env = get_env(exp)
    if env.artifacts_dir.exists():
        shutil.rmtree(env.artifacts_dir)


def _cleanup_collected_results(exp: str) -> None:
    env = get_env(exp)
    results_dir = env.results_dir
    _remove_file(results_dir / "runs.parquet")
    _remove_file(results_dir / "curves.parquet")


def _split_top_level_csv(value: str) -> list[str]:
    items: list[str] = []
    current: list[str] = []
    depth = 0
    for ch in value:
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            item = "".join(current).strip()
            if item:
                items.append(item)
            current = []
            continue
        current.append(ch)
    tail = "".join(current).strip()
    if tail:
        items.append(tail)
    return items or [value.strip()]


def _override_pairs(overrides: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for override in overrides:
        if "=" not in override:
            continue
        key, value = override.split("=", 1)
        pairs.append((key.strip(), value.strip()))
    return pairs


def _normalize_override_value(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_normalize_override_value(item) for item in value) + "]"
    return str(value).strip()


def _cfg_value_for_key(cfg: dict, key: str) -> object:
    current: object = cfg
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    if isinstance(current, dict) and "_name" in current:
        return current["_name"]
    return current


def _expected_jobs(overrides: list[str]) -> int:
    count = 1
    for _, value in _override_pairs(overrides):
        count *= max(1, len(_split_top_level_csv(value)))
    return count


def _group_complete(exp: str, exp_mode: str, overrides: list[str]) -> bool:
    cfgs = _load_completed_cfgs(exp, exp_mode)
    return _group_complete_from_cfgs(cfgs, overrides)


def _load_completed_cfgs(exp: str, exp_mode: str) -> list[dict]:
    env = get_env(exp)
    mode_root = env.results_dir / exp_mode
    if not mode_root.exists():
        return []

    cfgs: list[dict] = []
    for run_root in sorted([p for p in mode_root.iterdir() if p.is_dir()], reverse=True):
        for subdir in [p for p in run_root.iterdir() if p.is_dir()]:
            if not (subdir / "eval.json").exists():
                continue
            cfg_path = subdir / ".hydra" / "config.yaml"
            if not cfg_path.exists():
                continue
            cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
            if isinstance(cfg, dict):
                cfgs.append(cfg)
    return cfgs


def _group_complete_from_cfgs(cfgs: list[dict], overrides: list[str]) -> bool:
    target_pairs = []
    for key, value in _override_pairs(overrides):
        target_pairs.append((key, {item.strip() for item in _split_top_level_csv(value)}))
    expected = _expected_jobs(overrides)
    matched: set[tuple[str, ...]] = set()

    for cfg in cfgs:
        signature: list[str] = []
        ok = True
        for key, target_values in target_pairs:
            actual = _normalize_override_value(_cfg_value_for_key(cfg, key))
            if actual not in target_values:
                ok = False
                break
            signature.append(f"{key}={actual}")
        if not ok:
            continue
        matched.add(tuple(signature))
        if len(matched) >= expected:
            return True
    return False


def _run_project(
    exp: str,
    benchmark_groups: list[list[str]],
    ntdpl_groups: list[list[str]],
    common: list[str] | None = None,
    mode: str = "run",
    benchmark_parallel_jobs: int = 1,
    ntdpl_parallel_jobs: int = 1,
) -> None:
    """
    Run experiment with optional mode filtering.
    
    Args:
        mode: "run" (both), "benchmark" (non-NTDPL only), "ntdpl" (NTDPL only).
            For backward compatibility, "all" is treated as "run".
    """
    if mode == "all":
        mode = "run"
    if mode not in ("run", "benchmark", "ntdpl"):
        raise ValueError(f"Invalid mode: {mode}. Expected one of: run, benchmark, ntdpl")

    env = get_env(exp)
    _cleanup_experiment_artifacts(exp)
    _cleanup_collected_results(exp)

    python_exe = _project_python(env.project_root)
    child_env = _cpu_intensive_child_env()
    command_prefix = [python_exe, "run.py", "-m", f"exp={exp}"]
    if common:
        command_prefix.extend(common)

    # Validate grouping: benchmark_groups must NOT include ntdpl, ntdpl_groups MUST include ntdpl.
    for group in benchmark_groups:
        if _is_ntdpl_group(group):
            raise ValueError(
                "benchmark_groups contains an ntdpl group; move it to ntdpl_groups. "
                f"Group: {group}"
            )
    for group in ntdpl_groups:
        if not _is_ntdpl_group(group):
            raise ValueError(
                "ntdpl_groups contains a non-ntdpl group; move it to benchmark_groups. "
                f"Group: {group}"
            )

    # Run benchmark groups
    if mode in ("run", "benchmark") and benchmark_groups:
        benchmark_cfgs = _load_completed_cfgs(exp, "benchmark")
        benchmark_commands: list[list[str]] = []
        for group_overrides in benchmark_groups:
            full_overrides = (common or []) + ["exp_mode=benchmark"] + group_overrides
            if _group_complete_from_cfgs(benchmark_cfgs, [f"exp={exp}", *full_overrides]):
                continue
            unique_dir = (
                f"hydra.sweep.dir=artifacts/multirun/{exp}/benchmark/"
                f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}"
            )
            benchmark_commands.append(command_prefix + [unique_dir, "exp_mode=benchmark"] + group_overrides)
        _run_commands_parallel(
            benchmark_commands,
            cwd=env.project_root,
            max_parallel=benchmark_parallel_jobs,
            env=child_env,
        )

    # Run ntdpl groups (stored under exp_mode=run)
    if mode in ("run", "ntdpl") and ntdpl_groups:
        ntdpl_cfgs = _load_completed_cfgs(exp, "run")
        ntdpl_commands: list[list[str]] = []
        for group_overrides in ntdpl_groups:
            full_overrides = (common or []) + ["exp_mode=run"] + group_overrides
            if _group_complete_from_cfgs(ntdpl_cfgs, [f"exp={exp}", *full_overrides]):
                continue
            unique_dir = (
                f"hydra.sweep.dir=artifacts/multirun/{exp}/run/"
                f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}"
            )
            ntdpl_commands.append(command_prefix + [unique_dir, "exp_mode=run"] + group_overrides)
        _run_commands_parallel(
            ntdpl_commands,
            cwd=env.project_root,
            max_parallel=ntdpl_parallel_jobs,
            env=child_env,
        )

    # Collect all results (both benchmark and run modes)
    _run_command(
        [python_exe, "collect.py", f"--exp={exp}"],
        cwd=env.project_root,
        env=child_env,
    )

    if mode == "run":
        _run_command(
            [python_exe, "-m", "experiment", "postprocess", exp],
            cwd=env.project_root,
            env=child_env,
        )



def run_linear_consistency(mode: str = "run") -> None:
    filter_bias = "filter.bias=0,0.5"
    _run_project(
        exp="linear-consistency",
        benchmark_groups=[["method=tucker", filter_bias]],
        ntdpl_groups=[
            [
                "method=ntdpl",
                "method.p_max=6",
                "method.use_continuation=false",
                "method.allow_constant_term=false,true",
                filter_bias,
            ]
        ],
        common=[
            "filter=bias-filter",
            "filter.normalize_method=energy",
            "filter.snr_db=30",
            "data=tucker",
            seed_override(10),
        ],
        mode=mode,
    )


def run_nonlinear_approx(mode: str = "run") -> None:
    filter_alpha = "filter.alpha=0.1,0.15,0.2,0.25,0.3,0.35,0.4"
    filter_nonlinear = "filter.nonlinear=poly2,poly3,tanh,exp"
    _run_project(
        exp="nonlinear-approx",
        benchmark_groups=[
            ["method=tucker,cp,tr,tt", filter_nonlinear, filter_alpha],
        ],
        ntdpl_groups=[
            ["method=ntdpl", "method.p_max=6", filter_nonlinear, filter_alpha],
            ["method=ntdpl", "method.p_max=1,2,3,4,5,6", "task.log_level=1", filter_nonlinear, "filter.alpha=0.25"],
        ],
        common=[
            "data=tucker",
            "filter=nonlinear-filter",
            seed_override(10)
        ],
        mode=mode
    )


def run_geometry_visualization(mode: str = "run") -> None:
    _run_project(
        exp="geometry-visualization",
        benchmark_groups=[],
        ntdpl_groups=[["method=ntdpl", "method.p_max=1,2,3,4", "task.log_level=1"]],
        common=[
            "data=tucker",
            "filter=nonlinear-filter",
            "filter.nonlinear=poly3",
            "filter.alpha=0.1,0.2,0.3",
        ],
        mode=mode,
    )


def _id_override(ids: range) -> str:
    return "data.id=" + ",".join(str(i) for i in ids)


def _dir_id_override(root: Path) -> str:
    ids = range(1, len(sorted(path for path in root.iterdir() if path.is_dir())) + 1)
    return _id_override(ids)


REAL_HSI_ROBUSTNESS_DATASETS = (
    "jasper_ridge_hsi",
    "samson_hsi",
    "urban_hsi",
    "cuprite_hsi",
)
def _shape_override(shape: tuple[int, int, int]) -> str:
    return f"data.target_shape=[{shape[0]},{shape[1]}]"


def _rank_override(rank: tuple[int, int, int]) -> str:
    return f"method.rank=[{rank[0]},{rank[1]},{rank[2]}]"


def run_cave_representation(mode: str = "run") -> None:
    env = get_env("cave-representation")
    cave_ids = _dir_id_override(env.project_root / "data" / "CAVE")
    common = [
        "data=cave_hsi",
        "data.target_shape=[512,512]",
        "data.crop_shape=null",
        "filter=bias-filter",
        "filter.normalize_method=max",
        "method.n_iter_max=300",
    ]

    ranks = list(CAVE_RECON_RANKS)
    benchmark_groups: list[list[str]] = []
    ntdpl_groups: list[list[str]] = []
    for rank in ranks:
        benchmark_groups.append(
            [
                cave_ids,
                "method=tucker",
                _rank_override(rank),
            ]
        )
        ntdpl_groups.append(
            [
                cave_ids,
                "method=ntdpl",
                _rank_override(rank),
                "method.p_max=6",
                "method.init=tucker",
                "method.use_continuation=true",
                "method.factor_normalize=true",
                "method.beta_update_method=moments_normal_eq",
            ]
        )

    _run_project(
        exp="cave-representation",
        benchmark_groups=benchmark_groups,
        ntdpl_groups=ntdpl_groups,
        common=common,
        mode=mode,
    )


def run_cave_random_completion(mode: str = "run") -> None:
    env = get_env("cave-random-completion")
    cave_ids = _dir_id_override(env.project_root / "data" / "CAVE")
    rank = _rank_override(CAVE_RECON_MAIN_RANK)
    common = [
        "data=cave_hsi",
        "data.target_shape=[512,512]",
        "data.crop_shape=null",
        "task=random-missing-completion",
        "task.log_level=1",
        "filter=bias-filter",
        "filter.normalize_method=max",
        rank,
        "method.n_iter_max=300",
    ]

    scene_ids = [str(i) for i in range(1, 16)]
    benchmark_groups: list[list[str]] = []
    ntdpl_groups: list[list[str]] = []
    for scene_id in scene_ids:
        for seed in ("0", "1", "2"):
            for missing_rate in ("0.1", "0.3", "0.5", "0.7"):
                job_overrides = [
                    f"data.id={scene_id}",
                    f"task.seed={seed}",
                    f"task.missing_rate={missing_rate}",
                ]
                benchmark_groups.append(job_overrides + ["method=tucker"])
                ntdpl_groups.append(
                    job_overrides
                    + [
                        "method=ntdpl",
                        "method.p_max=4",
                        "method.init=tucker",
                        "method.use_continuation=true",
                        "method.factor_normalize=true",
                        "method.beta_update_method=ridge_lstsq",
                    ]
                )

    _run_project(
        exp="cave-random-completion",
        benchmark_groups=benchmark_groups,
        ntdpl_groups=ntdpl_groups,
        common=common,
        mode=mode,
        benchmark_parallel_jobs=6,
        ntdpl_parallel_jobs=12,
    )
def run_real_hsi_robustness(mode: str = "run") -> None:
    env = get_env("real-hsi-robustness")
    common = [
        "filter=bias-filter",
        "filter.normalize_method=max",
        "method.n_iter_max=300",
    ]

    benchmark_groups: list[list[str]] = []
    ntdpl_groups: list[list[str]] = []
    for dataset_name in REAL_HSI_ROBUSTNESS_DATASETS:
        shape = load_dataset_shape(env.project_root, dataset_name)
        rank = completion_rank_for_dataset(env.project_root, dataset_name)
        dataset_overrides = [
            f"data={dataset_name}",
            _shape_override(shape),
            "data.crop_shape=null",
            _rank_override(rank),
        ]
        benchmark_groups.append(
            dataset_overrides
            + [
                "task=decompose",
                "task.log_level=0",
                "method=tucker",
            ]
        )
        ntdpl_groups.append(
            dataset_overrides
            + [
                "task=decompose",
                "task.log_level=0",
                "method=ntdpl",
                "method.p_max=4",
                "method.init=tucker",
                "method.use_continuation=true",
                "method.factor_normalize=true",
                "method.beta_update_method=ridge_lstsq",
            ]
        )

    _run_project(
        exp="real-hsi-robustness",
        benchmark_groups=benchmark_groups,
        ntdpl_groups=ntdpl_groups,
        common=common,
        mode=mode,
        benchmark_parallel_jobs=_default_parallel_jobs(cap=6),
        ntdpl_parallel_jobs=_default_parallel_jobs(cap=6),
    )
