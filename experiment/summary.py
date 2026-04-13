from __future__ import annotations

from .config import get_env
from .utils.io import load_run_parquets
from .registry import EXPERIMENT_SPECS


SUMMARY_COLUMNS = {
    "exp2": ["ovr.data", "ovr.method", "ovr.filter.nonlinear", "ovr.method.p_max", "CR", "RMSE", "NMSE", "NMSE_dB"],
    "exp3": ["ovr.data", "ovr.method", "ovr.method.p_max", "CR", "RMSE", "NMSE", "NMSE_dB"],
    "test": ["ovr.data", "ovr.method", "ovr.filter.nonlinear", "ovr.method.p_max", "CR", "RMSE", "NMSE", "NMSE_dB"],
}


def print_summary(exp_name: str) -> None:
    env = get_env(exp_name)
    runs = load_run_parquets(env.results_dir)["runs"]
    columns = SUMMARY_COLUMNS.get(exp_name)
    if columns is None:
        raise ValueError(f"Unsupported summary experiment: {exp_name}")
    summary = runs.loc[:, columns].sort_values("RMSE", ascending=True)
    print(summary.to_string(index=False))


def summary_experiment_names() -> list[str]:
    return [name for name, spec in EXPERIMENT_SPECS.items() if spec.category == "summary"]
