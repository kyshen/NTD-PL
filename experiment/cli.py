from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from .registry import iter_specs_by_paper_order
from .runner import (
    postprocess_experiment,
    run_cave_random_completion,
    run_cave_representation,
    run_geometry_visualization,
    run_linear_consistency,
    run_nonlinear_approx,
    run_real_hsi_robustness,
)
from .paper_tables import feature_placeholder_tables
from .summary import print_summary, summary_experiment_names


def _venv_python() -> Path:
    return Path(__file__).resolve().parents[1] / ".venv" / "Scripts" / "python.exe"


def _maybe_reexec_in_venv(args: argparse.Namespace) -> bool:
    venv_python = _venv_python()
    if not venv_python.exists():
        return False
    if Path(sys.executable).resolve() == venv_python.resolve():
        return False
    if args.command == "postprocess" and args.experiment == "cave-representation":
        subprocess.run([str(venv_python), "-m", "experiment", *sys.argv[1:]], check=True)
        return True
    return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="exp")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("paper-tables")

    # Define common subcommands
    common_run_commands = ["benchmark", "run", "ntdpl"]

    # Linear consistency
    linear_parser = subparsers.add_parser("linear-consistency")
    linear_subparsers = linear_parser.add_subparsers(dest="linear_command", required=True)
    for cmd in common_run_commands:
        linear_subparsers.add_parser(cmd)

    # Nonlinear-approx
    smooth_parser = subparsers.add_parser("nonlinear-approx")
    smooth_subparsers = smooth_parser.add_subparsers(dest="nonlinear_command", required=True)
    for cmd in common_run_commands:
        smooth_subparsers.add_parser(cmd)

    # Geometry visualization
    geometry_parser = subparsers.add_parser("geometry-visualization")
    geometry_subparsers = geometry_parser.add_subparsers(dest="geometry_command", required=True)
    for cmd in common_run_commands:
        geometry_subparsers.add_parser(cmd)

    # Cave representation
    cave_parser = subparsers.add_parser("cave-representation")
    cave_subparsers = cave_parser.add_subparsers(dest="cave_command", required=True)
    for cmd in common_run_commands:
        cave_subparsers.add_parser(cmd)

    cave_completion_parser = subparsers.add_parser("cave-random-completion")
    cave_completion_subparsers = cave_completion_parser.add_subparsers(
        dest="cave_completion_command",
        required=True,
    )
    for cmd in common_run_commands:
        cave_completion_subparsers.add_parser(cmd)

    robustness_parser = subparsers.add_parser("real-hsi-robustness")
    robustness_subparsers = robustness_parser.add_subparsers(
        dest="robustness_command",
        required=True,
    )
    for cmd in common_run_commands:
        robustness_subparsers.add_parser(cmd)

    summary_parser = subparsers.add_parser("summary")
    summary_parser.add_argument("experiment", choices=summary_experiment_names())

    postprocess_parser = subparsers.add_parser("postprocess")
    postprocess_parser.add_argument(
        "experiment",
        choices=[spec.name for spec in iter_specs_by_paper_order()],
        help="Run postprocessing for an experiment (plots/tables), without re-running sweeps.",
    )

    projects_parser = subparsers.add_parser("projects")
    projects_parser.add_argument("--verbose", action="store_true")
    return parser


# Command handler mappings
RUN_MODE_MAP = {
    "benchmark": "benchmark",
    "run": "run",
    "ntdpl": "ntdpl",
}

EXPERIMENT_HANDLERS = {
    "linear-consistency": {
        "run_func": run_linear_consistency,
        "cmd_attr": "linear_command",
    },
    "nonlinear-approx": {
        "run_func": run_nonlinear_approx,
        "cmd_attr": "nonlinear_command",
    },
    "geometry-visualization": {
        "run_func": run_geometry_visualization,
        "cmd_attr": "geometry_command",
    },
    "cave-representation": {
        "run_func": run_cave_representation,
        "cmd_attr": "cave_command",
    },
    "cave-random-completion": {
        "run_func": run_cave_random_completion,
        "cmd_attr": "cave_completion_command",
    },
    "real-hsi-robustness": {
        "run_func": run_real_hsi_robustness,
        "cmd_attr": "robustness_command",
    },
}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if _maybe_reexec_in_venv(args):
        return

    # Handle special commands
    if args.command == "paper-tables":
        feature_placeholder_tables()
        return
    if args.command == "summary":
        print_summary(args.experiment)
        return
    if args.command == "postprocess":
        postprocess_experiment(args.experiment)
        return
    if args.command == "projects":
        for spec in iter_specs_by_paper_order():
            if args.verbose:
                print(
                    f"{spec.name}: {spec.category} | section={spec.paper_section} "
                    f"| multirun/{spec.multirun_dir} | {spec.description}"
                )
            else:
                print(spec.name)
        return

    # Handle experiment commands
    if args.command not in EXPERIMENT_HANDLERS:
        parser.error(f"Unknown command: {args.command}")
        return

    exp_config = EXPERIMENT_HANDLERS[args.command]
    cmd_attr = exp_config["cmd_attr"]
    cmd = getattr(args, cmd_attr)

    # Try mode-based command first
    if cmd in RUN_MODE_MAP:
        exp_config["run_func"](mode=RUN_MODE_MAP[cmd])
        return

    parser.error(f"Unsupported command: {cmd}")
