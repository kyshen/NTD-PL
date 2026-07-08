from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from omegaconf import OmegaConf


def flatten_dict(d, parent_key="", sep="."):
    """Flatten nested dictionaries into dot-separated keys for runs tables."""
    items = {}
    if isinstance(d, dict):
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
            if isinstance(v, dict):
                items.update(flatten_dict(v, new_key, sep=sep))
            else:
                items[new_key] = v
    return items


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def find_run_dirs(root: Path):
    """Yield Hydra run directories containing .hydra/config.yaml."""
    for p in root.rglob(".hydra"):
        cfg = p / "config.yaml"
        if cfg.exists():
            yield p.parent


def logs_to_curves(logs: List[Dict], run_id: str) -> pd.DataFrame:
    df = pd.DataFrame(logs)

    step_col = "step"
    df = df.reset_index().rename(columns={"index": step_col})

    value_cols = [c for c in df.columns if c != step_col]
    long_df = df.melt(
        id_vars=[step_col],
        value_vars=value_cols,
        var_name="metric",
        value_name="value",
    )
    long_df = long_df.rename(columns={step_col: step_col})
    long_df["run_id"] = run_id
    return long_df


def df_object2str(df: pd.DataFrame):
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(json.dumps)


def collect(root: str | Path):
    root = Path(root)
    runs_rows = []
    curves_rows = []

    for run_dir in find_run_dirs(root):
        run_id = str(run_dir.relative_to(root)).replace("\\", "/")

        cfg_path = run_dir / ".hydra" / "config.yaml"
        overrides_path = run_dir / ".hydra" / "overrides.yaml"
        cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
        overrides = OmegaConf.to_container(OmegaConf.load(overrides_path), resolve=True)

        eval_path = run_dir / "eval.json"
        logs_path = run_dir / "logs.json"
        eval_dict = read_json(eval_path) if eval_path.exists() else {}
        logs_dict = read_json(logs_path) if logs_path.exists() else [{}]

        flat_cfg = flatten_dict(cfg)
        flat_eval = flatten_dict(eval_dict) if isinstance(eval_dict, dict) else {"eval": eval_dict}

        row = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            **flat_cfg,
            **flat_eval,
            "state_path": str(run_dir / "state.mat") if (run_dir / "state.mat").exists() else None,
        }

        curves = logs_to_curves(logs_dict, run_id)
        if isinstance(overrides, list):
            for ovr in overrides:
                k, v = ovr.split("=", 1)
                k = "ovr." + k
                curves[k] = v
                row[k] = v

        runs_rows.append(row)
        curves_rows.append(curves)

    runs_df = pd.DataFrame(runs_rows)
    curves_df = pd.concat(curves_rows, ignore_index=True) if curves_rows else pd.DataFrame()
    df_object2str(runs_df)
    df_object2str(curves_df)

    return runs_df, curves_df


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Collect Hydra runs")
    argparser.add_argument("--exp", type=str)
    args = argparser.parse_args()

    run_dir = Path("artifacts") / "multirun" / args.exp

    runs_df, curves_df = collect(run_dir)

    runs_df.to_parquet(run_dir / "runs.parquet", index=False)
    curves_df.to_parquet(run_dir / "curves.parquet", index=False)

    print("Saved:", run_dir / "runs.parquet", run_dir / "curves.parquet")
