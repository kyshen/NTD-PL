from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd

from ..config import ExperimentEnv


def sync_artifact_to_latex(env: ExperimentEnv, source: str | Path, target_name: str | None = None) -> Path:
    src = Path(source)
    if not src.exists():
        raise FileNotFoundError(f"Artifact not found: {src}")
    latex_inputs_root = env.latex_root / "inputs"
    try:
        if os.path.lexists(latex_inputs_root):
            st = os.lstat(latex_inputs_root)
            if getattr(st, "st_file_attributes", 0) & 0x400:
                return src
            if getattr(st, "st_reparse_tag", 0):
                return src
    except OSError:
        pass
    if latex_inputs_root.exists() and latex_inputs_root.is_symlink():
        return src
    if latex_inputs_root.exists() and not latex_inputs_root.is_dir():
            return src
    dst = env.latex_inputs_dir / (target_name or src.name)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return dst


def write_csv_artifact(
    env: ExperimentEnv,
    frame: pd.DataFrame,
    artifact_name: str,
    latex_name: str | None = None,
    **to_csv_kwargs: object,
) -> tuple[Path, Path]:
    csv_path = env.artifacts_dir / artifact_name
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig", **to_csv_kwargs)
    latex_path = sync_artifact_to_latex(env, csv_path, latex_name or artifact_name)
    return csv_path, latex_path


def write_text_artifact(
    env: ExperimentEnv,
    text: str,
    artifact_name: str,
    latex_name: str | None = None,
    encoding: str = "utf-8",
) -> tuple[Path, Path]:
    path = env.artifacts_dir / artifact_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding=encoding)
    latex_path = sync_artifact_to_latex(env, path, latex_name or artifact_name)
    return path, latex_path
