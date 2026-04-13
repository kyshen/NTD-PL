from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from .registry import get_spec


METHOD_LABELS = {
    "tucker": "Tucker",
    "cp": "CP",
    "tr": "TR",
    "tt": "TT",
    "ntdpl": "NTD-PL",
    "softimpute": "SoftImpute",
}


@dataclass(frozen=True)
class ExperimentEnv:
    project_root: Path
    experiments_root: Path
    exp_name: str
    multirun_name: str

    @property
    def results_dir(self) -> Path:
        return self.project_root / "multirun" / self.multirun_name

    @property
    def artifacts_dir(self) -> Path:
        path = self.experiments_root / "outputs" / self.exp_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def latex_root(self) -> Path:
        return self.project_root / "latex-zh"

    @property
    def latex_inputs_dir(self) -> Path:
        path = self.latex_root / "inputs" / self.exp_name
        latex_inputs_root = self.latex_root / "inputs"
        try:
            if os.path.lexists(latex_inputs_root):
                st = os.lstat(latex_inputs_root)
                if getattr(st, "st_file_attributes", 0) & 0x400:
                    return path
                if getattr(st, "st_reparse_tag", 0):
                    return path
        except OSError:
            pass
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def figs_dir(self) -> Path:
        path = self.artifacts_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def method_labels(self) -> dict[str, str]:
        return METHOD_LABELS

    def label_for_method(self, method: str) -> str:
        return self.method_labels.get(method, method)


def get_env(exp_name: str) -> ExperimentEnv:
    experiments_root = Path(__file__).resolve().parent
    spec = get_spec(exp_name)
    return ExperimentEnv(
        project_root=experiments_root.parent,
        experiments_root=experiments_root,
        exp_name=spec.name,
        multirun_name=spec.multirun_dir,
    )
