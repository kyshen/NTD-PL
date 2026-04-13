from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    category: str
    multirun_dir: str
    description: str
    paper_section: str = "misc"
    paper_order: int = 999


EXPERIMENT_SPECS: dict[str, ExperimentSpec] = {
    "linear-consistency": ExperimentSpec(
        name="linear-consistency",
        category="analysis",
        multirun_dir="linear-consistency",
        description="Linear consistency validation on controlled synthetic data.",
        paper_section="synthetic",
        paper_order=10,
    ),
    "nonlinear-approx": ExperimentSpec(
        name="nonlinear-approx",
        category="analysis",
        multirun_dir="nonlinear-approx",
        description="Unified nonlinear link approximation analysis on controlled synthetic data.",
        paper_section="synthetic",
        paper_order=20,
    ),
    "geometry-visualization": ExperimentSpec(
        name="geometry-visualization",
        category="analysis",
        multirun_dir="geometry-visualization",
        description="Geometric surface visualization analysis on controlled synthetic data.",
        paper_section="synthetic",
        paper_order=40,
    ),
    "paper-tables": ExperimentSpec(
        name="paper-tables",
        category="paper",
        multirun_dir="paper-tables",
        description="Generate LaTeX table inputs used in the paper (placeholders and small utilities).",
        paper_section="misc",
        paper_order=900,
    ),
    "cave-representation": ExperimentSpec(
        name="cave-representation",
        category="analysis",
        multirun_dir="cave-representation",
        description="CAVE hyperspectral nonlinear low-rank representation with compression-reconstruction analysis.",
        paper_section="real-hsi",
        paper_order=100,
    ),
    "cave-random-completion": ExperimentSpec(
        name="cave-random-completion",
        category="analysis",
        multirun_dir="cave-random-completion",
        description="CAVE hyperspectral random missing completion with Tucker and NTD-PL.",
        paper_section="real-hsi",
        paper_order=110,
    ),
    "real-hsi-robustness": ExperimentSpec(
        name="real-hsi-robustness",
        category="analysis",
        multirun_dir="real-hsi-robustness",
        description="Cross-dataset robustness validation on real hyperspectral reconstruction and completion.",
        paper_section="real-hsi",
        paper_order=131,
    ),
}


def get_spec(exp_name: str) -> ExperimentSpec:
    try:
        return EXPERIMENT_SPECS[exp_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported experiment: {exp_name}") from exc


def iter_specs_by_paper_order() -> list[ExperimentSpec]:
    return sorted(EXPERIMENT_SPECS.values(), key=lambda spec: (spec.paper_order, spec.name))
