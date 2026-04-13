"""Experiment post-processing pipeline organized by function.

- plots/: viz-backed figure generation
- tables/: CSV/LaTeX table generation
- helpers/: utilities used by runner/CLI
- registry.py: experiment-agnostic processor registration and discovery
"""

from .postprocess import postprocess_experiment
from .registry import get_postprocessors, register_postprocessor, registered_experiments

__all__ = [
    "get_postprocessors",
    "postprocess_experiment",
    "register_postprocessor",
    "registered_experiments",
]
