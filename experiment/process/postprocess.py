from __future__ import annotations

from .registry import get_postprocessors, registered_experiments


def postprocess_experiment(exp: str) -> None:
    processors = get_postprocessors(exp)
    if not processors:
        available = ", ".join(registered_experiments())
        raise ValueError(f"Unknown experiment for postprocess: {exp}. Available: {available}")

    for processor in processors:
        processor.func()
