"""Utility helpers for the experiment package.

This package groups reusable, tool-like helpers (I/O, plotting, etc.) to keep
experiment scripts focused on experiment logic.
"""

try:
    from .io import load_run_parquets, load_state_mat, maybe_numeric
except ModuleNotFoundError as exc:  # pragma: no cover
    _io_import_error = exc

    def load_run_parquets(*args, **kwargs):  # type: ignore[no-redef]
        raise ModuleNotFoundError(
            "experiment.utils.io requires optional dependency 'scipy'. "
            "Install project dependencies (see experiment/pyproject.toml)."
        ) from _io_import_error

    def load_state_mat(*args, **kwargs):  # type: ignore[no-redef]
        raise ModuleNotFoundError(
            "experiment.utils.io requires optional dependency 'scipy'. "
            "Install project dependencies (see experiment/pyproject.toml)."
        ) from _io_import_error

    def maybe_numeric(*args, **kwargs):  # type: ignore[no-redef]
        raise ModuleNotFoundError(
            "experiment.utils.io requires optional dependency 'scipy'. "
            "Install project dependencies (see experiment/pyproject.toml)."
        ) from _io_import_error
from .paper import sync_artifact_to_latex, write_csv_artifact, write_text_artifact
from .plotting import (
    METHOD_STYLES,
    PALETTE,
    Palette,
    apply_theme,
    method_style,
    rounded_bounds,
    save_figure,
    style_axes,
)

__all__ = [
    "load_run_parquets",
    "load_state_mat",
    "maybe_numeric",
    "sync_artifact_to_latex",
    "write_csv_artifact",
    "write_text_artifact",
    "METHOD_STYLES",
    "PALETTE",
    "Palette",
    "apply_theme",
    "method_style",
    "rounded_bounds",
    "save_figure",
    "style_axes",
]
