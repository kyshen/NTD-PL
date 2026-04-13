"""Backward-compatible filter module.

New modular implementation lives in `src.filters`.
This module remains so Hydra `_target_` strings like `src.filter.MultiFilter`
continue to work.
"""

from src.filters import BiasFilter, DataFilter, MultiFilter

__all__ = [
    "DataFilter",
    "MultiFilter",
    "BiasFilter",
]
