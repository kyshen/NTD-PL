from __future__ import annotations

from ..helpers.cave_representation_original_space import (
    main_rank_significance_summary,
    reconstruction_table,
    significance_table,
)
from ..registry import register_postprocessor


@register_postprocessor(exp_name="cave-representation", order=10)
def recon_summary() -> None:
    reconstruction_table()


@register_postprocessor(exp_name="cave-representation", order=20)
def significance() -> None:
    significance_table()


@register_postprocessor(exp_name="cave-representation", order=30)
def main_rank_significance() -> None:
    main_rank_significance_summary()
