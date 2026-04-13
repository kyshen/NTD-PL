from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..config import get_env
from .nmse_alpha_plots import NmseAlphaSpec
from .nmse_pmax_plots import NmsePmaxSpec
from .nmse_common import DEFAULT_METHOD_ORDER, infer_nonlinears, load_runs
from .nmse_alpha_plots import save_nmse_vs_alpha as _save_nmse_vs_alpha
from .nmse_pmax_plots import save_nmse_vs_pmax as _save_nmse_vs_pmax


@dataclass(frozen=True)
class NmseAlphaPmaxSpec:
    exp_name: str
    alpha_ref: float = 0.25
    ntdpl_p_max: int = 8
    method_order: Sequence[str] = tuple(DEFAULT_METHOD_ORDER)
    alpha_values: Sequence[float] | None = None
    alpha_range: tuple[float, float] | None = None
    pmax_values: Sequence[int] | None = None
    pmax_range: tuple[int, int] | None = None
    nonlinears: Sequence[str] | None = None


def save_nmse_vs_alpha(exp_name: str, nonlinear: str, spec: NmseAlphaPmaxSpec) -> Path:
    alpha_spec = NmseAlphaSpec(
        exp_name=spec.exp_name,
        ntdpl_p_max=spec.ntdpl_p_max,
        method_order=spec.method_order,
        alpha_values=spec.alpha_values,
        alpha_range=spec.alpha_range,
    )
    return _save_nmse_vs_alpha(exp_name, nonlinear, alpha_spec)


def save_nmse_vs_pmax(exp_name: str, nonlinear: str, spec: NmseAlphaPmaxSpec) -> Path:
    pmax_spec = NmsePmaxSpec(
        exp_name=spec.exp_name,
        alpha_ref=spec.alpha_ref,
        pmax_values=spec.pmax_values,
        pmax_range=spec.pmax_range,
    )
    return _save_nmse_vs_pmax(exp_name, nonlinear, pmax_spec)


def save_nmse_alpha_pmax_plots(spec: NmseAlphaPmaxSpec) -> list[Path]:
    env = get_env(spec.exp_name)
    runs = load_runs(env)

    tbl = runs
    nonlinears = list(spec.nonlinears) if spec.nonlinears is not None else infer_nonlinears(tbl)

    written: list[Path] = []
    for nonlinear in nonlinears:
        written.append(save_nmse_vs_alpha(spec.exp_name, nonlinear, spec))
        written.append(save_nmse_vs_pmax(spec.exp_name, nonlinear, spec))
    return written
