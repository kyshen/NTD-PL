from __future__ import annotations

from typing import Any, TYPE_CHECKING
if TYPE_CHECKING:
    from src.data.base import BaseData

import numpy as np
from .base import DataFilter

class NonlinearFilter(DataFilter):
    def __init__(self, **filter_cfg: Any) -> None:
        super().__init__(**filter_cfg)

    def __call__(self, data: "BaseData") -> "BaseData":
        self.normalize(data)
        self.insert_nonlinearity(data)
        self.add_noise(data)
        return data

    def insert_nonlinearity(self, data: "BaseData") -> None:
        nonlinear = str(self.cfg["nonlinear"])
        alpha = float(self.cfg["alpha"])

        if nonlinear == "none":
            return
        if nonlinear == "sin":
            data._dense = apply_sin(data._dense, alpha)
        elif nonlinear == "cos":
            data._dense = apply_cos(data._dense, alpha)
        elif nonlinear == "tanh":
            data._dense = apply_tanh(data._dense, alpha)
        elif nonlinear == "poly2":
            data._dense = apply_poly2(data._dense, alpha)
        elif nonlinear == "poly3":
            data._dense = apply_poly3(data._dense, alpha)
        elif nonlinear == "poly34":
            data._dense = apply_poly34(data._dense, alpha)
        elif nonlinear == "exp":
            data._dense = apply_exp_response(data._dense, alpha)
        else:
            raise ValueError(f"Unsupported filter function: {nonlinear}")

        data._dense_eval = data._dense.copy()



from src.utils.filter_ops import (
    mix_with_exact_energy_ratio,
    orthogonal_nonlinear_part,
    typical_scale,
)


def apply_poly2(x: np.ndarray, alpha: float) -> np.ndarray:
    gx = x**2
    r = orthogonal_nonlinear_part(x, gx)
    return mix_with_exact_energy_ratio(x, r, alpha)


def apply_poly3(x: np.ndarray, alpha: float) -> np.ndarray:
    gx = x**2 + x**3
    r = orthogonal_nonlinear_part(x, gx)
    return mix_with_exact_energy_ratio(x, r, alpha)


def apply_poly34(x: np.ndarray, alpha: float) -> np.ndarray:
    gx = x**3 + x**4
    r = orthogonal_nonlinear_part(x, gx)
    return mix_with_exact_energy_ratio(x, r, alpha)


def apply_exp_response(x: np.ndarray, alpha: float) -> np.ndarray:
    k = 1.0 / typical_scale(x)
    gx = np.expm1(k * x) / k
    r = orthogonal_nonlinear_part(x, gx)
    return mix_with_exact_energy_ratio(x, r, alpha)


def apply_sin(x: np.ndarray, alpha: float) -> np.ndarray:
    k = np.pi / 2 / typical_scale(x)
    gx = np.sin(k * x)
    r = orthogonal_nonlinear_part(x, gx)
    return mix_with_exact_energy_ratio(x, r, alpha)


def apply_cos(x: np.ndarray, alpha: float) -> np.ndarray:
    k = np.pi / 2 / typical_scale(x)
    gx = np.cos(k * x)
    r = orthogonal_nonlinear_part(x, gx)
    return mix_with_exact_energy_ratio(x, r, alpha)


def apply_tanh(x: np.ndarray, alpha: float) -> np.ndarray:
    k = 1.5 / typical_scale(x)
    gx = np.tanh(k * x)
    r = orthogonal_nonlinear_part(x, gx)
    return mix_with_exact_energy_ratio(x, r, alpha)
