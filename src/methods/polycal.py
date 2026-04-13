from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from src.utils.completion_ops import mask_to_bool


@dataclass(frozen=True)
class PolynomialCalibrationDiagnostics:
    degree: int
    lambda_reg: float
    observed_count: int
    x_min: float
    x_max: float
    x_mean: float
    design_cond_raw: float
    design_cond_scaled: float
    fit_time_sec: float


class PolynomialCalibration:
    def __init__(self, degree: int, *, lambda_reg: float = 1e-6) -> None:
        degree = int(degree)
        if degree < 1:
            raise ValueError(f"`degree` must be >= 1, got {degree}.")
        self.degree = degree
        self.lambda_reg = float(lambda_reg)
        self.coefficients: np.ndarray | None = None
        self.diagnostics: PolynomialCalibrationDiagnostics | None = None

    def _design_matrix(self, x: np.ndarray) -> np.ndarray:
        return np.vander(x, N=self.degree + 1, increasing=True)

    def fit(
        self,
        x_pred: np.ndarray,
        x_target: np.ndarray,
        mask: np.ndarray,
    ) -> "PolynomialCalibration":
        x_pred = np.asarray(x_pred, dtype=np.float64)
        x_target = np.asarray(x_target, dtype=np.float64)
        mask_bool = mask_to_bool(mask, x_pred.shape)
        if x_pred.shape != x_target.shape:
            raise ValueError(f"Shape mismatch: {x_pred.shape} vs {x_target.shape}.")

        x_obs = x_pred[mask_bool].reshape(-1)
        y_obs = x_target[mask_bool].reshape(-1)
        if x_obs.size == 0:
            raise ValueError("Observed mask contains no entries for polynomial calibration.")

        start = perf_counter()
        phi = self._design_matrix(x_obs)
        col_scales = np.linalg.norm(phi, axis=0)
        col_scales = np.maximum(col_scales, 1e-12)
        phi_scaled = phi / col_scales

        gram = phi_scaled.T @ phi_scaled
        rhs = phi_scaled.T @ y_obs
        ridge = self.lambda_reg * np.eye(gram.shape[0], dtype=np.float64)
        coeff_scaled = np.linalg.solve(gram + ridge, rhs)
        coeff = coeff_scaled / col_scales
        fit_time = perf_counter() - start

        self.coefficients = coeff.astype(np.float64, copy=True)
        self.diagnostics = PolynomialCalibrationDiagnostics(
            degree=self.degree,
            lambda_reg=self.lambda_reg,
            observed_count=int(x_obs.size),
            x_min=float(np.min(x_obs)),
            x_max=float(np.max(x_obs)),
            x_mean=float(np.mean(x_obs)),
            design_cond_raw=float(np.linalg.cond(phi)),
            design_cond_scaled=float(np.linalg.cond(phi_scaled)),
            fit_time_sec=float(fit_time),
        )
        return self

    def apply(self, x_pred: np.ndarray) -> np.ndarray:
        if self.coefficients is None:
            raise ValueError("PolynomialCalibration must be fitted before apply().")
        x_pred = np.asarray(x_pred, dtype=np.float64)
        out = np.zeros_like(x_pred, dtype=np.float64)
        for coeff in self.coefficients[::-1]:
            out = out * x_pred + float(coeff)
        return out.astype(np.float32)

    def get_state_dict(self) -> dict[str, Any]:
        if self.coefficients is None or self.diagnostics is None:
            raise ValueError("PolynomialCalibration must be fitted before exporting state.")
        state = {
            "degree": int(self.degree),
            "lambda_reg": float(self.lambda_reg),
            "coefficients": np.array(self.coefficients, dtype=np.float64, copy=True),
        }
        state.update(self.diagnostics.__dict__)
        return state
