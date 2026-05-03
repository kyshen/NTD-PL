from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from src.utils.completion_ops import mask_to_bool


@dataclass(frozen=True)
class ScalarMLPCalibrationDiagnostics:
    hidden_units: int
    lambda_reg: float
    max_iter: int
    batch_size: int
    observed_count: int
    train_count: int
    x_mean: float
    x_std: float
    y_mean: float
    y_std: float
    final_loss: float
    fit_time_sec: float


class ScalarMLPCalibration:
    """Post-hoc scalar MLP calibration for a fixed reconstruction.

    This baseline learns a scalar map y ~= f(x), where x is a fixed Tucker
    reconstruction entry and y is the observed target entry. It is deliberately
    a post-processing model, not a joint low-rank model.
    """

    def __init__(
        self,
        hidden_units: int = 16,
        *,
        lambda_reg: float = 1e-5,
        lr: float = 1e-3,
        max_iter: int = 2000,
        batch_size: int = 8192,
        max_train_samples: int = 200_000,
        random_state: int = 0,
    ) -> None:
        hidden_units = int(hidden_units)
        if hidden_units < 1:
            raise ValueError(f"`hidden_units` must be >= 1, got {hidden_units}.")
        self.hidden_units = hidden_units
        self.lambda_reg = float(lambda_reg)
        self.lr = float(lr)
        self.max_iter = int(max_iter)
        self.batch_size = int(batch_size)
        self.max_train_samples = int(max_train_samples)
        self.random_state = int(random_state)

        self.x_mean: float | None = None
        self.x_std: float | None = None
        self.y_mean: float | None = None
        self.y_std: float | None = None
        self.w1: np.ndarray | None = None
        self.b1: np.ndarray | None = None
        self.w2: np.ndarray | None = None
        self.b2: float | None = None
        self.w_skip: float | None = None
        self.diagnostics: ScalarMLPCalibrationDiagnostics | None = None

    def fit(
        self,
        x_pred: np.ndarray,
        x_target: np.ndarray,
        mask: np.ndarray,
    ) -> "ScalarMLPCalibration":
        x_pred = np.asarray(x_pred, dtype=np.float64)
        x_target = np.asarray(x_target, dtype=np.float64)
        if x_pred.shape != x_target.shape:
            raise ValueError(f"Shape mismatch: {x_pred.shape} vs {x_target.shape}.")
        mask_bool = mask_to_bool(mask, x_pred.shape)

        x_obs = x_pred[mask_bool].reshape(-1)
        y_obs = x_target[mask_bool].reshape(-1)
        if x_obs.size == 0:
            raise ValueError("Observed mask contains no entries for MLP calibration.")

        rng = np.random.default_rng(self.random_state)
        if self.max_train_samples > 0 and x_obs.size > self.max_train_samples:
            idx = rng.choice(x_obs.size, size=self.max_train_samples, replace=False)
            x_train = x_obs[idx]
            y_train = y_obs[idx]
        else:
            x_train = x_obs
            y_train = y_obs

        self.x_mean = float(np.mean(x_train))
        self.x_std = float(max(np.std(x_train), 1e-8))
        self.y_mean = float(np.mean(y_train))
        self.y_std = float(max(np.std(y_train), 1e-8))
        x = ((x_train - self.x_mean) / self.x_std).reshape(-1, 1)
        y = ((y_train - self.y_mean) / self.y_std).reshape(-1, 1)

        scale = 1.0 / np.sqrt(1.0)
        self.w1 = rng.normal(0.0, scale, size=(1, self.hidden_units))
        self.b1 = np.zeros((self.hidden_units,), dtype=np.float64)
        self.w2 = rng.normal(0.0, 1.0 / np.sqrt(self.hidden_units), size=(self.hidden_units, 1))
        self.b2 = 0.0
        self.w_skip = 1.0

        m_w1 = np.zeros_like(self.w1)
        v_w1 = np.zeros_like(self.w1)
        m_b1 = np.zeros_like(self.b1)
        v_b1 = np.zeros_like(self.b1)
        m_w2 = np.zeros_like(self.w2)
        v_w2 = np.zeros_like(self.w2)
        m_b2 = 0.0
        v_b2 = 0.0
        m_w_skip = 0.0
        v_w_skip = 0.0

        start = perf_counter()
        n = x.shape[0]
        batch_size = max(1, min(self.batch_size, n))
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        final_loss = np.inf

        for step in range(1, self.max_iter + 1):
            if batch_size == n:
                xb = x
                yb = y
            else:
                idx = rng.integers(0, n, size=batch_size)
                xb = x[idx]
                yb = y[idx]

            z1 = xb @ self.w1 + self.b1
            h = np.tanh(z1)
            pred = h @ self.w2 + self.b2 + self.w_skip * xb
            residual = pred - yb
            data_loss = float(np.mean(residual**2))
            reg_loss = float(self.lambda_reg * (np.sum(self.w1**2) + np.sum(self.w2**2)))
            final_loss = data_loss + reg_loss

            grad_pred = (2.0 / xb.shape[0]) * residual
            grad_w2 = h.T @ grad_pred + 2.0 * self.lambda_reg * self.w2
            grad_b2 = float(np.sum(grad_pred))
            grad_w_skip = float(np.sum(grad_pred * xb))
            grad_h = grad_pred @ self.w2.T
            grad_z1 = grad_h * (1.0 - h**2)
            grad_w1 = xb.T @ grad_z1 + 2.0 * self.lambda_reg * self.w1
            grad_b1 = np.sum(grad_z1, axis=0)

            self.w1, m_w1, v_w1 = _adam_update(self.w1, grad_w1, m_w1, v_w1, step, self.lr, beta1, beta2, eps)
            self.b1, m_b1, v_b1 = _adam_update(self.b1, grad_b1, m_b1, v_b1, step, self.lr, beta1, beta2, eps)
            self.w2, m_w2, v_w2 = _adam_update(self.w2, grad_w2, m_w2, v_w2, step, self.lr, beta1, beta2, eps)
            b2_arr, m_b2_arr, v_b2_arr = _adam_update(
                np.array([self.b2], dtype=np.float64),
                np.array([grad_b2], dtype=np.float64),
                np.array([m_b2], dtype=np.float64),
                np.array([v_b2], dtype=np.float64),
                step,
                self.lr,
                beta1,
                beta2,
                eps,
            )
            self.b2 = float(b2_arr[0])
            m_b2 = float(m_b2_arr[0])
            v_b2 = float(v_b2_arr[0])
            w_skip_arr, m_w_skip_arr, v_w_skip_arr = _adam_update(
                np.array([self.w_skip], dtype=np.float64),
                np.array([grad_w_skip], dtype=np.float64),
                np.array([m_w_skip], dtype=np.float64),
                np.array([v_w_skip], dtype=np.float64),
                step,
                self.lr,
                beta1,
                beta2,
                eps,
            )
            self.w_skip = float(w_skip_arr[0])
            m_w_skip = float(m_w_skip_arr[0])
            v_w_skip = float(v_w_skip_arr[0])

        fit_time = perf_counter() - start
        self.diagnostics = ScalarMLPCalibrationDiagnostics(
            hidden_units=self.hidden_units,
            lambda_reg=self.lambda_reg,
            max_iter=self.max_iter,
            batch_size=batch_size,
            observed_count=int(x_obs.size),
            train_count=int(x_train.size),
            x_mean=float(self.x_mean),
            x_std=float(self.x_std),
            y_mean=float(self.y_mean),
            y_std=float(self.y_std),
            final_loss=float(final_loss),
            fit_time_sec=float(fit_time),
        )
        return self

    def apply(self, x_pred: np.ndarray) -> np.ndarray:
        self._check_fitted()
        x_pred = np.asarray(x_pred, dtype=np.float64)
        x = ((x_pred.reshape(-1, 1) - self.x_mean) / self.x_std)
        h = np.tanh(x @ self.w1 + self.b1)
        y = h @ self.w2 + self.b2 + self.w_skip * x
        y = y.reshape(x_pred.shape) * self.y_std + self.y_mean
        return y.astype(np.float32)

    def get_state_dict(self) -> dict[str, Any]:
        self._check_fitted()
        state = {
            "hidden_units": int(self.hidden_units),
            "lambda_reg": float(self.lambda_reg),
            "lr": float(self.lr),
            "max_iter": int(self.max_iter),
            "batch_size": int(self.batch_size),
            "max_train_samples": int(self.max_train_samples),
            "random_state": int(self.random_state),
            "x_mean": float(self.x_mean),
            "x_std": float(self.x_std),
            "y_mean": float(self.y_mean),
            "y_std": float(self.y_std),
            "w1": np.array(self.w1, dtype=np.float64, copy=True),
            "b1": np.array(self.b1, dtype=np.float64, copy=True),
            "w2": np.array(self.w2, dtype=np.float64, copy=True),
            "b2": float(self.b2),
            "w_skip": float(self.w_skip),
        }
        assert self.diagnostics is not None
        state.update(self.diagnostics.__dict__)
        return state

    def _check_fitted(self) -> None:
        if (
            self.x_mean is None
            or self.x_std is None
            or self.y_mean is None
            or self.y_std is None
            or self.w1 is None
            or self.b1 is None
            or self.w2 is None
            or self.b2 is None
            or self.w_skip is None
        ):
            raise ValueError("ScalarMLPCalibration must be fitted before use.")


def _adam_update(
    param: np.ndarray,
    grad: np.ndarray,
    m: np.ndarray,
    v: np.ndarray,
    step: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = beta1 * m + (1.0 - beta1) * grad
    v = beta2 * v + (1.0 - beta2) * (grad**2)
    m_hat = m / (1.0 - beta1**step)
    v_hat = v / (1.0 - beta2**step)
    param = param - lr * m_hat / (np.sqrt(v_hat) + eps)
    return param, m, v
