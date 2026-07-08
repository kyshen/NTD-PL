"""Scalar link dictionaries for NTD-PL.

The solver uses links that are linear in the coefficients beta:

    f_beta(s) = sum_q beta_q phi_q(s).

Each link provides basis values and derivatives with respect to the latent
scalar s.  Power links reproduce the original polynomial model.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Any, Dict, Optional

import numpy as np
from scipy.linalg import solve_triangular

from .beta import beta_update as beta_update_base


@dataclass
class ScalarLink:
    kind: str
    nested_basis: bool = True

    def fit(self, values: np.ndarray, active_q: int) -> "ScalarLink":
        return self

    def init_beta(self, active_q: int, values: Optional[np.ndarray] = None) -> np.ndarray:
        beta = np.zeros(active_q + 1, dtype=np.float32)
        if active_q >= 1:
            beta[1] = 1.0
        return beta

    def features(self, values: np.ndarray, active_q: int) -> np.ndarray:
        raise NotImplementedError

    def derivative_features(self, values: np.ndarray, active_q: int) -> np.ndarray:
        raise NotImplementedError

    def value(self, values: np.ndarray, beta: np.ndarray) -> np.ndarray:
        shape = values.shape
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        phi = self.features(flat, len(beta) - 1)
        return (phi @ np.asarray(beta, dtype=np.float64)).reshape(shape).astype(np.float32)

    def derivative(self, values: np.ndarray, beta: np.ndarray) -> np.ndarray:
        shape = values.shape
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        dphi = self.derivative_features(flat, len(beta) - 1)
        return (dphi @ np.asarray(beta, dtype=np.float64)).reshape(shape).astype(np.float32)

    def value_and_derivative(self, values: np.ndarray, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.value(values, beta), self.derivative(values, beta)

    def update_beta(
        self,
        X: np.ndarray,
        S: np.ndarray,
        active_q: int,
        lambda_beta: float,
        method: str,
        allow_constant_term: bool = True,
        mask: Optional[np.ndarray] = None,
        stable: bool = True,
    ) -> np.ndarray:
        return beta_update_link(
            X=X,
            S=S,
            link=self,
            active_q=active_q,
            lambda_beta=lambda_beta,
            allow_constant_term=allow_constant_term,
            mask=mask,
        )

    def state_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind}


@dataclass
class PowerLink(ScalarLink):
    kind: str = "power"

    def features(self, values: np.ndarray, active_q: int) -> np.ndarray:
        x = np.asarray(values, dtype=np.float64).reshape(-1)
        phi = np.empty((x.size, active_q + 1), dtype=np.float64)
        phi[:, 0] = 1.0
        for q in range(1, active_q + 1):
            phi[:, q] = phi[:, q - 1] * x
        return phi

    def derivative_features(self, values: np.ndarray, active_q: int) -> np.ndarray:
        x = np.asarray(values, dtype=np.float64).reshape(-1)
        dphi = np.zeros((x.size, active_q + 1), dtype=np.float64)
        if active_q >= 1:
            dphi[:, 1] = 1.0
        for q in range(2, active_q + 1):
            dphi[:, q] = q * np.power(x, q - 1)
        return dphi

    def value_and_derivative(self, values: np.ndarray, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        shape = values.shape
        x = np.asarray(values, dtype=np.float32).reshape(-1)
        coeffs = np.asarray(beta, dtype=np.float32)
        p = len(coeffs) - 1
        if p < 0:
            y = np.zeros_like(x)
            dy = np.zeros_like(x)
            return y.reshape(shape), dy.reshape(shape)
        y = np.full_like(x, coeffs[p], dtype=np.float32)
        dy = np.zeros_like(x)
        for q in range(p - 1, -1, -1):
            dy = dy * x + y
            y = y * x + coeffs[q]
        return y.reshape(shape), dy.reshape(shape)

    def update_beta(
        self,
        X: np.ndarray,
        S: np.ndarray,
        active_q: int,
        lambda_beta: float,
        method: str,
        allow_constant_term: bool = True,
        mask: Optional[np.ndarray] = None,
        stable: bool = True,
    ) -> np.ndarray:
        if not stable:
            return beta_update_base(
                X=X,
                S=S,
                p=active_q,
                lambda_beta=lambda_beta,
                method=method,
                allow_constant_term=allow_constant_term,
                mask=mask,
            )
        if not allow_constant_term:
            return beta_update_base(
                X=X,
                S=S,
                p=active_q,
                lambda_beta=lambda_beta,
                method=method,
                allow_constant_term=False,
                mask=mask,
            )
        if method == "moments_normal_eq":
            return _power_beta_update_moments_cached(X, S, active_q, lambda_beta, mask=mask)
        if method == "ridge_lstsq":
            return _power_beta_update_ridge_lstsq(X, S, active_q, lambda_beta, mask=mask)
        raise ValueError(f"Unknown beta update method: {method}")


@dataclass
class ChebyshevLink(ScalarLink):
    center: float = 0.0
    scale: float = 1.0
    kind: str = "chebyshev"

    def fit(self, values: np.ndarray, active_q: int) -> "ChebyshevLink":
        x = np.asarray(values, dtype=np.float64).reshape(-1)
        if x.size == 0:
            return self
        lo = float(np.percentile(x, 1.0))
        hi = float(np.percentile(x, 99.0))
        self.center = 0.5 * (lo + hi)
        self.scale = max(0.5 * (hi - lo), 1e-6)
        return self

    def init_beta(self, active_q: int, values: Optional[np.ndarray] = None) -> np.ndarray:
        beta = np.zeros(active_q + 1, dtype=np.float32)
        beta[0] = np.float32(self.center)
        if active_q >= 1:
            beta[1] = np.float32(self.scale)
        return beta

    def _z(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float64).reshape(-1) - self.center) / self.scale

    def features(self, values: np.ndarray, active_q: int) -> np.ndarray:
        z = self._z(values)
        phi = np.empty((z.size, active_q + 1), dtype=np.float64)
        phi[:, 0] = 1.0
        if active_q >= 1:
            phi[:, 1] = z
        for q in range(2, active_q + 1):
            phi[:, q] = 2.0 * z * phi[:, q - 1] - phi[:, q - 2]
        return phi

    def derivative_features(self, values: np.ndarray, active_q: int) -> np.ndarray:
        z = self._z(values)
        dphi = np.zeros((z.size, active_q + 1), dtype=np.float64)
        if active_q >= 1:
            dphi[:, 1] = 1.0 / self.scale
        if active_q >= 2:
            # U_0(z)=1, U_1(z)=2z and dT_q/ds = q U_{q-1}(z) / scale.
            u_prev = np.ones_like(z)
            dphi[:, 2] = 2.0 * (2.0 * z) / self.scale
            u_curr = 2.0 * z
            for q in range(3, active_q + 1):
                u_next = 2.0 * z * u_curr - u_prev
                dphi[:, q] = q * u_next / self.scale
                u_prev, u_curr = u_curr, u_next
        return dphi

    def state_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "center": float(self.center), "scale": float(self.scale)}


@dataclass
class RBFLink(ScalarLink):
    centers: Optional[np.ndarray] = None
    width: float = 1.0
    kind: str = "rbf"
    nested_basis: bool = False

    def fit(self, values: np.ndarray, active_q: int) -> "RBFLink":
        x = np.asarray(values, dtype=np.float64).reshape(-1)
        if x.size == 0:
            self.centers = np.zeros(max(active_q - 1, 0), dtype=np.float64)
            self.width = 1.0
            return self
        n_rbf = max(active_q - 1, 0)
        if n_rbf > 0:
            qs = np.linspace(5.0, 95.0, n_rbf)
            self.centers = np.percentile(x, qs).astype(np.float64)
            if n_rbf > 1:
                spacing = np.diff(self.centers)
                self.width = max(float(np.median(np.abs(spacing))), 1e-6)
            else:
                self.width = max(float(np.std(x)), 1e-6)
        else:
            self.centers = np.zeros(0, dtype=np.float64)
            self.width = 1.0
        return self

    def init_beta(self, active_q: int, values: Optional[np.ndarray] = None) -> np.ndarray:
        if values is None:
            return super().init_beta(active_q, values)
        x = np.asarray(values, dtype=np.float64).reshape(-1)
        phi = self.features(x, active_q)
        beta, *_ = np.linalg.lstsq(phi, x, rcond=None)
        return beta.astype(np.float32)

    def features(self, values: np.ndarray, active_q: int) -> np.ndarray:
        x = np.asarray(values, dtype=np.float64).reshape(-1)
        phi = np.empty((x.size, active_q + 1), dtype=np.float64)
        phi[:, 0] = 1.0
        if active_q >= 1:
            phi[:, 1] = x
        centers = np.asarray(self.centers if self.centers is not None else [], dtype=np.float64)
        n_rbf = min(max(active_q - 1, 0), centers.size)
        for j in range(n_rbf):
            q = j + 2
            diff = x - centers[j]
            phi[:, q] = np.exp(-0.5 * (diff / self.width) ** 2)
        for q in range(2 + n_rbf, active_q + 1):
            phi[:, q] = 0.0
        return phi

    def derivative_features(self, values: np.ndarray, active_q: int) -> np.ndarray:
        x = np.asarray(values, dtype=np.float64).reshape(-1)
        dphi = np.zeros((x.size, active_q + 1), dtype=np.float64)
        if active_q >= 1:
            dphi[:, 1] = 1.0
        centers = np.asarray(self.centers if self.centers is not None else [], dtype=np.float64)
        n_rbf = min(max(active_q - 1, 0), centers.size)
        inv_w2 = 1.0 / (self.width * self.width)
        for j in range(n_rbf):
            q = j + 2
            diff = x - centers[j]
            basis = np.exp(-0.5 * (diff / self.width) ** 2)
            dphi[:, q] = -diff * inv_w2 * basis
        return dphi

    def state_dict(self) -> Dict[str, Any]:
        centers = [] if self.centers is None else [float(c) for c in self.centers]
        return {"kind": self.kind, "centers": centers, "width": float(self.width)}


@dataclass
class LinearSplineLink(ScalarLink):
    knots: Optional[np.ndarray] = None
    kind: str = "spline"
    nested_basis: bool = False

    def fit(self, values: np.ndarray, active_q: int) -> "LinearSplineLink":
        x = np.asarray(values, dtype=np.float64).reshape(-1)
        n_knots = max(active_q + 1, 2)
        if x.size == 0:
            self.knots = np.linspace(-1.0, 1.0, n_knots, dtype=np.float64)
            return self

        knots = np.percentile(x, np.linspace(0.0, 100.0, n_knots)).astype(np.float64)
        knots = _strictly_increasing_knots(knots)
        if knots.size != n_knots:
            lo = float(np.min(x))
            hi = float(np.max(x))
            if hi - lo < 1e-8:
                lo -= 0.5
                hi += 0.5
            knots = np.linspace(lo, hi, n_knots, dtype=np.float64)
        self.knots = knots
        return self

    def init_beta(self, active_q: int, values: Optional[np.ndarray] = None) -> np.ndarray:
        knots = self._knots(active_q)
        return knots.astype(np.float32)

    def _knots(self, active_q: int) -> np.ndarray:
        n_knots = max(active_q + 1, 2)
        if self.knots is None or len(self.knots) != n_knots:
            return np.linspace(-1.0, 1.0, n_knots, dtype=np.float64)
        return np.asarray(self.knots, dtype=np.float64)

    def features(self, values: np.ndarray, active_q: int) -> np.ndarray:
        x = np.asarray(values, dtype=np.float64).reshape(-1)
        knots = self._knots(active_q)
        n_knots = knots.size
        phi = np.zeros((x.size, n_knots), dtype=np.float64)

        idx = np.searchsorted(knots, x, side="right") - 1
        idx = np.clip(idx, 0, n_knots - 2)
        left = knots[idx]
        right = knots[idx + 1]
        denom = np.maximum(right - left, 1e-12)
        weight_right = np.clip((x - left) / denom, 0.0, 1.0)
        weight_left = 1.0 - weight_right

        rows = np.arange(x.size)
        phi[rows, idx] = weight_left
        phi[rows, idx + 1] = weight_right
        below = x <= knots[0]
        above = x >= knots[-1]
        phi[below, :] = 0.0
        phi[below, 0] = 1.0
        phi[above, :] = 0.0
        phi[above, -1] = 1.0
        return phi

    def derivative_features(self, values: np.ndarray, active_q: int) -> np.ndarray:
        x = np.asarray(values, dtype=np.float64).reshape(-1)
        knots = self._knots(active_q)
        n_knots = knots.size
        dphi = np.zeros((x.size, n_knots), dtype=np.float64)

        interior = (x > knots[0]) & (x < knots[-1])
        if not np.any(interior):
            return dphi

        x_int = x[interior]
        idx = np.searchsorted(knots, x_int, side="right") - 1
        idx = np.clip(idx, 0, n_knots - 2)
        denom = np.maximum(knots[idx + 1] - knots[idx], 1e-12)
        rows = np.flatnonzero(interior)
        dphi[rows, idx] = -1.0 / denom
        dphi[rows, idx + 1] = 1.0 / denom
        return dphi

    def state_dict(self) -> Dict[str, Any]:
        knots = [] if self.knots is None else [float(k) for k in self.knots]
        return {"kind": self.kind, "knots": knots}


def make_link(kind: str = "power", state: Optional[Dict[str, Any]] = None) -> ScalarLink:
    link_kind = str(kind or "power").lower()
    if state is not None:
        link_kind = str(state.get("kind", link_kind)).lower()
    if link_kind in {"power", "poly", "polynomial"}:
        return PowerLink()
    if link_kind in {"chebyshev", "cheb"}:
        if state is None:
            return ChebyshevLink()
        return ChebyshevLink(center=float(state.get("center", 0.0)), scale=float(state.get("scale", 1.0)))
    if link_kind == "rbf":
        if state is None:
            return RBFLink()
        return RBFLink(
            centers=np.asarray(state.get("centers", []), dtype=np.float64),
            width=float(state.get("width", 1.0)),
        )
    if link_kind in {"spline", "linear_spline", "linear-spline"}:
        if state is None:
            return LinearSplineLink()
        return LinearSplineLink(knots=np.asarray(state.get("knots", []), dtype=np.float64))
    raise ValueError(f"Unknown scalar link kind: {kind}")


def _strictly_increasing_knots(knots: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    values = np.asarray(knots, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return values
    out = [float(values[0])]
    for value in values[1:]:
        if float(value) > out[-1] + eps:
            out.append(float(value))
    return np.asarray(out, dtype=np.float64)


def beta_update_link(
    X: np.ndarray,
    S: np.ndarray,
    link: ScalarLink,
    active_q: int,
    lambda_beta: float = 0.0,
    allow_constant_term: bool = True,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Ridge update for any link that is linear in beta."""
    Xv = np.asarray(X, dtype=np.float64).reshape(-1)
    Sv = np.asarray(S, dtype=np.float64).reshape(-1)

    if mask is not None:
        mv = np.asarray(mask).reshape(-1).astype(bool)
        Xv = Xv[mv]
        Sv = Sv[mv]

    if Sv.size == 0:
        raise ValueError("No valid entries available for beta update.")

    phi = link.features(Sv, active_q)
    if not allow_constant_term:
        phi_fit = phi[:, 1:]
    else:
        phi_fit = phi

    d = phi_fit.shape[1]
    if d == 0:
        return np.zeros(active_q + 1, dtype=np.float32)

    gram = phi_fit.T @ phi_fit
    rhs = phi_fit.T @ Xv
    if lambda_beta > 0.0:
        gram = gram + lambda_beta * np.eye(d, dtype=np.float64)

    try:
        coeff = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        coeff, *_ = np.linalg.lstsq(gram, rhs, rcond=None)

    if allow_constant_term:
        return coeff.astype(np.float32)

    beta = np.zeros(active_q + 1, dtype=np.float32)
    beta[1:] = coeff.astype(np.float32)
    return beta


def _build_power_change_of_basis(mu: float, sigma: float, degree: int) -> np.ndarray:
    """Build T so beta = T @ gamma maps standardized powers back to powers of S."""
    transform = np.zeros((degree + 1, degree + 1), dtype=np.float64)
    for j in range(degree + 1):
        for k in range(j, degree + 1):
            transform[j, k] = comb(k, j) * ((-mu) ** (k - j)) / (sigma ** k)
    return transform


def _power_standardized_observations(
    X: np.ndarray,
    S: np.ndarray,
    mask: Optional[np.ndarray],
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    Xv = np.asarray(X, dtype=np.float64).reshape(-1)
    Sv = np.asarray(S, dtype=np.float64).reshape(-1)
    if mask is not None:
        mv = np.asarray(mask).reshape(-1).astype(bool)
        Xv = Xv[mv]
        Sv = Sv[mv]
    if Sv.size == 0:
        raise ValueError("No valid entries available for beta update.")
    mu = float(Sv.mean())
    sigma = float(Sv.std())
    if sigma < eps:
        sigma = 1.0
    return Xv, (Sv - mu) / sigma, mu, sigma


def _power_beta_update_moments_cached(
    X: np.ndarray,
    S: np.ndarray,
    degree: int,
    lambda_beta: float = 0.0,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    if degree < 0:
        raise ValueError("degree must be >= 0")
    Xv, Zv, mu, sigma = _power_standardized_observations(X, S, mask)

    max_pow = 2 * degree
    powers = np.empty(max_pow + 1, dtype=np.float64)
    powers[0] = len(Zv)
    if max_pow >= 1:
        powers[1] = np.sum(Zv)
        z_power = Zv.copy()
        for m in range(2, max_pow + 1):
            z_power *= Zv
            powers[m] = np.sum(z_power)

    dim = degree + 1
    gram = np.empty((dim, dim), dtype=np.float64)
    rhs = np.empty(dim, dtype=np.float64)
    for i in range(dim):
        for j in range(dim):
            gram[i, j] = powers[i + j]

    rhs[0] = np.sum(Xv)
    if dim > 1:
        z_power_rhs = Zv.copy()
        for i in range(1, dim):
            rhs[i] = np.sum(Xv * z_power_rhs)
            z_power_rhs *= Zv

    transform = _build_power_change_of_basis(mu, sigma, degree)
    if lambda_beta > 0.0:
        gram = gram + lambda_beta * (transform.T @ transform)

    try:
        chol = np.linalg.cholesky(gram)
        y = solve_triangular(chol, rhs, lower=True)
        gamma = solve_triangular(chol.T, y, lower=False)
    except np.linalg.LinAlgError:
        gamma = np.linalg.solve(gram, rhs)

    return (transform @ gamma).astype(np.float32)


def _power_beta_update_ridge_lstsq(
    X: np.ndarray,
    S: np.ndarray,
    degree: int,
    lambda_beta: float = 0.0,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    if degree < 0:
        raise ValueError("degree must be >= 0")
    Xv, Zv, mu, sigma = _power_standardized_observations(X, S, mask)

    dim = degree + 1
    phi = np.empty((Zv.size, dim), dtype=np.float64)
    phi[:, 0] = 1.0
    if degree >= 1:
        phi[:, 1] = Zv
        for k in range(2, dim):
            phi[:, k] = phi[:, k - 1] * Zv

    transform = _build_power_change_of_basis(mu, sigma, degree)
    if lambda_beta > 0.0:
        gram = phi.T @ phi + lambda_beta * (transform.T @ transform)
        rhs = phi.T @ Xv
        gamma = np.linalg.solve(gram, rhs)
    else:
        gamma, *_ = np.linalg.lstsq(phi, Xv, rcond=None)

    return (transform @ gamma).astype(np.float32)
