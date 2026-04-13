"""
NTDPL beta update utilities.

This module provides two polynomial coefficient solvers:
1. `moments_normal_eq`: moment-based normal equations
2. `ridge_lstsq`: ridge least-squares on the explicit Vandermonde design
"""

from typing import Optional

import numpy as np


def beta_update(
    X,
    S,
    p: int,
    lambda_beta: float,
    method: str,
    allow_constant_term: bool = True,
    mask: Optional[np.ndarray] = None,
):
    """Dispatch beta updates to the requested solver."""
    if method == "moments_normal_eq":
        return beta_update_moments_normal_eq(
            X,
            S,
            p,
            lambda_beta,
            allow_constant_term=allow_constant_term,
            mask=mask,
        )
    if method == "ridge_lstsq":
        return beta_update_ridge_lstsq(
            X,
            S,
            p,
            lambda_beta,
            allow_constant_term=allow_constant_term,
            mask=mask,
        )
    raise ValueError(f"Unknown beta update method: {method}")


def beta_update_moments_normal_eq(
    X,
    S,
    p: int,
    lambda_beta: float = 0.0,
    allow_constant_term: bool = True,
    mask: Optional[np.ndarray] = None,
):
    """Solve beta with moment equations, optionally fixing beta_0 = 0."""
    if p < 0:
        raise ValueError("p must be >= 0")

    Xv = np.asarray(X, dtype=np.float64).reshape(-1)
    Sv = np.asarray(S, dtype=np.float64).reshape(-1)

    if mask is not None:
        mv = np.asarray(mask).reshape(-1).astype(bool)
        Xv = Xv[mv]
        Sv = Sv[mv]

    degrees = list(range(0, p + 1)) if allow_constant_term else list(range(1, p + 1))
    if not degrees:
        return np.zeros(1, dtype=np.float32)

    max_pow = degrees[-1] + degrees[-1]
    pow_s = np.empty(max_pow + 1, dtype=np.float64)
    pow_s[0] = Sv.size
    if max_pow >= 1:
        pow_s[1] = Sv.sum()
        for degree in range(2, max_pow + 1):
            pow_s[degree] = np.sum(Sv ** degree)

    d = len(degrees)
    b = np.empty(d, dtype=np.float64)
    for i, degree in enumerate(degrees):
        if degree == 0:
            b[i] = Xv.sum()
        else:
            b[i] = np.sum(Xv * (Sv ** degree))

    M = np.empty((d, d), dtype=np.float64)
    for i, left_degree in enumerate(degrees):
        for j, right_degree in enumerate(degrees):
            M[i, j] = pow_s[left_degree + right_degree]

    if lambda_beta > 0.0:
        M = M + lambda_beta * np.eye(d, dtype=np.float64)

    beta = np.linalg.solve(M, b).astype(np.float32)
    if allow_constant_term:
        return beta

    full_beta = np.zeros(p + 1, dtype=np.float32)
    full_beta[1:] = beta
    return full_beta


def beta_update_ridge_lstsq(
    X,
    S,
    p: int,
    lambda_beta: float = 0.0,
    allow_constant_term: bool = True,
    mask: Optional[np.ndarray] = None,
):
    """Solve beta with ridge least squares, optionally fixing beta_0 = 0."""
    if p < 0:
        raise ValueError("p must be >= 0")

    Xv = np.asarray(X, dtype=np.float64).reshape(-1)
    Sv = np.asarray(S, dtype=np.float64).reshape(-1)

    if mask is not None:
        mv = np.asarray(mask).reshape(-1).astype(bool)
        Xv = Xv[mv]
        Sv = Sv[mv]

    degrees = list(range(0, p + 1)) if allow_constant_term else list(range(1, p + 1))
    d = len(degrees)
    if d == 0:
        return np.zeros(1, dtype=np.float32)

    Phi = np.empty((Sv.size, d), dtype=np.float64)
    for idx, degree in enumerate(degrees):
        if degree == 0:
            Phi[:, idx] = 1.0
        elif degree == 1:
            Phi[:, idx] = Sv
        else:
            Phi[:, idx] = Sv ** degree

    if lambda_beta > 0.0:
        A = np.vstack([Phi, np.sqrt(lambda_beta) * np.eye(d, dtype=np.float64)])
        b = np.concatenate([Xv, np.zeros(d, dtype=np.float64)])
    else:
        A, b = Phi, Xv

    beta, *_ = np.linalg.lstsq(A, b, rcond=None)
    beta = beta.astype(np.float32)
    if allow_constant_term:
        return beta

    full_beta = np.zeros(p + 1, dtype=np.float32)
    full_beta[1:] = beta
    return full_beta
