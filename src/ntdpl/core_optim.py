"""
Optimized NTDPL implementation
Focuses on:
1. Avoiding duplicate Tucker reconstruction
2. Fusing polynomial and derivative computation
3. Caching powers in beta update
4. JIT compilation with Numba for hot paths

This version additionally stabilizes beta updates by:
- solving on centered/scaled latent variable Z = (S - mu) / sigma,
- then converting the solved coefficients back to the ORIGINAL power basis of S.

So the outer model, forward pass, derivative computation, logging, and history
all remain unchanged:
    X_hat = sum_j beta_j * S^j
"""

from typing import Dict, List, Optional, Tuple
from math import comb
import numpy as np
import tensorly as tl
from scipy.linalg import solve_triangular
from tensorly.decomposition import tucker
from tensorly.tucker_tensor import tucker_to_tensor
from tensorly.tenalg.core_tenalg import multi_mode_dot

from src.utils.completion_ops import mask_to_bool, mask_to_float
from .beta import beta_update as beta_update_base
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(func):
        return func


# ============================================================
# Optimized Polynomial Operations with Numba JIT
# ============================================================

if NUMBA_AVAILABLE:
    @njit
    def _poly_and_deriv_fused_jit(S_flat: np.ndarray, beta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute both polynomial and its derivative simultaneously.
        More efficient than computing separately due to shared computation.

        Returns
        -------
        y : np.ndarray
            Polynomial values
        dy : np.ndarray
            Derivative values
        """
        p = len(beta) - 1
        n = len(S_flat)
        y = np.empty(n, dtype=np.float32)
        dy = np.empty(n, dtype=np.float32)

        if p < 0:
            return y, dy

        # Horner initialization
        for i in range(n):
            y[i] = beta[p]
            dy[i] = 0.0

        # Simultaneous Horner evaluation:
        # dy <- dy * x + y_prev
        # y  <- y  * x + beta[k]
        for k in range(p - 1, -1, -1):
            for i in range(n):
                dy[i] = dy[i] * S_flat[i] + y[i]
                y[i] = y[i] * S_flat[i] + beta[k]

        return y, dy

else:
    # Fallback versions (non-JIT) for when Numba is unavailable
    def _poly_and_deriv_fused_jit(S_flat: np.ndarray, beta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        S_flat = np.asarray(S_flat, dtype=np.float32)
        beta = np.asarray(beta, dtype=np.float32)
        p = len(beta) - 1
        if p < 0:
            return np.zeros_like(S_flat), np.zeros_like(S_flat)

        y = np.full_like(S_flat, beta[p], dtype=np.float32)
        dy = np.zeros_like(S_flat)
        for k in range(p - 1, -1, -1):
            dy = dy * S_flat + y
            y = y * S_flat + beta[k]
        return y, dy


def _poly_and_deriv_fused(S: np.ndarray, beta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute both polynomial and derivative simultaneously.

    Optimization: Fused computation saves memory bandwidth and shares
    intermediate calculations between y and dy.

    Returns
    -------
    y : np.ndarray
        Polynomial values
    dy : np.ndarray
        Derivative values
    """
    original_shape = S.shape
    S_flat = S.ravel().astype(np.float32)
    y_flat, dy_flat = _poly_and_deriv_fused_jit(S_flat, beta)
    return y_flat.reshape(original_shape), dy_flat.reshape(original_shape)


# ============================================================
# Change of Basis Utilities
# ============================================================


def _build_change_of_basis(mu: float, sigma: float, p: int) -> np.ndarray:
    """
    Build T such that:
        beta = T @ gamma
    and
        sum_k gamma_k * ((s - mu) / sigma)^k
      = sum_j beta_j * s^j

    Therefore, gamma is solved on the centered/scaled basis Z = (S - mu)/sigma,
    while beta remains in the original power basis of S.
    """
    T = np.zeros((p + 1, p + 1), dtype=np.float64)
    for j in range(p + 1):
        for k in range(j, p + 1):
            T[j, k] = comb(k, j) * ((-mu) ** (k - j)) / (sigma ** k)
    return T


# ============================================================
# Optimized Beta Update with Power Caching
# ============================================================


def beta_update_moments_powers_cached(
    X: np.ndarray,
    S: np.ndarray,
    p: int,
    lambda_beta: float = 0.0,
    mask: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Beta update using moment equations with cached powers.

    This version solves in the centered/scaled variable
        Z = (S - mu) / sigma,
    and then converts the result back to the original power basis of S.

    If lambda_beta > 0, the ridge penalty is kept EXACTLY consistent with
    penalizing ||beta||^2 in the original basis:
        beta = T @ gamma
        ||beta||^2 = gamma^T (T^T T) gamma

    Parameters
    ----------
    X : np.ndarray
        Target tensor.
    S : np.ndarray
        Tucker tensor (reconstructed latent tensor).
    p : int
        Polynomial degree.
    lambda_beta : float
        Regularization strength on beta in the original basis.
    mask : np.ndarray, optional
        Boolean mask for observed entries.
    eps : float
        Numerical floor for sigma.

    Returns
    -------
    beta : np.ndarray
        Polynomial coefficients [beta_0, beta_1, ..., beta_p]
        in the ORIGINAL power basis of S.
    """
    if p < 0:
        raise ValueError("p must be >= 0")

    Xv = np.asarray(X, dtype=np.float64).ravel()
    Sv = np.asarray(S, dtype=np.float64).ravel()

    if mask is not None:
        mv = np.asarray(mask).ravel().astype(bool)
        Xv = Xv[mv]
        Sv = Sv[mv]

    if Sv.size == 0:
        raise ValueError("No valid entries available for beta update.")

    mu = float(Sv.mean())
    sigma = float(Sv.std())
    if sigma < eps:
        sigma = 1.0

    Zv = (Sv - mu) / sigma

    # Cache powers of Z instead of S.
    max_pow = 2 * p
    powers = np.empty(max_pow + 1, dtype=np.float64)
    powers[0] = len(Zv)
    if max_pow >= 1:
        powers[1] = np.sum(Zv)
        Zv_power = Zv.copy()
        for m in range(2, max_pow + 1):
            Zv_power *= Zv
            powers[m] = np.sum(Zv_power)

    d = p + 1
    M = np.empty((d, d), dtype=np.float64)
    b = np.empty(d, dtype=np.float64)

    for i in range(d):
        for j in range(d):
            M[i, j] = powers[i + j]

    b[0] = np.sum(Xv)
    if d > 1:
        Zv_power_for_b = Zv.copy()
        for i in range(1, d):
            b[i] = np.sum(Xv * Zv_power_for_b)
            Zv_power_for_b *= Zv

    T = _build_change_of_basis(mu, sigma, p)

    if lambda_beta > 0.0:
        M = M + lambda_beta * (T.T @ T)

    try:
        L = np.linalg.cholesky(M)
        y = solve_triangular(L, b, lower=True)
        gamma = solve_triangular(L.T, y, lower=False)
    except np.linalg.LinAlgError:
        gamma = np.linalg.solve(M, b)

    beta = T @ gamma
    return beta.astype(np.float32)


# ============================================================
# Main Optimized Solver
# ============================================================


def ntdpl_optimized(
    X,
    rank,
    init_n_iter_max: int,
    p_max: int,
    allow_constant_term: bool,
    n_iter_max: int,
    use_continuation: bool,
    factor_normalize: bool,
    lr_core: float,
    lr_factors: float,
    lambda_core: float,
    lambda_factors: float,
    lambda_beta: float,
    beta_update_method: str,
    init: str,
    random_state: int,
    beta_update_interval: int,
    stable_beta_update: bool,
    beta_update_stage: str,
    return_history: bool,
    mask: Optional[np.ndarray] = None,
):
    """
    Optimized NTD-PL with the following improvements:

    1. Avoid duplicate Tucker reconstruction
    2. Fuse polynomial and derivative computation
    3. Cache powers in beta update
    4. JIT compilation for hot paths
    5. Keep normalization behavior aligned with base NTD-PL
    6. Stabilize beta update via centered/scaled latent regression

    Expected speedup: 3-5x depending on tensor size and rank.
    """
    X = np.asarray(X, dtype=np.float32)
    mask_bool = mask_to_bool(mask, X.shape) if mask is not None else None
    mask_float = mask_to_float(mask, X.shape, dtype=X.dtype) if mask is not None else None

    if mask_float is None:
        fit_scale = np.float32(1.0 / X.size)
    else:
        n_obs = float(mask_float.sum())
        if n_obs <= 0:
            raise ValueError("`mask` contains no observed entries.")
        fit_scale = np.float32(1.0 / n_obs)

    # --------------------------------------------------------
    # Initialization
    # --------------------------------------------------------
    core, factors = _init_ntdpl_factors(
        X=X,
        rank=rank,
        init=init,
        init_n_iter_max=init_n_iter_max,
        mask_float=mask_float,
        random_state=random_state,
    )
    if factor_normalize:
        core, factors = normalize_tucker(core, factors)

    if p_max < 1:
        raise ValueError("`p_max` must be >= 1.")
    if beta_update_interval < 1:
        raise ValueError("`beta_update_interval` must be >= 1.")
    beta_stage = str(beta_update_stage).lower()
    if beta_stage not in {"before_grad", "after_grad"}:
        raise ValueError("`beta_update_stage` must be either 'before_grad' or 'after_grad'.")

    # beta are polynomial coefficients on the ORIGINAL power basis of S.
    if use_continuation:
        p = 1
        beta = np.zeros(p + 1, dtype=np.float32)
        beta[0] = 0.0
        beta[1] = 1.0
        continuation_schedule = _build_continuation_schedule(n_iter_max, p_max)
        continuation_idx = 0
    else:
        p = p_max
        beta = np.zeros(p + 1, dtype=np.float32)
        beta[0] = 0.0
        beta[1] = 1.0
        continuation_schedule = []
        continuation_idx = 0

    st_core = _adam_init(core.shape)
    st_factors = [_adam_init(f.shape) for f in factors]

    history: List[Dict[str, float]] = []

    def _append_history() -> None:
        S_hist = tucker_to_tensor((core, factors))
        X_hat, _ = _poly_and_deriv_fused(S_hist, beta)
        err = _rmse_on_target(X, X_hat, mask=mask_bool)
        record = {
            "p": int(p),
            "error": float(err),
            **_beta_to_dict(beta, p_max),
        }
        if mask_bool is None:
            record["RMSE"] = float(err)
        else:
            record["RMSE_obs"] = float(err)
        history.append(record)

    if return_history:
        _append_history()

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------
    N = X.ndim
    modes_all = list(range(N))

    for it in range(1, n_iter_max + 1):
        if use_continuation:
            while continuation_idx < len(continuation_schedule) and it >= continuation_schedule[continuation_idx]:
                p += 1
                beta_new = np.zeros(p + 1, dtype=np.float32)
                beta_new[:p] = beta
                beta_new[p] = 0.0
                beta = beta_new
                continuation_idx += 1

        # Reconstruct Tucker tensor once per iteration.
        S = tucker_to_tensor((core, factors))

        # Stable beta updates are solved on a centered/scaled latent basis and
        # then mapped back to the original power basis of S. When needed, we
        # can explicitly fall back to the original solver in `beta.py`.
        if beta_stage == "before_grad" and (it % beta_update_interval) == 0:
            if stable_beta_update:
                beta = beta_update(
                    X=X,
                    S=S,
                    p=p,
                    lambda_beta=lambda_beta,
                    method=beta_update_method,
                    allow_constant_term=allow_constant_term,
                    mask=mask_bool,
                )
            else:
                beta = beta_update_base(
                    X=X,
                    S=S,
                    p=p,
                    lambda_beta=lambda_beta,
                    method=beta_update_method,
                    allow_constant_term=allow_constant_term,
                    mask=mask_bool,
                )

        # Always use fused polynomial and derivative computation.
        Xhat, dfdS = _poly_and_deriv_fused(S, beta)

        # Masked residual for completion, full residual for decomposition.
        E = Xhat - X
        if mask_float is not None:
            E = E * mask_float
        E = E * fit_scale
        T = E * dfdS

        # core gradient
        grad_core = multi_mode_dot(T, [f.T for f in factors], modes=modes_all)
        grad_core = grad_core.astype(np.float32) + lambda_core * core
        _adam_step(core, grad_core, st_core, b1=0.9, b2=0.999, lr=lr_core, eps=1e-8)

        # factor gradients
        for n in range(N):
            other_modes = [k for k in range(N) if k != n]
            M = multi_mode_dot(core, [factors[k] for k in other_modes], modes=other_modes)
            Z = tl.unfold(M, mode=n)
            Tn = tl.unfold(T, mode=n)
            grad_A = np.dot(Tn, Z.T)
            grad_A = grad_A.astype(np.float32) + lambda_factors * factors[n]
            _adam_step(factors[n], grad_A, st_factors[n], b1=0.9, b2=0.999, lr=lr_factors, eps=1e-8)

        if factor_normalize:
            core, factors = normalize_tucker(core, factors)

        if beta_stage == "after_grad" and (it % beta_update_interval) == 0:
            S_beta = tucker_to_tensor((core, factors))
            if stable_beta_update:
                beta = beta_update(
                    X=X,
                    S=S_beta,
                    p=p,
                    lambda_beta=lambda_beta,
                    method=beta_update_method,
                    allow_constant_term=allow_constant_term,
                    mask=mask_bool,
                )
            else:
                beta = beta_update_base(
                    X=X,
                    S=S_beta,
                    p=p,
                    lambda_beta=lambda_beta,
                    method=beta_update_method,
                    allow_constant_term=allow_constant_term,
                    mask=mask_bool,
                )

        if return_history:
            _append_history()

    result = (core, factors, beta)
    if return_history:
        return result, history
    return result


# ============================================================
# Helper functions
# ============================================================


def _init_ntdpl_factors(
    X,
    rank,
    init: str,
    init_n_iter_max: int,
    mask_float: Optional[np.ndarray],
    random_state: Optional[int],
):
    init_name = str(init).lower()
    if init_name == "tucker":
        tucker_mask = None if mask_float is None else mask_float
        core, factors = tucker(
            X,
            rank=rank,
            n_iter_max=init_n_iter_max,
            init="svd",
            mask=tucker_mask,
            random_state=random_state,
        )
        core = np.asarray(core, dtype=np.float32)
        factors = [np.asarray(f, dtype=np.float32) for f in factors]
        return core, factors
    if init_name == "random":
        rng = np.random.default_rng(random_state)
        tensor_shape = tuple(int(dim) for dim in np.asarray(X).shape)
        rank_shape = tuple(int(r) for r in rank)
        core = rng.normal(size=rank_shape).astype(np.float32)
        factors = [
            rng.normal(size=(mode_dim, mode_rank)).astype(np.float32)
            for mode_dim, mode_rank in zip(tensor_shape, rank_shape)
        ]
        return core, factors
    raise ValueError(f"Unsupported init for NTD-PL: {init}")



def _rmse_on_target(X, Xhat, mask: Optional[np.ndarray] = None) -> float:
    X = np.asarray(X)
    Xhat = np.asarray(Xhat)

    if mask is None:
        return float(np.sqrt(np.mean((Xhat - X) ** 2)))

    diff = Xhat[mask] - X[mask]
    if diff.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(diff ** 2)))



def beta_update(
    X,
    S,
    p: int,
    lambda_beta: float,
    method: str,
    allow_constant_term: bool = True,
    mask: Optional[np.ndarray] = None,
):
    """Beta update where returned beta is always in the original power basis of S."""
    if not allow_constant_term:
        return beta_update_base(
            X=X,
            S=S,
            p=p,
            lambda_beta=lambda_beta,
            method=method,
            allow_constant_term=False,
            mask=mask,
        )
    if method == "moments_normal_eq":
        return beta_update_moments_powers_cached(X, S, p, lambda_beta, mask=mask)
    if method == "ridge_lstsq":
        return beta_update_ridge_lstsq(X, S, p, lambda_beta, mask=mask)
    raise ValueError(f"Unknown beta update method: {method}")



def beta_update_ridge_lstsq(
    X,
    S,
    p: int,
    lambda_beta: float = 0.0,
    mask: Optional[np.ndarray] = None,
    eps: float = 1e-12,
):
    """
    Ridge least-squares beta update solved on centered/scaled latent variable,
    but returned in the ORIGINAL power basis of S.

    We solve gamma in
        X ≈ sum_k gamma_k * ((S - mu) / sigma)^k
    and then convert exactly to beta such that
        X ≈ sum_j beta_j * S^j

    If lambda_beta > 0, the regularization remains exactly equivalent to
    penalizing ||beta||^2 in the original basis.
    """
    if p < 0:
        raise ValueError("p must be >= 0")

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

    Zv = (Sv - mu) / sigma

    n = Zv.size
    d = p + 1

    Phi_tilde = np.empty((n, d), dtype=np.float64)
    Phi_tilde[:, 0] = 1.0
    if p >= 1:
        Phi_tilde[:, 1] = Zv
        for k in range(2, d):
            Phi_tilde[:, k] = Phi_tilde[:, k - 1] * Zv

    T = _build_change_of_basis(mu, sigma, p)

    if lambda_beta > 0.0:
        A = Phi_tilde.T @ Phi_tilde + lambda_beta * (T.T @ T)
        b = Phi_tilde.T @ Xv
        gamma = np.linalg.solve(A, b)
    else:
        gamma, *_ = np.linalg.lstsq(Phi_tilde, Xv, rcond=None)

    beta = T @ gamma
    return beta.astype(np.float32)



def normalize_tucker(
    core,
    factors,
    eps: float = 1e-12,
):
    core_new = np.asarray(core, dtype=np.float32)
    factors_new = []

    for n, A in enumerate(factors):
        A = np.asarray(A, dtype=np.float32)
        s = np.linalg.norm(A, axis=0)
        s_safe = np.maximum(s, eps)

        A_norm = A / s_safe[None, :]
        core_new = _mode_scale_core(core_new, s_safe, mode=n)
        factors_new.append(A_norm)

    return core_new, factors_new

def _mode_scale_core(core, scale, mode: int) -> np.ndarray:
    X = np.moveaxis(core, mode, 0)
    X = X * scale.reshape((-1,) + (1,) * (X.ndim - 1))
    return np.moveaxis(X, 0, mode)



def _build_continuation_schedule(n_iter_max: int, p_max: int) -> List[int]:
    """Build iteration milestones for polynomial degree increase."""
    if p_max <= 1 or n_iter_max <= 0:
        return []

    raw = [(k * n_iter_max) / p_max for k in range(1, p_max)]
    schedule: List[int] = []
    last = 0
    for val in raw:
        it = int(np.round(val))
        if it > last:
            schedule.append(it)
            last = it
    return schedule



def _beta_to_dict(beta, p_max: int) -> Dict[str, float]:
    out = {}
    for i in range(p_max + 1):
        out[f"beta_{i}"] = float(beta[i]) if i < len(beta) else 0.0
    return out



def _adam_init(shape):
    return {
        "m": np.zeros(shape, dtype=np.float32),
        "v": np.zeros(shape, dtype=np.float32),
        "t": 0,
    }



def _adam_step(param, grad, state, b1, b2, lr, eps):
    state["t"] += 1
    t = state["t"]
    state["m"] = b1 * state["m"] + (1 - b1) * grad
    state["v"] = b2 * state["v"] + (1 - b2) * (grad * grad)
    mhat = state["m"] / (1 - b1 ** t)
    vhat = state["v"] / (1 - b2 ** t)
    param -= lr * mhat / (np.sqrt(vhat) + eps)


__all__ = ["ntdpl_optimized"]
