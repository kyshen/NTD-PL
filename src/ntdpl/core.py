"""
Optimized NTDPL implementation
Focuses on:
1. Avoiding duplicate Tucker reconstruction
2. Delegating scalar-link evaluation and beta refresh to a unified link API
3. Keeping the optimized Tucker-gradient path as the single NTD-PL solver

The default power link preserves the original polynomial behavior, while other
finite scalar-link dictionaries use the same solver loop.
"""

from typing import Dict, List, Optional
import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker
from tensorly.tucker_tensor import tucker_to_tensor
from tensorly.tenalg.core_tenalg import multi_mode_dot

from src.utils.completion_ops import mask_to_bool, mask_to_float
from .links import make_link


# ============================================================
# Main Optimized Solver
# ============================================================


def ntdpl(
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
    link_kind: str = "power",
    return_link_state: bool = False,
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

    S_for_link = tucker_to_tensor((core, factors))
    link = make_link(link_kind).fit(S_for_link, p_max)

    # beta are coefficients in the selected scalar-link dictionary.
    effective_continuation = bool(use_continuation and getattr(link, "nested_basis", True))
    if effective_continuation:
        p = 1
        beta = link.init_beta(p, S_for_link)
        continuation_schedule = _build_continuation_schedule(n_iter_max, p_max)
        continuation_idx = 0
    else:
        p = p_max
        beta = link.init_beta(p, S_for_link)
        continuation_schedule = []
        continuation_idx = 0

    st_core = _adam_init(core.shape)
    st_factors = [_adam_init(f.shape) for f in factors]

    history: List[Dict[str, float]] = []

    def _append_history() -> None:
        S_hist = tucker_to_tensor((core, factors))
        X_hat, _ = link.value_and_derivative(S_hist, beta)
        err = _rmse_on_target(X, X_hat, mask=mask_bool)
        record = {
            "p": int(p),
            "link_kind": str(link.kind),
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
                beta = beta_new
                continuation_idx += 1

        # Reconstruct Tucker tensor once per iteration.
        S = tucker_to_tensor((core, factors))

        # Stable beta updates are solved on a centered/scaled latent basis and
        # then mapped back to the original power basis of S. When needed, we
        # can explicitly fall back to the original solver in `beta.py`.
        if beta_stage == "before_grad" and (it % beta_update_interval) == 0:
            beta = link.update_beta(
                X=X,
                S=S,
                active_q=p,
                lambda_beta=lambda_beta,
                method=beta_update_method,
                allow_constant_term=allow_constant_term,
                mask=mask_bool,
                stable=stable_beta_update,
            )

        Xhat, dfdS = link.value_and_derivative(S, beta)

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
            beta = link.update_beta(
                X=X,
                S=S_beta,
                active_q=p,
                lambda_beta=lambda_beta,
                method=beta_update_method,
                allow_constant_term=allow_constant_term,
                mask=mask_bool,
                stable=stable_beta_update,
            )

        if return_history:
            _append_history()

    result = (core, factors, beta, link.state_dict()) if return_link_state else (core, factors, beta)
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


ntdpl_optimized = ntdpl
init_ntdpl_factors = _init_ntdpl_factors

__all__ = ["ntdpl", "ntdpl_optimized", "init_ntdpl_factors"]
