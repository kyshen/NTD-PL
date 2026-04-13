"""
NTDPL normalization helpers.
"""

import numpy as np


def normalize_tucker(core, factors, eps: float = 1e-12):
    """
    Normalize Tucker factors column-wise and absorb the scale into the core.
    """
    core_new = np.asarray(core, dtype=np.float32)
    factors_new = []

    for n, A in enumerate(factors):
        A = np.asarray(A, dtype=np.float32)
        s = np.linalg.norm(A, axis=0)
        s_safe = np.maximum(s, eps)

        A_norm = A / s_safe[None, :]
        core_new = mode_scale_core(core_new, s_safe, mode=n)
        factors_new.append(A_norm)

    return core_new, factors_new


def mode_scale_core(core, scale, mode: int) -> np.ndarray:
    """
    Scale the Tucker core along a given mode.
    """
    X = np.moveaxis(core, mode, 0)
    X = X * scale.reshape((-1,) + (1,) * (X.ndim - 1))
    return np.moveaxis(X, 0, mode)
