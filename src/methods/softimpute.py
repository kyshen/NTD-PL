from collections.abc import Sequence
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.methods.base import BaseDecomposeMethod
from src.types import Tensor
from src.utils.completion_ops import mask_to_bool


class _SoftImputeStateMixin:
    fitted: bool
    completed_dense: Optional[np.ndarray]
    num_params_: int
    mode_ranks_: List[int]

    def reconstruct(self) -> Tensor:
        if not self.fitted or self.completed_dense is None:
            raise ValueError("Model must be fitted before reconstruction.")
        dense = np.array(self.completed_dense, dtype=np.float32, copy=True)
        return Tensor(shape=dense.shape, dense=dense)

    def get_num_params(self) -> int:
        if not self.fitted:
            raise ValueError("Model must be fitted before reconstruction.")
        return int(self.num_params_)

    def get_state_dict(self) -> Dict[str, Any]:
        if not self.fitted or self.completed_dense is None:
            raise ValueError("Model must be fitted before exporting state.")
        return {
            "completed_dense": np.array(self.completed_dense, copy=True),
            "num_params": int(self.num_params_),
            "mode_ranks": list(self.mode_ranks_),
            "fitted": self.fitted,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.completed_dense = np.array(
            state_dict["completed_dense"], dtype=np.float32, copy=True
        )
        self.num_params_ = int(state_dict.get("num_params", np.prod(self.completed_dense.shape)))
        self.mode_ranks_ = [int(r) for r in state_dict.get("mode_ranks", [])]
        self.fitted = bool(state_dict.get("fitted", True))


class SoftImputeCompletion(_SoftImputeStateMixin, BaseDecomposeMethod):
    """
    Mode-wise matrix completion baseline for tensor completion.

    The tensor is unfolded along one or more modes, each unfolding is completed
    by a standard SoftImpute-style low-rank matrix completion procedure, and the
    completed tensors are folded back and averaged on missing entries.

    Expected input to `fit(data, mask, logcallback)`:
        data.dense : np.ndarray
            Observed tensor. Missing entries can be filled arbitrarily (e.g. 0).
        mask : np.ndarray
            Boolean / {0,1} mask with the same shape as `data.dense`.
            1/True = observed, 0/False = missing.

    Important config keys (all optional except rank / matrix_rank):
        rank / matrix_rank : int or sequence[int]
            Target matrix rank(s) used on unfoldings.
            - If `matrix_rank` is given, it takes precedence.
            - If only Tucker-style `rank` is given, per-mode matrix ranks are
              derived from it.
        n_iter_max : int
            Max inner SoftImpute iterations for each unfolded matrix.
        outer_n_iter_max : int
            Number of outer tensor aggregation iterations.
        tol : float
            Outer stopping tolerance.
        matrix_tol : float
            Inner stopping tolerance for each matrix completion.
        shrinkage_value : float or None
            Singular-value shrinkage. If None, a data-adaptive value is used.
        unfold_modes : sequence[int] or None
            Which modes to unfold. None means all modes.
        init_fill : str
            One of {"zero", "mean"}. Default is "mean".
    """

    def __init__(self, **method_cfg: Any) -> None:
        super().__init__()
        self.cfg = method_cfg
        self.fitted = False
        self.completed_dense: Optional[np.ndarray] = None
        self.num_params_: int = 0
        self.mode_ranks_: List[int] = []

    def fit(self, data, mask, logcallback) -> None:
        X_obs = np.array(data.dense, dtype=np.float32)
        mask_bool = np.ones_like(X_obs, dtype=bool) if mask is None else mask_to_bool(mask, X_obs.shape)
        if not np.any(mask_bool):
            raise ValueError("`mask` contains no observed entries.")

        modes = _resolve_unfold_modes(self.cfg, X_obs.ndim)
        mode_ranks = _resolve_matrix_ranks(self.cfg, X_obs.shape, modes)

        outer_n_iter_max = int(self.cfg.get("outer_n_iter_max", 5))
        inner_n_iter_max = int(self.cfg.get("n_iter_max", 100))
        tol = float(self.cfg.get("tol", 1e-5))
        matrix_tol = float(self.cfg.get("matrix_tol", tol))
        shrinkage_value = self.cfg.get("shrinkage_value", None)
        init_fill = str(self.cfg.get("init_fill", "mean"))

        T = _initialize_tensor(X_obs, mask_bool, init_fill=init_fill)
        last_mode_factors: List[Dict[str, np.ndarray]] = []

        for outer_it in range(outer_n_iter_max):
            completed_views: List[np.ndarray] = []
            last_mode_factors = []

            for mode, rank in zip(modes, mode_ranks):
                Xn_obs = _unfold(X_obs, mode)
                Mn = _unfold(mask_bool, mode)
                Xn_init = _unfold(T, mode)

                Xn_hat, stats = _softimpute_matrix(
                    X_obs=Xn_obs,
                    mask=Mn,
                    rank=rank,
                    n_iter_max=inner_n_iter_max,
                    tol=matrix_tol,
                    shrinkage_value=shrinkage_value,
                    init_matrix=Xn_init,
                )

                completed_views.append(_fold(Xn_hat, X_obs.shape, mode))
                last_mode_factors.append(stats)

            T_new = np.mean(np.stack(completed_views, axis=0), axis=0)
            T_new[mask_bool] = X_obs[mask_bool]

            rel_change = _relative_change(T_new, T)
            if getattr(logcallback, "log_level", 0) >= 1:
                logcallback.addlog({"error": float(rel_change)})

            T = T_new
            if rel_change < tol:
                break

        self.completed_dense = np.array(T, dtype=np.float32, copy=True)
        self.mode_ranks_ = [int(r) for r in mode_ranks]
        self.num_params_ = 0
        for item in last_mode_factors:
            self.num_params_ += int(item.get("num_params", 0))
        if self.num_params_ <= 0:
            self.num_params_ = int(np.prod(self.completed_dense.shape))
        self.fitted = True
        return None


def _softimpute_matrix(
    X_obs: np.ndarray,
    mask: np.ndarray,
    rank: Optional[int],
    n_iter_max: int,
    tol: float,
    shrinkage_value: Optional[float],
    init_matrix: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    X_obs = np.asarray(X_obs, dtype=np.float64)
    mask_bool = np.asarray(mask).astype(bool)

    if init_matrix is None:
        M = _initialize_matrix(X_obs, mask_bool, init_fill="mean")
    else:
        M = np.array(init_matrix, dtype=np.float64, copy=True)
        M[mask_bool] = X_obs[mask_bool]

    U_last = np.zeros((X_obs.shape[0], 0), dtype=np.float64)
    s_last = np.zeros((0,), dtype=np.float64)
    Vt_last = np.zeros((0, X_obs.shape[1]), dtype=np.float64)

    for _ in range(n_iter_max):
        Y = np.array(M, copy=True)
        Y[mask_bool] = X_obs[mask_bool]

        U, s, Vt = np.linalg.svd(Y, full_matrices=False)
        tau = _resolve_shrinkage(shrinkage_value, s)
        s_shrunk = np.maximum(s - tau, 0.0)

        keep = np.flatnonzero(s_shrunk > 0)
        if rank is not None:
            keep = keep[: int(rank)]

        if keep.size == 0:
            M_new = np.zeros_like(Y)
            U_last = np.zeros((Y.shape[0], 0), dtype=np.float64)
            s_last = np.zeros((0,), dtype=np.float64)
            Vt_last = np.zeros((0, Y.shape[1]), dtype=np.float64)
        else:
            U_last = U[:, keep]
            s_last = s_shrunk[keep]
            Vt_last = Vt[keep, :]
            M_new = (U_last * s_last) @ Vt_last

        M_new[mask_bool] = X_obs[mask_bool]
        rel_change = _relative_change(M_new, M)
        M = M_new
        if rel_change < tol:
            break

    num_params = int(U_last.size + s_last.size + Vt_last.size)
    stats = {
        "U": U_last.astype(np.float32, copy=False),
        "s": s_last.astype(np.float32, copy=False),
        "Vt": Vt_last.astype(np.float32, copy=False),
        "num_params": np.array(num_params, dtype=np.int64),
    }
    return M.astype(np.float32), stats


def _resolve_shrinkage(shrinkage_value: Optional[float], singular_values: np.ndarray) -> float:
    if singular_values.size == 0:
        return 0.0
    if shrinkage_value is None:
        return float(0.05 * singular_values[0])
    return float(shrinkage_value)


def _resolve_unfold_modes(cfg: Dict[str, Any], ndim: int) -> List[int]:
    modes = cfg.get("unfold_modes", None)
    if modes is None:
        return list(range(ndim))
    out = [int(m) for m in modes]
    for m in out:
        if m < 0 or m >= ndim:
            raise ValueError(f"Invalid unfold mode {m} for ndim={ndim}.")
    return out


def _resolve_matrix_ranks(
    cfg: Dict[str, Any], shape: Tuple[int, ...], modes: Sequence[int]
) -> List[Optional[int]]:
    if cfg.get("matrix_rank", None) is not None:
        mr = cfg["matrix_rank"]
        if isinstance(mr, Sequence) and not isinstance(mr, (str, bytes)):
            if len(mr) != len(modes):
                raise ValueError(
                    f"`matrix_rank` has length {len(mr)} but `unfold_modes` has length {len(modes)}."
                )
            return [int(r) if r is not None else None for r in mr]
        return [int(mr) for _ in modes]

    rank = cfg.get("rank", None)
    if rank is None:
        raise KeyError("SoftImputeCompletion requires `matrix_rank` or Tucker-style `rank`.")

    if isinstance(rank, Sequence) and not isinstance(rank, (str, bytes)):
        if len(rank) != len(shape):
            raise ValueError(f"`rank` length {len(rank)} does not match tensor order {len(shape)}.")
        return [int(rank[m]) for m in modes]

    return [int(rank) for _ in modes]


def _initialize_tensor(X_obs: np.ndarray, mask: np.ndarray, init_fill: str) -> np.ndarray:
    X = np.array(X_obs, dtype=np.float32, copy=True)
    if init_fill == "zero":
        X[~mask] = 0.0
        return X
    if init_fill == "mean":
        mean_val = float(np.mean(X_obs[mask]))
        X[~mask] = mean_val
        return X
    raise ValueError(f"Unknown init_fill: {init_fill}")


def _initialize_matrix(X_obs: np.ndarray, mask: np.ndarray, init_fill: str) -> np.ndarray:
    X = np.array(X_obs, dtype=np.float64, copy=True)
    if init_fill == "zero":
        X[~mask] = 0.0
        return X
    if init_fill == "mean":
        mean_val = float(np.mean(X_obs[mask]))
        X[~mask] = mean_val
        return X
    raise ValueError(f"Unknown init_fill: {init_fill}")
def _unfold(X: np.ndarray, mode: int) -> np.ndarray:
    X = np.asarray(X)
    return np.reshape(np.moveaxis(X, mode, 0), (X.shape[mode], -1))


def _fold(Xn: np.ndarray, shape: Tuple[int, ...], mode: int) -> np.ndarray:
    front_shape = [shape[mode]] + [shape[i] for i in range(len(shape)) if i != mode]
    X = np.reshape(Xn, front_shape)
    return np.moveaxis(X, 0, mode)


def _relative_change(X_new: np.ndarray, X_old: np.ndarray, eps: float = 1e-12) -> float:
    num = float(np.linalg.norm(X_new - X_old))
    den = float(np.linalg.norm(X_old))
    return num / max(den, eps)
