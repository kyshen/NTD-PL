from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tensorly import cp_to_tensor
from tensorly.decomposition import parafac

from src.methods.base import BaseDecomposeMethod
from src.types import Tensor
from src.utils.completion_ops import mask_to_bool, mean_fill_missing
from src.utils.tensor_ranks import cp_rank_from_tucker


class _CPStateMixin:
    fitted: bool
    weights: np.ndarray
    factors: List[np.ndarray]

    def reconstruct(self) -> Tensor:
        if not self.fitted:
            raise ValueError("Model must be fitted before reconstruction.")
        dense = np.array(cp_to_tensor((self.weights, self.factors)), dtype=np.float32)
        return Tensor(shape=dense.shape, dense=dense)

    def get_num_params(self) -> int:
        if not self.fitted:
            raise ValueError("Model must be fitted before reconstruction.")
        total = int(np.prod(self.weights.shape))
        for factor in self.factors:
            total += int(np.prod(factor.shape))
        return total

    def get_state_dict(self) -> Dict[str, Any]:
        if not self.fitted:
            raise ValueError("Model must be fitted before exporting state.")
        return {
            "weights": np.array(self.weights, copy=True),
            "factors": [np.array(f, copy=True) for f in self.factors],
            "fitted": self.fitted,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.weights = np.array(state_dict["weights"], dtype=np.float32, copy=True)
        self.factors = [
            np.array(f, dtype=np.float32, copy=True) for f in state_dict["factors"]
        ]
        self.fitted = bool(state_dict.get("fitted", True))


class CPDecomposition(_CPStateMixin, BaseDecomposeMethod):
    def __init__(self, **method_cfg: Any) -> None:
        super().__init__()
        self.cfg = method_cfg
        self.fitted = False

    def fit(self, data, mask, logcallback) -> None:
        X_obs = np.array(data.dense, dtype=np.float32)
        mask_bool = mask_to_bool(mask, X_obs.shape) if mask is not None else None
        rank = _resolve_cp_rank(X_obs.shape, self.cfg)
        return_errors = bool(getattr(logcallback, "log_level", 0) >= 1)

        X_fit = mean_fill_missing(X_obs, mask_bool) if mask_bool is not None else X_obs

        ret = parafac(
            X_fit,
            rank=rank,
            n_iter_max=int(self.cfg["n_iter_max"]),
            init=str(self.cfg.get("init_method", "svd")),
            tol=float(self.cfg.get("tol", 1e-8)),
            random_state=int(self.cfg.get("random_state", 0)),
            mask=None if mask_bool is None else mask_bool.astype(np.float32),
            return_errors=return_errors,
            normalize_factors=bool(self.cfg.get("normalize_factors", False)),
        )

        if return_errors:
            (weights, factors), errors = ret
            for e in errors:
                logcallback.addlog({"error": float(e)})
        else:
            weights, factors = ret

        if weights is None:
            weights = np.ones(factors[0].shape[1], dtype=np.float32)

        self.weights = np.array(weights, dtype=np.float32, copy=True)
        self.factors = [np.array(f, dtype=np.float32, copy=True) for f in factors]
        self.fitted = True
        return None


def _resolve_cp_rank(shape: Tuple[int, ...], cfg: Dict[str, Any]) -> int:
    cp_rank = cfg.get("cp_rank", None)
    if cp_rank is not None:
        return int(cp_rank)

    if "rank" not in cfg:
        raise KeyError("CP requires either `cp_rank` or Tucker-style `rank` in method_cfg.")
    return int(cp_rank_from_tucker(shape, cfg["rank"]))
