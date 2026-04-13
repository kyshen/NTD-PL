from typing import Any, Dict

import numpy as np
from tensorly import tr_to_tensor
from tensorly.decomposition import tensor_ring
from src.utils.tensor_ranks import tr_rank_from_tucker
from src.methods.base import BaseDecomposeMethod
from src.types import Tensor


class _TRStateMixin:
    fitted: bool
    factors: list[np.ndarray]

    def reconstruct(self) -> Tensor:
        if not self.fitted:
            raise ValueError("Model must be fitted before reconstruction.")
        dense = np.array(tr_to_tensor(self.factors))
        return Tensor(shape=dense.shape, dense=dense)

    def get_num_params(self) -> int:
        if not self.fitted:
            raise ValueError("Model must be fitted before reconstruction.")
        total = 0
        for factor in self.factors:
            total += int(np.prod(factor.shape))
        return total

    def get_state_dict(self) -> Dict[str, Any]:
        if not self.fitted:
            raise ValueError("Model must be fitted before exporting state.")
        return {
            "factors": [np.array(f, copy=True) for f in self.factors],
            "fitted": self.fitted,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.factors = [np.array(f, dtype=np.float32, copy=True) for f in state_dict["factors"]]
        self.fitted = bool(state_dict.get("fitted", True))


class TRDecomposition(_TRStateMixin, BaseDecomposeMethod):
    def __init__(self, **method_cfg: Any) -> None:
        super().__init__()
        self.cfg = method_cfg
        self.fitted = False

    def fit(self, data, mask, logcallback) -> None:
        X = data.dense
        if self.cfg["tr_rank"] is not None:
            rank = self.cfg["tr_rank"]
        else:            
            rank = tr_rank_from_tucker(X.shape, self.cfg["rank"])

        tr_tensor = tensor_ring(
            X,
            rank=rank,
            svd=self.cfg["svd"],
        )
        self.factors = [np.array(factor, dtype=np.float32) for factor in tr_tensor.factors]
        self.fitted = True
        return None
