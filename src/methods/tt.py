from typing import Any, Dict

import numpy as np
from tensorly import tt_to_tensor
from tensorly.decomposition import tensor_train
from src.utils.tensor_ranks import tt_rank_from_tucker
from src.methods.base import BaseDecomposeMethod
from src.types import Tensor


class _TTStateMixin:
    fitted: bool
    factors: list[np.ndarray]

    def reconstruct(self) -> Tensor:
        if not self.fitted:
            raise ValueError("Model must be fitted before reconstruction.")
        dense = np.array(tt_to_tensor(self.factors))
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


class TTDecomposition(_TTStateMixin, BaseDecomposeMethod):
    def __init__(self, **method_cfg: Any) -> None:
        super().__init__()

        self.cfg = method_cfg
        self.fitted = False

    def fit(self, data, mask, logcallback) -> None:
        X = data.dense
        if self.cfg["tt_rank"] is not None:
            rank = self.cfg["tt_rank"]
        else:            
            rank = tt_rank_from_tucker(X.shape, self.cfg["rank"])

        tt_tensor = tensor_train(
            X,
            rank=rank,
            svd=self.cfg["svd"],
        )
        self.factors = [np.array(factor, dtype=np.float32) for factor in tt_tensor.factors]
        self.fitted = True
        return None
