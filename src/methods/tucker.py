from typing import Any, Dict
import numpy as np
from tensorly import tucker_to_tensor
from tensorly.decomposition import tucker
from src.methods.base import BaseDecomposeMethod
from src.types import Tensor
from src.utils.completion_ops import mask_to_bool, mask_to_float, mean_fill_missing


class _TuckerStateMixin:
    fitted: bool
    core: np.ndarray
    factors: list[np.ndarray]

    def reconstruct(self) -> Tensor:
        if not self.fitted:
            raise ValueError("Model must be fitted before reconstruction.")
        dense = np.array(tucker_to_tensor((self.core, self.factors)))
        return Tensor(shape=dense.shape, dense=dense)

    def get_num_params(self) -> int:
        if not self.fitted:
            raise ValueError("Model must be fitted before reconstruction.")
        total = int(np.prod(self.core.shape))
        for factor in self.factors:
            total += int(np.prod(factor.shape))
        return total

    def get_state_dict(self) -> Dict[str, Any]:
        if not self.fitted:
            raise ValueError("Model must be fitted before exporting state.")
        return {
            "core": np.array(self.core, copy=True),
            "factors": [np.array(f, copy=True) for f in self.factors],
            "fitted": self.fitted,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.core = np.array(state_dict["core"], dtype=np.float32, copy=True)
        self.factors = [np.array(f, dtype=np.float32, copy=True) for f in state_dict["factors"]]
        self.fitted = bool(state_dict.get("fitted", True))


class TuckerDecomposition(_TuckerStateMixin, BaseDecomposeMethod):
    def __init__(self, **method_cfg: Any) -> None:
        super().__init__()

        self.cfg = method_cfg
        self.fitted = False

    def fit(self, data, mask, logcallback) -> None:
        X_obs = np.array(data.dense)
        mask_bool = mask_to_bool(mask, X_obs.shape) if mask is not None else None
        mask_float = None if mask_bool is None else mask_to_float(mask_bool, X_obs.shape, dtype=X_obs.dtype)
        X_fit = mean_fill_missing(X_obs, mask_bool) if mask_bool is not None else X_obs

        return_errors = True if logcallback.log_level >= 1 else False
        ret = tucker(
            X_fit,
            rank=self.cfg["rank"],
            n_iter_max=self.cfg["n_iter_max"],
            init=self.cfg["init"],
            tol=self.cfg["tol"],
            mask=mask_float,
            return_errors=return_errors,
            random_state=0,
        )
        if return_errors:
            tucker_tensor, errors = ret
        else:
            tucker_tensor = ret

        self.core = np.array(tucker_tensor[0])
        self.factors = [np.array(factor) for factor in tucker_tensor[1]]
        if logcallback.log_level >= 1:
            for e in errors:
                logcallback.addlog({"error": float(e)})
        self.fitted = True
        return None
