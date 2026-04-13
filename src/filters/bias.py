from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.data.base import BaseData

from .base import DataFilter
import numpy as np

class BiasFilter(DataFilter):

    def __init__(self, **filter_cfg: Any) -> None:
        super().__init__(**filter_cfg)

    def __call__(self, data: "BaseData") -> "BaseData":
        self.add_bias(data)
        self.normalize(data)
        self.add_noise(data)
        return data

    def add_bias(self, data: "BaseData") -> None:
        bias = self.cfg["bias"]
        if bias is None:
            return
        try:
            if np.all(np.asarray(bias) == 0):
                return
        except Exception:
            pass
        data._dense = data._dense + bias
        data._dense_eval = data._dense.copy()
