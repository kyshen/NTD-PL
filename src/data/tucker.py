from typing import Any, Dict
from pathlib import Path
from src.data.base import BaseData
from src.types import Float, Array
import numpy as np
from tensorly import tucker_to_tensor


class TuckerData(BaseData):
    def __init__(self, **data_cfg: Any):
        super().__init__(**data_cfg)

    def _make_dense(self) -> Array:
        np.random.seed(self.cfg["seed"])
        rank = self.cfg["rank"]
        shape = self.cfg["shape"]
        N = len(rank)
        factors = [np.random.normal(size=(shape[n], rank[n])).astype(Float) for n in range(N)]
        core = np.random.normal(size=rank).astype(Float)
        dense = tucker_to_tensor((core, factors))
        dense = dense.astype(Float)
        return dense
