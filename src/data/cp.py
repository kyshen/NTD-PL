from typing import Any, Dict
from pathlib import Path
from src.data.base import BaseData
from src.types import Float, Array
import numpy as np
from tensorly import cp_to_tensor
from src.utils.tensor_ranks import cp_rank_from_tucker


class CPData(BaseData):
    def __init__(self, **data_cfg: Any):
        super().__init__(**data_cfg)

    def _make_dense(self) -> Array:
        shape = self.cfg["shape"]
        np.random.seed(self.cfg["seed"])
        N = len(shape)

        if self.cfg['cp_rank'] is not None:
            rank = self.cfg['cp_rank']
        else:
            rank = cp_rank_from_tucker(shape, self.cfg['rank'], include_weights=True)
        factors = [np.random.normal(size=(shape[n], rank)).astype(Float) for n in range(N)]
        weights = np.random.normal(size=rank).astype(Float)
        dense = cp_to_tensor((weights, factors))
        dense = np.array(dense, dtype=Float)
        return dense
