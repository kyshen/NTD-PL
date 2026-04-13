from typing import Any, Dict
from pathlib import Path
from src.data.base import BaseData
from src.types import Float, Array
import numpy as np


class RandData(BaseData):
    def __init__(self, **data_cfg: Any):
        super().__init__(**data_cfg)

    def _make_dense(self) -> Array:
        np.random.seed(self.cfg["seed"])
        type_str = self.cfg["type"]
        if type_str == "normal":
            dense = np.random.normal(size=self.cfg["shape"]).astype(Float)
        elif type_str == "uniform":
            dense = np.random.uniform(size=self.cfg["shape"]).astype(Float)
        else:
            raise ValueError(f"Unsupported distribution type: {type_str}")
        return dense
