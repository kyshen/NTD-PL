from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.data.base import BaseData

import numpy as np


class DataFilter(ABC):
    def __init__(self, **filter_cfg: Any) -> None:
        self.cfg = filter_cfg
        np.random.seed(int(self.cfg["seed"]))

    @abstractmethod
    def __call__(self, data: "BaseData") -> "BaseData":
        raise NotImplementedError
    
    def normalize(self, data: "BaseData") -> None:
        if not bool(self.cfg["normalize_method"]):
            return

        method = self.cfg["normalize_method"]
        if method == "energy":
            energy = np.linalg.norm(data._dense)
            data._dense = data._dense / (energy + 1e-8) * np.sqrt(np.prod(data._dense.shape))
            data._dense_eval = data._dense.copy()
            return

        if method == "max":
            scale = float(np.max(np.asarray(data._dense, dtype=np.float32)))
            if scale > 1e-12:
                data._dense = data._dense / scale
                data._dense_eval = data._dense.copy()
            return

        raise ValueError(f"Unsupported normalize_method: {method}")

    def add_noise(self, data: "BaseData") -> None:
        snr_db = self.cfg.get("snr_db")
        if snr_db is None:
            return
        snr_db = float(snr_db)
        signal_power = np.mean(data._dense**2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), size=data._dense.shape)
        data._dense += noise
