from typing import Any, Dict
from pathlib import Path
from src.data.base import BaseData
from src.types import Float, Array
from src.utils.image_ops import downsample_image_to_shape
import matplotlib.image as mpimg
import numpy as np

_KODAK_FILES = [f"kodim{i:02d}.png" for i in range(1, 25)]


class KodakData(BaseData):
    def __init__(self, **data_cfg: Any):
        super().__init__(**data_cfg)

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        target_shape = self.cfg["target_shape"]
        return downsample_image_to_shape(image, target_shape)

    def _make_dense(self) -> Array:
        path = Path.cwd() / Path(self.cfg['path'])
        id = self.cfg["id"]
        if id == 0:
            dense = []
            for filename in _KODAK_FILES:
                img = mpimg.imread(path / filename)
                arr = np.asarray(img)
                if arr.dtype != Float:
                    arr = arr.astype(Float)
                arr = self._preprocess_image(arr)
                dense.append(arr)
            dense = np.asarray(dense)
        elif 1 <= id <= 24:
            img = mpimg.imread(path / _KODAK_FILES[id-1])
            dense = np.asarray(img)
            if dense.dtype != Float:
                dense = dense.astype(Float)
            dense = self._preprocess_image(dense)
        else:
            raise ValueError(f"Invalid id {id} for KodakData. Must be 0 or [1, 24].")
        return dense
    
