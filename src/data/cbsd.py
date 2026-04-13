from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import numpy as np

from src.data.base import BaseData
from src.types import Array, Float
from src.utils.image_ops import downsample_image_to_shape


def _list_cbsd_files(path: Path) -> list[str]:
    files = sorted(file.name for file in path.glob("*.png"))
    if not files:
        raise ValueError(f"No PNG files found in CBSD directory: {path}")
    return files


class CBSDData(BaseData):
    def __init__(self, **data_cfg: Any):
        super().__init__(**data_cfg)

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        target_shape = self.cfg["target_shape"]
        return downsample_image_to_shape(image, target_shape)

    def _make_dense(self) -> Array:
        path = Path.cwd() / Path(self.cfg["path"])
        files = _list_cbsd_files(path)
        image_id = self.cfg["id"]

        if image_id == 0:
            dense = []
            for filename in files:
                img = mpimg.imread(path / filename)
                arr = np.asarray(img)
                if arr.dtype != Float:
                    arr = arr.astype(Float)
                arr = self._preprocess_image(arr)
                dense.append(arr)
            return np.asarray(dense)

        if 1 <= image_id <= len(files):
            img = mpimg.imread(path / files[image_id - 1])
            dense = np.asarray(img)
            if dense.dtype != Float:
                dense = dense.astype(Float)
            return self._preprocess_image(dense)

        raise ValueError(f"Invalid id {image_id} for CBSDData. Must be 0 or [1, {len(files)}].")
