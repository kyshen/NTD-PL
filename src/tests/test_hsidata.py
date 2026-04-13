from pathlib import Path

import numpy as np
import pytest

from src.data.hsi import CAVEHSIData
class _FakeImage:
    def __init__(self, array: np.ndarray):
        self._array = array

    def __array__(self, dtype=None):
        if dtype is None:
            return self._array
        return self._array.astype(dtype)


def test_cave_hsi_loads_selected_scene(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("src.data.hsi._cave_scene_dirs", lambda _root: [Path("scene1"), Path("scene2")])
    monkeypatch.setattr("pathlib.Path.iterdir", lambda self: [self / "inner"] if self.name == "scene2" else [])
    monkeypatch.setattr("pathlib.Path.is_dir", lambda self: self.name in {"scene1", "scene2", "inner"})
    monkeypatch.setattr("pathlib.Path.glob", lambda self, pattern: [self / f"band_{idx:02d}.png" for idx in range(31)])
    monkeypatch.setattr("src.data.hsi.Image.open", lambda _path: _FakeImage(np.full((256, 256), 32, dtype=np.uint16)))

    dataset = CAVEHSIData(path="data/CAVE", id=2, target_shape=(64, 32))
    tensor = dataset.get("fit")

    assert tensor.shape == (64, 32, 31)
    assert tensor.dense.dtype == np.uint16
    assert np.isclose(float(tensor.dense.max()), 32.0)
