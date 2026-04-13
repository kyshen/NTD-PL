from pathlib import Path

import numpy as np
import pytest

from src.data.kodak import KodakData, _KODAK_FILES
from src.types import Float


def test_kodakdata_loads_all_images_when_id_is_zero(monkeypatch: pytest.MonkeyPatch):
    calls: list[Path] = []

    def fake_imread(path: Path) -> np.ndarray:
        calls.append(Path(path))
        return np.ones((256, 256, 3), dtype=np.uint8)

    monkeypatch.setattr("src.data.kodak.mpimg.imread", fake_imread)

    dataset = KodakData(path="data/kodak", id=0)
    tensor = dataset.get("fit")

    assert tensor.shape == (24, 128, 128, 3)
    assert tensor.dense.dtype == Float
    assert [p.name for p in calls] == _KODAK_FILES


@pytest.mark.parametrize(
    ("image_id", "expected_file"),
    [(1, "kodim01.png"), (24, "kodim24.png")],
)
def test_kodakdata_loads_selected_image(monkeypatch: pytest.MonkeyPatch, image_id: int, expected_file: str):
    calls: list[Path] = []

    def fake_imread(path: Path) -> np.ndarray:
        calls.append(Path(path))
        return np.zeros((256, 256, 3), dtype=np.float64)

    monkeypatch.setattr("src.data.kodak.mpimg.imread", fake_imread)

    dataset = KodakData(path="data/kodak", id=image_id)
    tensor = dataset.get("fit")

    assert len(calls) == 1
    assert calls[0].name == expected_file
    assert tensor.shape == (128, 128, 3)
    assert tensor.dense.dtype == Float


def test_kodakdata_supports_custom_target_shape(monkeypatch: pytest.MonkeyPatch):
    def fake_imread(path: Path) -> np.ndarray:
        return np.zeros((200, 180, 3), dtype=np.float32)

    monkeypatch.setattr("src.data.kodak.mpimg.imread", fake_imread)

    dataset = KodakData(path="data/kodak", id=1, target_shape=(64, 96))
    tensor = dataset.get("fit")

    assert tensor.shape == (64, 96, 3)


def test_kodakdata_downsamples_to_requested_shape(monkeypatch: pytest.MonkeyPatch):
    def fake_imread(path: Path) -> np.ndarray:
        return np.zeros((256, 256, 3), dtype=np.float32)

    monkeypatch.setattr("src.data.kodak.mpimg.imread", fake_imread)

    dataset = KodakData(path="data/kodak", id=1, target_shape=(64, 32))
    tensor = dataset.get("fit")

    assert tensor.shape == (64, 32, 3)


def test_kodakdata_rejects_too_large_target_shape(monkeypatch: pytest.MonkeyPatch):
    def fake_imread(path: Path) -> np.ndarray:
        return np.zeros((100, 100, 3), dtype=np.float32)

    monkeypatch.setattr("src.data.kodak.mpimg.imread", fake_imread)

    with pytest.raises(ValueError, match=r"target_shape .* exceeds image size"):
        KodakData(path="data/kodak", id=1, target_shape=(128, 128))


def test_kodakdata_rejects_invalid_target_shape(monkeypatch: pytest.MonkeyPatch):
    def fake_imread(path: Path) -> np.ndarray:
        return np.zeros((256, 256, 3), dtype=np.float32)

    monkeypatch.setattr("src.data.kodak.mpimg.imread", fake_imread)

    with pytest.raises(ValueError, match=r"target_shape must be positive"):
        KodakData(path="data/kodak", id=1, target_shape=(0, 2))


@pytest.mark.parametrize("invalid_id", [-1, 25])
def test_kodakdata_rejects_invalid_id(invalid_id: int):
    with pytest.raises(ValueError, match=r"Invalid id .* Must be 0 or \[1, 24\]\."):
        KodakData(path="data/kodak", id=invalid_id)
