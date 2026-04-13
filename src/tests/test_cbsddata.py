from pathlib import Path

import numpy as np
import pytest

from src.data.cbsd import CBSDData, _list_cbsd_files
from src.types import Float


def test_list_cbsd_files_sorts_png_names(tmp_path: Path):
    (tmp_path / "0002.png").write_bytes(b"")
    (tmp_path / "0000.png").write_bytes(b"")
    (tmp_path / "0001.png").write_bytes(b"")

    files = _list_cbsd_files(tmp_path)

    assert files == ["0000.png", "0001.png", "0002.png"]


def test_cbsddata_loads_all_images_when_id_is_zero(monkeypatch: pytest.MonkeyPatch):
    calls: list[Path] = []

    monkeypatch.setattr("src.data.cbsd._list_cbsd_files", lambda path: ["0000.png", "0001.png", "0002.png"])

    def fake_imread(path: Path) -> np.ndarray:
        calls.append(Path(path))
        return np.ones((256, 256, 3), dtype=np.uint8)

    monkeypatch.setattr("src.data.cbsd.mpimg.imread", fake_imread)

    dataset = CBSDData(path="data/cbsd", id=0, target_shape=(128, 128))
    tensor = dataset.get("fit")

    assert tensor.shape == (3, 128, 128, 3)
    assert tensor.dense.dtype == Float
    assert [p.name for p in calls] == ["0000.png", "0001.png", "0002.png"]


@pytest.mark.parametrize(
    ("image_id", "expected_file"),
    [(1, "0000.png"), (3, "0002.png")],
)
def test_cbsddata_loads_selected_image(monkeypatch: pytest.MonkeyPatch, image_id: int, expected_file: str):
    calls: list[Path] = []

    monkeypatch.setattr("src.data.cbsd._list_cbsd_files", lambda path: ["0000.png", "0001.png", "0002.png"])

    def fake_imread(path: Path) -> np.ndarray:
        calls.append(Path(path))
        return np.zeros((256, 256, 3), dtype=np.float64)

    monkeypatch.setattr("src.data.cbsd.mpimg.imread", fake_imread)

    dataset = CBSDData(path="data/cbsd", id=image_id, target_shape=(64, 96))
    tensor = dataset.get("fit")

    assert len(calls) == 1
    assert calls[0].name == expected_file
    assert tensor.shape == (64, 96, 3)
    assert tensor.dense.dtype == Float


def test_cbsddata_rejects_invalid_id(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("src.data.cbsd._list_cbsd_files", lambda path: ["0000.png", "0001.png", "0002.png"])

    with pytest.raises(ValueError, match=r"Invalid id .* Must be 0 or \[1, 3\]\."):
        CBSDData(path="data/cbsd", id=4, target_shape=(128, 128))
