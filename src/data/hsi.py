from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np
from PIL import Image
from scipy.io import loadmat

from src.data.base import BaseData
from src.types import Array, Float
from src.utils.image_ops import downsample_image_to_shape


def _ensure_hsi_cube(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D hyperspectral cube, got shape {arr.shape}")
    return arr


def _load_hsi_from_file(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return _ensure_hsi_cube(np.load(path))
    if path.suffix.lower() == ".img":
        return _load_envi_cube(path)
    if path.suffix.lower() == ".mat":
        mat = loadmat(path)
        if {"V", "nRow", "nCol"}.issubset(mat):
            v = np.asarray(mat["V"], dtype=np.float32)
            n_row = int(np.asarray(mat["nRow"]).reshape(-1)[0])
            n_col = int(np.asarray(mat["nCol"]).reshape(-1)[0])
            n_band = int(np.asarray(mat.get("nBand", [[v.shape[0]]])).reshape(-1)[0])
            if v.ndim != 2:
                raise ValueError(f"Expected benchmark matrix V to be 2D, got shape {v.shape}")
            if v.shape != (n_band, n_row * n_col):
                raise ValueError(
                    "Benchmark hyperspectral matrix shape mismatch: "
                    f"V has shape {v.shape}, expected ({n_band}, {n_row * n_col})"
                )
            return _ensure_hsi_cube(v.T.reshape(n_row, n_col, n_band))
        preferred_keys = (
            "indian_pines_corrected",
            "indian_pines",
            "salinas_corrected",
            "salinas",
            "Botswana",
            "botswana",
            "KSC",
            "ksc",
            "paviaU",
            "pavia",
            "cube",
            "data",
        )
        for key in preferred_keys:
            if key in mat:
                return _ensure_hsi_cube(mat[key])
        for key, value in mat.items():
            if key.startswith("__"):
                continue
            arr = np.asarray(value)
            if arr.ndim == 3:
                return _ensure_hsi_cube(arr)
        raise ValueError(f"No 3D hyperspectral cube found in {path}")
    raise ValueError(f"Unsupported hyperspectral file format: {path.suffix}")


_ENVI_DTYPES: dict[int, np.dtype] = {
    1: np.dtype(np.uint8),
    2: np.dtype(np.int16),
    3: np.dtype(np.int32),
    4: np.dtype(np.float32),
    5: np.dtype(np.float64),
    12: np.dtype(np.uint16),
    13: np.dtype(np.uint32),
    14: np.dtype(np.int64),
    15: np.dtype(np.uint64),
}


def _parse_envi_header_value(text: str) -> str | list[str]:
    value = text.strip()
    if value.startswith("{") and value.endswith("}"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [item.strip() for item in inner.split(",")]
    return value


def _load_envi_header(path: Path) -> dict[str, str | list[str]]:
    header_path = path.with_suffix(path.suffix + ".hdr")
    if not header_path.exists():
        header_path = path.with_suffix(".hdr")
    if not header_path.exists():
        raise FileNotFoundError(f"Missing ENVI header for {path}")

    header = header_path.read_text(encoding="utf-8", errors="ignore")
    fields: dict[str, str | list[str]] = {}
    pattern = re.compile(r"^\s*([^=\n]+?)\s*=\s*(\{.*?\}|[^\n]+)", re.MULTILINE | re.DOTALL)
    for key, value in pattern.findall(header):
        fields[key.strip().lower()] = _parse_envi_header_value(value)
    return fields


def _load_envi_cube(path: Path) -> np.ndarray:
    header = _load_envi_header(path)
    try:
        lines = int(str(header["lines"]))
        samples = int(str(header["samples"]))
        bands = int(str(header["bands"]))
        data_type = int(str(header["data type"]))
        byte_order = int(str(header.get("byte order", "0")))
        interleave = str(header.get("interleave", "bsq")).strip().lower()
        offset = int(str(header.get("header offset", "0")))
    except KeyError as exc:
        raise ValueError(f"Missing required ENVI header field {exc} for {path}") from exc

    if data_type not in _ENVI_DTYPES:
        raise ValueError(f"Unsupported ENVI data type {data_type} for {path}")

    dtype = _ENVI_DTYPES[data_type].newbyteorder("<" if byte_order == 0 else ">")
    raw = np.fromfile(path, dtype=dtype, offset=offset)
    expected = lines * samples * bands
    if raw.size != expected:
        raise ValueError(f"ENVI cube size mismatch for {path}: got {raw.size}, expected {expected}")

    if interleave == "bsq":
        cube = raw.reshape(bands, lines, samples).transpose(1, 2, 0)
    elif interleave == "bil":
        cube = raw.reshape(lines, bands, samples).transpose(0, 2, 1)
    elif interleave == "bip":
        cube = raw.reshape(lines, samples, bands)
    else:
        raise ValueError(f"Unsupported ENVI interleave {interleave} for {path}")
    return _ensure_hsi_cube(cube)


def _center_crop_spatial(array: np.ndarray, crop_shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim < 2:
        raise ValueError(f"Expected at least 2 spatial dimensions, got shape {arr.shape}")
    target_h, target_w = (int(crop_shape[0]), int(crop_shape[1]))
    src_h, src_w = int(arr.shape[0]), int(arr.shape[1])
    if target_h > src_h or target_w > src_w:
        raise ValueError(
            f"Cannot center-crop shape {(target_h, target_w)} from source spatial shape {(src_h, src_w)}."
        )
    start_h = (src_h - target_h) // 2
    start_w = (src_w - target_w) // 2
    return arr[start_h : start_h + target_h, start_w : start_w + target_w, ...]


class _BaseHSIData(BaseData):
    scene_name: str

    def __init__(self, **data_cfg: Any):
        self.scene_name = ""
        super().__init__(**data_cfg)

    def _postprocess_cube(self, cube: np.ndarray) -> np.ndarray:
        arr = _ensure_hsi_cube(cube)
        crop_shape = self.cfg.get("crop_shape")
        if crop_shape is not None:
            arr = _center_crop_spatial(arr, tuple(int(v) for v in crop_shape))
        else:
            arr = downsample_image_to_shape(arr, self.cfg["target_shape"])
        return arr

    def _postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        arr = np.asarray(mask)
        if arr.ndim != 2:
            raise ValueError(f"Expected a 2D hyperspectral support mask, got shape {arr.shape}")
        crop_shape = self.cfg.get("crop_shape")
        if crop_shape is not None:
            sampled = _center_crop_spatial(arr.astype(np.uint8)[..., None], tuple(int(v) for v in crop_shape))[..., 0]
        else:
            sampled = downsample_image_to_shape(arr.astype(np.uint8), self.cfg["target_shape"])
        return np.asarray(sampled > 0, dtype=bool)


def _cave_scene_dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.is_dir())


class CAVEHSIData(_BaseHSIData):
    def _make_dense(self) -> Array:
        root = Path.cwd() / Path(self.cfg["path"])
        scene_dirs = _cave_scene_dirs(root)
        scene_id = int(self.cfg["id"])
        if not 1 <= scene_id <= len(scene_dirs):
            raise ValueError(f"Invalid id {scene_id} for CAVEHSIData. Must be in [1, {len(scene_dirs)}].")

        scene_dir = scene_dirs[scene_id - 1]
        inner_dirs = [path for path in scene_dir.iterdir() if path.is_dir()]
        if not inner_dirs:
            raise ValueError(f"No spectral band directory found under {scene_dir}")
        band_dir = inner_dirs[0]

        band_paths = sorted(
            path for path in band_dir.glob("*.png") if "_RGB" not in path.name and "Thumbs" not in path.name
        )
        if not band_paths:
            raise ValueError(f"No spectral PNG bands found under {band_dir}")

        bands = [np.asarray(Image.open(path)) for path in band_paths]
        cube = np.stack(bands, axis=-1)
        self.scene_name = scene_dir.name.replace("_ms", "")
        return self._postprocess_cube(cube)


class IndianPinesHSIData(_BaseHSIData):
    def _make_dense(self) -> Array:
        path = Path.cwd() / Path(self.cfg["path"])
        cube = _load_hsi_from_file(path)
        self.scene_name = path.stem
        return self._postprocess_cube(cube)


class SalinasHSIData(_BaseHSIData):
    def _make_dense(self) -> Array:
        path = Path.cwd() / Path(self.cfg["path"])
        cube = _load_hsi_from_file(path)
        self.scene_name = path.stem
        return self._postprocess_cube(cube)


class BotswanaHSIData(_BaseHSIData):
    def _make_dense(self) -> Array:
        path = Path.cwd() / Path(self.cfg["path"])
        cube = _load_hsi_from_file(path)
        self.scene_name = path.stem
        return self._postprocess_cube(cube)


class PaviaHSIData(_BaseHSIData):
    def _make_dense(self) -> Array:
        path = Path.cwd() / Path(self.cfg["path"])
        cube = _load_hsi_from_file(path)
        self.scene_name = path.stem
        return self._postprocess_cube(cube)


class JasperRidgeHSIData(_BaseHSIData):
    def _make_dense(self) -> Array:
        path = Path.cwd() / Path(self.cfg["path"])
        cube = _load_hsi_from_file(path)
        self.scene_name = path.stem
        return self._postprocess_cube(cube)


class SamsonHSIData(_BaseHSIData):
    def _make_dense(self) -> Array:
        path = Path.cwd() / Path(self.cfg["path"])
        cube = _load_hsi_from_file(path)
        self.scene_name = path.stem
        return self._postprocess_cube(cube)


class UrbanHSIData(_BaseHSIData):
    def _make_dense(self) -> Array:
        path = Path.cwd() / Path(self.cfg["path"])
        cube = _load_hsi_from_file(path)
        self.scene_name = path.stem
        return self._postprocess_cube(cube)


class CupriteHSIData(_BaseHSIData):
    def _make_dense(self) -> Array:
        path = Path.cwd() / Path(self.cfg["path"])
        cube = _load_hsi_from_file(path)
        self.scene_name = path.stem
        return self._postprocess_cube(cube)
