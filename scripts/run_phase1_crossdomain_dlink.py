from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

for _thread_key in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "TBB_NUM_THREADS",
):
    os.environ.setdefault(_thread_key, "1")

import matplotlib.pyplot as plt
import imageio.v3 as iio
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.methods.ntdpl import NTDPLDecomposition
from src.methods.tucker import TuckerDecomposition
from src.metrics import val_RMSE, val_SAM
from src.types import LogCallback, Tensor
from src.utils.image_ops import downsample_image_to_shape


DEFAULT_OUT_ROOT = PROJECT_ROOT / "results" / "phase1_crossdomain_dlink"
DOMAINS = {
    "brdf": "Material reflectance",
    "lightfield": "Light-field",
    "harvard": "Spectral natural scenes",
    "kth": "Action video",
    "ucf101": "Action video",
}
DATASET_LABELS = {
    "brdf": "BRDF",
    "lightfield": "Stanford LF",
    "harvard": "Harvard HS",
    "kth": "KTH-Action",
    "ucf101": "UCF101",
}
RANKS = {
    "brdf": (12, 12, 16, 2),
    "lightfield": (4, 4, 16, 16, 2),
    "harvard": (32, 32, 8),
    "kth": (12, 18, 18, 3),
    "ucf101": (12, 18, 18, 3),
}
PROCESSED_SHAPES = {
    "brdf": (32, 32, 64, 3),
    "lightfield": (7, 7, 96, 96, 3),
    "harvard": (128, 128, 31),
    "kth": (24, 72, 96, 3),
    "ucf101": (24, 72, 96, 3),
}
PER_UNIT_COLUMNS = [
    "domain",
    "dataset",
    "unit_id",
    "unit_label",
    "source_path",
    "raw_shape",
    "processed_shape",
    "rank",
    "seed",
    "tucker_rmse",
    "ntdpl_rmse",
    "rmse_gain_pct",
    "tucker_sam",
    "ntdpl_sam",
    "sam_gain_pct",
    "d_link_db",
    "diagnostic_label",
    "tucker_params",
    "ntdpl_params",
    "runtime_tucker",
    "runtime_ntdpl",
    "status",
    "notes",
]


@dataclass(frozen=True)
class UnitSpec:
    dataset_key: str
    unit_id: str
    unit_label: str
    source_path: str
    raw_shape: str = ""
    notes: str = ""


def _worker_env() -> None:
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
        "TBB_NUM_THREADS",
    ):
        os.environ.setdefault(key, "1")


def _resolve_root(value: str, default_rel: str) -> Path:
    candidates: list[Path] = []
    if value:
        p = Path(value)
        candidates.append(p)
        if not p.is_absolute():
            candidates.append(PROJECT_ROOT / p)
            candidates.append(PROJECT_ROOT / "data" / p)
    candidates.append(PROJECT_ROOT / default_rel)
    candidates.append(PROJECT_ROOT / "data" / Path(default_rel).name)
    for path in candidates:
        if path.exists():
            return path.resolve()
    tried = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not find dataset root. Tried:\n  {tried}")


def _normalize_max(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr[arr < 0] = 0.0
    scale = float(np.max(arr))
    if np.isfinite(scale) and scale > 1e-12:
        arr = arr / scale
    return arr.astype(np.float32, copy=False)


def _sample_axis(length: int, target: int) -> np.ndarray:
    if target > length:
        raise ValueError(f"target {target} exceeds axis length {length}")
    return np.linspace(0, length - 1, num=target).round().astype(int)


def _load_merl_brdf(path: Path, target_shape: tuple[int, int, int, int]) -> tuple[np.ndarray, str, str]:
    with path.open("rb") as handle:
        dims = np.fromfile(handle, dtype=np.int32, count=3)
        if dims.size != 3:
            raise ValueError("MERL header is incomplete")
        n = int(np.prod(dims))
        values = np.fromfile(handle, dtype=np.float64, count=3 * n)
    if values.size != 3 * n:
        raise ValueError(f"MERL payload has {values.size} doubles, expected {3 * n}")
    raw = values.reshape(3, int(dims[0]), int(dims[1]), int(dims[2]))
    scales = np.asarray([1.0 / 1500.0, 1.15 / 1500.0, 1.66 / 1500.0], dtype=np.float64)
    raw = raw * scales[:, None, None, None]
    tensor = np.moveaxis(raw, 0, -1)
    th, td, ph, ch = target_shape
    tensor = tensor[_sample_axis(tensor.shape[0], th)]
    tensor = tensor[:, _sample_axis(tensor.shape[1], td)]
    tensor = tensor[:, :, _sample_axis(tensor.shape[2], ph)]
    tensor = tensor[..., :ch]
    neg_count = int(np.sum(~np.isfinite(tensor) | (tensor < 0)))
    notes = f"MERL binary; official RGB scales; clamped_invalid_or_negative={neg_count}"
    return _normalize_max(tensor), "x".join(str(int(v)) for v in dims) + "x3", notes


def _choose_mat_cube(path: Path) -> tuple[np.ndarray, np.ndarray | None, str]:
    mat = sio.loadmat(path)
    preferred = ["reflectances", "rad", "img", "hypercube", "cube", "ref"]
    arrays: dict[str, np.ndarray] = {
        key: value
        for key, value in mat.items()
        if not key.startswith("__") and isinstance(value, np.ndarray) and value.ndim == 3 and np.issubdtype(value.dtype, np.number)
    }
    if not arrays:
        raise ValueError("No 3D numeric array found in .mat file")
    name = next((key for key in preferred if key in arrays), sorted(arrays.keys())[0])
    mask = mat.get("lbl")
    if isinstance(mask, np.ndarray) and mask.ndim == 2:
        mask_out = mask
    else:
        mask_out = None
    return np.asarray(arrays[name]), mask_out, name


def _load_harvard(path: Path, target_shape: tuple[int, int, int]) -> tuple[np.ndarray, str, str]:
    if path.suffix.lower() == ".npy":
        cube = np.load(path)
        mask = None
        var_name = "npy"
    elif path.suffix.lower() == ".mat":
        cube, mask, var_name = _choose_mat_cube(path)
    else:
        raise ValueError(f"Unsupported Harvard cube suffix: {path.suffix}")
    if cube.ndim != 3:
        raise ValueError(f"Expected 3D hyperspectral cube, got {cube.shape}")
    raw_shape = "x".join(str(v) for v in cube.shape)
    cube = np.asarray(cube, dtype=np.float32)
    if mask is not None and mask.shape == cube.shape[:2]:
        cube = cube.copy()
        cube[mask == 0, :] = 0.0
    h, w, b = target_shape
    if cube.shape[2] < b:
        raise ValueError(f"Expected at least {b} bands, got {cube.shape[2]}")
    if cube.shape[0] < h or cube.shape[1] < w:
        raise ValueError(f"Target spatial shape {(h, w)} exceeds raw shape {cube.shape[:2]}")
    cube = downsample_image_to_shape(cube, (h, w))
    if cube.shape[2] != b:
        cube = cube[:, :, _sample_axis(cube.shape[2], b)]
    notes = f"mat_variable={var_name}; mask_zeroed={bool(mask is not None)}"
    return _normalize_max(cube), raw_shape, notes


def _image_files(root: Path) -> list[Path]:
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)


def _find_lightfield_units(root: Path) -> tuple[list[UnitSpec], list[str]]:
    image_files = _image_files(root)
    notes: list[str] = []
    if not image_files:
        suffix_counts: dict[str, int] = {}
        for path in root.rglob("*"):
            if path.is_file():
                suffix_counts[path.suffix.lower()] = suffix_counts.get(path.suffix.lower(), 0) + 1
        raw_count = suffix_counts.get(".raw", 0)
        txt_count = suffix_counts.get(".txt", 0)
        notes.append(
            "No usable light-field scene tensors found under "
            f"{root}; found {raw_count} RAW and {txt_count} metadata/calibration text files but no decoded sub-aperture images. "
            "Expected format: one scene directory containing a regular 7x7 or 9x9 grid of PNG/JPEG/TIFF sub-aperture views."
        )
        return [], notes
    by_dir: dict[Path, list[Path]] = {}
    for path in image_files:
        by_dir.setdefault(path.parent, []).append(path)
    units: list[UnitSpec] = []
    for folder, files in sorted(by_dir.items()):
        if len(files) >= 49:
            rel = folder.relative_to(root) if folder != root else Path(".")
            unit_id = re.sub(r"[^A-Za-z0-9]+", "_", str(rel)).strip("_") or "scene"
            units.append(UnitSpec("lightfield", f"lf_{unit_id}", str(rel), str(folder), notes=f"{len(files)} image files"))
    if not units:
        notes.append(
            f"Image files exist under {root}, but no directory has at least 49 images for a 7x7 sub-aperture tensor."
        )
    else:
        notes.append(f"Detected {len(units)} candidate sub-aperture image directories under {root}.")
    return units, notes


def _parse_kth_segments(sequence_file: Path) -> list[tuple[str, list[tuple[int, int]]]]:
    pattern = re.compile(r"^(person\d+_[a-z]+_d\d+)\s+frames\s+(.+)$")
    entries: list[tuple[str, list[tuple[int, int]]]] = []
    for raw_line in sequence_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        match = pattern.match(line)
        if not match:
            continue
        clip_key, spans_text = match.groups()
        spans: list[tuple[int, int]] = []
        for piece in spans_text.split(","):
            item = piece.strip()
            if not item:
                continue
            bounds = item.split("-")
            if len(bounds) != 2:
                continue
            start = int(bounds[0])
            end = int(bounds[1])
            if end >= start:
                spans.append((start, end))
        if spans:
            entries.append((clip_key, spans))
    return entries


def _find_kth_units(root: Path, limit: int) -> tuple[list[UnitSpec], list[str]]:
    sequence_file = root / "00sequences.txt"
    if not sequence_file.exists():
        raise FileNotFoundError(f"KTH sequence file not found: {sequence_file}")
    entries = _parse_kth_segments(sequence_file)
    video_map = {path.stem.replace("_uncomp", ""): path for path in sorted(root.rglob("*.avi"))}
    units: list[UnitSpec] = []
    missing = 0
    total_segments = 0
    for clip_key, spans in entries:
        total_segments += len(spans)
        video_path = video_map.get(clip_key)
        if video_path is None:
            missing += 1
            continue
        action = clip_key.split("_")[1]
        for seg_idx, (start, end) in enumerate(spans, start=1):
            unit_id = f"kth_{clip_key}_s{seg_idx}"
            label = f"{clip_key} seg{seg_idx}"
            notes = f"action={action}; segment={start}-{end}"
            units.append(UnitSpec("kth", unit_id, label, str(video_path), notes=notes))
    units = sorted(units, key=lambda spec: spec.unit_id)
    if limit > 0:
        units = units[:limit]
    notes = [
        f"KTH root={root}; avi_files={len(video_map)}; annotated_segments={total_segments}; units_selected={len(units)}; processed_shape=24x72x96x3; rank=(12,18,18,3)."
    ]
    if missing:
        notes.append(f"KTH annotation entries missing matching .avi file: {missing}.")
    return units, notes


def _find_ucf101_units(root: Path, split: str, limit: int) -> tuple[list[UnitSpec], list[str]]:
    csv_path = root / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"UCF101 split CSV not found: {csv_path}")
    frame = pd.read_csv(csv_path)
    needed = {"clip_name", "clip_path", "label"}
    if not needed.issubset(set(frame.columns)):
        raise ValueError(f"UCF101 CSV {csv_path} missing required columns {needed}")
    units: list[UnitSpec] = []
    for row in frame.itertuples(index=False):
        rel = str(row.clip_path).lstrip("/\\")
        video_path = root / rel
        unit_id = f"ucf101_{Path(str(row.clip_name)).stem}"
        label = f"{row.label}:{Path(str(row.clip_name)).stem}"
        notes = f"split={split}; action={row.label}"
        units.append(UnitSpec("ucf101", unit_id, label, str(video_path), notes=notes))
    units = sorted(units, key=lambda spec: spec.unit_id)
    total = len(units)
    if limit > 0:
        units = units[:limit]
    notes = [
        f"UCF101 root={root}; split={split}; csv_units={total}; units_selected={len(units)}; processed_shape=24x72x96x3; rank=(12,18,18,3)."
    ]
    return units, notes


def inspect_datasets(args: argparse.Namespace) -> tuple[list[UnitSpec], list[str]]:
    selected = {item.strip().lower() for item in args.datasets.split(",") if item.strip()}
    specs: list[UnitSpec] = []
    notes: list[str] = []
    if "brdf" in selected:
        root = _resolve_root(args.brdf_root, "data/BRDFDatabase")
        files = sorted(root.rglob("*.binary"))
        limit = int(args.limit_brdf) if int(args.limit_brdf) > 0 else len(files)
        for path in files[:limit]:
            specs.append(UnitSpec("brdf", f"brdf_{path.stem}", path.stem.replace("-", " "), str(path), "90x90x180x3"))
        notes.append(
            f"BRDF root={root}; binary_files={len(files)}; units_selected={min(limit, len(files))}; processed_shape=32x32x64x3; rank=(12,12,16,2)."
        )
    if "lightfield" in selected or "lf" in selected:
        root = _resolve_root(args.lightfield_root, "data/caldata-B5143104560")
        lf_specs, lf_notes = _find_lightfield_units(root)
        limit = int(args.limit_lightfield) if int(args.limit_lightfield) > 0 else len(lf_specs)
        specs.extend(lf_specs[:limit])
        notes.extend(lf_notes)
    if "harvard" in selected or "hs" in selected:
        root = _resolve_root(args.harvard_root, "data/CZ_hsdbi")
        files = sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".mat", ".npy"}])
        limit = int(args.limit_harvard) if int(args.limit_harvard) > 0 else len(files)
        for path in files[:limit]:
            raw_shape = ""
            try:
                cube, _, _ = _choose_mat_cube(path) if path.suffix.lower() == ".mat" else (np.load(path, mmap_mode="r"), None, "npy")
                raw_shape = "x".join(str(v) for v in cube.shape)
            except Exception:
                raw_shape = ""
            specs.append(UnitSpec("harvard", f"harvard_{path.stem}", path.stem, str(path), raw_shape=raw_shape))
        notes.append(
            f"Harvard root={root}; cube_files={len(files)}; units_selected={min(limit, len(files))}; processed_shape=128x128x31; rank=(32,32,8)."
        )
    if "kth" in selected or "kth-action" in selected:
        root = _resolve_root(args.kth_root, "data/KTH-Action")
        kth_limit = int(args.limit_kth) if int(args.limit_kth) > 0 else 0
        kth_specs, kth_notes = _find_kth_units(root, kth_limit)
        specs.extend(kth_specs)
        notes.extend(kth_notes)
    if "ucf101" in selected or "ucf" in selected:
        root = _resolve_root(args.ucf101_root, "data/UCF101")
        ucf_limit = int(args.limit_ucf101) if int(args.limit_ucf101) > 0 else 0
        ucf_specs, ucf_notes = _find_ucf101_units(root, str(args.ucf101_split), ucf_limit)
        specs.extend(ucf_specs)
        notes.extend(ucf_notes)
    if args.mode == "smoke":
        rng = np.random.default_rng(int(args.seed))
        picked: list[UnitSpec] = []
        for key in ("brdf", "lightfield", "harvard", "kth", "ucf101"):
            panel = [spec for spec in specs if spec.dataset_key == key]
            if not panel:
                continue
            count = min(int(args.smoke_units), len(panel))
            idx = sorted(rng.choice(len(panel), size=count, replace=False).tolist()) if count < len(panel) else list(range(len(panel)))
            picked.extend(panel[i] for i in idx)
        specs = picked
    return specs, notes


def _load_lightfield_from_images(folder: Path, target_shape: tuple[int, int, int, int, int]) -> tuple[np.ndarray, str, str]:
    from matplotlib import image as mpimg

    files = _image_files(folder)
    u, v, h, w, c = target_shape
    need = u * v
    if len(files) < need:
        raise ValueError(f"Need at least {need} sub-aperture images, found {len(files)}")
    files = files[:need]
    views: list[np.ndarray] = []
    raw_shape = ""
    for path in files:
        img = np.asarray(mpimg.imread(path), dtype=np.float32)
        raw_shape = raw_shape or "x".join(str(x) for x in img.shape)
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=-1)
        if img.shape[-1] > 3:
            img = img[..., :3]
        if img.max(initial=0.0) > 1.5:
            img = img / 255.0
        img = downsample_image_to_shape(img, (h, w))
        if img.shape[-1] < c:
            img = np.repeat(img, c, axis=-1)
        views.append(img[..., :c])
    tensor = np.asarray(views, dtype=np.float32).reshape(u, v, h, w, c)
    return _normalize_max(tensor), f"{len(files)}x{raw_shape}", f"first {need} sorted sub-aperture images"


def _load_kth_clip(spec: UnitSpec, target_shape: tuple[int, int, int, int]) -> tuple[np.ndarray, str, str]:
    video_path = Path(spec.source_path)
    match = re.search(r"segment=(\d+)-(\d+)", spec.notes)
    if not match:
        raise ValueError(f"KTH segment bounds missing in notes: {spec.notes}")
    start = int(match.group(1))
    end = int(match.group(2))
    frames = iio.imread(video_path, index=None)
    if frames.ndim != 4:
        raise ValueError(f"Expected video tensor with 4 dims, got {frames.shape}")
    raw_shape = "x".join(str(v) for v in frames.shape)
    total, _, _, _ = frames.shape
    start_idx = max(0, start - 1)
    end_idx = min(total, end)
    if end_idx <= start_idx:
        raise ValueError(f"Invalid KTH segment {start}-{end} for clip with {total} frames")
    clip = np.asarray(frames[start_idx:end_idx], dtype=np.float32)
    t, h, w, c = target_shape
    frame_idx = _sample_axis(clip.shape[0], min(t, clip.shape[0]))
    clip = clip[frame_idx]
    if clip.shape[0] < t:
        pad = np.repeat(clip[-1:], t - clip.shape[0], axis=0)
        clip = np.concatenate([clip, pad], axis=0)
    resized = np.empty((t, h, w, c), dtype=np.float32)
    for i in range(t):
        frame = clip[i]
        if frame.ndim == 2:
            frame = np.repeat(frame[..., None], 3, axis=-1)
        if frame.shape[-1] > c:
            frame = frame[..., :c]
        if frame.max(initial=0.0) > 1.5:
            frame = frame / 255.0
        resized[i] = downsample_image_to_shape(frame, (h, w))[..., :c]
    notes = f"{spec.notes}; raw_frames={total}; sampled_frames={t}"
    return _normalize_max(resized), raw_shape, notes


def _load_ucf101_clip(spec: UnitSpec, target_shape: tuple[int, int, int, int]) -> tuple[np.ndarray, str, str]:
    video_path = Path(spec.source_path)
    frames = iio.imread(video_path, index=None)
    if frames.ndim != 4:
        raise ValueError(f"Expected video tensor with 4 dims, got {frames.shape}")
    raw_shape = "x".join(str(v) for v in frames.shape)
    clip = np.asarray(frames, dtype=np.float32)
    total, _, _, _ = clip.shape
    t, h, w, c = target_shape
    frame_idx = _sample_axis(clip.shape[0], min(t, clip.shape[0]))
    clip = clip[frame_idx]
    if clip.shape[0] < t:
        pad = np.repeat(clip[-1:], t - clip.shape[0], axis=0)
        clip = np.concatenate([clip, pad], axis=0)
    resized = np.empty((t, h, w, c), dtype=np.float32)
    for i in range(t):
        frame = clip[i]
        if frame.ndim == 2:
            frame = np.repeat(frame[..., None], 3, axis=-1)
        if frame.shape[-1] > c:
            frame = frame[..., :c]
        if frame.max(initial=0.0) > 1.5:
            frame = frame / 255.0
        resized[i] = downsample_image_to_shape(frame, (h, w))[..., :c]
    notes = f"{spec.notes}; raw_frames={total}; sampled_frames={t}"
    return _normalize_max(resized), raw_shape, notes


def _load_unit(spec: UnitSpec) -> tuple[np.ndarray, str, str]:
    path = Path(spec.source_path)
    if spec.dataset_key == "brdf":
        return _load_merl_brdf(path, PROCESSED_SHAPES["brdf"])
    if spec.dataset_key == "lightfield":
        return _load_lightfield_from_images(path, PROCESSED_SHAPES["lightfield"])
    if spec.dataset_key == "harvard":
        return _load_harvard(path, PROCESSED_SHAPES["harvard"])
    if spec.dataset_key == "kth":
        return _load_kth_clip(spec, PROCESSED_SHAPES["kth"])
    if spec.dataset_key == "ucf101":
        return _load_ucf101_clip(spec, PROCESSED_SHAPES["ucf101"])
    raise ValueError(f"Unknown dataset key {spec.dataset_key}")


def _fit_tucker(array: np.ndarray, rank: tuple[int, ...], *, n_iter_max: int) -> tuple[TuckerDecomposition, np.ndarray, float]:
    tensor = Tensor(shape=array.shape, dense=array)
    method = TuckerDecomposition(rank=rank, n_iter_max=int(n_iter_max), init="svd", tol=1e-5)
    start = time.perf_counter()
    method.fit(tensor, None, LogCallback(log_level=0))
    elapsed = float(time.perf_counter() - start)
    return method, np.asarray(method.reconstruct().dense, dtype=np.float32), elapsed


def _fit_ntdpl(array: np.ndarray, rank: tuple[int, ...], args: argparse.Namespace) -> tuple[NTDPLDecomposition, np.ndarray, float]:
    tensor = Tensor(shape=array.shape, dense=array)
    method = NTDPLDecomposition(
        rank=rank,
        init_n_iter_max=int(args.init_n_iter_max),
        init="tucker",
        solver_variant="optimized",
        stable_beta_update=True,
        beta_update_stage="before_grad",
        random_state=int(args.seed),
        p_max=int(args.p_max),
        allow_constant_term=True,
        use_continuation=True,
        factor_normalize=True,
        lr_core=float(args.lr_core),
        lr_factors=float(args.lr_factors),
        lambda_core=1e-6,
        lambda_factors=1e-6,
        lambda_beta=float(args.lambda_beta),
        beta_update_method="ridge_lstsq",
        beta_update_interval=5,
        n_iter_max=int(args.n_iter_max),
    )
    start = time.perf_counter()
    method.fit(tensor, None, LogCallback(log_level=0))
    elapsed = float(time.perf_counter() - start)
    return method, np.asarray(method.reconstruct().dense, dtype=np.float32), elapsed


def _poly_design(x: np.ndarray, degree: int) -> np.ndarray:
    return np.vander(x, N=int(degree) + 1, increasing=True)


def _fit_scalar_poly_predict(
    x_pred: np.ndarray,
    target: np.ndarray,
    *,
    degree: int,
    lambda_reg: float,
    sample_size: int,
    seed: int,
) -> np.ndarray:
    x = np.asarray(x_pred, dtype=np.float64).reshape(-1)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(seed)
    if 0 < int(sample_size) < x.size:
        idx = rng.choice(x.size, size=int(sample_size), replace=False)
        x_fit = x[idx]
        y_fit = y[idx]
    else:
        x_fit = x
        y_fit = y
    phi = _poly_design(x_fit, degree)
    scales = np.maximum(np.linalg.norm(phi, axis=0), 1e-12)
    phi_scaled = phi / scales
    gram = phi_scaled.T @ phi_scaled
    rhs = phi_scaled.T @ y_fit
    coeff_scaled = np.linalg.solve(gram + float(lambda_reg) * np.eye(gram.shape[0]), rhs)
    coeff = coeff_scaled / scales
    pred = np.zeros_like(x, dtype=np.float64)
    for c in coeff[::-1]:
        pred = pred * x + float(c)
    return pred.reshape(x_pred.shape).astype(np.float32)


def _d_link_db(tucker_rmse: float, link_rmse: float) -> float:
    ratio = (float(link_rmse) ** 2) / max(float(tucker_rmse) ** 2, 1e-12)
    return float(10.0 * math.log10(1.0 / max(ratio, 1e-12)))


def _safe_sam(tensor: Tensor, recon: Tensor, dataset_key: str) -> float:
    if dataset_key == "brdf":
        return float("nan")
    if tensor.dense.shape[-1] <= 1:
        return float("nan")
    try:
        return float(val_SAM(tensor, recon))
    except Exception:
        return float("nan")


def _run_one(spec: UnitSpec, args: argparse.Namespace) -> dict[str, Any]:
    _worker_env()
    row: dict[str, Any] = {
        "domain": DOMAINS[spec.dataset_key],
        "dataset": DATASET_LABELS[spec.dataset_key],
        "unit_id": spec.unit_id,
        "unit_label": spec.unit_label,
        "source_path": spec.source_path,
        "raw_shape": spec.raw_shape,
        "processed_shape": "",
        "rank": "",
        "seed": int(args.seed),
        "tucker_rmse": float("nan"),
        "ntdpl_rmse": float("nan"),
        "rmse_gain_pct": float("nan"),
        "tucker_sam": float("nan"),
        "ntdpl_sam": float("nan"),
        "sam_gain_pct": float("nan"),
        "d_link_db": float("nan"),
        "diagnostic_label": "",
        "tucker_params": float("nan"),
        "ntdpl_params": float("nan"),
        "runtime_tucker": float("nan"),
        "runtime_ntdpl": float("nan"),
        "status": "failed",
        "notes": spec.notes,
    }
    try:
        array, raw_shape, load_notes = _load_unit(spec)
        rank = tuple(min(int(r), int(dim)) for r, dim in zip(RANKS[spec.dataset_key], array.shape))
        row["raw_shape"] = raw_shape or row["raw_shape"]
        row["processed_shape"] = "x".join(str(v) for v in array.shape)
        row["rank"] = "(" + ",".join(str(v) for v in rank) + ")"
        tensor = Tensor(shape=array.shape, dense=array)
        tucker_method, tucker_recon, runtime_tucker = _fit_tucker(array, rank, n_iter_max=int(args.tucker_n_iter_max))
        ntdpl_method, ntdpl_recon, runtime_ntdpl = _fit_ntdpl(array, rank, args)
        link_recon = _fit_scalar_poly_predict(
            tucker_recon,
            array,
            degree=int(args.p_max),
            lambda_reg=float(args.lambda_beta),
            sample_size=int(args.link_sample_size),
            seed=int(args.seed),
        )
        tucker_tensor = Tensor(shape=array.shape, dense=tucker_recon)
        ntdpl_tensor = Tensor(shape=array.shape, dense=ntdpl_recon)
        link_tensor = Tensor(shape=array.shape, dense=link_recon)
        tucker_rmse = val_RMSE(tensor, tucker_tensor)
        ntdpl_rmse = val_RMSE(tensor, ntdpl_tensor)
        link_rmse = val_RMSE(tensor, link_tensor)
        tucker_sam = _safe_sam(tensor, tucker_tensor, spec.dataset_key)
        ntdpl_sam = _safe_sam(tensor, ntdpl_tensor, spec.dataset_key)
        sam_gain = float("nan")
        if np.isfinite(tucker_sam) and abs(tucker_sam) > 1e-12:
            sam_gain = 100.0 * (tucker_sam - ntdpl_sam) / max(tucker_sam, 1e-12)
        row.update(
            {
                "tucker_rmse": tucker_rmse,
                "ntdpl_rmse": ntdpl_rmse,
                "rmse_gain_pct": 100.0 * (tucker_rmse - ntdpl_rmse) / max(tucker_rmse, 1e-12),
                "tucker_sam": tucker_sam,
                "ntdpl_sam": ntdpl_sam,
                "sam_gain_pct": sam_gain,
                "d_link_db": _d_link_db(tucker_rmse, link_rmse),
                "tucker_params": int(tucker_method.get_num_params()),
                "ntdpl_params": int(ntdpl_method.get_num_params()),
                "runtime_tucker": runtime_tucker,
                "runtime_ntdpl": runtime_ntdpl,
                "status": "ok",
                "notes": f"{load_notes}; link_refresh_rmse={link_rmse:.8g}; p_max={int(args.p_max)}",
            }
        )
    except Exception as exc:
        row["status"] = "failed"
        row["notes"] = f"{row.get('notes', '')}; {type(exc).__name__}: {exc}".strip("; ")
    return row


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=PER_UNIT_COLUMNS)


def _assign_labels(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if out.empty:
        return _empty_frame()
    out["diagnostic_label"] = ""
    ok = out["status"].eq("ok") & np.isfinite(pd.to_numeric(out["d_link_db"], errors="coerce"))
    for _, idx in out.loc[ok].groupby("dataset").groups.items():
        values = out.loc[idx, "d_link_db"].astype(float)
        low = float(values.quantile(1.0 / 3.0))
        high = float(values.quantile(2.0 / 3.0))
        for row_idx, value in values.items():
            label = "moderate"
            if value <= low or value <= 0.01:
                label = "boundary"
            if value >= high and value > 0.03:
                label = "effective"
            out.at[row_idx, "diagnostic_label"] = label
    out.loc[~ok, "diagnostic_label"] = "failed"
    return out[PER_UNIT_COLUMNS]


def _tercile_mean_gain(panel: pd.DataFrame, which: str) -> float:
    ok = panel.loc[panel["status"].eq("ok")].sort_values("d_link_db")
    if ok.empty:
        return float("nan")
    groups = np.array_split(ok, 3)
    mapping = {"low": groups[0], "mid": groups[1], "high": groups[2]}
    group = mapping[which]
    return float(group["rmse_gain_pct"].mean()) if not group.empty else float("nan")


def build_summary(frame: pd.DataFrame, specs: list[UnitSpec], selected_keys: set[str] | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dataset_keys: list[str] = []
    for key in ("brdf", "lightfield", "harvard", "kth", "ucf101"):
        if (selected_keys and key in selected_keys) or any(spec.dataset_key == key for spec in specs):
            dataset_keys.append(key)
    for key in dataset_keys:
        dataset = DATASET_LABELS[key]
        panel = frame.loc[frame["dataset"].eq(dataset)].copy()
        ok = panel.loc[panel["status"].eq("ok")].copy()
        if ok.shape[0] >= 2 and ok["d_link_db"].nunique() > 1 and ok["rmse_gain_pct"].nunique() > 1:
            corr = spearmanr(ok["d_link_db"], ok["rmse_gain_pct"], nan_policy="omit")
            spearman = float(corr.statistic)
        else:
            spearman = float("nan")
        rows.append(
            {
                "domain": DOMAINS[key],
                "dataset": dataset,
                "num_units": int(panel.shape[0]),
                "num_ok": int(ok.shape[0]),
                "num_failed": int(panel.shape[0] - ok.shape[0]),
                "median_d_link": float(ok["d_link_db"].median()) if not ok.empty else float("nan"),
                "mean_d_link": float(ok["d_link_db"].mean()) if not ok.empty else float("nan"),
                "spearman_dlink_gain": spearman,
                "mean_gain": float(ok["rmse_gain_pct"].mean()) if not ok.empty else float("nan"),
                "median_gain": float(ok["rmse_gain_pct"].median()) if not ok.empty else float("nan"),
                "low_tercile_gain": _tercile_mean_gain(panel, "low"),
                "mid_tercile_gain": _tercile_mean_gain(panel, "mid"),
                "high_tercile_gain": _tercile_mean_gain(panel, "high"),
                "num_effective": int(panel["diagnostic_label"].eq("effective").sum()) if not panel.empty else 0,
                "num_moderate": int(panel["diagnostic_label"].eq("moderate").sum()) if not panel.empty else 0,
                "num_boundary": int(panel["diagnostic_label"].eq("boundary").sum()) if not panel.empty else 0,
                "processed_shape": "x".join(str(v) for v in PROCESSED_SHAPES[key]),
                "rank": "(" + ",".join(str(v) for v in RANKS[key]) + ")",
            }
        )
    return pd.DataFrame(rows)


def build_representatives(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for _, panel in frame.loc[frame["status"].eq("ok")].groupby("dataset", sort=False):
        ordered = panel.sort_values("d_link_db")
        pieces = [
            ordered.head(2).assign(selection="bottom-2 by D_link"),
            ordered.tail(2).assign(selection="top-2 by D_link"),
        ]
        median = float(ordered["d_link_db"].median())
        med = ordered.iloc[(ordered["d_link_db"] - median).abs().argsort()[:1]].assign(selection="median-near by D_link")
        pieces.append(med)
        rows.append(pd.concat(pieces, ignore_index=True))
    if not rows:
        return pd.DataFrame()
    cols = [
        "domain",
        "dataset",
        "unit_id",
        "unit_label",
        "selection",
        "diagnostic_label",
        "d_link_db",
        "tucker_rmse",
        "ntdpl_rmse",
        "rmse_gain_pct",
        "rank",
        "processed_shape",
        "source_path",
    ]
    return pd.concat(rows, ignore_index=True).drop_duplicates(["dataset", "unit_id", "selection"])[cols]


def write_scatter(frame: pd.DataFrame, outdir: Path) -> None:
    ok = frame.loc[frame["status"].eq("ok")].copy()
    fig, ax = plt.subplots(figsize=(5.3, 3.6), constrained_layout=True)
    markers = {"BRDF": "o", "Stanford LF": "s", "Harvard HS": "^", "KTH-Action": "D", "UCF101": "P"}
    colors = {"BRDF": "#4C78A8", "Stanford LF": "#F58518", "Harvard HS": "#54A24B", "KTH-Action": "#E45756", "UCF101": "#B279A2"}
    for dataset, panel in ok.groupby("dataset", sort=False):
        ax.scatter(
            panel["d_link_db"],
            panel["rmse_gain_pct"],
            s=30,
            marker=markers.get(dataset, "o"),
            color=colors.get(dataset, "#666666"),
            alpha=0.78,
            linewidths=0.4,
            edgecolors="white",
            label=dataset,
        )
        if panel.shape[0] >= 2 and panel["d_link_db"].nunique() > 1 and panel["rmse_gain_pct"].nunique() > 1:
            rho = spearmanr(panel["d_link_db"], panel["rmse_gain_pct"], nan_policy="omit").statistic
            ax.text(
                0.02,
                0.96 - 0.07 * list(ok["dataset"].drop_duplicates()).index(dataset),
                f"{dataset}: rho={rho:.2f}",
                transform=ax.transAxes,
                fontsize=8,
                color=colors.get(dataset, "#666666"),
                va="top",
            )
    ax.axhline(0.0, color="#666666", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel(r"$D_{\mathrm{link}}$ (dB)")
    ax.set_ylabel("NTD-PL RMSE gain (%)")
    if not ok.empty:
        ax.legend(frameon=False, fontsize=8)
    ax.grid(True, color="#dddddd", linewidth=0.5, alpha=0.7)
    fig.savefig(outdir / "diagnostic_scatter.pdf")
    fig.savefig(outdir / "diagnostic_scatter.png", dpi=220)
    plt.close(fig)


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        val = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(val):
        return "--"
    return f"{val:.{digits}f}"


def _pct(value: Any, digits: int = 2) -> str:
    text = _fmt(value, digits)
    return "--" if text == "--" else f"{text}\\%"


def _gain_range(low: Any, high: Any, digits: int = 2) -> str:
    low_text = _fmt(low, digits)
    high_text = _fmt(high, digits)
    if low_text == "--" or high_text == "--":
        return "--"
    return f"{low_text}--{high_text}\\%"


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    text = frame.copy()
    for col in text.columns:
        if pd.api.types.is_float_dtype(text[col]):
            text[col] = text[col].map(lambda value: _fmt(value, 4))
        else:
            text[col] = text[col].map(lambda value: "" if pd.isna(value) else str(value))
    headers = [str(col) for col in text.columns]
    rows = text.astype(str).values.tolist()
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]

    def _row(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(values)) + " |"

    return "\n".join([_row(headers), "| " + " | ".join("-" * width for width in widths) + " |", *(_row(row) for row in rows)])


def _reading(row: Any) -> str:
    med = float(row.median_d_link) if np.isfinite(float(row.median_d_link)) else float("nan")
    if not np.isfinite(med):
        return "no usable units"
    if med <= 0.03:
        return "mostly boundary/modest"
    if med <= 0.15:
        return "moderate shared-link yield"
    return "clearer effective units"


def write_latex(summary: pd.DataFrame, reps: pd.DataFrame, outdir: Path) -> None:
    lines = [
        "% Candidate tables generated by scripts/run_phase1_crossdomain_dlink.py.",
        "% Labels and representatives are selected by D_link, not by NTD-PL gain.",
        "",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Dataset-level phase-1 cross-domain link diagnostics (candidate rows).}",
        r"\begin{tabular}{@{}l l l r r r r r l@{}}",
        r"\toprule",
        r"Domain & Dataset & Unit & \#Units & median $D_{\mathrm{link}}$ & Spearman & Median gain & Low--High gain & Reading \\",
        r"\midrule",
    ]
    unit_text = {"BRDF": "material BRDF", "Stanford LF": "scene LF", "Harvard HS": "spectral scene", "KTH-Action": "action clip", "UCF101": "video clip"}
    for row in summary.itertuples(index=False):
        lines.append(
            f"{row.domain} & {row.dataset} & {unit_text.get(row.dataset, 'unit')} & {int(row.num_ok)} & "
            f"{_fmt(row.median_d_link, 3)} & {_fmt(row.spearman_dlink_gain, 2)} & {_pct(row.median_gain, 2)} & "
            f"{_gain_range(row.low_tercile_gain, row.high_tercile_gain, 2)} & {_reading(row)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    lines.extend(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Representative phase-1 cross-domain units selected by $D_{\mathrm{link}}$ (candidate).}",
            r"\begin{tabular}{@{}l l l l r r r r@{}}",
            r"\toprule",
            r"Domain & Dataset & Unit ID & Diagnostic label & $D_{\mathrm{link}}$ & Tucker RMSE & NTD-PL RMSE & Gain \\",
            r"\midrule",
        ]
    )
    for row in reps.itertuples(index=False):
        lines.append(
            f"{row.domain} & {row.dataset} & {row.unit_id} & {row.diagnostic_label} & "
            f"{_fmt(row.d_link_db, 3)} & {_fmt(row.tucker_rmse, 4)} & {_fmt(row.ntdpl_rmse, 4)} & "
            f"{_fmt(row.rmse_gain_pct, 2)}\\% \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (outdir / "latex_table_candidates.tex").write_text("\n".join(lines), encoding="utf-8")


def write_data_inspection(outdir: Path, args: argparse.Namespace, specs: list[UnitSpec], notes: list[str]) -> None:
    ext_rows: list[dict[str, Any]] = []
    roots = {
        "BRDFDatabase": _resolve_root(args.brdf_root, "data/BRDFDatabase") if "brdf" in args.datasets.lower() else None,
        "caldata-B5143104560": _resolve_root(args.lightfield_root, "data/caldata-B5143104560") if "lightfield" in args.datasets.lower() else None,
        "CZ_hsdbi": _resolve_root(args.harvard_root, "data/CZ_hsdbi") if "harvard" in args.datasets.lower() else None,
        "KTH-Action": _resolve_root(args.kth_root, "data/KTH-Action") if "kth" in args.datasets.lower() else None,
        "UCF101": _resolve_root(args.ucf101_root, "data/UCF101") if "ucf101" in args.datasets.lower() or "ucf" in args.datasets.lower() else None,
    }
    for label, root in roots.items():
        if root is None:
            continue
        counts: dict[str, int] = {}
        for path in root.rglob("*"):
            if path.is_file():
                counts[path.suffix or "<none>"] = counts.get(path.suffix or "<none>", 0) + 1
        for ext, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            ext_rows.append({"root": label, "extension": ext, "count": count})
    spec_rows = [
        {
            "domain": DOMAINS[spec.dataset_key],
            "dataset": DATASET_LABELS[spec.dataset_key],
            "unit_id": spec.unit_id,
            "source_path": spec.source_path,
            "raw_shape": spec.raw_shape,
            "processed_shape": "x".join(str(v) for v in PROCESSED_SHAPES[spec.dataset_key]),
            "rank": "(" + ",".join(str(v) for v in RANKS[spec.dataset_key]) + ")",
        }
        for spec in specs
    ]
    lines = [
        "# Phase-1 Cross-Domain D_link Data Inspection",
        "",
        f"- Output directory: `{outdir}`",
        f"- Datasets requested: `{args.datasets}`",
        "",
        "## File Types",
        "",
        _markdown_table(pd.DataFrame(ext_rows)) if ext_rows else "No files found.",
        "",
        "## Loader And Unit Notes",
        "",
        *[f"- {note}" for note in notes],
        "- BRDF loader: minimal MERL binary reader using dimensions header and official RGB scales.",
        "- Harvard loader: MATLAB reader chooses a 3D numeric cube, preferring ref/reflectances/rad/img/hypercube/cube.",
        "- Light-field loader: requires decoded sub-aperture image grids; raw calibration-only Lytro unit data are not treated as scene tensors.",
        "- KTH loader: reads AVI clips with imageio-ffmpeg, then crops to annotated action segments from 00sequences.txt.",
        "- UCF101 loader: reads AVI clips with imageio-ffmpeg and treats each CSV-listed video clip as one unit.",
        "",
        "## Candidate Units",
        "",
        _markdown_table(pd.DataFrame(spec_rows)) if spec_rows else "No usable tensor units detected.",
        "",
    ]
    (outdir / "data_inspection_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_report(
    *,
    args: argparse.Namespace,
    outdir: Path,
    notes: list[str],
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    reps: pd.DataFrame,
    elapsed: float,
) -> None:
    failures = frame.loc[~frame["status"].eq("ok"), ["dataset", "unit_id", "notes"]]
    lines = [
        "# Phase-1 Cross-Domain D_link Diagnostic",
        "",
        f"- Mode: `{args.mode}`",
        f"- Output directory: `{outdir}`",
        f"- Elapsed wall time: {elapsed:.1f} s",
        f"- Seed: {int(args.seed)}",
        f"- Tucker iterations: {int(args.tucker_n_iter_max)}",
        f"- NTD-PL iterations: {int(args.n_iter_max)}",
        f"- NTD-PL polynomial degree: {int(args.p_max)}",
        f"- Link refresh sample size: {int(args.link_sample_size)}",
        "",
        "## Data Paths And Loader Notes",
        "",
        *[f"- {note}" for note in notes],
        "",
        "## Unit Definitions And Preprocessing",
        "",
        "- BRDF: one MERL material file per unit; processed to 32x32x64x3, rank (12,12,16,2); official RGB scaling; invalid/negative values clamped; per-material max-normalization.",
        "- Stanford LF: one decoded sub-aperture scene directory per unit when available; processed target 7x7x96x96x3, rank (4,4,16,16,2); no scene tensors are fabricated from calibration-only files.",
        "- Harvard HS: one hyperspectral image per unit; ref/lbl .mat cubes processed to 128x128x31, rank (32,32,8); mask pixels zeroed when lbl exists; per-scene max-normalization.",
        "- KTH-Action: one annotated action segment per unit; each clip is sampled to 24x72x96x3, rank (12,18,18,3); segments come from 00sequences.txt and are max-normalized independently.",
        "- UCF101: one CSV-listed video clip per unit; each clip is sampled to 24x72x96x3, rank (12,18,18,3); clips are max-normalized independently.",
        "- SAM is reported as NaN for BRDF because angular spectral error is not the target reflectance diagnostic here.",
        "",
        "## Label Rule",
        "",
        "Diagnostic labels are assigned only from D_link, not from NTD-PL gain: boundary is the bottom D_link tercile or D_link <= 0.01 dB; effective is the top tercile with D_link > 0.03 dB; all remaining successful units are moderate.",
        "",
        "## Dataset Summary",
        "",
        _markdown_table(summary),
        "",
        "## Representative Units",
        "",
        _markdown_table(reps) if not reps.empty else "No successful representative units.",
        "",
        "## Failures",
        "",
        _markdown_table(failures) if not failures.empty else "No failed fitted units. Some datasets may have no usable tensor units; see inspection notes.",
        "",
        "## Diagnostic Reading",
        "",
        "- BRDF/material reflectance: inspect median D_link, high-tercile gain, and representative materials before deciding whether it is a stronger link-yield domain.",
        "- Light-field: usable only if decoded sub-aperture scene tensors are present; calibration-only data are reported as unavailable.",
        "- Harvard HS: compare median D_link and high-tercile gain with RGB natural-image baselines before adding to Table 3.",
        "- KTH-Action: treat the result as action-clip diagnosis rather than video classification; coherent temporal clips are the diagnostic unit.",
        "- UCF101: treat the result as per-clip action-video diagnosis rather than classification; each clip is a self-contained tensor unit.",
        "- Table 3 recommendation: candidate evidence only; this script does not modify main.tex.",
        "",
        "## Reproduction Command",
        "",
        "```powershell",
        "python scripts/run_phase1_crossdomain_dlink.py "
        f"--mode {args.mode} --datasets {args.datasets} "
        f"--brdf_root {args.brdf_root} --lightfield_root {args.lightfield_root} --harvard_root {args.harvard_root} --kth_root {args.kth_root} --ucf101_root {args.ucf101_root} --ucf101_split {args.ucf101_split} "
        f"--seed {int(args.seed)} --n_iter_max {int(args.n_iter_max)} --tucker_n_iter_max {int(args.tucker_n_iter_max)} "
        f"--p_max {int(args.p_max)} --jobs {int(args.jobs)} --outdir {outdir}",
        "```",
        "",
    ]
    (outdir / ("smoke_test_report.md" if args.mode == "smoke" else "report.md")).write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) if args.outdir else DEFAULT_OUT_ROOT / timestamp
    outdir.mkdir(parents=True, exist_ok=True)
    selected_keys = {item.strip().lower() for item in args.datasets.split(",") if item.strip()}
    if "lf" in selected_keys:
        selected_keys.add("lightfield")
    if "hs" in selected_keys:
        selected_keys.add("harvard")
    specs, notes = inspect_datasets(args)
    write_data_inspection(outdir, args, specs, notes)
    if args.mode == "inspect":
        print(f"Wrote data inspection to {outdir}", flush=True)
        return outdir
    if not specs:
        frame = _empty_frame()
        summary = build_summary(frame, specs, selected_keys)
        reps = pd.DataFrame()
        frame.to_csv(outdir / "per_unit_results.csv", index=False)
        summary.to_csv(outdir / "summary_by_dataset.csv", index=False)
        reps.to_csv(outdir / "representative_units.csv", index=False)
        write_scatter(frame, outdir)
        write_latex(summary, reps, outdir)
        write_report(args=args, outdir=outdir, notes=notes, frame=frame, summary=summary, reps=reps, elapsed=0.0)
        print(f"No usable units. Wrote reports to {outdir}", flush=True)
        return outdir
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    jobs = max(1, int(args.jobs))
    if jobs == 1:
        for i, spec in enumerate(specs, start=1):
            print(f"[{i}/{len(specs)}] {spec.dataset_key} {spec.unit_id}", flush=True)
            rows.append(_run_one(spec, args))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = {executor.submit(_run_one, spec, args): spec for spec in specs}
            for i, future in enumerate(as_completed(futures), start=1):
                spec = futures[future]
                row = future.result()
                print(f"[{i}/{len(specs)}] {spec.dataset_key} {spec.unit_id}: {row['status']}", flush=True)
                rows.append(row)
    elapsed = float(time.perf_counter() - start)
    frame = _assign_labels(pd.DataFrame(rows))
    order = {spec.unit_id: i for i, spec in enumerate(specs)}
    frame = frame.assign(_order=frame["unit_id"].map(order)).sort_values("_order").drop(columns="_order")
    summary = build_summary(frame, specs, selected_keys)
    reps = build_representatives(frame)
    frame.to_csv(outdir / "per_unit_results.csv", index=False)
    summary.to_csv(outdir / "summary_by_dataset.csv", index=False)
    reps.to_csv(outdir / "representative_units.csv", index=False)
    write_scatter(frame, outdir)
    write_latex(summary, reps, outdir)
    write_report(args=args, outdir=outdir, notes=notes, frame=frame, summary=summary, reps=reps, elapsed=elapsed)
    print(f"Wrote results to {outdir}", flush=True)
    print(summary.to_string(index=False), flush=True)
    return outdir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run phase-1 cross-domain per-unit D_link diagnostics for NTD-PL.")
    parser.add_argument("--mode", choices=["inspect", "smoke", "full"], default="inspect")
    parser.add_argument("--datasets", default="brdf,lightfield,harvard")
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--jobs", type=int, default=max(1, min(6, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--p-max", "--p_max", dest="p_max", type=int, default=4)
    parser.add_argument("--n-iter-max", "--n_iter_max", dest="n_iter_max", type=int, default=180)
    parser.add_argument("--tucker-n-iter-max", "--tucker_n_iter_max", dest="tucker_n_iter_max", type=int, default=180)
    parser.add_argument("--init-n-iter-max", "--init_n_iter_max", dest="init_n_iter_max", type=int, default=50)
    parser.add_argument("--lr-core", type=float, default=7e-5)
    parser.add_argument("--lr-factors", type=float, default=2e-4)
    parser.add_argument("--lambda-beta", "--lambda_beta", dest="lambda_beta", type=float, default=1e-6)
    parser.add_argument("--link-sample-size", "--link_sample_size", dest="link_sample_size", type=int, default=400_000)
    parser.add_argument("--smoke-units", "--smoke_units", dest="smoke_units", type=int, default=3)
    parser.add_argument("--limit-brdf", "--limit_brdf", dest="limit_brdf", type=int, default=0)
    parser.add_argument("--limit-lightfield", "--limit_lightfield", dest="limit_lightfield", type=int, default=0)
    parser.add_argument("--limit-harvard", "--limit_harvard", dest="limit_harvard", type=int, default=0)
    parser.add_argument("--limit-kth", "--limit_kth", dest="limit_kth", type=int, default=0)
    parser.add_argument("--limit-ucf101", "--limit_ucf101", dest="limit_ucf101", type=int, default=0)
    parser.add_argument("--brdf-root", "--brdf_root", dest="brdf_root", default="BRDFDatabase")
    parser.add_argument("--lightfield-root", "--lightfield_root", dest="lightfield_root", default="caldata-B5143104560")
    parser.add_argument("--harvard-root", "--harvard_root", dest="harvard_root", default="CZ_hsdbi")
    parser.add_argument("--kth-root", "--kth_root", dest="kth_root", default="KTH-Action")
    parser.add_argument("--ucf101-root", "--ucf101_root", dest="ucf101_root", default="UCF101")
    parser.add_argument("--ucf101-split", "--ucf101_split", dest="ucf101_split", default="train")
    return parser.parse_args()


def main() -> None:
    _worker_env()
    run(parse_args())


if __name__ == "__main__":
    main()
