from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import gzip
import io
import math
import os
from pathlib import Path
import pickle
import re
import struct
import sys
import tarfile
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

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tensorly.tucker_tensor import tucker_to_tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.methods.ntdpl import NTDPLDecomposition
from src.methods.tucker import TuckerDecomposition
from src.metrics import val_RMSE, val_SAM
from src.types import LogCallback, Tensor
from src.utils.image_ops import downsample_image_to_shape


DEFAULT_OUT_ROOT = PROJECT_ROOT / "results" / "nonhsi_dlink_diagnostic"
RANKS = {
    "cbsd68": (32, 32, 3),
    "cifar10": (20, 20, 3),
    "coil100": (8, 12, 12, 2),
    "smallnorb": (12, 24, 24, 2),
}
DOMAINS = {
    "cbsd68": "Natural images",
    "cifar10": "Natural images",
    "coil100": "Object-view",
    "smallnorb": "Object-view",
}
DATASET_LABELS = {
    "cbsd68": "CBSD68",
    "cifar10": "CIFAR-10",
    "coil100": "COIL-100",
    "smallnorb": "smallNORB",
}

_CIFAR_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_SMALLNORB_CACHE: dict[str, tuple[np.ndarray, list[tuple[int, int]]]] = {}


@dataclass(frozen=True)
class UnitSpec:
    dataset_key: str
    unit_id: str
    unit_label: str
    source: str
    index: int = -1
    extra: tuple[Any, ...] = ()


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


def _normalize_max(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    scale = float(np.max(arr))
    if np.isfinite(scale) and scale > 1e-12:
        arr = arr / scale
    return arr.astype(np.float32, copy=False)


def _as_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"Expected an image array, got shape {arr.shape}")
    if arr.shape[-1] >= 3:
        arr = arr[..., :3]
    elif arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.max(initial=0.0) > 1.5:
        arr = arr / 255.0
    return arr.astype(np.float32, copy=False)


def _find_existing_root(candidates: list[Path], flag_name: str) -> Path:
    for root in candidates:
        if root.exists():
            return root
    tried = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not find dataset root for {flag_name}. Tried:\n  {tried}")


def build_unit_specs(args: argparse.Namespace) -> tuple[list[UnitSpec], list[str]]:
    notes: list[str] = []
    specs: list[UnitSpec] = []
    selected = {item.strip().lower() for item in args.datasets.split(",") if item.strip()}

    if "cbsd68" in selected or "cbsd" in selected:
        cbsd_default = PROJECT_ROOT / "data" / "CBSD68"
        root = _find_existing_root(
            [
                Path(args.cbsd68_root) if args.cbsd68_root else cbsd_default / "train",
                cbsd_default / "train",
                cbsd_default / "validation",
                PROJECT_ROOT / "data" / "CBSD68",
                PROJECT_ROOT / "data" / "cbsd",
            ],
            "--cbsd68_root",
        )
        if (root / "train").exists():
            root = root / "train"
        files = sorted([p for p in root.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}])
        files = files[:68]
        for i, path in enumerate(files):
            specs.append(UnitSpec("cbsd68", f"cbsd68_{i + 1:03d}", path.stem, str(path)))
        notes.append(f"CBSD68 root={root}; units={len(files)}; resize=96x96 RGB.")

    if "cifar10" in selected or "cifar" in selected:
        archive = _find_existing_root(
            [
                Path(args.cifar10_root) if args.cifar10_root else PROJECT_ROOT / "data" / "cifar-10-python.tar.gz",
                PROJECT_ROOT / "data" / "cifar-10-python.tar.gz",
                PROJECT_ROOT / "data" / "cifar-10-batches-py",
            ],
            "--cifar10_root",
        )
        images, labels = _load_cifar10_test(archive)
        indices = _class_balanced_indices(labels, per_class=max(1, int(args.cifar_per_class)), seed=int(args.seed))
        if args.limit_cifar and args.limit_cifar > 0:
            indices = indices[: int(args.limit_cifar)]
        for pos, idx in enumerate(indices):
            label = int(labels[idx])
            specs.append(UnitSpec("cifar10", f"cifar10_test_{idx:05d}", f"class{label}_test{idx}", str(archive), int(idx)))
        notes.append(f"CIFAR-10 source={archive}; class-balanced units={len(indices)}; shape=32x32x3.")

    if "coil100" in selected or "coil" in selected:
        root = _find_existing_root(
            [
                Path(args.coil100_root) if args.coil100_root else PROJECT_ROOT / "data" / "coil-100" / "coil-100",
                PROJECT_ROOT / "data" / "coil-100",
            ],
            "--coil100_root",
        )
        if (root / "coil-100").exists():
            root = root / "coil-100"
        object_ids = _coil_object_ids(root)
        if args.limit_coil and args.limit_coil > 0:
            object_ids = object_ids[: int(args.limit_coil)]
        for obj_id in object_ids:
            specs.append(UnitSpec("coil100", f"coil_obj{obj_id:03d}", f"object {obj_id}", str(root), int(obj_id)))
        notes.append(f"COIL-100 root={root}; objects={len(object_ids)}; views=36 via angles 0..350 step 10; resize=48x48 RGB.")

    if "smallnorb" in selected or "norb" in selected:
        root = _find_existing_root(
            [
                Path(args.smallnorb_root) if args.smallnorb_root else PROJECT_ROOT / "data" / "smallNORB",
                PROJECT_ROOT / "data" / "smallnorb",
            ],
            "--smallnorb_root",
        )
        _, keys = _load_smallnorb_units(root)
        if args.limit_smallnorb and args.limit_smallnorb > 0:
            keys = keys[: int(args.limit_smallnorb)]
        for idx, (cat, inst) in enumerate(keys):
            specs.append(UnitSpec("smallnorb", f"smallnorb_c{cat}_i{inst}", f"category {cat}, instance {inst}", str(root), idx, (cat, inst)))
        notes.append(f"smallNORB root={root}; object instances={len(keys)}; elevation=4 lighting=0 azimuth=18; resize=64x64 stereo.")

    if args.mode == "smoke":
        rng = np.random.default_rng(int(args.seed))
        smoke_specs = []
        for dataset_key in ("cbsd68", "cifar10", "coil100", "smallnorb"):
            panel = [spec for spec in specs if spec.dataset_key == dataset_key]
            if not panel:
                continue
            count = min(int(args.smoke_units), len(panel))
            if count >= len(panel):
                chosen = panel
            else:
                chosen_idx = sorted(rng.choice(len(panel), size=count, replace=False).tolist())
                chosen = [panel[i] for i in chosen_idx]
            smoke_specs.extend(chosen)
        specs = smoke_specs
    return specs, notes


def _load_cifar10_test(root: Path) -> tuple[np.ndarray, np.ndarray]:
    key = str(root.resolve())
    if key in _CIFAR_CACHE:
        return _CIFAR_CACHE[key]
    if root.is_dir():
        path = root / "test_batch"
        with path.open("rb") as handle:
            batch = pickle.load(handle, encoding="latin1")
    else:
        with tarfile.open(root, "r:gz") as archive:
            member = archive.getmember("cifar-10-batches-py/test_batch")
            handle = archive.extractfile(member)
            if handle is None:
                raise FileNotFoundError("test_batch not found inside CIFAR-10 archive")
            batch = pickle.load(handle, encoding="latin1")
    data = np.asarray(batch["data"], dtype=np.float32)
    labels = np.asarray(batch.get("labels", batch.get("fine_labels")), dtype=np.int32)
    images = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0
    _CIFAR_CACHE[key] = (images.astype(np.float32), labels)
    return _CIFAR_CACHE[key]


def _class_balanced_indices(labels: np.ndarray, *, per_class: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    out: list[int] = []
    for cls in sorted(np.unique(labels).tolist()):
        idx = np.flatnonzero(labels == cls)
        if idx.size < per_class:
            raise ValueError(f"CIFAR class {cls} has only {idx.size} examples, need {per_class}")
        chosen = rng.choice(idx, size=per_class, replace=False)
        out.extend(int(v) for v in chosen.tolist())
    return sorted(out)


def _coil_object_ids(root: Path) -> list[int]:
    ids: set[int] = set()
    for path in root.glob("obj*__*.png"):
        match = re.match(r"obj(\d+)__", path.name)
        if match:
            ids.add(int(match.group(1)))
    if not ids:
        raise FileNotFoundError(f"No COIL-100 object PNG files found under {root}")
    return sorted(ids)


def _load_unit(spec: UnitSpec) -> np.ndarray:
    if spec.dataset_key == "cbsd68":
        image = _as_rgb(mpimg.imread(Path(spec.source)))
        return _normalize_max(downsample_image_to_shape(image, (96, 96)))
    if spec.dataset_key == "cifar10":
        images, _ = _load_cifar10_test(Path(spec.source))
        return _normalize_max(images[int(spec.index)])
    if spec.dataset_key == "coil100":
        root = Path(spec.source)
        views = []
        for angle in range(0, 360, 10):
            path = root / f"obj{int(spec.index)}__{angle}.png"
            if not path.exists():
                raise FileNotFoundError(f"Missing COIL-100 view: {path}")
            image = _as_rgb(mpimg.imread(path))
            views.append(downsample_image_to_shape(image, (48, 48)))
        return _normalize_max(np.asarray(views, dtype=np.float32))
    if spec.dataset_key == "smallnorb":
        tensor, keys = _load_smallnorb_units(Path(spec.source))
        key_to_idx = {key: idx for idx, key in enumerate(keys)}
        cat, inst = int(spec.extra[0]), int(spec.extra[1])
        return _normalize_max(tensor[key_to_idx[(cat, inst)]])
    raise ValueError(f"Unknown dataset key: {spec.dataset_key}")


def _smallnorb_paths(root: Path) -> dict[str, Path]:
    files = {
        "dat": "smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat",
        "cat": "smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat",
        "info": "smallnorb-5x46789x9x18x6x2x96x96-training-info.mat",
    }
    out: dict[str, Path] = {}
    for key, name in files.items():
        path = root / name
        gz_path = root / f"{name}.gz"
        if path.exists():
            out[key] = path
        elif gz_path.exists():
            out[key] = gz_path
        else:
            raise FileNotFoundError(f"Missing smallNORB {key} file under {root}: {name}")
    return out


def _read_smallnorb_binary_matrix(path: Path) -> np.ndarray:
    magic_to_dtype = {
        0x1E3D4C51: np.float32,
        0x1E3D4C53: np.float64,
        0x1E3D4C54: np.int32,
        0x1E3D4C55: np.uint8,
        0x1E3D4C56: np.int16,
    }
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        header = handle.read(8)
        if len(header) != 8:
            raise ValueError(f"Invalid smallNORB header in {path}")
        magic, ndim = struct.unpack("<ii", header)
        if magic not in magic_to_dtype:
            raise ValueError(f"Unsupported smallNORB magic=0x{magic:08X} in {path}")
        base_dims = list(struct.unpack("<iii", handle.read(12)))
        dims = base_dims[: min(ndim, 3)]
        if ndim > 3:
            dims.extend(struct.unpack("<" + "i" * (ndim - 3), handle.read(4 * (ndim - 3))))
        dtype = magic_to_dtype[magic]
        count = int(np.prod(dims, dtype=np.int64))
        raw = handle.read()
    data = np.frombuffer(raw, dtype=dtype, count=count)
    if data.size != count:
        raise ValueError(f"Unexpected EOF in {path}: expected {count}, got {data.size}")
    return data.reshape(tuple(int(d) for d in dims))


def _load_smallnorb_units(root: Path) -> tuple[np.ndarray, list[tuple[int, int]]]:
    key = str(root.resolve())
    if key in _SMALLNORB_CACHE:
        return _SMALLNORB_CACHE[key]
    paths = _smallnorb_paths(root)
    images = _read_smallnorb_binary_matrix(paths["dat"]).astype(np.float32)
    cats = _read_smallnorb_binary_matrix(paths["cat"]).astype(np.int32).reshape(-1)
    info = _read_smallnorb_binary_matrix(paths["info"]).astype(np.int32).reshape(images.shape[0], -1)
    instance = info[:, 0]
    elevation = info[:, 1]
    azimuth = info[:, 2]
    lighting = info[:, 3]
    mask = (elevation == 4) & (lighting == 0)
    images = images[mask]
    cats = cats[mask]
    instance = instance[mask]
    azimuth = azimuth[mask]
    keys = sorted({(int(c), int(i)) for c, i in zip(cats.tolist(), instance.tolist())})
    out = np.empty((len(keys), 18, 64, 64, 2), dtype=np.float32)
    out.fill(np.nan)
    key_to_idx = {k: i for i, k in enumerate(keys)}
    for img_pair, cat, inst, az in zip(images, cats, instance, azimuth, strict=False):
        az_idx = int(az) // 2
        if not (0 <= az_idx < 18):
            continue
        img_hw2 = img_pair.transpose(1, 2, 0)
        out[key_to_idx[(int(cat), int(inst))], az_idx] = downsample_image_to_shape(img_hw2, (64, 64))
    if np.isnan(out).any():
        missing = int(np.isnan(out[..., 0]).sum())
        raise RuntimeError(f"smallNORB tensor has missing entries: {missing}")
    _SMALLNORB_CACHE[key] = (out, keys)
    return _SMALLNORB_CACHE[key]


def _fit_tucker(array: np.ndarray, rank: tuple[int, ...], *, n_iter_max: int) -> tuple[TuckerDecomposition, np.ndarray, float]:
    tensor = Tensor(shape=array.shape, dense=array)
    method = TuckerDecomposition(rank=rank, n_iter_max=int(n_iter_max), init="svd", tol=1e-5)
    start = time.perf_counter()
    method.fit(tensor, None, LogCallback(log_level=0))
    elapsed = float(time.perf_counter() - start)
    recon = np.asarray(method.reconstruct().dense, dtype=np.float32)
    return method, recon, elapsed


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
    recon = np.asarray(method.reconstruct().dense, dtype=np.float32)
    return method, recon, elapsed


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
        "shape": "",
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
        "notes": "",
    }
    try:
        array = _load_unit(spec)
        rank = tuple(min(int(r), int(dim)) for r, dim in zip(RANKS[spec.dataset_key], array.shape))
        row["shape"] = "x".join(str(v) for v in array.shape)
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
                "notes": f"link_refresh_rmse={link_rmse:.8g}; p_max={int(args.p_max)}",
            }
        )
    except Exception as exc:
        row["status"] = "failed"
        row["notes"] = f"{type(exc).__name__}: {exc}"
    return row


def _assign_labels(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["diagnostic_label"] = ""
    ok = out["status"].eq("ok") & np.isfinite(out["d_link_db"].astype(float))
    for dataset, idx in out.loc[ok].groupby("dataset").groups.items():
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
    return out


def _tercile_mean_gain(panel: pd.DataFrame, which: str) -> float:
    ok = panel.loc[panel["status"].eq("ok")].sort_values("d_link_db")
    if ok.empty:
        return float("nan")
    groups = np.array_split(ok, 3)
    mapping = {"low": groups[0], "mid": groups[1], "high": groups[2]}
    group = mapping[which]
    return float(group["rmse_gain_pct"].mean()) if not group.empty else float("nan")


def build_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset, panel in frame.groupby("dataset", sort=False):
        ok = panel.loc[panel["status"].eq("ok")].copy()
        if ok.shape[0] >= 2 and ok["d_link_db"].nunique() > 1 and ok["rmse_gain_pct"].nunique() > 1:
            corr = spearmanr(ok["d_link_db"], ok["rmse_gain_pct"], nan_policy="omit")
            spearman = float(corr.statistic)
        else:
            spearman = float("nan")
        rows.append(
            {
                "domain": str(panel["domain"].iloc[0]),
                "dataset": dataset,
                "num_units": int(ok.shape[0]),
                "median_d_link": float(ok["d_link_db"].median()) if not ok.empty else float("nan"),
                "mean_d_link": float(ok["d_link_db"].mean()) if not ok.empty else float("nan"),
                "spearman_dlink_gain": spearman,
                "mean_gain": float(ok["rmse_gain_pct"].mean()) if not ok.empty else float("nan"),
                "median_gain": float(ok["rmse_gain_pct"].median()) if not ok.empty else float("nan"),
                "low_tercile_gain": _tercile_mean_gain(panel, "low"),
                "mid_tercile_gain": _tercile_mean_gain(panel, "mid"),
                "high_tercile_gain": _tercile_mean_gain(panel, "high"),
                "num_effective": int(panel["diagnostic_label"].eq("effective").sum()),
                "num_boundary": int(panel["diagnostic_label"].eq("boundary").sum()),
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
        "shape",
    ]
    return pd.concat(rows, ignore_index=True).drop_duplicates(["dataset", "unit_id", "selection"])[cols]


def write_scatter(frame: pd.DataFrame, outdir: Path) -> None:
    ok = frame.loc[frame["status"].eq("ok")].copy()
    fig, ax = plt.subplots(figsize=(5.2, 3.5), constrained_layout=True)
    markers = {"CBSD68": "o", "CIFAR-10": "s", "COIL-100": "^", "smallNORB": "D"}
    colors = {"CBSD68": "#4C78A8", "CIFAR-10": "#72B7B2", "COIL-100": "#F58518", "smallNORB": "#54A24B"}
    for dataset, panel in ok.groupby("dataset", sort=False):
        ax.scatter(
            panel["d_link_db"],
            panel["rmse_gain_pct"],
            s=28,
            marker=markers.get(dataset, "o"),
            color=colors.get(dataset, "#666666"),
            alpha=0.78,
            linewidths=0.4,
            edgecolors="white",
            label=dataset,
        )
    ax.axhline(0.0, color="#666666", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel(r"$D_{\mathrm{link}}$ (dB)")
    ax.set_ylabel("NTD-PL RMSE gain (%)")
    ax.legend(frameon=False, fontsize=8, ncol=2)
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
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]
    def _row(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(values)) + " |"
    lines = [
        _row(headers),
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    lines.extend(_row(row) for row in rows)
    return "\n".join(lines)


def write_latex(summary: pd.DataFrame, reps: pd.DataFrame, outdir: Path) -> None:
    lines = [
        "% Candidate tables generated by scripts/run_nonhsi_dlink_diagnostic.py.",
        "% Labels and representatives are selected by D_link, not by NTD-PL gain.",
        "",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Dataset-level non-HSI link diagnostics (candidate).}",
        r"\begin{tabular}{@{}l l l r r r r r l@{}}",
        r"\toprule",
        r"Domain & Dataset & Unit & \#Units & median $D_{\mathrm{link}}$ & Spearman & Low-yield gain & High-yield gain & Reading \\",
        r"\midrule",
    ]
    unit_text = {"CBSD68": "image", "CIFAR-10": "image", "COIL-100": "object views", "smallNORB": "instance views"}
    for row in summary.itertuples(index=False):
        reading = "modest" if row.domain == "Natural images" else "clearer object-view units"
        lines.append(
            f"{row.domain} & {row.dataset} & {unit_text.get(row.dataset, 'unit')} & {int(row.num_units)} & "
            f"{_fmt(row.median_d_link, 3)} & {_fmt(row.spearman_dlink_gain, 2)} & "
            f"{_fmt(row.low_tercile_gain, 2)}\\% & {_fmt(row.high_tercile_gain, 2)}\\% & {reading} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    lines.extend(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Representative non-HSI units selected by $D_{\mathrm{link}}$ (candidate).}",
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


def write_report(
    *,
    args: argparse.Namespace,
    outdir: Path,
    data_notes: list[str],
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    reps: pd.DataFrame,
    elapsed: float,
) -> None:
    failures = frame.loc[~frame["status"].eq("ok"), ["dataset", "unit_id", "notes"]]
    natural = summary.loc[summary["domain"].eq("Natural images")]
    obj = summary.loc[summary["domain"].eq("Object-view")]
    natural_modest = bool((natural["median_d_link"].fillna(0.0) < 0.10).all()) if not natural.empty else False
    object_clear = bool((obj["num_effective"].fillna(0).sum() > 0)) if not obj.empty else False
    lines = [
        "# Non-HSI D_link Diagnostic",
        "",
        f"- Mode: `{args.mode}`",
        f"- Output directory: `{outdir}`",
        f"- Elapsed wall time: {elapsed:.1f} s",
        f"- Seed: {int(args.seed)}",
        f"- Tucker iterations: {int(args.tucker_n_iter_max)}",
        f"- NTD-PL iterations: {int(args.n_iter_max)}",
        f"- NTD-PL polynomial degree: {int(args.p_max)}",
        f"- NTD-PL ridge lambda_beta: {float(args.lambda_beta):g}",
        f"- Link refresh sample size: {int(args.link_sample_size)} (<=0 means all entries)",
        "",
        "## Data Paths And Preprocessing",
        "",
    ]
    lines.extend(f"- {note}" for note in data_notes)
    lines.extend(
        [
            "- All units are max-normalized independently before fitting.",
            "- Natural-image units are single images; object-view units are per-object/per-instance multi-view tensors.",
            "- No cross-class or cross-object giant tensor is used for the main diagnostic unit.",
            "",
            "## Unit Definitions And Ranks",
            "",
            "- CBSD68: one 96x96x3 image, rank (32,32,3).",
            "- CIFAR-10: one 32x32x3 test image, rank (20,20,3).",
            "- COIL-100: one 36x48x48x3 object-view tensor, rank (8,12,12,2).",
            "- smallNORB: one 18x64x64x2 instance-view tensor, rank (12,24,24,2).",
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
            _markdown_table(failures) if not failures.empty else "No failed units.",
            "",
            "## Main Conclusions",
            "",
            f"- Natural images mostly weak/modest shared-link yield: {'yes' if natural_modest else 'mixed or not supported in this run'}.",
            f"- Object-view tensors show clearer effective units: {'yes' if object_clear else 'not clearly in this run'}.",
            "- Table 3 update recommendation: use this output as candidate evidence only; main.tex was not modified.",
            "",
            "## Reproduction Command",
            "",
            "```powershell",
            "python scripts/run_nonhsi_dlink_diagnostic.py "
            f"--mode {args.mode} --datasets {args.datasets} --jobs {int(args.jobs)} "
            f"--seed {int(args.seed)} --n-iter-max {int(args.n_iter_max)} --tucker-n-iter-max {int(args.tucker_n_iter_max)} "
            f"--p-max {int(args.p_max)} --outdir {outdir}",
            "```",
            "",
        ]
    )
    (outdir / ("smoke_test_report.md" if args.mode == "smoke" else "report.md")).write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) if args.outdir else DEFAULT_OUT_ROOT / timestamp
    outdir.mkdir(parents=True, exist_ok=True)
    specs, data_notes = build_unit_specs(args)
    if not specs:
        raise RuntimeError("No units selected. Check --datasets and dataset roots.")
    start = time.perf_counter()
    rows: list[dict[str, Any]] = []
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
    summary = build_summary(frame)
    reps = build_representatives(frame)

    frame.to_csv(outdir / "per_unit_results.csv", index=False)
    summary.to_csv(outdir / "summary_by_dataset.csv", index=False)
    reps.to_csv(outdir / "representative_units.csv", index=False)
    write_scatter(frame, outdir)
    write_latex(summary, reps, outdir)
    write_report(args=args, outdir=outdir, data_notes=data_notes, frame=frame, summary=summary, reps=reps, elapsed=elapsed)
    print(f"Wrote results to {outdir}", flush=True)
    print(summary.to_string(index=False), flush=True)
    return outdir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run non-HSI per-unit D_link diagnostics for NTD-PL.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--datasets", default="cbsd68,cifar10,coil100,smallnorb")
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--jobs", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--p-max", dest="p_max", type=int, default=4)
    parser.add_argument("--n-iter-max", dest="n_iter_max", type=int, default=180)
    parser.add_argument("--tucker-n-iter-max", dest="tucker_n_iter_max", type=int, default=180)
    parser.add_argument("--init-n-iter-max", dest="init_n_iter_max", type=int, default=50)
    parser.add_argument("--lr-core", type=float, default=7e-5)
    parser.add_argument("--lr-factors", type=float, default=2e-4)
    parser.add_argument("--lambda-beta", type=float, default=1e-6)
    parser.add_argument("--link-sample-size", type=int, default=400_000)
    parser.add_argument("--smoke-units", type=int, default=3)
    parser.add_argument("--cifar-per-class", type=int, default=100)
    parser.add_argument("--limit-cifar", type=int, default=0)
    parser.add_argument("--limit-coil", type=int, default=0)
    parser.add_argument("--limit-smallnorb", type=int, default=0)
    parser.add_argument("--cbsd68-root", default="")
    parser.add_argument("--cifar10-root", default="")
    parser.add_argument("--coil100-root", default="")
    parser.add_argument("--smallnorb-root", default="")
    return parser.parse_args()


def main() -> None:
    _worker_env()
    run(parse_args())


if __name__ == "__main__":
    main()
