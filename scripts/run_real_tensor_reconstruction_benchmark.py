from __future__ import annotations

import argparse
import gzip
import pickle
import shutil
import struct
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import urllib.request
import tarfile
import zipfile
import os
import re
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment.process.helpers.real_hsi_robustness import (
    DATASET_ORDER as HSI_DATASET_ORDER,
    build_main_table,
    build_summary,
    load_main_runs,
)
from src.methods.ntdpl import NTDPLDecomposition
from src.methods.tucker import TuckerDecomposition
from src.metrics import val_NMSE_dB, val_RMSE, val_SAM, val_SSIM
from src.types import LogCallback, Tensor
from src.utils.image_ops import downsample_image_to_shape


OUT_PREFIX = PROJECT_ROOT / "papers" / "neurips" / "tables" / "real_tensor_reconstruction_main"


DATASET_ORDERING: dict[str, tuple[int, int]] = {
    "cbsd": (1, 0),
    "cifar": (1, 1),
    "coil": (2, 0),
    "norb": (2, 1),
    "eth8": (2, 2),
    "eth": (2, 3),
    "har": (3, 0),
    "dj": (4, 0),
}


BENCHMARKS: dict[str, dict[str, Any]] = {
    "coil": {
        "domain": "Object views",
        "dataset": "COIL-100",
        "rank": (4, 8, 12, 12, 2),
        "p_max": 3,
        "n_iter_max": 140,
        "loader": "_load_coil100",
    },
    "norb": {
        "domain": "Object views",
        "dataset": "smallNORB",
        "rank": (18, 12, 24, 24, 2),
        "p_max": 4,
        "n_iter_max": 180,
        "lr_core": 7e-5,
        "lr_factors": 2e-4,
        "loader": "_load_smallnorb_25obj_az18",
    },
    "eth": {
        "domain": "Object views",
        "dataset": "ETH-80",
        "rank": (8, 8, 16, 16, 3),
        "p_max": 3,
        "n_iter_max": 140,
        "loader": "_load_eth80",
    },
    "eth8": {
        "domain": "Object views",
        "dataset": "ETH-80 (8 objs)",
        "rank": (4, 8, 16, 16, 3),
        "p_max": 3,
        "n_iter_max": 140,
        "loader": "_load_eth80_8objs_first",
    },
    "cifar": {
        "domain": "Natural images",
        "dataset": "CIFAR-10",
        "rank": (96, 20, 20, 3),
        "p_max": 4,
        "n_iter_max": 180,
        "lr_core": 7e-5,
        "lr_factors": 2e-4,
        "loader": "_load_cifar10",
    },
    "cbsd": {
        "domain": "Natural images",
        "dataset": "CBSD68",
        "rank": (20, 32, 32, 3),
        "p_max": 4,
        "n_iter_max": 180,
        "lr_core": 7e-5,
        "lr_factors": 2e-4,
        "loader": "_load_cbsd68",
    },
    "har": {
        "domain": "Sensors",
        "dataset": "UCI HAR",
        "rank": (4, 2, 8, 2),
        "p_max": 3,
        "n_iter_max": 140,
        "loader": "_load_uci_har_subject_activity_tensor",
    },
    "dj": {
        "domain": "Finance",
        "dataset": "Dow Jones Index",
        "rank": (10, 10, 1),
        "p_max": 3,
        "n_iter_max": 140,
        "loader": "_load_dow_jones_index_tensor",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect real-tensor reconstruction results for NeurIPS 6.4.")
    parser.add_argument("--datasets", default="coil,cifar,cbsd")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-running selected datasets even if they exist in the cached CSV.",
    )
    args = parser.parse_args()

    rows = _load_hsi_rows()
    selected = [item.strip() for item in args.datasets.split(",") if item.strip()]
    existing = _read_existing_rows()
    for name in selected:
        if name not in BENCHMARKS:
            raise ValueError(f"Unknown dataset {name!r}. Choices: {sorted(BENCHMARKS)}")
        if args.collect_only and name not in existing:
            continue
        if name in existing and not args.force:
            rows.extend(existing[name])
            continue
        rows.extend(_run_benchmark(name, BENCHMARKS[name]))

    table = pd.DataFrame(rows)
    if table.empty:
        raise RuntimeError("No rows collected.")
    table = table.sort_values(["domain_order", "dataset_order"]).drop(columns=["domain_order", "dataset_order"])
    OUT_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_PREFIX.with_suffix(".csv"), index=False)
    OUT_PREFIX.with_suffix(".tex").write_text(_to_latex(table), encoding="utf-8")
    print(f"Wrote {OUT_PREFIX.with_suffix('.csv')}")
    print(f"Wrote {OUT_PREFIX.with_suffix('.tex')}")
    print(table.to_string(index=False))


def _load_hsi_rows() -> list[dict[str, Any]]:
    frame, metadata, _ = load_main_runs()
    summary = build_summary(frame)
    main_table = build_main_table(summary)
    rows: list[dict[str, Any]] = []
    keep = {"jasper_ridge_hsi", "samson_hsi", "urban_hsi"}
    meta_lookup = metadata.set_index("dataset").to_dict("index")
    for dataset_name in HSI_DATASET_ORDER:
        if dataset_name not in keep:
            continue
        panel = main_table.loc[
            main_table["dataset"].eq(dataset_name) & main_table["task_name"].eq("decompose")
        ]
        if panel.empty:
            continue
        row = panel.iloc[0]
        meta = meta_lookup[dataset_name]
        recon_runs = frame.loc[frame["dataset"].eq(dataset_name) & frame["task_name"].eq("decompose")]
        tucker_nmse = _lookup_metric(recon_runs, "tucker", "NMSE_dB")
        ntdpl_nmse = _lookup_metric(recon_runs, "ntdpl", "NMSE_dB")
        rows.append(
            {
                "domain_order": 0,
                "dataset_order": len(rows),
                "domain": "Hyperspectral",
                "dataset": row["dataset_label"],
                "shape": str(meta["source_shape"]).replace("x", r"$\times$"),
                "rank": row["rank"],
                "tucker_rmse": float(row["tucker_rmse"]),
                "ntdpl_rmse": float(row["ntdpl_rmse"]),
                "tucker_nmse_db": tucker_nmse,
                "ntdpl_nmse_db": ntdpl_nmse,
                "nmse_gain_pct": _nmse_gain_pct(tucker_nmse, ntdpl_nmse),
                "gain_pct": float(row["gain_pct"]),
                "delta_nmse_db": float(row["delta_nmse_db"]),
                "delta_sam": float(row["delta_sam"]),
            }
        )
    return rows


def _read_existing_rows() -> dict[str, list[dict[str, Any]]]:
    if not OUT_PREFIX.with_suffix(".csv").exists():
        return {}
    frame = pd.read_csv(OUT_PREFIX.with_suffix(".csv"))
    if "tucker_rmse" not in frame.columns or "ntdpl_rmse" not in frame.columns:
        return {}
    out: dict[str, list[dict[str, Any]]] = {}
    aliases = {
        "CBSD68": "cbsd",
        "CIFAR-10": "cifar",
        "COIL-100": "coil",
        "smallNORB": "norb",
        "ETH-80": "eth",
        "ETH-80 (8 objs)": "eth8",
        "UCI HAR": "har",
        "Dow Jones Index": "dj",
    }
    for dataset_label, key in aliases.items():
        panel = frame.loc[frame["dataset"].eq(dataset_label)].copy()
        if not panel.empty:
            panel["domain_order"] = DATASET_ORDERING[key][0]
            panel["dataset_order"] = DATASET_ORDERING[key][1]
            out[key] = panel.to_dict("records")
    return out


def _lookup_metric(frame: pd.DataFrame, method_name: str, metric: str) -> float:
    panel = frame.loc[frame["method_name"].eq(method_name)]
    if panel.empty or metric not in panel.columns:
        raise KeyError(f"Missing {metric} for {method_name}")
    return float(panel.iloc[0][metric])


def _run_benchmark(name: str, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    loader = globals()[str(cfg["loader"])]
    dense = np.asarray(loader(), dtype=np.float32)
    dense = _normalize_max(dense)
    rank = tuple(int(v) for v in cfg["rank"])
    tensor = Tensor(shape=dense.shape, dense=dense)

    results: dict[str, dict[str, float]] = {}
    for method_name in ("tucker", "ntdpl"):
        start = time.perf_counter()
        if method_name == "tucker":
            method = TuckerDecomposition(rank=rank, n_iter_max=int(cfg["n_iter_max"]), init="svd", tol=1e-4)
        else:
            method = NTDPLDecomposition(
                rank=rank,
                init_n_iter_max=40,
                init="tucker",
                solver_variant="optimized",
                stable_beta_update=True,
                beta_update_stage="before_grad",
                random_state=0,
                p_max=int(cfg["p_max"]),
                allow_constant_term=True,
                use_continuation=True,
                factor_normalize=True,
                lr_core=float(cfg.get("lr_core", 1e-4)),
                lr_factors=float(cfg.get("lr_factors", 3e-4)),
                lambda_core=1e-6,
                lambda_factors=1e-6,
                lambda_beta=1e-6,
                beta_update_method="ridge_lstsq",
                beta_update_interval=5,
                n_iter_max=int(cfg["n_iter_max"]),
            )
        method.fit(tensor, None, LogCallback(log_level=0))
        recon = method.reconstruct()
        results[method_name] = {
            "rmse": val_RMSE(tensor, recon),
            "nmse_db": val_NMSE_dB(tensor, recon),
            "ssim": val_SSIM(tensor, recon),
            "sam": val_SAM(tensor, recon),
            "fit_time_sec": float(time.perf_counter() - start),
        }
        print(f"{cfg['dataset']} {method_name}: {results[method_name]}")

    tucker = results["tucker"]
    ntdpl = results["ntdpl"]
    return [
        {
            "domain_order": DATASET_ORDERING[name][0],
            "dataset_order": DATASET_ORDERING[name][1],
            "domain": cfg["domain"],
            "dataset": cfg["dataset"],
            "shape": r"$\times$".join(str(v) for v in dense.shape),
            "rank": f"({','.join(str(v) for v in rank)})",
            "tucker_rmse": tucker["rmse"],
            "ntdpl_rmse": ntdpl["rmse"],
            "tucker_nmse_db": tucker["nmse_db"],
            "ntdpl_nmse_db": ntdpl["nmse_db"],
            "nmse_gain_pct": _nmse_gain_pct(tucker["nmse_db"], ntdpl["nmse_db"]),
            "gain_pct": 100.0 * (tucker["rmse"] - ntdpl["rmse"]) / max(tucker["rmse"], 1e-12),
            "delta_nmse_db": tucker["nmse_db"] - ntdpl["nmse_db"],
            "delta_ssim": ntdpl["ssim"] - tucker["ssim"],
            "delta_sam": tucker["sam"] - ntdpl["sam"],
        }
    ]


def _load_coil100() -> np.ndarray:
    root = PROJECT_ROOT / "data" / "coil-100"
    objects = list(range(1, 9))
    angles = list(range(0, 360, 10))
    frames = []
    for obj_id in objects:
        views = []
        for angle in angles:
            image = np.asarray(mpimg.imread(root / f"obj{obj_id}__{angle}.png"), dtype=np.float32)
            views.append(downsample_image_to_shape(image[..., :3], (48, 48)))
        frames.append(np.asarray(views, dtype=np.float32))
    return np.asarray(frames, dtype=np.float32)


def _load_eth80() -> np.ndarray:
    root = PROJECT_ROOT / "data" / "eth-80"
    if not root.exists():
        raise FileNotFoundError(f"ETH-80 dataset not found under {root}")

    # ETH-80 structure: 8 categories (1-8), each with 10 objects (1-10)
    # Each object folder contains PNG images at different viewing angles
    frames = []
    for category in range(1, 9):  # categories 1-8
        category_dir = root / str(category)
        if not category_dir.exists():
            print(f"Warning: category {category} not found")
            continue
        for obj_id in range(1, 11):  # objects 1-10 per category
            obj_dir = category_dir / str(obj_id)
            if not obj_dir.exists():
                print(f"Warning: object {category}/{obj_id} not found")
                continue
            views = []
            files = sorted(obj_dir.glob("*.png"))
            if not files:
                print(f"Warning: no PNG files in {obj_dir}")
                continue
            for f in files:
                try:
                    image = np.asarray(mpimg.imread(f), dtype=np.float32)
                    # Ensure we take only RGB channels
                    if image.ndim == 3 and image.shape[2] >= 3:
                        image = image[..., :3]
                    elif image.ndim == 2:
                        image = np.stack([image, image, image], axis=-1)
                    views.append(downsample_image_to_shape(image, (64, 64)))
                except Exception as e:
                    print(f"Warning: failed to load {f}: {e}")
            if views:
                frames.append(np.asarray(views, dtype=np.float32))
    
    if not frames:
        raise FileNotFoundError(f"No valid object frames found under {root}")
    print(f"Loaded {len(frames)} objects from ETH-80")
    result = np.asarray(frames, dtype=np.float32)
    print(f"ETH-80 tensor shape: {result.shape}")
    return result


def _download_uci_har(root: Path) -> Path:
    """Download and extract UCI HAR Dataset.

    Source: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
    """

    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "UCI HAR Dataset.zip"
    extracted_root = raw_dir / "UCI HAR Dataset"

    if extracted_root.exists():
        return extracted_root

    if not zip_path.exists():
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
        print(f"Downloading UCI HAR: {url} -> {zip_path}")
        urllib.request.urlretrieve(url, zip_path)

    print(f"Extracting UCI HAR: {zip_path} -> {raw_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)
    if not extracted_root.exists():
        raise FileNotFoundError(f"Expected extracted folder not found: {extracted_root}")
    return extracted_root


def _download_dow_jones_index(root: Path) -> Path:
    """Download and extract the UCI Dow Jones Index dataset."""

    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "dow+jones+index.zip"

    if not zip_path.exists():
        url = "https://archive.ics.uci.edu/static/public/312/dow+jones+index.zip"
        print(f"Downloading Dow Jones Index: {url} -> {zip_path}")
        urllib.request.urlretrieve(url, zip_path)

    print(f"Extracting Dow Jones Index: {zip_path} -> {raw_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)

    matches = list(raw_dir.rglob("dow_jones_index.data"))
    if not matches:
        raise FileNotFoundError(f"dow_jones_index.data not found under {raw_dir}")
    return matches[0]


def _load_dow_jones_index_tensor() -> np.ndarray:
    """Build a dense stock×week×feature tensor from the UCI Dow Jones Index dataset.

    Output shape: (30, T, 5)
      - stocks: 30 DJIA components (symbols)
      - weeks: common dates across all stocks
      - features: [open, high, low, close, log1p(volume)]
    """

    root = PROJECT_ROOT / "data" / "dow-jones"
    data_path = _download_dow_jones_index(root)

    df = pd.read_csv(data_path)
    required = {"stock", "date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dow Jones dataset missing columns: {sorted(missing)}")

    # Clean price columns like '$123.45'
    def _clean_money(series: pd.Series) -> pd.Series:
        return pd.to_numeric(
            series.astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False),
            errors="coerce",
        )

    for col in ("open", "high", "low", "close"):
        df[col] = _clean_money(df[col])
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["stock", "date", "open", "high", "low", "close", "volume"]).copy()

    # Align weeks by intersection of dates across all stocks.
    stocks = sorted(df["stock"].unique().tolist())
    if len(stocks) != 30:
        print(f"Warning: expected 30 stocks, got {len(stocks)}")

    date_sets = []
    for s in stocks:
        date_sets.append(set(df.loc[df["stock"].eq(s), "date"].tolist()))
    common_dates = sorted(set.intersection(*date_sets)) if date_sets else []
    if not common_dates:
        raise RuntimeError("No common dates across Dow Jones stocks")

    features = ["open", "high", "low", "close", "volume"]
    out = np.empty((len(stocks), len(common_dates), 5), dtype=np.float32)

    for i, s in enumerate(stocks):
        panel = df.loc[df["stock"].eq(s) & df["date"].isin(common_dates), ["date"] + features].copy()
        panel = panel.sort_values("date")
        if panel.shape[0] != len(common_dates):
            raise RuntimeError(f"Stock {s} missing aligned weeks")
        arr = panel[features].to_numpy(dtype=np.float32)
        # Stabilize scale for volume
        arr[:, 4] = np.log1p(arr[:, 4])
        out[i] = arr

    print(f"Loaded Dow Jones Index tensor shape: {out.shape}")
    return out


def _read_uci_har_inertial_split(dataset_root: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read inertial signals for a split.

    Returns:
      signals: (N, 128, 9) float32
      y:       (N,) int32 labels in 1..6
      subject: (N,) int32 in 1..30
    """

    split = str(split)
    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    base = dataset_root / split
    inertial = base / "Inertial Signals"
    if not inertial.exists():
        raise FileNotFoundError(f"Missing inertial signals folder: {inertial}")

    channel_specs = [
        ("body_acc_x", inertial / f"body_acc_x_{split}.txt"),
        ("body_acc_y", inertial / f"body_acc_y_{split}.txt"),
        ("body_acc_z", inertial / f"body_acc_z_{split}.txt"),
        ("body_gyro_x", inertial / f"body_gyro_x_{split}.txt"),
        ("body_gyro_y", inertial / f"body_gyro_y_{split}.txt"),
        ("body_gyro_z", inertial / f"body_gyro_z_{split}.txt"),
        ("total_acc_x", inertial / f"total_acc_x_{split}.txt"),
        ("total_acc_y", inertial / f"total_acc_y_{split}.txt"),
        ("total_acc_z", inertial / f"total_acc_z_{split}.txt"),
    ]

    channels = []
    for _, path in channel_specs:
        if not path.exists():
            raise FileNotFoundError(f"Missing UCI HAR file: {path}")
        # Each row is a 128-sample window.
        arr = np.loadtxt(path, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 128:
            raise ValueError(f"Unexpected inertial shape in {path}: {arr.shape}")
        channels.append(arr)

    # Stack to (N, 128, 9)
    stacked = np.stack(channels, axis=-1)

    y_path = base / f"y_{split}.txt"
    subject_path = base / f"subject_{split}.txt"
    y = np.loadtxt(y_path, dtype=np.int32).reshape(-1)
    subject = np.loadtxt(subject_path, dtype=np.int32).reshape(-1)
    if y.shape[0] != stacked.shape[0] or subject.shape[0] != stacked.shape[0]:
        raise ValueError("UCI HAR split length mismatch")
    return stacked, y, subject


def _load_uci_har_subject_activity_tensor() -> np.ndarray:
    """Build a dense tensor from UCI HAR raw inertial signals.

    We aggregate all windows (train+test) by taking the mean waveform for each
    (subject, activity) pair.

    Output shape: (30, 6, 128, 9)
    """

    root = PROJECT_ROOT / "data" / "uci-har"
    extracted = _download_uci_har(root)

    sig_tr, y_tr, subj_tr = _read_uci_har_inertial_split(extracted, "train")
    sig_te, y_te, subj_te = _read_uci_har_inertial_split(extracted, "test")

    signals = np.concatenate([sig_tr, sig_te], axis=0)
    y = np.concatenate([y_tr, y_te], axis=0)
    subject = np.concatenate([subj_tr, subj_te], axis=0)

    out = np.zeros((30, 6, 128, 9), dtype=np.float32)
    counts = np.zeros((30, 6), dtype=np.int32)

    for s, a, window in zip(subject.tolist(), y.tolist(), signals):
        s_idx = int(s) - 1
        a_idx = int(a) - 1
        if not (0 <= s_idx < 30 and 0 <= a_idx < 6):
            continue
        out[s_idx, a_idx] += window
        counts[s_idx, a_idx] += 1

    if np.any(counts == 0):
        missing = int((counts == 0).sum())
        raise RuntimeError(f"UCI HAR has missing (subject, activity) pairs: {missing}")

    out = out / counts[:, :, None, None].astype(np.float32)
    print("Loaded UCI HAR tensor shape:", out.shape)
    return out


def _download_smallnorb_files(root: Path) -> dict[str, Path]:
    """Download and extract the smallNORB training split (if needed).

    Data source: https://cs.nyu.edu/~yann/data/norb-v1.0-small/
    """

    base_url = "https://cs.nyu.edu/~yann/data/norb-v1.0-small/"
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "dat": "smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat",
        "cat": "smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat",
        "info": "smallnorb-5x46789x9x18x6x2x96x96-training-info.mat",
    }

    out: dict[str, Path] = {}
    for key, fname in files.items():
        mat_path = raw_dir / fname
        if not mat_path.exists():
            gz_path = raw_dir / f"{fname}.gz"
            if not gz_path.exists():
                url = base_url + f"{fname}.gz"
                print(f"Downloading smallNORB: {url} -> {gz_path}")
                urllib.request.urlretrieve(url, gz_path)
            print(f"Extracting smallNORB: {gz_path} -> {mat_path}")
            with gzip.open(gz_path, "rb") as src, mat_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        out[key] = mat_path
    return out


def _read_smallnorb_binary_matrix(path: Path) -> np.ndarray:
    """Read a smallNORB '.mat' binary-matrix file into a NumPy array."""

    magic_to_dtype = {
        0x1E3D4C51: np.float32,
        0x1E3D4C53: np.float64,
        0x1E3D4C54: np.int32,
        0x1E3D4C55: np.uint8,
        0x1E3D4C56: np.int16,
    }

    with path.open("rb") as handle:
        header = handle.read(8)
        if len(header) != 8:
            raise ValueError(f"Invalid smallNORB header in {path}")
        magic, ndim = struct.unpack("<ii", header)
        if magic not in magic_to_dtype:
            raise ValueError(f"Unsupported smallNORB magic=0x{magic:08X} in {path}")
        if ndim <= 0:
            raise ValueError(f"Invalid smallNORB ndim={ndim} in {path}")

        # The header always stores at least 3 dims. More dims are stored after.
        base_dims = list(struct.unpack("<iii", handle.read(12)))
        dims = base_dims[: min(ndim, 3)]
        if ndim > 3:
            dims.extend(struct.unpack("<" + "i" * (ndim - 3), handle.read(4 * (ndim - 3))))

        dtype = magic_to_dtype[magic]
        count = int(np.prod(dims, dtype=np.int64))
        data = np.fromfile(handle, dtype=dtype, count=count)
        if data.size != count:
            raise ValueError(f"Unexpected EOF in {path}: expected {count} elems, got {data.size}")
        return data.reshape(tuple(int(d) for d in dims))


def _load_smallnorb_25obj_az18() -> np.ndarray:
    """Build an object-views tensor from smallNORB.

    We use the training split only (25 objects total = 5 categories × 5 instances).
    For each object we fix elevation=4 and lighting=0 and sweep all 18 azimuths,
    keeping the stereo pair as the channel dimension.

    Output shape: (25, 18, 64, 64, 2)
    """

    root = PROJECT_ROOT / "data" / "smallnorb"
    paths = _download_smallnorb_files(root)

    images = _read_smallnorb_binary_matrix(paths["dat"]).astype(np.float32)
    cats = _read_smallnorb_binary_matrix(paths["cat"]).astype(np.int32).reshape(-1)
    info = _read_smallnorb_binary_matrix(paths["info"]).astype(np.int32)
    info = info.reshape(info.shape[0], -1)
    if info.shape[1] < 4:
        raise ValueError(f"Unexpected smallNORB info shape: {info.shape}")

    if images.ndim != 4 or images.shape[1] != 2 or images.shape[2] != 96 or images.shape[3] != 96:
        raise ValueError(f"Unexpected smallNORB image tensor shape: {images.shape}")
    if cats.shape[0] != images.shape[0] or info.shape[0] != images.shape[0]:
        raise ValueError("smallNORB cat/info length mismatch")

    instance = info[:, 0]
    elevation = info[:, 1]
    azimuth = info[:, 2]
    lighting = info[:, 3]

    elev_fixed = 4
    light_fixed = 0
    mask = (elevation == elev_fixed) & (lighting == light_fixed)

    images = images[mask]
    cats = cats[mask]
    instance = instance[mask]
    azimuth = azimuth[mask]

    # Expect 5 categories × 5 instances × 18 azimuths = 450 examples.
    if images.shape[0] != 450:
        print(f"Warning: expected 450 fixed-pose samples, got {images.shape[0]}")

    obj_keys = sorted({(int(c), int(i)) for c, i in zip(cats.tolist(), instance.tolist())})
    obj_index = {k: idx for idx, k in enumerate(obj_keys)}

    out_h, out_w = 64, 64
    tensor = np.empty((len(obj_keys), 18, out_h, out_w, 2), dtype=np.float32)
    tensor.fill(np.nan)

    for img_pair, c, inst, az in zip(images, cats, instance, azimuth):
        az_idx = int(az) // 2
        if not (0 <= az_idx < 18):
            continue
        o_idx = obj_index[(int(c), int(inst))]

        # (2, 96, 96) -> (96, 96, 2)
        img_hw2 = img_pair.transpose(1, 2, 0)
        img_hw2 = downsample_image_to_shape(img_hw2, (out_h, out_w))
        tensor[o_idx, az_idx] = img_hw2

    if np.isnan(tensor).any():
        missing = int(np.isnan(tensor[..., 0]).sum())
        raise RuntimeError(f"smallNORB tensor has missing entries (missing views: {missing})")

    print(f"Loaded {tensor.shape[0]} objects from smallNORB (train)")
    print(f"smallNORB tensor shape: {tensor.shape}")
    return tensor


def _load_eth80_8objs_first() -> np.ndarray:
    """Load an 8-object ETH-80 tensor for Table 5 comparability.

    We select the first object in each of the 8 categories (1..8), i.e.
    category k / object 1, giving 8 objects total. This keeps the object mode
    aligned with the COIL-100 benchmark (8 objects) while still covering all
    categories.
    """

    root = PROJECT_ROOT / "data" / "eth-80"
    if not root.exists():
        raise FileNotFoundError(f"ETH-80 dataset not found under {root}")

    frames = []
    for category in range(1, 9):
        obj_dir = root / str(category) / "1"
        if not obj_dir.exists():
            raise FileNotFoundError(f"Missing ETH-80 object dir: {obj_dir}")
        files = sorted(obj_dir.glob("*.png"))
        if not files:
            raise FileNotFoundError(f"No PNG files in {obj_dir}")

        views = []
        for f in files:
            image = np.asarray(mpimg.imread(f), dtype=np.float32)
            if image.ndim == 3 and image.shape[2] >= 3:
                image = image[..., :3]
            elif image.ndim == 2:
                image = np.stack([image, image, image], axis=-1)
            views.append(downsample_image_to_shape(image, (64, 64)))
        frames.append(np.asarray(views, dtype=np.float32))

    result = np.asarray(frames, dtype=np.float32)
    print(f"Loaded {len(frames)} objects from ETH-80 (8 objs)")
    print(f"ETH-80 (8 objs) tensor shape: {result.shape}")
    return result


def _load_cifar10() -> np.ndarray:
    path = PROJECT_ROOT / "data" / "cifar-10-batches-py" / "data_batch_1"
    with path.open("rb") as handle:
        batch = pickle.load(handle, encoding="latin1")
    data = np.asarray(batch["data"][:1000], dtype=np.float32)
    images = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0
    return images.astype(np.float32)


def _load_cbsd68() -> np.ndarray:
    root = PROJECT_ROOT / "data" / "cbsd"
    files = sorted(root.glob("*.png"))[:68]
    if not files:
        raise FileNotFoundError(f"No CBSD PNG files found under {root}")
    images = []
    for path in files:
        image = np.asarray(mpimg.imread(path), dtype=np.float32)
        images.append(downsample_image_to_shape(image[..., :3], (96, 96)))
    return np.asarray(images, dtype=np.float32)


def _normalize_max(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    scale = float(np.max(arr))
    if scale > 1e-12:
        arr = arr / scale
    return arr


def _nmse_gain_pct(tucker_nmse_db: float, ntdpl_nmse_db: float) -> float:
    tucker_nmse = 10.0 ** (float(tucker_nmse_db) / 10.0)
    ntdpl_nmse = 10.0 ** (float(ntdpl_nmse_db) / 10.0)
    return 100.0 * (tucker_nmse - ntdpl_nmse) / max(tucker_nmse, 1e-12)


def _to_latex(table: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{@{}l l c c c c c c@{}}",
        r"\toprule",
        r"\multirow{2}{*}{Domain} & \multirow{2}{*}{Dataset} & \multirow{2}{*}{Shape} & \multirow{2}{*}{Rank} & \multicolumn{3}{c}{RMSE$\downarrow$} & \multirow{2}{*}{Gain} \\",
        r"\cmidrule(lr){5-7}",
        r"& & & & Tucker & NTD-PL & $\Delta$ & \\",
        r"\midrule",
    ]
    last_domain = None
    for row in table.to_dict("records"):
        domain = str(row["domain"])
        if last_domain is not None and domain != last_domain:
            lines.append(r"\midrule")
        last_domain = domain
        gain = float(row["gain_pct"])
        tucker_rmse = float(row["tucker_rmse"])
        ntdpl_rmse = float(row["ntdpl_rmse"])
        delta_rmse = tucker_rmse - ntdpl_rmse
        gain_text = f"{gain:.1f}\\%"
        delta_text = f"{delta_rmse:.4f}"
        if gain > 0.0:
            gain_text = rf"\textbf{{{gain_text}}}"
        if delta_rmse > 0.0:
            delta_text = rf"\textbf{{{delta_text}}}"
        lines.append(
            " & ".join(
                [
                    domain,
                    str(row["dataset"]),
                    str(row["shape"]),
                    str(row["rank"]),
                    f"{tucker_rmse:.4f}",
                    f"{ntdpl_rmse:.4f}",
                    delta_text,
                    gain_text,
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
