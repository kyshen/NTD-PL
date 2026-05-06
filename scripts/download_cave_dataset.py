from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAVE_URL = "https://cave.cs.columbia.edu/old/databases/multispectral/zip/complete_ms_data.zip"
ZIP_PATH = PROJECT_ROOT / "data/downloads/complete_ms_data.zip"
TARGET_DIR = PROJECT_ROOT / "data/CAVE"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path, *, force: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        print(f"Skip {destination}: already exists (sha256={_sha256(destination)}).")
        return

    tmp_path = destination.with_suffix(destination.suffix + ".part")
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl is required for this large download.")
    command = [
        curl,
        "-L",
        "--retry",
        "5",
        "--continue-at",
        "-",
        "-o",
        str(tmp_path),
        url,
    ]
    subprocess.run(command, check=True)
    tmp_path.replace(destination)
    print(f"Saved {destination} (sha256={_sha256(destination)})")


def _extract(zip_path: Path, target_dir: Path, *, force: bool) -> None:
    if target_dir.exists() and not force and _looks_like_cave(target_dir):
        print(f"Skip extraction: {target_dir} already looks complete.")
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)
    print(f"Extracted {zip_path} -> {target_dir}")


def _looks_like_cave(path: Path) -> bool:
    if not path.exists():
        return False
    scene_dirs = sorted(p for p in path.iterdir() if p.is_dir())
    if len(scene_dirs) < 15:
        return False
    first = scene_dirs[0]
    inner_dirs = [p for p in first.iterdir() if p.is_dir()]
    if not inner_dirs:
        return False
    band_pngs = [
        p for p in inner_dirs[0].glob("*.png")
        if "_RGB" not in p.name and "Thumbs" not in p.name
    ]
    return len(band_pngs) >= 31


def _validate() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.data.hsi import CAVEHSIData
    from src.filters.bias import BiasFilter

    dataset = CAVEHSIData(path="data/CAVE", id=1, target_shape=(512, 512), crop_shape=None)
    BiasFilter(seed=0, normalize_method="max", snr_db=None, bias=None)(dataset)
    tensor = dataset.get(split="eval")
    print(f"Validated CAVE scene 1: {getattr(dataset, 'scene_name', 'unknown')} shape={tensor.dense.shape}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and extract the Columbia CAVE multispectral dataset.")
    parser.add_argument("--force", action="store_true", help="Re-download and re-extract even if files exist.")
    parser.add_argument("--no-validate", action="store_true", help="Skip CAVEHSIData validation after extraction.")
    args = parser.parse_args()

    _download(CAVE_URL, ZIP_PATH, force=args.force)
    _extract(ZIP_PATH, TARGET_DIR, force=args.force)
    if not args.no_validate:
        _validate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
