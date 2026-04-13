from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen


DATASETS = {
    "samson": {
        "description": "Samson scene mirrored from the public hyperspectral benchmark collection.",
        "files": [
            {
                "url": "https://hf-mirror.com/datasets/danaroth/samson/resolve/main/samson_1.img",
                "path": Path("data/hsi-similar/samson_1.img"),
            },
            {
                "url": "https://hf-mirror.com/datasets/danaroth/samson/resolve/main/samson_1.img.hdr",
                "path": Path("data/hsi-similar/samson_1.img.hdr"),
            },
        ],
    },
    "urban": {
        "description": "Urban scene mirrored from the public hyperspectral benchmark collection.",
        "files": [
            {
                "url": "https://hf-mirror.com/datasets/danaroth/urban/resolve/main/Urban_R162.mat",
                "path": Path("data/hsi-similar/Urban_R162.mat"),
            },
        ],
    },
    "cuprite": {
        "description": "Cuprite scene mirrored from the public hyperspectral benchmark collection.",
        "files": [
            {
                "url": "https://hf-mirror.com/datasets/danaroth/cuprite/resolve/main/Cuprite_S1_R188.img",
                "path": Path("data/hsi-similar/Cuprite_S1_R188.img"),
            },
            {
                "url": "https://hf-mirror.com/datasets/danaroth/cuprite/resolve/main/Cuprite_S1_R188.hdr",
                "path": Path("data/hsi-similar/Cuprite_S1_R188.hdr"),
            },
        ],
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    curl_exe = shutil.which("curl.exe") or shutil.which("curl")
    if curl_exe is not None:
        command = [
            curl_exe,
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
        return

    with urlopen(url) as response, tmp_path.open("wb") as handle:
        total = response.headers.get("Content-Length")
        total_bytes = int(total) if total is not None else None
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if total_bytes:
                percent = 100.0 * downloaded / total_bytes
                print(f"\r{destination.name}: {percent:5.1f}% ({downloaded}/{total_bytes} bytes)", end="", flush=True)
            else:
                print(f"\r{destination.name}: {downloaded} bytes", end="", flush=True)
    print()
    tmp_path.replace(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the real HSI robustness benchmark datasets.")
    parser.add_argument(
        "--dataset",
        choices=[*DATASETS.keys(), "all"],
        default="all",
        help="Dataset to download. Defaults to all Jasper-like robustness-study datasets.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the target file already exists.",
    )
    args = parser.parse_args()

    selected = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    for dataset_name in selected:
        spec = DATASETS[dataset_name]
        print(f"Preparing {dataset_name}: {spec['description']}")
        for file_spec in spec["files"]:
            destination = Path(file_spec["path"])
            if destination.exists() and not args.force:
                print(f"Skip {destination}: already exists (sha256={_sha256(destination)}).")
                continue

            print(f"Downloading {file_spec['url']} -> {destination}")
            _download(str(file_spec["url"]), destination)
            print(f"Saved {destination} (sha256={_sha256(destination)})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
