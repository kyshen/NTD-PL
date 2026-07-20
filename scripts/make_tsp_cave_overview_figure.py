from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAVE_ROOT = PROJECT_ROOT / "data" / "CAVE"
OUT_DIR = PROJECT_ROOT / "papers" / "tsp" / "figures"

SCENE_IDS = list(range(1, 16))
LABELS = {
    "balloons": "balloons",
    "beads": "beads",
    "cd": "CD",
    "chart_and_stuffed_toy": "chart/toy",
    "clay": "clay",
    "cloth": "cloth",
    "egyptian_statue": "statue",
    "face": "face",
    "fake_and_real_beers": "beers",
    "fake_and_real_food": "food",
    "fake_and_real_lemons": "lemons",
    "fake_and_real_lemon_slices": "slices",
    "fake_and_real_peppers": "peppers",
    "fake_and_real_strawberries": "strawberries",
    "fake_and_real_sushi": "sushi",
}


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 5.2,
            "axes.linewidth": 0.45,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.01,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _scene_dirs() -> list[Path]:
    dirs = sorted(path for path in CAVE_ROOT.iterdir() if path.is_dir())
    if len(dirs) < max(SCENE_IDS):
        raise RuntimeError(f"Expected at least {max(SCENE_IDS)} CAVE scenes under {CAVE_ROOT}")
    return dirs


def _load_rgb(scene_dir: Path) -> np.ndarray:
    inner = scene_dir / scene_dir.name
    rgb_paths = sorted(inner.glob("*_RGB.bmp"))
    if not rgb_paths:
        raise RuntimeError(f"No RGB preview found under {inner}")
    image = Image.open(rgb_paths[0]).convert("RGB")
    return np.asarray(image)


def _label(scene_dir: Path) -> str:
    scene_name = scene_dir.name.replace("_ms", "")
    return LABELS.get(scene_name, scene_name.replace("_", " "))


def main() -> None:
    _set_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dirs = _scene_dirs()
    selected = [dirs[idx - 1] for idx in SCENE_IDS]

    fig, axes = plt.subplots(3, 5, figsize=(3.42, 1.94))
    for ax, scene_dir in zip(axes.ravel(), selected):
        ax.imshow(_load_rgb(scene_dir))
        ax.set_title(_label(scene_dir), pad=1.1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#333333")
            spine.set_linewidth(0.35)
    fig.subplots_adjust(
        left=0.005,
        right=0.995,
        top=0.95,
        bottom=0.01,
        wspace=0.045,
        hspace=0.22,
    )

    for ext in ("pdf", "png"):
        path = OUT_DIR / f"cave_scene_montage.{ext}"
        if ext == "png":
            fig.savefig(path, dpi=320)
        else:
            fig.savefig(path)
    plt.close(fig)
    print(f"Wrote {OUT_DIR / 'cave_scene_montage.pdf'}")


if __name__ == "__main__":
    main()
