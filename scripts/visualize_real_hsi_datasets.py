from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.hsi import _load_hsi_from_file


DATASETS = [
    ("Jasper Ridge", Path("data/hsi/jasperRidge2_R198.mat")),
    ("Samson", Path("data/hsi-similar/samson_1.img")),
    ("Urban", Path("data/hsi-similar/Urban_R162.mat")),
    ("Cuprite", Path("data/hsi-similar/Cuprite_S1_R188.img")),
]


def _normalize_cube(cube: np.ndarray) -> np.ndarray:
    cube = np.asarray(cube, dtype=np.float32)
    cube = cube - np.nanmin(cube)
    scale = float(np.nanmax(cube))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Cube has invalid dynamic range")
    return cube / scale


def _pick_rgb_bands(num_bands: int) -> tuple[int, int, int]:
    # Use broad spectral positions to form a stable pseudo-RGB composite.
    band_positions = np.array([0.65, 0.35, 0.15], dtype=np.float32)
    indices = np.clip(np.round((num_bands - 1) * band_positions).astype(int), 0, num_bands - 1)
    return int(indices[0]), int(indices[1]), int(indices[2])


def _make_rgb(cube: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int]]:
    bands = _pick_rgb_bands(cube.shape[2])
    rgb = cube[..., list(bands)]
    lo = np.percentile(rgb, 2.0, axis=(0, 1), keepdims=True)
    hi = np.percentile(rgb, 98.0, axis=(0, 1), keepdims=True)
    rgb = np.clip((rgb - lo) / np.maximum(hi - lo, 1e-6), 0.0, 1.0)
    return rgb, bands


def _sample_pixel_spectra(cube: np.ndarray) -> np.ndarray:
    h, w, _ = cube.shape
    ys = np.linspace(0, h - 1, 5, dtype=int)
    xs = np.linspace(0, w - 1, 5, dtype=int)
    spectra = [cube[y, x, :] for y in ys for x in xs]
    return np.asarray(spectra, dtype=np.float32)


def main() -> None:
    out_dir = PROJECT_ROOT / "outputs" / "real_hsi_visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(DATASETS), 2, figsize=(12, 16), constrained_layout=True)
    fig.suptitle("Real HSI Dataset Visualizations", fontsize=18, fontweight="bold")

    for row, (name, rel_path) in enumerate(DATASETS):
        cube = _normalize_cube(_load_hsi_from_file(PROJECT_ROOT / rel_path))
        rgb, bands = _make_rgb(cube)
        ax_img = axes[row, 0]
        ax_plot = axes[row, 1]

        ax_img.imshow(rgb)
        ax_img.set_title(f"{name} pseudo-RGB\nshape={cube.shape}, bands={bands}")
        ax_img.axis("off")

        mean_spectrum = cube.mean(axis=(0, 1))
        sample_spectra = _sample_pixel_spectra(cube)
        for spectrum in sample_spectra:
            ax_plot.plot(spectrum, color="#9ecae1", alpha=0.35, linewidth=0.9)
        ax_plot.plot(mean_spectrum, color="#08519c", linewidth=2.2, label="mean spectrum")
        ax_plot.set_title(f"{name} spectral overview")
        ax_plot.set_xlabel("Band index")
        ax_plot.set_ylabel("Normalized intensity")
        ax_plot.set_ylim(0.0, 1.05)
        ax_plot.grid(alpha=0.25, linewidth=0.6)
        ax_plot.legend(loc="upper right")

        single_fig, single_ax = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
        single_ax[0].imshow(rgb)
        single_ax[0].set_title(f"{name} pseudo-RGB")
        single_ax[0].axis("off")
        for spectrum in sample_spectra:
            single_ax[1].plot(spectrum, color="#9ecae1", alpha=0.35, linewidth=0.9)
        single_ax[1].plot(mean_spectrum, color="#08519c", linewidth=2.2)
        single_ax[1].set_title(f"{name} spectral overview")
        single_ax[1].set_xlabel("Band index")
        single_ax[1].set_ylabel("Normalized intensity")
        single_ax[1].set_ylim(0.0, 1.05)
        single_ax[1].grid(alpha=0.25, linewidth=0.6)
        single_path = out_dir / f"{name.lower().replace(' ', '_')}.png"
        single_fig.savefig(single_path, dpi=180, bbox_inches="tight")
        plt.close(single_fig)

    combined_path = out_dir / "real_hsi_datasets_overview.png"
    fig.savefig(combined_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved combined figure to: {combined_path}")
    for name, _ in DATASETS:
        print(out_dir / f"{name.lower().replace(' ', '_')}.png")


if __name__ == "__main__":
    main()
