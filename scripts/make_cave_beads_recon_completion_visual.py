from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment.process.helpers import cave_random_completion as completion
from experiment.process.helpers import cave_representation_original_space as reconstruction
from experiment.utils.plotting import apply_theme


OUT_DIR = PROJECT_ROOT / "neurips" / "figures"
SCENE_ID = 2  # CAVE beads
MISSING_RATE = 0.5


def main() -> None:
    recon_frame, _ = reconstruction._load_runs()
    recon_payload = reconstruction._run_payload(recon_frame, SCENE_ID, reconstruction.MAIN_RANK)

    completion_frame, _ = completion.load_main_runs()
    comp_payload = completion.load_scene_payload(completion_frame, scene_id=SCENE_ID, missing_rate=MISSING_RATE)

    render_figure(recon_payload, comp_payload)


def render_figure(recon_payload: object, comp_payload: object) -> None:
    bands = _rgb_bands(recon_payload.original.shape[-1])
    recon_rgb_scale = _rgb_scale(recon_payload.original, bands)
    comp_rgb_scale = _rgb_scale(comp_payload.original, bands)

    recon_observation = _rgb_from_cube(recon_payload.original, bands, recon_rgb_scale)
    recon_tucker = _rgb_from_cube(recon_payload.recon_tucker, bands, recon_rgb_scale)
    recon_ntdpl = _rgb_from_cube(recon_payload.recon_ntdpl, bands, recon_rgb_scale)
    recon_error = _rmse_map(recon_payload.original, recon_payload.recon_tucker) - _rmse_map(
        recon_payload.original, recon_payload.recon_ntdpl
    )

    comp_observation = _masked_rgb(
        _rgb_from_cube(comp_payload.original, bands, comp_rgb_scale),
        np.asarray(comp_payload.observed_mask, dtype=bool),
        bands,
    )
    comp_tucker = _rgb_from_cube(comp_payload.recon_tucker, bands, comp_rgb_scale)
    comp_ntdpl = _rgb_from_cube(comp_payload.recon_ntdpl, bands, comp_rgb_scale)
    missing_mask = ~np.asarray(comp_payload.observed_mask, dtype=bool)
    comp_error = _rmse_map(comp_payload.original, comp_payload.recon_tucker, missing_mask) - _rmse_map(
        comp_payload.original, comp_payload.recon_ntdpl, missing_mask
    )

    diff_values = np.concatenate([np.ravel(recon_error), np.ravel(comp_error[np.isfinite(comp_error)])])
    diff_vmax = max(float(np.nanquantile(np.abs(diff_values), 0.99)), 1e-6)

    apply_theme()
    plt.rcParams.update(
        {
            "font.size": 7.6,
            "axes.titlesize": 8.1,
            "axes.labelsize": 7.6,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 4, figsize=(7.6, 3.15), constrained_layout=False)
    col_titles = ("Observation", "Tucker", "NTD-PL", "Error")
    rows = (
        ("Reconstruction", recon_observation, recon_tucker, recon_ntdpl, recon_error),
        ("Completion", comp_observation, comp_tucker, comp_ntdpl, comp_error),
    )

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#F2F4F7")
    error_image = None
    for row_idx, (row_label, observation, tucker, ntdpl, error_map) in enumerate(rows):
        panels = (observation, tucker, ntdpl, error_map)
        for col_idx, image in enumerate(panels):
            ax = axes[row_idx, col_idx]
            image = _display_downsample(np.asarray(image))
            if col_idx < 3:
                ax.imshow(np.clip(image, 0.0, 1.0))
            else:
                error_image = ax.imshow(image, cmap=cmap, vmin=-diff_vmax, vmax=diff_vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], loc="left", pad=4, fontweight="bold")
            if col_idx == 0:
                ax.text(
                    -0.06,
                    0.5,
                    row_label,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=8.2,
                    fontweight="bold",
                )

    if error_image is not None:
        cax = fig.add_axes([0.91, 0.21, 0.014, 0.56])
        cbar = fig.colorbar(error_image, cax=cax)
        cbar.ax.set_title("Error", fontsize=7.0, pad=3)
        cbar.ax.tick_params(labelsize=6.5, length=2)
        cbar.set_label("Tucker - NTD-PL RMSE", fontsize=7.0, labelpad=4)

    fig.subplots_adjust(left=0.075, right=0.89, top=0.925, bottom=0.055, wspace=0.032, hspace=0.045)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"cave_beads_recon_completion_visual.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _rgb_bands(num_bands: int) -> list[int]:
    return [int(round((num_bands - 1) * value)) for value in (0.75, 0.50, 0.20)]


def _rgb_scale(cube: np.ndarray, bands: list[int]) -> tuple[float, float]:
    rgb = np.stack([cube[..., idx] for idx in bands], axis=-1).astype(np.float32)
    lo, hi = np.quantile(rgb, [0.01, 0.995])
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def _rgb_from_cube(cube: np.ndarray, bands: list[int], scale: tuple[float, float]) -> np.ndarray:
    lo, hi = scale
    rgb = np.stack([cube[..., idx] for idx in bands], axis=-1).astype(np.float32)
    rgb = (np.clip(rgb, 0.0, None) - lo) / (hi - lo)
    return np.clip(rgb, 0.0, 1.0)


def _masked_rgb(rgb: np.ndarray, observed_mask: np.ndarray, bands: list[int]) -> np.ndarray:
    channel_mask = observed_mask[..., bands]
    missing_gray = np.array([0.94, 0.94, 0.94], dtype=np.float32)
    return np.where(channel_mask, rgb, missing_gray.reshape(1, 1, 3))


def _display_downsample(image: np.ndarray, max_side: int = 256) -> np.ndarray:
    height, width = image.shape[:2]
    stride = max(1, int(np.ceil(max(height, width) / max_side)))
    return image[::stride, ::stride]


def _rmse_map(original: np.ndarray, reconstruction_: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    diff2 = (np.asarray(original, dtype=np.float32) - np.asarray(reconstruction_, dtype=np.float32)) ** 2
    if mask is None:
        return np.sqrt(np.mean(diff2, axis=-1))
    mask = np.asarray(mask, dtype=bool)
    count = mask.sum(axis=-1)
    total = np.sum(np.where(mask, diff2, 0.0), axis=-1)
    out = np.full(count.shape, np.nan, dtype=np.float32)
    valid = count > 0
    out[valid] = np.sqrt(total[valid] / count[valid])
    return out


if __name__ == "__main__":
    main()
