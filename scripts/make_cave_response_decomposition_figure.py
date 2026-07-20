from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from make_cave_response_pathway_figure import (
    DEFAULT_OUTDIR,
    _fit_or_load,
    _poly_value,
    _select_crop,
    _spectral_rmse,
)


def _rgb_limits(measured: np.ndarray, bands: tuple[int, int, int]) -> np.ndarray:
    rgb = measured[:, :, list(bands)].astype(np.float64)
    return np.maximum(np.percentile(rgb, 99.5, axis=(0, 1)), 1e-8)


def _render_shared_rgb(cube: np.ndarray, bands: tuple[int, int, int], hi: np.ndarray) -> np.ndarray:
    rgb = cube[:, :, list(bands)].astype(np.float64)
    rgb = np.clip(rgb / hi, 0.0, 1.0)
    return np.power(rgb, 1.0 / 2.2)


def _add_equation_symbols(fig: plt.Figure, axes: list[plt.Axes]) -> None:
    for symbol, left, right in zip(("+", "=", r"$\approx$"), axes[:-1], axes[1:]):
        box_l = left.get_position()
        box_r = right.get_position()
        x = 0.5 * (box_l.x1 + box_r.x0)
        y = 0.5 * (box_l.y0 + box_l.y1)
        fig.text(x, y, symbol, ha="center", va="center", fontsize=11.0, color="#333333", weight="bold")


def _image_note(ax: plt.Axes, text: str, *, light: bool = False) -> None:
    ax.text(
        0.03,
        0.04,
        text,
        transform=ax.transAxes,
        color="#222222" if light else "white",
        fontsize=6.1,
        bbox={
            "facecolor": "white" if light else "black",
            "alpha": 0.76 if light else 0.62,
            "edgecolor": "none",
            "pad": 1.6,
        },
    )


def make_figure(data: dict[str, np.ndarray | str], outdir: Path, crop_size: int) -> None:
    measured = np.asarray(data["measured"], dtype=np.float32)
    signal = np.asarray(data["signal"], dtype=np.float32)
    prediction = np.asarray(data["prediction"], dtype=np.float32)
    beta = np.asarray(data["beta"], dtype=np.float64).reshape(-1)

    affine = float(beta[0]) + float(beta[1]) * signal
    nonlinear = prediction - affine

    bands = (25, 15, 5)
    hi = _rgb_limits(measured, bands)
    affine_rgb = _render_shared_rgb(affine, bands, hi)
    nonlinear_rgb = _render_shared_rgb(nonlinear, bands, hi)
    prediction_rgb = _render_shared_rgb(prediction, bands, hi)
    measured_rgb = _render_shared_rgb(measured, bands, hi)

    err_affine = _spectral_rmse(measured, affine)
    err_full = _spectral_rmse(measured, prediction)
    y0, y1, x0, x1 = _select_crop(err_affine - err_full, crop_size)
    crop = np.s_[y0:y1, x0:x1]

    target_rms = float(np.sqrt(np.mean(measured.astype(np.float64) ** 2)))
    affine_share = 100.0 * float(np.sqrt(np.mean(affine.astype(np.float64) ** 2))) / target_rms
    nonlinear_share = 100.0 * float(np.sqrt(np.mean(nonlinear.astype(np.float64) ** 2))) / target_rms
    rmse_full = float(np.sqrt(np.mean((measured - prediction) ** 2)))
    crop_rmse_affine = float(np.sqrt(np.mean((measured[crop] - affine[crop]) ** 2)))
    crop_rmse_full = float(np.sqrt(np.mean((measured[crop] - prediction[crop]) ** 2)))
    crop_reduction = 100.0 * (crop_rmse_affine - crop_rmse_full) / crop_rmse_affine

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(7.16, 3.55))
    grid = fig.add_gridspec(
        2,
        4,
        height_ratios=[1.0, 0.82],
        width_ratios=[1.0, 0.96, 1.0, 1.0],
        hspace=0.24,
        wspace=0.20,
    )
    axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(4)]

    top_images = (affine_rgb, nonlinear_rgb, prediction_rgb, measured_rgb)
    top_titles = (
        r"(a) affine terms $p=0,1$",
        r"(b) nonlinear terms $p\geq2$",
        r"(c) full prediction $\widehat{Y}$",
        r"(d) measured image $Y$",
    )
    for ax, image, title in zip(axes[:4], top_images, top_titles):
        ax.imshow(image)
        ax.set_title(title, pad=3)
        ax.axis("off")
    axes[3].add_patch(
        Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#F2C14E", linewidth=1.6)
    )
    _image_note(axes[0], rf"RMS(A) / RMS(Y) = {affine_share:.1f}%")
    _image_note(axes[1], rf"RMS(Rnl) / RMS(Y) = {nonlinear_share:.1f}%")
    _image_note(axes[2], rf"RMSE {rmse_full:.4f}")

    flat_signal = signal.reshape(-1).astype(np.float64)
    grid_s = np.quantile(flat_signal, np.linspace(0.01, 0.99, 201))
    response = _poly_value(grid_s, beta)
    affine_curve = float(beta[0]) + float(beta[1]) * grid_s
    nonlinear_curve = response - affine_curve
    axes[4].plot(grid_s, response, color="#2F6FBB", linewidth=1.7, label="sum")
    axes[4].plot(grid_s, affine_curve, color="#777777", linestyle="--", linewidth=1.0, label="affine")
    axes[4].plot(grid_s, nonlinear_curve, color="#D9822B", linewidth=1.25, label="nonlinear")
    axes[4].set_title("(e) response decomposition", pad=3)
    axes[4].set_xlabel(r"latent value $s$", labelpad=1)
    axes[4].set_ylabel("component value", labelpad=1)
    axes[4].grid(True, alpha=0.22)
    axes[4].legend(loc="upper left", fontsize=5.7, borderpad=0.35, handlelength=1.8, labelspacing=0.25)

    axes[5].imshow(affine_rgb[crop])
    axes[5].set_title("(f) affine detail", pad=3)
    _image_note(axes[5], rf"RMSE {crop_rmse_affine:.4f}")
    axes[5].axis("off")

    axes[6].imshow(prediction_rgb[crop])
    axes[6].set_title("(g) after nonlinear terms", pad=3)
    _image_note(axes[6], rf"RMSE {crop_rmse_full:.4f}  ({crop_reduction:.1f}% lower)")
    axes[6].axis("off")

    axes[7].imshow(measured_rgb[crop])
    axes[7].set_title("(h) measured detail", pad=3)
    axes[7].axis("off")

    fig.subplots_adjust(top=0.95, left=0.02, right=0.985, bottom=0.055)
    _add_equation_symbols(fig, axes[:4])
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "cave_response_decomposition.pdf", bbox_inches="tight")
    fig.savefig(outdir / "cave_response_decomposition.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize affine and nonlinear parts of an NTD-PL fit on CAVE.")
    parser.add_argument("--scene-id", type=int, default=10)
    parser.add_argument("--rank", default="24,24,4")
    parser.add_argument("--p-max", type=int, default=6)
    parser.add_argument("--n-iter-max", type=int, default=120)
    parser.add_argument("--crop-size", type=int, default=144)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    rank_values = tuple(int(part.strip()) for part in args.rank.split(","))
    if len(rank_values) != 3:
        raise ValueError("Expected three Tucker rank entries.")
    rank = (rank_values[0], rank_values[1], rank_values[2])
    outdir = args.outdir if args.outdir.is_absolute() else DEFAULT_OUTDIR.parent.parent.parent / args.outdir
    cache = outdir / f"scene{args.scene_id:02d}_r{rank[0]}_p{args.p_max}_state.npz"
    data = _fit_or_load(cache, args.scene_id, rank, int(args.p_max), int(args.n_iter_max))
    make_figure(data, outdir, int(args.crop_size))
    print(f"Output: {outdir}")


if __name__ == "__main__":
    main()
