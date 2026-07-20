from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
from scipy.ndimage import uniform_filter

from run_cave_rank_inflation_spectrum import (
    PROJECT_ROOT,
    _fit_ntdpl_direct,
    _fit_tucker_direct,
    _load_cave_scene,
)


DEFAULT_OUTDIR = PROJECT_ROOT / "artifacts" / "results" / "cave_response_pathway_figure"


def _shared_rgb_limits(measured: np.ndarray, bands: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    rgb = measured[:, :, list(bands)].astype(np.float64)
    lo = np.percentile(rgb, 1.0, axis=(0, 1))
    hi = np.percentile(rgb, 99.5, axis=(0, 1))
    return lo, np.maximum(hi, lo + 1e-8)


def _render_rgb(
    cube: np.ndarray,
    bands: tuple[int, int, int],
    lo: np.ndarray,
    hi: np.ndarray,
) -> np.ndarray:
    rgb = cube[:, :, list(bands)].astype(np.float64)
    rgb = np.clip((rgb - lo) / (hi - lo), 0.0, 1.0)
    return np.power(rgb, 1.0 / 2.2)


def _render_latent_rgb(signal: np.ndarray, bands: tuple[int, int, int]) -> np.ndarray:
    rgb = signal[:, :, list(bands)].astype(np.float64)
    lo = np.percentile(rgb, 1.0, axis=(0, 1))
    hi = np.percentile(rgb, 99.0, axis=(0, 1))
    rgb = np.clip((rgb - lo) / np.maximum(hi - lo, 1e-8), 0.0, 1.0)
    return np.power(rgb, 1.0 / 2.2)


def _spectral_rmse(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((reference.astype(np.float64) - prediction.astype(np.float64)) ** 2, axis=2))


def _select_crop(improvement: np.ndarray, crop_size: int) -> tuple[int, int, int, int]:
    height, width = improvement.shape
    score = uniform_filter(improvement.astype(np.float64), size=max(9, crop_size // 4), mode="nearest")
    margin = crop_size // 2
    masked = score.copy()
    masked[:margin, :] = -np.inf
    masked[-margin:, :] = -np.inf
    masked[:, :margin] = -np.inf
    masked[:, -margin:] = -np.inf
    cy, cx = np.unravel_index(np.argmax(masked), masked.shape)
    y0 = int(np.clip(cy - crop_size // 2, 0, height - crop_size))
    x0 = int(np.clip(cx - crop_size // 2, 0, width - crop_size))
    return y0, y0 + crop_size, x0, x0 + crop_size


def _fit_or_load(
    cache_path: Path,
    scene_id: int,
    rank: tuple[int, int, int],
    p_max: int,
    n_iter_max: int,
) -> dict[str, np.ndarray | str]:
    if cache_path.exists():
        loaded = np.load(cache_path, allow_pickle=False)
        return {key: loaded[key] for key in loaded.files}
    scene_name, measured = _load_cave_scene(scene_id, (512, 512))
    tucker, _ = _fit_tucker_direct(measured, rank, n_iter_max)
    signal, prediction, beta, _ = _fit_ntdpl_direct(measured, rank, n_iter_max, p_max)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        scene_name=np.asarray(scene_name),
        measured=measured,
        tucker=tucker,
        signal=signal,
        prediction=prediction,
        beta=beta,
    )
    return {
        "scene_name": np.asarray(scene_name),
        "measured": measured,
        "tucker": tucker,
        "signal": signal,
        "prediction": prediction,
        "beta": beta,
    }


def _poly_value(values: np.ndarray, beta: np.ndarray) -> np.ndarray:
    out = np.full_like(values, float(beta[-1]), dtype=np.float64)
    for q in range(len(beta) - 2, -1, -1):
        out = out * values + float(beta[q])
    return out


def _gain_text(gain: float, *, include_sign: bool = False) -> str:
    if include_sign:
        return f"-{gain:.1f}%" if gain >= 0.0 else f"+{abs(gain):.1f}%"
    direction = "lower" if gain >= 0.0 else "higher"
    return f"{abs(gain):.1f}% {direction}"


def _add_process_arrows(fig: plt.Figure, axes: list[plt.Axes]) -> None:
    for left, right in zip(axes[:2], axes[1:3]):
        box_l = left.get_position()
        box_r = right.get_position()
        y = 0.5 * (box_l.y0 + box_l.y1)
        arrow = FancyArrowPatch(
            (box_l.x1 + 0.006, y),
            (box_r.x0 - 0.006, y),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.0,
            color="#444444",
        )
        fig.add_artist(arrow)
    box_l = axes[2].get_position()
    box_r = axes[3].get_position()
    fig.text(
        0.5 * (box_l.x1 + box_r.x0),
        0.5 * (box_l.y0 + box_l.y1),
        r"$\approx$",
        ha="center",
        va="center",
        fontsize=13,
        color="#444444",
    )


def make_figure(data: dict[str, np.ndarray | str], outdir: Path, crop_size: int) -> None:
    measured = np.asarray(data["measured"], dtype=np.float32)
    tucker = np.asarray(data["tucker"], dtype=np.float32)
    signal = np.asarray(data["signal"], dtype=np.float32)
    prediction = np.asarray(data["prediction"], dtype=np.float32)
    beta = np.asarray(data["beta"], dtype=np.float64).reshape(-1)
    bands = (25, 15, 5)
    lo, hi = _shared_rgb_limits(measured, bands)
    measured_rgb = _render_rgb(measured, bands, lo, hi)
    tucker_rgb = _render_rgb(tucker, bands, lo, hi)
    prediction_rgb = _render_rgb(prediction, bands, lo, hi)
    signal_rgb = _render_latent_rgb(signal, bands)

    err_tucker = _spectral_rmse(measured, tucker)
    err_ntdpl = _spectral_rmse(measured, prediction)
    improvement = err_tucker - err_ntdpl
    y0, y1, x0, x1 = _select_crop(improvement, crop_size)

    rmse_tucker = float(np.sqrt(np.mean((measured - tucker) ** 2)))
    rmse_ntdpl = float(np.sqrt(np.mean((measured - prediction) ** 2)))
    gain = 100.0 * (rmse_tucker - rmse_ntdpl) / rmse_tucker
    crop = np.s_[y0:y1, x0:x1]
    crop_rmse_tucker = float(np.sqrt(np.mean((measured[crop] - tucker[crop]) ** 2)))
    crop_rmse_ntdpl = float(np.sqrt(np.mean((measured[crop] - prediction[crop]) ** 2)))
    local_gain = 100.0 * (crop_rmse_tucker - crop_rmse_ntdpl) / crop_rmse_tucker

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
    fig = plt.figure(figsize=(7.16, 3.78))
    grid = fig.add_gridspec(
        2,
        4,
        height_ratios=[1.0, 0.82],
        width_ratios=[1.0, 0.94, 1.0, 1.0],
        hspace=0.38,
        wspace=0.18,
    )
    axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(4)]

    axes[0].imshow(signal_rgb)
    axes[0].set_title(r"(a) latent Tucker signal $\widehat{S}$", pad=3)
    axes[0].axis("off")

    flat_signal = signal.reshape(-1).astype(np.float64)
    grid_s = np.quantile(flat_signal, np.linspace(0.01, 0.99, 201))
    response = _poly_value(grid_s, beta)
    affine = float(beta[0]) + float(beta[1]) * grid_s
    axes[1].plot(grid_s, response, color="#2F6FBB", linewidth=1.7, label=r"$f_{\widehat\beta}(s)$")
    axes[1].plot(grid_s, affine, color="#777777", linestyle="--", linewidth=1.0, label="affine part")
    axes[1].fill_between(grid_s, affine, response, color="#D9822B", alpha=0.22, label="nonlinear part")
    axes[1].set_title("(b) learned response", pad=3)
    axes[1].set_xlabel(r"latent value $s$", labelpad=0)
    axes[1].grid(True, alpha=0.22)
    axes[1].legend(loc="upper left", fontsize=5.7, borderpad=0.35, handlelength=2.0, labelspacing=0.25)

    axes[2].imshow(prediction_rgb)
    axes[2].set_title(r"(c) response output $\widehat{Y}$", pad=3)
    axes[2].axis("off")

    axes[3].imshow(measured_rgb)
    axes[3].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#F2C14E", linewidth=2.0))
    axes[3].set_title(r"(d) measured image $Y$", pad=3)
    axes[3].axis("off")

    axes[4].imshow(measured_rgb[crop])
    axes[4].set_title("(e) measured detail", pad=3)
    axes[4].axis("off")

    axes[5].imshow(tucker_rgb[crop])
    axes[5].set_title("(f) Tucker detail", pad=3)
    axes[5].text(
        0.03,
        0.04,
        rf"RMSE {crop_rmse_tucker:.4f}",
        transform=axes[5].transAxes,
        color="white",
        fontsize=6.4,
        bbox={"facecolor": "black", "alpha": 0.62, "edgecolor": "none", "pad": 1.7},
    )
    axes[5].axis("off")

    axes[6].imshow(prediction_rgb[crop])
    axes[6].set_title("(g) NTD-PL detail", pad=3)
    axes[6].text(
        0.03,
        0.04,
        rf"RMSE {crop_rmse_ntdpl:.4f}  ({_gain_text(local_gain, include_sign=True)})",
        transform=axes[6].transAxes,
        color="white",
        fontsize=6.2,
        bbox={"facecolor": "black", "alpha": 0.62, "edgecolor": "none", "pad": 1.7},
    )
    axes[6].axis("off")

    crop_improvement = improvement[crop]
    vmax = max(float(np.percentile(np.abs(crop_improvement), 98.0)), 1e-8)
    image = axes[7].imshow(crop_improvement, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[7].set_title("(h) per-pixel RMSE reduction", pad=3)
    axes[7].text(
        0.03,
        0.94,
        rf"{_gain_text(gain)} overall",
        transform=axes[7].transAxes,
        va="top",
        color="#222222",
        fontsize=6.2,
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 1.5},
    )
    axes[7].axis("off")
    colorbar = fig.colorbar(image, ax=axes[7], fraction=0.046, pad=0.02)
    colorbar.ax.tick_params(labelsize=5.8, length=2)
    colorbar.set_label("Tucker error - NTD-PL error", fontsize=6.2, labelpad=2)

    fig.subplots_adjust(top=0.95, left=0.018, right=0.985, bottom=0.055)
    _add_process_arrows(fig, axes[:4])
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "cave_response_pathway.pdf", bbox_inches="tight")
    fig.savefig(outdir / "cave_response_pathway.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a real-image visualization of the NTD-PL response pathway on CAVE.")
    parser.add_argument("--scene-id", type=int, default=3)
    parser.add_argument("--rank", default="96,96,8")
    parser.add_argument("--p-max", type=int, default=6)
    parser.add_argument("--n-iter-max", type=int, default=120)
    parser.add_argument("--crop-size", type=int, default=144)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()
    rank_values = tuple(int(part.strip()) for part in args.rank.split(","))
    if len(rank_values) != 3:
        raise ValueError("Expected three Tucker rank entries.")
    rank = (rank_values[0], rank_values[1], rank_values[2])
    outdir = args.outdir if args.outdir.is_absolute() else PROJECT_ROOT / args.outdir
    cache = outdir / f"scene{args.scene_id:02d}_r{rank[0]}_p{args.p_max}_state.npz"
    data = _fit_or_load(cache, args.scene_id, rank, int(args.p_max), int(args.n_iter_max))
    make_figure(data, outdir, int(args.crop_size))
    print(f"Output: {outdir}")


if __name__ == "__main__":
    main()
