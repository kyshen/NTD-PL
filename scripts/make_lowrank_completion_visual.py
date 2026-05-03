from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment.utils.io import load_state_mat
from src.data.hsi import CAVEHSIData
from src.filters.bias import BiasFilter


OUT_DIR = PROJECT_ROOT / "neurips" / "figures"
SCENE_ID = 2


def main() -> None:
    for protocol in ("random", "block"):
        _render_case(protocol=protocol)


def _render_case(*, protocol: str) -> None:
    case = _load_completion_case(protocol=protocol)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 8.6,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 5, figsize=(7.35, 1.65), constrained_layout=True)
    col_titles = ("Original", "Observed", "Tucker", "NTD-PL", "Error reduction")
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#F1F3F5")

    images = (
        case["original_rgb"],
        case["observed_rgb"],
        case["tucker_rgb"],
        case["ntdpl_rgb"],
        case["reduction"],
    )
    reduction = np.asarray(case["reduction"], dtype=np.float32)
    red_vmax = max(float(np.nanquantile(np.abs(reduction), 0.98)), 1e-6)
    for col_idx, (ax, title, image) in enumerate(zip(axes, col_titles, images, strict=True)):
        ax.set_title(title, loc="left", pad=3, fontweight="bold")
        if col_idx < 4:
            ax.imshow(np.clip(image, 0.0, 1.0))
        else:
            ax.imshow(image, cmap=cmap, vmin=-red_vmax, vmax=red_vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    _metric_box(axes[2], case["tucker_metrics"])
    _metric_box(axes[3], case["ntdpl_metrics"])
    axes[4].text(
        0.03,
        0.97,
        "red = NTD-PL lower error",
        transform=axes[4].transAxes,
        va="top",
        ha="left",
        fontsize=6.7,
        bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "#D7DCE2", "alpha": 0.92},
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = "lowrank_completion_visual" if protocol == "random" else f"lowrank_completion_visual_{protocol}"
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_DIR / (stem + '.pdf')} and .png")


def _load_completion_case(*, protocol: str) -> dict[str, object]:
    runs = pd.read_csv(PROJECT_ROOT / "neurips" / "tables" / "lowrank_core_cave_rates.per_run.csv")
    case = {
        "item_id": SCENE_ID,
        "protocol": protocol,
        "seed": 0,
        "missing_rate": 0.5,
        "rank": "(24,24,4)",
    }
    tucker_row = _select_completion(runs, method="tucker", **case)
    ntdpl_row = _select_completion(runs, method="ntdpl", **case)

    original = _load_original(scene_id=SCENE_ID, target_shape=(128, 128))
    tucker_state = load_state_mat(Path(tucker_row["run_dir"]) / "state.mat")
    ntdpl_state = load_state_mat(Path(ntdpl_row["run_dir"]) / "state.mat")
    recon_tucker = np.asarray(tucker_state["reconstruction"], dtype=np.float32)
    recon_ntdpl = np.asarray(ntdpl_state["reconstruction"], dtype=np.float32)
    observed_mask = np.asarray(tucker_state["observed_mask"], dtype=bool)

    original_rgb, rgb_bands = _pseudo_rgb(original)
    observed_rgb = _masked_rgb(original_rgb, observed_mask, rgb_bands)
    missing_mask = ~observed_mask
    reduction = _rmse_map(original, recon_tucker, missing_mask) - _rmse_map(original, recon_ntdpl, missing_mask)

    return {
        "original_rgb": original_rgb,
        "observed_rgb": observed_rgb,
        "tucker_rgb": _pseudo_rgb(recon_tucker)[0],
        "ntdpl_rgb": _pseudo_rgb(recon_ntdpl)[0],
        "reduction": reduction,
        "tucker_metrics": (float(tucker_row["RMSE_missing"]), float(tucker_row["SAM_missing"])),
        "ntdpl_metrics": (float(ntdpl_row["RMSE_missing"]), float(ntdpl_row["SAM_missing"])),
    }


def _select_completion(runs: pd.DataFrame, *, method: str, **case: object) -> pd.Series:
    panel = runs.loc[
        runs["item_id"].eq(case["item_id"])
        & runs["protocol"].eq(case["protocol"])
        & runs["seed"].eq(case["seed"])
        & np.isclose(runs["missing_rate"], float(case["missing_rate"]), atol=1e-12)
        & runs["rank"].eq(case["rank"])
        & runs["method"].eq(method)
    ].copy()
    if panel.empty:
        raise ValueError(f"No completion run found for {case}, method={method}.")
    return panel.iloc[0]


def _load_original(*, scene_id: int, target_shape: tuple[int, int]) -> np.ndarray:
    dataset = CAVEHSIData(path="data/CAVE", id=scene_id, target_shape=target_shape, crop_shape=None)
    BiasFilter(seed=0, normalize_method="max", snr_db=None, bias=None)(dataset)
    return np.asarray(dataset.get(split="eval").dense, dtype=np.float32)


def _pseudo_rgb(cube: np.ndarray) -> tuple[np.ndarray, list[int]]:
    bands = cube.shape[-1]
    idx = [int(round((bands - 1) * f)) for f in (0.75, 0.50, 0.20)]
    rgb = np.stack([cube[..., i] for i in idx], axis=-1).astype(np.float32)
    rgb = np.clip(rgb, 0.0, None)
    lo, hi = np.quantile(rgb, [0.01, 0.995])
    if hi > lo:
        rgb = (rgb - lo) / (hi - lo)
    return np.clip(rgb, 0.0, 1.0), idx


def _masked_rgb(rgb: np.ndarray, observed_mask: np.ndarray, rgb_bands: list[int]) -> np.ndarray:
    out = rgb.copy()
    channel_mask = observed_mask[..., rgb_bands]
    gray = np.array([0.96, 0.96, 0.96], dtype=np.float32)
    out = np.where(channel_mask, out, gray.reshape(1, 1, 3))
    return out


def _rmse_map(original: np.ndarray, reconstruction: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    diff2 = (np.asarray(original, dtype=np.float32) - np.asarray(reconstruction, dtype=np.float32)) ** 2
    if mask is None:
        return np.sqrt(np.mean(diff2, axis=-1))
    mask = np.asarray(mask, dtype=bool)
    count = mask.sum(axis=-1)
    total = np.sum(np.where(mask, diff2, 0.0), axis=-1)
    out = np.full(count.shape, np.nan, dtype=np.float32)
    valid = count > 0
    out[valid] = np.sqrt(total[valid] / count[valid])
    return out


def _metric_box(ax: plt.Axes, metrics: tuple[float, float]) -> None:
    rmse, sam = metrics
    ax.text(
        0.03,
        0.97,
        f"RMSE*: {rmse:.3f}\nSAM*: {sam:.1f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=6.8,
        bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "#D7DCE2", "alpha": 0.92},
    )


if __name__ == "__main__":
    main()
