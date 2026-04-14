from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PolynomialFeatures

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.hsi import _load_hsi_from_file


DATASETS = [
    ("jasper_ridge_hsi", "Jasper Ridge", Path("data/hsi/jasperRidge2_R198.mat")),
    ("samson_hsi", "Samson", Path("data/hsi-similar/samson_1.img")),
    ("urban_hsi", "Urban", Path("data/hsi-similar/Urban_R162.mat")),
    ("cuprite_hsi", "Cuprite", Path("data/hsi-similar/Cuprite_S1_R188.img")),
]


def _normalize_cube(cube: np.ndarray) -> np.ndarray:
    cube = np.asarray(cube, dtype=np.float32)
    cube = cube - float(np.min(cube))
    scale = float(np.max(cube))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Invalid normalization scale: {scale}")
    return cube / scale


def _load_project_gain() -> pd.DataFrame:
    path = PROJECT_ROOT / "experiment/outputs/real-hsi-robustness/real_hsi_robustness_main_table_numeric.csv"
    frame = pd.read_csv(path)
    recon = (
        frame.loc[frame["task_name"].eq("decompose"), ["dataset", "gain_pct", "delta_nmse_db", "delta_sam"]]
        .rename(
            columns={
                "gain_pct": "ntdpl_gain_pct_recon",
                "delta_nmse_db": "ntdpl_delta_nmse_db_recon",
                "delta_sam": "ntdpl_delta_sam_recon",
            }
        )
        .copy()
    )
    return recon


def _median_gamma(x: np.ndarray) -> float:
    diffs = x[:, None, :] - x[None, :, :]
    dist2 = np.sum(diffs * diffs, axis=2)
    tri = dist2[np.triu_indices_from(dist2, k=1)]
    med = float(np.median(tri))
    return 1.0 / max(med, 1e-6)


def _scene_metrics(cube: np.ndarray, *, seed: int = 0, sample_size: int = 5000) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    x = cube.reshape(-1, cube.shape[-1])
    idx = rng.choice(x.shape[0], size=min(sample_size, x.shape[0]), replace=False)
    xs = np.asarray(x[idx], dtype=np.float32)
    xc = xs - xs.mean(axis=0, keepdims=True)

    pca_full = PCA(n_components=min(20, xc.shape[1] - 1), svd_solver="randomized", random_state=seed)
    pca_full.fit(xc)
    cum_energy = np.cumsum(pca_full.explained_variance_ratio_)
    latent_dim = int(np.searchsorted(cum_energy, 0.99) + 1)
    latent_dim = max(3, min(8, latent_dim))

    pca = PCA(n_components=latent_dim, svd_solver="randomized", random_state=seed)
    z = pca.fit_transform(xc)
    x_lin = pca.inverse_transform(z)
    mse_lin = float(np.mean((xc - x_lin) ** 2))
    var = float(np.mean(xc**2))
    linear_residual_ratio = mse_lin / max(var, 1e-12)

    linear_decoder = Ridge(alpha=1e-6, fit_intercept=True)
    linear_decoder.fit(z, xs)
    x_hat_linear = linear_decoder.predict(z)
    mse_decoder_linear = float(np.mean((xs - x_hat_linear) ** 2))

    poly = PolynomialFeatures(degree=2, include_bias=False)
    z2 = poly.fit_transform(z)
    quadratic_decoder = Ridge(alpha=1e-4, fit_intercept=True)
    quadratic_decoder.fit(z2, xs)
    x_hat_quadratic = quadratic_decoder.predict(z2)
    mse_decoder_quadratic = float(np.mean((xs - x_hat_quadratic) ** 2))
    quadratic_decoder_gain = (mse_decoder_linear - mse_decoder_quadratic) / max(mse_decoder_linear, 1e-12)

    z_train, z_test, y_train, y_test = train_test_split(z, xs, test_size=0.3, random_state=seed)
    ridge = Ridge(alpha=1e-4)
    ridge.fit(z_train, y_train)
    linear_test_mse = float(np.mean((y_test - ridge.predict(z_test)) ** 2))

    gamma_sample = z_train[rng.choice(len(z_train), size=min(1000, len(z_train)), replace=False)]
    kernel = KernelRidge(alpha=1e-2, kernel="rbf", gamma=_median_gamma(gamma_sample))
    kernel.fit(z_train, y_train)
    kernel_test_mse = float(np.mean((y_test - kernel.predict(z_test)) ** 2))
    kernel_regression_gain = (linear_test_mse - kernel_test_mse) / max(linear_test_mse, 1e-12)

    local_latent = PCA(
        n_components=min(10, xs.shape[1] - 1),
        svd_solver="randomized",
        random_state=seed,
    ).fit_transform(xs)
    nn = NearestNeighbors(n_neighbors=min(31, len(local_latent))).fit(local_latent)
    _, neighbor_ids = nn.kneighbors(local_latent)
    local_residuals: list[float] = []
    for i in range(len(local_latent)):
        neighborhood = xs[neighbor_ids[i, 1:]]
        center = neighborhood.mean(axis=0, keepdims=True)
        q = min(3, neighborhood.shape[0] - 1, neighborhood.shape[1] - 1)
        tangent = PCA(n_components=q, svd_solver="randomized", random_state=seed).fit(neighborhood - center)
        point = xs[i : i + 1] - center
        point_proj = tangent.inverse_transform(tangent.transform(point))
        local_residuals.append(float(np.mean((point - point_proj) ** 2) / max(np.mean(point**2), 1e-12)))
    local_linearity_defect_median = float(np.median(local_residuals))

    return {
        "sampled_pixels": int(xs.shape[0]),
        "latent_dim": int(latent_dim),
        "linear_residual_ratio": linear_residual_ratio,
        "quadratic_decoder_gain": quadratic_decoder_gain,
        "kernel_regression_gain": kernel_regression_gain,
        "local_linearity_defect_median": local_linearity_defect_median,
    }


def analyze() -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for dataset, label, rel_path in DATASETS:
        cube = _normalize_cube(_load_hsi_from_file(PROJECT_ROOT / rel_path))
        row: dict[str, float | int | str] = {
            "dataset": dataset,
            "dataset_label": label,
            "shape": f"{cube.shape[0]}x{cube.shape[1]}x{cube.shape[2]}",
            "bands": int(cube.shape[2]),
            "pixels": int(cube.shape[0] * cube.shape[1]),
        }
        row.update(_scene_metrics(cube))
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame = frame.merge(_load_project_gain(), on="dataset", how="left")

    score_columns = [
        "linear_residual_ratio",
        "quadratic_decoder_gain",
        "kernel_regression_gain",
        "local_linearity_defect_median",
        "ntdpl_gain_pct_recon",
    ]
    for col in score_columns:
        frame[f"{col}_rank"] = frame[col].rank(ascending=False, method="dense")
    frame["nonlinearity_rank_mean"] = frame[[f"{col}_rank" for col in score_columns]].mean(axis=1)
    frame["overall_order"] = frame["nonlinearity_rank_mean"].rank(ascending=True, method="dense").astype(int)
    return frame.sort_values(["overall_order", "dataset_label"]).reset_index(drop=True)


def main() -> None:
    out_dir = PROJECT_ROOT / "outputs" / "real_hsi_nonlinearity"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = analyze()
    csv_path = out_dir / "real_hsi_nonlinearity_metrics.csv"
    frame.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    with pd.option_context("display.max_columns", None, "display.width", 240):
        print(frame)


if __name__ == "__main__":
    main()
