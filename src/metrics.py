from typing import Any, Dict

import numpy as np

from src.types import Tensor


def _sq_abs(x):
    return np.abs(x) ** 2


def _validate_same_shape(original: Tensor, reconstructed: Tensor) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(original.dense)
    y = np.asarray(reconstructed.dense)
    if x.shape != y.shape:
        raise ValueError(f"tensor shape mismatch: {x.shape} vs {y.shape}")
    return x, y


def _observed_eval_mask(tensor: Tensor, shape: tuple[int, ...]) -> np.ndarray | None:
    if tensor.mask is None:
        return None
    eval_mask = np.asarray(tensor.mask, dtype=bool)
    if eval_mask.shape != shape:
        raise ValueError(f"mask shape {eval_mask.shape} does not match tensor shape {shape}")
    if not np.any(eval_mask):
        return None
    return eval_mask


def _resolve_data_range(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if 0.0 <= x_min and x_max <= 1.0:
        return 1.0
    data_range = x_max - x_min
    return float(data_range if data_range > 1e-12 else 1.0)


def _uniform_filter2d(image: np.ndarray, window_size: int) -> np.ndarray:
    pad = window_size // 2
    padded = np.pad(image, ((pad, pad), (pad, pad)), mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant", constant_values=0.0).cumsum(axis=0).cumsum(axis=1)
    total = (
        integral[window_size:, window_size:]
        - integral[:-window_size, window_size:]
        - integral[window_size:, :-window_size]
        + integral[:-window_size, :-window_size]
    )
    return total / float(window_size * window_size)


def _ssim_single_channel(x: np.ndarray, y: np.ndarray, data_range: float, window_size: int = 7) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mu_x = _uniform_filter2d(x, window_size)
    mu_y = _uniform_filter2d(y, window_size)
    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x_sq = _uniform_filter2d(x * x, window_size) - mu_x_sq
    sigma_y_sq = _uniform_filter2d(y * y, window_size) - mu_y_sq
    sigma_xy = _uniform_filter2d(x * y, window_size) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    ssim_map = numerator / np.maximum(denominator, 1e-12)
    return float(np.mean(ssim_map))


def val_CR(num_original: int, num_factors: int):
    if num_factors == 0:
        return float("inf")
    return num_original / num_factors


def val_RMSE(original: Tensor, reconstructed: Tensor):
    x, y = _validate_same_shape(original, reconstructed)
    err = x - y

    eval_mask = _observed_eval_mask(original, err.shape)
    if eval_mask is not None:
        err = err[eval_mask]

    mse = np.mean(_sq_abs(err))
    return float(np.sqrt(mse))


def val_NMSE(original: Tensor, reconstructed: Tensor):
    x, y = _validate_same_shape(original, reconstructed)
    err = x - y
    ref = x

    eval_mask = _observed_eval_mask(original, err.shape)
    if eval_mask is not None:
        err = err[eval_mask]
        ref = ref[eval_mask]

    mse = np.mean(_sq_abs(err))
    ref_power = np.mean(_sq_abs(ref))
    if ref_power <= 1e-12:
        return float("inf")
    return float(mse / ref_power)


def val_NMSE_dB(original: Tensor, reconstructed: Tensor):
    nmse = val_NMSE(original, reconstructed)
    if nmse == 0:
        return float("-inf")
    return float(10 * np.log10(nmse))


def val_PSNR(original: Tensor, reconstructed: Tensor):
    x, y = _validate_same_shape(original, reconstructed)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    diff = x - y
    eval_mask = _observed_eval_mask(original, diff.shape)
    if eval_mask is not None:
        diff = diff[eval_mask]
    mse = float(np.mean(_sq_abs(diff)))
    if mse <= 1e-12:
        return float("inf")
    data_range = _resolve_data_range(x)
    return float(10.0 * np.log10((data_range ** 2) / mse))


def val_SSIM(original: Tensor, reconstructed: Tensor):
    x, y = _validate_same_shape(original, reconstructed)
    eval_mask = _observed_eval_mask(original, x.shape)
    if eval_mask is not None:
        flat_x = np.asarray(x, dtype=np.float64)[eval_mask]
        flat_y = np.asarray(y, dtype=np.float64)[eval_mask]
        data_range = _resolve_data_range(flat_x)
        mu_x = float(np.mean(flat_x))
        mu_y = float(np.mean(flat_y))
        sigma_x_sq = float(np.var(flat_x))
        sigma_y_sq = float(np.var(flat_y))
        sigma_xy = float(np.mean((flat_x - mu_x) * (flat_y - mu_y)))
        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2
        numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
        denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x_sq + sigma_y_sq + c2)
        return float(numerator / max(denominator, 1e-12))

    data_range = _resolve_data_range(x)

    if x.ndim == 2:
        return _ssim_single_channel(x, y, data_range=data_range)

    if x.ndim == 3 and x.shape[-1] <= 4:
        values = [
            _ssim_single_channel(x[..., c], y[..., c], data_range=data_range)
            for c in range(x.shape[-1])
        ]
        return float(np.mean(values))

    flat_x = np.asarray(x, dtype=np.float64).reshape(-1)
    flat_y = np.asarray(y, dtype=np.float64).reshape(-1)
    mu_x = float(np.mean(flat_x))
    mu_y = float(np.mean(flat_y))
    sigma_x_sq = float(np.var(flat_x))
    sigma_y_sq = float(np.var(flat_y))
    sigma_xy = float(np.mean((flat_x - mu_x) * (flat_y - mu_y)))
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x_sq + sigma_y_sq + c2)
    return float(numerator / max(denominator, 1e-12))


def val_SAM(original: Tensor, reconstructed: Tensor):
    x, y = _validate_same_shape(original, reconstructed)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    eval_mask = _observed_eval_mask(original, x.shape)

    if x.ndim == 1:
        x_spec = x.reshape(1, -1)
        y_spec = y.reshape(1, -1)
        mask_spec = None if eval_mask is None else eval_mask.reshape(1, -1)
    else:
        x_spec = x.reshape(-1, x.shape[-1])
        y_spec = y.reshape(-1, y.shape[-1])
        mask_spec = None if eval_mask is None else eval_mask.reshape(-1, eval_mask.shape[-1])

    if mask_spec is None:
        x_norm = np.linalg.norm(x_spec, axis=1)
        y_norm = np.linalg.norm(y_spec, axis=1)
        valid = (x_norm > 1e-12) & (y_norm > 1e-12)
        if not np.any(valid):
            return 0.0

        cosine = np.sum(x_spec[valid] * y_spec[valid], axis=1) / (x_norm[valid] * y_norm[valid])
        cosine = np.clip(cosine, -1.0, 1.0)
        angles = np.degrees(np.arccos(cosine))
        return float(np.mean(angles))

    row_valid = np.any(mask_spec, axis=1)
    if not np.any(row_valid):
        return 0.0

    numerators: list[float] = []
    denominators: list[float] = []
    for x_row, y_row, row_mask in zip(x_spec[row_valid], y_spec[row_valid], mask_spec[row_valid], strict=False):
        xv = x_row[row_mask]
        yv = y_row[row_mask]
        x_norm = float(np.linalg.norm(xv))
        y_norm = float(np.linalg.norm(yv))
        if x_norm <= 1e-12 or y_norm <= 1e-12:
            continue
        numerators.append(float(np.dot(xv, yv)))
        denominators.append(x_norm * y_norm)

    if not denominators:
        return 0.0

    cosine = np.divide(
        np.asarray(numerators, dtype=np.float64),
        np.asarray(denominators, dtype=np.float64),
    )
    cosine = np.clip(cosine, -1.0, 1.0)
    angles = np.degrees(np.arccos(cosine))
    return float(np.mean(angles))


def val_ERGAS(original: Tensor, reconstructed: Tensor, *, ratio: float = 1.0) -> float:
    x, y = _validate_same_shape(original, reconstructed)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    eval_mask = _observed_eval_mask(original, x.shape)

    if x.ndim == 1:
        x = x.reshape(1, 1, -1)
        y = y.reshape(1, 1, -1)
        eval_mask = None if eval_mask is None else eval_mask.reshape(1, 1, -1)
    elif x.ndim == 2:
        x = x[..., None]
        y = y[..., None]
        eval_mask = None if eval_mask is None else eval_mask[..., None]

    if x.ndim != 3:
        raise ValueError(f"ERGAS expects 1D, 2D, or 3D arrays, got {x.shape}")

    terms: list[float] = []
    for band_idx in range(x.shape[-1]):
        x_band = x[..., band_idx]
        y_band = y[..., band_idx]
        if eval_mask is not None:
            band_mask = np.asarray(eval_mask[..., band_idx], dtype=bool)
            if not np.any(band_mask):
                continue
            x_band = x_band[band_mask]
            y_band = y_band[band_mask]

        ref_mean = float(np.mean(np.abs(x_band)))
        if ref_mean <= 1e-12:
            continue
        rmse = float(np.sqrt(np.mean((x_band - y_band) ** 2)))
        terms.append((rmse / ref_mean) ** 2)

    if not terms:
        return 0.0
    return float((100.0 / float(ratio)) * np.sqrt(np.mean(terms)))


def CR(num_original: int, num_factors: int) -> Dict[str, Any]:
    return {"CR": val_CR(num_original, num_factors)}


def RMSE(original: Tensor, reconstructed: Tensor) -> Dict[str, Any]:
    return {"RMSE": val_RMSE(original, reconstructed)}


def NMSE(original: Tensor, reconstructed: Tensor) -> Dict[str, Any]:
    return {"NMSE": val_NMSE(original, reconstructed)}


def NMSE_dB(original: Tensor, reconstructed: Tensor) -> Dict[str, Any]:
    return {"NMSE_dB": val_NMSE_dB(original, reconstructed)}


def PSNR(original: Tensor, reconstructed: Tensor) -> Dict[str, Any]:
    return {"PSNR": val_PSNR(original, reconstructed)}


def SSIM(original: Tensor, reconstructed: Tensor) -> Dict[str, Any]:
    return {"SSIM": val_SSIM(original, reconstructed)}


def SAM(original: Tensor, reconstructed: Tensor) -> Dict[str, Any]:
    return {"SAM": val_SAM(original, reconstructed)}


def ERGAS(original: Tensor, reconstructed: Tensor) -> Dict[str, Any]:
    return {"ERGAS": val_ERGAS(original, reconstructed)}
