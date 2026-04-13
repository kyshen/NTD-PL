import numpy as np


def mask_to_bool(mask: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    mask_arr = np.asarray(mask)
    if mask_arr.shape != shape:
        raise ValueError(
            f"`mask.shape` must match data shape, got {mask_arr.shape} vs {shape}."
        )
    if np.issubdtype(mask_arr.dtype, np.bool_):
        return mask_arr
    return mask_arr > 0


def mask_to_float(
    mask: np.ndarray, shape: tuple[int, ...], dtype: np.dtype | type = np.float32
) -> np.ndarray:
    mask_bool = mask_to_bool(mask, shape)
    return mask_bool.astype(dtype, copy=False)


def mean_fill_missing(X_obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask_bool = mask_to_bool(mask, np.asarray(X_obs).shape)
    if not np.any(mask_bool):
        raise ValueError("`mask` contains no observed entries.")
    X_init = np.array(X_obs, copy=True)
    obs_mean = float(np.mean(X_obs[mask_bool]))
    X_init[~mask_bool] = obs_mean
    return X_init


def random_observed_mask(
    shape: tuple[int, ...],
    *,
    missing_rate: float,
    seed: int,
) -> np.ndarray:
    missing_rate = float(missing_rate)
    if not 0.0 <= missing_rate < 1.0:
        raise ValueError(f"`missing_rate` must be in [0, 1), got {missing_rate}.")

    observed_rate = 1.0 - missing_rate
    rng = np.random.default_rng(int(seed))
    mask = rng.random(shape) < observed_rate

    if not np.any(mask):
        flat = mask.reshape(-1)
        flat[rng.integers(0, flat.size)] = True
        mask = flat.reshape(shape)

    if np.all(mask) and missing_rate > 0.0:
        flat = mask.reshape(-1)
        flat[rng.integers(0, flat.size)] = False
        mask = flat.reshape(shape)

    return np.asarray(mask, dtype=bool)
