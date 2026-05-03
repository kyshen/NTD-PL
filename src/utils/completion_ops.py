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


def block_observed_mask(
    shape: tuple[int, ...],
    *,
    missing_rate: float,
    seed: int,
    block_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Observed mask with spatial rectangular holes shared across channels."""
    if len(shape) < 2:
        raise ValueError(f"Block missingness requires at least two dimensions, got {shape}.")
    missing_rate = _validate_missing_rate(missing_rate)
    rng = np.random.default_rng(int(seed))
    mask = np.ones(shape, dtype=bool)
    height, width = int(shape[0]), int(shape[1])
    if block_shape is None:
        block_shape = (max(1, height // 8), max(1, width // 8))
    block_h, block_w = (max(1, int(block_shape[0])), max(1, int(block_shape[1])))
    target_missing = int(round(missing_rate * mask.size))

    attempts = 0
    max_attempts = max(256, int(np.ceil(target_missing / max(1, block_h * block_w))) * 16)
    while np.count_nonzero(~mask) < target_missing and attempts < max_attempts:
        attempts += 1
        top = int(rng.integers(0, max(1, height - block_h + 1)))
        left = int(rng.integers(0, max(1, width - block_w + 1)))
        mask[top : top + block_h, left : left + block_w, ...] = False

    return _ensure_nontrivial_mask(mask, rng, missing_rate)


def band_observed_mask(
    shape: tuple[int, ...],
    *,
    missing_rate: float,
    seed: int,
    band_axis: int = -1,
) -> np.ndarray:
    """Observed mask with entire channels/bands missing."""
    missing_rate = _validate_missing_rate(missing_rate)
    axis = int(band_axis) % len(shape)
    band_count = int(shape[axis])
    missing_count = int(round(missing_rate * band_count))
    missing_count = max(1 if missing_rate > 0.0 else 0, min(band_count - 1, missing_count))
    rng = np.random.default_rng(int(seed))
    missing_bands = rng.choice(band_count, size=missing_count, replace=False)
    mask = np.ones(shape, dtype=bool)
    index = [slice(None)] * len(shape)
    index[axis] = missing_bands
    mask[tuple(index)] = False
    return _ensure_nontrivial_mask(mask, rng, missing_rate)


def stripe_observed_mask(
    shape: tuple[int, ...],
    *,
    missing_rate: float,
    seed: int,
    stripe_axis: int = 1,
    stripe_width: int | None = None,
) -> np.ndarray:
    """Observed mask with spatial line/stripe dropouts shared across channels."""
    if len(shape) < 2:
        raise ValueError(f"Stripe missingness requires at least two dimensions, got {shape}.")
    missing_rate = _validate_missing_rate(missing_rate)
    axis = int(stripe_axis) % len(shape)
    axis_size = int(shape[axis])
    if stripe_width is None:
        stripe_width = max(1, axis_size // 64)
    stripe_width = max(1, int(stripe_width))
    target_positions = int(round(missing_rate * axis_size))
    target_positions = max(1 if missing_rate > 0.0 else 0, min(axis_size - 1, target_positions))
    rng = np.random.default_rng(int(seed))
    missing = np.zeros(axis_size, dtype=bool)

    attempts = 0
    max_attempts = max(256, int(np.ceil(target_positions / stripe_width)) * 16)
    while np.count_nonzero(missing) < target_positions and attempts < max_attempts:
        attempts += 1
        start = int(rng.integers(0, max(1, axis_size - stripe_width + 1)))
        missing[start : start + stripe_width] = True

    if np.count_nonzero(missing) < target_positions:
        remaining = np.flatnonzero(~missing)
        fill = rng.choice(remaining, size=target_positions - int(np.count_nonzero(missing)), replace=False)
        missing[fill] = True

    mask = np.ones(shape, dtype=bool)
    index = [slice(None)] * len(shape)
    index[axis] = missing
    mask[tuple(index)] = False
    return _ensure_nontrivial_mask(mask, rng, missing_rate)


def structured_observed_mask(
    shape: tuple[int, ...],
    *,
    pattern: str,
    missing_rate: float,
    seed: int,
    block_shape: tuple[int, int] | None = None,
    stripe_axis: int = 1,
    stripe_width: int | None = None,
    band_axis: int = -1,
) -> np.ndarray:
    pattern_key = str(pattern).strip().lower().replace("_", "-")
    if pattern_key == "random":
        return random_observed_mask(shape, missing_rate=missing_rate, seed=seed)
    if pattern_key in {"block", "blocks", "spatial-block"}:
        return block_observed_mask(shape, missing_rate=missing_rate, seed=seed, block_shape=block_shape)
    if pattern_key in {"band", "bands", "spectral-band"}:
        return band_observed_mask(shape, missing_rate=missing_rate, seed=seed, band_axis=band_axis)
    if pattern_key in {"stripe", "stripes", "dropout"}:
        return stripe_observed_mask(
            shape,
            missing_rate=missing_rate,
            seed=seed,
            stripe_axis=stripe_axis,
            stripe_width=stripe_width,
        )
    raise ValueError(f"Unsupported structured missingness pattern: {pattern!r}.")


def _validate_missing_rate(missing_rate: float) -> float:
    value = float(missing_rate)
    if not 0.0 <= value < 1.0:
        raise ValueError(f"`missing_rate` must be in [0, 1), got {missing_rate}.")
    return value


def _ensure_nontrivial_mask(mask: np.ndarray, rng: np.random.Generator, missing_rate: float) -> np.ndarray:
    out = np.asarray(mask, dtype=bool)
    if not np.any(out):
        flat = out.reshape(-1)
        flat[int(rng.integers(0, flat.size))] = True
        out = flat.reshape(out.shape)
    if np.all(out) and float(missing_rate) > 0.0:
        flat = out.reshape(-1)
        flat[int(rng.integers(0, flat.size))] = False
        out = flat.reshape(out.shape)
    return out
