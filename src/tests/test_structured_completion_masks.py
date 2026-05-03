import numpy as np

from src.utils.completion_ops import structured_observed_mask


def test_block_mask_has_spatial_holes_across_bands():
    mask = structured_observed_mask(
        (16, 20, 5),
        pattern="block",
        missing_rate=0.25,
        seed=0,
        block_shape=(4, 5),
    )

    assert mask.shape == (16, 20, 5)
    assert mask.dtype == np.bool_
    assert np.any(mask)
    assert np.any(~mask)
    missing_spatial = np.any(~mask, axis=-1)
    assert np.any(missing_spatial)
    assert np.all(mask[missing_spatial] == mask[missing_spatial, :1])


def test_band_mask_drops_complete_bands():
    mask = structured_observed_mask((8, 9, 10), pattern="spectral-band", missing_rate=0.3, seed=1)

    missing_by_band = np.all(~mask, axis=(0, 1))
    observed_by_band = np.all(mask, axis=(0, 1))
    assert int(np.count_nonzero(missing_by_band)) == 3
    assert np.all(missing_by_band | observed_by_band)


def test_stripe_mask_drops_complete_columns():
    mask = structured_observed_mask(
        (12, 18, 4),
        pattern="stripe",
        missing_rate=0.25,
        seed=2,
        stripe_axis=1,
        stripe_width=2,
    )

    missing_cols = np.all(~mask, axis=(0, 2))
    observed_cols = np.all(mask, axis=(0, 2))
    assert np.any(missing_cols)
    assert np.all(missing_cols | observed_cols)
