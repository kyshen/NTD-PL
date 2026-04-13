import numpy as np

from src.metrics import val_ERGAS, val_PSNR, val_SAM, val_SSIM
from src.types import Tensor


def test_psnr_and_ssim_are_perfect_for_identical_images():
    x = np.ones((16, 16, 3), dtype=np.float32) * 0.5
    original = Tensor(shape=x.shape, dense=x)
    reconstructed = Tensor(shape=x.shape, dense=x.copy())

    assert np.isinf(val_PSNR(original, reconstructed))
    assert np.isclose(val_SSIM(original, reconstructed), 1.0)


def test_metrics_use_observed_mask_entries():
    x = np.zeros((2, 2, 1), dtype=np.float32)
    y = x.copy()
    y[0, 0, 0] = 1.0

    original = Tensor(
        shape=x.shape,
        dense=x,
        mask=np.array([[[True], [False]], [[False], [False]]], dtype=bool),
    )
    reconstructed = Tensor(shape=y.shape, dense=y)

    assert np.isclose(val_PSNR(original, reconstructed), 0.0)


def test_sam_uses_masked_entries_only():
    x = np.array([[[1.0, 0.0], [1.0, 0.0]]], dtype=np.float32)
    y = np.array([[[0.0, 1.0], [0.0, 1.0]]], dtype=np.float32)
    mask = np.array([[[True, False], [False, True]]], dtype=bool)

    original = Tensor(shape=x.shape, dense=x, mask=mask)
    reconstructed = Tensor(shape=y.shape, dense=y)

    assert np.isclose(val_SAM(original, reconstructed), 0.0)


def test_ergas_uses_masked_entries_only():
    x = np.ones((1, 2, 2), dtype=np.float32)
    y = x.copy()
    y[0, 0, 0] = 2.0
    y[0, 1, 1] = 3.0
    mask = np.array([[[True, True], [False, False]]], dtype=bool)

    original = Tensor(shape=x.shape, dense=x, mask=mask)
    reconstructed = Tensor(shape=y.shape, dense=y)

    expected = 100.0 * np.sqrt(np.mean(np.array([1.0**2, 0.0**2], dtype=np.float64)))
    assert np.isclose(val_ERGAS(original, reconstructed), expected)
