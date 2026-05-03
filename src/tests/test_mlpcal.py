import numpy as np

from src.methods.mlpcal import ScalarMLPCalibration


def test_mlpcal_learns_smooth_scalar_mapping():
    x = np.linspace(-2.0, 2.0, 256, dtype=np.float64)
    y = np.sin(x) + 0.2 * x
    mask = np.ones_like(x, dtype=bool)

    model = ScalarMLPCalibration(
        hidden_units=24,
        lr=2e-2,
        max_iter=1500,
        batch_size=256,
        lambda_reg=1e-6,
        random_state=0,
    ).fit(x, y, mask)
    y_hat = model.apply(x)

    assert np.mean((y_hat - y) ** 2) < 2e-3


def test_mlpcal_uses_only_observed_entries():
    x = np.linspace(-1.0, 1.0, 64, dtype=np.float64)
    y = 2.0 * x + 0.5
    y[-8:] = 100.0
    mask = np.ones_like(x, dtype=bool)
    mask[-8:] = False

    model = ScalarMLPCalibration(
        hidden_units=8,
        lr=1e-2,
        max_iter=1000,
        batch_size=32,
        lambda_reg=1e-6,
        random_state=1,
    ).fit(x, y, mask)
    y_hat = model.apply(x)

    assert np.mean((y_hat[mask] - (2.0 * x[mask] + 0.5)) ** 2) < 1e-3
