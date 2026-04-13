import numpy as np

from src.methods.polycal import PolynomialCalibration


def test_polycal_recovers_observed_quadratic_mapping():
    x = np.linspace(-1.0, 1.0, 128, dtype=np.float64)
    y = 0.2 + 1.1 * x + 0.3 * x**2
    mask = np.zeros_like(x, dtype=bool)
    mask[::2] = True

    model = PolynomialCalibration(degree=2, lambda_reg=1e-12).fit(x, y, mask)
    y_hat = model.apply(x)

    assert np.allclose(y_hat[mask], y[mask], atol=1e-7)


def test_polycal_uses_only_observed_entries():
    x = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float64)
    y = np.array([0.0, 1.0, 2.0, 99.0], dtype=np.float64)
    mask = np.array([True, True, True, False], dtype=bool)

    model = PolynomialCalibration(degree=1, lambda_reg=1e-12).fit(x, y, mask)
    y_hat = model.apply(x)

    assert np.allclose(y_hat[:3], y[:3], atol=1e-7)
