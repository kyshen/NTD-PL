import numpy as np

from src.ntdpl import beta_update_link, make_link, ntdpl


def _finite_difference(link, values, beta, eps=1e-5):
    values = np.asarray(values, dtype=np.float64)
    plus = link.value(values + eps, beta).astype(np.float64)
    minus = link.value(values - eps, beta).astype(np.float64)
    return (plus - minus) / (2.0 * eps)


def test_link_derivatives_match_finite_differences():
    values = np.linspace(-0.9, 0.9, 11, dtype=np.float64)
    specs = [
        ("power", np.array([0.2, 0.9, -0.4, 0.1], dtype=np.float32)),
        ("chebyshev", np.array([0.1, 0.7, -0.2, 0.05], dtype=np.float32)),
        ("rbf", np.array([0.1, 0.8, -0.3, 0.2, -0.1], dtype=np.float32)),
        ("spline", np.array([-0.8, -0.2, 0.3, 0.7, 1.0], dtype=np.float32)),
    ]

    for kind, beta in specs:
        link = make_link(kind).fit(values, len(beta) - 1)
        if kind == "spline":
            values_eval = np.array([-0.75, -0.25, 0.25, 0.75], dtype=np.float64)
        else:
            values_eval = values
        numerical = _finite_difference(link, values_eval, beta)
        analytic = link.derivative(values_eval, beta)
        assert np.allclose(analytic, numerical, atol=2e-3, rtol=2e-3), kind


def test_spline_link_update_recovers_piecewise_linear_response():
    s = np.linspace(-1.0, 1.0, 101, dtype=np.float64)
    y = np.where(s < 0.0, 0.5 * s, 1.5 * s)
    link = make_link("spline").fit(s, active_q=4)

    beta = beta_update_link(y, s, link, active_q=4, lambda_beta=1e-12)
    y_hat = link.value(s, beta)

    assert np.sqrt(np.mean((y_hat - y) ** 2)) < 1e-5


def test_link_state_round_trip_preserves_prediction():
    s = np.linspace(-1.0, 1.0, 31, dtype=np.float64)
    beta = np.array([-1.0, -0.2, 0.3, 0.8, 1.2], dtype=np.float32)
    link = make_link("spline").fit(s, active_q=4)

    restored = make_link("spline", link.state_dict())

    assert np.allclose(restored.value(s, beta), link.value(s, beta))
    assert np.allclose(restored.derivative(s, beta), link.derivative(s, beta))


def test_non_nested_links_disable_degree_continuation():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 4, 3)).astype(np.float32)

    (_, _, beta, link_state), history = ntdpl(
        X=x,
        rank=(2, 2, 2),
        init_n_iter_max=2,
        p_max=5,
        allow_constant_term=True,
        n_iter_max=1,
        use_continuation=True,
        factor_normalize=True,
        lr_core=1e-4,
        lr_factors=3e-4,
        lambda_core=1e-6,
        lambda_factors=1e-6,
        lambda_beta=1e-6,
        beta_update_method="ridge_lstsq",
        init="tucker",
        random_state=0,
        beta_update_interval=1,
        stable_beta_update=True,
        beta_update_stage="before_grad",
        return_history=True,
        link_kind="spline",
        return_link_state=True,
    )

    assert link_state["kind"] == "spline"
    assert len(beta) == 6
    assert history[0]["p"] == 5
