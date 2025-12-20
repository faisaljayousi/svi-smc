import numpy as np

from src.model import forward_model


def test_svi_non_negativity():
    """
    Ensure SVI never returns negative total variance for valid parameters.
    """
    ks = np.linspace(-1.0, 1.0, 50)
    # Typical 'extreme' but valid parameters
    params = [0.001, 0.1, -0.9, 0.0, 0.01]
    w = forward_model(ks, *params)
    assert np.all(w >= 0), "SVI produced negative total variance."


def test_svi_at_m():
    """At k=m, the value should be a + b * sigma."""
    a, b, rho, m, sigma = 0.04, 0.1, -0.7, 0.1, 0.1
    val = forward_model(m, a, b, rho, m, sigma)
    expected = a + b * sigma
    assert np.isclose(val, expected), f"Expected {expected}, got {val}"
