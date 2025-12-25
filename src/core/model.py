import numpy as np


def forward_model(k, a, b, rho, m, sigma):
    inner = (k - m) ** 2 + sigma**2
    return a + b * (rho * (k - m) + np.sqrt(inner))


def svi_residuals(params, ks, target_total_vars):
    model_vars = forward_model(ks, *params)
    return model_vars - target_total_vars


def svi_slope(k, a, b, rho, m, sigma):
    """
    Calculates dw/dk (the slope of total variance).
    """
    inner_sqrt = np.sqrt((k - m) ** 2 + sigma**2)
    return b * (rho + (k - m) / inner_sqrt)
