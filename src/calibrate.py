from scipy.optimize import least_squares

from src.model import svi_residuals


def calibrate_svi_ls(ks, target_total_vars, x0, bounds):
    res = least_squares(
        svi_residuals,
        x0,
        bounds=bounds,
        args=(ks, target_total_vars),
        method="trf",
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
        max_nfev=1000,
    )

    return res.x, res.success
