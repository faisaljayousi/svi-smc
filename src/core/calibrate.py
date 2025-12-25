import numpy as np
from scipy.optimize import least_squares

from .model import forward_model


class SVICalibrator:
    def __init__(self, cfg, method="pro"):
        """
        Stateful SVI Calibrator.
        """
        self.cfg = cfg
        self.method = method
        self.bounds = (cfg.model.bounds.lower, cfg.model.bounds.upper)

        # Memory: Initialise with the config's starting guess
        self.current_params = np.array(cfg.model.initial_guess)
        self.history = []

    def objective(self, params, ks, market_w, lambda_reg):
        """Dispatches to the chosen strategy."""
        if self.method.lower() == "tikhonov":
            return self._objective_tikhonov(
                params, ks, market_w, self.current_params, lambda_reg
            )
        return self._objective_simple(
            params, ks, market_w, self.current_params, lambda_reg
        )

    @staticmethod
    def _objective_simple(params, ks, market_w, prior_params, lambda_reg):
        model_w = forward_model(ks, *params)
        residuals = model_w - market_w
        rho_penalty = np.sqrt(lambda_reg) * (params[2] - prior_params[2])
        return np.append(residuals, [rho_penalty])

    @staticmethod
    def _objective_tikhonov(params, ks, market_w, prior_params, lambda_reg):
        model_w = forward_model(ks, *params)
        # Normalisation for regime-agnostic lambda
        scale = 1.0 / (np.mean(market_w) + 1e-6)
        residuals = (model_w - market_w) * scale

        stiff_l = lambda_reg * scale * 10.0
        rho_pen = np.sqrt(stiff_l) * (params[2] - prior_params[2])
        # Penalty on 'b' to stabilise slope
        b_pen = (
            np.sqrt(stiff_l * 0.2)
            * (params[1] - prior_params[1])
            / (prior_params[1] + 1e-6)
        )

        return np.append(residuals, [rho_pen, b_pen])

    def find_best_lambda(self, ks, market_w):
        """4-fold Interleaved Cross-Validation."""
        indices = np.arange(len(ks))
        train_mask = indices % 4 != 0
        val_mask = ~train_mask

        candidates = [1e-4, 1e-3, 1e-2, 1e-1, 0.5]
        best_l, best_score = candidates[0], float("inf")

        for l_test in candidates:
            res = least_squares(
                self.objective,
                x0=self.current_params,
                args=(ks[train_mask], market_w[train_mask], l_test),
                bounds=self.bounds,
                ftol=1e-7,
            )
            val_pred = forward_model(ks[val_mask], *res.x)
            val_rmse = np.sqrt(np.mean((val_pred - market_w[val_mask]) ** 2))
            if val_rmse < best_score:
                best_score, best_l = val_rmse, l_test
        return best_l

    def calibrate_tick(self, ks, market_w):
        """The main method to call per market update."""
        opt_l = self.find_best_lambda(ks, market_w)

        res = least_squares(
            self.objective,
            x0=self.current_params,
            args=(ks, market_w, opt_l),
            bounds=self.bounds,
            ftol=1e-9,
        )

        # Update internal state (Warm Start)
        self.current_params = res.x

        # Track history for TCA
        self.history.append(
            {
                "params": res.x.copy(),
                "lambda": opt_l,
                "rmse": np.sqrt(np.mean((forward_model(ks, *res.x) - market_w) ** 2)),
            }
        )
        return res.x
