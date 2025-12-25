import logging
from datetime import datetime

import numpy as np
import pandas as pd

from src.core.calibrate import SVICalibrator
from src.core.model import forward_model
from src.utils.config_loader import load_config
from src.utils.logger import setup_logging

# --- Setup Logging ---
logger = logging.getLogger(__name__)


def run_production_simulation():
    cfg = load_config("config/params.yaml")

    true_params = cfg.sim.true_params
    n_ticks = cfg.sim.n_ticks
    ks = np.linspace(cfg.sim.strike_range[0], cfg.sim.strike_range[1], cfg.sim.n_points)

    engine_ls = SVICalibrator(cfg, method="simple")
    engine_tikh = SVICalibrator(cfg, method="tikhonov")

    logger.info(
        f"Simulation: {cfg.model.name} | Ticks: {n_ticks} | Noise: {cfg.sim.noise_level}"
    )

    for _ in range(n_ticks):
        noise = np.random.normal(0, cfg.sim.noise_level, len(ks))
        market_w = forward_model(ks, *true_params) + noise

        engine_ls.calibrate_tick(ks, market_w)
        engine_tikh.calibrate_tick(ks, market_w)

    return engine_ls.history, engine_tikh.history


def export_results(history_ls, history_tikh):
    """Saves raw simulation data to CSV for Notebook analysis."""
    data = []
    for t in range(len(history_ls)):
        data.append(
            {
                "tick": t,
                "rho_ls": history_ls[t]["params"][2],
                "rmse_ls": history_ls[t]["rmse"],
                "rho_tikh": history_tikh[t]["params"][2],
                "rmse_tikh": history_tikh[t]["rmse"],
                "lambda_tikh": history_tikh[t]["lambda"],
            }
        )

    df = pd.DataFrame(data)
    filename = f"data/sim_results_{datetime.now().strftime('%Y%H%M')}.csv"
    df.to_csv(filename, index=False)
    logger.info(f"Simulation complete. Results exported to {filename}")
    return df


def generate_live_summary(df):
    """Calculates final 'Quant' metrics for the console."""
    jitter_ls = np.sum(np.abs(np.diff(df["rho_ls"])))
    jitter_tikh = np.sum(np.abs(np.diff(df["rho_tikh"])))
    rmse_ls = df["rmse_ls"].mean()
    rmse_tikh = df["rmse_tikh"].mean()

    summary = pd.DataFrame(
        {
            "Metric": ["Avg OOS RMSE", "Total Parameter Jitter", "Efficiency Index"],
            "Simple LS": [rmse_ls, jitter_ls, 1 / (rmse_ls * jitter_ls)],
            "Tikhonov": [rmse_tikh, jitter_tikh, 1 / (rmse_tikh * jitter_tikh)],
        }
    )

    # 1. Prepare the report string
    report_header = (
        "\n"
        + "=" * 65
        + "\n"
        + " " * 12
        + "FINAL PRODUCTION PERFORMANCE REPORT\n"
        + "=" * 65
    )
    report_table = summary.to_string(index=False)
    improvement = (1 - jitter_tikh / jitter_ls) * 100
    conclusion = f"CONCLUSION: Tikhonov reduced parameter noise by {improvement:.1f}%"
    footer = "=" * 65 + "\n"

    # 2. Combine and Log
    full_report = f"{report_header}\n{report_table}\n{'=' * 65}\n{conclusion}\n{footer}"
    logging.info(full_report)


if __name__ == "__main__":
    setup_logging()
    h_ls, h_tikh = run_production_simulation()
    results_df = export_results(h_ls, h_tikh)
    generate_live_summary(results_df)
