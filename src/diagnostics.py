import logging

import matplotlib.pyplot as plt
import numpy as np

from src.calibrate import calibrate_svi_ls
from src.decorators import diagnostic_report
from src.model import forward_model, svi_slope

logger = logging.getLogger(__name__)


@diagnostic_report("Rho Instability")
def diagnostic_instability(cfg):
    n_ticks = 100
    ks = np.linspace(
        cfg.data.strike_range[0], cfg.data.strike_range[1], cfg.data.n_points
    )
    true_params = [0.04, 0.1, -0.7, 0.0, 0.1]
    ls_rho_history = []

    bounds = (cfg.model.bounds.lower, cfg.model.bounds.upper)
    initial_guess = np.array(cfg.model.initial_guess)

    for i in range(n_ticks):
        market_w = forward_model(ks, *true_params) + np.random.normal(
            0, cfg.data.noise_level, len(ks)
        )
        fitted, _ = calibrate_svi_ls(ks, market_w, initial_guess, bounds)
        ls_rho_history.append(fitted[2])

    rho_std = np.std(ls_rho_history)
    fig = plt.figure(figsize=(12, 5))
    plt.plot(ls_rho_history, "ro-", markersize=3, alpha=0.6, label="LS Estimate (Rho)")
    plt.axhline(-0.7, color="green", linestyle="--", label="True Rho")
    plt.fill_between(
        range(n_ticks), -0.7 - rho_std, -0.7 + rho_std, color="green", alpha=0.1
    )
    plt.title(f"Diagnostic: Parameter Instability (Rho)\nStdDev: {rho_std:.4f}")
    plt.legend()
    return fig


@diagnostic_report("Vanna Jitter")
def diagnostic_vanna(cfg):
    n_ticks = 100
    ks_market = np.linspace(
        cfg.data.strike_range[0], cfg.data.strike_range[1], cfg.data.n_points
    )
    true_params = [0.04, 0.1, -0.7, 0.0, 0.1]
    ls_vanna_history = []
    true_vanna = svi_slope(0.0, *true_params)

    bounds = (cfg.model.bounds.lower, cfg.model.bounds.upper)
    initial_guess = np.array(cfg.model.initial_guess)

    for _ in range(n_ticks):
        market_w = forward_model(ks_market, *true_params) + np.random.normal(
            0, cfg.data.noise_level, cfg.data.n_points
        )
        fitted, _ = calibrate_svi_ls(ks_market, market_w, initial_guess, bounds)
        ls_vanna_history.append(svi_slope(0.0, *fitted))

    vanna_vol = np.std(ls_vanna_history)
    fig = plt.figure(figsize=(12, 5))
    plt.plot(ls_vanna_history, color="crimson", label="LS ATM Vanna (dw/dk)", alpha=0.8)
    plt.axhline(true_vanna, color="black", linestyle="--", label="True Vanna")
    plt.fill_between(
        range(n_ticks),
        true_vanna - vanna_vol,
        true_vanna + vanna_vol,
        color="crimson",
        alpha=0.1,
    )
    plt.title("Risk Diagnostic: Vanna Instability")
    plt.legend()
    return fig


@diagnostic_report("Flat Valley")
def diagnostic_cost_surface(cfg):
    ks = np.linspace(
        cfg.data.strike_range[0], cfg.data.strike_range[1], cfg.data.n_points
    )
    true_params = [0.04, 0.1, -0.7, 0.0, 0.1]
    market_w = forward_model(ks, *true_params) + np.random.normal(
        0, cfg.data.noise_level, len(ks)
    )

    rhos = np.linspace(-0.99, 0.99, 100)
    errors = [
        np.sum(
            (
                market_w
                - forward_model(
                    ks,
                    true_params[0],
                    true_params[1],
                    r,
                    true_params[3],
                    true_params[4],
                )
            )
            ** 2
        )
        for r in rhos
    ]

    fig = plt.figure(figsize=(10, 5))
    plt.plot(rhos, errors, color="purple", lw=2)
    plt.axvline(-0.7, color="green", linestyle="--", label="True Rho")
    plt.title("Cost Function Sensitivity: The 'Flat Valley' Problem")
    plt.yscale("log")
    plt.xlabel("Rho Value")
    plt.ylabel("SSE (Log Scale)")
    plt.grid(True, alpha=0.2)
    return fig
