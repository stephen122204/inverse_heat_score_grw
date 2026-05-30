"""
fields.py — Analytical field definitions for the heat equation.

For a Gaussian initial condition
    u0(x) = A * exp(-(x - mu)^2 / (2 sigma0^2))

the exact solution under free-space heat diffusion is another Gaussian:
    u(x, t) = A * (sigma0 / sigma_t) * exp(-(x - mu)^2 / (2 sigma_t^2))
where
    sigma_t^2 = sigma0^2 + 2 * alpha * t.

This module provides:
  - make_grid         : uniform 1-D spatial grid on [x_min, x_max]
  - true_u0           : exact initial Gaussian
  - exact_heat_solution : exact u(x, t) for any t >= 0
  - observed_final    : shorthand for exact_heat_solution at t = T
"""

from __future__ import annotations

import numpy as np
from .config import Config


def make_grid(cfg: Config) -> np.ndarray:
    """Return uniform grid with n_grid points on [x_min, x_max]."""
    return np.linspace(cfg.domain.x_min, cfg.domain.x_max, cfg.domain.n_grid)


def true_u0(x: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Exact initial Gaussian:
        u0(x) = amplitude * exp(-(x - mu)^2 / (2 sigma0^2))
    """
    ic = cfg.initial_condition
    return ic.amplitude * np.exp(-0.5 * ((x - ic.mu) / ic.sigma0) ** 2)


def exact_heat_solution(x: np.ndarray, t: float, cfg: Config) -> np.ndarray:
    """
    Exact Gaussian heat solution at time t:
        sigma_t^2 = sigma0^2 + 2 * alpha * t
        u(x, t) = amplitude * (sigma0 / sigma_t) * exp(-(x - mu)^2 / (2 sigma_t^2))

    Note: at t = 0 this reduces to true_u0.
    """
    ic = cfg.initial_condition
    alpha = cfg.heat.alpha
    sigma_t2 = ic.sigma0 ** 2 + 2.0 * alpha * t
    sigma_t = np.sqrt(sigma_t2)
    return ic.amplitude * (ic.sigma0 / sigma_t) * np.exp(-0.5 * ((x - ic.mu) ** 2) / sigma_t2)


def observed_final(x: np.ndarray, cfg: Config) -> np.ndarray:
    """
    The observed (forward-diffused) field at t = T.
    This is the starting point for all backward reconstructions.
    """
    return exact_heat_solution(x, cfg.heat.T, cfg)
