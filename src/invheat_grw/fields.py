"""
fields.py — Analytical field definitions for the heat equation.

For a Gaussian initial condition
    u0(x) = A * exp(-(x - mu)^2 / (2 sigma0^2))

the exact solution under free-space heat diffusion is another Gaussian:
    u(x, t) = A * (sigma0 / sigma_t) * exp(-(x - mu)^2 / (2 sigma_t^2))
where
    sigma_t^2 = sigma0^2 + 2 * alpha * t.

Manuscript labels (draft): the constant-coefficient problem is `eq:heat`;
these fields define the test cases of `sec:setup`.

This module provides:
  - make_grid         : uniform 1-D spatial grid on [x_min, x_max]
  - true_u0           : exact initial Gaussian
  - exact_heat_solution : exact u(x, t) for any t >= 0
  - observed_final    : shorthand for exact_heat_solution at t = T
"""

from __future__ import annotations

import numpy as np
from .config import Config

# The spatial discretization convention make_grid implements.  Frozen figure
# datasets and run manifests record this string; the live figure-freeze gate
# compares it against the archive it is asked to verify.
GRID_CONVENTION = "endpoint"


def make_grid(cfg: Config) -> np.ndarray:
    """Return uniform grid with n_grid points on [x_min, x_max]."""
    return np.linspace(cfg.domain.x_min, cfg.domain.x_max, cfg.domain.n_grid)


def true_u0(x: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Exact initial condition.
    - type=gaussian: A * exp(-(x-mu)^2 / (2*sigma0^2))
    - type=gaussian_mixture: background + sum_i A_i * exp(-(x-mu_i)^2 / (2*sigma_i0^2))
    """
    ic = cfg.initial_condition
    if ic.type == "gaussian":
        return ic.amplitude * np.exp(-0.5 * ((x - ic.mu) / ic.sigma0) ** 2)
    elif ic.type == "gaussian_mixture":
        u = np.full_like(x, ic.background, dtype=float)
        for comp in ic.components:
            u += comp.amplitude * np.exp(-0.5 * ((x - comp.mu) / comp.sigma0) ** 2)
        return u
    else:
        raise ValueError(f"Unknown IC type: {ic.type!r}")


def exact_heat_solution(x: np.ndarray, t: float, cfg: Config) -> np.ndarray:
    """
    Exact heat solution at time t.

    For Gaussian IC:
        sigma_t^2 = sigma0^2 + 2*alpha*t
        u(x,t) = amplitude*(sigma0/sigma_t)*exp(-(x-mu)^2/(2*sigma_t^2))

    For Gaussian mixture IC:
        Each component evolves independently as a Gaussian.
        Background stays constant.
        u(x,t) = background + sum_i A_i*(sigma_i0/sigma_it)*exp(-(x-mu_i)^2/(2*sigma_it^2))
    """
    ic = cfg.initial_condition
    alpha = cfg.heat.alpha
    if ic.type == "gaussian":
        sigma_t2 = ic.sigma0 ** 2 + 2.0 * alpha * t
        sigma_t = np.sqrt(sigma_t2)
        return ic.amplitude * (ic.sigma0 / sigma_t) * np.exp(-0.5 * ((x - ic.mu) ** 2) / sigma_t2)
    elif ic.type == "gaussian_mixture":
        u = np.full_like(x, ic.background, dtype=float)
        for comp in ic.components:
            sigma_t2 = comp.sigma0 ** 2 + 2.0 * alpha * t
            sigma_t = np.sqrt(sigma_t2)
            u += comp.amplitude * (comp.sigma0 / sigma_t) * np.exp(-0.5 * (x - comp.mu) ** 2 / sigma_t2)
        return u
    else:
        raise ValueError(f"Unknown IC type: {ic.type!r}")


def observed_final(x: np.ndarray, cfg: Config) -> np.ndarray:
    """
    The observed (forward-diffused) field at t = T.
    This is the starting point for all backward reconstructions.
    """
    return exact_heat_solution(x, cfg.heat.T, cfg)
