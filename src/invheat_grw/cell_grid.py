"""cell_grid.py — the prospective cell-centered grid contract.

This module defines the discretization the corrected numerical method will
use, ahead of any production wiring:

    x_j = x_min + (j + 1/2) dx,   dx = L / m,   j = 0, ..., m-1,
    k_n = n pi / L,               n = 0, ..., m-1,

with the midpoint mass dx * sum(u_j), the midpoint norm sqrt(dx * sum u_j^2),
and DCT-II (orthonormal) as the exactly matched cosine transform: the DCT-II
basis vectors are precisely cos(k_n (x_j - x_min)) sampled at the cell
centers, so a cosine pseudospectral heat solve applies the exact continuum
multiplier exp(-alpha k_n^2 t) to each discrete mode.

Nothing in the production paths imports this module yet; the contract is
tested standalone so the later grid migration only rewires callers.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.fft import dct, idct


def _validate_domain(x_min: float, x_max: float, m: int) -> float:
    if not (math.isfinite(x_min) and math.isfinite(x_max) and x_max > x_min):
        raise ValueError("x_max must be greater than x_min and both finite")
    if m < 1:
        raise ValueError("the grid needs at least one cell")
    return float(x_max - x_min)


def cell_spacing(x_min: float, x_max: float, m: int) -> float:
    """dx = L / m for m cells on [x_min, x_max]."""
    return _validate_domain(x_min, x_max, m) / m


def cell_centers(x_min: float, x_max: float, m: int) -> np.ndarray:
    """Cell-center coordinates x_min + (j + 1/2) dx, j = 0..m-1."""
    dx = cell_spacing(x_min, x_max, m)
    return x_min + (np.arange(m) + 0.5) * dx


def midpoint_mass(u: np.ndarray, dx: float) -> float:
    """Discrete mass dx * sum(u_j) of a cell-centered field."""
    return float(dx * np.sum(u))


def midpoint_norm(u: np.ndarray, dx: float) -> float:
    """Discrete L2 norm sqrt(dx * sum u_j^2) of a cell-centered field."""
    return float(np.sqrt(dx * np.sum(np.asarray(u, dtype=float) ** 2)))


def wave_numbers(m: int, length: float) -> np.ndarray:
    """Cosine wavenumbers k_n = n pi / L for n = 0..m-1."""
    if not (math.isfinite(length) and length > 0.0):
        raise ValueError("length must be finite and positive")
    return np.arange(m) * math.pi / length


def dct2_forward(u: np.ndarray) -> np.ndarray:
    """Orthonormal DCT-II of cell-centered samples."""
    return dct(np.asarray(u, dtype=float), type=2, norm="ortho")


def dct2_inverse(c: np.ndarray) -> np.ndarray:
    """Inverse of dct2_forward."""
    return idct(np.asarray(c, dtype=float), type=2, norm="ortho")


def propagate_heat(u: np.ndarray, length: float, alpha: float, t: float) -> np.ndarray:
    """Cosine pseudospectral heat propagation of a cell-centered field.

    Applies the exact continuum multiplier exp(-alpha k_n^2 t) to each DCT-II
    mode; on cell centers this diagonalization is exact for every resolved
    mode, not an approximation.
    """
    if not (math.isfinite(alpha) and alpha >= 0.0 and math.isfinite(t) and t >= 0.0):
        raise ValueError("alpha and t must be finite and nonnegative")
    u = np.asarray(u, dtype=float)
    k = wave_numbers(len(u), length)
    return dct2_inverse(dct2_forward(u) * np.exp(-alpha * k ** 2 * t))


def cell_edge_quantile_positions(
    u: np.ndarray,
    x_min: float,
    x_max: float,
    n_particles: int,
) -> tuple[np.ndarray, float]:
    """Deterministic quantile particle positions for a cell-centered field.

    Each grid value is treated as the density of its cell, so the cell mass
    is m_j = max(u_j, 0) dx, the CDF is exact and piecewise linear at the
    cell edges, and each quantile target (i + 1/2)/N is inverted linearly
    within its cell.  This preserves the piecewise-constant mass
    representation instead of shifting quantiles toward cell centers.

    Returns (positions, total_mass); total_mass is the midpoint mass of the
    clipped field.  A nonpositive-mass field falls back to uniform placement
    with total mass zero, matching the endpoint-grid initializer's contract.
    """
    u = np.asarray(u, dtype=float)
    m = u.shape[0]
    dx = cell_spacing(x_min, x_max, m)
    if n_particles < 1:
        raise ValueError("n_particles must be at least one")
    cell_mass = np.maximum(u, 0.0) * dx
    total_mass = float(np.sum(cell_mass))
    edges = x_min + np.arange(m + 1) * dx
    if total_mass <= 0.0:
        return np.linspace(x_min, x_max, n_particles), 0.0
    cdf = np.concatenate([[0.0], np.cumsum(cell_mass)]) / total_mass
    targets = (np.arange(n_particles) + 0.5) / n_particles
    return np.interp(targets, cdf, edges), total_mass
