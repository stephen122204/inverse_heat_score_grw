"""Mass-preserving Neumann Gaussian kernels on a bounded interval.

For ``[a,b]`` with ``L=b-a``, the reflected Gaussian/Neumann heat kernel is

    K_h(x,y) = 1/L + (2/L) sum_{k>=1} exp(-(k*pi*h/L)^2/2)
                                      cos(k*pi*(x-a)/L)
                                      cos(k*pi*(y-a)/L).

The cosine representation has three useful numerical properties: its integral
in ``x`` is exactly one through the constant mode, its ``x`` derivative is
zero at both walls term by term, and a weighted particle KDE can be evaluated
in O(K(N+M)) operations rather than by forming an M-by-N kernel matrix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NeumannKDEDiagnostics:
    n_modes: int
    omitted_multiplier_bound: float
    total_mass: float
    min_density: float
    max_density: float


def neumann_mode_count(
    bandwidth: float,
    length: float,
    tolerance: float = 1e-14,
    max_modes: int = 100_000,
) -> int:
    """Return K so the first omitted heat-kernel multiplier is <= tolerance."""
    if not math.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError("bandwidth must be finite and positive")
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("domain length must be finite and positive")
    if not math.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
        raise ValueError("tolerance must lie strictly between zero and one")
    scale = math.pi * bandwidth / length
    # Choose K with exp(-0.5*((K+1)*scale)^2) <= tolerance.
    threshold = math.sqrt(2.0 * math.log(1.0 / tolerance)) / scale
    n_modes = max(0, math.ceil(threshold) - 1)
    if n_modes > max_modes:
        raise ValueError(
            f"Neumann kernel requires {n_modes} modes, above max_modes={max_modes}; "
            "increase the bandwidth or the explicit safety limit"
        )
    return n_modes


def _validated_inputs(
    x_eval: np.ndarray,
    positions: np.ndarray,
    weights: np.ndarray,
    x_min: float,
    x_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    x = np.asarray(x_eval, dtype=float)
    pos = np.asarray(positions, dtype=float)
    w = np.asarray(weights, dtype=float)
    if x.ndim != 1 or pos.ndim != 1 or w.ndim != 1:
        raise ValueError("x_eval, positions, and weights must be one-dimensional")
    if pos.shape != w.shape:
        raise ValueError("positions and weights must have the same shape")
    if pos.size == 0:
        raise ValueError("at least one particle is required")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(pos)) and np.all(np.isfinite(w))):
        raise ValueError("kernel inputs must be finite")
    if np.any(w < 0.0):
        raise ValueError("Neumann density-particle weights must be nonnegative")
    if not (math.isfinite(x_min) and math.isfinite(x_max) and x_max > x_min):
        raise ValueError("x_max must be greater than x_min")
    slack = 64.0 * np.finfo(float).eps * max(1.0, abs(x_min), abs(x_max))
    if np.any(pos < x_min - slack) or np.any(pos > x_max + slack):
        raise ValueError("particle positions must lie inside the closed domain")
    if np.any(x < x_min - slack) or np.any(x > x_max + slack):
        raise ValueError("evaluation points must lie inside the closed domain")
    return x, pos, w, float(x_max - x_min)


def neumann_kde_density_derivative(
    x_eval: np.ndarray,
    positions: np.ndarray,
    weights: np.ndarray,
    x_min: float,
    x_max: float,
    bandwidth: float,
    *,
    tolerance: float = 1e-14,
    chunk_size: int = 8192,
) -> tuple[np.ndarray, np.ndarray, NeumannKDEDiagnostics]:
    """Evaluate a weighted Neumann KDE and its analytic x derivative."""
    x, pos, w, length = _validated_inputs(
        x_eval, positions, weights, x_min, x_max
    )
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    n_modes = neumann_mode_count(bandwidth, length, tolerance)
    total_mass = float(np.sum(w))
    density = np.full(x.shape, total_mass / length, dtype=float)
    derivative = np.zeros(x.shape, dtype=float)

    if n_modes:
        modes = np.arange(1, n_modes + 1, dtype=float)
        wave_numbers = math.pi * modes / length
        damping = np.exp(-0.5 * (bandwidth * wave_numbers) ** 2)
        coefficients = np.zeros(n_modes, dtype=float)
        theta_pos = math.pi * (pos - x_min) / length
        for start in range(0, pos.size, chunk_size):
            stop = min(start + chunk_size, pos.size)
            basis = np.cos(theta_pos[start:stop, None] * modes[None, :])
            coefficients += basis.T @ w[start:stop]

        theta_eval = math.pi * (x - x_min) / length
        weighted = damping * coefficients
        density += (2.0 / length) * (
            np.cos(theta_eval[:, None] * modes[None, :]) @ weighted
        )
        derivative -= (2.0 / length) * (
            np.sin(theta_eval[:, None] * modes[None, :])
            @ (wave_numbers * weighted)
        )
        omitted = float(
            math.exp(-0.5 * (bandwidth * math.pi * (n_modes + 1) / length) ** 2)
        )
    else:
        omitted = float(
            math.exp(-0.5 * (bandwidth * math.pi / length) ** 2)
        )

    diagnostics = NeumannKDEDiagnostics(
        n_modes=n_modes,
        omitted_multiplier_bound=omitted,
        total_mass=total_mass,
        min_density=float(np.min(density)) if density.size else float("nan"),
        max_density=float(np.max(density)) if density.size else float("nan"),
    )
    return density, derivative, diagnostics


def neumann_kde_score(
    x_eval: np.ndarray,
    positions: np.ndarray,
    weights: np.ndarray,
    x_min: float,
    x_max: float,
    bandwidth: float,
    epsilon: float = 0.0,
    *,
    tolerance: float = 1e-14,
    chunk_size: int = 8192,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, NeumannKDEDiagnostics]:
    """Evaluate density, derivative, and d/dx log(density + epsilon)."""
    if not math.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("epsilon must be finite and nonnegative")
    density, derivative, diagnostics = neumann_kde_density_derivative(
        x_eval,
        positions,
        weights,
        x_min,
        x_max,
        bandwidth,
        tolerance=tolerance,
        chunk_size=chunk_size,
    )
    denominator = density + epsilon
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.where(denominator > 0.0, derivative / denominator, np.nan)
    return density, derivative, score, diagnostics

