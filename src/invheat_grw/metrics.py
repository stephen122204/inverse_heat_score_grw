"""
metrics.py — Quantitative metrics for evaluating backward GRW reconstructions.

All metrics use the discrete L2 norm with grid spacing dx:
    L2_h(f) = sqrt( dx * sum(f_i^2) )

Metrics computed per method:
  - l2_error        : L2_h(candidate - true_u0)
  - peak_value      : max(candidate)
  - peak_location   : argmax location
  - peak_width      : width at half maximum (FWHM)
  - total_variation : sum |candidate[i+1] - candidate[i]|
  - forward_consistency_l2 : L2_h(forward_solve(candidate) - observed_final)

The forward consistency diagnostic uses DCT-based exact heat diffusion to
check whether the candidate (if run forward) reproduces the observation.
This is the one place DCT is allowed; it is a diagnostic only, not part
of the inverse method.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional
from .config import Config
from .methods import MethodResult


# ---------------------------------------------------------------------------
# Forward heat solve for diagnostics (DCT-based, exact for the grid)
# ---------------------------------------------------------------------------

def forward_heat_solve_dct(u: np.ndarray, x_grid: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Apply exact heat diffusion u -> u(., T) using DCT-II eigenfunctions on [x_min, x_max].

    This is used ONLY for the forward-consistency diagnostic.  It is NOT part
    of the inverse method.

    Uses Neumann boundary conditions (derivative = 0 at endpoints), consistent
    with reflecting-boundary GRW.
    """
    from scipy.fft import dct, idct

    alpha = cfg.heat.alpha
    T = cfg.heat.T
    N = len(u)
    L = cfg.domain.x_max - cfg.domain.x_min

    # DCT-II coefficients
    c = dct(u, type=2, norm="ortho")

    # Eigenvalues: lambda_k = (pi * k / L)^2 for k = 0, 1, ..., N-1
    k = np.arange(N)
    lam = (np.pi * k / L) ** 2
    decay = np.exp(-alpha * T * lam)

    c_diffused = c * decay
    return idct(c_diffused, type=2, norm="ortho")


# ---------------------------------------------------------------------------
# Peak width (FWHM)
# ---------------------------------------------------------------------------

def compute_fwhm(u: np.ndarray, x: np.ndarray) -> float:
    """
    Full-width at half maximum of u on grid x.
    Returns NaN if the half-maximum level is not crossed.
    """
    peak = np.max(u)
    half = 0.5 * peak
    above = u >= half

    if not np.any(above):
        return float("nan")

    # Find leftmost and rightmost crossing
    idxs = np.where(above)[0]
    left_idx = idxs[0]
    right_idx = idxs[-1]

    dx = x[1] - x[0]
    return float((right_idx - left_idx + 1) * dx)


# ---------------------------------------------------------------------------
# Metric dataclass
# ---------------------------------------------------------------------------

@dataclass
class MethodMetrics:
    method_name: str
    completed: bool
    failure_step: Optional[int]
    failure_msg: str
    l2_error: float
    peak_value: float
    peak_location: float
    peak_width_fwhm: float
    total_variation: float
    forward_consistency_l2: float


# ---------------------------------------------------------------------------
# Compute metrics for one result
# ---------------------------------------------------------------------------

def compute_metrics(
    result: MethodResult,
    true_u0: np.ndarray,
    observed_final: np.ndarray,
    x_grid: np.ndarray,
    cfg: Config,
) -> MethodMetrics:
    """
    Compute all metrics for a single MethodResult.

    If the run did not complete, metrics are still computed on the
    best-available candidate field.
    """
    dx = x_grid[1] - x_grid[0]
    candidate = result.candidate

    # L2 recovery error
    diff = candidate - true_u0
    l2_err = float(np.sqrt(dx * np.sum(diff ** 2)))

    # Peak
    peak_val = float(np.max(candidate))
    peak_loc = float(x_grid[np.argmax(candidate)])

    # FWHM
    fwhm = compute_fwhm(candidate, x_grid)

    # Total variation
    tv = float(np.sum(np.abs(np.diff(candidate))))

    # Forward consistency: run candidate forward to T, compare to observed_final
    candidate_forward = forward_heat_solve_dct(candidate, x_grid, cfg)
    fwd_diff = candidate_forward - observed_final
    fwd_l2 = float(np.sqrt(dx * np.sum(fwd_diff ** 2)))

    return MethodMetrics(
        method_name=result.method_name,
        completed=result.completed,
        failure_step=result.failure_step,
        failure_msg=result.failure_msg,
        l2_error=l2_err,
        peak_value=peak_val,
        peak_location=peak_loc,
        peak_width_fwhm=fwhm,
        total_variation=tv,
        forward_consistency_l2=fwd_l2,
    )
