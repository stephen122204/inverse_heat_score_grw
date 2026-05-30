"""
metrics.py — Quantitative metrics for evaluating backward GRW reconstructions.

All metrics use the discrete L2 norm with grid spacing dx:
    L2_h(f) = sqrt( dx * sum(f_i^2) )

Metrics computed per method:
  - l2_error              : L2_h(candidate - true_u0)
  - relative_l2           : l2_error / L2_h(true_u0)
  - linf_error            : max |candidate - true_u0|
  - peak_value            : max(candidate)
  - peak_ratio            : max(candidate) / max(true_u0)
  - peak_location         : argmax location
  - peak_width_fwhm       : FWHM of candidate
  - total_variation       : sum |candidate[i+1] - candidate[i]|
  - forward_consistency_l2: L2_h(forward_solve(candidate) - observed_final)
  - A_fit, mu_fit, sigma_fit, fit_success, fit_rmse : Gaussian fit to candidate
  - sigma_moment, width_moment : second-moment width
  - mass_candidate, mass_true, mass_error, mass_rel_error : integral conservation
  - max_abs_score_final   : max |score| at last completed step
  - max_score_error_L2    : max over steps of score L2 error vs oracle
  - runtime_seconds       : wall-clock time for integration

The forward consistency diagnostic uses DCT-based exact heat diffusion.
This is the one place DCT is allowed; it is diagnostic only, not part of
the inverse method.

This module intentionally does NOT apply regularization.
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
    Used ONLY for the forward-consistency diagnostic, not the inverse method.
    Neumann boundary conditions (consistent with reflecting-boundary GRW).
    """
    from scipy.fft import dct, idct

    alpha = cfg.heat.alpha
    T = cfg.heat.T
    N = len(u)
    L = cfg.domain.x_max - cfg.domain.x_min

    c = dct(u, type=2, norm="ortho")
    k = np.arange(N)
    lam = (np.pi * k / L) ** 2
    decay = np.exp(-alpha * T * lam)
    c_diffused = c * decay
    return idct(c_diffused, type=2, norm="ortho")


# ---------------------------------------------------------------------------
# Peak width (FWHM)
# ---------------------------------------------------------------------------

def compute_fwhm(u: np.ndarray, x: np.ndarray) -> float:
    """Full-width at half maximum of u on grid x. Returns NaN if not found."""
    peak = np.max(u)
    half = 0.5 * peak
    above = u >= half
    if not np.any(above):
        return float("nan")
    idxs = np.where(above)[0]
    dx = x[1] - x[0]
    return float((idxs[-1] - idxs[0] + 1) * dx)


# ---------------------------------------------------------------------------
# Gaussian fit
# ---------------------------------------------------------------------------

def fit_gaussian(
    x: np.ndarray,
    u: np.ndarray,
    cfg: Config,
) -> tuple[float, float, float, bool, float]:
    """
    Fit u(x) ~ A * exp(-(x-mu)^2 / (2*sigma^2)) via scipy curve_fit.

    Returns (A_fit, mu_fit, sigma_fit, fit_success, fit_rmse).
    If fit fails, returns (nan, nan, nan, False, nan).
    """
    from scipy.optimize import curve_fit

    def gaussian(x, A, mu, sigma):
        return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    # Safe initial guesses
    A0 = float(np.max(u)) if np.max(u) > 0 else 1.0
    mu0 = float(x[np.argmax(u)])
    u_pos = np.maximum(u, 0.0)
    mass = np.trapz(u_pos, x)
    if mass > 0:
        mean = np.trapz(x * u_pos, x) / mass
        var = np.trapz((x - mean) ** 2 * u_pos, x) / mass
        sigma0_g = float(np.sqrt(var)) if var > 0 else 0.08
    else:
        sigma0_g = 0.08

    x_min = cfg.domain.x_min
    x_max = cfg.domain.x_max

    try:
        popt, _ = curve_fit(
            gaussian, x, u,
            p0=[A0, mu0, sigma0_g],
            bounds=([0.0, x_min, 1e-4], [np.inf, x_max, x_max - x_min]),
            maxfev=10000,
        )
        A_fit, mu_fit, sigma_fit = popt
        residuals = u - gaussian(x, *popt)
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        return float(A_fit), float(mu_fit), float(sigma_fit), True, rmse
    except Exception:
        return float("nan"), float("nan"), float("nan"), False, float("nan")


# ---------------------------------------------------------------------------
# Second-moment width
# ---------------------------------------------------------------------------

def compute_moment_width(u: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """
    Mass-weighted second-moment width.
    Returns (sigma_moment, width_moment) where width_moment = 2 * sigma_moment.
    """
    u_pos = np.maximum(u, 0.0)
    mass = np.trapz(u_pos, x)
    if mass <= 0:
        return float("nan"), float("nan")
    mean = np.trapz(x * u_pos, x) / mass
    var = np.trapz((x - mean) ** 2 * u_pos, x) / mass
    sigma_m = float(np.sqrt(var)) if var > 0 else 0.0
    return sigma_m, 2.0 * sigma_m


# ---------------------------------------------------------------------------
# Metric dataclass
# ---------------------------------------------------------------------------

@dataclass
class MethodMetrics:
    method_name: str
    completed: bool
    failure_step: Optional[int]
    failure_msg: str
    # L2 family
    l2_error: float
    relative_l2: float
    linf_error: float
    # Peak
    peak_value: float
    peak_ratio: float
    peak_location: float
    # Width
    peak_width_fwhm: float
    sigma_moment: float
    width_moment: float
    # Gaussian fit
    A_fit: float
    mu_fit: float
    sigma_fit: float
    fit_success: bool
    fit_rmse: float
    # TV and forward consistency
    total_variation: float
    forward_consistency_l2: float
    # Mass conservation
    mass_candidate: float
    mass_true: float
    mass_error: float
    mass_rel_error: float
    # Score diagnostics (final step summary)
    max_abs_score_final: float
    max_score_error_L2: float
    # Timing
    runtime_seconds: float
    # --- Optional extended fields (Phase 5+) ---
    # Wasserstein-1 distance between candidate and true_u0 (treated as densities)
    wasserstein: float = float("nan")
    # Regularization info
    regularization_enabled: bool = False
    epsilon_value: float = float("nan")
    clipping_enabled: bool = False
    max_abs_score_clip: float = float("nan")
    n_denom_below_eps_total: int = 0
    n_clipped_total: int = 0
    # Ratio: this method's rel_L2 / spectral_cutoff rel_L2 (for particle methods)
    particle_rel_L2_over_spectral: float = float("nan")
    # Method category: "particle", "baseline_spectral", "baseline_tikhonov"
    method_category: str = "particle"
    # Score estimator type label for CSV output
    # Values: "position_ratio_raw", "grid_ratio_raw", "grid_ratio_epsilon",
    #         "oracle", "none" (naive), or "" (unset / baseline)
    score_estimator_type: str = ""


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
    If the run failed, metrics are computed on the best available candidate.
    """
    dx = x_grid[1] - x_grid[0]
    candidate = result.candidate

    # L2 family
    diff = candidate - true_u0
    l2_err = float(np.sqrt(dx * np.sum(diff ** 2)))
    l2_true = float(np.sqrt(dx * np.sum(true_u0 ** 2)))
    rel_l2 = l2_err / l2_true if l2_true > 0 else float("nan")
    linf_err = float(np.max(np.abs(diff)))

    # Peak
    true_peak = float(np.max(true_u0))
    peak_val = float(np.max(candidate))
    peak_ratio = peak_val / true_peak if true_peak > 0 else float("nan")
    peak_loc = float(x_grid[np.argmax(candidate)])

    # Width
    fwhm = compute_fwhm(candidate, x_grid)
    sigma_m, width_m = compute_moment_width(candidate, x_grid)

    # Gaussian fit
    A_fit, mu_fit, sigma_fit, fit_ok, fit_rmse = fit_gaussian(x_grid, candidate, cfg)

    # TV
    tv = float(np.sum(np.abs(np.diff(candidate))))

    # Forward consistency
    candidate_forward = forward_heat_solve_dct(candidate, x_grid, cfg)
    fwd_diff = candidate_forward - observed_final
    fwd_l2 = float(np.sqrt(dx * np.sum(fwd_diff ** 2)))

    # Mass conservation
    mass_cand = float(np.trapz(candidate, x_grid))
    mass_true_val = float(np.trapz(true_u0, x_grid))
    mass_err = mass_cand - mass_true_val
    mass_rel_err = mass_err / mass_true_val if mass_true_val > 0 else float("nan")

    # Score diagnostics
    max_abs_score_final = (
        result.score_max_abs[-1]
        if result.score_max_abs and np.isfinite(result.score_max_abs[-1])
        else float("nan")
    )
    if result.score_L2_error_vs_oracle:
        finite_errs = [v for v in result.score_L2_error_vs_oracle if np.isfinite(v)]
        max_score_err_L2 = float(max(finite_errs)) if finite_errs else float("nan")
    else:
        max_score_err_L2 = float("nan")

    return MethodMetrics(
        method_name=result.method_name,
        completed=result.completed,
        failure_step=result.failure_step,
        failure_msg=result.failure_msg,
        l2_error=l2_err,
        relative_l2=rel_l2,
        linf_error=linf_err,
        peak_value=peak_val,
        peak_ratio=peak_ratio,
        peak_location=peak_loc,
        peak_width_fwhm=fwhm,
        sigma_moment=sigma_m,
        width_moment=width_m,
        A_fit=A_fit,
        mu_fit=mu_fit,
        sigma_fit=sigma_fit,
        fit_success=fit_ok,
        fit_rmse=fit_rmse,
        total_variation=tv,
        forward_consistency_l2=fwd_l2,
        mass_candidate=mass_cand,
        mass_true=mass_true_val,
        mass_error=mass_err,
        mass_rel_error=mass_rel_err,
        max_abs_score_final=max_abs_score_final,
        max_score_error_L2=max_score_err_L2,
        runtime_seconds=result.runtime_seconds,
        # Regularization diagnostics (from MethodResult extended fields)
        epsilon_value=getattr(result, "epsilon_used", float("nan")),
        n_denom_below_eps_total=int(sum(getattr(result, "n_denominator_below_epsilon", []))),
        n_clipped_total=int(sum(getattr(result, "n_clipped_scores", []))),
        score_estimator_type=getattr(result, "score_estimator_type", ""),
    )


# ---------------------------------------------------------------------------
# Wasserstein-1 helper
# ---------------------------------------------------------------------------

def compute_wasserstein(
    candidate: np.ndarray,
    true_u0: np.ndarray,
    x_grid: np.ndarray,
) -> float:
    """
    Compute the Wasserstein-1 (Earth Mover's) distance between the candidate
    and true_u0 treated as nonnegative distributions on x_grid.
    Both are normalized to unit mass before comparison.
    Returns NaN if either distribution has zero mass.
    """
    try:
        from scipy.stats import wasserstein_distance
    except ImportError:
        return float("nan")

    c = np.maximum(candidate, 0.0)
    t = np.maximum(true_u0, 0.0)
    # Guard against overflow (large k_cut_mult can blow up spectral reconstructions)
    if not (np.all(np.isfinite(c)) and np.all(np.isfinite(t))):
        return float("nan")
    mc = float(np.trapz(c, x_grid))
    mt = float(np.trapz(t, x_grid))
    if not (np.isfinite(mc) and mc > 0.0 and np.isfinite(mt) and mt > 0.0):
        return float("nan")
    c_norm = c / mc
    t_norm = t / mt
    return float(wasserstein_distance(x_grid, x_grid, c_norm, t_norm))


# ---------------------------------------------------------------------------
# Baseline metrics helper (spectral / Tikhonov)
# ---------------------------------------------------------------------------

def compute_baseline_metrics(
    candidate: np.ndarray,
    true_u0: np.ndarray,
    observed_final: np.ndarray,
    x_grid: np.ndarray,
    cfg: "Config",
    method_name: str,
    method_category: str = "baseline_spectral",
) -> MethodMetrics:
    """
    Compute metrics for a deterministic baseline (spectral cutoff, Tikhonov).
    No MethodResult needed — takes the candidate array directly.
    """
    from .methods import MethodResult
    # Wrap candidate in a minimal MethodResult so compute_metrics can be reused
    fake_result = MethodResult(
        method_name=method_name,
        completed=True,
        failure_step=None,
        failure_msg="",
        candidate=candidate,
    )
    m = compute_metrics(fake_result, true_u0, observed_final, x_grid, cfg)
    m.method_category = method_category
    return m
