"""
scores.py — Score function evaluation for the reverse-time SDE.

The score is:
    s(x, t) = partial_x log u(x, t) = u_x(x, t) / u(x, t)

Two variants are provided:

1. ORACLE score (exact Gaussian formula):
   For u(x, t) = A * (sigma0 / sigma_t) * exp(-(x-mu)^2 / (2 sigma_t^2)):
       s_exact(x, t_phys) = -(x - mu) / sigma_t^2

   This requires knowledge of the true initial condition parameters.
   It is used as an idealized test of whether the score drift mechanism works.

2. ESTIMATED score (raw, unregularized):
   Reconstruct u from the current glob state, then numerically differentiate:
       s_est(x, t) = u_x_numerical(x) / u_reconstructed(x)

   If u is near zero, the score can blow up.  This is NOT silently fixed here.
   Instability (NaN, Inf, or |s| > threshold) is detected and reported.

Timing convention (backward integration):
    At reverse step k, elapsed reverse time tau = k * dt.
    The physical time is t_phys = T - tau.
    At start (k=0): t_phys = T.
    At end (k=n_steps): t_phys -> 0.
"""

from __future__ import annotations

import numpy as np
from .config import Config
from .globs import GlobState, reconstruct_field


# ---------------------------------------------------------------------------
# Oracle score
# ---------------------------------------------------------------------------

def oracle_score(positions: np.ndarray, t_phys: float, cfg: Config) -> np.ndarray:
    """
    Exact score for the Gaussian solution at physical time t_phys:
        s(x, t_phys) = -(x - mu) / sigma_t^2
    where sigma_t^2 = sigma0^2 + 2 * alpha * t_phys.

    Parameters
    ----------
    positions : glob x-coordinates at which to evaluate the score.
    t_phys    : physical time (T - tau), where tau = k * dt.
    cfg       : experiment config.

    Returns
    -------
    score values at each position.
    """
    ic = cfg.initial_condition
    alpha = cfg.heat.alpha

    if t_phys <= 0.0:
        # At t_phys = 0 the Gaussian has width sigma0; score still defined
        sigma_t2 = ic.sigma0 ** 2
    else:
        sigma_t2 = ic.sigma0 ** 2 + 2.0 * alpha * t_phys

    return -(positions - ic.mu) / sigma_t2


# ---------------------------------------------------------------------------
# Estimated score (raw, no regularization)
# ---------------------------------------------------------------------------

def estimated_score_raw(
    positions: np.ndarray,
    state: GlobState,
    x_grid: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, bool, str]:
    """
    Estimate the score s = u_x / u at each glob position using the current
    glob-reconstructed field.  No regularization is applied.

    Steps:
      1. Reconstruct u on the spatial grid from current glob state.
      2. Numerically differentiate u_x using central differences.
      3. Interpolate u and u_x to glob positions.
      4. Compute s = u_x / u pointwise.
      5. Check for NaN, Inf, or |s| > threshold.

    Returns
    -------
    scores    : array of score values (may contain NaN/Inf if unstable).
    is_stable : False if any NaN/Inf or |score| > threshold was detected.
    msg       : human-readable diagnostic message.
    """
    threshold = cfg.safety.score_abs_fail_threshold

    # Step 1: reconstruct u on grid
    u_grid = reconstruct_field(state, x_grid)

    # Step 2: numerical derivative u_x via central differences
    dx = x_grid[1] - x_grid[0]
    u_x_grid = np.gradient(u_grid, dx)

    # Step 3: interpolate to glob positions
    u_at_pos = np.interp(positions, x_grid, u_grid)
    u_x_at_pos = np.interp(positions, x_grid, u_x_grid)

    # Step 4: raw score = u_x / u  (no epsilon, no clipping)
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = np.where(u_at_pos != 0.0, u_x_at_pos / u_at_pos, np.nan)

    # Step 5: stability check
    has_nan = np.any(~np.isfinite(scores))
    has_large = np.any(np.abs(scores[np.isfinite(scores)]) > threshold) if np.any(np.isfinite(scores)) else False

    if has_nan:
        return scores, False, f"NaN/Inf in estimated score (u ≈ 0 at {np.sum(~np.isfinite(scores))} globs)"
    if has_large:
        max_s = np.max(np.abs(scores))
        return scores, False, f"Score magnitude {max_s:.3e} exceeds threshold {threshold:.3e}"

    return scores, True, "OK"
