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
    Exact oracle score s(x, t_phys) = partial_x log u(x, t_phys) = u_x / u.

    For Gaussian IC:
        s = -(x - mu) / sigma_t^2,  sigma_t^2 = sigma0^2 + 2*alpha*t_phys

    For Gaussian mixture IC:
        u = background + sum_i A_i(t)*exp(-(x-mu_i)^2/(2*sigma_it^2))
        u_x = sum_i A_i(t)*exp(...) * (-(x-mu_i)/sigma_it^2)
        s = u_x / u  (analytic, no numerical differentiation)
        Where u = 0 (background-only tail), score is set to 0.

    Parameters
    ----------
    positions : positions at which to evaluate the score.
    t_phys    : physical time T - tau (tau = k*dt, k = current backward step).
    cfg       : experiment config.
    """
    ic = cfg.initial_condition
    alpha = cfg.heat.alpha
    t = max(t_phys, 0.0)

    if ic.type == "gaussian":
        sigma_t2 = ic.sigma0 ** 2 + 2.0 * alpha * t
        return -(positions - ic.mu) / sigma_t2

    elif ic.type == "gaussian_mixture":
        # Analytically compute u and u_x at each position
        u = np.full_like(positions, ic.background, dtype=float)
        u_x = np.zeros_like(positions, dtype=float)
        for comp in ic.components:
            sigma_t2 = comp.sigma0 ** 2 + 2.0 * alpha * t
            sigma_t = np.sqrt(sigma_t2)
            A_t = comp.amplitude * (comp.sigma0 / sigma_t)
            gauss = A_t * np.exp(-0.5 * (positions - comp.mu) ** 2 / sigma_t2)
            u += gauss
            u_x += gauss * (-(positions - comp.mu) / sigma_t2)
        with np.errstate(divide="ignore", invalid="ignore"):
            scores = np.where(u > 0.0, u_x / u, 0.0)
        return scores

    else:
        raise ValueError(f"Unknown IC type for oracle score: {ic.type!r}")


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


# ---------------------------------------------------------------------------
# Estimated score (regularized)
# ---------------------------------------------------------------------------

def estimated_score_regularized(
    positions: np.ndarray,
    state: GlobState,
    x_grid: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, bool, str, dict]:
    """
    Regularized score estimate.  Uses cfg.regularization to apply:
      - Epsilon floor: denom = u + epsilon  (epsilon >= 0)
      - Score clipping: |s| clipped to max_abs_score
      - (Smoothing stub: not yet implemented)

    At epsilon=0 and clipping disabled, this score estimator is still NOT
    equivalent to estimated_score_raw.  This function computes the score
    on the reconstruction grid as u_x / (u + epsilon) and then interpolates
    the score to particle positions (divide-then-interp, grid-ratio path).
    estimated_score_raw interpolates u and u_x separately to particle
    positions and then divides (interp-then-divide, position-ratio path).
    Division and interpolation do not commute, so the two paths are
    distinct discretizations even when epsilon=0.

    INVARIANT: estimated_score_raw is NEVER modified.  This function is
    always separate.

    Parameters
    ----------
    positions : glob positions at which to evaluate the score.
    state     : current GlobState.
    x_grid    : spatial grid.
    cfg       : full Config (reads cfg.regularization).

    Returns
    -------
    scores     : regularized score at each glob position.
    is_stable  : False if NaN/Inf or magnitude > threshold.
    msg        : diagnostic string.
    diag       : dict with keys:
                   epsilon_used           (float)
                   n_denom_below_epsilon  (int)  -- grid pts where u < eps
                   n_clipped              (int)  -- grid pts where score clipped
                   max_abs_before_clip    (float)
                   max_abs_after          (float)
                   s_grid                 (np.ndarray) -- regularized score on x_grid
                   u_grid                 (np.ndarray) -- reconstructed u on x_grid
    """
    threshold = cfg.safety.score_abs_fail_threshold
    reg = cfg.regularization

    # Step 1: reconstruct u on grid
    u_grid = reconstruct_field(state, x_grid)

    # Step 2: numerical derivative
    dx = x_grid[1] - x_grid[0]
    u_x_grid = np.gradient(u_grid, dx)

    # Step 3: compute effective epsilon
    if reg.enabled and reg.epsilon_floor.enabled:
        if reg.epsilon_floor.scale_by_peak:
            peak_u = float(np.max(u_grid))
            eps = reg.epsilon_floor.value * max(peak_u, 0.0)
        else:
            eps = reg.epsilon_floor.value
    else:
        eps = 0.0

    # Count denominator values below epsilon (diagnostic)
    n_below_eps = int(np.sum(u_grid < eps)) if eps > 0.0 else 0

    # Step 4: regularized score on grid
    denom = u_grid + eps
    with np.errstate(divide="ignore", invalid="ignore"):
        s_grid = np.where(denom != 0.0, u_x_grid / denom, np.nan)

    # Step 5: clipping
    finite_mask = np.isfinite(s_grid)
    max_abs_before_clip = float(np.max(np.abs(s_grid[finite_mask]))) if np.any(finite_mask) else float("nan")
    n_clipped = 0
    if reg.enabled and reg.score_clipping.enabled:
        max_clip = reg.score_clipping.max_abs_score
        clip_mask = finite_mask & (np.abs(s_grid) > max_clip)
        n_clipped = int(np.sum(clip_mask))
        s_grid = np.clip(s_grid, -max_clip, max_clip)

    max_abs_after = float(np.max(np.abs(s_grid[np.isfinite(s_grid)]))) if np.any(np.isfinite(s_grid)) else float("nan")

    # Step 6: interpolate to positions
    scores = np.interp(positions, x_grid, s_grid)

    # Step 7: stability check — do NOT silently zero out NaN/Inf
    has_nan = np.any(~np.isfinite(scores))
    has_large = (np.any(np.abs(scores[np.isfinite(scores)]) > threshold)
                 if np.any(np.isfinite(scores)) else False)

    diag = {
        "epsilon_used": eps,
        "n_denom_below_epsilon": n_below_eps,
        "n_clipped": n_clipped,
        "max_abs_before_clip": max_abs_before_clip,
        "max_abs_after": max_abs_after,
        "s_grid": s_grid.copy(),
        "u_grid": u_grid.copy(),
    }

    if has_nan:
        n_nan = int(np.sum(~np.isfinite(scores)))
        return scores, False, f"NaN/Inf in regularized score ({n_nan} globs)", diag
    if has_large:
        max_s = float(np.max(np.abs(scores)))
        return scores, False, f"Regularized score magnitude {max_s:.3e} exceeds threshold {threshold:.3e}", diag

    return scores, True, "OK", diag
