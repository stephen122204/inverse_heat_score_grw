"""
scores.py — Score function evaluation for the reverse-time SDE.

Manuscript labels (draft): the estimators smoothed_log_score,
log_density_fd_score, and direct_kde_score are `eq:smoothed_log`,
`eq:fd_score`, and `eq:direct_kde_score` (`sec:score_estimation`).

The score is:
    s(x, t) = partial_x log u(x, t) = u_x(x, t) / u(x, t)

Two variants are provided:

1. EXACT score (closed-form Gaussian formula; labeled "oracle" in method names):
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
# Exact score
# ---------------------------------------------------------------------------

def oracle_score(positions: np.ndarray, t_phys: float, cfg: Config) -> np.ndarray:
    """
    Exact score s(x, t_phys) = partial_x log u(x, t_phys) = u_x / u.

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
        raise ValueError(f"Unknown IC type for exact score: {ic.type!r}")


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


# ---------------------------------------------------------------------------
# Direct KDE derivative score (particle-native, no finite differences)
# ---------------------------------------------------------------------------

def direct_kde_score(
    x_eval: np.ndarray,
    positions: np.ndarray,
    weights: np.ndarray,
    bandwidth: float,
    epsilon: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """
    Compute the score s = partial_x log u at x_eval directly from the
    Gaussian KDE representation, without finite-difference grid operations.

    For a weighted Gaussian KDE:
        u(x) = sum_i w_i * K_h(x - X_i)
        K_h(z) = (1/(h*sqrt(2*pi))) * exp(-z^2 / (2*h^2))
        K_h'(z) = -(z / h^2) * K_h(z)

    so:
        u_x(x) = sum_i w_i * K_h'(x - X_i) = -sum_i w_i * ((x-X_i)/h^2) * K_h(x-X_i)

    Score:
        s(x) = u_x(x) / (u(x) + epsilon)

    This avoids the grid-ratio / finite-difference path entirely:
      - No np.gradient amplifying high-frequency noise.
      - No interpolation from grid to particle positions.
      - Score is evaluated directly at any x (particle positions or grid).

    Memory: for N particles × M eval points, allocates one (M, N) matrix.
    For N=10,000 and M=800 this is ~64 MB; chunked if needed.

    Parameters
    ----------
    x_eval    : (M,) evaluation points (particle positions or x_grid).
    positions : (N,) particle positions.
    weights   : (N,) particle weights (mass per particle = total_mass / N).
    bandwidth : Gaussian KDE bandwidth h (in x-units).
    epsilon   : denominator regularization floor.

    Returns
    -------
    scores : (M,) score at each eval point.
    diag   : dict with diagnostics:
               density_at_eval  : u(x_eval)
               numerator        : u_x(x_eval)
               denominator      : u(x_eval) + epsilon
               max_abs_score    : max|s| over finite values
               score_mean       : mean s over finite values
               score_std        : std s over finite values
               n_nonfinite      : number of non-finite score values
               epsilon_used     : epsilon argument (possibly 0)
               bandwidth_used   : bandwidth argument
    """
    M = len(x_eval)
    N = len(positions)

    # Chunk size chosen to stay within ~256 MB: M*N*8 bytes per array,
    # we need 3 arrays → limit chunk to 256 MB / (3 * 8) / M elements.
    max_chunk = max(1, int(256 * 1024 * 1024 / (3 * 8 * M)))

    u_vals = np.zeros(M, dtype=np.float64)
    ux_vals = np.zeros(M, dtype=np.float64)

    h = bandwidth
    h2 = h * h
    norm = 1.0 / (h * np.sqrt(2.0 * np.pi))

    for start in range(0, N, max_chunk):
        end = min(start + max_chunk, N)
        pos_chunk = positions[start:end]    # (chunk,)
        w_chunk = weights[start:end]        # (chunk,)

        diff = x_eval[:, None] - pos_chunk[None, :]   # (M, chunk)
        kern = norm * np.exp(-0.5 * (diff / h) ** 2)  # (M, chunk)  K_h
        kern_prime = -(diff / h2) * kern               # (M, chunk)  K_h'

        u_vals  += kern @ w_chunk       # (M,)
        ux_vals += kern_prime @ w_chunk  # (M,)

    denom = u_vals + epsilon
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = np.where(denom > 0.0, ux_vals / denom, np.nan)

    finite_mask = np.isfinite(scores)
    n_nonfinite = int(np.sum(~finite_mask))
    finite_scores = scores[finite_mask]

    diag = {
        "density_at_eval": u_vals,
        "numerator": ux_vals,
        "denominator": denom,
        "max_abs_score": float(np.max(np.abs(finite_scores))) if len(finite_scores) else float("nan"),
        "score_mean": float(np.mean(finite_scores)) if len(finite_scores) else float("nan"),
        "score_std": float(np.std(finite_scores)) if len(finite_scores) else float("nan"),
        "n_nonfinite": n_nonfinite,
        "epsilon_used": epsilon,
        "bandwidth_used": bandwidth,
    }
    return scores, diag


# ---------------------------------------------------------------------------
# Log-density finite-difference score
# ---------------------------------------------------------------------------

def log_density_fd_score(
    x_eval: np.ndarray,
    u_grid: np.ndarray,
    x_grid: np.ndarray,
    epsilon: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """
    Estimate score as gradient of log(u + epsilon) on the grid, then
    interpolate to x_eval.

    s(x) = partial_x log(u + epsilon) = u_x / (u + epsilon)

    This is mathematically the same as the fd_grid_ratio path but takes
    the gradient of log rather than dividing gradient(u) by u.  The two
    are numerically equivalent for smooth u; differ near zeros due to
    float arithmetic ordering.

    Returns
    -------
    scores : (len(x_eval),) score at eval points.
    diag   : dict with density_at_grid, log_u, log_grad, max_abs_score,
             n_nonfinite, epsilon_used.
    """
    dx = x_grid[1] - x_grid[0]
    log_u = np.log(np.maximum(u_grid + epsilon, 1e-300))
    s_grid = np.gradient(log_u, dx)

    scores = np.interp(x_eval, x_grid, s_grid)
    finite_mask = np.isfinite(scores)
    n_nonfinite = int(np.sum(~finite_mask))

    diag = {
        "density_at_grid": u_grid,
        "log_u": log_u,
        "log_grad": s_grid,
        "max_abs_score": float(np.max(np.abs(scores[finite_mask]))) if np.any(finite_mask) else float("nan"),
        "n_nonfinite": n_nonfinite,
        "epsilon_used": epsilon,
    }
    return scores, diag


# ---------------------------------------------------------------------------
# Smoothed-log score (Gaussian pre-smooth before log/gradient)
# ---------------------------------------------------------------------------

def smoothed_log_score(
    x_eval: np.ndarray,
    u_grid: np.ndarray,
    x_grid: np.ndarray,
    smooth_sigma: float,
    epsilon: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """
    Smooth u with a Gaussian filter, then compute log-derivative score.

    s(x) = partial_x log(smooth(u) + epsilon)

    smooth_sigma is the Gaussian smoothing sigma in x-units.
    This is the particle analogue of spectral truncation: it suppresses
    high-frequency noise in the reconstructed density before differentiating.

    Uses scipy.ndimage.gaussian_filter1d.

    Returns
    -------
    scores : (len(x_eval),) score at eval points.
    diag   : dict with u_smoothed, log_u, log_grad, smooth_sigma_used,
             max_abs_score, n_nonfinite, epsilon_used.
    """
    from scipy.ndimage import gaussian_filter1d
    dx = x_grid[1] - x_grid[0]
    sigma_pixels = smooth_sigma / dx
    u_smooth = gaussian_filter1d(u_grid, sigma=sigma_pixels, mode="nearest")
    u_smooth = np.maximum(u_smooth, 0.0)

    log_u = np.log(np.maximum(u_smooth + epsilon, 1e-300))
    s_grid = np.gradient(log_u, dx)

    scores = np.interp(x_eval, x_grid, s_grid)
    finite_mask = np.isfinite(scores)
    n_nonfinite = int(np.sum(~finite_mask))

    diag = {
        "u_smoothed": u_smooth,
        "log_u": log_u,
        "log_grad": s_grid,
        "smooth_sigma_used": smooth_sigma,
        "max_abs_score": float(np.max(np.abs(scores[finite_mask]))) if np.any(finite_mask) else float("nan"),
        "n_nonfinite": n_nonfinite,
        "epsilon_used": epsilon,
    }
    return scores, diag
