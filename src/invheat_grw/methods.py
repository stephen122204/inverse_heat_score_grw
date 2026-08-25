"""
methods.py — Backward GRW integration methods.

Each method integrates glob positions backward from t = T to t = 0
by running n_steps of size dt.  After integration, the field is
reconstructed from the final glob positions.

Manuscript labels (draft): the naive update is `eq:naive` (`sec:naive`);
the probability-flow ODE update is `eq:pfode`; the reverse-time SDE update
is `eq:reverse_sde`; the density-particle method is Algorithm
`alg:density_particle` (`sec:density_formulation`).

The method names below use the label ``oracle`` for the exact analytic
score; the paper calls this the exact score.

METHODS IMPLEMENTED
-------------------

1. naive_backward
   Update: X_{k+1} = X_k - sqrt(2 alpha dt) xi_k
   This is NOT a valid inverse of forward Brownian motion.
   xi and -xi have the same Gaussian distribution, so this
   produces the same diffusive spreading, not a reversal.
   Expected result: fails to recover the sharp initial peak.

2. oracle_score_deterministic (probability-flow ODE)
   Update: X_{k+1} = X_k + alpha * s_exact(X_k, t_phys) * dt
   No noise term.  The score drift alone should move particles
   anti-diffusively toward the Gaussian peak.
   Coefficient is alpha (not 2*alpha) for the deterministic flow.

3. oracle_score_stochastic (reverse-time SDE)
   Update: X_{k+1} = X_k + 2 * alpha * s_exact(X_k, t_phys) * dt
                          + sqrt(2 alpha dt) xi_k
   Full reverse-time SDE with noise.
   Coefficient is 2*alpha for the stochastic drift.

4. estimated_score_deterministic_raw
   Same as oracle_score_deterministic but uses the raw estimated
   score from the current glob reconstruction.  May blow up.

5. estimated_score_stochastic_raw
   Same as oracle_score_stochastic but uses the raw estimated score.
   May blow up.

SAFETY
------
At each step, if:
  - any glob position is NaN/Inf, or
  - any score value is NaN/Inf, or
  - max |score| > score_abs_fail_threshold, or
  - max |position| > value_abs_fail_threshold
the run is marked UNSTABLE and integration halts.
The partial result up to the failure step is returned.
"""

from __future__ import annotations

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from .config import Config
from .globs import GlobState, field_to_globs, reconstruct_field, apply_reflecting_boundary
from .scores import (
    oracle_score,
    estimated_score_raw,
    estimated_score_regularized,
    direct_kde_score,
    log_density_fd_score,
    smoothed_log_score,
)

# numpy>=2.0 renamed np.trapz -> np.trapezoid (np.trapz removed in 2.x).
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class MethodResult:
    """
    Stores the outcome of one backward GRW integration run.

    Attributes
    ----------
    method_name   : identifier string.
    completed     : True if all n_steps finished without instability.
    failure_step  : step index at which instability was detected (or None).
    failure_msg   : human-readable reason for failure (or "").
    candidate     : reconstructed field on x_grid (or best available partial).
    step_snapshots: dict mapping step index -> reconstructed field (for plotting).
    score_max_abs : max |score| recorded at each step (for diagnostics).
    score_mean    : mean score at each step.
    score_std     : std of score at each step.
    final_positions : final glob positions.
    """
    method_name: str
    completed: bool
    failure_step: Optional[int]
    failure_msg: str
    candidate: np.ndarray
    step_snapshots: dict = field(default_factory=dict)
    score_max_abs: list = field(default_factory=list)
    score_mean: list = field(default_factory=list)
    score_std: list = field(default_factory=list)
    final_positions: Optional[np.ndarray] = None
    # Per-step score error vs the exact score (for estimated-score methods)
    score_L2_error_vs_oracle: list = field(default_factory=list)
    score_Linf_error_vs_oracle: list = field(default_factory=list)
    score_core_L2_error_vs_oracle: list = field(default_factory=list)
    score_core_Linf_error_vs_oracle: list = field(default_factory=list)
    # Score overlay snapshots: step -> (x_grid, s_est, s_oracle)
    score_overlay_snapshots: dict = field(default_factory=dict)
    # Steps where estimated score error was suspiciously exactly 0.0
    score_suspicious_steps: list = field(default_factory=list)
    # Wall-clock runtime in seconds
    runtime_seconds: float = 0.0
    # Regularization diagnostics (populated only for regularized methods)
    n_denominator_below_epsilon: list = field(default_factory=list)  # per step
    n_clipped_scores: list = field(default_factory=list)             # per step
    epsilon_used: float = 0.0  # effective epsilon (0 = raw/unregularized)
    # Score estimator type label for CSV output
    # Values: "position_ratio_raw", "grid_ratio_raw", "grid_ratio_epsilon",
    #         "oracle", "none" (naive), or "" (unset)
    score_estimator_type: str = ""
    # Density-particle method metadata (populated only for density-particle runs)
    n_particles: int = 0
    reconstruction_method: str = ""
    bandwidth_used: float = float("nan")
    bandwidth_factor: float = float("nan")
    step_zero_recon_error: float = float("nan")
    mass_initial: float = float("nan")
    # Per-step diagnostics for density-particle estimated methods
    mean_abs_score: list = field(default_factory=list)          # mean |score| at particle positions
    mass_per_step: list = field(default_factory=list)           # reconstructed mass per step
    epsilon_actual_per_step: list = field(default_factory=list) # actual epsilon per step (varies if scale_by_peak)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_positions(positions: np.ndarray, threshold: float) -> tuple[bool, str]:
    if not np.all(np.isfinite(positions)):
        return False, "NaN/Inf in glob positions"
    if np.any(np.abs(positions) > threshold):
        return False, f"Glob position magnitude > {threshold:.3e}"
    return True, "OK"


def _record_snapshot_steps(n_steps: int) -> set:
    """Steps at which to save field snapshots for plotting.
    Always include step 0, 1, 2, the midpoint, and the final step.
    """
    mid = n_steps // 2
    return {0, 1, 2, 5, 10, mid, n_steps}


def _quantile_init_particles(
    u_obs: np.ndarray,
    x_grid: np.ndarray,
    n_particles: int,
) -> tuple[np.ndarray, float]:
    """
    Initialize density particles from u_obs by deterministic quantile placement.

    Treats u_obs as a non-negative mass distribution.  Places n_particles at
    the quantile positions (i + 0.5) / n_particles for i = 0..n_particles-1,
    so the empirical CDF matches u_obs as closely as possible.

    Returns
    -------
    positions : (n_particles,) array of initial particle x-coordinates.
    total_mass : float, integral of u_obs over x_grid.
    """
    u_pos = np.maximum(u_obs, 0.0)
    dx = x_grid[1] - x_grid[0]
    total_mass = float(_trapz(u_pos, x_grid))  # type: ignore[attr-defined]
    if total_mass <= 0.0:
        # Fallback: uniform placement
        positions = np.linspace(x_grid[0], x_grid[-1], n_particles)
        return positions, 0.0

    # Trapezoidal CDF on x_grid: cdf[0]=0, cdf[-1]=total_mass (before normalizing)
    segments = 0.5 * (u_pos[:-1] + u_pos[1:]) * dx
    cdf = np.concatenate([[0.0], np.cumsum(segments)])
    cdf /= total_mass  # normalize to [0, 1]

    # Deterministic quantile inversion
    quantiles = (np.arange(n_particles) + 0.5) / n_particles
    positions = np.interp(quantiles, cdf, x_grid)
    return positions, total_mass


def _reconstruct_density_particles(
    positions: np.ndarray,
    total_mass: float,
    x_grid: np.ndarray,
    recon_method: str,
    bandwidth: float,
) -> np.ndarray:
    """
    Reconstruct field u on x_grid from density particle positions.

    The reconstruction preserves total mass (integral = total_mass).

    Parameters
    ----------
    positions    : (n_particles,) array of particle x-coordinates.
    total_mass   : target total mass for the reconstruction.
    x_grid       : uniform spatial grid.
    recon_method : "histogram" or "kde".
    bandwidth    : KDE bandwidth in x-units (ignored for histogram).
                   If <= 0 for KDE, falls back to Scott's rule.

    Returns
    -------
    u_recon : (n_grid,) non-negative array with integral = total_mass.
    """
    N = len(positions)
    n_grid = len(x_grid)
    dx = x_grid[1] - x_grid[0]

    if recon_method == "histogram":
        x_min_h = x_grid[0] - 0.5 * dx
        x_max_h = x_grid[-1] + 0.5 * dx
        counts, _ = np.histogram(positions, bins=n_grid, range=(x_min_h, x_max_h))
        u_recon = counts.astype(float) * (total_mass / N) / dx
        return u_recon

    elif recon_method == "kde":
        bw = bandwidth
        if bw <= 0.0:
            # Scott's rule
            std_pos = float(np.std(positions))
            bw = 1.06 * std_pos * N ** (-0.2) if std_pos > 0.0 else dx
        # Vectorized Gaussian KDE: evaluate at all x_grid points simultaneously
        diff = x_grid[:, None] - positions[None, :]  # (n_grid, N)
        kernels = np.exp(-0.5 * (diff / bw) ** 2) / (bw * np.sqrt(2.0 * np.pi))
        u_recon = np.sum(kernels, axis=1) * (total_mass / N)
        return u_recon

    else:
        raise ValueError(f"Unknown recon_method: {recon_method!r}. Use 'histogram' or 'kde'.")


# ---------------------------------------------------------------------------
# Shared integration loop skeleton
# ---------------------------------------------------------------------------

def _run_integration(
    method_name: str,
    state: GlobState,
    x_grid: np.ndarray,
    cfg: Config,
    rng: np.random.Generator,
    use_score: bool,
    use_noise: bool,
    stochastic_coeff: float,  # 2*alpha for stochastic, alpha for deterministic
    score_source: str,        # "oracle" or "estimated"
    oracle_mode: str = "normal",    # "normal" | "zero" | "flip"  (ablation)
    estimated_mode: str = "normal", # "normal" | "zero" | "flip"  (ablation)
    use_regularization: bool = False,  # if True, use estimated_score_regularized
) -> MethodResult:
    """
    Generic integration loop.

    Parameters
    ----------
    use_score        : if False, the score drift is omitted (naive backward).
    use_noise        : if True, Gaussian noise increment is added.
    stochastic_coeff : multiplier for score * dt in drift term.
    score_source     : "oracle" uses exact formula; "estimated" uses globs.
    """
    alpha = cfg.heat.alpha
    dt = cfg.heat.dt
    n_steps = cfg.n_steps
    noise_scale = np.sqrt(2.0 * alpha * dt)   # std of one Brownian increment
    snapshot_steps = _record_snapshot_steps(n_steps)
    # Score overlay snapshots at step 0, midpoint, final
    overlay_steps = {0, n_steps // 2, n_steps}

    result = MethodResult(
        method_name=method_name,
        completed=False,
        failure_step=None,
        failure_msg="",
        candidate=np.zeros_like(x_grid),
    )

    # Snapshot at step 0 (initial glob state = observed_final)
    snap0 = reconstruct_field(state, x_grid)
    result.step_snapshots[0] = snap0.copy()

    positions = state.positions.copy()
    t_start = time.perf_counter()

    for k in range(n_steps):
        # Physical time at this backward step: tau = k*dt, t_phys = T - tau
        tau = k * dt
        t_phys = cfg.heat.T - tau

        # --- Compute score drift ---
        if use_score:
            if score_source == "oracle":
                scores = oracle_score(positions, t_phys, cfg)
                if not np.all(np.isfinite(scores)):
                    result.failure_step = k
                    result.failure_msg = "Oracle score NaN/Inf (unexpected)"
                    result.candidate = reconstruct_field(
                        GlobState(positions, state.weights, state.u_left,
                                  state.x_min, state.x_max), x_grid)
                    result.final_positions = positions.copy()
                    result.runtime_seconds = time.perf_counter() - t_start
                    return result

                # Record zero error for oracle (oracle vs oracle = 0)
                result.score_L2_error_vs_oracle.append(0.0)
                result.score_Linf_error_vs_oracle.append(0.0)
                result.score_core_L2_error_vs_oracle.append(0.0)
                result.score_core_Linf_error_vs_oracle.append(0.0)

                # Apply oracle ablation mode AFTER recording zero error
                if oracle_mode == "zero":
                    scores = np.zeros_like(scores)
                elif oracle_mode == "flip":
                    scores = -scores
                drift = stochastic_coeff * scores * dt

            elif score_source == "estimated":
                tmp_state = GlobState(
                    positions=positions,
                    weights=state.weights,
                    u_left=state.u_left,
                    x_min=state.x_min,
                    x_max=state.x_max,
                )
                # --- Score computation (raw or regularized) ---
                if use_regularization:
                    scores, is_stable, msg, reg_diag = estimated_score_regularized(
                        positions, tmp_state, x_grid, cfg)
                else:
                    scores, is_stable, msg = estimated_score_raw(positions, tmp_state, x_grid, cfg)
                    reg_diag = None

                if not is_stable:
                    result.failure_step = k
                    result.failure_msg = f"{'Regularized' if use_regularization else 'Estimated'} score instability at step {k}: {msg}"
                    result.candidate = reconstruct_field(tmp_state, x_grid)
                    result.final_positions = positions.copy()
                    result.runtime_seconds = time.perf_counter() - t_start
                    return result

                # Record regularization diagnostics
                if reg_diag is not None:
                    result.n_denominator_below_epsilon.append(reg_diag["n_denom_below_epsilon"])
                    result.n_clipped_scores.append(reg_diag["n_clipped"])
                    if k == 0:
                        result.epsilon_used = reg_diag["epsilon_used"]

                # --- Score error vs oracle on grid (using UNMODIFIED s_est) ---
                s_oracle_grid = oracle_score(x_grid, t_phys, cfg)
                dx = x_grid[1] - x_grid[0]
                if reg_diag is not None:
                    # Use the regularized score on the grid (what the method actually uses)
                    u_grid_now = reg_diag["u_grid"]
                    s_est_grid = reg_diag["s_grid"]
                else:
                    u_grid_now = reconstruct_field(tmp_state, x_grid)
                    u_x_grid = np.gradient(u_grid_now, dx)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        s_est_grid = np.where(u_grid_now != 0.0,
                                             u_x_grid / u_grid_now, np.nan)

                finite_mask = np.isfinite(s_est_grid) & np.isfinite(s_oracle_grid)
                core_mask = finite_mask & (u_grid_now > 0.05 * np.max(u_grid_now))
                if np.any(finite_mask):
                    err_all = s_est_grid[finite_mask] - s_oracle_grid[finite_mask]
                    s_L2 = float(np.sqrt(dx * np.sum(err_all ** 2)))
                    s_Linf = float(np.max(np.abs(err_all)))
                else:
                    s_L2 = float("nan")
                    s_Linf = float("nan")
                if np.any(core_mask):
                    err_core = s_est_grid[core_mask] - s_oracle_grid[core_mask]
                    s_core_L2 = float(np.sqrt(dx * np.sum(err_core ** 2)))
                    s_core_Linf = float(np.max(np.abs(err_core)))
                else:
                    s_core_L2 = float("nan")
                    s_core_Linf = float("nan")

                result.score_L2_error_vs_oracle.append(s_L2)
                result.score_Linf_error_vs_oracle.append(s_Linf)
                result.score_core_L2_error_vs_oracle.append(s_core_L2)
                result.score_core_Linf_error_vs_oracle.append(s_core_Linf)

                # Suspicious zero: s_L2 exactly 0.0 despite finite comparisons
                if s_L2 == 0.0 and np.any(finite_mask):
                    result.score_suspicious_steps.append(k)

                # Score overlay snapshot at key steps (before mode override)
                if k in overlay_steps:
                    result.score_overlay_snapshots[k] = (
                        x_grid.copy(), s_est_grid.copy(), s_oracle_grid.copy()
                    )

                # Apply estimated ablation mode AFTER recording errors
                if estimated_mode == "zero":
                    scores = np.zeros_like(scores)
                elif estimated_mode == "flip":
                    scores = -scores
                drift = stochastic_coeff * scores * dt

            else:
                raise ValueError(f"Unknown score_source: {score_source}")

            # Record score diagnostics (from the (possibly overridden) scores)
            finite_scores = scores[np.isfinite(scores)]
            result.score_max_abs.append(
                float(np.max(np.abs(finite_scores))) if len(finite_scores) else np.nan)
            result.score_mean.append(
                float(np.mean(finite_scores)) if len(finite_scores) else np.nan)
            result.score_std.append(
                float(np.std(finite_scores)) if len(finite_scores) else np.nan)
        else:
            # Naive backward: no score drift
            drift = np.zeros_like(positions)
            result.score_max_abs.append(0.0)
            result.score_mean.append(0.0)
            result.score_std.append(0.0)
            result.score_L2_error_vs_oracle.append(float("nan"))
            result.score_Linf_error_vs_oracle.append(float("nan"))
            result.score_core_L2_error_vs_oracle.append(float("nan"))
            result.score_core_Linf_error_vs_oracle.append(float("nan"))

        # --- Noise increment ---
        if use_noise:
            xi = rng.standard_normal(len(positions))
            if not use_score:
                noise = -noise_scale * xi   # naive backward: flip sign
            else:
                noise = noise_scale * xi    # reverse SDE: add noise
        else:
            noise = np.zeros_like(positions)

        # --- Update positions ---
        positions = positions + drift + noise

        # --- Boundary ---
        if cfg.grw.boundary == "reflecting":
            positions = apply_reflecting_boundary(positions, state.x_min, state.x_max)

        # --- Safety check on positions ---
        ok, msg = _check_positions(positions, cfg.safety.value_abs_fail_threshold)
        if not ok:
            result.failure_step = k + 1
            result.failure_msg = msg
            tmp_state = GlobState(positions, state.weights, state.u_left,
                                  state.x_min, state.x_max)
            result.candidate = reconstruct_field(tmp_state, x_grid)
            result.final_positions = positions.copy()
            result.runtime_seconds = time.perf_counter() - t_start
            return result

        # --- Field snapshot ---
        step_idx = k + 1
        if step_idx in snapshot_steps:
            tmp_state = GlobState(positions, state.weights, state.u_left,
                                  state.x_min, state.x_max)
            result.step_snapshots[step_idx] = reconstruct_field(tmp_state, x_grid)

        # Score overlay at end-of-step (uses current field for overlay)
        if score_source == "estimated" and use_score and step_idx in overlay_steps:
            tmp_state = GlobState(positions, state.weights, state.u_left,
                                  state.x_min, state.x_max)
            u_g = reconstruct_field(tmp_state, x_grid)
            dx_g = x_grid[1] - x_grid[0]
            ux_g = np.gradient(u_g, dx_g)
            with np.errstate(divide="ignore", invalid="ignore"):
                s_est_end = np.where(u_g != 0.0, ux_g / u_g, np.nan)
            s_or_end = oracle_score(x_grid, cfg.heat.T - step_idx * dt, cfg)
            result.score_overlay_snapshots[step_idx] = (
                x_grid.copy(), s_est_end.copy(), s_or_end.copy()
            )

    # All steps completed
    final_state = GlobState(positions, state.weights, state.u_left,
                            state.x_min, state.x_max)
    result.completed = True
    result.failure_msg = ""
    result.candidate = reconstruct_field(final_state, x_grid)
    result.final_positions = positions.copy()
    result.runtime_seconds = time.perf_counter() - t_start
    return result


# ---------------------------------------------------------------------------
# Public method functions
# ---------------------------------------------------------------------------

def run_naive_backward(
    u_obs: np.ndarray,
    x_grid: np.ndarray,
    cfg: Config,
    rng: np.random.Generator,
) -> MethodResult:
    """
    Naive backward GRW: X_{k+1} = X_k - sqrt(2 alpha dt) xi_k.

    No score drift.  Merely flips the sign of the Brownian increment.
    This does not invert diffusion: xi and -xi are identically distributed.
    Expected to fail at recovering the initial peak.
    """
    state = field_to_globs(u_obs, x_grid, cfg)
    result = _run_integration(
        method_name="naive_backward",
        state=state,
        x_grid=x_grid,
        cfg=cfg,
        rng=rng,
        use_score=False,
        use_noise=True,
        stochastic_coeff=0.0,   # no drift
        score_source="oracle",  # unused
    )
    result.score_estimator_type = "none"
    return result


def run_oracle_score_deterministic(
    u_obs: np.ndarray,
    x_grid: np.ndarray,
    cfg: Config,
    rng: np.random.Generator,
    oracle_mode: str = "normal",
) -> MethodResult:
    """
    Oracle probability-flow ODE: X_{k+1} = X_k + alpha * s_exact(X_k, t_phys) * dt.

    Uses the exact Gaussian score.  No noise.  Coefficient is alpha (not 2*alpha).
    This is the deterministic reverse-time probability-flow ODE.
    Expected to concentrate particles toward the Gaussian peak.
    """
    state = field_to_globs(u_obs, x_grid, cfg)
    result = _run_integration(
        method_name="oracle_score_deterministic",
        state=state,
        x_grid=x_grid,
        cfg=cfg,
        rng=rng,
        use_score=True,
        use_noise=False,
        stochastic_coeff=cfg.heat.alpha,
        score_source="oracle",
        oracle_mode=oracle_mode,
        estimated_mode="normal",
    )
    result.score_estimator_type = "oracle"
    return result


def run_oracle_score_stochastic(
    u_obs: np.ndarray,
    x_grid: np.ndarray,
    cfg: Config,
    rng: np.random.Generator,
    oracle_mode: str = "normal",
) -> MethodResult:
    """
    Oracle reverse-time SDE:
        X_{k+1} = X_k + 2*alpha * s_exact(X_k, t_phys) * dt + sqrt(2*alpha*dt) xi_k.

    Uses the exact Gaussian score.  Includes noise.  Coefficient is 2*alpha.
    This is the full reverse-time SDE (Anderson 1982).
    Expected to show similar recovery as deterministic but with noise.
    """
    state = field_to_globs(u_obs, x_grid, cfg)
    result = _run_integration(
        method_name="oracle_score_stochastic",
        state=state,
        x_grid=x_grid,
        cfg=cfg,
        rng=rng,
        use_score=True,
        use_noise=True,
        stochastic_coeff=2.0 * cfg.heat.alpha,
        score_source="oracle",
        oracle_mode=oracle_mode,
        estimated_mode="normal",
    )
    result.score_estimator_type = "oracle"
    return result


def run_estimated_score_deterministic_raw(
    u_obs: np.ndarray,
    x_grid: np.ndarray,
    cfg: Config,
    rng: np.random.Generator,
    estimated_mode: str = "normal",
) -> MethodResult:
    """
    Estimated probability-flow ODE with raw score:
        X_{k+1} = X_k + alpha * s_est(X_k, t) * dt

    Score is estimated as u_x / u from the current glob reconstruction.
    No regularization applied.  If u ≈ 0, score may blow up.
    This will either show promising anti-diffusion or expose the need
    for regularization.
    """
    state = field_to_globs(u_obs, x_grid, cfg)
    result = _run_integration(
        method_name="estimated_score_deterministic_raw",
        state=state,
        x_grid=x_grid,
        cfg=cfg,
        rng=rng,
        use_score=True,
        use_noise=False,
        stochastic_coeff=cfg.heat.alpha,
        score_source="estimated",
        oracle_mode="normal",
        estimated_mode=estimated_mode,
    )
    result.score_estimator_type = "position_ratio_raw"
    return result


def run_estimated_score_stochastic_raw(
    u_obs: np.ndarray,
    x_grid: np.ndarray,
    cfg: Config,
    rng: np.random.Generator,
    estimated_mode: str = "normal",
) -> MethodResult:
    """
    Estimated reverse-time SDE with raw score:
        X_{k+1} = X_k + 2*alpha * s_est(X_k, t) * dt + sqrt(2*alpha*dt) xi_k

    Score is estimated as u_x / u from the current glob reconstruction.
    No regularization.  May blow up.
    """
    state = field_to_globs(u_obs, x_grid, cfg)
    result = _run_integration(
        method_name="estimated_score_stochastic_raw",
        state=state,
        x_grid=x_grid,
        cfg=cfg,
        rng=rng,
        use_score=True,
        use_noise=True,
        stochastic_coeff=2.0 * cfg.heat.alpha,
        score_source="estimated",
        oracle_mode="normal",
        estimated_mode=estimated_mode,
    )
    result.score_estimator_type = "position_ratio_raw"
    return result


# ---------------------------------------------------------------------------
# Regularized estimated-score methods (Phase 5+)
# NOTE: The raw methods above are PERMANENTLY unmodified.
# ---------------------------------------------------------------------------

def run_estimated_score_deterministic_regularized(
    u_obs: np.ndarray,
    x_grid: np.ndarray,
    cfg: Config,
    rng: np.random.Generator,
) -> MethodResult:
    """
    Regularized probability-flow ODE:
        X_{k+1} = X_k + alpha * s_reg(X_k, t) * dt

    Score is estimated with regularization as configured in cfg.regularization:
      - epsilon floor: s = u_x / (u + eps)
      - score clipping: |s| <= max_abs_score
    If cfg.regularization.enabled=False, eps=0 (same as raw).
    """
    state = field_to_globs(u_obs, x_grid, cfg)
    result = _run_integration(
        method_name="estimated_score_deterministic_regularized",
        state=state,
        x_grid=x_grid,
        cfg=cfg,
        rng=rng,
        use_score=True,
        use_noise=False,
        stochastic_coeff=cfg.heat.alpha,
        score_source="estimated",
        oracle_mode="normal",
        estimated_mode="normal",
        use_regularization=True,
    )
    result.score_estimator_type = "grid_ratio_epsilon"
    return result


def run_estimated_score_stochastic_regularized(
    u_obs: np.ndarray,
    x_grid: np.ndarray,
    cfg: Config,
    rng: np.random.Generator,
) -> MethodResult:
    """
    Regularized reverse-time SDE:
        X_{k+1} = X_k + 2*alpha * s_reg(X_k, t) * dt + sqrt(2*alpha*dt) xi_k

    Score is estimated with regularization as configured in cfg.regularization.
    If cfg.regularization.enabled=False, eps=0 (same as raw).
    """
    state = field_to_globs(u_obs, x_grid, cfg)
    result = _run_integration(
        method_name="estimated_score_stochastic_regularized",
        state=state,
        x_grid=x_grid,
        cfg=cfg,
        rng=rng,
        use_score=True,
        use_noise=True,
        stochastic_coeff=2.0 * cfg.heat.alpha,
        score_source="estimated",
        oracle_mode="normal",
        estimated_mode="normal",
        use_regularization=True,
    )
    result.score_estimator_type = "grid_ratio_epsilon"
    return result


def run_estimated_score_deterministic_grid_ratio_raw(
    u_obs: np.ndarray,
    x_grid: np.ndarray,
    cfg: Config,
    rng: np.random.Generator,
) -> MethodResult:
    """
    Grid-ratio probability-flow ODE (no epsilon, no clipping):
        X_{k+1} = X_k + alpha * s_gr(X_k, t) * dt

    Score is estimated on the grid as u_x / u (divide-then-interp, grid-ratio
    path), with epsilon=0 and clipping disabled.

    This is a DISTINCT discretization from estimated_score_deterministic_raw.
    That method uses interp-then-divide (position-ratio path).  Division and
    interpolation do not commute, so the two paths give numerically different
    results even though both nominally compute u_x / u.

    This wrapper makes the grid-ratio path accessible under an unambiguous
    name for convergence studies and CSV labeling.  It calls the same
    estimated_score_regularized code path with epsilon=0 and clipping off
    (a deepcopy of cfg is used so the caller's cfg is not modified).
    """
    import copy
    cfg_gr = copy.deepcopy(cfg)
    cfg_gr.regularization.enabled = False
    cfg_gr.regularization.epsilon_floor.enabled = False
    cfg_gr.regularization.score_clipping.enabled = False

    state = field_to_globs(u_obs, x_grid, cfg_gr)
    result = _run_integration(
        method_name="estimated_score_deterministic_grid_ratio_raw",
        state=state,
        x_grid=x_grid,
        cfg=cfg_gr,
        rng=rng,
        use_score=True,
        use_noise=False,
        stochastic_coeff=cfg_gr.heat.alpha,
        score_source="estimated",
        oracle_mode="normal",
        estimated_mode="normal",
        use_regularization=True,   # use the grid-ratio code path (eps=0)
    )
    result.score_estimator_type = "grid_ratio_raw"
    return result


# ---------------------------------------------------------------------------
# Density-particle oracle method (representation audit)
# ---------------------------------------------------------------------------

def run_density_particle_oracle_score_deterministic(
    u_obs: np.ndarray,
    x_grid: np.ndarray,
    cfg: Config,
    n_particles: int,
    recon_method: str = "kde",
    bandwidth_factor: float = 2.0,
) -> MethodResult:
    """
    Density-particle probability-flow ODE with oracle score.

    Representation-consistent reverse diffusion: particles carry the density
    field u directly (not u_x).  The oracle score s = partial_x log u is the
    exact drift for particles representing u under the reverse-time heat
    equation.

    The reverse-time probability-flow ODE for density particles is:
        X_{k+1} = X_k + alpha * s(X_k, t_phys) * dt
    where s = u_x / u is the field score.  This is equivalent to the
    backward heat equation in the continuum limit:
        partial_tau u = -partial_x(u * alpha * u_x/u) = -alpha * u_xx.

    This is DISTINCT from the gradient-glob method, which moves globs
    carrying u_x with the same field score.  That method is representation-
    mismatched: the correct continuity equation for u_x-particles would
    require the gradient score partial_x log(u_x), not the field score.

    Algorithm
    ---------
    1. Initialize n_particles from u_obs via deterministic quantile placement.
    2. Record step-zero reconstruction error (before any movement).
    3. For k = 0 .. n_steps-1:
           t_phys = T - k*dt
           s = oracle_score(positions, t_phys, cfg)   [exact analytic]
           positions += alpha * s * dt
           apply reflecting boundary
    4. Reconstruct u_0 from final positions via histogram or KDE.

    Parameters
    ----------
    u_obs           : observed field u(., T) on x_grid.
    x_grid          : uniform spatial grid.
    cfg             : experiment config.
    n_particles     : number of density particles.
    recon_method    : "histogram" or "kde".
    bandwidth_factor: KDE bandwidth = bandwidth_factor * dx.
                      Ignored for histogram.

    Returns
    -------
    MethodResult with:
        method_name             = "density_particle_oracle_deterministic"
        score_estimator_type    = "density_particle_oracle_field_score"
        n_particles             = n_particles
        reconstruction_method   = recon_method
        bandwidth_used          = actual KDE bandwidth (nan for histogram)
        bandwidth_factor        = bandwidth_factor argument
        step_zero_recon_error   = relative L2 of reconstruction at t=T
        mass_initial            = total mass of u_obs
        completed, failure_step, failure_msg, runtime_seconds as usual
    """
    alpha = cfg.heat.alpha
    dt = cfg.heat.dt
    n_steps = cfg.n_steps
    x_min = float(x_grid[0])
    x_max = float(x_grid[-1])
    dx = float(x_grid[1] - x_grid[0])
    bw = bandwidth_factor * dx

    result = MethodResult(
        method_name="density_particle_oracle_deterministic",
        completed=False,
        failure_step=None,
        failure_msg="",
        candidate=np.zeros_like(x_grid),
    )
    result.score_estimator_type = "density_particle_oracle_field_score"
    result.n_particles = n_particles
    result.reconstruction_method = recon_method
    result.bandwidth_factor = bandwidth_factor
    result.bandwidth_used = bw if recon_method == "kde" else float("nan")

    # Step 1: initialize density particles
    positions, total_mass = _quantile_init_particles(u_obs, x_grid, n_particles)
    result.mass_initial = total_mass

    # Step-zero reconstruction error (quantifies representation quality at t=T)
    u_recon_t0 = _reconstruct_density_particles(positions, total_mass, x_grid, recon_method, bw)
    u_obs_norm = float(np.sqrt(dx * np.sum(u_obs ** 2)))
    if u_obs_norm > 0.0:
        step0_diff = u_recon_t0 - u_obs
        result.step_zero_recon_error = float(np.sqrt(dx * np.sum(step0_diff ** 2))) / u_obs_norm
    else:
        result.step_zero_recon_error = float("nan")

    score_fail_thresh = cfg.safety.score_abs_fail_threshold
    pos_fail_thresh = cfg.safety.value_abs_fail_threshold

    t_start = time.perf_counter()

    # Step 2: backward integration
    for k in range(n_steps):
        t_phys = cfg.heat.T - k * dt

        # Oracle score at particle positions (exact analytic formula)
        scores = oracle_score(positions, t_phys, cfg)

        if not np.all(np.isfinite(scores)):
            result.failure_step = k
            result.failure_msg = f"Oracle score NaN/Inf at step {k}"
            result.candidate = _reconstruct_density_particles(
                positions, total_mass, x_grid, recon_method, bw)
            result.final_positions = positions.copy()
            result.runtime_seconds = time.perf_counter() - t_start
            return result

        if np.any(np.abs(scores) > score_fail_thresh):
            result.failure_step = k
            result.failure_msg = (
                f"Oracle score magnitude {float(np.max(np.abs(scores))):.3e} "
                f"> {score_fail_thresh:.3e} at step {k}"
            )
            result.candidate = _reconstruct_density_particles(
                positions, total_mass, x_grid, recon_method, bw)
            result.final_positions = positions.copy()
            result.runtime_seconds = time.perf_counter() - t_start
            return result

        # Record score diagnostics
        result.score_max_abs.append(float(np.max(np.abs(scores))))
        result.score_mean.append(float(np.mean(scores)))
        result.score_std.append(float(np.std(scores)))
        result.score_L2_error_vs_oracle.append(0.0)   # oracle vs oracle = 0

        # Deterministic probability-flow update
        positions = positions + alpha * scores * dt

        # Reflecting boundary
        positions = apply_reflecting_boundary(positions, x_min, x_max)

        # Safety check
        if not np.all(np.isfinite(positions)):
            result.failure_step = k + 1
            result.failure_msg = f"NaN/Inf in particle positions at step {k + 1}"
            result.candidate = _reconstruct_density_particles(
                positions, total_mass, x_grid, recon_method, bw)
            result.final_positions = positions.copy()
            result.runtime_seconds = time.perf_counter() - t_start
            return result

        if np.any(np.abs(positions) > pos_fail_thresh):
            result.failure_step = k + 1
            result.failure_msg = (
                f"Particle position magnitude > {pos_fail_thresh:.3e} at step {k + 1}"
            )
            result.candidate = _reconstruct_density_particles(
                positions, total_mass, x_grid, recon_method, bw)
            result.final_positions = positions.copy()
            result.runtime_seconds = time.perf_counter() - t_start
            return result

    # All steps completed successfully
    result.completed = True
    result.candidate = _reconstruct_density_particles(
        positions, total_mass, x_grid, recon_method, bw)
    result.final_positions = positions.copy()
    result.runtime_seconds = time.perf_counter() - t_start
    return result


# ---------------------------------------------------------------------------
# Density-particle estimated-score method
# ---------------------------------------------------------------------------

def run_density_particle_estimated_score_deterministic(
    u_obs: np.ndarray,
    x_grid: np.ndarray,
    cfg: Config,
    n_particles: int,
    rng=None,
    recon_method: str = "kde",
    bandwidth: "float | None" = None,
    bandwidth_factor: float = 2.0,
    epsilon: float = 0.0,
    scale_epsilon_by_peak: bool = False,
    score_clipping: "float | None" = None,
    save_snapshots: bool = True,
    score_method: str = "fd_grid_ratio",
    smooth_sigma_factor: float = 1.0,
) -> MethodResult:
    """
    Density-particle probability-flow ODE with estimated score.

    Like run_density_particle_oracle_score_deterministic, but computes the
    score from the current reconstructed density rather than the exact oracle
    formula.

    score_method controls how the score is estimated:

    "fd_grid_ratio" (original):
        reconstruct u on grid → np.gradient(u) → u_x/(u+ε) → interpolate.
    "direct_kde":
        evaluate KDE derivative directly at particle positions (no grid step).
        s(X_i) = -sum_j w_j (X_i-X_j)/h^2 K_h(X_i-X_j)
                  / (sum_j w_j K_h(X_i-X_j) + ε)
        Particle-native; avoids finite-difference noise amplification.
        Also evaluates on x_grid for oracle comparison diagnostics.
    "log_density_fd":
        reconstruct u on grid → log(u+ε) → np.gradient(log u) → interpolate.
        Mathematically equivalent to fd_grid_ratio but numerically different
        near zeros.
    "smoothed_log":
        reconstruct u on grid → Gaussian smooth (sigma = smooth_sigma_factor * bw)
        → log(u+ε) → np.gradient → interpolate.  Suppresses high-frequency
        KDE noise before differentiating.

    NaN/Inf scores are NEVER silently replaced with zero.

    Parameters
    ----------
    u_obs              : observed field u(., T) on x_grid.
    x_grid             : uniform spatial grid.
    cfg                : experiment config.
    n_particles        : number of density particles.
    rng                : random generator (unused; accepted for API symmetry).
    recon_method       : "histogram" or "kde".
    bandwidth          : explicit KDE bandwidth in x-units.  If None,
                         bandwidth = bandwidth_factor * dx.
    bandwidth_factor   : KDE bandwidth as a multiple of dx.
    epsilon            : denominator regularization floor.
    scale_epsilon_by_peak : if True, epsilon_actual = epsilon * max(u_recon).
    score_clipping     : if not None, clip |s| to score_clipping.
    save_snapshots     : if True, save field snapshots at key steps.
    score_method       : one of "fd_grid_ratio", "direct_kde", "log_density_fd",
                         "smoothed_log".
    smooth_sigma_factor: for "smoothed_log" only — smooth sigma = factor * bw.
    """
    _VALID_SCORE_METHODS = ("fd_grid_ratio", "direct_kde", "log_density_fd", "smoothed_log")
    if score_method not in _VALID_SCORE_METHODS:
        raise ValueError(f"score_method must be one of {_VALID_SCORE_METHODS}, got {score_method!r}")
    alpha = cfg.heat.alpha
    dt = cfg.heat.dt
    n_steps = cfg.n_steps
    x_min = float(x_grid[0])
    x_max = float(x_grid[-1])
    dx = float(x_grid[1] - x_grid[0])

    # Determine KDE bandwidth
    bw = float(bandwidth) if bandwidth is not None else bandwidth_factor * dx

    # Smooth sigma for "smoothed_log" method
    smooth_sigma = smooth_sigma_factor * bw

    # Method naming encodes score_method and epsilon for CSV readability
    if epsilon == 0.0 and not scale_epsilon_by_peak:
        eps_tag = "raw"
    else:
        eps_tag = f"eps={epsilon:.0e}"

    # Map score_method to short canonical names used in MethodResult
    _SCORE_METHOD_NAMES = {
        "fd_grid_ratio":  "density_est_fd_grid_ratio",
        "direct_kde":     "density_est_direct_kde",
        "log_density_fd": "density_est_log_density_fd",
        "smoothed_log":   "density_est_smoothed_log",
    }
    _SCORE_EST_TYPES = {
        "fd_grid_ratio":  "density_particle_estimated_grid_ratio_raw" if epsilon == 0.0 else "density_particle_estimated_grid_ratio_epsilon",
        "direct_kde":     "density_particle_estimated_direct_kde",
        "log_density_fd": "density_particle_estimated_log_density_fd",
        "smoothed_log":   "density_particle_estimated_smoothed_log",
    }
    method_name = f"{_SCORE_METHOD_NAMES[score_method]}_{eps_tag}"
    score_est_type = _SCORE_EST_TYPES[score_method]

    result = MethodResult(
        method_name=method_name,
        completed=False,
        failure_step=None,
        failure_msg="",
        candidate=np.zeros_like(x_grid),
    )
    result.score_estimator_type = score_est_type
    result.n_particles = n_particles
    result.reconstruction_method = recon_method
    result.bandwidth_factor = bandwidth_factor
    result.bandwidth_used = bw if recon_method == "kde" else float("nan")
    result.epsilon_used = epsilon  # base epsilon before scale_by_peak

    # Initialize density particles from u_obs
    positions, total_mass = _quantile_init_particles(u_obs, x_grid, n_particles)
    result.mass_initial = total_mass

    # Step-zero reconstruction error (quantifies KDE/histogram fit at t=T)
    u_recon_t0 = _reconstruct_density_particles(positions, total_mass, x_grid, recon_method, bw)
    u_obs_norm = float(np.sqrt(dx * np.sum(u_obs ** 2)))
    if u_obs_norm > 0.0:
        step0_diff = u_recon_t0 - u_obs
        result.step_zero_recon_error = float(np.sqrt(dx * np.sum(step0_diff ** 2))) / u_obs_norm
    else:
        result.step_zero_recon_error = float("nan")

    score_fail_thresh = cfg.safety.score_abs_fail_threshold
    pos_fail_thresh = cfg.safety.value_abs_fail_threshold
    snapshot_steps = _record_snapshot_steps(n_steps)

    t_start = time.perf_counter()

    for k in range(n_steps):
        t_phys = cfg.heat.T - k * dt

        # --- Reconstruct density on grid from current particle positions ---
        u_recon = _reconstruct_density_particles(positions, total_mass, x_grid, recon_method, bw)

        # --- Compute effective epsilon for this step ---
        if scale_epsilon_by_peak:
            peak_u = float(np.max(u_recon)) if np.any(u_recon > 0.0) else 0.0
            epsilon_actual = epsilon * max(peak_u, 0.0)
        else:
            epsilon_actual = epsilon
        result.epsilon_actual_per_step.append(epsilon_actual)

        # --- Per-step mass tracking ---
        mass_recon = float(_trapz(u_recon, x_grid))  # type: ignore[attr-defined]
        result.mass_per_step.append(mass_recon)

        # --- Denominator diagnostic: count grid pts where u < epsilon_actual ---
        if epsilon_actual > 0.0:
            n_below_eps = int(np.sum(u_recon < epsilon_actual))
        else:
            n_below_eps = int(np.sum(u_recon <= 0.0))
        result.n_denominator_below_epsilon.append(n_below_eps)

        # --- Estimate score and get s_grid (for oracle comparison) ---
        n_clipped = 0

        if score_method == "fd_grid_ratio":
            # Original path: gradient(u) / (u + ε) on grid → interpolate
            u_x = np.gradient(u_recon, dx)
            denom = u_recon + epsilon_actual
            with np.errstate(divide="ignore", invalid="ignore"):
                s_grid = np.where(denom > 0.0, u_x / denom, np.nan)
            if score_clipping is not None and score_clipping > 0.0:
                finite_mask_clip = np.isfinite(s_grid)
                clip_mask = finite_mask_clip & (np.abs(s_grid) > score_clipping)
                n_clipped = int(np.sum(clip_mask))
                s_grid = np.where(finite_mask_clip,
                                  np.clip(s_grid, -score_clipping, score_clipping),
                                  s_grid)
            scores = np.interp(positions, x_grid, s_grid)

        elif score_method == "direct_kde":
            # Direct KDE derivative at particle positions (no grid step)
            w_uniform = np.full(n_particles, total_mass / n_particles)
            scores, kde_diag = direct_kde_score(
                positions, positions, w_uniform, bw, epsilon=epsilon_actual
            )
            # Also evaluate on grid for oracle comparison diagnostics
            s_grid, _ = direct_kde_score(
                x_grid, positions, w_uniform, bw, epsilon=epsilon_actual
            )
            # Score clipping (applied to scores-at-particles and s_grid)
            if score_clipping is not None and score_clipping > 0.0:
                finite_mask_clip = np.isfinite(scores)
                clip_mask = finite_mask_clip & (np.abs(scores) > score_clipping)
                n_clipped = int(np.sum(clip_mask))
                scores = np.where(finite_mask_clip,
                                  np.clip(scores, -score_clipping, score_clipping),
                                  scores)
                s_grid = np.where(np.isfinite(s_grid),
                                  np.clip(s_grid, -score_clipping, score_clipping),
                                  s_grid)

        elif score_method == "log_density_fd":
            # log-density finite-difference path
            scores_grid, _ = log_density_fd_score(
                x_grid, u_recon, x_grid, epsilon=epsilon_actual
            )
            s_grid = scores_grid
            if score_clipping is not None and score_clipping > 0.0:
                finite_mask_clip = np.isfinite(s_grid)
                clip_mask = finite_mask_clip & (np.abs(s_grid) > score_clipping)
                n_clipped = int(np.sum(clip_mask))
                s_grid = np.where(finite_mask_clip,
                                  np.clip(s_grid, -score_clipping, score_clipping),
                                  s_grid)
            scores = np.interp(positions, x_grid, s_grid)

        elif score_method == "smoothed_log":
            # Gaussian-smoothed log-density gradient
            scores_grid, _ = smoothed_log_score(
                x_grid, u_recon, x_grid,
                smooth_sigma=smooth_sigma, epsilon=epsilon_actual
            )
            s_grid = scores_grid
            if score_clipping is not None and score_clipping > 0.0:
                finite_mask_clip = np.isfinite(s_grid)
                clip_mask = finite_mask_clip & (np.abs(s_grid) > score_clipping)
                n_clipped = int(np.sum(clip_mask))
                s_grid = np.where(finite_mask_clip,
                                  np.clip(s_grid, -score_clipping, score_clipping),
                                  s_grid)
            scores = np.interp(positions, x_grid, s_grid)

        result.n_clipped_scores.append(n_clipped)

        # --- Compare estimated score vs oracle on grid ---
        s_oracle_grid = oracle_score(x_grid, t_phys, cfg)
        finite_mask = np.isfinite(s_grid) & np.isfinite(s_oracle_grid)
        peak_u_for_core = float(np.max(u_recon)) if np.any(u_recon > 0.0) else 1.0
        core_mask = finite_mask & (u_recon > 0.05 * peak_u_for_core)

        if np.any(finite_mask):
            err_all = s_grid[finite_mask] - s_oracle_grid[finite_mask]
            s_L2 = float(np.sqrt(dx * np.sum(err_all ** 2)))
            s_Linf = float(np.max(np.abs(err_all)))
        else:
            s_L2 = float("nan")
            s_Linf = float("nan")

        if np.any(core_mask):
            err_core = s_grid[core_mask] - s_oracle_grid[core_mask]
            s_core_L2 = float(np.sqrt(dx * np.sum(err_core ** 2)))
            s_core_Linf = float(np.max(np.abs(err_core)))
        else:
            s_core_L2 = float("nan")
            s_core_Linf = float("nan")

        result.score_L2_error_vs_oracle.append(s_L2)
        result.score_Linf_error_vs_oracle.append(s_Linf)
        result.score_core_L2_error_vs_oracle.append(s_core_L2)
        result.score_core_Linf_error_vs_oracle.append(s_core_Linf)

        # --- Safety: NaN/Inf in interpolated scores — do NOT zero them out ---
        if not np.all(np.isfinite(scores)):
            n_bad = int(np.sum(~np.isfinite(scores)))
            result.failure_step = k
            result.failure_msg = (
                f"NaN/Inf in estimated score ({score_method}) at step {k} "
                f"({n_bad}/{n_particles} particles); "
                f"epsilon_actual={epsilon_actual:.2e}, "
                f"n_denom_zero={n_below_eps}"
            )
            result.candidate = _reconstruct_density_particles(
                positions, total_mass, x_grid, recon_method, bw)
            result.final_positions = positions.copy()
            result.runtime_seconds = time.perf_counter() - t_start
            return result

        if np.any(np.abs(scores) > score_fail_thresh):
            max_s = float(np.max(np.abs(scores)))
            result.failure_step = k
            result.failure_msg = (
                f"Score magnitude {max_s:.3e} > {score_fail_thresh:.3e} at step {k}; "
                f"score_method={score_method}, epsilon_actual={epsilon_actual:.2e}"
            )
            result.candidate = _reconstruct_density_particles(
                positions, total_mass, x_grid, recon_method, bw)
            result.final_positions = positions.copy()
            result.runtime_seconds = time.perf_counter() - t_start
            return result

        # --- Record score diagnostics ---
        result.score_max_abs.append(float(np.max(np.abs(scores))))
        result.score_mean.append(float(np.mean(scores)))
        result.mean_abs_score.append(float(np.mean(np.abs(scores))))
        result.score_std.append(float(np.std(scores)))

        # --- Deterministic probability-flow update ---
        positions = positions + alpha * scores * dt

        # --- Reflecting boundary ---
        positions = apply_reflecting_boundary(positions, x_min, x_max)

        # --- Position safety ---
        if not np.all(np.isfinite(positions)):
            result.failure_step = k + 1
            result.failure_msg = f"NaN/Inf in particle positions at step {k + 1}"
            result.candidate = _reconstruct_density_particles(
                positions, total_mass, x_grid, recon_method, bw)
            result.final_positions = positions.copy()
            result.runtime_seconds = time.perf_counter() - t_start
            return result

        if np.any(np.abs(positions) > pos_fail_thresh):
            result.failure_step = k + 1
            result.failure_msg = (
                f"Particle position magnitude > {pos_fail_thresh:.3e} at step {k + 1}"
            )
            result.candidate = _reconstruct_density_particles(
                positions, total_mass, x_grid, recon_method, bw)
            result.final_positions = positions.copy()
            result.runtime_seconds = time.perf_counter() - t_start
            return result

        # --- Optional field snapshot at key steps ---
        if save_snapshots and (k + 1) in snapshot_steps:
            snap = _reconstruct_density_particles(
                positions, total_mass, x_grid, recon_method, bw)
            result.step_snapshots[k + 1] = snap.copy()

    # All steps completed successfully
    result.completed = True
    result.candidate = _reconstruct_density_particles(
        positions, total_mass, x_grid, recon_method, bw)
    result.final_positions = positions.copy()
    result.runtime_seconds = time.perf_counter() - t_start
    return result
