"""
methods.py — Backward GRW integration methods.

Each method integrates glob positions backward from t = T to t = 0
by running n_steps of size dt.  After integration, the field is
reconstructed from the final glob positions.

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
from .scores import oracle_score, estimated_score_raw, estimated_score_regularized


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
    # Per-step score error vs oracle (for estimated-score methods)
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
