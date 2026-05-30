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

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from .config import Config
from .globs import GlobState, field_to_globs, reconstruct_field, apply_reflecting_boundary
from .scores import oracle_score, estimated_score_raw


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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_positions(positions: np.ndarray, threshold: float) -> tuple[bool, str]:
    if not np.all(np.isfinite(positions)):
        return False, "NaN/Inf in glob positions"
    if np.any(np.abs(positions) > threshold):
        return False, f"Glob position magnitude > {threshold:.3e}"
    return True, "OK"


def _record_snapshot_steps() -> set:
    """Steps at which to save field snapshots for plotting."""
    return {0, 1, 2, 5, 10}


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
    snapshot_steps = _record_snapshot_steps()

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

    for k in range(n_steps):
        # Physical time at this backward step: tau = k*dt, t_phys = T - tau
        tau = k * dt
        t_phys = cfg.heat.T - tau

        # --- Compute score drift ---
        if use_score:
            if score_source == "oracle":
                scores = oracle_score(positions, t_phys, cfg)
                # Oracle score is always finite for Gaussian; still check
                if not np.all(np.isfinite(scores)):
                    result.failure_step = k
                    result.failure_msg = "Oracle score NaN/Inf (unexpected)"
                    result.candidate = reconstruct_field(
                        GlobState(positions, state.weights, state.u_left,
                                  state.x_min, state.x_max), x_grid)
                    result.final_positions = positions.copy()
                    return result

                drift = stochastic_coeff * scores * dt

            elif score_source == "estimated":
                tmp_state = GlobState(
                    positions=positions,
                    weights=state.weights,
                    u_left=state.u_left,
                    x_min=state.x_min,
                    x_max=state.x_max,
                )
                scores, is_stable, msg = estimated_score_raw(positions, tmp_state, x_grid, cfg)
                if not is_stable:
                    result.failure_step = k
                    result.failure_msg = f"Estimated score instability at step {k}: {msg}"
                    result.candidate = reconstruct_field(tmp_state, x_grid)
                    result.final_positions = positions.copy()
                    return result
                drift = stochastic_coeff * scores * dt
            else:
                raise ValueError(f"Unknown score_source: {score_source}")

            # Record score diagnostics
            finite_scores = scores[np.isfinite(scores)]
            result.score_max_abs.append(float(np.max(np.abs(finite_scores))) if len(finite_scores) else np.nan)
            result.score_mean.append(float(np.mean(finite_scores)) if len(finite_scores) else np.nan)
            result.score_std.append(float(np.std(finite_scores)) if len(finite_scores) else np.nan)
        else:
            drift = np.zeros_like(positions)
            # No score; still record zeros for consistent plotting
            result.score_max_abs.append(0.0)
            result.score_mean.append(0.0)
            result.score_std.append(0.0)

        # --- Noise increment ---
        if use_noise:
            xi = rng.standard_normal(len(positions))
            # Naive backward: subtract noise; score methods: add noise
            if not use_score:
                noise = -noise_scale * xi   # naive backward flip
            else:
                noise = noise_scale * xi    # forward noise term in reverse SDE
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
            return result

        # --- Snapshot ---
        step_idx = k + 1
        if step_idx in snapshot_steps:
            tmp_state = GlobState(positions, state.weights, state.u_left,
                                  state.x_min, state.x_max)
            result.step_snapshots[step_idx] = reconstruct_field(tmp_state, x_grid)

    # All steps completed
    final_state = GlobState(positions, state.weights, state.u_left,
                            state.x_min, state.x_max)
    result.completed = True
    result.failure_msg = ""
    result.candidate = reconstruct_field(final_state, x_grid)
    result.final_positions = positions.copy()
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
    return _run_integration(
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


def run_oracle_score_deterministic(
    u_obs: np.ndarray,
    x_grid: np.ndarray,
    cfg: Config,
    rng: np.random.Generator,
) -> MethodResult:
    """
    Oracle probability-flow ODE: X_{k+1} = X_k + alpha * s_exact(X_k, t_phys) * dt.

    Uses the exact Gaussian score.  No noise.  Coefficient is alpha (not 2*alpha).
    This is the deterministic reverse-time probability-flow ODE.
    Expected to concentrate particles toward the Gaussian peak.
    """
    state = field_to_globs(u_obs, x_grid, cfg)
    return _run_integration(
        method_name="oracle_score_deterministic",
        state=state,
        x_grid=x_grid,
        cfg=cfg,
        rng=rng,
        use_score=True,
        use_noise=False,
        stochastic_coeff=cfg.heat.alpha,   # alpha for deterministic flow
        score_source="oracle",
    )


def run_oracle_score_stochastic(
    u_obs: np.ndarray,
    x_grid: np.ndarray,
    cfg: Config,
    rng: np.random.Generator,
) -> MethodResult:
    """
    Oracle reverse-time SDE:
        X_{k+1} = X_k + 2*alpha * s_exact(X_k, t_phys) * dt + sqrt(2*alpha*dt) xi_k.

    Uses the exact Gaussian score.  Includes noise.  Coefficient is 2*alpha.
    This is the full reverse-time SDE (Anderson 1982).
    Expected to show similar recovery as deterministic but with noise.
    """
    state = field_to_globs(u_obs, x_grid, cfg)
    return _run_integration(
        method_name="oracle_score_stochastic",
        state=state,
        x_grid=x_grid,
        cfg=cfg,
        rng=rng,
        use_score=True,
        use_noise=True,
        stochastic_coeff=2.0 * cfg.heat.alpha,  # 2*alpha for stochastic SDE
        score_source="oracle",
    )


def run_estimated_score_deterministic_raw(
    u_obs: np.ndarray,
    x_grid: np.ndarray,
    cfg: Config,
    rng: np.random.Generator,
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
    return _run_integration(
        method_name="estimated_score_deterministic_raw",
        state=state,
        x_grid=x_grid,
        cfg=cfg,
        rng=rng,
        use_score=True,
        use_noise=False,
        stochastic_coeff=cfg.heat.alpha,
        score_source="estimated",
    )


def run_estimated_score_stochastic_raw(
    u_obs: np.ndarray,
    x_grid: np.ndarray,
    cfg: Config,
    rng: np.random.Generator,
) -> MethodResult:
    """
    Estimated reverse-time SDE with raw score:
        X_{k+1} = X_k + 2*alpha * s_est(X_k, t) * dt + sqrt(2*alpha*dt) xi_k

    Score is estimated as u_x / u from the current glob reconstruction.
    No regularization.  May blow up.
    """
    state = field_to_globs(u_obs, x_grid, cfg)
    return _run_integration(
        method_name="estimated_score_stochastic_raw",
        state=state,
        x_grid=x_grid,
        cfg=cfg,
        rng=rng,
        use_score=True,
        use_noise=True,
        stochastic_coeff=2.0 * cfg.heat.alpha,
        score_source="estimated",
    )
