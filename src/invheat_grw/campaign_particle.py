"""campaign_particle.py — canonical density-particle runner for Phase 2C.

Implements the production estimated-score backward run with the Section 4
score-floor and truncation contract of PHASE2C_PROTOCOL.md:

- additive relative regularization ``epsilon_abs = epsilon_rel * M0 / L``;
- certified truncation envelopes ``B0(K)``/``B1(K)`` from the geometric and
  arithmetic-geometric tail majorants, and the ``epsilon_abs >= 100 B0`` gate;
- particle states retained immediately before reverse steps ``0``,
  ``floor(n/2)``, and ``n-1``, with the tightened-truncation score comparison
  on those states (both envelopes at least 100 times smaller; relative L2
  change at most 1e-8 where the tightened score norm is nonzero, absolute
  change at most 1e-10/L otherwise);
- fail-loud behavior: a nonpositive denominator, a nonfinite score, or a
  score magnitude above the preexisting safety-abort ceiling 1e6 terminates
  the run with its step recorded, and a failed tightened comparison fails the
  row rather than permitting a post hoc tolerance choice.

The runner supports constant diffusivity (``alpha``) and variable diffusivity
(``a_of_x``), whose deterministic velocity is ``a(x)`` times the estimated
score with no ``a'(x)`` term.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .cell_grid import (
    cell_centers,
    cell_spacing,
    cell_edge_quantile_positions,
    midpoint_norm,
)
from .globs import apply_reflecting_boundary
from .neumann_kernels import (
    neumann_kde_density_derivative,
    neumann_kde_score,
    neumann_mode_count,
)

SCORE_ABORT_CEILING = 1e6
EPS_FLOOR_FACTOR = 100.0
TIGHTEN_FACTOR = 100.0
TIGHTENED_REL_TOL = 1e-8
TIGHTENED_ABS_OVER_L = 1e-10
KERNEL_TOLERANCE = 1e-14


class CampaignContractError(ValueError):
    """Raised for configuration errors that violate the frozen contract."""


def exact_step_count(T: float, dt: float) -> int:
    """n with n * dt = T exactly; a mismatched horizon is a refusal."""
    if not (math.isfinite(T) and T > 0.0 and math.isfinite(dt) and dt > 0.0):
        raise CampaignContractError("T and dt must be finite and positive")
    n = round(T / dt)
    if n < 1 or abs(n * dt - T) > 1e-12 * max(1.0, T):
        raise CampaignContractError(
            f"time-step contract violated: {n} * {dt} != {T}"
        )
    return n


def certified_b0(n_modes: int, bandwidth: float, length: float,
                 total_mass: float) -> float:
    """Certified upper bound for B0(K) = (2 M0/L) sum_{n>K} exp(-theta n^2)."""
    theta = 0.5 * (math.pi * bandwidth / length) ** 2
    ratio = math.exp(-theta * (2 * n_modes + 3))
    lead = math.exp(-theta * (n_modes + 1) ** 2)
    return (2.0 * total_mass / length) * lead / (1.0 - ratio)


def certified_b1(n_modes: int, bandwidth: float, length: float,
                 total_mass: float) -> float:
    """Certified upper bound for B1(K) = (2 pi M0/L^2) sum_{n>K} n exp(-theta n^2)."""
    theta = 0.5 * (math.pi * bandwidth / length) ** 2
    ratio = math.exp(-theta * (2 * n_modes + 3))
    lead = math.exp(-theta * (n_modes + 1) ** 2)
    tail = lead * ((n_modes + 1) / (1.0 - ratio) + ratio / (1.0 - ratio) ** 2)
    return (2.0 * math.pi * total_mass / length ** 2) * tail


def tightened_mode_count(n_modes: int, bandwidth: float, length: float,
                         total_mass: float) -> int:
    """Smallest K whose certified envelopes are both at least TIGHTEN_FACTOR
    below the production envelopes."""
    b0_target = certified_b0(n_modes, bandwidth, length, total_mass) / TIGHTEN_FACTOR
    b1_target = certified_b1(n_modes, bandwidth, length, total_mass) / TIGHTEN_FACTOR
    k = n_modes
    while (certified_b0(k, bandwidth, length, total_mass) > b0_target
           or certified_b1(k, bandwidth, length, total_mass) > b1_target):
        k += 1
        if k > n_modes + 100000:
            raise CampaignContractError("tightened mode search did not terminate")
    return k


def tolerance_for_modes(n_modes: int, bandwidth: float, length: float) -> float:
    """A kernel tolerance that makes neumann_mode_count return n_modes."""
    scale = math.pi * bandwidth / length
    return math.exp(-0.5 * ((n_modes + 0.5) * scale) ** 2)


def score_check_message(score: np.ndarray, denominator: np.ndarray) -> str:
    """Empty string if the score passes the fail-loud checks."""
    if np.any(denominator <= 0.0):
        return "nonpositive score denominator"
    if not np.all(np.isfinite(score)):
        return "nonfinite score"
    if float(np.max(np.abs(score))) > SCORE_ABORT_CEILING:
        return "score magnitude above the safety-abort ceiling"
    return ""


@dataclass
class CampaignRunResult:
    status: str
    failure_step: int | None
    failure_message: str
    reconstruction: np.ndarray | None
    metrics: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    gates: dict = field(default_factory=dict)
    snapshots: dict = field(default_factory=dict)


def run_campaign_density(
    datum: np.ndarray,
    *,
    x_min: float,
    x_max: float,
    T: float,
    dt: float,
    n_particles: int,
    bandwidth: float,
    eps_rel: float,
    alpha: float | None = None,
    a_of_x: Callable[[np.ndarray], np.ndarray] | None = None,
    u0_reference: np.ndarray | None = None,
    forward_residual: Callable[[np.ndarray], float] | None = None,
    keep_snapshots: bool = True,
) -> CampaignRunResult:
    if (alpha is None) == (a_of_x is None):
        raise CampaignContractError("provide exactly one of alpha and a_of_x")
    if eps_rel < 0.0:
        raise CampaignContractError("eps_rel must be nonnegative")

    datum = np.asarray(datum, dtype=float)
    m_grid = datum.shape[0]
    length = float(x_max - x_min)
    x = cell_centers(x_min, x_max, m_grid)
    dx = cell_spacing(x_min, x_max, m_grid)
    n_steps = exact_step_count(T, dt)

    positions, total_mass = cell_edge_quantile_positions(
        datum, x_min, x_max, n_particles)
    weights = np.full(n_particles, total_mass / n_particles)
    eps_abs = eps_rel * total_mass / length

    n_modes = neumann_mode_count(bandwidth, length, KERNEL_TOLERANCE)
    b0 = certified_b0(n_modes, bandwidth, length, total_mass)
    b1 = certified_b1(n_modes, bandwidth, length, total_mass)

    snapshot_steps = sorted({0, n_steps // 2, n_steps - 1})
    snapshots: dict[int, np.ndarray] = {}

    min_raw_density = math.inf
    min_denominator = math.inf
    max_abs_score = 0.0
    result = CampaignRunResult(
        status="completed", failure_step=None, failure_message="",
        reconstruction=None,
    )

    for step in range(n_steps):
        if keep_snapshots and step in snapshot_steps:
            snapshots[step] = positions.copy()
        density, _deriv, score, _diag = neumann_kde_score(
            positions, positions, weights, x_min, x_max, bandwidth, eps_abs,
            tolerance=KERNEL_TOLERANCE,
        )
        denominator = density + eps_abs
        message = score_check_message(score, denominator)
        if message:
            result.status = "failed"
            result.failure_step = step
            result.failure_message = message
            break
        min_raw_density = min(min_raw_density, float(np.min(density)))
        min_denominator = min(min_denominator, float(np.min(denominator)))
        max_abs_score = max(max_abs_score, float(np.max(np.abs(score))))
        speed = alpha if a_of_x is None else a_of_x(positions)
        positions = apply_reflecting_boundary(
            positions + speed * score * dt, x_min, x_max)
        if not np.all(np.isfinite(positions)):
            result.status = "failed"
            result.failure_step = step
            result.failure_message = "nonfinite particle position"
            break

    result.snapshots = snapshots
    result.diagnostics = {
        "n_modes": n_modes,
        "first_omitted_multiplier": (math.exp(
            -0.5 * ((n_modes + 1) * math.pi * bandwidth / length) ** 2)
            if n_modes else 1.0),
        "B0": b0,
        "B1": b1,
        "eps_abs": eps_abs,
        "min_raw_density": (None if math.isinf(min_raw_density)
                            else min_raw_density),
        "min_regularized_denominator": (None if math.isinf(min_denominator)
                                        else min_denominator),
        "max_abs_score": max_abs_score,
        "total_mass": total_mass,
        "n_steps": n_steps,
    }
    result.gates = {"eps_floor": eps_abs >= EPS_FLOOR_FACTOR * b0}

    if result.status == "failed":
        return result

    # Tightened-truncation score comparison on the retained states.
    k_tight = tightened_mode_count(n_modes, bandwidth, length, total_mass)
    tol_tight = tolerance_for_modes(k_tight, bandwidth, length)
    tight_rel: dict[int, float] = {}
    tight_ok = True
    for step, state in snapshots.items():
        _, _, s_prod, _ = neumann_kde_score(
            state, state, weights, x_min, x_max, bandwidth, eps_abs,
            tolerance=KERNEL_TOLERANCE)
        _, _, s_tight, _ = neumann_kde_score(
            state, state, weights, x_min, x_max, bandwidth, eps_abs,
            tolerance=tol_tight)
        norm_tight = float(np.linalg.norm(s_tight))
        change = float(np.linalg.norm(s_prod - s_tight))
        if norm_tight > 0.0:
            rel = change / norm_tight
            tight_rel[step] = rel
            if rel > TIGHTENED_REL_TOL:
                tight_ok = False
        else:
            abs_change = float(np.max(np.abs(s_prod - s_tight)))
            tight_rel[step] = abs_change
            if abs_change > TIGHTENED_ABS_OVER_L / length:
                tight_ok = False
    result.diagnostics["n_modes_tightened"] = k_tight
    result.diagnostics["tightened_score_change"] = tight_rel
    result.gates["tightened_score"] = tight_ok
    if not tight_ok:
        result.status = "failed"
        result.failure_step = None
        result.failure_message = "tightened-truncation score comparison failed"
        return result

    reconstruction, _, _ = neumann_kde_density_derivative(
        x, positions, weights, x_min, x_max, bandwidth,
        tolerance=KERNEL_TOLERANCE)
    result.reconstruction = reconstruction

    metrics: dict[str, float] = {
        "mass_reconstruction": float(dx * np.sum(reconstruction)),
        "mass_error_rel": (abs(float(dx * np.sum(reconstruction)) - total_mass)
                           / total_mass if total_mass > 0 else float("nan")),
        "min_reconstruction": float(np.min(reconstruction)),
        "tv_reconstruction": float(np.sum(np.abs(np.diff(reconstruction)))),
    }
    if u0_reference is not None:
        u0 = np.asarray(u0_reference, dtype=float)
        diff = reconstruction - u0
        metrics["E2"] = midpoint_norm(diff, dx) / midpoint_norm(u0, dx)
        metrics["Linf_rel"] = float(np.max(np.abs(diff)) / np.max(np.abs(u0)))
        metrics["tv_reference"] = float(np.sum(np.abs(np.diff(u0))))
    if forward_residual is not None:
        metrics["forward_residual"] = float(forward_residual(reconstruction))
    result.metrics = metrics
    return result
