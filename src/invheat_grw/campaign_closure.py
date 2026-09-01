"""campaign_closure.py — gradient-carrier machinery for Phase 2C.

Implements PHASE2C_PROTOCOL.md Section 8: the two reconstruction closures,
the closure-aware smoothed score S_{h,eps}[U], deterministic transport of
signed carriers with the split-invariance construction, the finite-volume
reference solver (piecewise-linear monotonized-central reconstruction, local
Rusanov flux, zero numerical flux at the walls, SSPRK3 time stepping) for the
closure-matched regularized equation and the unregularized wrong-limit
equation, the exact four-field error decomposition with its squared-norm
reconciliation, and the analytic frozen-left closure anchor.

Positivity of the reconstructed field is required at every carrier and
reference stage; loss of positivity is a failed run and is never repaired by
clipping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .cell_grid import (
    cell_centers,
    cell_spacing,
    dct2_forward,
    midpoint_norm,
    wave_numbers,
)
from .globs import apply_reflecting_boundary

CLOSURES = ("frozen_left", "mass")


class ClosureError(RuntimeError):
    """Raised for contract violations in the closure machinery."""


# ---------------------------------------------------------------------------
# Closure reconstruction of U from a cell-average gradient field q
# ---------------------------------------------------------------------------

def closure_constant(q: np.ndarray, closure: str, *, dx: float,
                     anchor_value: float, total_mass: float,
                     length: float) -> float:
    """Additive constant of U(x) = C + int q for the requested closure.

    frozen_left: U at the first cell center equals anchor_value (the observed
    terminal first-cell value, held fixed in reverse time).
    mass: dx * sum(U at centers) equals total_mass at every time.
    """
    q = np.asarray(q, dtype=float)
    s = dx * (np.concatenate([[0.0], np.cumsum(q)[:-1]]) + 0.5 * q)
    if closure == "frozen_left":
        return float(anchor_value - s[0])
    if closure == "mass":
        return float((total_mass - dx * np.sum(s)) / length)
    raise ClosureError(f"unknown closure {closure!r}")


def reconstruct_centers(q: np.ndarray, constant: float, dx: float) -> np.ndarray:
    """U at cell centers from cell-average q and the additive constant."""
    q = np.asarray(q, dtype=float)
    s = dx * (np.concatenate([[0.0], np.cumsum(q)[:-1]]) + 0.5 * q)
    return constant + s


def reconstruct_edges(q: np.ndarray, constant: float, dx: float) -> np.ndarray:
    """U at cell edges (piecewise-linear integral of the cell averages)."""
    q = np.asarray(q, dtype=float)
    return constant + dx * np.concatenate([[0.0], np.cumsum(q)])


# ---------------------------------------------------------------------------
# Closure-aware smoothed score S_{h,eps}[U] via the cosine representation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SmoothedField:
    coefficients: np.ndarray      # damped orthonormal DCT-II coefficients
    x_min: float
    length: float
    m: int

    def values(self, x_eval: np.ndarray) -> np.ndarray:
        k = wave_numbers(self.m, self.length)
        theta = (np.asarray(x_eval, dtype=float)[:, None] - self.x_min) * k[None, 1:]
        out = np.full(len(np.atleast_1d(x_eval)),
                      self.coefficients[0] / math.sqrt(self.m))
        out += math.sqrt(2.0 / self.m) * (np.cos(theta) @ self.coefficients[1:])
        return out

    def derivative(self, x_eval: np.ndarray) -> np.ndarray:
        k = wave_numbers(self.m, self.length)
        theta = (np.asarray(x_eval, dtype=float)[:, None] - self.x_min) * k[None, 1:]
        return -math.sqrt(2.0 / self.m) * (
            np.sin(theta) @ (self.coefficients[1:] * k[1:]))


def smooth_field(u_centers: np.ndarray, bandwidth: float, *, x_min: float,
                 x_max: float) -> SmoothedField:
    """Neumann-kernel smoothing of a cell-centered field (DCT-II damping)."""
    u_centers = np.asarray(u_centers, dtype=float)
    m = u_centers.shape[0]
    length = x_max - x_min
    k = wave_numbers(m, length)
    damped = dct2_forward(u_centers) * np.exp(-0.5 * (bandwidth * k) ** 2)
    return SmoothedField(damped, x_min, length, m)


def smoothed_center_values_and_derivative(
    u_centers: np.ndarray, bandwidth: float, *, length: float
) -> tuple[np.ndarray, np.ndarray]:
    """(K_h U, d/dx K_h U) at the cell centers in O(M log M).

    The derivative uses the exact identity sin(k_n x_j) = (-1)^j
    cos(k_{M-n} x_j) at cell centers, so the sine series is a cosine series
    with reversed coefficients; a test verifies equivalence with the dense
    evaluation, so this is an algebraically equivalent optimization.
    """
    from .cell_grid import dct2_inverse
    u_centers = np.asarray(u_centers, dtype=float)
    m = u_centers.shape[0]
    k = wave_numbers(m, length)
    damped = dct2_forward(u_centers) * np.exp(-0.5 * (bandwidth * k) ** 2)
    values = dct2_inverse(damped)
    reversed_coeff = np.zeros(m)
    reversed_coeff[1:] = (damped * k)[m - 1:0:-1]
    signs = np.where(np.arange(m) % 2 == 0, 1.0, -1.0)
    derivative = -signs * dct2_inverse(reversed_coeff)
    return values, derivative


def carrier_score(u_centers: np.ndarray, bandwidth: float, eps_abs: float, *,
                  x_min: float, x_max: float
                  ) -> Callable[[np.ndarray], np.ndarray]:
    """S_{h,eps}[U](x) = d/dx (K_h U)(x) / ((K_h U)(x) + eps_abs)."""
    smooth = smooth_field(u_centers, bandwidth, x_min=x_min, x_max=x_max)

    def score(x_eval: np.ndarray) -> np.ndarray:
        return smooth.derivative(x_eval) / (smooth.values(x_eval) + eps_abs)

    return score


# ---------------------------------------------------------------------------
# Deterministic transport of signed carriers
# ---------------------------------------------------------------------------

@dataclass
class CarrierRunResult:
    status: str
    failure_step: int | None
    failure_message: str
    u_final: np.ndarray | None    # closure reconstruction on cell centers
    q_final: np.ndarray | None    # binned cell-average gradient field
    min_u: float
    snapshots: dict | None = None  # reverse time -> (u, q) at archived times


def _snapshot_steps(snapshot_times, dt: float, n_steps: int, T: float) -> dict:
    """Map archived reverse times to exact step indices (contract-checked)."""
    wanted: dict[int, float] = {}
    for t_snap in snapshot_times:
        k = round(t_snap / dt)
        if k < 1 or k > n_steps or abs(k * dt - t_snap) > 1e-12 * max(1.0, T):
            raise ClosureError(
                f"snapshot time {t_snap} is not an exact step multiple of "
                f"dt = {dt}")
        wanted[k] = float(t_snap)
    return wanted


def run_gradient_carriers(
    g_grid: np.ndarray,
    *,
    x_min: float,
    x_max: float,
    T: float,
    dt: float,
    bandwidth: float,
    eps_rel: float,
    alpha: float,
    closure: str,
    subparticles: int = 1,
    snapshot_times: Sequence[float] = (),
) -> CarrierRunResult:
    """Transport one signed carrier per initial grid jump (optionally split
    into coincident equal-weight subparticles) by alpha * S_{h,eps}[U]."""
    g_grid = np.asarray(g_grid, dtype=float)
    m = g_grid.shape[0]
    dx = cell_spacing(x_min, x_max, m)
    length = x_max - x_min
    x = cell_centers(x_min, x_max, m)
    n_steps = round(T / dt)
    if n_steps < 1 or abs(n_steps * dt - T) > 1e-12 * max(1.0, T):
        raise ClosureError(f"time-step contract violated: {n_steps} * {dt} != {T}")
    if subparticles < 1:
        raise ClosureError("subparticles must be at least one")
    wanted = _snapshot_steps(snapshot_times, dt, n_steps, T)

    edges = x_min + np.arange(1, m) * dx
    jumps = np.diff(g_grid)
    positions = np.repeat(edges, subparticles)
    weights = np.repeat(jumps / subparticles, subparticles)
    anchor_value = float(g_grid[0])
    total_mass = float(dx * np.sum(g_grid))
    eps_abs = eps_rel * total_mass / length

    def reconstruct(pos: np.ndarray) -> np.ndarray:
        order = np.argsort(pos, kind="stable")
        csum = np.concatenate([[0.0], np.cumsum(weights[order])])
        idx = np.searchsorted(pos[order], x, side="right")
        u = anchor_value + csum[idx]
        if closure == "mass":
            u = u + (total_mass - dx * np.sum(u)) / length
        elif closure != "frozen_left":
            raise ClosureError(f"unknown closure {closure!r}")
        return u

    def bin_q(pos: np.ndarray) -> np.ndarray:
        counts = np.zeros(m)
        cell_index = np.clip(((pos - x_min) / dx).astype(int), 0, m - 1)
        np.add.at(counts, cell_index, weights)
        return counts / dx

    snapshots: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    min_u = math.inf
    for step in range(n_steps):
        u = reconstruct(positions)
        min_u = min(min_u, float(np.min(u)))
        if min_u <= 0.0:
            return CarrierRunResult("failed", step, "loss of positivity of U",
                                    None, None, min_u)
        if step in wanted:
            snapshots[wanted[step]] = (u.copy(), bin_q(positions))
        score = carrier_score(u, bandwidth, eps_abs, x_min=x_min, x_max=x_max)
        s_at = score(positions)
        if (not np.all(np.isfinite(s_at))) or np.max(np.abs(s_at)) > 1e6:
            return CarrierRunResult("failed", step, "invalid carrier score",
                                    None, None, min_u)
        positions = apply_reflecting_boundary(
            positions + alpha * s_at * dt, x_min, x_max)

    u_final = reconstruct(positions)
    min_u = min(min_u, float(np.min(u_final)))
    q_final = bin_q(positions)
    if n_steps in wanted:
        snapshots[wanted[n_steps]] = (u_final.copy(), q_final.copy())
    return CarrierRunResult("completed", None, "", u_final, q_final, min_u,
                            snapshots)


# ---------------------------------------------------------------------------
# Finite-volume reference solver (MC limiter, Rusanov, SSPRK3)
# ---------------------------------------------------------------------------

def mc_slopes(q: np.ndarray) -> np.ndarray:
    """Monotonized-central limited slopes (undivided, per cell)."""
    q = np.asarray(q, dtype=float)
    slopes = np.zeros_like(q)
    left = q[1:-1] - q[:-2]
    right = q[2:] - q[1:-1]
    center = 0.5 * (left + right)
    sign = np.sign(center)
    agree = (np.sign(left) == np.sign(right)) & (left != 0.0)
    limited = sign * np.minimum(np.abs(center),
                                np.minimum(2.0 * np.abs(left),
                                           2.0 * np.abs(right)))
    slopes[1:-1] = np.where(agree, limited, 0.0)
    return slopes


@dataclass
class ReferenceRunResult:
    status: str
    failure_message: str
    q: np.ndarray | None
    u: np.ndarray | None
    max_cfl: float
    speed_bound: float
    min_u: float
    snapshots: dict | None = None  # reverse time -> (u, q) at archived times


def run_reference(
    q0: np.ndarray,
    *,
    kind: str,
    closure: str,
    anchor_value: float,
    total_mass: float,
    x_min: float,
    x_max: float,
    T: float,
    dt: float,
    alpha: float,
    bandwidth: float = 0.0,
    eps_rel: float = 0.0,
    snapshot_times: Sequence[float] = (),
) -> ReferenceRunResult:
    """Solve q_t = -d/dx f by MUSCL/Rusanov/SSPRK3 with zero wall flux.

    kind = 'regularized':  f = alpha * q * S_{h,eps}[U]
    kind = 'unregularized': f = alpha * q^2 / U
    with U reconstructed from q under the closure at every Runge--Kutta stage.
    """
    q = np.asarray(q0, dtype=float).copy()
    m = q.shape[0]
    dx = cell_spacing(x_min, x_max, m)
    length = x_max - x_min
    edges = x_min + np.arange(m + 1) * dx
    n_steps = round(T / dt)
    if n_steps < 1 or abs(n_steps * dt - T) > 1e-12 * max(1.0, T):
        raise ClosureError(f"time-step contract violated: {n_steps} * {dt} != {T}")
    if kind not in ("regularized", "unregularized"):
        raise ClosureError(f"unknown reference kind {kind!r}")
    wanted = _snapshot_steps(snapshot_times, dt, n_steps, T)
    eps_abs = eps_rel * total_mass / length

    max_cfl = 0.0
    speed_bound = 0.0
    min_u_seen = math.inf

    def rhs(state: np.ndarray) -> tuple[np.ndarray | None, str, float, float]:
        nonlocal min_u_seen
        constant = closure_constant(state, closure, dx=dx,
                                    anchor_value=anchor_value,
                                    total_mass=total_mass, length=length)
        u_centers = reconstruct_centers(state, constant, dx)
        min_u = float(np.min(u_centers))
        min_u_seen = min(min_u_seen, min_u)
        if min_u <= 0.0:
            return None, "loss of positivity of U", 0.0, 0.0
        slopes = mc_slopes(state)
        q_left = state + 0.5 * slopes          # left state of interface j+1/2
        q_right = state - 0.5 * slopes         # right state of interface j-1/2
        ql = q_left[:-1]
        qr = q_right[1:]
        if kind == "regularized":
            smooth_values, smooth_derivative = (
                smoothed_center_values_and_derivative(
                    u_centers, bandwidth, length=length))
            s_centers = smooth_derivative / (smooth_values + eps_abs)
            v_edges = 0.5 * (s_centers[:-1] + s_centers[1:])
            bound = float(np.max(
                alpha * (np.abs(s_centers)
                         + 2.0 * np.abs(state)
                         / (smooth_values + eps_abs))))
            flux = (0.5 * alpha * v_edges * (ql + qr)
                    - 0.5 * bound * (qr - ql))
        else:
            u_edges = reconstruct_edges(state, constant, dx)[1:-1]
            if float(np.min(u_edges)) <= 0.0:
                return None, "loss of positivity of U at an interface", 0.0, 0.0
            local = np.maximum(np.abs(2.0 * alpha * ql / u_edges),
                               np.abs(2.0 * alpha * qr / u_edges))
            bound = float(np.max(local))
            flux = (0.5 * alpha * (ql ** 2 + qr ** 2) / u_edges
                    - 0.5 * local * (qr - ql))
        full_flux = np.concatenate([[0.0], flux, [0.0]])   # zero wall flux
        return -np.diff(full_flux) / dx, "", bound, bound * dt / dx

    snapshots: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for _step in range(n_steps):
        stage_state = q
        stages = []
        for _stage in range(3):
            out, message, bound, cfl = rhs(stage_state)
            if out is None:
                return ReferenceRunResult("failed", message, None, None,
                                          max_cfl, speed_bound, min_u_seen)
            speed_bound = max(speed_bound, bound)
            max_cfl = max(max_cfl, cfl)
            stages.append(out)
            if _stage == 0:
                stage_state = q + dt * stages[0]
            elif _stage == 1:
                stage_state = 0.75 * q + 0.25 * (stage_state + dt * stages[1])
        q = q / 3.0 + (2.0 / 3.0) * (stage_state + dt * stages[2])
        if _step + 1 in wanted:
            constant = closure_constant(q, closure, dx=dx,
                                        anchor_value=anchor_value,
                                        total_mass=total_mass, length=length)
            snapshots[wanted[_step + 1]] = (
                reconstruct_centers(q, constant, dx), q.copy())

    constant = closure_constant(q, closure, dx=dx, anchor_value=anchor_value,
                                total_mass=total_mass, length=length)
    u_final = reconstruct_centers(q, constant, dx)
    return ReferenceRunResult("completed", "", q, u_final, max_cfl,
                              speed_bound, min_u_seen, snapshots)


# ---------------------------------------------------------------------------
# Exact four-field decomposition and analytic anchor
# ---------------------------------------------------------------------------

def exact_decomposition(u_truth: np.ndarray, u0_mass: np.ndarray,
                        u0_c: np.ndarray, u_heps_c: np.ndarray,
                        u_particle_c: np.ndarray, dx: float) -> dict:
    """Protocol Section 8 identity with all six pairwise inner products and
    the squared-norm reconciliation residual."""
    e = [np.asarray(u0_mass) - np.asarray(u_truth),
         np.asarray(u0_c) - np.asarray(u0_mass),
         np.asarray(u_heps_c) - np.asarray(u0_c),
         np.asarray(u_particle_c) - np.asarray(u_heps_c)]
    total = np.asarray(u_particle_c) - np.asarray(u_truth)
    names = ("wrong_transport", "closure", "score_regularization",
             "particle_discretization")
    norms = {name: midpoint_norm(component, dx)
             for name, component in zip(names, e)}
    inner = {}
    for i in range(4):
        for j in range(i + 1, 4):
            inner[f"{names[i]}|{names[j]}"] = float(dx * np.dot(e[i], e[j]))
    total_sq = midpoint_norm(total, dx) ** 2
    recon = sum(norms[n] ** 2 for n in names) + 2.0 * sum(inner.values())
    residual = abs(total_sq - recon) / max(total_sq, 1e-300)
    return {
        "total_norm": midpoint_norm(total, dx),
        "component_norms": norms,
        "inner_products": inner,
        "reconciliation_residual": float(residual),
        "identity_max_abs": float(np.max(np.abs(sum(e) - total))),
    }


def frozen_left_offset(a: float, alpha: float, T: float) -> float:
    """Signed frozen-left closure error for u0 = c + a cos(pi x)."""
    return -a * (1.0 - math.exp(-alpha * math.pi ** 2 * T))
