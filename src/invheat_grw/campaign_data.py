"""campaign_data.py — case fields and data builders for Phase 2C.

Analytic terminal and initial fields for the preregistered cases, the
conservative Crank--Nicolson finite-volume forward solver for variable
diffusivity, the de-crimed terminal-data builder of PHASE2C_PROTOCOL.md
Section 7, and the paired-noise and positivity-projection bookkeeping of
Sections 4 and 6.

Sampling conventions (load-bearing; see CAMPAIGN_GATE_LEDGER.md item 12):
- cosine primaries and the approximate-Neumann Gaussian secondaries use
  point values at cell centers, preserving the exact DCT-II modal
  correspondence;
- variable-coefficient production data use analytic cell averages on the
  fine grid, a conservative fine-to-coarse cell average, and truth given by
  analytic cell averages of the initial condition on the inverse grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.linalg import solve_banded
from scipy.special import erf

from .cell_grid import cell_centers, cell_spacing, midpoint_norm

SQRT2 = math.sqrt(2.0)


# ---------------------------------------------------------------------------
# Analytic fields (point values at cell centers)
# ---------------------------------------------------------------------------

def cosine_field(x: np.ndarray, background: float,
                 modes: Sequence[tuple[int, float]], *, length: float,
                 alpha: float = 0.0, t: float = 0.0) -> np.ndarray:
    """Exact Neumann cosine solution background + sum a_n cos(n pi x / L)
    evolved for time t (t = 0 gives the initial field)."""
    u = np.full_like(np.asarray(x, dtype=float), float(background))
    for n, amplitude in modes:
        k = n * math.pi / length
        u += amplitude * math.exp(-alpha * k * k * t) * np.cos(k * np.asarray(x))
    return u


def gaussian_free_space(x: np.ndarray, mu: float, sigma0: float,
                        amplitude: float, *, alpha: float = 0.0,
                        t: float = 0.0) -> np.ndarray:
    """Free-space heat evolution of an amplitude-A Gaussian (variance grows
    by 2 alpha t; the amplitude scales to conserve mass)."""
    variance = sigma0 * sigma0 + 2.0 * alpha * t
    scale = amplitude * sigma0 / math.sqrt(variance)
    return scale * np.exp(-(np.asarray(x, dtype=float) - mu) ** 2
                          / (2.0 * variance))


def mixture_free_space(x: np.ndarray, background: float,
                       components: Sequence[tuple[float, float, float]], *,
                       alpha: float = 0.0, t: float = 0.0) -> np.ndarray:
    """Constant background plus free-space Gaussians (A, mu, sigma0)."""
    u = np.full_like(np.asarray(x, dtype=float), float(background))
    for amplitude, mu, sigma0 in components:
        u += gaussian_free_space(x, mu, sigma0, amplitude, alpha=alpha, t=t)
    return u


# ---------------------------------------------------------------------------
# Analytic cell averages (variable-coefficient convention)
# ---------------------------------------------------------------------------

def gaussian_cell_averages(edges: np.ndarray, mu: float, sigma0: float,
                           amplitude: float) -> np.ndarray:
    """Exact cell averages of A exp(-(x-mu)^2/(2 sigma0^2)) between edges."""
    edges = np.asarray(edges, dtype=float)
    z = (edges - mu) / (sigma0 * SQRT2)
    antiderivative = amplitude * sigma0 * math.sqrt(math.pi / 2.0) * erf(z)
    return np.diff(antiderivative) / np.diff(edges)


def mixture_cell_averages(edges: np.ndarray, background: float,
                          components: Sequence[tuple[float, float, float]]
                          ) -> np.ndarray:
    u = np.full(len(edges) - 1, float(background))
    for amplitude, mu, sigma0 in components:
        u += gaussian_cell_averages(edges, mu, sigma0, amplitude)
    return u


def cell_edges(x_min: float, x_max: float, m: int) -> np.ndarray:
    dx = cell_spacing(x_min, x_max, m)
    return x_min + np.arange(m + 1) * dx


# ---------------------------------------------------------------------------
# Conservative Crank--Nicolson finite-volume forward solver
# ---------------------------------------------------------------------------

def variable_diffusivity(alpha0: float, beta: float,
                         *, x_min: float = 0.0,
                         length: float = 1.0) -> Callable[[np.ndarray], np.ndarray]:
    """a(x) = alpha0 (1 + beta sin(2 pi (x - x_min)/L))."""
    def a_of_x(z: np.ndarray) -> np.ndarray:
        return alpha0 * (1.0 + beta * np.sin(
            2.0 * math.pi * (np.asarray(z, dtype=float) - x_min) / length))
    return a_of_x


def flux_operator_diagonals(a_faces: np.ndarray, dx: float
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric tridiagonal operator (A u)_j = (F_{j+1/2}-F_{j-1/2})/dx with
    F = a_face (u_{j+1}-u_j)/dx and zero flux at both walls.  Returns
    (main diagonal, off diagonal)."""
    m = a_faces.shape[0] + 1
    off = a_faces / dx ** 2
    main = np.zeros(m)
    main[:-1] -= off
    main[1:] -= off
    return main, off


def cn_forward_solve(u0: np.ndarray, a_of_x: Callable[[np.ndarray], np.ndarray],
                     T: float, dt: float, *, x_min: float, x_max: float
                     ) -> np.ndarray:
    """Crank--Nicolson evolution of cell averages under the conservative
    variable-coefficient operator with analytic face coefficients."""
    u = np.asarray(u0, dtype=float).copy()
    m = u.shape[0]
    dx = cell_spacing(x_min, x_max, m)
    faces = x_min + np.arange(1, m) * dx
    a_faces = np.asarray(a_of_x(faces), dtype=float)
    if np.any(a_faces <= 0.0):
        raise ValueError("face diffusivities must be positive")
    n_steps = round(T / dt)
    if n_steps < 1 or abs(n_steps * dt - T) > 1e-12 * max(1.0, T):
        raise ValueError(f"time-step contract violated: {n_steps} * {dt} != {T}")
    main, off = flux_operator_diagonals(a_faces, dx)

    half = 0.5 * dt
    # Banded form of (I - half A) for solve_banded.
    ab = np.zeros((3, m))
    ab[0, 1:] = -half * off
    ab[1, :] = 1.0 - half * main
    ab[2, :-1] = -half * off
    for _ in range(n_steps):
        rhs = u * (1.0 + half * main)
        rhs[:-1] += half * off * u[1:]
        rhs[1:] += half * off * u[:-1]
        u = solve_banded((1, 1), ab, rhs)
    return u


def project_mean(u_fine: np.ndarray, factor: int) -> np.ndarray:
    """Conservative fine-to-coarse cell average by grouping `factor` cells."""
    u_fine = np.asarray(u_fine, dtype=float)
    if u_fine.shape[0] % factor:
        raise ValueError("fine grid is not a multiple of the coarse grid")
    return u_fine.reshape(-1, factor).mean(axis=1)


@dataclass(frozen=True)
class DecrimedData:
    terminal: np.ndarray          # projected 4x terminal data (clean)
    terminal_check: np.ndarray    # projected 8x terminal data
    truth_cell_averages: np.ndarray
    gate_difference: float        # relative L2 difference of the projections
    mass_discrepancy: float       # relative mass difference of the projections
    passes_gate: bool


def decrimed_variable_data(
    ic_cell_average_fn: Callable[[np.ndarray], np.ndarray],
    a_of_x: Callable[[np.ndarray], np.ndarray],
    T: float,
    *,
    x_min: float,
    x_max: float,
    m_inverse: int,
    fine: tuple[int, float],
    finer: tuple[int, float],
    gate: float,
) -> DecrimedData:
    """Protocol Section 7: independently refined forward data, conservatively
    projected, with the convergence gate recorded."""
    dx = cell_spacing(x_min, x_max, m_inverse)
    projections = []
    for m_fine, dt_fine in (fine, finer):
        if m_fine % m_inverse:
            raise ValueError("fine resolution must refine the inverse grid")
        edges = cell_edges(x_min, x_max, m_fine)
        u0_fine = ic_cell_average_fn(edges)
        terminal_fine = cn_forward_solve(
            u0_fine, a_of_x, T, dt_fine, x_min=x_min, x_max=x_max)
        projections.append(project_mean(terminal_fine, m_fine // m_inverse))
    g4, g8 = projections
    gate_difference = midpoint_norm(g4 - g8, dx) / midpoint_norm(g8, dx)
    mass4 = float(dx * np.sum(g4))
    mass8 = float(dx * np.sum(g8))
    mass_discrepancy = abs(mass4 - mass8) / abs(mass8)
    truth = ic_cell_average_fn(cell_edges(x_min, x_max, m_inverse))
    return DecrimedData(
        terminal=g4,
        terminal_check=g8,
        truth_cell_averages=truth,
        gate_difference=float(gate_difference),
        mass_discrepancy=float(mass_discrepancy),
        passes_gate=bool(gate_difference <= gate),
    )


# ---------------------------------------------------------------------------
# Paired noise and positivity projection (protocol Sections 4 and 6)
# ---------------------------------------------------------------------------

def apply_noise(g_clean: np.ndarray, eta: float, xi: np.ndarray) -> np.ndarray:
    """d = g + eta * max(g) * xi with a shared standard-normal vector xi."""
    g_clean = np.asarray(g_clean, dtype=float)
    if xi.shape != g_clean.shape:
        raise ValueError("noise vector shape must match the datum")
    return g_clean + eta * float(np.max(g_clean)) * np.asarray(xi, dtype=float)


def projection_stats(d: np.ndarray, dx: float) -> dict:
    """Common nonnegative projection P+(d) with the recorded diagnostics."""
    d = np.asarray(d, dtype=float)
    projected = np.maximum(d, 0.0)
    negative = d < 0.0
    mass_before = float(dx * np.sum(d))
    mass_after = float(dx * np.sum(projected))
    return {
        "projected": projected,
        "negative_fraction": float(np.mean(negative)),
        "negative_mass_removed": float(dx * np.sum(np.abs(d[negative]))),
        "mass_change_rel": ((mass_after - mass_before) / abs(mass_before)
                            if mass_before != 0.0 else float("nan")),
    }


def nominal_delta(eta: float, g_clean: np.ndarray, d_arm: np.ndarray,
                  dx: float, length: float) -> float:
    """delta_nom = eta * max(g_clean) * sqrt(L) / ||d_arm||_2 (Section 6)."""
    return (eta * float(np.max(np.asarray(g_clean)))
            * math.sqrt(length) / midpoint_norm(np.asarray(d_arm), dx))


def real_delta(g_clean: np.ndarray, d_arm: np.ndarray, dx: float) -> float:
    """Simulation diagnostic ||d - g||/||d||; never used for selection."""
    d_arm = np.asarray(d_arm, dtype=float)
    return (midpoint_norm(d_arm - np.asarray(g_clean, dtype=float), dx)
            / midpoint_norm(d_arm, dx))
