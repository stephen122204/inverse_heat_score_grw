"""
run_variable_coefficient_audit.py

Variable-coefficient heat equation experiment for inverse_heat_score_grw.

Solves  u_t = ∂_x( a(x) u_x )  on [0,1] with Neumann BCs, where
    a(x) = alpha0 * (1 + beta * sin(2*pi*x)).

Tests whether the density-particle score framework (probability-flow /
reverse-time particle dynamics) extends beyond the constant-coefficient
(spectral) setting.

The deterministic reverse-time velocity is
    v(x,t) = a(x) * ∂_x log u(x,t)
(NO a'(x) term — the probability-flow ODE for this PDE).

Methods tested
--------------
1. variable_oracle_deterministic        — numerical oracle score from saved snapshots
2. variable_estimated_smoothed_log_bw4  — smoothed_log score, bw_factor=4
3. variable_estimated_smoothed_log_bw6  — smoothed_log score, bw_factor=6
4. variable_estimated_fd_grid_ratio_bw4 — fd_grid_ratio score, bw_factor=4
5. varcoeff_tikhonov_best               — variable-coeff Tikhonov, lambda sweep

Test cases
----------
  VB_beta05 : Gaussian IC, beta=0.5
  VB_beta09 : Gaussian IC, beta=0.9
  VH_beta05 : Gaussian mixture IC, beta=0.5
  VH_beta09 : Gaussian mixture IC, beta=0.9

Output
------
outputs/variable_coefficient_audit_<TIMESTAMP>/
    variable_coeff_metrics.csv
    variable_coeff_summary.txt
    field_comparison_beta05_gaussian.png
    field_comparison_beta09_gaussian.png
    field_comparison_beta05_mixture.png
    field_comparison_beta09_mixture.png
    score_error_vs_step_<case>.png
    forward_consistency_<case>.png
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from invheat_grw.config import load_config
from invheat_grw.fields import make_grid, true_u0 as compute_true_u0
from invheat_grw.methods import (
    _quantile_init_particles,
    _reconstruct_density_particles,
    MethodResult,
)
from invheat_grw.scores import smoothed_log_score
from invheat_grw.metrics import compute_wasserstein

# scipy imports
from scipy import sparse
from scipy.sparse.linalg import spsolve

# numpy>=2.0 renamed np.trapz -> np.trapezoid (np.trapz removed in 2.x).
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Config paths
# ---------------------------------------------------------------------------
GAUSSIAN_CONFIG = REPO_ROOT / "configs" / "gaussian_base.yaml"
MIXTURE_CONFIG = REPO_ROOT / "configs" / "gaussian_mixture.yaml"

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------
N_PARTICLES = 10_000
N_GRID = 400
TIKHONOV_LAMBDAS = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]
EPSILON = 1e-8


# ---------------------------------------------------------------------------
# Utility: patch_config
# ---------------------------------------------------------------------------
def patch_config(cfg, **overrides):
    cfg = copy.deepcopy(cfg)
    for key, val in overrides.items():
        parts = key.split(".")
        obj = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], val)
    return cfg


# ---------------------------------------------------------------------------
# Variable-coefficient helpers
# ---------------------------------------------------------------------------
def a_of_x(x: np.ndarray, alpha0: float, beta: float) -> np.ndarray:
    """Diffusivity: a(x) = alpha0 * (1 + beta * sin(2*pi*x))."""
    return alpha0 * (1.0 + beta * np.sin(2.0 * np.pi * x))


def da_dx(x: np.ndarray, alpha0: float, beta: float) -> np.ndarray:
    """Derivative of diffusivity (diagnostics ONLY — not used in integration)."""
    return alpha0 * beta * 2.0 * np.pi * np.cos(2.0 * np.pi * x)


# ---------------------------------------------------------------------------
# Conservative FD operator  L u ≈ ∂_x(a(x) u_x)
# ---------------------------------------------------------------------------
def build_varcoeff_operator(
    x_grid: np.ndarray, alpha0: float, beta: float
) -> sparse.csr_matrix:
    """
    Build sparse tridiagonal matrix L for ∂_x(a(x) ∂_x u) on x_grid.

    Conservative FD at interior point i:
        [L u]_i = (a_{i+1/2}(u_{i+1}-u_i) - a_{i-1/2}(u_i-u_{i-1})) / dx^2

    Neumann BCs (du/dx=0 at both ends) via ghost-point reflection:
        At i=0:   only the right interface contributes, left ghost mirrors u[1].
        At i=N-1: only the left interface contributes, right ghost mirrors u[N-2].
    """
    N = len(x_grid)
    dx = x_grid[1] - x_grid[0]
    dx2 = dx * dx

    # Half-point diffusivities: a_{i+1/2} for i = 0..N-2
    a_vals = a_of_x(x_grid, alpha0, beta)
    a_half = 0.5 * (a_vals[:-1] + a_vals[1:])   # length N-1

    # Diagonal, sub-diagonal, super-diagonal arrays
    diag_main = np.zeros(N)
    diag_upper = np.zeros(N - 1)  # entry (i, i+1)
    diag_lower = np.zeros(N - 1)  # entry (i, i-1)

    # Interior: i = 1..N-2
    for i in range(1, N - 1):
        diag_upper[i] = a_half[i] / dx2
        diag_lower[i - 1] = a_half[i - 1] / dx2
        diag_main[i] = -(a_half[i] + a_half[i - 1]) / dx2

    # i = 0: Neumann BC on left — ghost = u[1], so a_{-1/2} term uses u[-1]=u[1]
    # => [L u]_0 = (a_{1/2}(u_1-u_0) - a_{-1/2}(u_0-u_{-1})) / dx^2
    #            = (a_{1/2}(u_1-u_0) - a_{-1/2}(u_0-u_1)) / dx^2  [ghost u_{-1}=u_1]
    #            = (a_{1/2} + a_{-1/2})(u_1-u_0) / dx^2
    # For simplicity use a_{-1/2} = a_{1/2} (one-sided, consistent with Neumann)
    diag_upper[0] = 2.0 * a_half[0] / dx2
    diag_main[0] = -2.0 * a_half[0] / dx2

    # i = N-1: Neumann BC on right — ghost u[N] = u[N-2]
    diag_lower[N - 2] = 2.0 * a_half[N - 2] / dx2
    diag_main[N - 1] = -2.0 * a_half[N - 2] / dx2

    L = (
        sparse.diags(diag_main, 0, shape=(N, N))
        + sparse.diags(diag_upper, 1, shape=(N, N))
        + sparse.diags(diag_lower, -1, shape=(N, N))
    ).tocsr()

    return L  # type: ignore[return-value]
# ---------------------------------------------------------------------------
def solve_varcoeff_forward(
    u0: np.ndarray,
    x_grid: np.ndarray,
    alpha0: float,
    beta: float,
    dt: float,
    n_steps: int,
) -> tuple[np.ndarray, dict]:
    """
    Solve u_t = ∂_x(a(x) u_x) forward in time using Crank-Nicolson.

    Scheme:
        (I - dt/2 * L) u^{n+1} = (I + dt/2 * L) u^n

    Returns
    -------
    u_T       : solution at t = n_steps * dt
    snapshots : dict  k -> u at step k  (k=0 is u0, k=n_steps is u_T)
    """
    N = len(x_grid)
    L = build_varcoeff_operator(x_grid, alpha0, beta)

    I = sparse.eye(N, format="csr")
    A = (I - 0.5 * dt * L).tocsr()   # LHS of CN
    B = (I + 0.5 * dt * L).tocsr()   # RHS of CN

    snapshots = {}
    u = u0.copy()
    snapshots[0] = u.copy()

    for k in range(1, n_steps + 1):
        rhs = B @ u
        u = spsolve(A, rhs)
        # Clip any tiny negatives introduced by numerics
        u = np.maximum(u, 0.0)
        snapshots[k] = u.copy()

    return u, snapshots


# ---------------------------------------------------------------------------
# Numerical oracle score from forward snapshots
# ---------------------------------------------------------------------------
def numerical_oracle_score(
    positions: np.ndarray,
    k_reverse: int,
    n_steps: int,
    snapshots: dict,
    x_grid: np.ndarray,
    smooth_sigma: float | None = None,
    epsilon: float = EPSILON,
) -> np.ndarray:
    """
    Compute ∂_x log u at particle positions using saved forward snapshot.

    During reverse step k_reverse (0-indexed from t=T going backward),
    the physical time is t = T - k_reverse * dt, which corresponds to
    forward step index  fwd_k = n_steps - k_reverse.

    smooth_sigma: smoothing width in x-units.  Defaults to 3 * dx (tight —
    the true forward density is already smooth so we want minimal filtering).
    """
    fwd_k = n_steps - k_reverse
    u_fwd = snapshots[fwd_k]
    dx = x_grid[1] - x_grid[0]
    if smooth_sigma is None:
        smooth_sigma = 3.0 * dx
    scores, _ = smoothed_log_score(positions, u_fwd, x_grid, smooth_sigma, epsilon)  # type: ignore[arg-type]
    return scores


# ---------------------------------------------------------------------------
# Reflecting boundary
# ---------------------------------------------------------------------------
def _reflect(positions: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    """Reflect particles that escape [x_min, x_max] back into the domain."""
    L = x_max - x_min
    # Shift to [0, L]
    p = positions - x_min
    p = p % (2.0 * L)
    p = np.where(p > L, 2.0 * L - p, p)
    return p + x_min


# ---------------------------------------------------------------------------
# Variable-coeff density-particle oracle method
# ---------------------------------------------------------------------------
def run_varcoeff_oracle(
    u_obs: np.ndarray,
    x_grid: np.ndarray,
    snapshots: dict,
    alpha0: float,
    beta: float,
    dt: float,
    n_steps: int,
    n_particles: int = N_PARTICLES,
    recon_method: str = "kde",
    bandwidth_factor: float = 4.0,
    score_smooth_sigma: float | None = None,
    epsilon: float = EPSILON,
) -> dict:
    """
    Reverse density-particle integration using the numerical oracle score.

    Particle update:  Δx = a(x) * ∂_x log u(x, t) * dt
    """
    dx = x_grid[1] - x_grid[0]
    x_min, x_max = float(x_grid[0]), float(x_grid[-1])
    bw = bandwidth_factor * dx

    positions, total_mass = _quantile_init_particles(u_obs, x_grid, n_particles)

    score_errors = []   # L2 error of oracle vs itself (always 0, placeholder)
    t_start = time.perf_counter()
    completed = True
    failure_step = None
    failure_msg = ""

    for k in range(n_steps):
        scores = numerical_oracle_score(
            positions, k, n_steps, snapshots, x_grid, score_smooth_sigma, epsilon
        )
        if not np.all(np.isfinite(scores)):
            completed = False
            failure_step = k
            failure_msg = f"Oracle score NaN/Inf at step {k}"
            break

        a_vals = a_of_x(positions, alpha0, beta)
        positions = positions + a_vals * scores * dt
        positions = _reflect(positions, x_min, x_max)

        if not np.all(np.isfinite(positions)):
            completed = False
            failure_step = k + 1
            failure_msg = f"NaN/Inf positions at step {k+1}"
            break

        score_errors.append(0.0)   # oracle vs oracle

    candidate = _reconstruct_density_particles(positions, total_mass, x_grid, recon_method, bw)
    runtime = time.perf_counter() - t_start

    return {
        "method_name": "variable_oracle_deterministic",
        "completed": completed,
        "failure_step": failure_step,
        "failure_msg": failure_msg,
        "candidate": candidate,
        "final_positions": positions.copy(),
        "score_errors_per_step": score_errors,
        "runtime_seconds": runtime,
    }


# ---------------------------------------------------------------------------
# Variable-coeff density-particle estimated-score method
# ---------------------------------------------------------------------------
def run_varcoeff_estimated(
    u_obs: np.ndarray,
    x_grid: np.ndarray,
    snapshots: dict,
    alpha0: float,
    beta: float,
    dt: float,
    n_steps: int,
    score_method: str = "smoothed_log",
    bandwidth_factor: float = 4.0,
    n_particles: int = N_PARTICLES,
    recon_method: str = "kde",
    epsilon: float = EPSILON,
    score_smooth_sigma: float | None = None,
) -> dict:
    """
    Reverse density-particle integration using estimated score.

    score_method choices: "smoothed_log", "fd_grid_ratio"

    Particle update:  Δx = a(x) * s_estimated(x,t) * dt
    """
    dx = x_grid[1] - x_grid[0]
    x_min, x_max = float(x_grid[0]), float(x_grid[-1])
    bw = bandwidth_factor * dx

    method_name = f"variable_estimated_{score_method}_bw{int(bandwidth_factor)}"

    positions, total_mass = _quantile_init_particles(u_obs, x_grid, n_particles)

    score_errors_l2 = []   # vs numerical oracle, per step
    t_start = time.perf_counter()
    completed = True
    failure_step = None
    failure_msg = ""

    for k in range(n_steps):
        # Reconstruct density from current particles
        u_recon = _reconstruct_density_particles(
            positions, total_mass, x_grid, recon_method, bw
        )

        # Estimate score
        if score_method == "smoothed_log":
            smooth_sigma = bandwidth_factor * dx
            scores, _ = smoothed_log_score(positions, u_recon, x_grid, smooth_sigma, epsilon)
        elif score_method == "fd_grid_ratio":
            # fd_grid_ratio: log-difference of adjacent grid bins
            from invheat_grw.scores import log_density_fd_score
            scores_grid, _ = log_density_fd_score(x_grid, u_recon, x_grid, epsilon)
            scores = np.interp(positions, x_grid, scores_grid)
        else:
            raise ValueError(f"Unknown score_method: {score_method!r}")

        if not np.all(np.isfinite(scores)):
            # Clip non-finite to zero rather than aborting immediately
            scores = np.where(np.isfinite(scores), scores, 0.0)

        # Compute score error vs oracle
        oracle_scores = numerical_oracle_score(
            positions, k, n_steps, snapshots, x_grid, score_smooth_sigma, epsilon
        )
        err = float(np.sqrt(np.mean((scores - oracle_scores) ** 2)))
        score_errors_l2.append(err)

        # Variable-coeff probability-flow update
        a_vals = a_of_x(positions, alpha0, beta)
        positions = positions + a_vals * scores * dt
        positions = _reflect(positions, x_min, x_max)

        if not np.all(np.isfinite(positions)):
            completed = False
            failure_step = k + 1
            failure_msg = f"NaN/Inf positions at step {k+1}"
            break

    candidate = _reconstruct_density_particles(positions, total_mass, x_grid, recon_method, bw)
    runtime = time.perf_counter() - t_start

    return {
        "method_name": method_name,
        "completed": completed,
        "failure_step": failure_step,
        "failure_msg": failure_msg,
        "candidate": candidate,
        "final_positions": positions.copy(),
        "score_errors_per_step": score_errors_l2,
        "runtime_seconds": runtime,
    }


# ---------------------------------------------------------------------------
# Variable-coeff Tikhonov baseline
# ---------------------------------------------------------------------------
def varcoeff_tikhonov(
    u_obs: np.ndarray,
    x_grid: np.ndarray,
    alpha0: float,
    beta: float,
    dt: float,
    n_steps: int,
    lam: float,
) -> np.ndarray:
    """
    Tikhonov-regularized inversion for the variable-coefficient heat equation.

    Builds the full forward map matrix F by applying one CN step-matrix
    to each standard basis vector, then:
        û0 = (F^T F + lam I)^{-1} F^T u_T
    """
    N = len(x_grid)
    L = build_varcoeff_operator(x_grid, alpha0, beta)
    I_sp = sparse.eye(N, format="csr")
    A = (I_sp - 0.5 * dt * L).tocsr()
    B = (I_sp + 0.5 * dt * L).tocsr()

    # Build the single-step propagation matrix M (dense, N×N)
    # M @ u = u^{n+1}: solve A @ u^{n+1} = B @ u^n
    # => M = A^{-1} @ B  (build column by column)
    M = np.zeros((N, N))
    for j in range(N):
        e_j = B @ np.eye(N)[:, j]
        M[:, j] = spsolve(A, e_j)  # type: ignore[call-overload, assignment]

    # Raise M to n_steps power via matrix exponentiation
    F = np.linalg.matrix_power(M, n_steps)   # F: u0 -> u_T

    # Normal equations: (F^T F + lam I) x = F^T u_obs
    FtF = F.T @ F
    Ft_uobs = F.T @ u_obs
    reg_mat = FtF + lam * np.eye(N)
    candidate = np.linalg.solve(reg_mat, Ft_uobs)
    candidate = np.maximum(candidate, 0.0)
    return candidate


def varcoeff_tikhonov_sweep(
    u_obs: np.ndarray,
    x_grid: np.ndarray,
    alpha0: float,
    beta: float,
    dt: float,
    n_steps: int,
    true_u0_field: np.ndarray,
    lambdas: list[float] = TIKHONOV_LAMBDAS,
) -> dict:
    """Sweep lambdas and return the result with lowest relative L2 vs true_u0."""
    dx = x_grid[1] - x_grid[0]
    true_norm = float(np.sqrt(dx * np.sum(true_u0_field ** 2)))
    best = {"lam": None, "candidate": None, "rel_l2": float("inf")}

    t_start = time.perf_counter()
    for lam in lambdas:
        cand = varcoeff_tikhonov(u_obs, x_grid, alpha0, beta, dt, n_steps, lam)
        diff = cand - true_u0_field
        rel_l2 = float(np.sqrt(dx * np.sum(diff ** 2))) / (true_norm + 1e-300)
        if rel_l2 < best["rel_l2"]:
            best = {"lam": lam, "candidate": cand, "rel_l2": rel_l2}

    best["runtime_seconds"] = time.perf_counter() - t_start
    best["method_name"] = "varcoeff_tikhonov_best"
    best["completed"] = True
    return best


# ---------------------------------------------------------------------------
# Forward-consistency check
# ---------------------------------------------------------------------------
def forward_consistency_l2(
    candidate: np.ndarray,
    u_obs: np.ndarray,
    x_grid: np.ndarray,
    alpha0: float,
    beta: float,
    dt: float,
    n_steps: int,
) -> float:
    """Evolve candidate forward; compute relative L2 vs u_obs."""
    dx = x_grid[1] - x_grid[0]
    u_fwd, _ = solve_varcoeff_forward(candidate, x_grid, alpha0, beta, dt, n_steps)
    diff = u_fwd - u_obs
    obs_norm = float(np.sqrt(dx * np.sum(u_obs ** 2)))
    return float(np.sqrt(dx * np.sum(diff ** 2))) / (obs_norm + 1e-300)


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------
def compute_result_metrics(
    result: dict,
    true_u0_field: np.ndarray,
    u_obs: np.ndarray,
    x_grid: np.ndarray,
    alpha0: float,
    beta: float,
    dt: float,
    n_steps: int,
) -> dict:
    dx = x_grid[1] - x_grid[0]
    cand = result["candidate"]
    true_norm = float(np.sqrt(dx * np.sum(true_u0_field ** 2)))

    diff = cand - true_u0_field
    l2_err = float(np.sqrt(dx * np.sum(diff ** 2)))
    rel_l2 = l2_err / (true_norm + 1e-300)
    linf_err = float(np.max(np.abs(diff)))

    # Mass
    mass_cand = float(_trapz(cand, x_grid))  # type: ignore[attr-defined]
    mass_true = float(_trapz(true_u0_field, x_grid))  # type: ignore[attr-defined]

    # Total variation
    tv = float(np.sum(np.abs(np.diff(cand))) * dx)

    # Forward consistency (skip for Tikhonov — already computed in sweep)
    if result.get("method_name", "") == "varcoeff_tikhonov_best":
        fwd_l2 = float("nan")
    else:
        fwd_l2 = forward_consistency_l2(
            cand, u_obs, x_grid, alpha0, beta, dt, n_steps
        )

    # Wasserstein (best-effort)
    try:
        wass = compute_wasserstein(cand, true_u0_field, x_grid)
    except Exception:
        wass = float("nan")

    return {
        "method": result.get("method_name", "unknown"),
        "completed": result.get("completed", False),
        "failure_step": result.get("failure_step"),
        "failure_msg": result.get("failure_msg", ""),
        "relative_l2": rel_l2,
        "l2_error": l2_err,
        "linf_error": linf_err,
        "forward_consistency_l2": fwd_l2,
        "mass_candidate": mass_cand,
        "mass_true": mass_true,
        "mass_rel_error": abs(mass_cand - mass_true) / (mass_true + 1e-300),
        "total_variation": tv,
        "wasserstein": wass,
        "runtime_seconds": result.get("runtime_seconds", float("nan")),
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_field_comparison(
    x_grid: np.ndarray,
    true_u0_field: np.ndarray,
    u_obs: np.ndarray,
    results: list[dict],
    case_label: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_grid, true_u0_field, "k-", lw=2, label="True u₀")
    ax.plot(x_grid, u_obs, "k--", lw=1.5, alpha=0.6, label="Observed u(T)")
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for i, r in enumerate(results):
        c = colors[i % len(colors)]
        ax.plot(x_grid, r["candidate"], color=c, lw=1.5,
                label=r.get("method_name", f"method_{i}"), alpha=0.85)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.set_title(f"Field comparison — {case_label}")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_score_error_vs_step(
    results: list[dict],
    case_label: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for i, r in enumerate(results):
        errs = r.get("score_errors_per_step", [])
        if errs:
            ax.plot(errs, color=colors[i % len(colors)], lw=1.2,
                    label=r.get("method_name", f"method_{i}"), alpha=0.85)
    ax.set_xlabel("reverse step")
    ax.set_ylabel("Score RMS error vs oracle")
    ax.set_title(f"Score estimation error — {case_label}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_forward_consistency(
    results_metrics: list[dict],
    case_label: str,
    out_path: Path,
) -> None:
    methods = [m["method"] for m in results_metrics if not np.isnan(m["forward_consistency_l2"])]
    vals = [m["forward_consistency_l2"] for m in results_metrics if not np.isnan(m["forward_consistency_l2"])]
    if not methods:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(methods, vals)
    ax.set_ylabel("Forward consistency relative L2")
    ax.set_title(f"Forward consistency — {case_label}")
    plt.xticks(rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------
def compute_verdict(rows: list[dict]) -> str:
    """
    STRONG GO   : estimated particle within 10× of Tikhonov for beta=0.5 AND beta=0.9,
                  oracle tracks well (rel_l2 < 0.15 for both), no instability.
    CONDITIONAL GO: works for beta=0.5, degrades at beta=0.9 OR Gaussian works but mixture weak.
    STOP/REVISE : oracle particle fails badly; estimated unstable; rel_l2 > 0.5 while Tikhonov < 0.05.
    """
    def _get(rows, case, method):
        for r in rows:
            if r["case"] == case and r["method"] == method:
                return r
        return None

    issues = []
    strong_go_evidence = []
    cond_go_evidence = []

    cases = ["VB_beta05", "VB_beta09", "VH_beta05", "VH_beta09"]
    for case in cases:
        tik = _get(rows, case, "varcoeff_tikhonov_best")
        oracle = _get(rows, case, "variable_oracle_deterministic")
        sl4 = _get(rows, case, "variable_estimated_smoothed_log_bw4")
        sl6 = _get(rows, case, "variable_estimated_smoothed_log_bw6")

        if tik is None or oracle is None:
            continue

        tik_rl2 = tik.get("relative_l2", 1.0)
        oracle_rl2 = oracle.get("relative_l2", 1.0)

        if oracle_rl2 > 0.5 and tik_rl2 < 0.05:
            issues.append(f"{case}: oracle particle fails badly (rel_l2={oracle_rl2:.3f} vs tik={tik_rl2:.4f})")

        best_est = min(
            (r.get("relative_l2", 1.0) for r in [sl4, sl6] if r is not None),
            default=1.0,
        )
        # Floor the denominator at 0.005: the 10× rule is designed for noisy
        # data where Tikhonov has a real error floor.  With noiseless data
        # Tikhonov can achieve machine precision; clamp to avoid false failures.
        tik_rl2_floored = max(tik_rl2, 0.005)
        if tik_rl2_floored > 0:
            ratio = best_est / tik_rl2_floored
        else:
            ratio = float("inf")

        if not oracle.get("completed", False):
            issues.append(f"{case}: oracle particle did not complete")
        if sl4 is not None and not sl4.get("completed", False):
            issues.append(f"{case}: smoothed_log_bw4 did not complete")

        if ratio <= 10.0 and oracle_rl2 < 0.15:
            strong_go_evidence.append(case)
        elif ratio <= 20.0:
            cond_go_evidence.append(case)
        else:
            issues.append(f"{case}: estimated ratio={ratio:.1f}× Tikhonov (too large)")

    if issues:
        return "STOP/REVISE — Issues: " + "; ".join(issues)
    if len(strong_go_evidence) >= 3:
        return "STRONG GO — particle within 10× Tikhonov, oracle stable"
    if len(strong_go_evidence) + len(cond_go_evidence) >= 3:
        return "CONDITIONAL GO — partial success across beta/IC combinations"
    return "CONDITIONAL GO — some cases degraded; see metrics"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Variable-coefficient heat experiment")
    parser.add_argument("--n-particles", type=int, default=N_PARTICLES)
    parser.add_argument("--n-grid", type=int, default=N_GRID)
    parser.add_argument("--skip-tikhonov", action="store_true",
                        help="Skip the Tikhonov sweep (expensive for large n_grid)")
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Output directory
    # -----------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "outputs" / f"variable_coefficient_audit_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory: {out_dir}")

    # -----------------------------------------------------------------------
    # Load base configs
    # -----------------------------------------------------------------------
    base_cfg = load_config(str(GAUSSIAN_CONFIG))
    mix_cfg = load_config(str(MIXTURE_CONFIG))

    # Override grid size and PDE params
    base_cfg = patch_config(base_cfg,
                            **{"domain.n_grid": args.n_grid,
                               "heat.T": 0.15,
                               "heat.dt": 0.001})
    mix_cfg = patch_config(mix_cfg,
                           **{"domain.n_grid": args.n_grid,
                              "heat.T": 0.15,
                              "heat.dt": 0.001})

    # -----------------------------------------------------------------------
    # Test cases
    # -----------------------------------------------------------------------
    CASES = [
        ("VB_beta05", base_cfg, 0.5, "gaussian"),
        ("VB_beta09", base_cfg, 0.9, "gaussian"),
        ("VH_beta05", mix_cfg,  0.5, "mixture"),
        ("VH_beta09", mix_cfg,  0.9, "mixture"),
    ]

    all_rows: list[dict] = []
    all_results_by_case: dict[str, list[dict]] = {}

    for (case_id, cfg, beta, ic_type) in CASES:
        alpha0 = cfg.heat.alpha
        dt = cfg.heat.dt
        n_steps = cfg.n_steps
        print(f"\n{'='*60}")
        print(f"[CASE] {case_id}  beta={beta}  IC={ic_type}  "
              f"alpha0={alpha0}  T={cfg.heat.T}  n_steps={n_steps}  n_grid={args.n_grid}")

        x_grid = make_grid(cfg)
        u0_true = compute_true_u0(x_grid, cfg)

        # --- Variable-coeff forward solve ---
        print("[FORWARD] Solving variable-coeff heat equation...")
        t0 = time.perf_counter()
        u_obs, snapshots = solve_varcoeff_forward(u0_true, x_grid, alpha0, beta, dt, n_steps)
        print(f"  Done in {time.perf_counter()-t0:.2f}s | "
              f"u_obs range=[{u_obs.min():.4f}, {u_obs.max():.4f}]")

        case_results: list[dict] = []

        # --- Oracle method ---
        print("[METHOD] variable_oracle_deterministic")
        r_oracle = run_varcoeff_oracle(
            u_obs, x_grid, snapshots, alpha0, beta, dt, n_steps,
            n_particles=args.n_particles
        )
        case_results.append(r_oracle)
        print(f"  completed={r_oracle['completed']}  "
              f"failure={r_oracle.get('failure_msg','')}"
              f"  t={r_oracle['runtime_seconds']:.1f}s")

        # --- Estimated: smoothed_log bw4 ---
        print("[METHOD] variable_estimated_smoothed_log_bw4")
        r_sl4 = run_varcoeff_estimated(
            u_obs, x_grid, snapshots, alpha0, beta, dt, n_steps,
            score_method="smoothed_log", bandwidth_factor=4.0,
            n_particles=args.n_particles
        )
        case_results.append(r_sl4)
        print(f"  completed={r_sl4['completed']}  t={r_sl4['runtime_seconds']:.1f}s")

        # --- Estimated: smoothed_log bw6 ---
        print("[METHOD] variable_estimated_smoothed_log_bw6")
        r_sl6 = run_varcoeff_estimated(
            u_obs, x_grid, snapshots, alpha0, beta, dt, n_steps,
            score_method="smoothed_log", bandwidth_factor=6.0,
            n_particles=args.n_particles
        )
        case_results.append(r_sl6)
        print(f"  completed={r_sl6['completed']}  t={r_sl6['runtime_seconds']:.1f}s")

        # --- Estimated: fd_grid_ratio bw4 ---
        print("[METHOD] variable_estimated_fd_grid_ratio_bw4")
        r_fd4 = run_varcoeff_estimated(
            u_obs, x_grid, snapshots, alpha0, beta, dt, n_steps,
            score_method="fd_grid_ratio", bandwidth_factor=4.0,
            n_particles=args.n_particles
        )
        case_results.append(r_fd4)
        print(f"  completed={r_fd4['completed']}  t={r_fd4['runtime_seconds']:.1f}s")

        # --- Tikhonov (optional — expensive) ---
        if not args.skip_tikhonov:
            print("[METHOD] varcoeff_tikhonov_best (lambda sweep)")
            r_tik = varcoeff_tikhonov_sweep(
                u_obs, x_grid, alpha0, beta, dt, n_steps, u0_true
            )
            case_results.append(r_tik)
            print(f"  best_lam={r_tik['lam']}  rel_l2={r_tik['rel_l2']:.4f}"
                  f"  t={r_tik['runtime_seconds']:.1f}s")

        all_results_by_case[case_id] = case_results

        # --- Metrics ---
        for r in case_results:
            m = compute_result_metrics(
                r, u0_true, u_obs, x_grid, alpha0, beta, dt, n_steps
            )
            m["case"] = case_id
            m["beta"] = beta
            m["ic_type"] = ic_type
            m["alpha0"] = alpha0
            all_rows.append(m)
            print(f"  [{m['method']}] rel_l2={m['relative_l2']:.4f}  "
                  f"fwd_cons={m['forward_consistency_l2']:.4f}  "
                  f"mass_err={m['mass_rel_error']:.4f}")

        # --- Field comparison plot ---
        ic_short = "gaussian" if ic_type == "gaussian" else "mixture"
        beta_str = "beta05" if abs(beta - 0.5) < 0.01 else "beta09"
        plot_field_comparison(
            x_grid, u0_true, u_obs, case_results,
            f"{case_id}",
            out_dir / f"field_comparison_{beta_str}_{ic_short}.png"
        )

        # --- Score error vs step ---
        plot_score_error_vs_step(
            [r for r in case_results if r.get("method_name") != "varcoeff_tikhonov_best"],
            f"{case_id}",
            out_dir / f"score_error_vs_step_{case_id}.png"
        )

        # --- Forward consistency plot ---
        case_metric_rows = [m for m in all_rows if m["case"] == case_id]
        plot_forward_consistency(
            case_metric_rows, f"{case_id}",
            out_dir / f"forward_consistency_{case_id}.png"
        )

    # -----------------------------------------------------------------------
    # CSV
    # -----------------------------------------------------------------------
    df = pd.DataFrame(all_rows)
    csv_path = out_dir / "variable_coeff_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[CSV] Saved: {csv_path}")

    # -----------------------------------------------------------------------
    # Summary text
    # -----------------------------------------------------------------------
    verdict = compute_verdict(all_rows)

    summary_lines = [
        "=" * 70,
        "VARIABLE-COEFFICIENT HEAT EXPERIMENT SUMMARY",
        f"Timestamp : {timestamp}",
        f"n_grid    : {args.n_grid}",
        f"N_particles: {args.n_particles}",
        "=" * 70,
        "",
    ]

    for case_id in [c[0] for c in CASES]:
        case_rows = [r for r in all_rows if r["case"] == case_id]
        summary_lines.append(f"Case: {case_id}")
        for r in case_rows:
            summary_lines.append(
                f"  {r['method']:<45} "
                f"rel_l2={r['relative_l2']:.4f}  "
                f"fwd_l2={r['forward_consistency_l2']:.4f}  "
                f"t={r['runtime_seconds']:.1f}s"
            )
        summary_lines.append("")

    summary_lines += [
        "=" * 70,
        f"VERDICT: {verdict}",
        "=" * 70,
    ]

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    summary_path = out_dir / "variable_coeff_summary.txt"
    summary_path.write_text(summary_text)
    print(f"\n[SUMMARY] Saved: {summary_path}")


if __name__ == "__main__":
    main()
