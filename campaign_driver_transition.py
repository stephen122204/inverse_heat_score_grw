"""campaign_driver_transition.py — the C1 estimator transition table.

This is the only campaign driver allowed to touch legacy code paths
(protocol Section 9), so it lives apart from campaign_drivers and shares
only the campaign_results output format with the canonical runners.

The three preregistered eras, in historical sequence:

1. endpoint_free_space_legacy — the pre-migration pipeline: endpoint grid
   x = linspace(0, 1, M) with dx = L/(M-1), trapezoidal quantile
   initialization, free-space Gaussian KDE reconstruction on the grid
   coordinates, Gaussian-smoothed log-density score (a second smoothing),
   mirror-reflecting walls. Vendored below from src/invheat_grw/methods.py
   and scores.py at commit e80c61f, the last commit before the
   cell-centered migration (0b936d7).
2. cell_centered_free_space_legacy — the retained legacy branch of the
   current package: cell-centered grid with the same free-space KDE and
   smoothed-log score (recon_method="kde", score_method="smoothed_log").
3. cell_centered_neumann_canonical — the production bounded-domain runner
   with one Neumann heat-kernel smoothing at the C1 anchor bandwidth.

Both legacy eras use the archived frozen configuration (N = 10000,
n_grid = 400, bandwidth = 4 dx, smooth_sigma_factor = 1, absolute
epsilon = 1e-8); their effective smoothing is bandwidth * sqrt(2) because
the KDE kernel and the log-smoothing filter compose. The canonical era
uses the campaign configuration, where nominal and effective smoothing
coincide. This is a bundled historical transition, not a one-factor
ablation, and no primary claim may rely on the legacy rows.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import campaign_results as results
import campaign_schema as schema

LEGACY_N_PARTICLES = 10000
LEGACY_N_GRID = 400
LEGACY_BANDWIDTH_FACTOR = 4.0
LEGACY_SMOOTH_SIGMA_FACTOR = 1.0
LEGACY_EPSILON = 1e-8
LEGACY_SOURCE_COMMIT = "e80c61f"

X_MIN, X_MAX = 0.0, 1.0
LENGTH = X_MAX - X_MIN

TRANSITION_PAYLOAD_KEYS = (
    "E2", "h_nominal", "h_effective", "smoothing_operations", "n_particles",
    "n_grid", "epsilon_style", "epsilon_value", "grid_convention",
    "score_estimator", "mass_initial",
)


def c1_fields(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from invheat_grw.campaign_data import cosine_field
    spec = schema.CASES[schema.TRANSITION_CASE]
    u0 = cosine_field(x, spec["background"], spec["modes"], length=LENGTH)
    g = cosine_field(x, spec["background"], spec["modes"], length=LENGTH,
                     alpha=spec["alpha"], t=spec["T"])
    return u0, g


def _rel_l2(candidate: np.ndarray, truth: np.ndarray, dx: float) -> float:
    return float(np.sqrt(dx * np.sum((candidate - truth) ** 2))
                 / np.sqrt(dx * np.sum(truth ** 2)))


# ---------------------------------------------------------------------------
# Era 1: vendored pre-migration endpoint/free-space pipeline (e80c61f)
# ---------------------------------------------------------------------------

def _legacy_quantile_init(u_obs: np.ndarray, x_grid: np.ndarray,
                          n_particles: int) -> tuple[np.ndarray, float]:
    """Pre-migration initializer: trapezoidal CDF on the endpoint nodes."""
    u_pos = np.maximum(u_obs, 0.0)
    dx = x_grid[1] - x_grid[0]
    segments = 0.5 * (u_pos[:-1] + u_pos[1:]) * dx   # trapezoidal masses
    total_mass = float(np.sum(segments))
    if total_mass <= 0.0:
        return np.linspace(x_grid[0], x_grid[-1], n_particles), 0.0
    cdf = np.concatenate([[0.0], np.cumsum(segments)]) / total_mass
    quantiles = (np.arange(n_particles) + 0.5) / n_particles
    return np.interp(quantiles, cdf, x_grid), total_mass


def _legacy_free_space_kde(positions: np.ndarray, total_mass: float,
                           x_grid: np.ndarray, bandwidth: float
                           ) -> np.ndarray:
    diff = x_grid[:, None] - positions[None, :]
    kernels = (np.exp(-0.5 * (diff / bandwidth) ** 2)
               / (bandwidth * math.sqrt(2.0 * math.pi)))
    return np.sum(kernels, axis=1) * (total_mass / len(positions))


def _legacy_smoothed_log_score(u_grid: np.ndarray, x_grid: np.ndarray,
                               smooth_sigma: float, epsilon: float
                               ) -> np.ndarray:
    from scipy.ndimage import gaussian_filter1d
    dx = x_grid[1] - x_grid[0]
    u_smooth = gaussian_filter1d(u_grid, sigma=smooth_sigma / dx,
                                 mode="nearest")
    u_smooth = np.maximum(u_smooth, 0.0)
    log_u = np.log(np.maximum(u_smooth + epsilon, 1e-300))
    return np.gradient(log_u, dx)


def _legacy_reflect(positions: np.ndarray) -> np.ndarray:
    pos = positions.copy()
    below = pos < X_MIN
    pos[below] = 2.0 * X_MIN - pos[below]
    above = pos > X_MAX
    pos[above] = 2.0 * X_MAX - pos[above]
    return np.clip(pos, X_MIN, X_MAX)


def run_endpoint_free_space_legacy() -> dict:
    """The pre-migration smoothed-log run on the endpoint grid."""
    spec = schema.CASES[schema.TRANSITION_CASE]
    alpha, big_t, dt = spec["alpha"], spec["T"], schema.DEFAULT_DT
    x = np.linspace(X_MIN, X_MAX, LEGACY_N_GRID)
    dx = x[1] - x[0]
    bandwidth = LEGACY_BANDWIDTH_FACTOR * dx
    smooth_sigma = LEGACY_SMOOTH_SIGMA_FACTOR * bandwidth
    u0, g = c1_fields(x)
    n_steps = round(big_t / dt)
    positions, total_mass = _legacy_quantile_init(g, x, LEGACY_N_PARTICLES)
    for _ in range(n_steps):
        u_recon = _legacy_free_space_kde(positions, total_mass, x, bandwidth)
        s_grid = _legacy_smoothed_log_score(u_recon, x, smooth_sigma,
                                            LEGACY_EPSILON)
        scores = np.interp(positions, x, s_grid)
        if not np.all(np.isfinite(scores)):
            return {"completed": False,
                    "failure_message": "nonfinite legacy score"}
        positions = _legacy_reflect(positions + alpha * scores * dt)
    candidate = _legacy_free_space_kde(positions, total_mass, x, bandwidth)
    return {"completed": True, "failure_message": "",
            "E2": _rel_l2(candidate, u0, dx), "mass_initial": total_mass,
            "h_nominal": bandwidth,
            "h_effective": bandwidth * math.sqrt(2.0)}


# ---------------------------------------------------------------------------
# Eras 2 and 3: current-package paths
# ---------------------------------------------------------------------------

def run_cell_centered_free_space_legacy() -> dict:
    """The retained legacy branch of the migrated package."""
    import copy

    from invheat_grw.cell_grid import cell_centers, cell_spacing
    from invheat_grw.config import load_config
    from invheat_grw.methods import (
        run_density_particle_estimated_score_deterministic,
    )
    spec = schema.CASES[schema.TRANSITION_CASE]
    cfg = copy.deepcopy(load_config(str(REPO / "configs"
                                        / "gaussian_base.yaml")))
    cfg.heat.alpha = spec["alpha"]
    cfg.heat.T = spec["T"]
    cfg.heat.dt = schema.DEFAULT_DT
    cfg.domain.x_min, cfg.domain.x_max = X_MIN, X_MAX
    cfg.domain.n_grid = LEGACY_N_GRID
    x = cell_centers(X_MIN, X_MAX, LEGACY_N_GRID)
    dx = cell_spacing(X_MIN, X_MAX, LEGACY_N_GRID)
    bandwidth = LEGACY_BANDWIDTH_FACTOR * dx
    u0, g = c1_fields(x)
    result = run_density_particle_estimated_score_deterministic(
        g, x, cfg, LEGACY_N_PARTICLES, recon_method="kde",
        bandwidth_factor=LEGACY_BANDWIDTH_FACTOR, epsilon=LEGACY_EPSILON,
        scale_epsilon_by_peak=False, score_clipping=None,
        save_snapshots=False, score_method="smoothed_log",
        smooth_sigma_factor=LEGACY_SMOOTH_SIGMA_FACTOR)
    return {"completed": result.completed,
            "failure_message": result.failure_msg,
            "E2": _rel_l2(result.candidate, u0, dx),
            "mass_initial": result.mass_initial,
            "h_nominal": bandwidth,
            "h_effective": bandwidth * math.sqrt(2.0)}


def run_cell_centered_neumann_canonical() -> dict:
    """The production bounded-domain runner at the campaign settings."""
    from invheat_grw.campaign_particle import run_campaign_density
    from invheat_grw.cell_grid import cell_centers
    spec = schema.CASES[schema.TRANSITION_CASE]
    x = cell_centers(X_MIN, X_MAX, schema.DEFAULT_M)
    u0, g = c1_fields(x)
    h = schema.anchor_h(schema.TRANSITION_CASE)
    result = run_campaign_density(
        g, x_min=X_MIN, x_max=X_MAX, T=spec["T"], dt=schema.DEFAULT_DT,
        n_particles=schema.DEFAULT_N, bandwidth=h,
        eps_rel=schema.HEADLINE_EPS_REL, alpha=spec["alpha"],
        u0_reference=u0)
    return {"completed": result.status == "completed",
            "failure_message": result.failure_message,
            "E2": result.metrics.get("E2", float("nan")),
            "mass_initial": result.metrics.get("mass_reconstruction",
                                               float("nan")),
            "h_nominal": h, "h_effective": h}


ERA_RUNNERS = {
    "endpoint_free_space_legacy": run_endpoint_free_space_legacy,
    "cell_centered_free_space_legacy": run_cell_centered_free_space_legacy,
    "cell_centered_neumann_canonical": run_cell_centered_neumann_canonical,
}

ERA_METADATA = {
    "endpoint_free_space_legacy": {
        "smoothing_operations": 2, "n_particles": LEGACY_N_PARTICLES,
        "n_grid": LEGACY_N_GRID, "epsilon_style": "absolute",
        "epsilon_value": LEGACY_EPSILON, "grid_convention": "endpoint",
        "score_estimator": "free_space_kde_smoothed_log",
    },
    "cell_centered_free_space_legacy": {
        "smoothing_operations": 2, "n_particles": LEGACY_N_PARTICLES,
        "n_grid": LEGACY_N_GRID, "epsilon_style": "absolute",
        "epsilon_value": LEGACY_EPSILON, "grid_convention": "cell_centered",
        "score_estimator": "free_space_kde_smoothed_log",
    },
    "cell_centered_neumann_canonical": {
        "smoothing_operations": 1, "n_particles": schema.DEFAULT_N,
        "n_grid": schema.DEFAULT_M, "epsilon_style": "relative",
        "epsilon_value": schema.HEADLINE_EPS_REL,
        "grid_convention": "cell_centered",
        "score_estimator": "neumann_kde",
    },
}


def drive_transition_table(out_dir: Path,
                           rows: Sequence[dict] | None = None
                           ) -> results.Accounting:
    writer = results.StudyWriter("transition_table", Path(out_dir))
    for row in rows or results.enumerate_rows("transition_table"):
        if results.study_row_key(writer.study, row) in writer.done_keys:
            continue
        outcome = ERA_RUNNERS[row["era"]]()
        payload = dict(ERA_METADATA[row["era"]])
        payload["E2"] = outcome.get("E2", float("nan"))
        payload["h_nominal"] = outcome.get("h_nominal", float("nan"))
        payload["h_effective"] = outcome.get("h_effective", float("nan"))
        payload["mass_initial"] = outcome.get("mass_initial", float("nan"))
        assert tuple(sorted(payload)) == tuple(sorted(TRANSITION_PAYLOAD_KEYS))
        writer.append(row,
                      results.STATUS_COMPLETED if outcome["completed"]
                      else results.STATUS_FAILED,
                      payload, failure_message=outcome["failure_message"])
    accounting = results.reconcile("transition_table", writer.csv_path)
    results.write_summary(
        Path(out_dir), accounting,
        {"accounting_consistent": accounting.consistent},
        extra={"legacy_source_commit": LEGACY_SOURCE_COMMIT})
    return accounting


DRIVERS = {"transition_table": drive_transition_table}
