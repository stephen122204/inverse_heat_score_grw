"""
run_validation_stage.py — Validation stage for density-particle estimated-score method.

Manuscript labels (draft): `fig:particle_count` (`sec:particle_count`).

Scientific questions:
  TASK 1 — N-convergence on Test B (Gaussian T=0.15)
      Does rel_L2 improve as N increases from 5000 → 10000 → 20000?
      If it plateaus, bandwidth bias dominates over particle variance.

  TASK 2 — Noise robustness on Test B
      u_obs_noisy = u_obs + eta * max(u_obs) * Normal(0,1)
      eta in {0.001, 0.005, 0.01}, seeds {0, 1, 2}
      Does bandwidth regularization stabilize noisy observations?
      Does fixed-bandwidth particle score compare reasonably to fixed-lam Tikhonov?

Methods:
  - density_particle_oracle_score_deterministic (oracle ceiling)
  - density_particle_estimated_score_deterministic, smoothed_log, bw=4
  - density_particle_estimated_score_deterministic, smoothed_log, bw=6
  - density_particle_estimated_score_deterministic, fd_grid_ratio, bw=4
  - tikhonov_best (tuned per noise level)
  - tikhonov_fixed_lam (fixed at clean-data best lambda)
  - spectral_cutoff_best (tuned per noise level)

Decision criteria:
  STRONG GO:    N=10000 drives Test B rel_L2 < 0.01 or close to oracle
                Noisy eta=0.001 and 0.005 remain stable (rel_L2 < 0.05)
  CONDITIONAL GO: N=10000 gives little improvement, rel_L2 ~ 0.01-0.02
                  eta=0.001 stable, eta=0.005/0.01 degrade gracefully
  STOP/REVISE:  N=10000 worsens or is unstable
                eta=0.001 causes failure

Outputs: outputs/validation_stage_TIMESTAMP/
  validation_metrics.csv
  validation_summary.txt
  N_convergence_TestB.png
  noisy_rel_L2_vs_eta_TestB.png
  field_comparison_N10000_TestB.png
  field_comparison_noisy_TestB_eta001.png
  field_comparison_noisy_TestB_eta005.png
  field_comparison_noisy_TestB_eta01.png

Usage:
  cd inverse_heat_score_grw
  PYTHONPATH=src python scripts/run_validation_stage.py \\
      --base-config configs/gaussian_base.yaml \\
      --mixture-config configs/gaussian_mixture.yaml

  # Fast smoke test:
  PYTHONPATH=src python scripts/run_validation_stage.py \\
      --base-config configs/gaussian_base.yaml \\
      --mixture-config configs/gaussian_mixture.yaml \\
      --n-particles-convergence 1000 2000 \\
      --skip-n20000 --n-grid 200

  # Skip noise task:
  PYTHONPATH=src python scripts/run_validation_stage.py \\
      --base-config configs/gaussian_base.yaml \\
      --mixture-config configs/gaussian_mixture.yaml \\
      --skip-noise
"""

from __future__ import annotations

import sys
import argparse
import copy
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from invheat_grw.config import load_config, Config
from invheat_grw.fields import (
    make_grid,
    true_u0 as compute_true_u0,
    observed_final as compute_observed_final,
)
from invheat_grw.methods import (
    run_density_particle_oracle_score_deterministic,
    run_density_particle_estimated_score_deterministic,
)
from invheat_grw.metrics import (
    compute_metrics,
    compute_wasserstein,
)
from invheat_grw.baselines import tikhonov_inverse, spectral_cutoff_inverse

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RECON_METHOD = "kde"
TIKHONOV_LAMBDAS = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
SPECTRAL_NOISE_DELTAS = [1e-8, 1e-6, 1e-4, 1e-3, 1e-2]
NOISE_SEEDS = [0, 1, 2]

# Best known clean-data Tikhonov lambda for Test B at n_grid=400
# (will be re-discovered at runtime if None)
TIKHONOV_CLEAN_BEST_LAM: Optional[float] = None

COLORS = {
    "oracle":       "#1f77b4",
    "smoothed_log_bw4": "#9467bd",
    "smoothed_log_bw6": "#e377c2",
    "fd_ratio_bw4": "#ff7f0e",
    "tikhonov_best":  "#8c564b",
    "tikhonov_fixed": "#bcbd22",
    "spectral_best":  "#17becf",
    "grad_glob":    "#aaaaaa",
}

# Grad-glob oracle plateau (from previous study)
GRAD_GLOB_PLATEAU_B = 0.175


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(v) -> float:
    try:
        f = float(v)
        return f if np.isfinite(f) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _listval(lst: list, idx: int) -> float:
    if not lst:
        return float("nan")
    if idx == -1:
        idx = len(lst) - 1
    if 0 <= idx < len(lst):
        return _safe(lst[idx])
    return float("nan")


def patch_config(cfg: Config, **overrides) -> Config:
    new_cfg = copy.deepcopy(cfg)
    for key, val in overrides.items():
        parts = key.split(".")
        obj = new_cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], val)
    return new_cfg


def build_test_b_config(base_cfg: Config, n_grid: int) -> Config:
    """Test B: Gaussian, sigma0=0.08, T=0.15, n_steps=150."""
    return patch_config(base_cfg, **{
        "heat.T": 0.15,
        "initial_condition.sigma0": 0.08,
        "domain.n_grid": n_grid,
    })


def make_noisy_obs(u_obs: np.ndarray, eta: float, seed: int) -> np.ndarray:
    """u_obs_noisy = u_obs + eta * max(u_obs) * N(0,1)."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(u_obs.shape)
    return u_obs + eta * float(np.max(u_obs)) * noise


# ---------------------------------------------------------------------------
# Metric aggregation helpers
# ---------------------------------------------------------------------------

def _fill_row_from_result(row: dict, result, true_u, u_obs, x_grid, cfg) -> dict:
    """Compute metrics and fill the row dict in-place. Returns row."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = compute_metrics(result, true_u, u_obs, x_grid, cfg)
        wass = compute_wasserstein(result.candidate, true_u, x_grid)

    score_l2_errs = getattr(result, "score_L2_error_vs_oracle", [])
    score_core_l2_errs = getattr(result, "score_core_L2_error_vs_oracle", [])

    finite_l2 = [v for v in score_l2_errs if np.isfinite(v)]
    finite_core = [v for v in score_core_l2_errs if np.isfinite(v)]

    row.update({
        "completed": result.completed,
        "failure_step": str(getattr(result, "failure_step", "") or ""),
        "failure_msg": str(getattr(result, "failure_msg", "") or ""),
        "rel_L2": _safe(m.relative_l2),
        "l2_error": _safe(m.l2_error),
        "forward_consistency_l2": _safe(m.forward_consistency_l2),
        "mass_rel_error": _safe(m.mass_rel_error),
        "wasserstein": _safe(wass),
        "peak_value": _safe(m.peak_value),
        "peak_ratio": _safe(m.peak_ratio),
        "runtime_seconds": _safe(getattr(result, "runtime_seconds", float("nan"))),
        "max_abs_score_final": _listval(getattr(result, "score_max_abs", []), -1),
        "max_score_L2_error_vs_oracle": float(max(finite_l2)) if finite_l2 else float("nan"),
        "mean_score_L2_error_vs_oracle": float(np.mean(finite_l2)) if finite_l2 else float("nan"),
        "max_score_core_L2_error_vs_oracle": float(max(finite_core)) if finite_core else float("nan"),
        "step_zero_recon_error": _safe(getattr(result, "step_zero_recon_error", float("nan"))),
    })
    return row


def _base_row(task: str, n_particles: int, n_grid: int, method_label: str,
              score_method: str, bw_factor: float, eta: float,
              seed: int, epsilon: float) -> dict:
    return {
        "task": task,
        "n_particles": n_particles,
        "n_grid": n_grid,
        "method_label": method_label,
        "score_method": score_method,
        "bandwidth_factor": bw_factor,
        "eta": eta,
        "seed": seed,
        "epsilon": epsilon,
        # to be filled by _fill_row_from_result
        "completed": False,
        "failure_step": "",
        "failure_msg": "",
        "rel_L2": float("nan"),
        "l2_error": float("nan"),
        "forward_consistency_l2": float("nan"),
        "mass_rel_error": float("nan"),
        "wasserstein": float("nan"),
        "peak_value": float("nan"),
        "peak_ratio": float("nan"),
        "runtime_seconds": float("nan"),
        "max_abs_score_final": float("nan"),
        "max_score_L2_error_vs_oracle": float("nan"),
        "mean_score_L2_error_vs_oracle": float("nan"),
        "max_score_core_L2_error_vs_oracle": float("nan"),
        "step_zero_recon_error": float("nan"),
    }


# ---------------------------------------------------------------------------
# Tikhonov helpers
# ---------------------------------------------------------------------------

def _find_tikhonov_best(u_obs: np.ndarray, x_grid: np.ndarray,
                        true_u: np.ndarray, alpha: float, T: float,
                        cfg: Config, lambdas=None) -> tuple[float, np.ndarray | None, float]:
    """Grid-search Tikhonov lambdas. Returns (best_lam, best_cand, best_rel_l2)."""
    if lambdas is None:
        lambdas = TIKHONOV_LAMBDAS
    best_lam, best_cand, best_rl2 = float("nan"), None, float("inf")
    for lam in lambdas:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            tr = tikhonov_inverse(u_obs, x_grid, alpha, T, lam,
                                  length=float(cfg.domain.x_max - cfg.domain.x_min))
        if not np.all(np.isfinite(tr.candidate)):
            continue
        dx = x_grid[1] - x_grid[0]
        diff = tr.candidate - true_u
        rl2 = float(np.sqrt(dx * np.sum(diff ** 2))) / float(np.sqrt(dx * np.sum(true_u ** 2)))
        if rl2 < best_rl2:
            best_rl2 = rl2
            best_cand = tr.candidate.copy()
            best_lam = lam
    return best_lam, best_cand, best_rl2


def _find_spectral_best(u_obs: np.ndarray, x_grid: np.ndarray,
                        true_u: np.ndarray, alpha: float, T: float,
                        cfg: Config) -> tuple[float, np.ndarray | None, float]:
    """Grid-search spectral cutoff noise_deltas. Returns (best_nd, best_cand, best_rel_l2)."""
    best_nd, best_cand, best_rl2 = float("nan"), None, float("inf")
    for nd in SPECTRAL_NOISE_DELTAS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sr = spectral_cutoff_inverse(u_obs, x_grid, alpha, T, noise_delta=nd,
                                         length=float(cfg.domain.x_max - cfg.domain.x_min))
        if not np.all(np.isfinite(sr.candidate)):
            continue
        dx = x_grid[1] - x_grid[0]
        diff = sr.candidate - true_u
        rl2 = float(np.sqrt(dx * np.sum(diff ** 2))) / float(np.sqrt(dx * np.sum(true_u ** 2)))
        if rl2 < best_rl2:
            best_rl2 = rl2
            best_cand = sr.candidate.copy()
            best_nd = nd
    return best_nd, best_cand, best_rl2


class _SimpleResult:
    """Minimal result wrapper for Tikhonov/spectral candidates."""
    def __init__(self, cand: np.ndarray, method_name: str):
        self.candidate = cand
        self.method_name = method_name
        self.completed = True
        self.failure_step = None
        self.failure_msg = ""
        self.runtime_seconds = float("nan")
        self.step_zero_recon_error = float("nan")
        self.score_L2_error_vs_oracle = []
        self.score_core_L2_error_vs_oracle = []
        self.score_max_abs = []
        self.mean_abs_score = []
        self.score_std = []
        self.n_denominator_below_epsilon = []
        self.n_clipped_scores = []
        self.mass_per_step = []
        self.epsilon_actual_per_step = []


# ---------------------------------------------------------------------------
# TASK 1: N-convergence on Test B (clean observations)
# ---------------------------------------------------------------------------

def run_task1_convergence(
    base_cfg: Config,
    n_particles_list: list[int],
    n_grid: int,
    epsilon: float,
    run_n20000: bool,
) -> tuple[list[dict], float]:
    """
    Task 1: run oracle + smoothed_log bw=4 + fd_ratio bw=4 for each N.
    Returns list of row dicts.
    """
    rows = []
    cfg = build_test_b_config(base_cfg, n_grid)
    x_grid = make_grid(cfg)
    true_u = compute_true_u0(x_grid, cfg)
    u_obs = compute_observed_final(x_grid, cfg)

    # Find Tikhonov best on clean data (once; independent of N)
    print("  [Task 1] Finding Tikhonov best on clean data ...", flush=True)
    tik_lam, tik_cand, tik_rl2 = _find_tikhonov_best(
        u_obs, x_grid, true_u, cfg.heat.alpha, cfg.heat.T, cfg)
    print(f"    Tikhonov best: lam={tik_lam:.0e}  rel_L2={tik_rl2:.5f}", flush=True)

    # Add Tikhonov row (no N dependence)
    if tik_cand is not None:
        tr_result = _SimpleResult(tik_cand, f"tikhonov_lam={tik_lam:.0e}")
        row = _base_row("convergence", 0, n_grid, "tikhonov_best", "tikhonov",
                        float("nan"), 0.0, -1, tik_lam)
        _fill_row_from_result(row, tr_result, true_u, u_obs, x_grid, cfg)
        rows.append(row)

    # Spectral best on clean data
    sp_nd, sp_cand, sp_rl2 = _find_spectral_best(
        u_obs, x_grid, true_u, cfg.heat.alpha, cfg.heat.T, cfg)
    print(f"    Spectral best: nd={sp_nd:.0e}  rel_L2={sp_rl2:.5f}", flush=True)
    if sp_cand is not None:
        sp_result = _SimpleResult(sp_cand, f"spectral_nd={sp_nd:.0e}")
        row = _base_row("convergence", 0, n_grid, "spectral_best", "spectral",
                        float("nan"), 0.0, -1, sp_nd)
        _fill_row_from_result(row, sp_result, true_u, u_obs, x_grid, cfg)
        rows.append(row)

    particle_ns = [n for n in n_particles_list if n <= 10000 or run_n20000]

    for n_p in particle_ns:
        print(f"  [Task 1] N={n_p} ...", flush=True)

        # Oracle
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            r_oracle = run_density_particle_oracle_score_deterministic(
                u_obs, x_grid, cfg, n_p,
                recon_method=RECON_METHOD, bandwidth_factor=4.0)
        print(f"    oracle: rel_L2={_safe(compute_metrics(r_oracle, true_u, u_obs, x_grid, cfg).relative_l2):.5f}  "
              f"t={time.perf_counter()-t0:.1f}s", flush=True)
        row = _base_row("convergence", n_p, n_grid, "oracle", "oracle", 4.0, 0.0, -1, 0.0)
        _fill_row_from_result(row, r_oracle, true_u, u_obs, x_grid, cfg)
        rows.append(row)

        # smoothed_log bw=4
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            r_sl4 = run_density_particle_estimated_score_deterministic(
                u_obs, x_grid, cfg, n_p,
                recon_method=RECON_METHOD, bandwidth_factor=4.0,
                epsilon=epsilon, scale_epsilon_by_peak=False,
                score_clipping=None, save_snapshots=False,
                score_method="smoothed_log", smooth_sigma_factor=1.0)
        rl2_sl4 = _safe(compute_metrics(r_sl4, true_u, u_obs, x_grid, cfg).relative_l2)
        print(f"    smoothed_log bw=4: rel_L2={rl2_sl4:.5f}  t={time.perf_counter()-t0:.1f}s", flush=True)
        row = _base_row("convergence", n_p, n_grid, "smoothed_log_bw4", "smoothed_log", 4.0, 0.0, -1, epsilon)
        _fill_row_from_result(row, r_sl4, true_u, u_obs, x_grid, cfg)
        rows.append(row)

        # smoothed_log bw=6
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            r_sl6 = run_density_particle_estimated_score_deterministic(
                u_obs, x_grid, cfg, n_p,
                recon_method=RECON_METHOD, bandwidth_factor=6.0,
                epsilon=epsilon, scale_epsilon_by_peak=False,
                score_clipping=None, save_snapshots=False,
                score_method="smoothed_log", smooth_sigma_factor=1.0)
        rl2_sl6 = _safe(compute_metrics(r_sl6, true_u, u_obs, x_grid, cfg).relative_l2)
        print(f"    smoothed_log bw=6: rel_L2={rl2_sl6:.5f}  t={time.perf_counter()-t0:.1f}s", flush=True)
        row = _base_row("convergence", n_p, n_grid, "smoothed_log_bw6", "smoothed_log", 6.0, 0.0, -1, epsilon)
        _fill_row_from_result(row, r_sl6, true_u, u_obs, x_grid, cfg)
        rows.append(row)

        # fd_grid_ratio bw=4
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            r_fd4 = run_density_particle_estimated_score_deterministic(
                u_obs, x_grid, cfg, n_p,
                recon_method=RECON_METHOD, bandwidth_factor=4.0,
                epsilon=epsilon, scale_epsilon_by_peak=False,
                score_clipping=None, save_snapshots=False,
                score_method="fd_grid_ratio", smooth_sigma_factor=1.0)
        rl2_fd4 = _safe(compute_metrics(r_fd4, true_u, u_obs, x_grid, cfg).relative_l2)
        print(f"    fd_grid_ratio bw=4: rel_L2={rl2_fd4:.5f}  t={time.perf_counter()-t0:.1f}s", flush=True)
        row = _base_row("convergence", n_p, n_grid, "fd_ratio_bw4", "fd_grid_ratio", 4.0, 0.0, -1, epsilon)
        _fill_row_from_result(row, r_fd4, true_u, u_obs, x_grid, cfg)
        rows.append(row)

    return rows, tik_lam


# ---------------------------------------------------------------------------
# TASK 2: Noise robustness on Test B
# ---------------------------------------------------------------------------

def run_task2_noise(
    base_cfg: Config,
    n_particles: int,
    n_grid: int,
    epsilon: float,
    eta_list: list[float],
    seeds: list[int],
    tikhonov_clean_lam: float,
) -> list[dict]:
    """
    Task 2: noise robustness. Returns list of row dicts.
    """
    rows = []
    cfg = build_test_b_config(base_cfg, n_grid)
    x_grid = make_grid(cfg)
    true_u = compute_true_u0(x_grid, cfg)
    u_obs_clean = compute_observed_final(x_grid, cfg)

    # Also run eta=0 (clean) as baseline in noise table
    all_etas = [0.0] + list(eta_list)

    for eta in all_etas:
        eta_label = f"{eta:.3f}"
        for seed in (seeds if eta > 0 else [0]):
            if eta == 0.0:
                u_obs = u_obs_clean.copy()
                print(f"  [Task 2] eta=0.000 (clean) ...", flush=True)
            else:
                u_obs = make_noisy_obs(u_obs_clean, eta, seed)
                print(f"  [Task 2] eta={eta_label} seed={seed} ...", flush=True)

            # ---- Particle methods ----
            configs_est = [
                ("smoothed_log_bw4", "smoothed_log", 4.0),
                ("smoothed_log_bw6", "smoothed_log", 6.0),
                ("fd_ratio_bw4",     "fd_grid_ratio", 4.0),
            ]
            for label, score_method, bw in configs_est:
                t0 = time.perf_counter()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    r = run_density_particle_estimated_score_deterministic(
                        u_obs, x_grid, cfg, n_particles,
                        recon_method=RECON_METHOD, bandwidth_factor=bw,
                        epsilon=epsilon, scale_epsilon_by_peak=False,
                        score_clipping=None, save_snapshots=False,
                        score_method=score_method, smooth_sigma_factor=1.0)
                rl2_m = compute_metrics(r, true_u, u_obs_clean, x_grid, cfg)
                print(f"    {label}: rel_L2={_safe(rl2_m.relative_l2):.5f}  "
                      f"completed={r.completed}  t={time.perf_counter()-t0:.1f}s", flush=True)
                row = _base_row("noise", n_particles, n_grid, label, score_method,
                                bw, eta, seed, epsilon)
                _fill_row_from_result(row, r, true_u, u_obs_clean, x_grid, cfg)
                rows.append(row)

            # ---- Tikhonov tuned per noise level ----
            tik_lam, tik_cand, tik_rl2 = _find_tikhonov_best(
                u_obs, x_grid, true_u, cfg.heat.alpha, cfg.heat.T, cfg)
            print(f"    tikhonov_best: lam={tik_lam:.0e}  rel_L2={tik_rl2:.5f}", flush=True)
            if tik_cand is not None:
                tr = _SimpleResult(tik_cand, f"tikhonov_lam={tik_lam:.0e}")
                row = _base_row("noise", 0, n_grid, "tikhonov_best", "tikhonov",
                                float("nan"), eta, seed, tik_lam)
                _fill_row_from_result(row, tr, true_u, u_obs_clean, x_grid, cfg)
                rows.append(row)

            # ---- Tikhonov fixed at clean-data best lambda ----
            if np.isfinite(tikhonov_clean_lam):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    tr_fixed = tikhonov_inverse(u_obs, x_grid, cfg.heat.alpha,
                                               cfg.heat.T, tikhonov_clean_lam,
                                               length=float(cfg.domain.x_max - cfg.domain.x_min))
                fr = _SimpleResult(tr_fixed.candidate, f"tikhonov_fixed_lam={tikhonov_clean_lam:.0e}")
                dx = x_grid[1] - x_grid[0]
                diff = tr_fixed.candidate - true_u
                fixed_rl2 = float(np.sqrt(dx * np.sum(diff**2))) / float(np.sqrt(dx * np.sum(true_u**2)))
                print(f"    tikhonov_fixed: rel_L2={fixed_rl2:.5f}", flush=True)
                row = _base_row("noise", 0, n_grid, "tikhonov_fixed", "tikhonov",
                                float("nan"), eta, seed, tikhonov_clean_lam)
                _fill_row_from_result(row, fr, true_u, u_obs_clean, x_grid, cfg)
                rows.append(row)

            # ---- Spectral cutoff tuned per noise level ----
            sp_nd, sp_cand, sp_rl2 = _find_spectral_best(
                u_obs, x_grid, true_u, cfg.heat.alpha, cfg.heat.T, cfg)
            print(f"    spectral_best: nd={sp_nd:.0e}  rel_L2={sp_rl2:.5f}", flush=True)
            if sp_cand is not None:
                sr = _SimpleResult(sp_cand, f"spectral_nd={sp_nd:.0e}")
                row = _base_row("noise", 0, n_grid, "spectral_best", "spectral",
                                float("nan"), eta, seed, sp_nd)
                _fill_row_from_result(row, sr, true_u, u_obs_clean, x_grid, cfg)
                rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_n_convergence(df: pd.DataFrame, out_dir: Path, n_grid: int) -> None:
    """N_convergence_TestB.png: rel_L2 vs N for each method."""
    conv = df[(df["task"] == "convergence") & (df["n_particles"] > 0)].copy()
    if conv.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))

    method_style = {
        "oracle":           (COLORS["oracle"],            "-",  "o", "Oracle (density-particle)"),
        "smoothed_log_bw4": (COLORS["smoothed_log_bw4"], "-",  "s", "smoothed_log bw=4"),
        "smoothed_log_bw6": (COLORS["smoothed_log_bw6"], "--", "^", "smoothed_log bw=6"),
        "fd_ratio_bw4":     (COLORS["fd_ratio_bw4"],     "--", "D", "fd_grid_ratio bw=4"),
    }

    ns_sorted = sorted(conv["n_particles"].unique())
    for mlabel, (col, ls, mk, name) in method_style.items():
        sub = conv[conv["method_label"] == mlabel].sort_values("n_particles")
        if sub.empty:
            continue
        ax.plot(sub["n_particles"], sub["rel_L2"],
                color=col, linestyle=ls, marker=mk, label=name, linewidth=2, markersize=7)

    # Tikhonov and spectral reference lines
    for blabel, col, name in [
        ("tikhonov_best", COLORS["tikhonov_best"], "Tikhonov (tuned)"),
        ("spectral_best", COLORS["spectral_best"], "Spectral cutoff (tuned)"),
    ]:
        sub = df[(df["task"] == "convergence") & (df["method_label"] == blabel)]
        if not sub.empty:
            rl2 = float(sub["rel_L2"].iloc[0])
            ax.axhline(rl2, color=col, linestyle=":", linewidth=1.5, label=name)

    ax.axhline(GRAD_GLOB_PLATEAU_B, color=COLORS["grad_glob"], linestyle=":",
               linewidth=1, alpha=0.7, label="Grad-glob plateau (0.175)")

    ax.set_xlabel("N particles", fontsize=12)
    ax.set_ylabel("rel_L2", fontsize=12)
    ax.set_title("Test B — N-convergence (Gaussian, T=0.15)", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "N_convergence_TestB.png", dpi=150)
    plt.close(fig)


def plot_noisy_rel_l2(df: pd.DataFrame, out_dir: Path) -> None:
    """noisy_rel_L2_vs_eta_TestB.png: rel_L2 vs eta (mean over seeds) for each method."""
    noise_df = df[df["task"] == "noise"].copy()
    if noise_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))

    method_style = {
        "smoothed_log_bw4": (COLORS["smoothed_log_bw4"], "-",  "s", "smoothed_log bw=4"),
        "smoothed_log_bw6": (COLORS["smoothed_log_bw6"], "--", "^", "smoothed_log bw=6"),
        "fd_ratio_bw4":     (COLORS["fd_ratio_bw4"],     "--", "D", "fd_grid_ratio bw=4"),
        "tikhonov_best":    (COLORS["tikhonov_best"],    "-",  "o", "Tikhonov (tuned)"),
        "tikhonov_fixed":   (COLORS["tikhonov_fixed"],   ":",  "o", "Tikhonov (fixed λ)"),
        "spectral_best":    (COLORS["spectral_best"],    "-.",  "x", "Spectral (tuned)"),
    }

    etas = sorted(noise_df["eta"].unique())
    for mlabel, (col, ls, mk, name) in method_style.items():
        sub = noise_df[noise_df["method_label"] == mlabel]
        if sub.empty:
            continue
        # Mean rel_L2 over seeds per eta
        grp = sub.groupby("eta")["rel_L2"].mean().reset_index().sort_values("eta")
        ax.plot(grp["eta"], grp["rel_L2"],
                color=col, linestyle=ls, marker=mk, label=name, linewidth=2, markersize=7)

    ax.axhline(GRAD_GLOB_PLATEAU_B, color=COLORS["grad_glob"], linestyle=":",
               linewidth=1, alpha=0.7, label="Grad-glob plateau (0.175)")
    ax.set_xlabel("Noise level η", fontsize=12)
    ax.set_ylabel("rel_L2 vs true u₀", fontsize=12)
    ax.set_title("Test B — Noise robustness (Gaussian, T=0.15)", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "noisy_rel_L2_vs_eta_TestB.png", dpi=150)
    plt.close(fig)


def plot_field_comparison(
    x_grid: np.ndarray,
    true_u: np.ndarray,
    u_obs: np.ndarray,
    candidates: dict[str, np.ndarray],   # label -> candidate array
    title: str,
    out_path: Path,
) -> None:
    """Field comparison plot."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_grid, true_u, "k-", linewidth=2, label="True u₀")
    ax.plot(x_grid, u_obs, "k:", linewidth=1.5, alpha=0.6, label="Observed u(T)")

    style_cycle = [
        (COLORS["oracle"],            "-",  "Oracle"),
        (COLORS["smoothed_log_bw4"], "-",  "smoothed_log bw=4"),
        (COLORS["smoothed_log_bw6"], "--", "smoothed_log bw=6"),
        (COLORS["fd_ratio_bw4"],     "-.", "fd_grid_ratio bw=4"),
        (COLORS["tikhonov_best"],    ":",  "Tikhonov (tuned)"),
        (COLORS["tikhonov_fixed"],   ":",  "Tikhonov (fixed λ)"),
        (COLORS["spectral_best"],    "--", "Spectral (tuned)"),
    ]
    label_to_style = {name: (c, ls) for c, ls, name in style_cycle}

    for label, cand in candidates.items():
        if cand is None or not np.any(np.isfinite(cand)):
            continue
        c, ls = label_to_style.get(label, ("#333333", "-"))
        ax.plot(x_grid, cand, color=c, linestyle=ls, linewidth=1.8, label=label)

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("u", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------

def write_summary(
    out_dir: Path,
    df: pd.DataFrame,
    n_grid: int,
    epsilon: float,
    n_particles_convergence: list[int],
    tikhonov_clean_lam: float,
) -> None:
    lines = []
    lines.append("=" * 70)
    lines.append("VALIDATION STAGE SUMMARY")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"n_grid={n_grid}  epsilon={epsilon:.0e}")
    lines.append("=" * 70)

    # ---- Task 1: N-convergence ----
    lines.append("")
    lines.append("TASK 1 — N-CONVERGENCE (Test B, Gaussian, T=0.15, clean data)")
    lines.append("-" * 70)
    conv = df[df["task"] == "convergence"].copy()

    # Reference baselines (N=0)
    for blabel in ["tikhonov_best", "spectral_best"]:
        sub = conv[conv["method_label"] == blabel]
        if not sub.empty:
            lam = float(sub["epsilon"].iloc[0])
            rl2 = float(sub["rel_L2"].iloc[0])
            lines.append(f"  {blabel:30s}: rel_L2={rl2:.5f}  (lambda/nd={lam:.0e})")
    lines.append(f"  {'Grad-glob plateau':30s}: rel_L2={GRAD_GLOB_PLATEAU_B:.5f}")
    lines.append("")

    # Particle methods vs N
    conv_p = conv[conv["n_particles"] > 0]
    if not conv_p.empty:
        ns = sorted(conv_p["n_particles"].unique())
        mlabels = ["oracle", "smoothed_log_bw4", "smoothed_log_bw6", "fd_ratio_bw4"]
        # header
        header = f"  {'Method':<22s}" + "".join(f"  N={n:>6d}" for n in ns)
        lines.append(header)
        lines.append("  " + "-" * (22 + 10 * len(ns)))
        for ml in mlabels:
            sub = conv_p[conv_p["method_label"] == ml].sort_values("n_particles")
            if sub.empty:
                continue
            row_str = f"  {ml:<22s}"
            prev_rl2 = None
            for n in ns:
                r = sub[sub["n_particles"] == n]
                if r.empty:
                    row_str += f"  {'N/A':>8s}"
                else:
                    rl2 = float(r["rel_L2"].iloc[0])
                    row_str += f"  {rl2:8.5f}"
                    if prev_rl2 is not None and np.isfinite(rl2) and np.isfinite(prev_rl2):
                        chg = (rl2 - prev_rl2) / prev_rl2 * 100
                        row_str += f"({chg:+.1f}%)"
                    prev_rl2 = rl2
            lines.append(row_str)

    lines.append("")

    # Q1: Does N=10000 improve over N=5000?
    sl4_conv = conv_p[conv_p["method_label"] == "smoothed_log_bw4"].sort_values("n_particles")
    if len(sl4_conv) >= 2:
        ns = sl4_conv["n_particles"].tolist()
        rl2s = sl4_conv["rel_L2"].tolist()
        idx5k = next((i for i, n in enumerate(ns) if n == 5000), None)
        idx10k = next((i for i, n in enumerate(ns) if n == 10000), None)
        if idx5k is not None and idx10k is not None:
            r5, r10 = rl2s[idx5k], rl2s[idx10k]
            chg = (r10 - r5) / r5 * 100
            q1 = (f"  Q1: N=10000 rel_L2={r10:.5f} vs N=5000 rel_L2={r5:.5f} "
                  f"({chg:+.1f}%)")
            if r10 < r5 * 0.90:
                q1 += " → IMPROVEMENT (variance still matters)"
            elif r10 < r5 * 1.05:
                q1 += " → PLATEAU (bandwidth bias dominates)"
            else:
                q1 += " → WARNING: regression"
            lines.append(q1)

    # Q2: Dominant error source
    oracle_sub = conv_p[conv_p["method_label"] == "oracle"]
    if not oracle_sub.empty and not sl4_conv.empty:
        oracle_rl2 = float(oracle_sub["rel_L2"].min())
        best_est_rl2 = float(sl4_conv["rel_L2"].min())
        ratio = best_est_rl2 / oracle_rl2 if oracle_rl2 > 0 else float("nan")
        lines.append(f"  Q2: Best estimated rel_L2 = {best_est_rl2:.5f}, "
                     f"oracle = {oracle_rl2:.5f}, ratio = {ratio:.2f}×")
        if ratio < 2.0:
            lines.append("      → STRONG GO: estimated/oracle < 2×")
        elif ratio < 5.0:
            lines.append("      → CONDITIONAL GO: estimated/oracle < 5×")
        else:
            lines.append("      → STOP/REVISE: estimated/oracle ≥ 5×")

    # ---- Task 2: Noise robustness ----
    lines.append("")
    lines.append("TASK 2 — NOISE ROBUSTNESS (Test B, Gaussian, T=0.15)")
    lines.append("-" * 70)
    noise_df = df[df["task"] == "noise"].copy()

    if not noise_df.empty:
        etas = sorted(noise_df["eta"].unique())
        mlabels_noise = [
            "smoothed_log_bw4", "smoothed_log_bw6", "fd_ratio_bw4",
            "tikhonov_best", "tikhonov_fixed", "spectral_best",
        ]
        # Mean rel_L2 over seeds
        pivot = noise_df.groupby(["method_label", "eta"])["rel_L2"].mean().unstack(level="eta")

        header2 = f"  {'Method':<22s}" + "".join(f"  eta={e:5.3f}" for e in etas)
        lines.append(header2)
        lines.append("  " + "-" * (22 + 12 * len(etas)))
        for ml in mlabels_noise:
            if ml not in pivot.index:
                continue
            row_str = f"  {ml:<22s}"
            for e in etas:
                if e in pivot.columns:
                    v = pivot.loc[ml, e]
                    row_str += f"  {float(v):9.5f}"  # type: ignore[arg-type]
                else:
                    row_str += f"  {'N/A':>9s}"
            lines.append(row_str)

        lines.append("")

        # Q3-Q5 summary
        sl4_noise = noise_df[noise_df["method_label"] == "smoothed_log_bw4"]
        if not sl4_noise.empty:
            eta001 = sl4_noise[sl4_noise["eta"].round(4) == 0.001]["rel_L2"]
            eta005 = sl4_noise[sl4_noise["eta"].round(4) == 0.005]["rel_L2"]
            eta01  = sl4_noise[sl4_noise["eta"].round(4) == 0.01]["rel_L2"]

            eta001_mean = float(eta001.mean()) if not eta001.empty else float("nan")
            eta005_mean = float(eta005.mean()) if not eta005.empty else float("nan")
            eta01_mean  = float(eta01.mean())  if not eta01.empty  else float("nan")

            lines.append(f"  Q3: Noisy observations (smoothed_log bw=4):")
            lines.append(f"      eta=0.001 mean rel_L2 = {eta001_mean:.5f}")
            lines.append(f"      eta=0.005 mean rel_L2 = {eta005_mean:.5f}")
            lines.append(f"      eta=0.010 mean rel_L2 = {eta01_mean:.5f}")

            if np.isfinite(eta001_mean) and eta001_mean < 0.05:
                lines.append("      Q3 RESULT: eta=0.001 is stable → bandwidth acts as regularizer ✓")
            elif np.isfinite(eta001_mean):
                lines.append("      Q3 RESULT: eta=0.001 is unstable — noise breaks the method")

        # Tikhonov vs particle comparison
        tik_best_noise = noise_df[noise_df["method_label"] == "tikhonov_best"]
        if not tik_best_noise.empty and not sl4_noise.empty:
            eta_vals = sorted(noise_df["eta"].unique())
            for e in eta_vals:
                if e == 0.0:
                    continue
                tik_v = float(tik_best_noise[tik_best_noise["eta"].round(4) == round(e, 4)]["rel_L2"].mean())
                sl4_v = float(sl4_noise[sl4_noise["eta"].round(4) == round(e, 4)]["rel_L2"].mean())
                if np.isfinite(tik_v) and np.isfinite(sl4_v):
                    ratio_v = sl4_v / tik_v
                    lines.append(f"  Q4: eta={e:.3f}: particle/tikhonov = {sl4_v:.5f}/{tik_v:.5f} = {ratio_v:.2f}×")

    # ---- Final verdict ----
    lines.append("")
    lines.append("DECISION")
    lines.append("-" * 70)

    # Determine verdict from the data
    oracle_rl2 = float("nan")
    best_est_rl2 = float("nan")

    o_sub = df[(df["task"] == "convergence") & (df["method_label"] == "oracle")]
    if not o_sub.empty:
        oracle_rl2 = float(o_sub["rel_L2"].min())
    e_sub = df[(df["task"] == "convergence") & (df["method_label"] == "smoothed_log_bw4")
               & (df["n_particles"] > 0)]
    if not e_sub.empty:
        best_est_rl2 = float(e_sub["rel_L2"].min())

    # Noise stability at eta=0.001
    n_sub = df[(df["task"] == "noise") & (df["method_label"] == "smoothed_log_bw4")]
    eta001_stable = False
    eta005_ok = False
    if not n_sub.empty:
        eta001_v = n_sub[n_sub["eta"].round(4) == 0.001]["rel_L2"]
        eta005_v = n_sub[n_sub["eta"].round(4) == 0.005]["rel_L2"]
        eta001_stable = (not eta001_v.empty) and float(eta001_v.mean()) < 0.05
        eta005_ok = (not eta005_v.empty) and float(eta005_v.mean()) < 0.15

    ratio = best_est_rl2 / oracle_rl2 if np.isfinite(oracle_rl2) and oracle_rl2 > 0 else float("nan")

    if np.isfinite(ratio) and ratio < 2.0 and eta001_stable and eta005_ok:
        verdict = "STRONG GO"
        reason = (f"estimated/oracle = {ratio:.2f}× (< 2×) AND "
                  f"eta=0.001 stable AND eta=0.005 ok")
    elif (np.isfinite(ratio) and ratio < 5.0 and eta001_stable):
        verdict = "CONDITIONAL GO"
        reason = (f"estimated/oracle = {ratio:.2f}× (< 5×) AND eta=0.001 stable")
    else:
        verdict = "STOP/REVISE"
        if not np.isfinite(ratio):
            reason = "Could not compute estimated/oracle ratio"
        elif ratio >= 5.0:
            reason = f"estimated/oracle = {ratio:.2f}× (≥ 5×)"
        else:
            reason = "eta=0.001 noise caused instability"

    lines.append(f"  VERDICT: {verdict}")
    lines.append(f"  REASON:  {reason}")
    lines.append("")

    # Next step recommendation
    lines.append("RECOMMENDED NEXT STEP")
    lines.append("-" * 70)
    if verdict == "STRONG GO":
        lines.append("  → Write bandwidth-as-cutoff theory section.")
        lines.append("    h_opt ~ sqrt(2*alpha*T) (diffusion length scale).")
        lines.append("    Then proceed to variable-coefficient heat equation.")
    elif verdict == "CONDITIONAL GO":
        lines.append("  → Proceed to variable-coefficient heat equation.")
        lines.append("    Test H (mixture) is the remaining bottleneck.")
        lines.append("    Consider adaptive bandwidth selection before variable diffusivity.")
    else:
        lines.append("  → Investigate score blow-up mechanism.")
        lines.append("    Run hybrid oracle/estimated diagnostic to isolate")
        lines.append("    reconstruction error vs score estimation error.")

    lines.append("")
    lines.append("=" * 70)

    summary_text = "\n".join(lines)
    print("\n" + summary_text, flush=True)
    (out_dir / "validation_summary.txt").write_text(summary_text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run validation stage for density-particle method.")
    parser.add_argument("--base-config",    required=True, help="gaussian_base.yaml path")
    parser.add_argument("--mixture-config", required=True, help="gaussian_mixture.yaml path")
    parser.add_argument("--n-grid",    type=int,   default=400,
                        help="Grid resolution (default 400)")
    parser.add_argument("--epsilon",   type=float, default=1e-8,
                        help="Score denominator epsilon (default 1e-8)")
    parser.add_argument("--n-particles-convergence", type=int, nargs="+",
                        default=[5000, 10000],
                        help="Particle counts for N-convergence task (default 5000 10000)")
    parser.add_argument("--skip-n20000", action="store_true",
                        help="Do not run N=20000 even if specified")
    parser.add_argument("--n-particles-noise", type=int, default=10000,
                        help="Particle count for noise task (default 10000)")
    parser.add_argument("--eta", type=float, nargs="+", default=[0.001, 0.005, 0.01],
                        help="Noise levels for Task 2 (default 0.001 0.005 0.01)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="RNG seeds for noise task (default 0 1 2)")
    parser.add_argument("--skip-noise", action="store_true",
                        help="Skip Task 2 (noise robustness)")
    parser.add_argument("--skip-tikhonov", action="store_true",
                        help="Skip Tikhonov reference")
    parser.add_argument("--out-dir", default=None,
                        help="Override output directory")
    args = parser.parse_args()

    base_cfg = load_config(args.base_config)
    # n_steps must be set on the config; for T=0.15 and dt=0.001 → 150 steps
    # The actual value is computed by the n_steps property in Config.
    # We just patch T here and rely on cfg.n_steps.

    run_n20000 = (20000 in args.n_particles_convergence) and not args.skip_n20000

    # Output directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = args.out_dir or f"outputs/validation_stage_{ts}"
    out_dir = Path(__file__).resolve().parent.parent / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}", flush=True)

    all_rows: list[dict] = []

    # ---- TASK 1: N-convergence ----
    print("\n=== TASK 1: N-CONVERGENCE ===", flush=True)
    t1_rows, tikhonov_clean_lam = run_task1_convergence(
        base_cfg=base_cfg,
        n_particles_list=args.n_particles_convergence,
        n_grid=args.n_grid,
        epsilon=args.epsilon,
        run_n20000=run_n20000,
    )
    all_rows.extend(t1_rows)

    # ---- TASK 2: Noise robustness ----
    if not args.skip_noise:
        print("\n=== TASK 2: NOISE ROBUSTNESS ===", flush=True)
        t2_rows = run_task2_noise(
            base_cfg=base_cfg,
            n_particles=args.n_particles_noise,
            n_grid=args.n_grid,
            epsilon=args.epsilon,
            eta_list=args.eta,
            seeds=args.seeds,
            tikhonov_clean_lam=tikhonov_clean_lam,
        )
        all_rows.extend(t2_rows)

    # ---- Save CSV ----
    df = pd.DataFrame(all_rows)
    csv_path = out_dir / "validation_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}", flush=True)

    # ---- Plots ----
    print("\n=== GENERATING PLOTS ===", flush=True)
    cfg_b = build_test_b_config(base_cfg, args.n_grid)
    x_grid = make_grid(cfg_b)
    true_u = compute_true_u0(x_grid, cfg_b)
    u_obs_clean = compute_observed_final(x_grid, cfg_b)

    plot_n_convergence(df, out_dir, args.n_grid)
    print("  N_convergence_TestB.png ✓", flush=True)

    if not args.skip_noise:
        plot_noisy_rel_l2(df, out_dir)
        print("  noisy_rel_L2_vs_eta_TestB.png ✓", flush=True)

    # Field comparison at N=max(convergence list)
    max_n = max(args.n_particles_convergence)
    cands_n = {}
    for mlabel, name in [
        ("oracle",            "Oracle"),
        ("smoothed_log_bw4",  "smoothed_log bw=4"),
        ("smoothed_log_bw6",  "smoothed_log bw=6"),
        ("fd_ratio_bw4",      "fd_grid_ratio bw=4"),
        ("tikhonov_best",     "Tikhonov (tuned)"),
        ("spectral_best",     "Spectral (tuned)"),
    ]:
        sub = df[(df["task"] == "convergence") & (df["method_label"] == mlabel)]
        if mlabel in ("tikhonov_best", "spectral_best"):
            pass  # These don't store candidates in df; re-run inline
        elif not sub.empty:
            sub_n = sub[sub["n_particles"] == max_n]
            if not sub_n.empty:
                # We only have rel_L2 in CSV, need to re-run to get candidate
                pass

    # Re-run the best configurations at max N to get candidates for field plots
    print(f"  Re-running N={max_n} for field comparison plots ...", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r_oracle_max = run_density_particle_oracle_score_deterministic(
            u_obs_clean, x_grid, cfg_b, max_n,
            recon_method=RECON_METHOD, bandwidth_factor=4.0)
        r_sl4_max = run_density_particle_estimated_score_deterministic(
            u_obs_clean, x_grid, cfg_b, max_n,
            recon_method=RECON_METHOD, bandwidth_factor=4.0,
            epsilon=args.epsilon, score_method="smoothed_log", smooth_sigma_factor=1.0,
            save_snapshots=False)
        r_sl6_max = run_density_particle_estimated_score_deterministic(
            u_obs_clean, x_grid, cfg_b, max_n,
            recon_method=RECON_METHOD, bandwidth_factor=6.0,
            epsilon=args.epsilon, score_method="smoothed_log", smooth_sigma_factor=1.0,
            save_snapshots=False)
        r_fd4_max = run_density_particle_estimated_score_deterministic(
            u_obs_clean, x_grid, cfg_b, max_n,
            recon_method=RECON_METHOD, bandwidth_factor=4.0,
            epsilon=args.epsilon, score_method="fd_grid_ratio", smooth_sigma_factor=1.0,
            save_snapshots=False)

    tik_lam2, tik_cand2, _ = _find_tikhonov_best(
        u_obs_clean, x_grid, true_u, cfg_b.heat.alpha, cfg_b.heat.T, cfg_b)
    _, sp_cand2, _ = _find_spectral_best(
        u_obs_clean, x_grid, true_u, cfg_b.heat.alpha, cfg_b.heat.T, cfg_b)

    candidates_max = {
        "Oracle":            r_oracle_max.candidate,
        "smoothed_log bw=4": r_sl4_max.candidate,
        "smoothed_log bw=6": r_sl6_max.candidate,
        "fd_grid_ratio bw=4": r_fd4_max.candidate,
        "Tikhonov (tuned)":  tik_cand2,
        "Spectral (tuned)":  sp_cand2,
    }
    plot_field_comparison(
        x_grid, true_u, u_obs_clean, candidates_max,
        title=f"Test B field comparison — N={max_n} (Gaussian, T=0.15)",
        out_path=out_dir / f"field_comparison_N{max_n}_TestB.png",
    )
    print(f"  field_comparison_N{max_n}_TestB.png ✓", flush=True)

    # Noise field comparison plots
    if not args.skip_noise:
        for eta_plot, eta_tag in [(0.001, "eta001"), (0.005, "eta005"), (0.01, "eta01")]:
            if eta_plot not in args.eta:
                continue
            u_obs_n = make_noisy_obs(u_obs_clean, eta_plot, seed=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                r_sl4_n = run_density_particle_estimated_score_deterministic(
                    u_obs_n, x_grid, cfg_b, args.n_particles_noise,
                    recon_method=RECON_METHOD, bandwidth_factor=4.0,
                    epsilon=args.epsilon, score_method="smoothed_log", smooth_sigma_factor=1.0,
                    save_snapshots=False)
                r_sl6_n = run_density_particle_estimated_score_deterministic(
                    u_obs_n, x_grid, cfg_b, args.n_particles_noise,
                    recon_method=RECON_METHOD, bandwidth_factor=6.0,
                    epsilon=args.epsilon, score_method="smoothed_log", smooth_sigma_factor=1.0,
                    save_snapshots=False)
                r_fd4_n = run_density_particle_estimated_score_deterministic(
                    u_obs_n, x_grid, cfg_b, args.n_particles_noise,
                    recon_method=RECON_METHOD, bandwidth_factor=4.0,
                    epsilon=args.epsilon, score_method="fd_grid_ratio", smooth_sigma_factor=1.0,
                    save_snapshots=False)
            tik_lam_n, tik_cand_n, _ = _find_tikhonov_best(
                u_obs_n, x_grid, true_u, cfg_b.heat.alpha, cfg_b.heat.T, cfg_b)
            _, sp_cand_n, _ = _find_spectral_best(
                u_obs_n, x_grid, true_u, cfg_b.heat.alpha, cfg_b.heat.T, cfg_b)

            cands_n_plot = {
                "smoothed_log bw=4": r_sl4_n.candidate,
                "smoothed_log bw=6": r_sl6_n.candidate,
                "fd_grid_ratio bw=4": r_fd4_n.candidate,
                "Tikhonov (tuned)":  tik_cand_n,
                "Spectral (tuned)":  sp_cand_n,
            }
            plot_field_comparison(
                x_grid, true_u, u_obs_n, cands_n_plot,
                title=f"Test B noisy field comparison — η={eta_plot:.3f} seed=0",
                out_path=out_dir / f"field_comparison_noisy_TestB_{eta_tag}.png",
            )
            print(f"  field_comparison_noisy_TestB_{eta_tag}.png ✓", flush=True)

    # ---- Summary ----
    write_summary(out_dir, df, args.n_grid, args.epsilon,
                  args.n_particles_convergence, tikhonov_clean_lam)
    print(f"\nSummary: {out_dir / 'validation_summary.txt'}", flush=True)
    print(f"Done. Output: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
