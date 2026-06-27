"""
run_density_estimated_audit.py — Density-particle estimated score audit.

Purpose:
    Test whether the density-particle method works with estimated score
    (u_x / (u + epsilon)) rather than oracle score.

    The representation audit confirmed oracle density-particle beats the
    gradient-glob oracle by 55-94×.  Now the key practical question is:

    Can estimated density-particle stay within 2-5× of oracle density-particle
    and remain far below the old gradient-glob plateau (0.175 / 0.239 / 0.155)?

Scientific framing:
    gradient-glob field-score:  GRW-native but representation-mismatched.
    density-particle oracle:    representation-consistent; oracle score = ceiling.
    density-particle estimated: practical; score estimated from reconstructed density.

Methods compared:
    1. density_particle_oracle_score_deterministic   (reference upper bound)
    2. density_particle_estimated_score_deterministic (sweep epsilon/bandwidth)
    3. gradient_glob_oracle_score_deterministic       (old plateau baseline)
    REF: Tikhonov best, spectral best

Tests:
    B : Gaussian, sigma0=0.08, T=0.15
    H : Gaussian mixture, T=0.15
    Z : near-zero tail stress, T=0.05

Sweeps — fast defaults:
    n_grid           : [200, 400]
    n_particles      : [1000, 5000]
    recon_method     : "kde"
    bandwidth_factor : [1.0, 2.0]
    epsilon          : [0.0, 1e-10, 1e-8, 1e-6]

Full production sweep (--full flag):
    n_grid           : [200, 400, 800]
    n_particles      : [500, 1000, 2000, 5000, 10000]
    bandwidth_factor : [1.0, 2.0, 4.0]
    epsilon          : [0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4]

Outputs:
    outputs/density_estimated_audit_TIMESTAMP/
        density_estimated_audit_metrics.csv
        density_estimated_audit_step_diagnostics.csv
        best_by_test.csv
        density_estimated_audit_summary.txt
        oracle_vs_estimated_density_rel_L2_{test}.png
        estimated_density_rel_L2_vs_epsilon_{test}.png
        estimated_density_rel_L2_vs_bandwidth_{test}.png
        score_diagnostics_vs_step_{test}.png
        step_zero_reconstruction_error_vs_n_particles_{test}.png
        field_comparison_best_estimated_density_{test}.png
        residual_best_estimated_density_{test}.png

Usage:
    PYTHONPATH=src python scripts/run_density_estimated_audit.py \\
        --base-config configs/gaussian_base.yaml \\
        --mixture-config configs/gaussian_mixture.yaml

    # Full sweep:
    PYTHONPATH=src python scripts/run_density_estimated_audit.py \\
        --base-config configs/gaussian_base.yaml \\
        --mixture-config configs/gaussian_mixture.yaml \\
        --full
"""

from __future__ import annotations

import sys
import argparse
import copy
import time
import warnings
from datetime import datetime
from pathlib import Path

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
    run_oracle_score_deterministic,
    run_density_particle_oracle_score_deterministic,
    run_density_particle_estimated_score_deterministic,
)
from invheat_grw.metrics import (
    compute_metrics,
    compute_baseline_metrics,
    compute_wasserstein,
)
from invheat_grw.baselines import spectral_cutoff_inverse, tikhonov_inverse


# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------

# Fast defaults (~1-3 min runtime)
N_GRID_DEFAULT = [200, 400]
N_PARTICLES_DEFAULT = [1000, 5000]
BANDWIDTH_FACTORS_DEFAULT = [1.0, 2.0]
EPSILON_DEFAULT = [0.0, 1e-10, 1e-8, 1e-6]

# Full production sweep (may take 15-45 min depending on hardware)
N_GRID_FULL = [200, 400, 800]
N_PARTICLES_FULL = [500, 1000, 2000, 5000, 10000]
BANDWIDTH_FACTORS_FULL = [1.0, 2.0, 4.0]
EPSILON_FULL = [0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4]

RECON_METHOD = "kde"  # first-pass focus on KDE
GLOB_VALUES = [20, 80]
TIKHONOV_LAMBDAS = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
NOISE_DELTA_VALUES = [1e-8, 1e-6, 1e-4, 1e-2]
RNG_SEED = 42

# Known gradient-glob oracle plateaus from convergence study
GRAD_GLOB_ORACLE_PLATEAUS = {"B": 0.175, "H": 0.239, "Z": 0.155}


# ---------------------------------------------------------------------------
# Color scheme
# ---------------------------------------------------------------------------

COLOR_ORACLE_DENSITY = "#1f77b4"   # blue
COLOR_ESTIMATED_RAW  = "#ff7f0e"   # orange
COLOR_ESTIMATED_BEST = "#2ca02c"   # green
COLOR_GRAD_GLOB      = "#9467bd"   # purple
COLOR_TIKHONOV       = "#8c564b"   # brown
COLOR_SPECTRAL       = "#e377c2"   # pink
COLOR_PLATEAU        = "#aaaaaa"   # gray


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def patch_config(cfg: Config, **overrides) -> Config:
    new_cfg = copy.deepcopy(cfg)
    for key, val in overrides.items():
        parts = key.split(".")
        obj = new_cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], val)
    return new_cfg


def build_test_configs(base_cfg: Config, mixture_cfg: Config) -> dict[str, Config]:
    return {
        "B": patch_config(base_cfg, **{
            "heat.T": 0.15,
            "initial_condition.sigma0": 0.08,
        }),
        "H": patch_config(mixture_cfg, **{"heat.T": 0.15}),
        "Z": patch_config(base_cfg, **{
            "heat.T": 0.05,
            "initial_condition.sigma0": 0.05,
            "initial_condition.mu": 0.4,
        }),
    }


def _safe(v) -> float:
    try:
        f = float(v)
        return f if np.isfinite(f) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _listval(lst: list, idx: int) -> float:
    if lst and idx < len(lst):
        return _safe(lst[idx])
    if lst and idx == -1 and len(lst) > 0:
        return _safe(lst[-1])
    return float("nan")


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------

def _base_row(test: str, n_grid: int, method_name: str, method_type: str) -> dict:
    return {
        "test": test,
        "n_grid": n_grid,
        "method_name": method_name,
        "method_type": method_type,
        "n_particles": 0,
        "globs_per_jump": 0,
        "recon_method": "",
        "bandwidth_factor": float("nan"),
        "bandwidth_used": float("nan"),
        "epsilon": float("nan"),
        "completed": False,
        "failure_step": "",
        "failure_msg": "",
        "step_zero_recon_error": float("nan"),
        "relative_l2": float("nan"),
        "l2_error": float("nan"),
        "linf_error": float("nan"),
        "forward_consistency_l2": float("nan"),
        "wasserstein": float("nan"),
        "mass_rel_error": float("nan"),
        "total_variation": float("nan"),
        "peak_value": float("nan"),
        "peak_ratio": float("nan"),
        "A_fit": float("nan"),
        "mu_fit": float("nan"),
        "sigma_fit": float("nan"),
        "fit_success": False,
        "fit_rmse": float("nan"),
        "mass_candidate": float("nan"),
        "mass_true": float("nan"),
        "runtime_seconds": float("nan"),
        "max_score_L2_error_vs_oracle": float("nan"),
        "mean_score_L2_error_vs_oracle": float("nan"),
        "max_abs_score_final": float("nan"),
        "n_denom_below_eps_total": 0,
        "n_clipped_total": 0,
    }


def _fill_metrics(row: dict, result, m, wass: float) -> dict:
    """Fill a row dict from a MethodResult + MethodMetrics + wasserstein."""
    row.update({
        "completed": result.completed,
        "failure_step": result.failure_step if result.failure_step is not None else "",
        "failure_msg": result.failure_msg or "",
        "step_zero_recon_error": _safe(getattr(result, "step_zero_recon_error", float("nan"))),
        "relative_l2": _safe(m.relative_l2),
        "l2_error": _safe(m.l2_error),
        "linf_error": _safe(m.linf_error),
        "forward_consistency_l2": _safe(m.forward_consistency_l2),
        "wasserstein": _safe(wass),
        "mass_rel_error": _safe(m.mass_rel_error),
        "total_variation": _safe(m.total_variation),
        "peak_value": _safe(m.peak_value),
        "peak_ratio": _safe(m.peak_ratio),
        "A_fit": _safe(m.A_fit),
        "mu_fit": _safe(m.mu_fit),
        "sigma_fit": _safe(m.sigma_fit),
        "fit_success": m.fit_success,
        "fit_rmse": _safe(m.fit_rmse),
        "mass_candidate": _safe(m.mass_candidate),
        "mass_true": _safe(m.mass_true),
        "runtime_seconds": _safe(result.runtime_seconds),
        "max_abs_score_final": _listval(result.score_max_abs, -1),
        "max_score_L2_error_vs_oracle": (
            max((v for v in result.score_L2_error_vs_oracle if np.isfinite(v)),
                default=float("nan"))
            if result.score_L2_error_vs_oracle else float("nan")
        ),
        "mean_score_L2_error_vs_oracle": (
            float(np.mean([v for v in result.score_L2_error_vs_oracle
                           if np.isfinite(v)]))
            if any(np.isfinite(v) for v in result.score_L2_error_vs_oracle)
            else float("nan")
        ),
        "n_denom_below_eps_total": int(sum(result.n_denominator_below_epsilon)),
        "n_clipped_total": int(sum(result.n_clipped_scores)),
    })
    return row


# ---------------------------------------------------------------------------
# Cell functions
# ---------------------------------------------------------------------------

def run_density_oracle_cell(
    test: str,
    cfg: Config,
    n_grid: int,
    n_particles: int,
    bw_factor: float,
) -> tuple[dict, list[dict]]:
    """Run density-particle oracle cell. Returns (metric_row, step_rows)."""
    cfg_res = patch_config(cfg, **{"domain.n_grid": n_grid})
    x_grid = make_grid(cfg_res)
    true_u = compute_true_u0(x_grid, cfg_res)
    u_obs = compute_observed_final(x_grid, cfg_res)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = run_density_particle_oracle_score_deterministic(
            u_obs, x_grid, cfg_res, n_particles,
            recon_method=RECON_METHOD, bandwidth_factor=bw_factor)
        m = compute_metrics(result, true_u, u_obs, x_grid, cfg_res)
        wass = compute_wasserstein(result.candidate, true_u, x_grid)

    row = _base_row(test, n_grid, result.method_name, "density_particle_oracle")
    row.update({
        "n_particles": n_particles,
        "recon_method": RECON_METHOD,
        "bandwidth_factor": bw_factor,
        "bandwidth_used": _safe(result.bandwidth_used),
        "epsilon": 0.0,
    })
    _fill_metrics(row, result, m, wass)

    # Per-step diagnostics (score error = 0 for oracle; max_abs_score is informative)
    n_steps_recorded = len(result.score_max_abs)
    step_rows = [
        {
            "test": test, "n_grid": n_grid, "n_particles": n_particles,
            "epsilon": 0.0, "bandwidth_factor": bw_factor,
            "method_type": "density_particle_oracle",
            "step": step,
            "score_L2_error_vs_oracle": 0.0,
            "score_core_L2_error_vs_oracle": 0.0,
            "score_Linf_error_vs_oracle": 0.0,
            "score_core_Linf_error_vs_oracle": 0.0,
            "max_abs_score": _listval(result.score_max_abs, step),
            "mean_abs_score": _listval(result.score_max_abs, step),
            "score_std": _listval(result.score_std, step),
            "n_denom_below_eps": 0,
            "n_clipped": 0,
            "mass": float("nan"),
            "epsilon_actual": 0.0,
        }
        for step in range(n_steps_recorded)
    ]
    return row, step_rows


def run_density_estimated_cell(
    test: str,
    cfg: Config,
    n_grid: int,
    n_particles: int,
    bw_factor: float,
    epsilon: float,
) -> tuple[dict, list[dict]]:
    """Run density-particle estimated cell. Returns (metric_row, step_rows)."""
    cfg_res = patch_config(cfg, **{"domain.n_grid": n_grid})
    x_grid = make_grid(cfg_res)
    true_u = compute_true_u0(x_grid, cfg_res)
    u_obs = compute_observed_final(x_grid, cfg_res)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = run_density_particle_estimated_score_deterministic(
            u_obs, x_grid, cfg_res, n_particles,
            recon_method=RECON_METHOD,
            bandwidth_factor=bw_factor,
            epsilon=epsilon,
            scale_epsilon_by_peak=False,
            score_clipping=None,
            save_snapshots=False,
        )
        m = compute_metrics(result, true_u, u_obs, x_grid, cfg_res)
        wass = compute_wasserstein(result.candidate, true_u, x_grid)

    row = _base_row(test, n_grid, result.method_name, "density_particle_estimated")
    row.update({
        "n_particles": n_particles,
        "recon_method": RECON_METHOD,
        "bandwidth_factor": bw_factor,
        "bandwidth_used": _safe(result.bandwidth_used),
        "epsilon": epsilon,
    })
    _fill_metrics(row, result, m, wass)

    # Per-step diagnostics
    n_steps_recorded = len(result.score_L2_error_vs_oracle)
    step_rows = [
        {
            "test": test, "n_grid": n_grid, "n_particles": n_particles,
            "epsilon": epsilon, "bandwidth_factor": bw_factor,
            "method_type": "density_particle_estimated",
            "step": step,
            "score_L2_error_vs_oracle": _listval(result.score_L2_error_vs_oracle, step),
            "score_core_L2_error_vs_oracle": _listval(result.score_core_L2_error_vs_oracle, step),
            "score_Linf_error_vs_oracle": _listval(result.score_Linf_error_vs_oracle, step),
            "score_core_Linf_error_vs_oracle": _listval(result.score_core_Linf_error_vs_oracle, step),
            "max_abs_score": _listval(result.score_max_abs, step),
            "mean_abs_score": _listval(result.mean_abs_score, step),
            "score_std": _listval(result.score_std, step),
            "n_denom_below_eps": _listval(result.n_denominator_below_epsilon, step),
            "n_clipped": _listval(result.n_clipped_scores, step),
            "mass": _listval(result.mass_per_step, step),
            "epsilon_actual": _listval(result.epsilon_actual_per_step, step),
        }
        for step in range(n_steps_recorded)
    ]
    return row, step_rows


def run_gradient_glob_cell(
    test: str,
    cfg: Config,
    n_grid: int,
    globs_per_jump: int,
) -> dict:
    cfg_res = patch_config(cfg, **{
        "domain.n_grid": n_grid,
        "grw.gradient_globs_per_jump": globs_per_jump,
    })
    x_grid = make_grid(cfg_res)
    true_u = compute_true_u0(x_grid, cfg_res)
    u_obs = compute_observed_final(x_grid, cfg_res)
    rng = np.random.default_rng(RNG_SEED)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = run_oracle_score_deterministic(u_obs, x_grid, cfg_res, rng)
        m = compute_metrics(result, true_u, u_obs, x_grid, cfg_res)
        wass = compute_wasserstein(result.candidate, true_u, x_grid)

    row = _base_row(test, n_grid, result.method_name, "gradient_glob_oracle")
    row.update({"globs_per_jump": globs_per_jump})
    _fill_metrics(row, result, m, wass)
    return row


def run_baseline_cells(
    test: str,
    cfg: Config,
    n_grid: int,
) -> list[dict]:
    cfg_res = patch_config(cfg, **{"domain.n_grid": n_grid})
    x_grid = make_grid(cfg_res)
    true_u = compute_true_u0(x_grid, cfg_res)
    u_obs = compute_observed_final(x_grid, cfg_res)
    alpha = cfg_res.heat.alpha
    T_val = cfg_res.heat.T
    rows = []

    # Tikhonov: sweep, keep best
    best_tik_cand = None
    best_tik_rl2 = float("inf")
    best_tik_lam = float("nan")
    for lam in TIKHONOV_LAMBDAS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            tr = tikhonov_inverse(u_obs, x_grid, alpha, T_val, lam)
        if np.all(np.isfinite(tr.candidate)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                bm = compute_baseline_metrics(tr.candidate, true_u, u_obs, x_grid, cfg_res,
                                              method_name="tikhonov_best")
            if np.isfinite(bm.relative_l2) and bm.relative_l2 < best_tik_rl2:
                best_tik_rl2 = bm.relative_l2
                best_tik_cand = tr.candidate.copy()
                best_tik_lam = lam

    if best_tik_cand is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = compute_baseline_metrics(best_tik_cand, true_u, u_obs, x_grid, cfg_res,
                                         method_name="tikhonov_best")
            wass = compute_wasserstein(best_tik_cand, true_u, x_grid)
        row = _base_row(test, n_grid, "tikhonov_best", "baseline_tikhonov")
        row.update({
            "completed": True,
            "relative_l2": _safe(m.relative_l2),
            "l2_error": _safe(m.l2_error),
            "linf_error": _safe(m.linf_error),
            "forward_consistency_l2": _safe(m.forward_consistency_l2),
            "wasserstein": _safe(wass),
            "mass_rel_error": _safe(m.mass_rel_error),
            "total_variation": _safe(m.total_variation),
            "peak_value": _safe(m.peak_value),
            "peak_ratio": _safe(m.peak_ratio),
            "A_fit": _safe(m.A_fit), "mu_fit": _safe(m.mu_fit), "sigma_fit": _safe(m.sigma_fit),
            "fit_success": m.fit_success, "fit_rmse": _safe(m.fit_rmse),
            "mass_candidate": _safe(m.mass_candidate), "mass_true": _safe(m.mass_true),
            "bandwidth_factor": best_tik_lam,  # repurpose column to store lambda
        })
        rows.append(row)

    # Spectral: sweep, keep best
    best_sc_cand = None
    best_sc_rl2 = float("inf")
    best_sc_nd = float("nan")
    for nd in NOISE_DELTA_VALUES:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sr = spectral_cutoff_inverse(u_obs, x_grid, alpha, T_val, noise_delta=nd)
        if np.all(np.isfinite(sr.candidate)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                bm = compute_baseline_metrics(sr.candidate, true_u, u_obs, x_grid, cfg_res,
                                              method_name="spectral_cutoff_best")
            if np.isfinite(bm.relative_l2) and bm.relative_l2 < best_sc_rl2:
                best_sc_rl2 = bm.relative_l2
                best_sc_cand = sr.candidate.copy()
                best_sc_nd = nd

    if best_sc_cand is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = compute_baseline_metrics(best_sc_cand, true_u, u_obs, x_grid, cfg_res,
                                         method_name="spectral_cutoff_best")
            wass = compute_wasserstein(best_sc_cand, true_u, x_grid)
        row = _base_row(test, n_grid, "spectral_cutoff_best", "baseline_spectral")
        row.update({
            "completed": True,
            "relative_l2": _safe(m.relative_l2),
            "l2_error": _safe(m.l2_error),
            "linf_error": _safe(m.linf_error),
            "forward_consistency_l2": _safe(m.forward_consistency_l2),
            "wasserstein": _safe(wass),
            "mass_rel_error": _safe(m.mass_rel_error),
            "total_variation": _safe(m.total_variation),
            "peak_value": _safe(m.peak_value),
            "peak_ratio": _safe(m.peak_ratio),
            "A_fit": _safe(m.A_fit), "mu_fit": _safe(m.mu_fit), "sigma_fit": _safe(m.sigma_fit),
            "fit_success": m.fit_success, "fit_rmse": _safe(m.fit_rmse),
            "mass_candidate": _safe(m.mass_candidate), "mass_true": _safe(m.mass_true),
            "bandwidth_factor": best_sc_nd,  # repurpose column for noise_delta
        })
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_oracle_vs_estimated(df: pd.DataFrame, out_dir: Path) -> None:
    """Per test: rel_L2 vs n_particles.  Oracle + estimated per epsilon + plateau + Tikhonov."""
    for test in sorted(df["test"].unique()):
        sub = df[df["test"] == test]
        n_grid_max = int(sub["n_grid"].max())
        sub_max = sub[sub["n_grid"] == n_grid_max]

        fig, ax = plt.subplots(figsize=(9, 5))

        # Oracle density: best bw per n_particles
        dp_or = sub_max[sub_max["method_type"] == "density_particle_oracle"].copy()
        if len(dp_or) > 0:
            best_or = (dp_or[np.isfinite(dp_or["relative_l2"])]
                       .groupby("n_particles")["relative_l2"].min()
                       .reset_index()
                       .sort_values("n_particles"))
            ax.plot(best_or["n_particles"], best_or["relative_l2"],
                    "o-", color=COLOR_ORACLE_DENSITY, linewidth=2.0,
                    label="density-particle oracle (best bw)", zorder=4)

        # Estimated density: per epsilon, best bw
        dp_est = sub_max[sub_max["method_type"] == "density_particle_estimated"].copy()
        if len(dp_est) > 0:
            eps_sorted = sorted(dp_est["epsilon"].unique())
            colors_est = plt.cm.tab10(np.linspace(0.15, 0.85, len(eps_sorted)))  # type: ignore[attr-defined]
            for i, eps_val in enumerate(eps_sorted):
                sub_eps = dp_est[dp_est["epsilon"] == eps_val]
                best_eps = (sub_eps[np.isfinite(sub_eps["relative_l2"])]
                            .groupby("n_particles")["relative_l2"].min()
                            .reset_index()
                            .sort_values("n_particles"))
                if len(best_eps) == 0:
                    continue
                label = f"estimated raw (ε=0)" if eps_val == 0.0 else f"estimated ε={eps_val:.0e}"
                color = COLOR_ESTIMATED_RAW if eps_val == 0.0 else colors_est[i]
                ls = "-" if eps_val == 0.0 else "--"
                ax.plot(best_eps["n_particles"], best_eps["relative_l2"],
                        marker="s", linestyle=ls, color=color, alpha=0.8, label=label)

        # Gradient-glob plateau
        plateau = GRAD_GLOB_ORACLE_PLATEAUS.get(test, float("nan"))
        if np.isfinite(plateau):
            ax.axhline(plateau, color=COLOR_PLATEAU, linestyle="--", linewidth=1.0,
                       label=f"grad-glob oracle plateau ({plateau:.3f})", alpha=0.7)

        # Tikhonov reference
        tik = sub_max[sub_max["method_type"] == "baseline_tikhonov"]
        if len(tik) > 0:
            rl2_tik = _safe(tik["relative_l2"].values[0])
            if np.isfinite(rl2_tik):
                ax.axhline(rl2_tik, color=COLOR_TIKHONOV, linestyle=":",
                           linewidth=1.5, label=f"Tikhonov best ({rl2_tik:.4f})", alpha=0.85)

        ax.set_xlabel("n_particles")
        ax.set_ylabel("relative L2")
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_title(f"Test {test}: oracle vs estimated density-particle (n_grid={n_grid_max})")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"oracle_vs_estimated_density_rel_L2_{test}.png", dpi=150)
        plt.close(fig)


def plot_estimated_vs_epsilon(df: pd.DataFrame, out_dir: Path) -> None:
    """Per test: estimated rel_L2 vs epsilon, per n_particles."""
    for test in sorted(df["test"].unique()):
        sub = df[df["test"] == test]
        n_grid_max = int(sub["n_grid"].max())
        sub_max = sub[sub["n_grid"] == n_grid_max]
        dp_est = sub_max[sub_max["method_type"] == "density_particle_estimated"].copy()

        if len(dp_est) == 0:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))

        for n_part in sorted(dp_est["n_particles"].unique()):
            sub_np = dp_est[dp_est["n_particles"] == n_part]
            # best bw per epsilon
            by_eps = (sub_np[np.isfinite(sub_np["relative_l2"])]
                      .groupby("epsilon")["relative_l2"].min()
                      .reset_index()
                      .sort_values("epsilon"))
            if len(by_eps) < 2:
                continue
            # Log x-axis: replace eps=0 with a placeholder for display
            eps_plot = np.where(by_eps["epsilon"].values == 0.0, 1e-14,
                                by_eps["epsilon"].values)
            ax.plot(eps_plot, by_eps["relative_l2"].values, "o-",
                    label=f"N={n_part}")

        # Oracle reference
        dp_or = sub_max[sub_max["method_type"] == "density_particle_oracle"]
        if len(dp_or) > 0:
            best_or_rl2 = float(dp_or["relative_l2"].min())
            if np.isfinite(best_or_rl2):
                ax.axhline(best_or_rl2, color=COLOR_ORACLE_DENSITY, linestyle=":",
                           linewidth=1.5, label=f"oracle best ({best_or_rl2:.4f})")

        plateau = GRAD_GLOB_ORACLE_PLATEAUS.get(test, float("nan"))
        if np.isfinite(plateau):
            ax.axhline(plateau, color=COLOR_PLATEAU, linestyle="--", linewidth=1.0,
                       label=f"grad-glob plateau ({plateau:.3f})", alpha=0.7)

        ax.set_xlabel("epsilon (denominator floor) — 1e-14 marker = raw (ε=0)")
        ax.set_xscale("log")
        ax.set_ylabel("relative L2")
        ax.set_yscale("log")
        ax.set_title(f"Test {test}: estimated density rel_L2 vs epsilon (n_grid={n_grid_max})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"estimated_density_rel_L2_vs_epsilon_{test}.png", dpi=150)
        plt.close(fig)


def plot_estimated_vs_bandwidth(df: pd.DataFrame, out_dir: Path) -> None:
    """Per test: estimated rel_L2 vs bandwidth_factor, per n_particles."""
    for test in sorted(df["test"].unique()):
        sub = df[df["test"] == test]
        n_grid_max = int(sub["n_grid"].max())
        sub_max = sub[sub["n_grid"] == n_grid_max]
        dp_est = sub_max[sub_max["method_type"] == "density_particle_estimated"].copy()

        if len(dp_est) == 0:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))

        for n_part in sorted(dp_est["n_particles"].unique()):
            sub_np = dp_est[dp_est["n_particles"] == n_part]
            # best epsilon per bandwidth_factor
            by_bw = (sub_np[np.isfinite(sub_np["relative_l2"])]
                     .groupby("bandwidth_factor")["relative_l2"].min()
                     .reset_index()
                     .sort_values("bandwidth_factor"))
            if len(by_bw) < 2:
                continue
            ax.plot(by_bw["bandwidth_factor"].values, by_bw["relative_l2"].values,
                    "o-", label=f"N={n_part}")

        # Oracle reference
        dp_or = sub_max[sub_max["method_type"] == "density_particle_oracle"]
        if len(dp_or) > 0:
            best_or_rl2 = float(dp_or["relative_l2"].min())
            if np.isfinite(best_or_rl2):
                ax.axhline(best_or_rl2, color=COLOR_ORACLE_DENSITY, linestyle=":",
                           linewidth=1.5, label=f"oracle best ({best_or_rl2:.4f})")

        plateau = GRAD_GLOB_ORACLE_PLATEAUS.get(test, float("nan"))
        if np.isfinite(plateau):
            ax.axhline(plateau, color=COLOR_PLATEAU, linestyle="--", linewidth=1.0,
                       label=f"grad-glob plateau ({plateau:.3f})", alpha=0.7)

        ax.set_xlabel("bandwidth_factor (× dx)")
        ax.set_ylabel("relative L2")
        ax.set_yscale("log")
        ax.set_title(f"Test {test}: estimated density rel_L2 vs bandwidth (n_grid={n_grid_max})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"estimated_density_rel_L2_vs_bandwidth_{test}.png", dpi=150)
        plt.close(fig)


def plot_score_diagnostics_vs_step(
    df: pd.DataFrame,
    step_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Per test: score_L2_error_vs_oracle and max_abs_score vs step.
    Shows best estimated config, raw (eps=0) config, and oracle (zero line)."""
    if len(step_df) == 0:
        return

    for test in sorted(df["test"].unique()):
        sub = df[df["test"] == test]
        n_grid_max = int(sub["n_grid"].max())
        dp_est = sub[(sub["method_type"] == "density_particle_estimated") &
                     (sub["n_grid"] == n_grid_max) & np.isfinite(sub["relative_l2"])]

        if len(dp_est) == 0:
            continue

        # Best estimated config
        best_idx = dp_est["relative_l2"].idxmin()
        best_row = dp_est.loc[best_idx]
        best_eps = float(best_row["epsilon"])
        best_bw = float(best_row["bandwidth_factor"])
        best_np = int(best_row["n_particles"])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax_err, ax_score = axes

        def _filter_steps(eps, bw, np_val):
            return step_df[
                (step_df["test"] == test) &
                (step_df["n_grid"] == n_grid_max) &
                (step_df["n_particles"] == np_val) &
                (step_df["epsilon"].round(20) == round(eps, 20)) &
                (step_df["bandwidth_factor"].round(6) == round(bw, 6)) &
                (step_df["method_type"] == "density_particle_estimated")
            ].sort_values("step")

        # Best config
        steps_best = _filter_steps(best_eps, best_bw, best_np)
        if len(steps_best) > 0:
            label_best = f"best (ε={best_eps:.0e}, bw={best_bw}, N={best_np})"
            ax_err.plot(steps_best["step"], steps_best["score_L2_error_vs_oracle"],
                        color=COLOR_ESTIMATED_BEST, linewidth=1.5, label=label_best)
            ax_score.plot(steps_best["step"], steps_best["max_abs_score"],
                          color=COLOR_ESTIMATED_BEST, linewidth=1.5, label=label_best)

        # Raw (eps=0, same bw, same N) — if exists and differs
        if best_eps != 0.0:
            steps_raw = _filter_steps(0.0, best_bw, best_np)
            if len(steps_raw) > 0:
                label_raw = f"raw (ε=0, bw={best_bw}, N={best_np})"
                ax_err.plot(steps_raw["step"], steps_raw["score_L2_error_vs_oracle"],
                            color=COLOR_ESTIMATED_RAW, linewidth=1.5, linestyle="--",
                            label=label_raw)
                ax_score.plot(steps_raw["step"], steps_raw["max_abs_score"],
                              color=COLOR_ESTIMATED_RAW, linewidth=1.5, linestyle="--",
                              label=label_raw)

        # Oracle step rows (score error = 0)
        oracle_steps = step_df[
            (step_df["test"] == test) &
            (step_df["n_grid"] == n_grid_max) &
            (step_df["n_particles"] == best_np) &
            (step_df["method_type"] == "density_particle_oracle")
        ].sort_values("step")
        if len(oracle_steps) > 0:
            ax_err.plot(oracle_steps["step"],
                        np.zeros(len(oracle_steps)),
                        color=COLOR_ORACLE_DENSITY, linewidth=1.0, linestyle=":",
                        label="oracle (score error = 0)")
            ax_score.plot(oracle_steps["step"], oracle_steps["max_abs_score"],
                          color=COLOR_ORACLE_DENSITY, linewidth=1.0, linestyle=":",
                          label=f"oracle (N={best_np})")

        ax_err.set_xlabel("backward step")
        ax_err.set_ylabel("score L2 error vs oracle")
        ax_err.set_title(f"Test {test}: score error vs step (n_grid={n_grid_max})")
        ax_err.legend(fontsize=8)
        ax_err.grid(True, alpha=0.3)

        ax_score.set_xlabel("backward step")
        ax_score.set_ylabel("max |score| at particles")
        ax_score.set_title(f"Test {test}: max abs score vs step (n_grid={n_grid_max})")
        ax_score.legend(fontsize=8)
        ax_score.grid(True, alpha=0.3)

        fig.suptitle(f"Test {test}: per-step score diagnostics", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / f"score_diagnostics_vs_step_{test}.png", dpi=150)
        plt.close(fig)


def plot_step_zero_error(df: pd.DataFrame, out_dir: Path) -> None:
    """Per test: step_zero_recon_error vs n_particles for oracle and estimated."""
    for test in sorted(df["test"].unique()):
        sub = df[df["test"] == test]
        n_grid_max = int(sub["n_grid"].max())
        sub_max = sub[sub["n_grid"] == n_grid_max]

        fig, ax = plt.subplots(figsize=(7, 5))

        dp_or = sub_max[sub_max["method_type"] == "density_particle_oracle"].copy()
        if len(dp_or) > 0:
            by_np = (dp_or[np.isfinite(dp_or["step_zero_recon_error"])]
                     .groupby("n_particles")["step_zero_recon_error"].min()
                     .reset_index().sort_values("n_particles"))
            if len(by_np) > 0:
                ax.plot(by_np["n_particles"], by_np["step_zero_recon_error"],
                        "o-", color=COLOR_ORACLE_DENSITY,
                        label="oracle (step-zero recon error)")

        dp_est = sub_max[sub_max["method_type"] == "density_particle_estimated"].copy()
        if len(dp_est) > 0:
            by_np = (dp_est[np.isfinite(dp_est["step_zero_recon_error"])]
                     .groupby("n_particles")["step_zero_recon_error"].min()
                     .reset_index().sort_values("n_particles"))
            if len(by_np) > 0:
                ax.plot(by_np["n_particles"], by_np["step_zero_recon_error"],
                        "s--", color=COLOR_ESTIMATED_RAW,
                        label="estimated (step-zero recon error)")

        ax.set_xlabel("n_particles")
        ax.set_ylabel("step-zero reconstruction error (relative L2)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"Test {test}: step-zero error vs n_particles (n_grid={n_grid_max})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"step_zero_reconstruction_error_vs_n_particles_{test}.png",
                    dpi=150)
        plt.close(fig)


def plot_field_comparison(
    df: pd.DataFrame,
    test: str,
    test_cfgs: dict,
    out_dir: Path,
) -> None:
    """Overlay: true u0, uT, oracle density, best estimated density, grad-glob, Tikhonov."""
    sub = df[df["test"] == test]
    n_grid_max = int(sub["n_grid"].max())
    test_cfg = test_cfgs[test]
    cfg_res = patch_config(test_cfg, **{"domain.n_grid": n_grid_max})
    x_grid = make_grid(cfg_res)
    true_u = compute_true_u0(x_grid, cfg_res)
    u_obs = compute_observed_final(x_grid, cfg_res)
    T_val = cfg_res.heat.T

    sub_max = sub[sub["n_grid"] == n_grid_max]

    # Best estimated
    dp_est = sub_max[(sub_max["method_type"] == "density_particle_estimated") &
                     np.isfinite(sub_max["relative_l2"])]
    if len(dp_est) == 0:
        return
    best_est_idx = dp_est["relative_l2"].idxmin()
    best_est = dp_est.loc[best_est_idx]
    est_np = int(best_est["n_particles"])  # type: ignore[arg-type]
    est_bw = float(best_est["bandwidth_factor"])  # type: ignore[arg-type]
    est_eps = float(best_est["epsilon"])  # type: ignore[arg-type]
    est_rl2 = float(best_est["relative_l2"])  # type: ignore[arg-type]

    # Best oracle
    dp_or = sub_max[(sub_max["method_type"] == "density_particle_oracle") &
                    np.isfinite(sub_max["relative_l2"])]
    if len(dp_or) > 0:
        best_or_idx = dp_or["relative_l2"].idxmin()
        or_np = int(dp_or.loc[best_or_idx, "n_particles"])  # type: ignore[arg-type]
        or_bw = float(dp_or.loc[best_or_idx, "bandwidth_factor"])  # type: ignore[arg-type]
        or_rl2 = float(dp_or.loc[best_or_idx, "relative_l2"])  # type: ignore[arg-type]
    else:
        or_np, or_bw, or_rl2 = est_np, est_bw, float("nan")

    # Best gradient-glob
    gg_sub = sub_max[(sub_max["method_type"] == "gradient_glob_oracle") &
                     np.isfinite(sub_max["relative_l2"])]
    gg_globs = (int(gg_sub.loc[gg_sub["relative_l2"].idxmin(), "globs_per_jump"])  # type: ignore[arg-type]
                if len(gg_sub) > 0 else GLOB_VALUES[-1])

    # Recompute candidates
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        est_result = run_density_particle_estimated_score_deterministic(
            u_obs, x_grid, cfg_res, est_np,
            recon_method=RECON_METHOD, bandwidth_factor=est_bw,
            epsilon=est_eps, scale_epsilon_by_peak=False,
            score_clipping=None, save_snapshots=False)

        or_result = run_density_particle_oracle_score_deterministic(
            u_obs, x_grid, cfg_res, or_np,
            recon_method=RECON_METHOD, bandwidth_factor=or_bw)

        gg_cfg = patch_config(cfg_res, **{"grw.gradient_globs_per_jump": gg_globs})
        rng = np.random.default_rng(RNG_SEED)
        gg_result = run_oracle_score_deterministic(u_obs, x_grid, gg_cfg, rng)

    # Tikhonov best
    alpha = cfg_res.heat.alpha
    best_tik_cand = None
    best_tik_rl2 = float("inf")
    for lam in TIKHONOV_LAMBDAS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            tr = tikhonov_inverse(u_obs, x_grid, alpha, T_val, lam)
        if np.all(np.isfinite(tr.candidate)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                bm = compute_baseline_metrics(tr.candidate, true_u, u_obs, x_grid, cfg_res,
                                              method_name="tikhonov_best")
            if np.isfinite(bm.relative_l2) and bm.relative_l2 < best_tik_rl2:
                best_tik_rl2 = bm.relative_l2
                best_tik_cand = tr.candidate.copy()

    # Field comparison + residual subplot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax, ax2 = axes

    ax.plot(x_grid, true_u, "k-", linewidth=2.0, label="true u₀", zorder=5)
    ax.plot(x_grid, u_obs, "k--", linewidth=1.0,
            label=f"observed u(T={T_val:.2f})", alpha=0.5)
    ax.plot(x_grid, or_result.candidate, "-",
            color=COLOR_ORACLE_DENSITY, linewidth=1.5,
            label=f"oracle (N={or_np}, bw={or_bw}, rl2={or_rl2:.4f})")
    ax.plot(x_grid, est_result.candidate, "-",
            color=COLOR_ESTIMATED_BEST, linewidth=1.5,
            label=f"estimated (ε={est_eps:.0e}, bw={est_bw}, N={est_np}, rl2={est_rl2:.4f})")
    ax.plot(x_grid, gg_result.candidate, "-",
            color=COLOR_GRAD_GLOB, linewidth=1.5,
            label=f"gradient-glob oracle (g={gg_globs})")
    if best_tik_cand is not None:
        ax.plot(x_grid, best_tik_cand, ":",
                color=COLOR_TIKHONOV, linewidth=1.5,
                label=f"Tikhonov best (rl2={best_tik_rl2:.4f})")

    ax.set_title(f"Test {test}: field comparison (n_grid={n_grid_max})")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    ax2.plot(x_grid, or_result.candidate - true_u, "-",
             color=COLOR_ORACLE_DENSITY, linewidth=1.5, label="oracle residual")
    ax2.plot(x_grid, est_result.candidate - true_u, "-",
             color=COLOR_ESTIMATED_BEST, linewidth=1.5, label="estimated residual")
    ax2.plot(x_grid, gg_result.candidate - true_u, "-",
             color=COLOR_GRAD_GLOB, linewidth=1.5, label="gradient-glob residual")
    if best_tik_cand is not None:
        ax2.plot(x_grid, best_tik_cand - true_u, ":",
                 color=COLOR_TIKHONOV, linewidth=1.5, label="Tikhonov residual")
    ax2.axhline(0, color="k", linewidth=0.5)
    ax2.set_title(f"Test {test}: residuals (candidate − true u₀)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("residual")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / f"field_comparison_best_estimated_density_{test}.png", dpi=150)
    plt.close(fig)

    # Standalone residual
    fig2, ax3 = plt.subplots(figsize=(8, 4))
    ax3.plot(x_grid, est_result.candidate - true_u, "-",
             color=COLOR_ESTIMATED_BEST,
             label=f"estimated residual (ε={est_eps:.0e}, bw={est_bw}, N={est_np})")
    ax3.axhline(0, color="k", linewidth=0.5)
    ax3.set_title(f"Test {test}: residual — best estimated density (n_grid={n_grid_max})")
    ax3.set_xlabel("x")
    ax3.set_ylabel("candidate − true u₀")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(out_dir / f"residual_best_estimated_density_{test}.png", dpi=150)
    plt.close(fig2)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def make_summary(
    df: pd.DataFrame,
    step_df: pd.DataFrame,
    out_dir: Path,
) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append("DENSITY-PARTICLE ESTIMATED SCORE AUDIT SUMMARY")
    lines.append(f"Output directory: {out_dir}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 72)
    lines.append("")
    lines.append("SCIENTIFIC FRAMING:")
    lines.append("  gradient-glob field-score: GRW-native but representation-mismatched.")
    lines.append("  density-particle oracle:   representation-consistent; score = exact.")
    lines.append("  density-particle estimated: practical; score = u_x / (u + epsilon).")
    lines.append("")
    lines.append("MAIN QUESTION:")
    lines.append("  Can estimated density-particle stay within 2-5x of oracle and")
    lines.append("  remain far below the old gradient-glob plateau?")
    lines.append("")

    go_verdicts: dict[str, str] = {}

    for test in sorted(df["test"].unique()):
        sub = df[df["test"] == test]
        n_grid_max = int(sub["n_grid"].max())
        sub_max = sub[sub["n_grid"] == n_grid_max]
        plateau = GRAD_GLOB_ORACLE_PLATEAUS.get(test, float("nan"))

        lines.append("-" * 60)
        lines.append(f"TEST {test}")
        lines.append("-" * 60)

        # Oracle density best
        dp_or = sub_max[(sub_max["method_type"] == "density_particle_oracle") &
                        np.isfinite(sub_max["relative_l2"])]
        best_or_rl2 = float(dp_or["relative_l2"].min()) if len(dp_or) > 0 else float("nan")

        # Estimated density best
        dp_est = sub_max[(sub_max["method_type"] == "density_particle_estimated") &
                         np.isfinite(sub_max["relative_l2"])]
        best_est_rl2 = float("nan")
        best_est_eps = float("nan")
        best_est_bw = float("nan")
        best_est_np = None
        best_est_step0: float = float("nan")
        best_est_fwd: float = float("nan")
        best_est_score_L2: float = float("nan")
        best_est_mean_score_L2: float = float("nan")
        n_failures = 0

        if len(sub_max[sub_max["method_type"] == "density_particle_estimated"]) > 0:
            n_failures = int((~sub_max[sub_max["method_type"] == "density_particle_estimated"]["completed"]).sum())

        if len(dp_est) > 0:
            idx = dp_est["relative_l2"].idxmin()
            best_est_rl2 = float(dp_est.loc[idx, "relative_l2"])
            best_est_eps = float(dp_est.loc[idx, "epsilon"])
            best_est_bw = float(dp_est.loc[idx, "bandwidth_factor"])
            best_est_np = int(dp_est.loc[idx, "n_particles"])
            best_est_step0 = _safe(dp_est.loc[idx, "step_zero_recon_error"])
            best_est_fwd = _safe(dp_est.loc[idx, "forward_consistency_l2"])
            best_est_score_L2 = _safe(dp_est.loc[idx, "max_score_L2_error_vs_oracle"])
            best_est_mean_score_L2 = _safe(dp_est.loc[idx, "mean_score_L2_error_vs_oracle"])

        # Gradient-glob best
        gg = sub_max[(sub_max["method_type"] == "gradient_glob_oracle") &
                     np.isfinite(sub_max["relative_l2"])]
        best_gg_rl2 = float(gg["relative_l2"].min()) if len(gg) > 0 else float("nan")

        # Tikhonov best
        tik = sub_max[(sub_max["method_type"] == "baseline_tikhonov") &
                      np.isfinite(sub_max["relative_l2"])]
        best_tik_rl2 = float(tik["relative_l2"].min()) if len(tik) > 0 else float("nan")

        # Spectral best
        sc = sub_max[(sub_max["method_type"] == "baseline_spectral") &
                     np.isfinite(sub_max["relative_l2"])]
        best_sc_rl2 = float(sc["relative_l2"].min()) if len(sc) > 0 else float("nan")

        lines.append(f"  density-particle oracle best     : rel_L2 = {best_or_rl2:.6f}")
        lines.append(f"  density-particle estimated best  : rel_L2 = {best_est_rl2:.6f}")
        if best_est_np is not None:
            eps_str = f"{best_est_eps:.0e}" if best_est_eps > 0 else "0 (raw)"
            lines.append(f"    config: N={best_est_np}, bw={best_est_bw}, epsilon={eps_str}")
            lines.append(f"    step-zero recon error          : {best_est_step0:.6f}")
            lines.append(f"    forward consistency L2         : {best_est_fwd:.6e}")
            lines.append(f"    max score L2 error vs oracle   : {best_est_score_L2:.6e}")
            lines.append(f"    mean score L2 error vs oracle  : {best_est_mean_score_L2:.6e}")
        lines.append(f"  gradient-glob oracle plateau (ref): {plateau:.3f}")
        lines.append(f"  gradient-glob oracle best         : rel_L2 = {best_gg_rl2:.6f}")
        lines.append(f"  Tikhonov best                     : rel_L2 = {best_tik_rl2:.6f}")
        lines.append(f"  spectral best                     : rel_L2 = {best_sc_rl2:.6f}")
        lines.append(f"  failures in estimated sweep       : {n_failures}")

        # Ratios
        ratio_est_oracle = float("nan")
        ratio_est_plateau = float("nan")
        ratio_est_tik = float("nan")
        if np.isfinite(best_est_rl2):
            if np.isfinite(best_or_rl2) and best_or_rl2 > 0:
                ratio_est_oracle = best_est_rl2 / best_or_rl2
                lines.append(f"  estimated / oracle ratio          : {ratio_est_oracle:.2f}×")
            if np.isfinite(plateau) and plateau > 0:
                ratio_est_plateau = best_est_rl2 / plateau
                lines.append(f"  estimated / grad-glob plateau     : {ratio_est_plateau:.4f}")
            if np.isfinite(best_tik_rl2) and best_tik_rl2 > 0:
                ratio_est_tik = best_est_rl2 / best_tik_rl2
                lines.append(f"  estimated / Tikhonov best         : {ratio_est_tik:.2f}×")

        # Convergence with n_particles
        if len(dp_est) >= 2:
            by_np = (dp_est.groupby("n_particles")["relative_l2"].min()
                     .reset_index()
                     .sort_values("n_particles"))
            by_np = by_np[np.isfinite(by_np["relative_l2"])]
            if len(by_np) >= 2:
                first = float(by_np.iloc[0]["relative_l2"])
                last = float(by_np.iloc[-1]["relative_l2"])
                conv = "CONVERGING" if last < first * 0.95 else "PLATEAUED"
                lines.append(f"  convergence: first={first:.5f}, last={last:.5f} → {conv}")

        # Bottleneck diagnosis
        if best_est_np is not None and np.isfinite(best_est_step0) and np.isfinite(best_est_rl2):
            if best_est_step0 > 0.8 * best_est_rl2:
                diag = ("RECONSTRUCTION BOTTLENECK — step-zero error ≈ final inverse error.\n"
                        "    KDE bandwidth / n_particles controls accuracy; not backward dynamics.")
            elif best_est_step0 > 0.5 * best_est_rl2:
                diag = ("PARTIAL RECONSTRUCTION BOTTLENECK — step-zero error significant.\n"
                        "    Both reconstruction and score estimation contribute to error.")
            elif (np.isfinite(best_est_score_L2) and
                  best_est_score_L2 > best_est_rl2):
                diag = ("SCORE ESTIMATION BOTTLENECK — max score L2 error > final inverse error.\n"
                        "    Score estimation diverges during integration; increase epsilon or N.")
            else:
                diag = ("BACKWARD DYNAMICS — step-zero small, score reasonable.\n"
                        "    Remaining error comes from time discretization or ill-posedness.")
            lines.append(f"  BOTTLENECK: {diag}")

        # Epsilon diagnosis
        if len(dp_est) > 0 and np.isfinite(best_est_eps):
            eps_str = f"{best_est_eps:.0e}" if best_est_eps > 0 else "0 (raw)"
            lines.append(f"  best epsilon                      : {eps_str}")
            if best_est_eps == 0.0:
                lines.append("    => raw score (ε=0) is sufficient for clean data.")
            else:
                lines.append(f"    => epsilon stabilization needed; use ε={best_est_eps:.0e}.")

        # Bandwidth diagnosis
        if np.isfinite(best_est_bw):
            lines.append(f"  best bandwidth_factor             : {best_est_bw}")

        # GO / STOP verdict
        if np.isfinite(best_est_rl2) and np.isfinite(best_or_rl2) and np.isfinite(plateau):
            if best_est_rl2 < plateau * 0.3 and (
                    not np.isfinite(ratio_est_oracle) or ratio_est_oracle < 5.0):
                verdict = "STRONG GO"
            elif best_est_rl2 < plateau * 0.5:
                pct = int((1.0 - best_est_rl2 / plateau) * 100)
                verdict = f"GO: estimated beats old plateau by >{pct}%"
            elif best_est_rl2 < plateau * 0.85:
                verdict = "CONDITIONAL GO: estimated beats old plateau, not strongly"
            elif best_est_rl2 < plateau * 1.05:
                verdict = "STOP/PIVOT: estimated ≈ old gradient-glob plateau"
            else:
                verdict = "STOP/PIVOT: estimated WORSE than old gradient-glob plateau"
        else:
            verdict = "INCONCLUSIVE (no finite result)"

        go_verdicts[test] = verdict
        lines.append(f"  VERDICT: {verdict}")
        lines.append("")

    # -----------------------------------------------------------------------
    # Overall recommendation
    # -----------------------------------------------------------------------
    lines.append("=" * 72)
    lines.append("OVERALL VERDICT")
    lines.append("=" * 72)
    for test, v in go_verdicts.items():
        lines.append(f"  Test {test}: {v}")
    lines.append("")

    n_strong = sum(1 for v in go_verdicts.values() if "STRONG GO" in v)
    n_go = sum(1 for v in go_verdicts.values() if v.startswith("GO:"))
    n_cond = sum(1 for v in go_verdicts.values() if "CONDITIONAL GO" in v)
    n_stop = sum(1 for v in go_verdicts.values() if "STOP" in v)

    if n_strong >= 2 or (n_strong + n_go) >= 2:
        lines.append("RECOMMENDATION: STRONG GO")
        lines.append("Estimated density-particle score is practical.")
        lines.append("")
        lines.append("Next steps:")
        lines.append("  1. Run noise-robustness study (add observation noise to u_obs).")
        lines.append("  2. Test scale_epsilon_by_peak=True for automatic stabilization.")
        lines.append("  3. Characterize how epsilon/bandwidth depend on noise level.")
        lines.append("  4. Consider variable-diffusivity forward model.")
    elif n_cond >= 2 or (n_go + n_cond) >= 2:
        lines.append("RECOMMENDATION: CONDITIONAL GO")
        lines.append("Estimated score beats old plateau but improvements remain limited.")
        lines.append("")
        lines.append("Next steps:")
        lines.append("  1. Increase n_particles to reduce reconstruction bottleneck.")
        lines.append("  2. Investigate adaptive KDE bandwidth (Silverman/Scott auto-select).")
        lines.append("  3. Check score_L2_error_vs_oracle: if large, score estimation is")
        lines.append("     the bottleneck — consider iterative density refinement.")
    else:
        lines.append("RECOMMENDATION: STOP/PIVOT")
        lines.append("Estimated density-particle score does not improve over old plateau.")
        lines.append("Score estimation error is too large for this problem scale.")
        lines.append("")
        lines.append("Next steps:")
        lines.append("  1. Try much larger n_particles (50000+) to test if KDE improves.")
        lines.append("  2. Try iterative density refinement (multiple passes).")
        lines.append("  3. Accept negative-result framing: score estimation is fundamentally")
        lines.append("     limited by reconstruction quality at moderate n_particles.")
        lines.append("  4. Pivot to: noise-robustness study or variable-diffusivity.")

    # Practical epsilon/bandwidth recommendation across all tests
    lines.append("")
    lines.append("PRACTICAL RECOMMENDATIONS:")
    best_eps_by_test: dict[str, float] = {}
    best_bw_by_test: dict[str, float] = {}
    for test in sorted(df["test"].unique()):
        sub = df[(df["test"] == test) &
                 (df["method_type"] == "density_particle_estimated") &
                 np.isfinite(df["relative_l2"])]
        n_grid_max_t = int(sub["n_grid"].max()) if len(sub) > 0 else -1
        sub_max = sub[sub["n_grid"] == n_grid_max_t] if n_grid_max_t > 0 else sub
        if len(sub_max) == 0:
            continue
        idx = sub_max["relative_l2"].idxmin()
        best_eps_by_test[test] = float(sub_max.loc[idx, "epsilon"])
        best_bw_by_test[test] = float(sub_max.loc[idx, "bandwidth_factor"])

    if best_eps_by_test:
        eps_vals = list(best_eps_by_test.values())
        lines.append(f"  Best epsilon per test: {best_eps_by_test}")
        if all(e == 0.0 for e in eps_vals):
            lines.append("  => Raw score (epsilon=0) is universally best on clean data.")
            lines.append("     Add epsilon only when observation noise is present.")
        else:
            nonzero = [e for e in eps_vals if e > 0]
            lines.append(f"  => Use epsilon={min(nonzero):.0e} as conservative default.")

    if best_bw_by_test:
        bw_vals_rec = list(best_bw_by_test.values())
        lines.append(f"  Best bandwidth_factor per test: {best_bw_by_test}")
        lines.append(f"  => Use bandwidth_factor={min(bw_vals_rec):.1f} as conservative default.")

    lines.append("")
    lines.append("NOTE: density-particle estimated method is NOT 'GRW' in the gradient-glob")
    lines.append("  sense.  It is a score-consistent particle inverse: particles carry u,")
    lines.append("  score is u_x/(u+eps) estimated from the reconstructed density.")
    lines.append("  Superiority over Tikhonov/spectral is NOT claimed or expected.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Density-particle estimated score audit")
    parser.add_argument("--base-config", required=True,
                        help="Path to base Gaussian YAML config")
    parser.add_argument("--mixture-config", required=True,
                        help="Path to Gaussian mixture YAML config")
    parser.add_argument("--n-grid", nargs="+", type=int, default=None,
                        help="Grid sizes to sweep (overrides defaults)")
    parser.add_argument("--n-particles", nargs="+", type=int, default=None,
                        help="Particle counts to sweep (overrides defaults)")
    parser.add_argument("--epsilon", nargs="+", type=float, default=None,
                        help="Epsilon values to sweep (overrides defaults)")
    parser.add_argument("--bandwidth-factor", nargs="+", type=float, default=None,
                        help="Bandwidth factors to sweep (overrides defaults)")
    parser.add_argument("--full", action="store_true",
                        help="Use full production sweep parameters")
    args = parser.parse_args()

    if args.full:
        n_grid_vals = args.n_grid or N_GRID_FULL
        n_part_vals = args.n_particles or N_PARTICLES_FULL
        bw_vals = args.bandwidth_factor or BANDWIDTH_FACTORS_FULL
        eps_vals = args.epsilon or EPSILON_FULL
    else:
        n_grid_vals = args.n_grid or N_GRID_DEFAULT
        n_part_vals = args.n_particles or N_PARTICLES_DEFAULT
        bw_vals = args.bandwidth_factor or BANDWIDTH_FACTORS_DEFAULT
        eps_vals = args.epsilon or EPSILON_DEFAULT

    base_cfg = load_config(args.base_config)
    mixture_cfg = load_config(args.mixture_config)
    test_cfgs = build_test_configs(base_cfg, mixture_cfg)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs") / f"density_estimated_audit_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    all_rows: list[dict] = []
    all_step_rows: list[dict] = []

    # Cell count estimate
    n_or = len(test_cfgs) * len(n_grid_vals) * len(n_part_vals) * len(bw_vals)
    n_est = len(test_cfgs) * len(n_grid_vals) * len(n_part_vals) * len(bw_vals) * len(eps_vals)
    n_gg = len(test_cfgs) * len(n_grid_vals) * len(GLOB_VALUES)
    n_bl = len(test_cfgs) * len(n_grid_vals)
    total = n_or + n_est + n_gg + n_bl

    print(f"\nRunning density estimated audit ({total} cells: "
          f"{n_or} oracle, {n_est} estimated, {n_gg} grad-glob, {n_bl} baselines)...")
    print(f"n_grid={n_grid_vals}  n_particles={n_part_vals}  "
          f"bw_factors={bw_vals}  epsilon={eps_vals}")

    t0_total = time.perf_counter()

    for test_name, test_cfg in test_cfgs.items():
        print(f"\n  Test {test_name}")

        for n_grid in n_grid_vals:
            print(f"    n_grid={n_grid}", end="", flush=True)

            # Oracle density cells
            for n_part in n_part_vals:
                for bw in bw_vals:
                    row, step_rows = run_density_oracle_cell(
                        test_name, test_cfg, n_grid, n_part, bw)
                    all_rows.append(row)
                    all_step_rows.extend(step_rows)
                    print("o", end="", flush=True)

            # Estimated density cells
            for n_part in n_part_vals:
                for bw in bw_vals:
                    for eps in eps_vals:
                        row, step_rows = run_density_estimated_cell(
                            test_name, test_cfg, n_grid, n_part, bw, eps)
                        all_rows.append(row)
                        all_step_rows.extend(step_rows)
                        print("e", end="", flush=True)

            # Gradient-glob oracle
            for globs in GLOB_VALUES:
                row = run_gradient_glob_cell(test_name, test_cfg, n_grid, globs)
                all_rows.append(row)
                print("g", end="", flush=True)

            # Classical baselines
            baseline_rows = run_baseline_cells(test_name, test_cfg, n_grid)
            all_rows.extend(baseline_rows)
            print("b", end="", flush=True)

        print()

    elapsed = time.perf_counter() - t0_total
    print(f"\nAll cells complete in {elapsed:.1f}s")

    # Save metrics CSV
    df = pd.DataFrame(all_rows)
    metrics_csv = out_dir / "density_estimated_audit_metrics.csv"
    df.to_csv(metrics_csv, index=False)
    print(f"Metrics CSV: {metrics_csv}")

    # Save step diagnostics CSV
    if all_step_rows:
        step_df = pd.DataFrame(all_step_rows)
        step_csv = out_dir / "density_estimated_audit_step_diagnostics.csv"
        step_df.to_csv(step_csv, index=False)
        print(f"Step diagnostics CSV: {step_csv}")
    else:
        step_df = pd.DataFrame()

    # Best-by-test CSV
    best_rows = []
    for test in df["test"].unique():
        sub = df[df["test"] == test]
        for mtype in ["density_particle_oracle", "density_particle_estimated",
                      "gradient_glob_oracle", "baseline_tikhonov", "baseline_spectral"]:
            sub2 = sub[(sub["method_type"] == mtype) & np.isfinite(sub["relative_l2"])]
            if len(sub2) == 0:
                continue
            idx = sub2["relative_l2"].idxmin()
            best_rows.append(sub2.loc[idx].to_dict())
    best_df = pd.DataFrame(best_rows)
    best_csv = out_dir / "best_by_test.csv"
    best_df.to_csv(best_csv, index=False)
    print(f"Best-by-test CSV: {best_csv}")

    # Plots
    print("\nGenerating plots...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plot_oracle_vs_estimated(df, out_dir)
        plot_estimated_vs_epsilon(df, out_dir)
        plot_estimated_vs_bandwidth(df, out_dir)
        plot_score_diagnostics_vs_step(df, step_df, out_dir)
        plot_step_zero_error(df, out_dir)
        for test in sorted(df["test"].unique()):
            plot_field_comparison(df, test, test_cfgs, out_dir)
    print("Plots saved.")

    # Summary
    summary_text = make_summary(df, step_df, out_dir)
    summary_path = out_dir / "density_estimated_audit_summary.txt"
    summary_path.write_text(summary_text)
    print(f"\nSummary: {summary_path}")
    print("\n" + summary_text)


if __name__ == "__main__":
    main()
