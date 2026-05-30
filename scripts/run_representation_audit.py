"""
run_representation_audit.py — Representation consistency audit.

Purpose:
    Test whether the gradient-glob oracle plateau (~0.15–0.24 relative L2) is
    caused by a representation mismatch, or by irreducible reconstruction /
    inverse ill-posedness.

    The current score-guided GRW method (Formulation A) moves globs carrying
    u_x with the field-density score s_u = u_x/u.  This is mathematically
    inconsistent: the continuity equation for u_x-particles evolving under
    s_u gives  q_tau = -alpha * partial_x(q^2/u),  NOT the gradient backward
    heat equation  q_tau = -alpha * q_xx.

    This audit tests Formulation B: density-particle oracle.
        - Particles carry u itself (density field).
        - Oracle score s = partial_x log u is the CORRECT drift for u-particles.
        - Reverse-time continuity gives u_tau = -alpha * u_xx (exact).
        - If the oracle plateau is representation-driven, B should break below
          the 0.15–0.24 ceiling as n_particles increases.

Methods compared:
    A. gradient_glob_oracle     : existing run_oracle_score_deterministic
    B. density_particle_oracle  : new run_density_particle_oracle_score_deterministic
    REF (Tikhonov)              : best lambda from sweep
    REF (spectral)              : best noise_delta from sweep

Tests:
    B : Gaussian, sigma0=0.08, T=0.15
    H : Gaussian mixture, T=0.15
    Z : near-zero tail stress, T=0.05

Sweeps:
    n_grid            : [200, 400, 800]
    n_particles       : [200, 500, 1000, 2000, 5000]
    recon_method      : ["histogram", "kde"]
    bandwidth_factor  : [1.0, 2.0, 4.0]   (KDE only; dx-relative)
    globs_per_jump    : [20, 80]           (gradient-glob baseline)

Outputs:
    outputs/representation_audit_TIMESTAMP/
        representation_audit_metrics.csv
        representation_audit_summary.txt
        best_by_test.csv
        oracle_A_vs_B_rel_L2_by_resolution.png
        density_particle_rel_L2_vs_n_particles_<test>.png
        density_particle_forward_consistency_vs_n_particles_<test>.png
        step_zero_reconstruction_error_vs_n_particles_<test>.png
        field_comparison_best_density_oracle_<test>.png
        residual_best_density_oracle_<test>.png
        density_particle_bandwidth_sensitivity_<test>.png

Usage:
    PYTHONPATH=src python scripts/run_representation_audit.py \\
        --base-config configs/gaussian_base.yaml \\
        --mixture-config configs/gaussian_mixture.yaml
"""

from __future__ import annotations

import sys
import argparse
import copy
import math
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

from invheat_grw.config import (
    load_config, Config,
    RegularizationConfig, EpsilonFloorConfig, ScoreClippingConfig, SmoothingConfig,
)
from invheat_grw.fields import make_grid, true_u0 as compute_true_u0, observed_final as compute_observed_final
from invheat_grw.methods import (
    run_oracle_score_deterministic,
    run_density_particle_oracle_score_deterministic,
    MethodResult,
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

N_GRID_VALUES = [200, 400, 800]
N_PARTICLES_VALUES = [200, 500, 1000, 2000, 5000]
RECON_METHODS = ["histogram", "kde"]
KDE_BANDWIDTH_FACTORS = [1.0, 2.0, 4.0]
GLOB_VALUES = [20, 80]
TIKHONOV_LAMBDAS = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
NOISE_DELTA_VALUES = [1e-8, 1e-6, 1e-4, 1e-2]
RNG_SEED = 42

# Known gradient-glob oracle plateaus from convergence study (used for GO verdict)
GRAD_GLOB_ORACLE_PLATEAUS = {"B": 0.175, "H": 0.239, "Z": 0.155}


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
    tests = {}
    tests["B"] = patch_config(base_cfg, **{"heat.T": 0.15, "initial_condition.sigma0": 0.08})
    tests["H"] = patch_config(mixture_cfg, **{"heat.T": 0.15})
    tests["Z"] = patch_config(base_cfg, **{
        "heat.T": 0.05,
        "initial_condition.sigma0": 0.05,
        "initial_condition.mu": 0.4,
    })
    return tests


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
    }


def run_density_particle_cell(
    test: str,
    cfg: Config,
    n_grid: int,
    n_particles: int,
    recon_method: str,
    bandwidth_factor: float,
) -> dict:
    """Run one density-particle oracle cell and return a metrics row."""
    cfg_res = patch_config(cfg, **{"domain.n_grid": n_grid})
    x_grid = make_grid(cfg_res)
    true_u = compute_true_u0(x_grid, cfg_res)
    u_obs = compute_observed_final(x_grid, cfg_res)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = run_density_particle_oracle_score_deterministic(
            u_obs, x_grid, cfg_res, n_particles,
            recon_method=recon_method,
            bandwidth_factor=bandwidth_factor,
        )
        m = compute_metrics(result, true_u, u_obs, x_grid, cfg_res)
        wass = compute_wasserstein(result.candidate, true_u, x_grid)

    bw_label = bandwidth_factor if recon_method == "kde" else float("nan")
    bw_actual = float(result.bandwidth_used) if recon_method == "kde" else float("nan")

    row = _base_row(test, n_grid, result.method_name, "density_particle_oracle")
    row.update({
        "n_particles": n_particles,
        "recon_method": recon_method,
        "bandwidth_factor": bw_label,
        "bandwidth_used": bw_actual,
        "completed": result.completed,
        "failure_step": result.failure_step if result.failure_step is not None else "",
        "failure_msg": result.failure_msg or "",
        "step_zero_recon_error": float(result.step_zero_recon_error),
        "relative_l2": m.relative_l2,
        "l2_error": m.l2_error,
        "linf_error": m.linf_error,
        "forward_consistency_l2": m.forward_consistency_l2,
        "wasserstein": wass,
        "mass_rel_error": m.mass_rel_error,
        "total_variation": m.total_variation,
        "peak_value": m.peak_value,
        "peak_ratio": m.peak_ratio,
        "A_fit": m.A_fit,
        "mu_fit": m.mu_fit,
        "sigma_fit": m.sigma_fit,
        "fit_success": m.fit_success,
        "fit_rmse": m.fit_rmse,
        "mass_candidate": m.mass_candidate,
        "mass_true": m.mass_true,
        "runtime_seconds": result.runtime_seconds,
    })
    return row


def run_gradient_glob_cell(
    test: str,
    cfg: Config,
    n_grid: int,
    globs_per_jump: int,
) -> dict:
    """Run gradient-glob oracle at one resolution cell."""
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
    row.update({
        "globs_per_jump": globs_per_jump,
        "completed": result.completed,
        "failure_step": result.failure_step if result.failure_step is not None else "",
        "failure_msg": result.failure_msg or "",
        "relative_l2": m.relative_l2,
        "l2_error": m.l2_error,
        "linf_error": m.linf_error,
        "forward_consistency_l2": m.forward_consistency_l2,
        "wasserstein": wass,
        "mass_rel_error": m.mass_rel_error,
        "total_variation": m.total_variation,
        "peak_value": m.peak_value,
        "peak_ratio": m.peak_ratio,
        "A_fit": m.A_fit,
        "mu_fit": m.mu_fit,
        "sigma_fit": m.sigma_fit,
        "fit_success": m.fit_success,
        "fit_rmse": m.fit_rmse,
        "mass_candidate": m.mass_candidate,
        "mass_true": m.mass_true,
        "runtime_seconds": result.runtime_seconds,
    })
    return row


def run_baseline_cells(
    test: str,
    cfg: Config,
    n_grid: int,
) -> list[dict]:
    """Run Tikhonov (best-lambda) and spectral (best-delta) at one resolution."""
    cfg_res = patch_config(cfg, **{"domain.n_grid": n_grid})
    x_grid = make_grid(cfg_res)
    true_u = compute_true_u0(x_grid, cfg_res)
    u_obs = compute_observed_final(x_grid, cfg_res)
    alpha = cfg_res.heat.alpha
    T_val = cfg_res.heat.T
    rows = []

    # Tikhonov: sweep lambdas, keep best
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
            "relative_l2": m.relative_l2,
            "l2_error": m.l2_error,
            "linf_error": m.linf_error,
            "forward_consistency_l2": m.forward_consistency_l2,
            "wasserstein": wass,
            "mass_rel_error": m.mass_rel_error,
            "total_variation": m.total_variation,
            "peak_value": m.peak_value,
            "peak_ratio": m.peak_ratio,
            "A_fit": m.A_fit, "mu_fit": m.mu_fit, "sigma_fit": m.sigma_fit,
            "fit_success": m.fit_success, "fit_rmse": m.fit_rmse,
            "mass_candidate": m.mass_candidate, "mass_true": m.mass_true,
            "bandwidth_factor": best_tik_lam,  # repurpose column to store lambda
        })
        rows.append(row)

    # Spectral: sweep noise_delta, keep best
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
            "relative_l2": m.relative_l2,
            "l2_error": m.l2_error,
            "linf_error": m.linf_error,
            "forward_consistency_l2": m.forward_consistency_l2,
            "wasserstein": wass,
            "mass_rel_error": m.mass_rel_error,
            "total_variation": m.total_variation,
            "peak_value": m.peak_value,
            "peak_ratio": m.peak_ratio,
            "A_fit": m.A_fit, "mu_fit": m.mu_fit, "sigma_fit": m.sigma_fit,
            "fit_success": m.fit_success, "fit_rmse": m.fit_rmse,
            "mass_candidate": m.mass_candidate, "mass_true": m.mass_true,
            "bandwidth_factor": best_sc_nd,
        })
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

COLORS_DP = {
    ("histogram", float("nan")): "#1f77b4",
    ("kde", 1.0): "#ff7f0e",
    ("kde", 2.0): "#2ca02c",
    ("kde", 4.0): "#d62728",
}
COLOR_GRAD_GLOB = "#9467bd"
COLOR_TIKHONOV = "#8c564b"
COLOR_SPECTRAL = "#e377c2"
COLOR_PLATEAU = "#aaaaaa"


def _color_for(recon_method: str, bw_factor) -> str:
    key = (recon_method, float("nan") if recon_method == "histogram" else float(bw_factor))
    return COLORS_DP.get(key, "#17becf")


def _label_for(recon_method: str, bw_factor) -> str:
    if recon_method == "histogram":
        return "density-particle (histogram)"
    else:
        return f"density-particle (KDE bw={bw_factor}×dx)"


def plot_oracle_A_vs_B(df: pd.DataFrame, out_dir: Path) -> None:
    """
    One figure with 3 subplots (one per test).
    X-axis: n_particles (for B) or n_grid (for A).
    Y-axis: relative_l2.
    Shows: best density-particle oracle curve, gradient-glob oracle (globs=80),
    Tikhonov best, spectral best, and the gradient-glob plateau line.
    """
    tests = ["B", "H", "Z"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, test in zip(axes, tests):
        sub = df[df["test"] == test]

        # Gradient-glob oracle at globs=80 across n_grid
        gg = sub[(sub["method_type"] == "gradient_glob_oracle") &
                 (sub["globs_per_jump"] == 80)].sort_values("n_grid")
        if len(gg) > 0:
            ax.plot(gg["n_grid"], gg["relative_l2"], "o-",
                    color=COLOR_GRAD_GLOB, label="gradient-glob oracle (g=80)", zorder=3)

        # Density-particle oracle: best (recon_method, bw_factor) combo per n_particles
        dp = sub[sub["method_type"] == "density_particle_oracle"].copy()
        if len(dp) > 0:
            # Group by n_particles, pick best relative_l2 across all recon configs
            best_dp = dp.groupby("n_particles")["relative_l2"].min().reset_index()
            best_dp = best_dp.sort_values("n_particles")
            ax.plot(best_dp["n_particles"], best_dp["relative_l2"], "s--",
                    color="#1f77b4", label="density-particle oracle (best recon)", zorder=3)

        # Reference lines: Tikhonov and spectral (largest n_grid available)
        tik = sub[(sub["method_type"] == "baseline_tikhonov") &
                  (sub["n_grid"] == sub["n_grid"].max())]
        if len(tik) > 0:
            rl2_tik = float(tik["relative_l2"].values[0])
            ax.axhline(rl2_tik, color=COLOR_TIKHONOV, linestyle=":", linewidth=1.5,
                       label=f"Tikhonov best ({rl2_tik:.4f})")

        sc = sub[(sub["method_type"] == "baseline_spectral") &
                 (sub["n_grid"] == sub["n_grid"].max())]
        if len(sc) > 0:
            rl2_sc = float(sc["relative_l2"].values[0])
            ax.axhline(rl2_sc, color=COLOR_SPECTRAL, linestyle=":", linewidth=1.5,
                       label=f"Spectral best ({rl2_sc:.4f})")

        # Gradient-glob plateau line
        plateau = GRAD_GLOB_ORACLE_PLATEAUS.get(test, float("nan"))
        if np.isfinite(plateau):
            ax.axhline(plateau, color=COLOR_PLATEAU, linestyle="--", linewidth=1.0,
                       label=f"grad-glob plateau ({plateau:.3f})", alpha=0.7)

        ax.set_title(f"Test {test}")
        ax.set_xlabel("n_particles (density-B) / n_grid (glob-A)")
        ax.set_ylabel("relative L2")
        ax.set_yscale("log")
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Representation Audit: Oracle A (gradient-glob) vs Oracle B (density-particle)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "oracle_A_vs_B_rel_L2_by_resolution.png", dpi=150)
    plt.close(fig)


def plot_density_particle_vs_n_particles(df: pd.DataFrame, metric: str, ylabel: str,
                                          file_prefix: str, out_dir: Path) -> None:
    """One figure per test: density-particle oracle metric vs n_particles,
    split by recon_method and bandwidth_factor."""
    dp = df[df["method_type"] == "density_particle_oracle"].copy()
    tests = df["test"].unique()

    for test in sorted(tests):
        sub = dp[dp["test"] == test]
        fig, ax = plt.subplots(figsize=(7, 5))

        for recon in RECON_METHODS:
            if recon == "histogram":
                bw_vals = [float("nan")]
            else:
                bw_vals = KDE_BANDWIDTH_FACTORS

            for bw in bw_vals:
                if recon == "histogram":
                    mask = (sub["recon_method"] == "histogram")
                else:
                    mask = (sub["recon_method"] == "kde") & (sub["bandwidth_factor"].round(4) == round(bw, 4))
                rows = sub[mask].sort_values("n_particles")
                if len(rows) == 0:
                    continue
                valid = rows[np.isfinite(rows[metric])]
                if len(valid) == 0:
                    continue
                # Use largest n_grid available for clearest signal
                n_grid_max = valid["n_grid"].max()
                curve = valid[valid["n_grid"] == n_grid_max]
                color = _color_for(recon, bw)
                label = _label_for(recon, bw) + f" (n_grid={n_grid_max})"
                ax.plot(curve["n_particles"], curve[metric], "o-",
                        color=color, label=label)

        # Gradient-glob plateau reference
        plateau = GRAD_GLOB_ORACLE_PLATEAUS.get(test, float("nan"))
        if np.isfinite(plateau) and metric == "relative_l2":
            ax.axhline(plateau, color=COLOR_PLATEAU, linestyle="--", linewidth=1.0,
                       label=f"grad-glob oracle plateau ({plateau:.3f})", alpha=0.7)

        # Tikhonov reference for relative_l2
        tik = df[(df["test"] == test) & (df["method_type"] == "baseline_tikhonov")]
        if len(tik) > 0 and metric == "relative_l2":
            rl2_tik = float(tik.sort_values("n_grid").iloc[-1]["relative_l2"])
            ax.axhline(rl2_tik, color=COLOR_TIKHONOV, linestyle=":", linewidth=1.5,
                       label=f"Tikhonov best ({rl2_tik:.4f})", alpha=0.8)

        ax.set_xlabel("n_particles")
        ax.set_ylabel(ylabel)
        if metric in ("relative_l2", "forward_consistency_l2", "step_zero_recon_error"):
            ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_title(f"Test {test}: {ylabel} vs n_particles")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"{file_prefix}_{test}.png", dpi=150)
        plt.close(fig)


def plot_bandwidth_sensitivity(df: pd.DataFrame, out_dir: Path) -> None:
    """Per test, per n_particles: relative_l2 vs bandwidth_factor for KDE."""
    dp = df[(df["method_type"] == "density_particle_oracle") &
            (df["recon_method"] == "kde")].copy()
    tests = df["test"].unique()

    for test in sorted(tests):
        sub = dp[dp["test"] == test]
        n_grid_max = sub["n_grid"].max() if len(sub) > 0 else None
        if n_grid_max is None:
            continue
        sub = sub[sub["n_grid"] == n_grid_max]

        fig, ax = plt.subplots(figsize=(7, 5))
        for np_val in sorted(sub["n_particles"].unique()):
            rows = sub[sub["n_particles"] == np_val].sort_values("bandwidth_factor")
            valid = rows[np.isfinite(rows["relative_l2"])]
            if len(valid) < 2:
                continue
            ax.plot(valid["bandwidth_factor"], valid["relative_l2"], "o-",
                    label=f"n_particles={np_val}")

        ax.set_xlabel("bandwidth_factor (× dx)")
        ax.set_ylabel("relative L2")
        ax.set_yscale("log")
        ax.set_title(f"Test {test}: KDE bandwidth sensitivity (n_grid={n_grid_max})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"density_particle_bandwidth_sensitivity_{test}.png", dpi=150)
        plt.close(fig)


def plot_field_comparison(
    df: pd.DataFrame,
    test: str,
    cfg: Config,
    best_dp_row: dict,
    best_gg_row: dict,
    out_dir: Path,
) -> None:
    """
    Overlay: true u0, observed uT, best density-particle oracle candidate,
    gradient-glob oracle candidate, Tikhonov best candidate.
    """
    n_grid = int(best_dp_row["n_grid"])
    cfg_res = patch_config(cfg, **{"domain.n_grid": n_grid})
    x_grid = make_grid(cfg_res)
    true_u = compute_true_u0(x_grid, cfg_res)
    u_obs = compute_observed_final(x_grid, cfg_res)

    # Recompute best density-particle candidate
    recon_method = str(best_dp_row["recon_method"])
    bw_factor = float(best_dp_row["bandwidth_factor"]) if recon_method == "kde" else 2.0
    n_particles = int(best_dp_row["n_particles"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dp_result = run_density_particle_oracle_score_deterministic(
            u_obs, x_grid, cfg_res, n_particles,
            recon_method=recon_method, bandwidth_factor=bw_factor)

    # Recompute gradient-glob oracle candidate
    globs = int(best_gg_row["globs_per_jump"])
    cfg_gg = patch_config(cfg_res, **{"grw.gradient_globs_per_jump": globs})
    rng = np.random.default_rng(RNG_SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        gg_result = run_oracle_score_deterministic(u_obs, x_grid, cfg_gg, rng)

    # Tikhonov best
    alpha = cfg_res.heat.alpha
    T_val = cfg_res.heat.T
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

    # Field comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(x_grid, true_u, "k-", linewidth=2.0, label="true u₀", zorder=5)
    ax.plot(x_grid, u_obs, "k--", linewidth=1.0, label=f"observed u(T={T_val})", alpha=0.6)
    ax.plot(x_grid, dp_result.candidate, "-",
            color="#1f77b4", linewidth=1.5,
            label=f"density-particle oracle ({recon_method}, N={n_particles})")
    ax.plot(x_grid, gg_result.candidate, "-",
            color=COLOR_GRAD_GLOB, linewidth=1.5,
            label=f"gradient-glob oracle (g={globs})")
    if best_tik_cand is not None:
        ax.plot(x_grid, best_tik_cand, ":",
                color=COLOR_TIKHONOV, linewidth=1.5,
                label=f"Tikhonov best (rel_L2={best_tik_rl2:.4f})")
    ax.set_title(f"Test {test}: field comparison (n_grid={n_grid})")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Residual plot
    ax2 = axes[1]
    ax2.plot(x_grid, dp_result.candidate - true_u, "-",
             color="#1f77b4", linewidth=1.5,
             label="density-particle oracle residual")
    ax2.plot(x_grid, gg_result.candidate - true_u, "-",
             color=COLOR_GRAD_GLOB, linewidth=1.5,
             label="gradient-glob oracle residual")
    if best_tik_cand is not None:
        ax2.plot(x_grid, best_tik_cand - true_u, ":",
                 color=COLOR_TIKHONOV, linewidth=1.5,
                 label="Tikhonov residual")
    ax2.axhline(0, color="k", linewidth=0.5)
    ax2.set_title(f"Test {test}: residuals (candidate − true u₀)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("residual")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / f"field_comparison_best_density_oracle_{test}.png", dpi=150)
    plt.close(fig)


def plot_residual_only(
    df: pd.DataFrame,
    test: str,
    cfg: Config,
    best_dp_row: dict,
    out_dir: Path,
) -> None:
    """Standalone residual plot for the best density-particle oracle."""
    n_grid = int(best_dp_row["n_grid"])
    cfg_res = patch_config(cfg, **{"domain.n_grid": n_grid})
    x_grid = make_grid(cfg_res)
    true_u = compute_true_u0(x_grid, cfg_res)
    u_obs = compute_observed_final(x_grid, cfg_res)

    recon_method = str(best_dp_row["recon_method"])
    bw_factor = float(best_dp_row["bandwidth_factor"]) if recon_method == "kde" else 2.0
    n_particles = int(best_dp_row["n_particles"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dp_result = run_density_particle_oracle_score_deterministic(
            u_obs, x_grid, cfg_res, n_particles,
            recon_method=recon_method, bandwidth_factor=bw_factor)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x_grid, dp_result.candidate - true_u, "-", color="#1f77b4",
            label=f"density-particle oracle residual ({recon_method}, N={n_particles})")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_title(f"Test {test}: residual — best density-particle oracle (n_grid={n_grid})")
    ax.set_xlabel("x")
    ax.set_ylabel("candidate − true u₀")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"residual_best_density_oracle_{test}.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary text generation
# ---------------------------------------------------------------------------

def _finite(v) -> float:
    try:
        f = float(v)
        return f if np.isfinite(f) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def make_summary(
    df: pd.DataFrame,
    out_dir: Path,
    best_by_test: pd.DataFrame,
) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append("REPRESENTATION AUDIT SUMMARY")
    lines.append(f"Output directory: {out_dir}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 72)
    lines.append("")
    lines.append("GOAL: Determine if gradient-glob oracle plateau is due to")
    lines.append("representation mismatch (carrying u_x with field score s_u).")
    lines.append("")
    lines.append("Known gradient-glob oracle plateaus (from convergence study):")
    for t, p in GRAD_GLOB_ORACLE_PLATEAUS.items():
        lines.append(f"  Test {t}: {p:.3f}")
    lines.append("")

    tests = sorted(df["test"].unique())
    go_conditions = {}

    for test in tests:
        sub = df[df["test"] == test]
        lines.append("-" * 60)
        lines.append(f"TEST {test}")
        lines.append("-" * 60)

        plateau = GRAD_GLOB_ORACLE_PLATEAUS.get(test, float("nan"))

        # Best density-particle oracle
        dp = sub[sub["method_type"] == "density_particle_oracle"]
        best_dp_rl2 = float("nan")
        best_dp_n_part = None
        best_dp_recon = ""
        best_dp_bw = float("nan")
        best_dp_n_grid = None
        if len(dp) > 0:
            valid_dp = dp[np.isfinite(dp["relative_l2"])]
            if len(valid_dp) > 0:
                idx = valid_dp["relative_l2"].idxmin()
                best_dp_rl2 = float(valid_dp.loc[idx, "relative_l2"])
                best_dp_n_part = int(valid_dp.loc[idx, "n_particles"])
                best_dp_recon = str(valid_dp.loc[idx, "recon_method"])
                best_dp_bw = _finite(valid_dp.loc[idx, "bandwidth_factor"])
                best_dp_n_grid = int(valid_dp.loc[idx, "n_grid"])
                best_dp_step0 = _finite(valid_dp.loc[idx, "step_zero_recon_error"])
                best_dp_fwd = _finite(valid_dp.loc[idx, "forward_consistency_l2"])

        # Best gradient-glob oracle
        gg = sub[sub["method_type"] == "gradient_glob_oracle"]
        best_gg_rl2 = float("nan")
        if len(gg) > 0:
            valid_gg = gg[np.isfinite(gg["relative_l2"])]
            if len(valid_gg) > 0:
                best_gg_rl2 = float(valid_gg["relative_l2"].min())

        # Best Tikhonov
        tik = sub[sub["method_type"] == "baseline_tikhonov"]
        best_tik_rl2 = float("nan")
        if len(tik) > 0:
            valid_tik = tik[np.isfinite(tik["relative_l2"])]
            if len(valid_tik) > 0:
                best_tik_rl2 = float(valid_tik["relative_l2"].min())

        # Best spectral
        sc = sub[sub["method_type"] == "baseline_spectral"]
        best_sc_rl2 = float("nan")
        if len(sc) > 0:
            valid_sc = sc[np.isfinite(sc["relative_l2"])]
            if len(valid_sc) > 0:
                best_sc_rl2 = float(valid_sc["relative_l2"].min())

        lines.append(f"  Best density-particle oracle : rel_L2 = {best_dp_rl2:.6f}")
        if best_dp_n_part is not None:
            bw_str = f"bw_factor={best_dp_bw:.1f}" if np.isfinite(best_dp_bw) else "histogram"
            lines.append(f"    config: n_particles={best_dp_n_part}, n_grid={best_dp_n_grid}, "
                         f"recon={best_dp_recon}, {bw_str}")
            lines.append(f"    step-zero recon error     : {best_dp_step0:.6f}")
            lines.append(f"    forward consistency L2    : {best_dp_fwd:.6e}")
        lines.append(f"  Best gradient-glob oracle   : rel_L2 = {best_gg_rl2:.6f}")
        lines.append(f"  Grad-glob plateau (ref)     : {plateau:.3f}")
        lines.append(f"  Tikhonov best               : rel_L2 = {best_tik_rl2:.6f}")
        lines.append(f"  Spectral best               : rel_L2 = {best_sc_rl2:.6f}")

        # Ratios
        if np.isfinite(best_dp_rl2) and np.isfinite(plateau) and plateau > 0:
            ratio_vs_plateau = best_dp_rl2 / plateau
            lines.append(f"  density_oracle / grad_glob_plateau : {ratio_vs_plateau:.4f}")
        if np.isfinite(best_dp_rl2) and np.isfinite(best_gg_rl2) and best_gg_rl2 > 0:
            ratio_vs_gg = best_dp_rl2 / best_gg_rl2
            lines.append(f"  density_oracle / grad_glob_oracle  : {ratio_vs_gg:.4f}")
        if np.isfinite(best_dp_rl2) and np.isfinite(best_tik_rl2) and best_tik_rl2 > 0:
            ratio_vs_tik = best_dp_rl2 / best_tik_rl2
            lines.append(f"  density_oracle / tikhonov_best     : {ratio_vs_tik:.4f}")

        # Convergence check: does density oracle decrease with n_particles?
        if len(dp) > 0:
            dp_max_ng = dp[dp["n_grid"] == dp["n_grid"].max()]
            by_np = dp_max_ng.groupby("n_particles")["relative_l2"].min().reset_index()
            by_np = by_np[np.isfinite(by_np["relative_l2"])].sort_values("n_particles")
            if len(by_np) >= 3:
                first = float(by_np.iloc[0]["relative_l2"])
                last = float(by_np.iloc[-1]["relative_l2"])
                converging = (last < first * 0.95)
                lines.append(f"  Convergence with n_particles: "
                              f"first={first:.5f}, last={last:.5f}, "
                              f"{'CONVERGING' if converging else 'PLATEAUED'}")

        # Step-zero vs final error analysis
        if best_dp_n_part is not None and np.isfinite(best_dp_step0):
            if best_dp_step0 > 0.5 * best_dp_rl2:
                lines.append("  DIAGNOSIS: step-zero reconstruction error is DOMINANT")
                lines.append("    => reconstruction error bottleneck, not backward dynamics")
            else:
                lines.append("  DIAGNOSIS: step-zero error is smaller than final error")
                lines.append("    => backward dynamics contribute to the remaining error")

        # GO / STOP verdict
        if np.isfinite(best_dp_rl2) and np.isfinite(plateau):
            if best_dp_rl2 < 0.05 and test in ("B", "Z"):
                verdict = "STRONG GO"
            elif best_dp_rl2 < 0.10 and test == "H":
                verdict = "STRONG GO"
            elif best_dp_rl2 < plateau * 0.85:
                verdict = "CONDITIONAL GO (density oracle improves, but not strongly)"
            elif best_dp_rl2 < plateau * 1.05:
                verdict = "STOP/PIVOT (density oracle ≈ gradient-glob plateau)"
            else:
                verdict = "STOP/PIVOT (density oracle WORSE than gradient-glob plateau)"
        else:
            verdict = "INCONCLUSIVE (no finite result)"
        go_conditions[test] = verdict
        lines.append(f"  VERDICT: {verdict}")
        lines.append("")

    # Overall recommendation
    lines.append("=" * 72)
    lines.append("OVERALL VERDICT")
    lines.append("=" * 72)
    for test, v in go_conditions.items():
        lines.append(f"  Test {test}: {v}")
    lines.append("")

    n_strong_go = sum(1 for v in go_conditions.values() if "STRONG GO" in v)
    n_cond_go = sum(1 for v in go_conditions.values() if "CONDITIONAL GO" in v)
    n_stop = sum(1 for v in go_conditions.values() if "STOP" in v)

    lines.append("METHODOLOGICAL INTERPRETATION:")
    lines.append("")
    lines.append("Formulation A (gradient-glob + field score):")
    lines.append("  Globs carry u_x. Score drift is alpha * partial_x log u.")
    lines.append("  Continuity eq. for u_x-particles: q_tau = -alpha * partial_x(q^2/u).")
    lines.append("  This is NOT the gradient backward heat equation (q_tau = -alpha*q_xx).")
    lines.append("  => representation mismatch.")
    lines.append("")
    lines.append("Formulation B (density-particle + field score):")
    lines.append("  Particles carry u. Score drift is alpha * partial_x log u.")
    lines.append("  Continuity eq. for u-particles: u_tau = -alpha * u_xx.")
    lines.append("  => representation consistent.")
    lines.append("")

    if n_strong_go >= 2:
        lines.append("RECOMMENDATION: STRONG GO — density-particle oracle clearly breaks")
        lines.append("the gradient-glob plateau.  Implement estimated density-particle score")
        lines.append("(Formulation B-estimated) as the main new method.")
        lines.append("")
        lines.append("Next steps:")
        lines.append("  1. Implement run_density_particle_estimated_score_deterministic")
        lines.append("     with grid-ratio epsilon score on the reconstructed density.")
        lines.append("  2. Run noise-robustness study at best (n_grid, n_particles).")
        lines.append("  3. Future work: sign-split gradient-native (Formulation C).")
    elif n_cond_go >= 2:
        lines.append("RECOMMENDATION: CONDITIONAL GO — density oracle improves over")
        lines.append("gradient-glob but reconstruction error still limits accuracy.")
        lines.append("The representation mismatch is real but not the sole bottleneck.")
        lines.append("")
        lines.append("Next steps:")
        lines.append("  1. Test larger n_particles (up to 20000) to separate reconstruction")
        lines.append("     floor from backward-dynamics error.")
        lines.append("  2. Implement KDE smoothing improvements (adaptive bandwidth).")
        lines.append("  3. Consider estimated density-particle score if oracle already")
        lines.append("     shows improvement trajectory.")
    else:
        lines.append("RECOMMENDATION: STOP/PIVOT — density-particle oracle does not")
        lines.append("break the gradient-glob plateau.  The ceiling is NOT primarily")
        lines.append("representation mismatch; it is inverse ill-posedness itself.")
        lines.append("")
        lines.append("Next steps:")
        lines.append("  1. Accept that particle methods plateau near rel_L2~0.15-0.24")
        lines.append("     for this problem with clean data.")
        lines.append("  2. Pivot to: (a) noise-robustness study (particle vs spectral),")
        lines.append("     or (b) negative-result framing (ceiling is fundamental),")
        lines.append("     or (c) variable-diffusivity / non-spectral forward model.")
        lines.append("  3. Do NOT invest further in representation reformulation.")

    lines.append("")
    lines.append("NOTE: Density-particle oracle is NOT 'GRW' in the gradient-glob sense.")
    lines.append("  Gradient-glob GRW: representation-native to gradients (u_x).")
    lines.append("  Density-particle: representation-consistent with field-score reverse")
    lines.append("  diffusion, but departs from GRW gradient representation.")
    lines.append("  Beating gradient-glob does not imply beating Tikhonov/spectral.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Representation audit: A vs B oracle")
    parser.add_argument("--base-config", required=True,
                        help="Path to base Gaussian YAML config")
    parser.add_argument("--mixture-config", required=True,
                        help="Path to Gaussian mixture YAML config")
    parser.add_argument("--n-grid", nargs="+", type=int, default=None,
                        help=f"Grid sizes to sweep (default: {N_GRID_VALUES})")
    parser.add_argument("--n-particles", nargs="+", type=int, default=None,
                        help=f"Particle counts to sweep (default: {N_PARTICLES_VALUES})")
    args = parser.parse_args()

    n_grid_vals = args.n_grid if args.n_grid else N_GRID_VALUES
    n_part_vals = args.n_particles if args.n_particles else N_PARTICLES_VALUES

    base_cfg = load_config(args.base_config)
    mixture_cfg = load_config(args.mixture_config)
    test_cfgs = build_test_configs(base_cfg, mixture_cfg)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs") / f"representation_audit_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    all_rows = []
    total_cells = (
        len(test_cfgs) * len(n_grid_vals) *
        (len(n_part_vals) * (1 + len(KDE_BANDWIDTH_FACTORS)) +  # dp cells
         len(GLOB_VALUES) + 1)  # glob cells + baseline cell
    )
    done = 0

    print(f"\nRunning representation audit (~{total_cells} cells)...")
    t0_total = time.perf_counter()

    for test_name, test_cfg in test_cfgs.items():
        print(f"\n  Test {test_name}")

        for n_grid in n_grid_vals:
            print(f"    n_grid={n_grid}", end="", flush=True)

            # === Density-particle oracle sweep ===
            for n_part in n_part_vals:
                # Histogram
                row = run_density_particle_cell(
                    test_name, test_cfg, n_grid, n_part, "histogram", 2.0)
                all_rows.append(row)
                # KDE with each bandwidth factor
                for bw in KDE_BANDWIDTH_FACTORS:
                    row = run_density_particle_cell(
                        test_name, test_cfg, n_grid, n_part, "kde", bw)
                    all_rows.append(row)
                done += 1 + len(KDE_BANDWIDTH_FACTORS)
                print(".", end="", flush=True)

            # === Gradient-glob oracle baseline ===
            for globs in GLOB_VALUES:
                row = run_gradient_glob_cell(test_name, test_cfg, n_grid, globs)
                all_rows.append(row)
                done += 1
                print("g", end="", flush=True)

            # === Classical baselines ===
            baseline_rows = run_baseline_cells(test_name, test_cfg, n_grid)
            all_rows.extend(baseline_rows)
            done += 1
            print("b", end="", flush=True)

        print()

    elapsed = time.perf_counter() - t0_total
    print(f"\nAll cells complete in {elapsed:.1f}s")

    # Save CSV
    df = pd.DataFrame(all_rows)
    csv_path = out_dir / "representation_audit_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"Metrics CSV: {csv_path}")

    # Best-by-test CSV
    best_rows = []
    for test in df["test"].unique():
        sub = df[df["test"] == test]
        for mtype in ["density_particle_oracle", "gradient_glob_oracle",
                      "baseline_tikhonov", "baseline_spectral"]:
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

        plot_oracle_A_vs_B(df, out_dir)

        plot_density_particle_vs_n_particles(
            df, "relative_l2", "relative L2",
            "density_particle_rel_L2_vs_n_particles", out_dir)

        plot_density_particle_vs_n_particles(
            df, "forward_consistency_l2", "forward consistency L2",
            "density_particle_forward_consistency_vs_n_particles", out_dir)

        plot_density_particle_vs_n_particles(
            df, "step_zero_recon_error", "step-zero reconstruction error",
            "step_zero_reconstruction_error_vs_n_particles", out_dir)

        plot_bandwidth_sensitivity(df, out_dir)

        # Field comparison and residual for each test
        for test in sorted(df["test"].unique()):
            sub = df[df["test"] == test]

            # Best density-particle oracle row
            dp_sub = sub[(sub["method_type"] == "density_particle_oracle") &
                         np.isfinite(sub["relative_l2"])]
            if len(dp_sub) == 0:
                continue
            best_dp_idx = dp_sub["relative_l2"].idxmin()
            best_dp_row = dp_sub.loc[best_dp_idx].to_dict()

            # Best gradient-glob oracle row
            gg_sub = sub[(sub["method_type"] == "gradient_glob_oracle") &
                         np.isfinite(sub["relative_l2"])]
            if len(gg_sub) == 0:
                continue
            best_gg_idx = gg_sub["relative_l2"].idxmin()
            best_gg_row = gg_sub.loc[best_gg_idx].to_dict()

            cfg_for_plot = test_cfgs[test]
            plot_field_comparison(df, test, cfg_for_plot, best_dp_row, best_gg_row, out_dir)
            plot_residual_only(df, test, cfg_for_plot, best_dp_row, out_dir)

    print("Plots saved.")

    # Summary
    summary_text = make_summary(df, out_dir, best_df)
    summary_path = out_dir / "representation_audit_summary.txt"
    summary_path.write_text(summary_text)
    print(f"\nSummary: {summary_path}")
    print("\n" + summary_text)


if __name__ == "__main__":
    main()
