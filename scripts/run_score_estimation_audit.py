"""
run_score_estimation_audit.py — Score-estimation method audit.

Manuscript labels (draft): `tab:bandwidth`, `fig:bandwidth_sweep`
(`sec:score_estimation`, `sec:bandwidth_sweep`).

Scientific question:
    Can direct KDE derivative score (direct_kde), with larger bandwidth
    (factor 8–16), close the estimated-density vs oracle-density gap that
    fd_grid_ratio with bw_factor=2 failed to close?

Methods compared:
    1. density_particle_oracle_score         — oracle ceiling
    2. density_est_fd_grid_ratio             — baseline (np.gradient path)
    3. density_est_direct_kde                — direct KDE derivative (no FD)
    4. density_est_log_density_fd            — log(u) FD path
    5. density_est_smoothed_log              — Gaussian-smoothed log FD path
    REF: Tikhonov best

Sweep (fast defaults):
    bandwidth_factor : [1, 2, 4, 8, 12, 16]
    epsilon          : [0.0, 1e-10, 1e-8, 1e-6, 1e-4]
    n_grid           : [400]
    n_particles      : [5000]

Full sweep (--full):
    n_grid           : [200, 400]
    n_particles      : [1000, 5000]

Tests:
    B : Gaussian, sigma0=0.08, T=0.15, dt=0.001 (150 steps)
    H : Gaussian mixture, T=0.15
    Z : near-zero tail, sigma0=0.05, mu=0.4, T=0.05

Outputs: outputs/score_estimation_audit_TIMESTAMP/
    score_estimation_audit_metrics.csv
    score_estimation_audit_summary.txt
    best_by_test.csv
    per_step_{test}_{method_type}_bw{bw}_eps{eps}.csv
    rel_L2_vs_bandwidth_{test}.png
    score_error_vs_bandwidth_{test}.png
    rel_L2_vs_epsilon_{test}.png
    max_abs_score_vs_step_{test}.png
    score_error_vs_step_{test}.png
    field_comparison_best_{test}.png

Usage:
    PYTHONPATH=src python scripts/run_score_estimation_audit.py \\
        --base-config configs/gaussian_base.yaml \\
        --mixture-config configs/gaussian_mixture.yaml

    # Smoke test (fast):
    PYTHONPATH=src python scripts/run_score_estimation_audit.py \\
        --base-config configs/gaussian_base.yaml \\
        --mixture-config configs/gaussian_mixture.yaml \\
        --n-grid 100 --n-particles 200 \\
        --bandwidth-factor 1 4 8 --epsilon 0.0 1e-8

    # Full sweep:
    PYTHONPATH=src python scripts/run_score_estimation_audit.py \\
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
from invheat_grw.baselines import tikhonov_inverse


# ---------------------------------------------------------------------------
# Sweep defaults
# ---------------------------------------------------------------------------

BANDWIDTH_FACTORS_DEFAULT = [1.0, 2.0, 4.0, 8.0, 12.0, 16.0]
EPSILON_DEFAULT = [0.0, 1e-10, 1e-8, 1e-6, 1e-4]
N_GRID_DEFAULT = [400]
N_PARTICLES_DEFAULT = [5000]

N_GRID_FULL = [200, 400]
N_PARTICLES_FULL = [1000, 5000]

RECON_METHOD = "kde"
TIKHONOV_LAMBDAS = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
RNG_SEED = 42

SCORE_METHODS = ["fd_grid_ratio", "direct_kde", "log_density_fd", "smoothed_log"]

# Known gradient-glob oracle plateaus from previous convergence study
GRAD_GLOB_ORACLE_PLATEAUS = {"B": 0.175, "H": 0.239, "Z": 0.155}


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

COLORS = {
    "density_particle_oracle": "#1f77b4",   # blue
    "fd_grid_ratio":            "#ff7f0e",   # orange
    "direct_kde":               "#2ca02c",   # green
    "log_density_fd":           "#d62728",   # red
    "smoothed_log":             "#9467bd",   # purple
    "tikhonov":                 "#8c564b",   # brown
    "oracle_plateau":           "#aaaaaa",   # gray
}
LINESTYLES = {
    "density_particle_oracle": "-",
    "fd_grid_ratio": "--",
    "direct_kde": "-",
    "log_density_fd": "-.",
    "smoothed_log": ":",
    "tikhonov": "--",
    "oracle_plateau": ":",
}


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
    if not lst:
        return float("nan")
    if idx == -1:
        idx = len(lst) - 1
    if 0 <= idx < len(lst):
        return _safe(lst[idx])
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
        "score_method": "",
        "n_particles": 0,
        "recon_method": "",
        "bandwidth_factor": float("nan"),
        "bandwidth_used": float("nan"),
        "epsilon": float("nan"),
        "smooth_sigma_factor": float("nan"),
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
    row.update({
        "completed": result.completed,
        "failure_step": (getattr(result, "failure_step", None) or ""),
        "failure_msg": (getattr(result, "failure_msg", "") or ""),
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


def _make_step_rows(result, test: str, n_grid: int, n_particles: int,
                    score_method: str, bw_factor: float, epsilon: float) -> list[dict]:
    n_steps_recorded = len(result.score_L2_error_vs_oracle)
    return [
        {
            "test": test,
            "n_grid": n_grid,
            "n_particles": n_particles,
            "score_method": score_method,
            "bandwidth_factor": bw_factor,
            "epsilon": epsilon,
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


# ---------------------------------------------------------------------------
# Cell runners
# ---------------------------------------------------------------------------

def run_oracle_cell(test: str, cfg: Config, n_grid: int, n_particles: int,
                    bw_factor: float) -> tuple[dict, list[dict]]:
    cfg_r = patch_config(cfg, **{"domain.n_grid": n_grid})
    x_grid = make_grid(cfg_r)
    true_u = compute_true_u0(x_grid, cfg_r)
    u_obs = compute_observed_final(x_grid, cfg_r)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = run_density_particle_oracle_score_deterministic(
            u_obs, x_grid, cfg_r, n_particles,
            recon_method=RECON_METHOD, bandwidth_factor=bw_factor)
        m = compute_metrics(result, true_u, u_obs, x_grid, cfg_r)
        wass = compute_wasserstein(result.candidate, true_u, x_grid)

    row = _base_row(test, n_grid, result.method_name, "density_particle_oracle")
    row.update({
        "score_method": "oracle",
        "n_particles": n_particles,
        "recon_method": RECON_METHOD,
        "bandwidth_factor": bw_factor,
        "bandwidth_used": _safe(result.bandwidth_used),
        "epsilon": 0.0,
    })
    _fill_metrics(row, result, m, wass)

    n_recorded = len(result.score_max_abs)
    step_rows = [
        {
            "test": test, "n_grid": n_grid, "n_particles": n_particles,
            "score_method": "oracle", "bandwidth_factor": bw_factor, "epsilon": 0.0,
            "step": step,
            "score_L2_error_vs_oracle": 0.0,
            "score_core_L2_error_vs_oracle": 0.0,
            "score_Linf_error_vs_oracle": 0.0,
            "score_core_Linf_error_vs_oracle": 0.0,
            "max_abs_score": _listval(result.score_max_abs, step),
            "mean_abs_score": _listval(result.mean_abs_score, step),
            "score_std": _listval(result.score_std, step),
            "n_denom_below_eps": 0,
            "n_clipped": 0,
            "mass": _listval(result.mass_per_step, step),
            "epsilon_actual": 0.0,
        }
        for step in range(n_recorded)
    ]
    return row, step_rows


def run_estimated_cell(
    test: str,
    cfg: Config,
    n_grid: int,
    n_particles: int,
    bw_factor: float,
    epsilon: float,
    score_method: str,
    smooth_sigma_factor: float = 1.0,
) -> tuple[dict, list[dict]]:
    cfg_r = patch_config(cfg, **{"domain.n_grid": n_grid})
    x_grid = make_grid(cfg_r)
    true_u = compute_true_u0(x_grid, cfg_r)
    u_obs = compute_observed_final(x_grid, cfg_r)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = run_density_particle_estimated_score_deterministic(
            u_obs, x_grid, cfg_r, n_particles,
            recon_method=RECON_METHOD,
            bandwidth_factor=bw_factor,
            epsilon=epsilon,
            scale_epsilon_by_peak=False,
            score_clipping=None,
            save_snapshots=False,
            score_method=score_method,
            smooth_sigma_factor=smooth_sigma_factor,
        )
        m = compute_metrics(result, true_u, u_obs, x_grid, cfg_r)
        wass = compute_wasserstein(result.candidate, true_u, x_grid)

    row = _base_row(test, n_grid, result.method_name, f"density_particle_{score_method}")
    row.update({
        "score_method": score_method,
        "n_particles": n_particles,
        "recon_method": RECON_METHOD,
        "bandwidth_factor": bw_factor,
        "bandwidth_used": _safe(result.bandwidth_used),
        "epsilon": epsilon,
        "smooth_sigma_factor": smooth_sigma_factor if score_method == "smoothed_log" else float("nan"),
    })
    _fill_metrics(row, result, m, wass)

    step_rows = _make_step_rows(result, test, n_grid, n_particles, score_method, bw_factor, epsilon)
    return row, step_rows


def run_tikhonov_best_cell(test: str, cfg: Config, n_grid: int) -> dict:
    cfg_r = patch_config(cfg, **{"domain.n_grid": n_grid})
    x_grid = make_grid(cfg_r)
    true_u = compute_true_u0(x_grid, cfg_r)
    u_obs = compute_observed_final(x_grid, cfg_r)
    alpha = cfg_r.heat.alpha
    T_val = cfg_r.heat.T

    best_cand = None
    best_rl2 = float("inf")
    best_lam = float("nan")

    for lam in TIKHONOV_LAMBDAS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            tr = tikhonov_inverse(u_obs, x_grid, alpha, T_val, lam,
                                  length=float(cfg_r.domain.x_max - cfg_r.domain.x_min))
        if not np.all(np.isfinite(tr.candidate)):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            tm = compute_metrics(tr, true_u, u_obs, x_grid, cfg_r)  # type: ignore[arg-type]
        if _safe(tm.relative_l2) < best_rl2:
            best_rl2 = _safe(tm.relative_l2)
            best_cand = tr.candidate.copy()
            best_lam = lam

    if best_cand is None:
        row = _base_row(test, n_grid, "tikhonov_best", "tikhonov")
        return row

    class _FakeResult:
        def __init__(self, cand):
            self.completed = True
            self.failure_step = None
            self.failure_msg = ""
            self.method_name = f"tikhonov_lam={best_lam:.0e}"
            self.candidate = cand
            self.runtime_seconds = float("nan")
            self.step_zero_recon_error = float("nan")
            self.score_L2_error_vs_oracle = []
            self.score_max_abs = []
            self.mean_abs_score = []
            self.score_std = []
            self.n_denominator_below_epsilon = []
            self.n_clipped_scores = []
            self.mass_per_step = []
            self.epsilon_actual_per_step = []

    fr = _FakeResult(best_cand)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fm = compute_metrics(fr, true_u, u_obs, x_grid, cfg_r)  # type: ignore[arg-type]
        fwass = compute_wasserstein(best_cand, true_u, x_grid)

    row = _base_row(test, n_grid, f"tikhonov_lam={best_lam:.0e}", "tikhonov")
    row.update({
        "epsilon": best_lam,
        "score_method": "tikhonov",
    })
    _fill_metrics(row, fr, fm, fwass)
    return row


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _oracle_rl2(rows_df: pd.DataFrame, test: str, n_grid: int, n_particles: int) -> float:
    sub = rows_df[
        (rows_df["test"] == test) &
        (rows_df["n_grid"] == n_grid) &
        (rows_df["n_particles"] == n_particles) &
        (rows_df["score_method"] == "oracle")
    ]
    if sub.empty:
        return float("nan")
    return float(sub["relative_l2"].min())


def plot_rel_l2_vs_bandwidth(df: pd.DataFrame, test: str, out_dir: Path,
                              n_grid: int, n_particles: int) -> None:
    """rel_L2 vs bandwidth_factor for each score_method (at best epsilon)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    bw_methods = ["fd_grid_ratio", "direct_kde", "log_density_fd", "smoothed_log"]
    sub_all = df[
        (df["test"] == test) & (df["n_grid"] == n_grid) &
        (df["n_particles"] == n_particles)
    ]

    oracle_rl2 = _oracle_rl2(df, test, n_grid, n_particles)
    if np.isfinite(oracle_rl2):
        ax.axhline(oracle_rl2, color=COLORS["density_particle_oracle"],
                   linestyle=LINESTYLES["density_particle_oracle"],
                   label=f"oracle ({oracle_rl2:.4f})", linewidth=1.5)

    for sm in bw_methods:
        sub = sub_all[sub_all["score_method"] == sm]
        if sub.empty:
            continue
        # For each bandwidth_factor, pick the best (lowest) rel_L2 over epsilon values
        bw_vals = sorted(sub["bandwidth_factor"].dropna().unique())
        rl2_vals = []
        for bw in bw_vals:
            bw_sub = sub[sub["bandwidth_factor"] == bw]
            best_rl2 = float(bw_sub["relative_l2"].dropna().min()) if not bw_sub.empty else float("nan")
            rl2_vals.append(best_rl2)
        ax.plot(bw_vals, rl2_vals,
                color=COLORS.get(sm, "black"),
                linestyle=LINESTYLES.get(sm, "-"),
                marker="o", label=sm)

    tik_row = sub_all[sub_all["score_method"] == "tikhonov"]
    if not tik_row.empty:
        tik_rl2 = float(tik_row["relative_l2"].min())
        ax.axhline(tik_rl2, color=COLORS["tikhonov"], linestyle=LINESTYLES["tikhonov"],
                   label=f"Tikhonov best ({tik_rl2:.4f})", linewidth=1.0)

    ax.set_xlabel("bandwidth_factor")
    ax.set_ylabel("relative L2 error")
    ax.set_title(f"Test {test}: rel_L2 vs bandwidth_factor (best ε, n_grid={n_grid}, N={n_particles})")
    ax.legend(fontsize=8)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"rel_L2_vs_bandwidth_{test}.png", dpi=120)
    plt.close(fig)


def plot_score_error_vs_bandwidth(df: pd.DataFrame, test: str, out_dir: Path,
                                   n_grid: int, n_particles: int) -> None:
    """max score L2 error vs bandwidth_factor."""
    fig, ax = plt.subplots(figsize=(8, 5))
    bw_methods = ["fd_grid_ratio", "direct_kde", "log_density_fd", "smoothed_log"]
    sub_all = df[
        (df["test"] == test) & (df["n_grid"] == n_grid) &
        (df["n_particles"] == n_particles)
    ]
    for sm in bw_methods:
        sub = sub_all[sub_all["score_method"] == sm]
        if sub.empty:
            continue
        bw_vals = sorted(sub["bandwidth_factor"].dropna().unique())
        err_vals = []
        for bw in bw_vals:
            bw_sub = sub[sub["bandwidth_factor"] == bw]
            # At best epsilon (minimise score error)
            best_err = float(bw_sub["max_score_L2_error_vs_oracle"].dropna().min()) if not bw_sub.empty else float("nan")
            err_vals.append(best_err)
        ax.plot(bw_vals, err_vals,
                color=COLORS.get(sm, "black"),
                linestyle=LINESTYLES.get(sm, "-"),
                marker="o", label=sm)

    ax.set_xlabel("bandwidth_factor")
    ax.set_ylabel("max score L2 error vs oracle")
    ax.set_title(f"Test {test}: score error vs bandwidth_factor (n_grid={n_grid}, N={n_particles})")
    ax.legend(fontsize=8)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"score_error_vs_bandwidth_{test}.png", dpi=120)
    plt.close(fig)


def plot_rel_l2_vs_epsilon(df: pd.DataFrame, test: str, out_dir: Path,
                            n_grid: int, n_particles: int) -> None:
    """rel_L2 vs epsilon at fixed bandwidth_factor=8 (or closest available)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    bw_methods = ["fd_grid_ratio", "direct_kde", "log_density_fd", "smoothed_log"]
    sub_all = df[
        (df["test"] == test) & (df["n_grid"] == n_grid) &
        (df["n_particles"] == n_particles)
    ]

    oracle_rl2 = _oracle_rl2(df, test, n_grid, n_particles)
    if np.isfinite(oracle_rl2):
        ax.axhline(oracle_rl2, color=COLORS["density_particle_oracle"],
                   linestyle=LINESTYLES["density_particle_oracle"],
                   label=f"oracle ({oracle_rl2:.4f})", linewidth=1.5)

    # For epsilon plot, pick bandwidth_factor = 8 (or closest)
    target_bw = 8.0
    for sm in bw_methods:
        sub = sub_all[sub_all["score_method"] == sm]
        if sub.empty:
            continue
        avail_bw = sub["bandwidth_factor"].dropna().unique()
        if len(avail_bw) == 0:
            continue
        chosen_bw = avail_bw[np.argmin(np.abs(avail_bw - target_bw))]
        sub_bw = sub[sub["bandwidth_factor"] == chosen_bw]
        eps_vals = sorted(sub_bw["epsilon"].dropna().unique())
        rl2_vals = [
            float(sub_bw[sub_bw["epsilon"] == e]["relative_l2"].dropna().min())
            if not sub_bw[sub_bw["epsilon"] == e].empty else float("nan")
            for e in eps_vals
        ]
        # Replace 0 epsilon with small value for log-x axis
        eps_plot = [max(e, 1e-13) for e in eps_vals]
        ax.semilogx(eps_plot, rl2_vals,
                    color=COLORS.get(sm, "black"),
                    linestyle=LINESTYLES.get(sm, "-"),
                    marker="o", label=f"{sm} (bw={chosen_bw:.0f})")

    ax.set_xlabel("epsilon")
    ax.set_ylabel("relative L2 error")
    ax.set_title(f"Test {test}: rel_L2 vs epsilon (bw_factor≈8, n_grid={n_grid}, N={n_particles})")
    ax.legend(fontsize=8)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"rel_L2_vs_epsilon_{test}.png", dpi=120)
    plt.close(fig)


def _load_step_diagnostics(step_csv_paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in step_csv_paths:
        if p.exists():
            frames.append(pd.read_csv(p))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_max_abs_score_vs_step(step_rows_by_key: dict, test: str, out_dir: Path,
                                n_grid: int, n_particles: int, target_bw: float = 8.0,
                                target_eps: float = 1e-8) -> None:
    """max |score| vs step for different score_methods at bw≈8, eps≈1e-8."""
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_methods = ["oracle", "fd_grid_ratio", "direct_kde", "log_density_fd", "smoothed_log"]
    for sm in plot_methods:
        key = (test, n_grid, n_particles, sm, target_bw, target_eps)
        if key not in step_rows_by_key:
            # Try oracle (eps=0, any bw)
            if sm == "oracle":
                oracle_keys = [k for k in step_rows_by_key if k[0] == test and k[2] == n_particles and k[3] == "oracle"]
                if oracle_keys:
                    key = oracle_keys[0]
                else:
                    continue
            else:
                # Fallback: closest bw
                cands = [k for k in step_rows_by_key
                         if k[0] == test and k[2] == n_particles and k[3] == sm]
                if not cands:
                    continue
                cands.sort(key=lambda k: abs(k[4] - target_bw))
                key = cands[0]

        rows = step_rows_by_key[key]
        if not rows:
            continue
        steps = [r["step"] for r in rows]
        vals = [r.get("max_abs_score", float("nan")) for r in rows]
        ax.plot(steps, vals,
                color=COLORS.get(sm, "black"),
                linestyle=LINESTYLES.get(sm, "-"),
                label=sm, linewidth=1.2)

    ax.set_xlabel("step")
    ax.set_ylabel("max |score| at particles")
    ax.set_title(f"Test {test}: max |score| vs step (bw≈{target_bw:.0f}, ε≈{target_eps:.0e})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"max_abs_score_vs_step_{test}.png", dpi=120)
    plt.close(fig)


def plot_score_error_vs_step(step_rows_by_key: dict, test: str, out_dir: Path,
                              n_grid: int, n_particles: int, target_bw: float = 8.0,
                              target_eps: float = 1e-8) -> None:
    """Score L2 error vs oracle vs step for different score_methods."""
    fig, ax = plt.subplots(figsize=(8, 5))
    est_methods = ["fd_grid_ratio", "direct_kde", "log_density_fd", "smoothed_log"]
    for sm in est_methods:
        key = (test, n_grid, n_particles, sm, target_bw, target_eps)
        if key not in step_rows_by_key:
            cands = [k for k in step_rows_by_key
                     if k[0] == test and k[2] == n_particles and k[3] == sm]
            if not cands:
                continue
            cands.sort(key=lambda k: abs(k[4] - target_bw))
            key = cands[0]

        rows = step_rows_by_key[key]
        if not rows:
            continue
        steps = [r["step"] for r in rows]
        vals = [r.get("score_L2_error_vs_oracle", float("nan")) for r in rows]
        ax.plot(steps, vals,
                color=COLORS.get(sm, "black"),
                linestyle=LINESTYLES.get(sm, "-"),
                label=sm, linewidth=1.2)

    ax.set_xlabel("step")
    ax.set_ylabel("score L2 error vs oracle")
    ax.set_title(f"Test {test}: score error vs step (bw≈{target_bw:.0f}, ε≈{target_eps:.0e})")
    ax.legend(fontsize=8)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"score_error_vs_step_{test}.png", dpi=120)
    plt.close(fig)


def plot_field_comparison_best(
    df: pd.DataFrame,
    test: str,
    cfg: Config,
    n_grid: int,
    n_particles: int,
    out_dir: Path,
    bw_factors: list[float],
    epsilons: list[float],
) -> None:
    """Overlay true u0, observed uT, oracle density, best direct_kde, fd_grid_ratio, Tikhonov."""
    cfg_r = patch_config(cfg, **{"domain.n_grid": n_grid})
    x_grid = make_grid(cfg_r)
    true_u = compute_true_u0(x_grid, cfg_r)
    u_obs = compute_observed_final(x_grid, cfg_r)

    sub = df[
        (df["test"] == test) & (df["n_grid"] == n_grid) &
        (df["n_particles"] == n_particles)
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_grid, true_u, "k-", linewidth=2, label="true u0", zorder=5)
    ax.plot(x_grid, u_obs, "k--", linewidth=1.2, alpha=0.5, label="observed uT")

    def _run_and_plot(score_method: str, color: str, label: str):
        sub_sm = sub[sub["score_method"] == score_method]
        if sub_sm.empty:
            return
        best_row = sub_sm.loc[sub_sm["relative_l2"].idxmin()]
        bw_f = float(best_row["bandwidth_factor"])  # type: ignore[arg-type]
        eps = float(best_row["epsilon"])  # type: ignore[arg-type]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = run_density_particle_estimated_score_deterministic(
                u_obs, x_grid, cfg_r, n_particles,
                recon_method=RECON_METHOD,
                bandwidth_factor=bw_f,
                epsilon=eps,
                scale_epsilon_by_peak=False,
                score_clipping=None,
                save_snapshots=False,
                score_method=score_method,
            )
        rl2 = float(best_row["relative_l2"])  # type: ignore[arg-type]
        ax.plot(x_grid, result.candidate,
                color=color, linestyle="-", linewidth=1.2,
                label=f"{label} (bw={bw_f:.0f}, ε={eps:.0e}) rl2={rl2:.4f}")

    # Oracle density
    sub_oracle = sub[sub["score_method"] == "oracle"]
    if not sub_oracle.empty:
        best_oc = sub_oracle.loc[sub_oracle["relative_l2"].idxmin()]
        bw_oc = float(best_oc["bandwidth_factor"])  # type: ignore[arg-type]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            r_oc = run_density_particle_oracle_score_deterministic(
                u_obs, x_grid, cfg_r, n_particles,
                recon_method=RECON_METHOD, bandwidth_factor=bw_oc)
        oc_rl2 = float(best_oc["relative_l2"])  # type: ignore[arg-type]
        ax.plot(x_grid, r_oc.candidate,
                color=COLORS["density_particle_oracle"], linestyle="-", linewidth=2,
                label=f"oracle (bw={bw_oc:.0f}) rl2={oc_rl2:.4f}")

    _run_and_plot("direct_kde", COLORS["direct_kde"], "direct_kde best")
    _run_and_plot("fd_grid_ratio", COLORS["fd_grid_ratio"], "fd_grid_ratio best")

    # Tikhonov best
    alpha = cfg_r.heat.alpha
    T_val = cfg_r.heat.T
    best_tik_cand = None
    best_tik_rl2 = float("inf")
    best_tik_lam = float("nan")
    for lam in TIKHONOV_LAMBDAS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            tr = tikhonov_inverse(u_obs, x_grid, alpha, T_val, lam,
                                  length=float(cfg_r.domain.x_max - cfg_r.domain.x_min))
        if not np.all(np.isfinite(tr.candidate)):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            tm = compute_metrics(tr, true_u, u_obs, x_grid, cfg_r)  # type: ignore[arg-type]
        if _safe(tm.relative_l2) < best_tik_rl2:
            best_tik_rl2 = _safe(tm.relative_l2)
            best_tik_cand = tr.candidate.copy()
            best_tik_lam = lam
    if best_tik_cand is not None:
        ax.plot(x_grid, best_tik_cand,
                color=COLORS["tikhonov"], linestyle=LINESTYLES["tikhonov"], linewidth=1.2,
                label=f"Tikhonov (λ={best_tik_lam:.0e}) rl2={best_tik_rl2:.4f}")

    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_title(f"Test {test}: field comparison best methods (n_grid={n_grid}, N={n_particles})")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"field_comparison_best_{test}.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------

def _verdict(test: str, oracle_rl2: float, direct_kde_rl2: float,
             fd_rl2: float) -> str:
    """Return GO / CONDITIONAL_GO / STOP based on the study's decision thresholds."""
    if not np.isfinite(oracle_rl2) or not np.isfinite(direct_kde_rl2):
        return "PIVOT – insufficient data"
    ratio = direct_kde_rl2 / oracle_rl2
    if ratio < 2.0:
        return f"STRONG_GO  (ratio={ratio:.2f}× oracle)"
    if ratio < 5.0:
        return f"CONDITIONAL_GO  (ratio={ratio:.2f}× oracle)"
    return f"STOP/PIVOT  (ratio={ratio:.2f}× oracle — bandwidth alone insufficient)"


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------

def write_summary(
    df: pd.DataFrame,
    out_dir: Path,
    bw_factors: list[float],
    epsilons: list[float],
    n_grid: int,
    n_particles: int,
) -> None:
    lines = [
        "=" * 72,
        "SCORE ESTIMATION AUDIT — SUMMARY",
        f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  n_grid    : {n_grid}",
        f"  n_particles: {n_particles}",
        f"  bw_factors: {bw_factors}",
        f"  epsilons  : {epsilons}",
        "=" * 72,
        "",
        "Scientific question:",
        "  Can direct KDE derivative score (direct_kde) with larger bandwidth",
        "  close the estimated/oracle density-particle gap?",
        "",
        "Note: Bandwidth IS the regularization parameter for direct_kde.",
        "The goal is not to beat Tikhonov on constant-coefficient heat,",
        "but to determine if the particle score framework is viable for",
        "non-spectral problems.",
        "",
        "=" * 72,
        "PER-TEST RESULTS",
        "",
    ]

    best_rows = []

    for test in ["B", "H", "Z"]:
        sub = df[
            (df["test"] == test) & (df["n_grid"] == n_grid) &
            (df["n_particles"] == n_particles)
        ]
        if sub.empty:
            lines.append(f"Test {test}: NO DATA")
            continue

        lines.append(f"Test {test}:")
        lines.append(f"  {'Method':<35}  {'bw_factor':>10}  {'epsilon':>10}  {'rel_L2':>10}  {'completed':>10}")
        lines.append(f"  {'-'*35}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

        oracle_rl2 = _oracle_rl2(df, test, n_grid, n_particles)

        # Print best row for each score_method
        for sm in ["oracle", "fd_grid_ratio", "direct_kde", "log_density_fd",
                   "smoothed_log", "tikhonov"]:
            sub_sm = sub[sub["score_method"] == sm]
            if sub_sm.empty:
                continue
            sub_sm_completed = sub_sm[sub_sm["completed"] == True]
            src = sub_sm_completed if not sub_sm_completed.empty else sub_sm
            best = src.loc[src["relative_l2"].idxmin()] if "relative_l2" in src.columns else src.iloc[0]
            bw_f = best.get("bandwidth_factor", float("nan"))
            eps = best.get("epsilon", float("nan"))
            rl2 = best.get("relative_l2", float("nan"))
            comp = best.get("completed", False)
            lines.append(
                f"  {sm:<35}  {bw_f:>10.1f}  {eps:>10.2e}  {_safe(rl2):>10.5f}  {str(comp):>10}"
            )
            if sm not in ("oracle", "tikhonov"):
                brow = {
                    "test": test,
                    "score_method": sm,
                    "bandwidth_factor": bw_f,
                    "epsilon": eps,
                    "relative_l2": _safe(rl2),
                    "oracle_relative_l2": oracle_rl2,
                    "ratio_to_oracle": _safe(rl2) / oracle_rl2 if np.isfinite(oracle_rl2) and np.isfinite(_safe(rl2)) else float("nan"),
                    "completed": comp,
                }
                best_rows.append(brow)

        # Verdicts for this test: direct_kde at bw in [8, 12]
        bw_target = [bw for bw in [8.0, 12.0] if bw in bw_factors]
        sub_dk = sub[(sub["score_method"] == "direct_kde") & (sub["bandwidth_factor"].isin(bw_target))]
        if not sub_dk.empty:
            dk_rl2_best = float(sub_dk["relative_l2"].dropna().min())
        else:
            dk_rl2_best = float("nan")

        sub_fd = sub[(sub["score_method"] == "fd_grid_ratio")]
        fd_rl2_best = float(sub_fd["relative_l2"].dropna().min()) if not sub_fd.empty else float("nan")

        verdict = _verdict(test, oracle_rl2, dk_rl2_best, fd_rl2_best)
        lines.append(f"")
        lines.append(f"  Oracle rel_L2            : {oracle_rl2:.5f}")
        lines.append(f"  Best fd_grid_ratio rel_L2: {fd_rl2_best:.5f}")
        lines.append(f"  Best direct_kde rel_L2   : {dk_rl2_best:.5f}  (bw=[8,12] if available)")
        lines.append(f"  VERDICT: {verdict}")
        lines.append(f"  Grad-glob oracle plateau : {GRAD_GLOB_ORACLE_PLATEAUS.get(test, 'N/A')}")
        lines.append("")

    lines += [
        "=" * 72,
        "FIVE KEY QUESTIONS",
        "",
        "1. Did direct_kde outperform fd_grid_ratio?",
    ]

    for test in ["B", "H", "Z"]:
        sub = df[(df["test"] == test) & (df["n_grid"] == n_grid) & (df["n_particles"] == n_particles)]
        dk = sub[sub["score_method"] == "direct_kde"]
        fd = sub[sub["score_method"] == "fd_grid_ratio"]
        dk_rl2 = float(dk["relative_l2"].dropna().min()) if not dk.empty else float("nan")
        fd_rl2 = float(fd["relative_l2"].dropna().min()) if not fd.empty else float("nan")
        if np.isfinite(dk_rl2) and np.isfinite(fd_rl2):
            winner = "direct_kde" if dk_rl2 < fd_rl2 else "fd_grid_ratio"
            lines.append(f"   Test {test}: direct_kde={dk_rl2:.5f} vs fd={fd_rl2:.5f} → {winner}")
        else:
            lines.append(f"   Test {test}: insufficient data")

    lines += [
        "",
        "2. Is best bandwidth ≈ 8–12 for T=0.15 (Tests B, H)?",
    ]
    for test in ["B", "H"]:
        sub = df[(df["test"] == test) & (df["n_grid"] == n_grid) &
                 (df["n_particles"] == n_particles) & (df["score_method"] == "direct_kde")]
        if not sub.empty:
            best_bw_row = sub.loc[sub["relative_l2"].idxmin()]
            lines.append(f"   Test {test}: best bw_factor = {best_bw_row['bandwidth_factor']:.1f} (rl2={best_bw_row['relative_l2']:.5f})")
        else:
            lines.append(f"   Test {test}: no direct_kde data")

    lines += [
        "",
        "3. Is rel_L2 curve U-shaped vs bandwidth?",
        "   (See rel_L2_vs_bandwidth_{test}.png for visual inspection)",
        "",
        "4. Score error vs oracle (max over steps):",
    ]
    for test in ["B", "H", "Z"]:
        sub = df[(df["test"] == test) & (df["n_grid"] == n_grid) & (df["n_particles"] == n_particles)]
        for sm in ["fd_grid_ratio", "direct_kde"]:
            sub_sm = sub[sub["score_method"] == sm]
            if not sub_sm.empty:
                max_err = float(sub_sm["max_score_L2_error_vs_oracle"].dropna().min())
                lines.append(f"   Test {test} {sm}: max score L2 error = {max_err:.3e}")

    lines += [
        "",
        "5. Framework viability for non-spectral problems:",
        "   If any test achieves direct_kde ratio < 5× oracle,",
        "   the particle score framework warrants further investigation.",
        "   (See VERDICTS above)",
        "",
        "=" * 72,
    ]

    summary_text = "\n".join(lines)
    (out_dir / "score_estimation_audit_summary.txt").write_text(summary_text)
    print(summary_text)

    if best_rows:
        pd.DataFrame(best_rows).to_csv(out_dir / "best_by_test.csv", index=False)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score estimation audit sweep.")
    p.add_argument("--base-config", default="configs/gaussian_base.yaml")
    p.add_argument("--mixture-config", default="configs/gaussian_mixture.yaml")
    p.add_argument("--full", action="store_true", help="Use full n_grid/n_particles sweep.")
    p.add_argument("--n-grid", type=int, nargs="+", default=None)
    p.add_argument("--n-particles", type=int, nargs="+", default=None)
    p.add_argument("--bandwidth-factor", type=float, nargs="+", default=None)
    p.add_argument("--epsilon", type=float, nargs="+", default=None)
    p.add_argument("--score-methods", nargs="+", default=None,
                   choices=SCORE_METHODS,
                   help="Score methods to run (default: all four).")
    p.add_argument("--skip-tikhonov", action="store_true", help="Skip Tikhonov baseline.")
    p.add_argument("--skip-field-comparison", action="store_true",
                   help="Skip field comparison plot (re-runs best cells; slow).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Load configs
    base_cfg = load_config(args.base_config)
    mixture_cfg = load_config(args.mixture_config)
    test_cfgs = build_test_configs(base_cfg, mixture_cfg)

    # Resolve sweep parameters
    if args.full:
        n_grids = args.n_grid or N_GRID_FULL
        n_parts_list = args.n_particles or N_PARTICLES_FULL
    else:
        n_grids = args.n_grid or N_GRID_DEFAULT
        n_parts_list = args.n_particles or N_PARTICLES_DEFAULT

    bw_factors = args.bandwidth_factor or BANDWIDTH_FACTORS_DEFAULT
    epsilons = args.epsilon or EPSILON_DEFAULT
    score_methods = args.score_methods or SCORE_METHODS

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs") / f"score_estimation_audit_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    all_metric_rows: list[dict] = []
    step_rows_by_key: dict = {}  # (test, n_grid, n_particles, score_method, bw, eps) → list[dict]

    total_cells = (
        len(test_cfgs) * len(n_grids) * len(n_parts_list) *
        (1 + len(score_methods) * len(bw_factors) * len(epsilons))
    )
    print(f"Total cells: {total_cells}")

    cell_idx = 0
    t_global_start = time.perf_counter()

    for n_grid in n_grids:
        for n_particles in n_parts_list:
            for test, cfg in test_cfgs.items():
                # --- Oracle ---
                cell_idx += 1
                # Use a middle bw_factor for oracle reference (not swept — oracle ignores it)
                oracle_bw = 4.0
                print(f"[{cell_idx}/{total_cells}] Oracle  test={test} n_grid={n_grid} N={n_particles} bw={oracle_bw}")
                try:
                    row, step_rows = run_oracle_cell(test, cfg, n_grid, n_particles, oracle_bw)
                    all_metric_rows.append(row)
                    key = (test, n_grid, n_particles, "oracle", oracle_bw, 0.0)
                    step_rows_by_key[key] = step_rows
                except Exception as exc:
                    print(f"  ERROR: {exc}")

                # --- Tikhonov baseline ---
                if not args.skip_tikhonov:
                    try:
                        tik_row = run_tikhonov_best_cell(test, cfg, n_grid)
                        all_metric_rows.append(tik_row)
                    except Exception as exc:
                        print(f"  Tikhonov ERROR: {exc}")

                # --- Estimated score sweep ---
                for sm in score_methods:
                    for bw in bw_factors:
                        for eps in epsilons:
                            cell_idx += 1
                            print(
                                f"[{cell_idx}/{total_cells}] {sm:20s} "
                                f"test={test} n_grid={n_grid} N={n_particles} "
                                f"bw={bw:.0f} eps={eps:.0e}"
                            )
                            try:
                                row, step_rows = run_estimated_cell(
                                    test, cfg, n_grid, n_particles,
                                    bw, eps, sm,
                                )
                                all_metric_rows.append(row)
                                key = (test, n_grid, n_particles, sm, bw, eps)
                                step_rows_by_key[key] = step_rows
                            except Exception as exc:
                                print(f"  ERROR: {exc}")
                                import traceback
                                traceback.print_exc()

    print(f"\nAll cells done in {time.perf_counter() - t_global_start:.1f}s")

    # --- Save metrics CSV ---
    df = pd.DataFrame(all_metric_rows)
    metrics_csv = out_dir / "score_estimation_audit_metrics.csv"
    df.to_csv(metrics_csv, index=False)
    print(f"Saved {metrics_csv}")

    # --- Save per-step diagnostics ---
    all_step_rows = []
    for rows in step_rows_by_key.values():
        all_step_rows.extend(rows)
    if all_step_rows:
        step_df = pd.DataFrame(all_step_rows)
        step_csv = out_dir / "score_estimation_audit_step_diagnostics.csv"
        step_df.to_csv(step_csv, index=False)
        print(f"Saved {step_csv}")

    # --- Plots and summary (use first n_grid, first n_particles for focus plots) ---
    focus_n_grid = n_grids[0]
    focus_n_parts = n_parts_list[0]

    # Pick the highest bw_factor and a medium epsilon for step plots
    target_bw_step = max(bw for bw in bw_factors if bw <= 12.0) if any(bw <= 12.0 for bw in bw_factors) else bw_factors[-1]
    target_eps_step = 1e-8 if 1e-8 in epsilons else epsilons[0]

    for test in test_cfgs:
        try:
            plot_rel_l2_vs_bandwidth(df, test, out_dir, focus_n_grid, focus_n_parts)
        except Exception as exc:
            print(f"  plot_rel_l2_vs_bandwidth {test}: {exc}")
        try:
            plot_score_error_vs_bandwidth(df, test, out_dir, focus_n_grid, focus_n_parts)
        except Exception as exc:
            print(f"  plot_score_error_vs_bandwidth {test}: {exc}")
        try:
            plot_rel_l2_vs_epsilon(df, test, out_dir, focus_n_grid, focus_n_parts)
        except Exception as exc:
            print(f"  plot_rel_l2_vs_epsilon {test}: {exc}")
        try:
            plot_max_abs_score_vs_step(
                step_rows_by_key, test, out_dir, focus_n_grid, focus_n_parts,
                target_bw=target_bw_step, target_eps=target_eps_step)
        except Exception as exc:
            print(f"  plot_max_abs_score_vs_step {test}: {exc}")
        try:
            plot_score_error_vs_step(
                step_rows_by_key, test, out_dir, focus_n_grid, focus_n_parts,
                target_bw=target_bw_step, target_eps=target_eps_step)
        except Exception as exc:
            print(f"  plot_score_error_vs_step {test}: {exc}")
        if not args.skip_field_comparison:
            try:
                plot_field_comparison_best(
                    df, test, test_cfgs[test], focus_n_grid, focus_n_parts,
                    out_dir, bw_factors, epsilons)
            except Exception as exc:
                print(f"  plot_field_comparison_best {test}: {exc}")

    # --- Summary ---
    write_summary(df, out_dir, bw_factors, epsilons, focus_n_grid, focus_n_parts)

    print(f"\nDone. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
