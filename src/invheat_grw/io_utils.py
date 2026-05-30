"""
io_utils.py — Save and load experiment artifacts.

Handles:
  - timestamped output directory creation
  - saving metrics to CSV (pandas)
  - saving arrays to NPZ and CSV
  - writing run_summary.txt
  - copying config_used.yaml
"""

from __future__ import annotations

import json
import shutil
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from .config import Config
from .methods import MethodResult
from .metrics import MethodMetrics


def make_output_dir(base_outputs: Path) -> Path:
    """Create a timestamped directory under base_outputs/."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base_outputs / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_config(cfg: Config, config_path: Path, out_dir: Path) -> None:
    """Copy the original config YAML to the output directory."""
    shutil.copy(config_path, out_dir / "config_used.yaml")


def save_metrics_csv(metrics_list: List[MethodMetrics], out_dir: Path) -> Path:
    """Save metrics to metrics_summary.csv."""
    rows = []
    for m in metrics_list:
        rows.append({
            "method": m.method_name,
            "completed": m.completed,
            "failure_step": m.failure_step if m.failure_step is not None else "",
            "failure_msg": m.failure_msg,
            "l2_error": m.l2_error,
            "relative_l2": getattr(m, "relative_l2", ""),
            "linf_error": getattr(m, "linf_error", ""),
            "peak_value": m.peak_value,
            "peak_ratio": getattr(m, "peak_ratio", ""),
            "peak_location": m.peak_location,
            "peak_width_fwhm": m.peak_width_fwhm,
            "sigma_moment": getattr(m, "sigma_moment", ""),
            "width_moment": getattr(m, "width_moment", ""),
            "A_fit": getattr(m, "A_fit", ""),
            "mu_fit": getattr(m, "mu_fit", ""),
            "sigma_fit": getattr(m, "sigma_fit", ""),
            "fit_success": getattr(m, "fit_success", ""),
            "fit_rmse": getattr(m, "fit_rmse", ""),
            "total_variation": m.total_variation,
            "forward_consistency_l2": m.forward_consistency_l2,
            "mass_candidate": getattr(m, "mass_candidate", ""),
            "mass_true": getattr(m, "mass_true", ""),
            "mass_error": getattr(m, "mass_error", ""),
            "mass_rel_error": getattr(m, "mass_rel_error", ""),
            "max_abs_score_final": getattr(m, "max_abs_score_final", ""),
            "max_score_error_L2": getattr(m, "max_score_error_L2", ""),
            "runtime_seconds": getattr(m, "runtime_seconds", ""),
        })
    df = pd.DataFrame(rows)
    path = out_dir / "metrics_summary.csv"
    df.to_csv(path, index=False)
    return path


def save_arrays(
    x_grid: np.ndarray,
    true_u0: np.ndarray,
    observed_final: np.ndarray,
    results: Dict[str, MethodResult],
    out_dir: Path,
) -> None:
    """Save grids and candidate arrays to NPZ and CSV."""
    arrays = {"x_grid": x_grid, "true_u0": true_u0, "observed_final": observed_final}
    for name, res in results.items():
        arrays[f"candidate_{name}"] = res.candidate

    np.savez(out_dir / "arrays.npz", **arrays)

    # Also save as CSV for easy inspection
    df = pd.DataFrame({"x": x_grid, "true_u0": true_u0, "observed_final": observed_final})
    for name, res in results.items():
        df[f"candidate_{name}"] = res.candidate
    df.to_csv(out_dir / "fields.csv", index=False)


METHOD_DESCRIPTIONS = {
    "naive_backward": (
        "Naive backward GRW: X_{k+1} = X_k - sqrt(2*alpha*dt) xi_k.\n"
        "Motivation: hypothesis that simply flipping the sign of the Brownian\n"
        "  increment inverts diffusion.\n"
        "Result: FAILS. xi and -xi are identically distributed (both Gaussian),\n"
        "  so this produces the same diffusive spreading as forward evolution.\n"
        "Expected: candidate does NOT recover the sharp initial peak."
    ),
    "oracle_score_deterministic": (
        "Oracle probability-flow ODE: X_{k+1} = X_k + alpha * s_exact(X_k, t_phys) * dt.\n"
        "Uses exact Gaussian score s = -(x-mu)/sigma_t^2. No noise. Coefficient = alpha.\n"
        "This is the deterministic reverse-time ODE (probability-flow).\n"
        "Expected: particles should drift toward the Gaussian peak, recovering true_u0."
    ),
    "oracle_score_stochastic": (
        "Oracle reverse-time SDE: X_{k+1} = X_k + 2*alpha * s_exact(X_k, t_phys) * dt\n"
        "  + sqrt(2*alpha*dt) xi_k.\n"
        "Uses exact Gaussian score. Includes Brownian noise. Coefficient = 2*alpha.\n"
        "This is the full Anderson (1982) reverse-time SDE.\n"
        "Expected: similar recovery as deterministic, but with stochastic fluctuations."
    ),
    "estimated_score_deterministic_raw": (
        "Estimated probability-flow ODE with RAW score: X_{k+1} = X_k + alpha * s_est * dt.\n"
        "Score estimated as s = u_x/u from current glob reconstruction. NO regularization.\n"
        "If u approx 0, score can blow up. This is recorded as a failure mode, not silently fixed.\n"
        "Expected: either shows anti-diffusive improvement (promising) or blows up\n"
        "  (justifies adding regularization in the next version)."
    ),
    "estimated_score_stochastic_raw": (
        "Estimated reverse-time SDE with RAW score: X_{k+1} = X_k + 2*alpha * s_est * dt\n"
        "  + sqrt(2*alpha*dt) xi_k.\n"
        "Same as estimated deterministic but with noise. Same instability risk.\n"
        "Expected: same as estimated deterministic but noisier."
    ),
}


def write_run_summary(
    cfg: Config,
    metrics_list: List[MethodMetrics],
    true_metrics: MethodMetrics,
    obs_metrics: MethodMetrics,
    out_dir: Path,
) -> Path:
    """Write a human-readable run_summary.txt."""
    lines = []
    lines.append("=" * 70)
    lines.append("INVERSE HEAT SCORE GRW — RUN SUMMARY")
    lines.append("=" * 70)
    lines.append("")

    lines.append("CONFIG")
    lines.append("-" * 40)
    lines.append(f"  Domain:    [{cfg.domain.x_min}, {cfg.domain.x_max}], n_grid={cfg.domain.n_grid}")
    lines.append(f"  Heat:      alpha={cfg.heat.alpha}, T={cfg.heat.T}, dt={cfg.heat.dt}")
    lines.append(f"  IC:        {cfg.initial_condition.type}, "
                 f"mu={cfg.initial_condition.mu}, "
                 f"sigma0={cfg.initial_condition.sigma0}, "
                 f"amplitude={cfg.initial_condition.amplitude}")
    lines.append(f"  GRW:       gradient_globs_per_jump={cfg.grw.gradient_globs_per_jump}, "
                 f"seed={cfg.grw.rng_seed}, boundary={cfg.grw.boundary}")
    lines.append(f"  n_steps:   {cfg.n_steps}")
    lines.append(f"  Safety:    score_threshold={cfg.safety.score_abs_fail_threshold:.2e}, "
                 f"value_threshold={cfg.safety.value_abs_fail_threshold:.2e}")
    lines.append("")

    lines.append("REFERENCE METRICS")
    lines.append("-" * 40)
    lines.append(f"  True u0:  peak={true_metrics.peak_value:.4f}, "
                 f"location={true_metrics.peak_location:.4f}, "
                 f"FWHM={true_metrics.peak_width_fwhm:.4f}, "
                 f"TV={true_metrics.total_variation:.4f}")
    lines.append(f"  Observed: peak={obs_metrics.peak_value:.4f}, "
                 f"location={obs_metrics.peak_location:.4f}, "
                 f"FWHM={obs_metrics.peak_width_fwhm:.4f}, "
                 f"TV={obs_metrics.total_variation:.4f}")
    lines.append("")

    lines.append("METHOD DESCRIPTIONS AND RESULTS")
    lines.append("-" * 40)
    for m in metrics_list:
        lines.append(f"\n  [{m.method_name}]")
        desc = METHOD_DESCRIPTIONS.get(m.method_name, "No description.")
        for dl in desc.split("\n"):
            lines.append(f"    {dl}")
        lines.append(f"  Status:    {'COMPLETED' if m.completed else f'FAILED at step {m.failure_step}'}")
        if m.failure_msg:
            lines.append(f"  Failure:   {m.failure_msg}")
        lines.append(f"  L2 error:     {m.l2_error:.6f}  (rel: {getattr(m, 'relative_l2', float('nan')):.4f}, Linf: {getattr(m, 'linf_error', float('nan')):.6f})")
        lines.append(f"  Peak:         {m.peak_value:.4f} at x={m.peak_location:.4f}  (ratio vs true: {getattr(m, 'peak_ratio', float('nan')):.4f})")
        lines.append(f"  FWHM:         {m.peak_width_fwhm:.4f}  (sigma_moment: {getattr(m, 'sigma_moment', float('nan')):.4f})")
        lines.append(f"  Gauss fit:    A={getattr(m, 'A_fit', float('nan')):.4f}, mu={getattr(m, 'mu_fit', float('nan')):.4f}, sigma={getattr(m, 'sigma_fit', float('nan')):.4f}, ok={getattr(m, 'fit_success', '?')}")
        lines.append(f"  Mass:         cand={getattr(m, 'mass_candidate', float('nan')):.4f}, true={getattr(m, 'mass_true', float('nan')):.4f}, rel_err={getattr(m, 'mass_rel_error', float('nan')):.4f}")
        lines.append(f"  TV:           {m.total_variation:.4f}")
        lines.append(f"  Fwd cons:     {m.forward_consistency_l2:.6f}")
        lines.append(f"  Score err L2: {getattr(m, 'max_score_error_L2', float('nan')):.4f}")
        lines.append(f"  Runtime:      {getattr(m, 'runtime_seconds', float('nan')):.3f}s")
    lines.append("")

    lines.append("METRICS TABLE")
    lines.append("-" * 40)
    header = f"{'Method':<40} {'Status':<15} {'L2 error':>10} {'rel_L2':>8} {'Peak ratio':>10} {'FwdCons':>10} {'Runtime':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    for m in metrics_list:
        status = "OK" if m.completed else f"FAIL@{m.failure_step}"
        lines.append(
            f"{m.method_name:<40} {status:<15} {m.l2_error:>10.5f} "
            f"{getattr(m, 'relative_l2', float('nan')):>8.4f} "
            f"{getattr(m, 'peak_ratio', float('nan')):>10.4f} "
            f"{m.forward_consistency_l2:>10.5f} "
            f"{getattr(m, 'runtime_seconds', float('nan')):>8.3f}"
        )
    lines.append("")

    lines.append("KEY INTERPRETATION")
    lines.append("-" * 40)
    lines.append(
        "  1. Naive backward should NOT recover the peak: L2 error should remain high.\n"
        "  2. Oracle score deterministic tests sign and coefficient of score drift.\n"
        "     If correct, L2 error should be lower than naive backward.\n"
        "  3. Raw estimated score tests whether globs can compute u_x/u without\n"
        "     regularization.  Blow-up or NaN justifies adding epsilon/clipping next.\n"
        "  4. Forward consistency: a good candidate should forward-diffuse close to\n"
        "     the observed final field (low forward_consistency_l2).\n"
        "  5. NO regularization is applied in this run.  All instabilities are raw.\n"
    )

    lines.append("=" * 70)

    path = out_dir / "run_summary.txt"
    path.write_text("\n".join(lines))
    return path
