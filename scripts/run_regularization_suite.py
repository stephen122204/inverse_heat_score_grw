"""
run_regularization_suite.py — Regularization + baseline experiment suite.

Tests:
  A : T=0.05, sigma0=0.08  (standard Gaussian, moderate diffusion)
  B : T=0.15, sigma0=0.08  (heavy diffusion — hardest inversion)
  D : T=0.05, sigma0=0.05  (moderate diffusion, narrow peak)
  G : T=0.05, Gaussian mixture (2 components)
  H : T=0.15, Gaussian mixture (heavy diffusion)
  Z : T=0.05, sigma0=0.05, mu=0.4, no background (near-zero tail stress test)

For each test:
  - Epsilon sweep: est_det_regularized with epsilon in [0,1e-10,1e-8,1e-6,1e-5,1e-4,1e-3]
  - Clipping sweep (second pass): best_epsilon + max_abs_score in [50,100,200]
  - Baselines: spectral_cutoff, tikhonov (lambda=1e-6)
  - Reference methods: naive_backward, oracle_score_deterministic, est_det_raw

Noisy tests (B and H):
  - noise_levels = [0.001, 0.005, 0.01]
  - Methods: raw, regularized (best epsilon), spectral_cutoff (noise_delta=noise_level)

Console summary:
  - Per test: rescue vs refinement verdict
  - Best epsilon recommendation
  - Ratio: raw_rel_L2 / oracle_rel_L2  (how much regularization helps)
  - particle_rel_L2_over_spectral (is particle competitive with classical?)

Usage:
    python scripts/run_regularization_suite.py \\
        --base-config configs/gaussian_base.yaml \\
        --mixture-config configs/gaussian_mixture.yaml
"""

from __future__ import annotations

import sys
import argparse
import copy
import math
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from invheat_grw.config import load_config, Config, RegularizationConfig, EpsilonFloorConfig, ScoreClippingConfig, SmoothingConfig
from invheat_grw.fields import make_grid, true_u0 as compute_true_u0, observed_final as compute_observed_final
from invheat_grw.globs import field_to_globs, reconstruct_field
from invheat_grw.methods import (
    run_naive_backward,
    run_oracle_score_deterministic,
    run_estimated_score_deterministic_raw,
    run_estimated_score_deterministic_regularized,
    MethodResult,
)
from invheat_grw.metrics import (
    compute_metrics,
    compute_baseline_metrics,
    compute_wasserstein,
    MethodMetrics,
    forward_heat_solve_dct,
)
from invheat_grw.baselines import spectral_cutoff_inverse, tikhonov_inverse


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPSILON_VALUES = [0.0, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3]
CLIP_VALUES = [50.0, 100.0, 200.0]
NOISE_LEVELS = [0.001, 0.005, 0.01]
TIKHONOV_LAMBDAS = [1e-6, 1e-4]


# ---------------------------------------------------------------------------
# Config patching helper (handles arbitrary depth via dot notation)
# ---------------------------------------------------------------------------

def patch_config(cfg: Config, **overrides) -> Config:
    """Return a deep-copied config with nested fields patched.
    Supports arbitrary depth: e.g. 'regularization.epsilon_floor.value'.
    """
    new_cfg = copy.deepcopy(cfg)
    for key, val in overrides.items():
        parts = key.split(".")
        obj = new_cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], val)
    return new_cfg


def make_regularization_cfg(
    enabled: bool = True,
    eps_enabled: bool = False,
    eps_value: float = 0.0,
    scale_by_peak: bool = False,
    clip_enabled: bool = False,
    max_abs_score: float = 100.0,
) -> RegularizationConfig:
    """Build a RegularizationConfig from explicit parameters."""
    return RegularizationConfig(
        enabled=enabled,
        epsilon_floor=EpsilonFloorConfig(
            enabled=eps_enabled,
            value=eps_value,
            scale_by_peak=scale_by_peak,
        ),
        score_clipping=ScoreClippingConfig(
            enabled=clip_enabled,
            max_abs_score=max_abs_score,
        ),
        smoothing=SmoothingConfig(enabled=False),
    )


# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------

def build_test_configs(base_cfg: Config, mixture_cfg: Config) -> dict:
    """Return dict of test_name -> Config."""
    tests = {}

    # Test A: standard Gaussian, moderate diffusion
    tests["A"] = patch_config(base_cfg, **{"heat.T": 0.05, "initial_condition.sigma0": 0.08})

    # Test B: heavy diffusion
    tests["B"] = patch_config(base_cfg, **{"heat.T": 0.15, "initial_condition.sigma0": 0.08})

    # Test D: narrow peak
    tests["D"] = patch_config(base_cfg, **{
        "heat.T": 0.05,
        "initial_condition.sigma0": 0.05,
        "grw.gradient_globs_per_jump": 40,
    })

    # Test G: Gaussian mixture, moderate diffusion
    tests["G"] = patch_config(mixture_cfg, **{"heat.T": 0.05})

    # Test H: Gaussian mixture, heavy diffusion
    tests["H"] = patch_config(mixture_cfg, **{"heat.T": 0.15})

    # Test Z: near-zero tail stress (narrow Gaussian, no background, n_grid=300)
    tests["Z"] = patch_config(base_cfg, **{
        "heat.T": 0.05,
        "initial_condition.sigma0": 0.05,
        "initial_condition.mu": 0.4,
        "initial_condition.amplitude": 1.0,
        "domain.n_grid": 300,
        "grw.gradient_globs_per_jump": 40,
    })

    return tests


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------

def run_single(name: str, fn, u_obs: np.ndarray, x_grid: np.ndarray,
               cfg: Config, rng_seed: int = 0) -> MethodResult:
    rng = np.random.default_rng(rng_seed)
    print(f"    {name:<50}", end=" ", flush=True)
    res = fn(u_obs, x_grid, cfg, rng)
    status = "OK" if res.completed else f"FAIL@{res.failure_step}"
    print(status)
    return res


def run_regularized_eps(u_obs, x_grid, cfg, epsilon, scale_by_peak=False, rng_seed=0):
    """Run est_det_regularized with the given epsilon setting."""
    new_cfg = patch_config(cfg)
    new_cfg.regularization = make_regularization_cfg(
        enabled=True,
        eps_enabled=(epsilon > 0.0),
        eps_value=epsilon,
        scale_by_peak=scale_by_peak,
        clip_enabled=False,
    )
    return run_single(
        f"est_det_reg eps={epsilon:.1e}",
        run_estimated_score_deterministic_regularized,
        u_obs, x_grid, new_cfg, rng_seed,
    )


def run_regularized_eps_clip(u_obs, x_grid, cfg, epsilon, max_clip, rng_seed=0):
    """Run est_det_regularized with epsilon floor + score clipping."""
    new_cfg = patch_config(cfg)
    new_cfg.regularization = make_regularization_cfg(
        enabled=True,
        eps_enabled=(epsilon > 0.0),
        eps_value=epsilon,
        clip_enabled=True,
        max_abs_score=max_clip,
    )
    return run_single(
        f"est_det_reg eps={epsilon:.1e} clip={max_clip:.0f}",
        run_estimated_score_deterministic_regularized,
        u_obs, x_grid, new_cfg, rng_seed,
    )


def run_spectral(u_obs, x_grid, cfg, noise_delta=None):
    """Run spectral cutoff baseline."""
    label = f"spectral_cutoff nd={noise_delta:.1e}" if noise_delta else "spectral_cutoff (auto)"
    print(f"    {label:<50}", end=" ", flush=True)
    result = spectral_cutoff_inverse(
        u_obs, x_grid, cfg.heat.alpha, cfg.heat.T, noise_delta=noise_delta)
    print("OK (deterministic)")
    return result


def run_tikhonov(u_obs, x_grid, cfg, lam):
    """Run Tikhonov baseline."""
    label = f"tikhonov lam={lam:.1e}"
    print(f"    {label:<50}", end=" ", flush=True)
    result = tikhonov_inverse(u_obs, x_grid, cfg.heat.alpha, cfg.heat.T, lam)
    print("OK (deterministic)")
    return result


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def metrics_to_row(m: MethodMetrics, test: str, run_label: str,
                   epsilon: float = float("nan"),
                   scale_by_peak: bool = False,
                   clip: float = float("nan"),
                   noise_level: float = float("nan"),
                   **extra_cfg) -> dict:
    return {
        "test": test,
        "run_label": run_label,
        "method": m.method_name,
        "method_category": m.method_category,
        "completed": m.completed,
        "failure_step": m.failure_step if m.failure_step is not None else "",
        "failure_msg": m.failure_msg,
        # Regularization settings
        "epsilon": epsilon,
        "scale_by_peak": scale_by_peak,
        "clip": clip,
        "noise_level": noise_level,
        # Accuracy
        "relative_l2": m.relative_l2,
        "l2_error": m.l2_error,
        "linf_error": m.linf_error,
        "peak_ratio": m.peak_ratio,
        "sigma_fit": m.sigma_fit,
        "fit_success": m.fit_success,
        # Consistency
        "forward_consistency_l2": m.forward_consistency_l2,
        "mass_rel_error": m.mass_rel_error,
        # Wasserstein
        "wasserstein": m.wasserstein,
        # Score diagnostics
        "max_abs_score_final": m.max_abs_score_final,
        "max_score_error_L2": m.max_score_error_L2,
        # Regularization diagnostics
        "epsilon_value": m.epsilon_value,
        "n_denom_below_eps_total": m.n_denom_below_eps_total,
        "n_clipped_total": m.n_clipped_total,
        # Ratio
        "particle_rel_L2_over_spectral": m.particle_rel_L2_over_spectral,
        # Timing
        "runtime_seconds": m.runtime_seconds,
        **extra_cfg,
    }


def _rel_l2(candidate, true_u0, dx):
    diff = candidate - true_u0
    denom = float(np.sqrt(dx * np.sum(true_u0 ** 2)))
    return float(np.sqrt(dx * np.sum(diff ** 2))) / denom if denom > 0 else float("nan")


def _compute_wasserstein_safe(candidate, true_u0, x_grid):
    try:
        return compute_wasserstein(candidate, true_u0, x_grid)
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_eps_sweep(rows_by_test: dict, out_dir: Path) -> None:
    """Plot rel_L2 vs epsilon for all tests (log x-scale)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=False)
    axes = axes.flatten()
    test_names = sorted(rows_by_test.keys())

    for ax, tname in zip(axes, test_names):
        rows = rows_by_test[tname]
        eps_rows = [r for r in rows if r["method"] == "estimated_score_deterministic_regularized"]
        if not eps_rows:
            ax.set_title(f"Test {tname} (no regularized data)")
            continue

        eps_vals = [r["epsilon"] for r in eps_rows]
        rel_l2_vals = [r["relative_l2"] for r in eps_rows]

        # Plot epsilon sweep
        valid = [(e, v) for e, v in zip(eps_vals, rel_l2_vals) if math.isfinite(v)]
        if valid:
            eps_v, rl2_v = zip(*valid)
            # Replace epsilon=0 with a small placeholder for log scale
            eps_plot = [max(e, 1e-12) for e in eps_v]
            ax.semilogx(eps_plot, rl2_v, "o-", color="#ff7f00", label="est_det_reg",
                        zorder=5)

        # Horizontal reference lines
        for rr in rows:
            if rr["method"] == "naive_backward" and math.isfinite(rr["relative_l2"]):
                ax.axhline(rr["relative_l2"], color="#e41a1c", ls=":", lw=1.5, label="naive")
                break
        for rr in rows:
            if rr["method"] == "oracle_score_deterministic" and math.isfinite(rr["relative_l2"]):
                ax.axhline(rr["relative_l2"], color="#377eb8", ls=":", lw=1.5, label="oracle_det")
                break
        for rr in rows:
            if rr["method"] == "estimated_score_deterministic_raw" and math.isfinite(rr["relative_l2"]):
                ax.axhline(rr["relative_l2"], color="#984ea3", ls=":", lw=1.5, label="est_raw")
                break
        for rr in rows:
            if rr["method"] == "spectral_cutoff_baseline" and math.isfinite(rr["relative_l2"]):
                ax.axhline(rr["relative_l2"], color="green", ls="--", lw=1.5, label="spectral")
                break

        ax.set_title(f"Test {tname}")
        ax.set_xlabel("epsilon")
        ax.set_ylabel("relative L2")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(len(test_names), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Relative L2 error vs epsilon-floor value", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "relative_L2_vs_epsilon.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_fwd_consistency_sweep(rows_by_test: dict, out_dir: Path) -> None:
    """Plot forward consistency L2 vs epsilon for all tests."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=False)
    axes = axes.flatten()
    test_names = sorted(rows_by_test.keys())

    for ax, tname in zip(axes, test_names):
        rows = rows_by_test[tname]
        eps_rows = [r for r in rows if r["method"] == "estimated_score_deterministic_regularized"]
        if not eps_rows:
            ax.set_visible(False)
            continue

        valid = [(r["epsilon"], r["forward_consistency_l2"]) for r in eps_rows
                 if math.isfinite(r["forward_consistency_l2"])]
        if valid:
            eps_v, fc_v = zip(*valid)
            ax.semilogx([max(e, 1e-12) for e in eps_v], fc_v, "s-", color="#ff7f00",
                        label="est_det_reg")

        for rr in rows:
            if rr["method"] == "spectral_cutoff_baseline" and math.isfinite(rr["forward_consistency_l2"]):
                ax.axhline(rr["forward_consistency_l2"], color="green", ls="--", lw=1.5, label="spectral")
                break
        for rr in rows:
            if rr["method"] == "oracle_score_deterministic" and math.isfinite(rr["forward_consistency_l2"]):
                ax.axhline(rr["forward_consistency_l2"], color="#377eb8", ls=":", lw=1.5, label="oracle_det")
                break

        ax.set_title(f"Test {tname}")
        ax.set_xlabel("epsilon")
        ax.set_ylabel("forward consistency L2")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for i in range(len(test_names), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Forward consistency L2 vs epsilon", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "forward_consistency_vs_epsilon.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_max_score_sweep(rows_by_test: dict, out_dir: Path) -> None:
    """Plot max |score| vs epsilon for all tests."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=False)
    axes = axes.flatten()
    test_names = sorted(rows_by_test.keys())

    for ax, tname in zip(axes, test_names):
        rows = rows_by_test[tname]
        eps_rows = [r for r in rows if r["method"] == "estimated_score_deterministic_regularized"]

        valid = [(r["epsilon"], r["max_abs_score_final"]) for r in eps_rows
                 if math.isfinite(r["max_abs_score_final"])]
        if valid:
            eps_v, ms_v = zip(*valid)
            ax.semilogy([max(e, 1e-12) for e in eps_v], ms_v, "^-", color="#ff7f00",
                        label="est_det_reg")

        ax.set_title(f"Test {tname}")
        ax.set_xlabel("epsilon")
        ax.set_ylabel("max |score| at final step")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for i in range(len(test_names), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Max |score| vs epsilon", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "max_abs_score_vs_epsilon.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_spectral_comparison(summary_rows: list, out_dir: Path) -> None:
    """Grouped bar chart: naive/oracle/raw/best_reg/spectral per test."""
    tests = sorted(set(r["test"] for r in summary_rows if not r["test"].startswith("noisy")))
    methods_order = [
        "naive_backward",
        "oracle_score_deterministic",
        "estimated_score_deterministic_raw",
        "estimated_score_deterministic_regularized",
        "spectral_cutoff_baseline",
        "tikhonov_baseline",
    ]
    colors = {
        "naive_backward": "#e41a1c",
        "oracle_score_deterministic": "#377eb8",
        "estimated_score_deterministic_raw": "#984ea3",
        "estimated_score_deterministic_regularized": "#ff7f00",
        "spectral_cutoff_baseline": "#4daf4a",
        "tikhonov_baseline": "#a65628",
    }
    labels = {
        "naive_backward": "Naive",
        "oracle_score_deterministic": "Oracle det",
        "estimated_score_deterministic_raw": "Est raw",
        "estimated_score_deterministic_regularized": "Est reg (best ε)",
        "spectral_cutoff_baseline": "Spectral cutoff",
        "tikhonov_baseline": "Tikhonov",
    }

    # For each test, pick best regularized (min rel_L2 among completed)
    best_by_test = {}
    for tname in tests:
        test_rows = [r for r in summary_rows if r["test"] == tname]
        reg_rows = [r for r in test_rows
                    if r["method"] == "estimated_score_deterministic_regularized"
                    and r.get("completed", False)
                    and math.isfinite(r["relative_l2"])]
        if reg_rows:
            best = min(reg_rows, key=lambda r: r["relative_l2"])
            best_by_test[tname] = best

    n_tests = len(tests)
    n_methods = len(methods_order)
    bar_w = 0.13
    x = np.arange(n_tests)

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, mname in enumerate(methods_order):
        vals = []
        for tname in tests:
            test_rows = [r for r in summary_rows if r["test"] == tname]
            if mname == "estimated_score_deterministic_regularized":
                # Use best epsilon result
                v = best_by_test.get(tname, {}).get("relative_l2", float("nan"))
            else:
                matched = [r for r in test_rows if r["method"] == mname]
                v = matched[0]["relative_l2"] if matched else float("nan")
            vals.append(v if math.isfinite(v) else 0.0)
        ax.bar(x + i * bar_w, vals, bar_w, label=labels.get(mname, mname),
               color=colors.get(mname, "gray"))

    ax.set_xticks(x + bar_w * (n_methods - 1) / 2)
    ax.set_xticklabels([f"Test {t}" for t in tests])
    ax.set_ylabel("Relative L2 error")
    ax.set_title("Method comparison by test (best regularized epsilon selected)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "spectral_comparison_by_test.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_field_comparison_test(test_name: str, x_grid: np.ndarray,
                                true_u0: np.ndarray, u_obs: np.ndarray,
                                candidates: dict, out_dir: Path,
                                filename: str = "field_comparison.png") -> None:
    """Plot candidate fields vs true_u0 and u_obs for one test."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {
        "naive_backward": "#e41a1c",
        "oracle_score_deterministic": "#377eb8",
        "estimated_score_deterministic_raw": "#984ea3",
        "spectral_cutoff_baseline": "#4daf4a",
        "tikhonov_baseline": "#a65628",
    }

    for ax in axes:
        ax.plot(x_grid, true_u0, "k-", lw=2.5, label="True u₀", zorder=10)
        ax.plot(x_grid, u_obs, "k--", lw=1.5, alpha=0.5, label="Observed u(T)")

    for name, (candidate, label) in candidates.items():
        ls = "-"
        c = colors.get(name, "#555555")
        for ax in axes:
            ax.plot(x_grid, candidate, color=c, lw=1.5, ls=ls, label=label)

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("u(x)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_title(f"Test {test_name}: field comparison (all candidates)")
    axes[1].set_title(f"Test {test_name}: residuals (candidate - true_u0)")
    for name, (candidate, label) in candidates.items():
        c = colors.get(name, "#555555")
        axes[1].plot(x_grid, candidate - true_u0, color=c, lw=1.5, label=label)
    axes[1].axhline(0, color="black", lw=0.8, ls="--")
    for ax in axes:
        ax.legend(fontsize=8)

    fig.suptitle(f"Test {test_name}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_noisy_comparison(test_name: str, noise_rows: list, out_dir: Path) -> None:
    """Summary plot for noisy data: rel_L2 vs noise_level per method."""
    if not noise_rows:
        return
    methods_order = [
        "estimated_score_deterministic_raw",
        "estimated_score_deterministic_regularized",
        "spectral_cutoff_baseline",
    ]
    colors = {
        "estimated_score_deterministic_raw": "#984ea3",
        "estimated_score_deterministic_regularized": "#ff7f00",
        "spectral_cutoff_baseline": "#4daf4a",
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    for mname in methods_order:
        mrows = sorted([r for r in noise_rows if r["method"] == mname],
                       key=lambda r: r["noise_level"])
        if mrows:
            nls = [r["noise_level"] for r in mrows]
            rls = [r["relative_l2"] for r in mrows]
            ax.plot(nls, rls, "o-", color=colors.get(mname, "gray"),
                    label=mname, lw=1.5)
    ax.set_xlabel("noise level (fraction of max)")
    ax.set_ylabel("relative L2 error")
    ax.set_title(f"Test {test_name}: rel_L2 vs noise level")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"noisy_{test_name}_rel_l2_vs_noise.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Rescue / refinement verdict
# ---------------------------------------------------------------------------

def rescue_or_refinement(
    raw_rel_l2: float,
    raw_completed: bool,
    oracle_rel_l2: float,
    best_reg_rel_l2: float,
) -> tuple[str, str]:
    """
    Determine if regularization provides rescue or refinement.

    Rescue:
      - Raw failed (not completed), OR
      - Raw rel_L2 > 1.5 * oracle rel_L2

    Refinement:
      - Raw succeeded and was already close to oracle, but regularized
        reduces error further.

    Returns (verdict, explanation)
    """
    if not raw_completed:
        verb = "RESCUE"
        exp = "raw method failed; regularization rescued the integration"
    elif not math.isnan(raw_rel_l2) and not math.isnan(oracle_rel_l2) and raw_rel_l2 > 1.5 * oracle_rel_l2:
        verb = "RESCUE"
        exp = f"raw rel_L2={raw_rel_l2:.4f} >> oracle rel_L2={oracle_rel_l2:.4f} (ratio {raw_rel_l2/oracle_rel_l2:.2f}x)"
    else:
        verb = "REFINEMENT"
        improvement = (raw_rel_l2 - best_reg_rel_l2) / raw_rel_l2 if (not math.isnan(raw_rel_l2) and raw_rel_l2 > 0) else float("nan")
        pct = f"{100*improvement:.1f}%" if math.isfinite(improvement) else "N/A"
        exp = f"raw already worked (rel_L2={raw_rel_l2:.4f}); regularized reduces by {pct}"
    return verb, exp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run regularization + baseline suite.")
    p.add_argument("--base-config", type=str, default="configs/gaussian_base.yaml")
    p.add_argument("--mixture-config", type=str, default="configs/gaussian_mixture.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    base_cfg = load_config(project_root / args.base_config)
    mixture_cfg = load_config(project_root / args.mixture_config)

    outputs_root = project_root / "outputs"
    outputs_root.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_dir = outputs_root / f"regularization_suite_{ts}"
    suite_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSuite output: {suite_dir}\n")

    tests = build_test_configs(base_cfg, mixture_cfg)
    all_rows: list[dict] = []
    summary_rows: list[dict] = []  # one row per (test, method, best_eps_or_fixed)
    per_test_rows: dict[str, list] = {}  # test -> all epsilon sweep rows

    # -----------------------------------------------------------------------
    # Main test loop
    # -----------------------------------------------------------------------
    for test_name, cfg in tests.items():
        print(f"\n{'='*60}")
        print(f"Test {test_name}  T={cfg.heat.T}  IC={cfg.initial_condition.type}")
        print("="*60)
        test_dir = suite_dir / f"test_{test_name}"
        test_dir.mkdir(exist_ok=True)

        x_grid = make_grid(cfg)
        u0_true = compute_true_u0(x_grid, cfg)
        u_obs = compute_observed_final(x_grid, cfg)
        dx = x_grid[1] - x_grid[0]

        # Reconstruct sanity check
        init_state = field_to_globs(u_obs, x_grid, cfg)
        recon = reconstruct_field(init_state, x_grid)
        recon_err = float(np.sqrt(dx * np.sum((recon - u_obs) ** 2)))
        print(f"  Step-0 recon L2 error: {recon_err:.6e}")

        test_eps_rows = []  # rows for epsilon sweep
        candidates_for_plot = {}  # name -> (candidate_array, label)

        # -------------------------------------------------------------------
        # Reference methods (naive, oracle, raw)
        # -------------------------------------------------------------------
        print("\n  Reference methods:")
        rng_base = base_cfg.grw.rng_seed

        res_naive = run_single("naive_backward", run_naive_backward,
                               u_obs, x_grid, cfg, rng_base)
        m_naive = compute_metrics(res_naive, u0_true, u_obs, x_grid, cfg)
        m_naive.wasserstein = _compute_wasserstein_safe(res_naive.candidate, u0_true, x_grid)
        m_naive.method_category = "particle"
        row_naive = metrics_to_row(m_naive, test_name, "reference")
        summary_rows.append(row_naive)
        candidates_for_plot["naive_backward"] = (res_naive.candidate, "Naive")

        res_oracle = run_single("oracle_score_deterministic", run_oracle_score_deterministic,
                                u_obs, x_grid, cfg, rng_base)
        m_oracle = compute_metrics(res_oracle, u0_true, u_obs, x_grid, cfg)
        m_oracle.wasserstein = _compute_wasserstein_safe(res_oracle.candidate, u0_true, x_grid)
        m_oracle.method_category = "particle"
        row_oracle = metrics_to_row(m_oracle, test_name, "reference")
        summary_rows.append(row_oracle)
        candidates_for_plot["oracle_score_deterministic"] = (res_oracle.candidate, "Oracle det")

        res_raw = run_single("estimated_score_deterministic_raw",
                             run_estimated_score_deterministic_raw,
                             u_obs, x_grid, cfg, rng_base)
        m_raw = compute_metrics(res_raw, u0_true, u_obs, x_grid, cfg)
        m_raw.wasserstein = _compute_wasserstein_safe(res_raw.candidate, u0_true, x_grid)
        m_raw.method_category = "particle"
        row_raw = metrics_to_row(m_raw, test_name, "reference")
        summary_rows.append(row_raw)
        candidates_for_plot["estimated_score_deterministic_raw"] = (res_raw.candidate, "Est raw")

        # -------------------------------------------------------------------
        # Classical baselines
        # -------------------------------------------------------------------
        print("\n  Classical baselines:")

        bl_spectral = run_spectral(u_obs, x_grid, cfg)
        m_spectral = compute_baseline_metrics(
            bl_spectral.candidate, u0_true, u_obs, x_grid, cfg,
            "spectral_cutoff_baseline", "baseline_spectral")
        m_spectral.wasserstein = _compute_wasserstein_safe(bl_spectral.candidate, u0_true, x_grid)
        row_spectral = metrics_to_row(m_spectral, test_name, "baseline",
                                      **{"k_cut": bl_spectral.k_cut,
                                         "n_modes_kept": bl_spectral.n_modes_kept})
        summary_rows.append(row_spectral)
        candidates_for_plot["spectral_cutoff_baseline"] = (bl_spectral.candidate, "Spectral cutoff")

        spectral_rel_l2 = m_spectral.relative_l2

        for lam in TIKHONOV_LAMBDAS:
            bl_tik = run_tikhonov(u_obs, x_grid, cfg, lam)
            m_tik = compute_baseline_metrics(
                bl_tik.candidate, u0_true, u_obs, x_grid, cfg,
                f"tikhonov_baseline_lam{lam:.0e}", "baseline_tikhonov")
            m_tik.wasserstein = _compute_wasserstein_safe(bl_tik.candidate, u0_true, x_grid)
            row_tik = metrics_to_row(m_tik, test_name, "baseline", **{"tikhonov_lam": lam})
            summary_rows.append(row_tik)
            if lam == TIKHONOV_LAMBDAS[0]:
                candidates_for_plot["tikhonov_baseline"] = (bl_tik.candidate, f"Tikhonov λ={lam:.0e}")

        # -------------------------------------------------------------------
        # Epsilon sweep: est_det_regularized
        # -------------------------------------------------------------------
        print("\n  Epsilon sweep (est_det_regularized):")
        eps_results = {}
        for eps in EPSILON_VALUES:
            res_reg = run_regularized_eps(u_obs, x_grid, cfg, eps, rng_seed=rng_base)
            m_reg = compute_metrics(res_reg, u0_true, u_obs, x_grid, cfg)
            m_reg.wasserstein = _compute_wasserstein_safe(res_reg.candidate, u0_true, x_grid)
            m_reg.method_category = "particle"
            m_reg.particle_rel_L2_over_spectral = (
                m_reg.relative_l2 / spectral_rel_l2
                if (math.isfinite(spectral_rel_l2) and spectral_rel_l2 > 0) else float("nan"))
            eps_results[eps] = (res_reg, m_reg)
            row = metrics_to_row(m_reg, test_name, f"eps={eps:.1e}",
                                 epsilon=eps, scale_by_peak=False)
            test_eps_rows.append(row)
            all_rows.append(row)

        # Best epsilon by rel_L2 (completed runs only)
        completed_eps = {eps: m for eps, (res, m) in eps_results.items()
                         if res.completed and math.isfinite(m.relative_l2)}
        if completed_eps:
            best_eps = min(completed_eps, key=lambda e: completed_eps[e].relative_l2)
            best_m_reg = completed_eps[best_eps]
        else:
            best_eps = 0.0
            best_m_reg = m_raw  # fallback

        best_m_reg.particle_rel_L2_over_spectral = (
            best_m_reg.relative_l2 / spectral_rel_l2
            if (math.isfinite(spectral_rel_l2) and spectral_rel_l2 > 0) else float("nan"))

        row_best_reg = metrics_to_row(best_m_reg, test_name, "best_regularized",
                                      epsilon=best_eps)
        summary_rows.append(row_best_reg)
        if best_eps in eps_results:
            candidates_for_plot["estimated_score_deterministic_regularized"] = (
                eps_results[best_eps][0].candidate, f"Est reg (ε={best_eps:.1e})")

        per_test_rows[test_name] = test_eps_rows

        # -------------------------------------------------------------------
        # Clipping sweep (using best epsilon)
        # -------------------------------------------------------------------
        print(f"\n  Clipping sweep (best eps={best_eps:.1e}):")
        for max_clip in CLIP_VALUES:
            res_clip = run_regularized_eps_clip(u_obs, x_grid, cfg, best_eps, max_clip,
                                                rng_seed=rng_base)
            m_clip = compute_metrics(res_clip, u0_true, u_obs, x_grid, cfg)
            m_clip.wasserstein = _compute_wasserstein_safe(res_clip.candidate, u0_true, x_grid)
            m_clip.method_category = "particle"
            row = metrics_to_row(m_clip, test_name, f"clip={max_clip:.0f}",
                                 epsilon=best_eps, clip=max_clip)
            all_rows.append(row)
            summary_rows.append(row)

        # -------------------------------------------------------------------
        # Field comparison plot
        # -------------------------------------------------------------------
        plot_field_comparison_test(
            test_name, x_grid, u0_true, u_obs, candidates_for_plot, test_dir)

        # Save per-test epsilon CSV
        if test_eps_rows:
            pd.DataFrame(test_eps_rows).to_csv(
                test_dir / "epsilon_sweep_metrics.csv", index=False)

        # -------------------------------------------------------------------
        # Rescue / refinement verdict
        # -------------------------------------------------------------------
        verb, exp = rescue_or_refinement(
            raw_rel_l2=m_raw.relative_l2,
            raw_completed=res_raw.completed,
            oracle_rel_l2=m_oracle.relative_l2,
            best_reg_rel_l2=best_m_reg.relative_l2,
        )
        n_below_eps_total = sum(eps_results[best_eps][0].n_denominator_below_epsilon) if best_eps in eps_results else 0
        n_clipped_total = sum(eps_results[best_eps][0].n_clipped_scores) if best_eps in eps_results else 0

        print(f"\n  >> Test {test_name} verdict: [{verb}] {exp}")
        print(f"     best eps={best_eps:.1e}  reg_rel_L2={best_m_reg.relative_l2:.4f}  "
              f"spectral_rel_L2={spectral_rel_l2:.4f}  "
              f"ratio={best_m_reg.particle_rel_L2_over_spectral:.3f}")
        print(f"     safeguards: n_below_eps={n_below_eps_total}  n_clipped={n_clipped_total}")

    # -----------------------------------------------------------------------
    # Noisy data experiments (Tests B and H)
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("NOISY DATA EXPERIMENTS (Tests B and H)")
    print("="*60)

    noisy_dir = suite_dir / "noisy"
    noisy_dir.mkdir(exist_ok=True)
    noisy_all_rows: list[dict] = []

    for test_name in ["B", "H"]:
        cfg = tests[test_name]
        x_grid = make_grid(cfg)
        u0_true = compute_true_u0(x_grid, cfg)
        u_obs_clean = compute_observed_final(x_grid, cfg)
        dx = x_grid[1] - x_grid[0]

        # Determine best epsilon for this test (from clean run)
        best_eps_for_test = min(
            (r["epsilon"] for r in per_test_rows.get(test_name, [])
             if r["method"] == "estimated_score_deterministic_regularized"
             and r.get("completed", False)
             and math.isfinite(r["relative_l2"])),
            key=lambda e: next(
                r["relative_l2"] for r in per_test_rows[test_name]
                if r["epsilon"] == e and r["method"] == "estimated_score_deterministic_regularized"),
            default=1e-6,
        )

        noise_rows_for_test = []
        rng_noise = np.random.default_rng(42)
        for noise_level in NOISE_LEVELS:
            print(f"\n  Test {test_name} noise_level={noise_level}")
            u_obs_noisy = u_obs_clean + noise_level * np.max(u_obs_clean) * rng_noise.standard_normal(len(u_obs_clean))
            # Note: may become slightly negative at tails — record but do NOT clip

            n_negative = int(np.sum(u_obs_noisy < 0))
            if n_negative > 0:
                print(f"    WARNING: {n_negative} grid points have u_obs_noisy < 0 (not clipped)")

            rng_base = base_cfg.grw.rng_seed

            # Raw
            res_raw_n = run_single("estimated_score_deterministic_raw",
                                   run_estimated_score_deterministic_raw,
                                   u_obs_noisy, x_grid, cfg, rng_base)
            m_raw_n = compute_metrics(res_raw_n, u0_true, u_obs_noisy, x_grid, cfg)
            m_raw_n.wasserstein = _compute_wasserstein_safe(res_raw_n.candidate, u0_true, x_grid)
            m_raw_n.method_category = "particle"
            row = metrics_to_row(m_raw_n, test_name, f"noisy_raw", noise_level=noise_level)
            noisy_all_rows.append(row)
            noise_rows_for_test.append(row)
            all_rows.append(row)

            # Regularized (best epsilon)
            res_reg_n = run_regularized_eps(u_obs_noisy, x_grid, cfg, best_eps_for_test,
                                            rng_seed=rng_base)
            m_reg_n = compute_metrics(res_reg_n, u0_true, u_obs_noisy, x_grid, cfg)
            m_reg_n.wasserstein = _compute_wasserstein_safe(res_reg_n.candidate, u0_true, x_grid)
            m_reg_n.method_category = "particle"
            row = metrics_to_row(m_reg_n, test_name, f"noisy_reg", noise_level=noise_level,
                                 epsilon=best_eps_for_test)
            noisy_all_rows.append(row)
            noise_rows_for_test.append(row)
            all_rows.append(row)

            # Spectral with noise_delta = noise_level
            bl_n = run_spectral(u_obs_noisy, x_grid, cfg, noise_delta=noise_level)
            m_sp_n = compute_baseline_metrics(
                bl_n.candidate, u0_true, u_obs_noisy, x_grid, cfg,
                "spectral_cutoff_baseline", "baseline_spectral")
            m_sp_n.wasserstein = _compute_wasserstein_safe(bl_n.candidate, u0_true, x_grid)
            row = metrics_to_row(m_sp_n, test_name, f"noisy_spectral", noise_level=noise_level)
            noisy_all_rows.append(row)
            noise_rows_for_test.append(row)
            all_rows.append(row)

        plot_noisy_comparison(test_name, noise_rows_for_test, noisy_dir)

    # -----------------------------------------------------------------------
    # Save aggregate CSVs
    # -----------------------------------------------------------------------
    agg_df = pd.DataFrame(all_rows)
    agg_df.to_csv(suite_dir / "aggregate_regularization_metrics.csv", index=False)
    print(f"\nAggregate CSV saved: {suite_dir/'aggregate_regularization_metrics.csv'}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(suite_dir / "summary_by_test_method.csv", index=False)

    noisy_df = pd.DataFrame(noisy_all_rows)
    noisy_df.to_csv(noisy_dir / "noisy_metrics.csv", index=False)

    # -----------------------------------------------------------------------
    # Aggregate plots
    # -----------------------------------------------------------------------
    print("\nGenerating aggregate plots ...")
    plot_eps_sweep(per_test_rows, suite_dir)
    plot_fwd_consistency_sweep(per_test_rows, suite_dir)
    plot_max_score_sweep(per_test_rows, suite_dir)
    plot_spectral_comparison(summary_rows, suite_dir)

    # -----------------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("REGULARIZATION SUITE SUMMARY")
    print("="*70)
    header = (f"{'Test':<6} {'Method':<45} {'Status':<12} {'rel_L2':>8} "
              f"{'fwd_cons':>9} {'wasserstein':>12} {'rt(s)':>7}")
    print(header)
    print("-" * len(header))

    ref_methods = [
        "naive_backward",
        "oracle_score_deterministic",
        "estimated_score_deterministic_raw",
        "estimated_score_deterministic_regularized",
        "spectral_cutoff_baseline",
        "tikhonov_baseline_lam1e-06",
    ]
    for tname in sorted(tests.keys()):
        for mname in ref_methods:
            matched = [r for r in summary_rows if r["test"] == tname and r["method"] == mname]
            if not matched:
                continue
            r = matched[0]
            status = "OK" if r.get("completed", True) else f"FAIL@{r.get('failure_step','?')}"
            rl2 = r.get("relative_l2", float("nan"))
            fc = r.get("forward_consistency_l2", float("nan"))
            ws = r.get("wasserstein", float("nan"))
            rt = r.get("runtime_seconds", float("nan"))
            print(f"{tname:<6} {mname:<45} {status:<12} "
                  f"{rl2:>8.4f} {fc:>9.5f} {ws:>12.6f} {rt:>7.3f}")
        print()

    print("\nRESCUE / REFINEMENT VERDICTS:")
    for tname in sorted(tests.keys()):
        cfg = tests[tname]
        raw_rows = [r for r in summary_rows
                    if r["test"] == tname and r["method"] == "estimated_score_deterministic_raw"]
        oracle_rows = [r for r in summary_rows
                       if r["test"] == tname and r["method"] == "oracle_score_deterministic"]
        reg_rows = [r for r in summary_rows
                    if r["test"] == tname and r["method"] == "estimated_score_deterministic_regularized"
                    and r["run_label"] == "best_regularized"]
        if not raw_rows:
            continue
        raw_rl2 = raw_rows[0].get("relative_l2", float("nan"))
        raw_ok = raw_rows[0].get("completed", True)
        oracle_rl2 = oracle_rows[0].get("relative_l2", float("nan")) if oracle_rows else float("nan")
        best_rl2 = reg_rows[0].get("relative_l2", float("nan")) if reg_rows else float("nan")
        best_eps = reg_rows[0].get("epsilon", float("nan")) if reg_rows else float("nan")
        verb, exp = rescue_or_refinement(raw_rl2, raw_ok, oracle_rl2, best_rl2)
        spectral_rl2 = next((r.get("relative_l2", float("nan")) for r in summary_rows
                              if r["test"] == tname and r["method"] == "spectral_cutoff_baseline"),
                             float("nan"))
        ratio_over_spectral = best_rl2 / spectral_rl2 if math.isfinite(spectral_rl2) and spectral_rl2 > 0 else float("nan")
        print(f"  {tname}: [{verb}]  best_eps={best_eps:.1e}  "
              f"raw={raw_rl2:.4f}  reg={best_rl2:.4f}  oracle={oracle_rl2:.4f}  "
              f"spectral={spectral_rl2:.4f}  reg/spectral={ratio_over_spectral:.3f}")
        print(f"       {exp}")

    # Write summary text file
    summary_lines = ["REGULARIZATION SUITE SUMMARY", "=" * 70, ""]
    summary_lines.append(f"Suite dir: {suite_dir}")
    summary_lines.append(f"Tests run: {', '.join(sorted(tests.keys()))}")
    summary_lines.append(f"Epsilon values: {EPSILON_VALUES}")
    summary_lines.append(f"Clip values: {CLIP_VALUES}")
    summary_lines.append(f"Noise levels (Tests B, H): {NOISE_LEVELS}")
    summary_lines.append("")
    summary_lines.append("See aggregate_regularization_metrics.csv for full data.")
    summary_lines.append("See regularization_suite_plots for figures.")
    (suite_dir / "aggregate_regularization_summary.txt").write_text("\n".join(summary_lines))

    print(f"\nAll outputs written to: {suite_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
