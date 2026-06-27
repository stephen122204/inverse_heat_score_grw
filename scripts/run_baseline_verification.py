"""
run_baseline_verification.py — Diagnostic verification pass before trusting Phase 5 conclusions.

Two concerns addressed:

PART 1 — Raw vs eps=0 path equivalence
  Compares estimated_score_deterministic_raw vs estimated_score_deterministic_regularized
  (with eps=0, clipping off) and the stochastic counterparts.  Both paths should
  yield identical trajectories if the code paths are truly equivalent — they are NOT
  expected to be identical because the two code paths differ in interpolation order:

    Raw path  : interpolate(u) → interp(u_x) → divide at particle positions
    Reg path  : divide on grid (u_x / (u+0)) → interpolate score to positions

  This is a known structural difference.  PART 1 quantifies the divergence and
  identifies which factor is dominant.

PART 2 — Tuned baseline sweep
  The Phase 5 suite used default noise_delta=1e-8 for spectral cutoff, which may be
  over-conservative.  PART 2 sweeps noise_delta and Tikhonov lambda to find the
  accuracy ceiling of classical baselines before claiming particle GRW is competitive.

Usage:
    python scripts/run_baseline_verification.py \\
        --base-config configs/gaussian_base.yaml \\
        --mixture-config configs/gaussian_mixture.yaml

Outputs (all under outputs/baseline_verification_TIMESTAMP/):
  PART 1:
    path_equivalence_metrics.csv
    path_equivalence_summary.txt
    path_equivalence_diff_{test}.png  (one per test)

  PART 2:
    baseline_sweep_metrics.csv
    baseline_sweep_summary.txt
    best_baseline_comparison.csv
    rel_L2_vs_noise_delta_ALL.png
    rel_L2_vs_tikhonov_lam_ALL.png
    rel_L2_vs_kcut_mult_ALL.png
    best_baseline_vs_particle_comparison.png
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
import warnings

from invheat_grw.config import (
    load_config, Config, RegularizationConfig,
    EpsilonFloorConfig, ScoreClippingConfig, SmoothingConfig,
)
from invheat_grw.fields import (
    make_grid,
    true_u0 as compute_true_u0,
    observed_final as compute_observed_final,
)
from invheat_grw.globs import field_to_globs, reconstruct_field
from invheat_grw.methods import (
    run_estimated_score_deterministic_raw,
    run_estimated_score_stochastic_raw,
    run_estimated_score_deterministic_regularized,
    run_estimated_score_stochastic_regularized,
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

PARTICLE_SEED = 42
TEST_NAMES = ["A", "B", "D", "G", "H", "Z"]

# PART 1 sweeps
PART1_SEED = 42  # deterministic seed for exact comparison

# PART 2 sweeps
NOISE_DELTA_VALUES = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 0.1, 0.5]
K_CUT_MULT_VALUES  = [0.5, 1.0, 2.0, 5.0, 10.0]
TIKHONOV_LAMBDAS   = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]


# ---------------------------------------------------------------------------
# Config helpers (duplicated from run_regularization_suite.py for self-containment)
# ---------------------------------------------------------------------------

def patch_config(cfg: Config, **overrides) -> Config:
    """Deep-copy cfg and apply nested dot-notation overrides."""
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
    max_abs_score: float = 1e12,
) -> RegularizationConfig:
    return RegularizationConfig(
        enabled=enabled,
        epsilon_floor=EpsilonFloorConfig(enabled=eps_enabled, value=eps_value, scale_by_peak=scale_by_peak),
        score_clipping=ScoreClippingConfig(enabled=clip_enabled, max_abs_score=max_abs_score),
        smoothing=SmoothingConfig(enabled=False),
    )


def eps0_regularization_cfg() -> RegularizationConfig:
    """eps=0, clipping off, smoothing off — closest possible to raw path via reg code."""
    return make_regularization_cfg(enabled=True, eps_enabled=True, eps_value=0.0,
                                   scale_by_peak=False, clip_enabled=False)


# ---------------------------------------------------------------------------
# Test definitions (mirrors run_regularization_suite.py)
# ---------------------------------------------------------------------------

def build_test_configs(base_cfg: Config, mixture_cfg: Config) -> dict[str, Config]:
    tests: dict[str, Config] = {}

    tests["A"] = patch_config(base_cfg, **{"heat.T": 0.05, "initial_condition.sigma0": 0.08})
    tests["B"] = patch_config(base_cfg, **{"heat.T": 0.15, "initial_condition.sigma0": 0.08})
    tests["D"] = patch_config(base_cfg, **{
        "heat.T": 0.05,
        "initial_condition.sigma0": 0.05,
        "grw.gradient_globs_per_jump": 40,
    })
    tests["G"] = patch_config(mixture_cfg, **{"heat.T": 0.05})
    tests["H"] = patch_config(mixture_cfg, **{"heat.T": 0.15})
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
# Metric extraction helpers
# ---------------------------------------------------------------------------

def result_to_particle_metrics(
    result: MethodResult,
    true_u: np.ndarray,
    obs_final: np.ndarray,
    x_grid: np.ndarray,
    cfg: Config,
) -> dict:
    """Extract a flat metrics dict from a MethodResult."""
    dx = x_grid[1] - x_grid[0]
    m = compute_metrics(result, true_u, obs_final, x_grid, cfg)
    w = compute_wasserstein(result.candidate, true_u, x_grid)

    # Count NaN-containing score-error entries (proxy for nonfinite score steps)
    score_nan_steps = sum(1 for v in result.score_L2_error_vs_oracle if not np.isfinite(v))

    # Max abs score over all steps (over finite entries)
    finite_max = [v for v in result.score_max_abs if np.isfinite(v)]
    max_abs_score_all = float(max(finite_max)) if finite_max else float("nan")

    # Max score error vs oracle (L2, over steps)
    finite_err = [v for v in result.score_L2_error_vs_oracle if np.isfinite(v)]
    max_score_err = float(max(finite_err)) if finite_err else float("nan")

    # n_denominator_below_epsilon total (0 for raw path)
    n_denom_below = int(sum(getattr(result, "n_denominator_below_epsilon", [])))
    n_clipped = int(sum(getattr(result, "n_clipped_scores", [])))
    eps_used = float(getattr(result, "epsilon_used", 0.0))

    return {
        "completed": result.completed,
        "failure_step": result.failure_step,
        "failure_reason": result.failure_msg,
        "rel_L2": m.relative_l2,
        "L2_h": m.l2_error,
        "Linf": m.linf_error,
        "forward_consistency_L2": m.forward_consistency_l2,
        "max_abs_score_all_steps": max_abs_score_all,
        "max_score_error_L2": max_score_err,
        "n_score_nan_steps": score_nan_steps,
        "n_denom_below_epsilon_total": n_denom_below,
        "n_clipped_scores_total": n_clipped,
        "epsilon_used": eps_used,
        "wasserstein": w,
    }


def field_diff_metrics(
    raw_cand: np.ndarray,
    reg_cand: np.ndarray,
    x_grid: np.ndarray,
) -> dict:
    """Compute L2, Linf, max-abs-pointwise difference between two field arrays."""
    dx = x_grid[1] - x_grid[0]
    diff = raw_cand - reg_cand
    l2_diff = float(np.sqrt(dx * np.sum(diff ** 2)))
    linf_diff = float(np.max(np.abs(diff)))
    l2_raw = float(np.sqrt(dx * np.sum(raw_cand ** 2)))
    rel_diff = l2_diff / l2_raw if l2_raw > 0.0 else float("nan")
    return {
        "L2_field_diff": l2_diff,
        "Linf_field_diff": linf_diff,
        "rel_field_diff": rel_diff,
    }


def pass_fail_criterion(raw_d: dict, reg_d: dict, diff_d: dict) -> tuple[str, str]:
    """
    PASS: both complete/fail at same step+reason AND rel_field_diff < 1e-10
    FAIL: any divergence → identify root cause
    Returns (verdict, diagnosis)
    """
    # Check completion parity
    same_complete = (raw_d["completed"] == reg_d["completed"])
    same_fail_step = (raw_d["failure_step"] == reg_d["failure_step"])

    rel_diff = diff_d["rel_field_diff"]
    is_tiny = (np.isfinite(rel_diff) and rel_diff < 1e-10)

    if same_complete and same_fail_step and is_tiny:
        return "PASS", "Paths are bit-equivalent (rel_diff < 1e-10)"

    # Diagnose cause(s)
    causes = []

    if not same_complete:
        causes.append(
            f"completion_status_differs: raw={raw_d['completed']} vs reg={reg_d['completed']}"
        )

    if not same_fail_step:
        causes.append(
            f"failure_step_differs: raw={raw_d['failure_step']} vs reg={reg_d['failure_step']}"
        )

    if not is_tiny:
        causes.append(
            f"STRUCTURAL: interpolation_order_differs — raw path divides u_x_pos/u_pos "
            f"(interp-then-divide) while reg path divides u_x_grid/u_grid then interps "
            f"(divide-then-interp); rel_diff={rel_diff:.3e} >> 1e-10"
        )
        # Further distinguish: if both completed, difference is pure interpolation artifact
        if raw_d["completed"] and reg_d["completed"]:
            causes.append(
                f"both_completed: difference is purely due to operation ordering "
                f"(L2_diff={diff_d['L2_field_diff']:.3e})"
            )
        else:
            causes.append(
                f"failure_divergence: one or both failed — nonfinite score detection "
                f"may trigger differently because zero-denom check is at different level "
                f"(raw checks u_at_positions; reg checks u_on_grid)"
            )

    # Check if eps0 regularization is accidentally active
    if reg_d["n_denom_below_epsilon_total"] > 0:
        causes.append(
            f"eps_floor_triggered: n_denom_below_epsilon={reg_d['n_denom_below_epsilon_total']} "
            f"even with eps=0 (eps>0 counting uses eps>0 guard, so this should be 0)"
        )

    if reg_d["n_clipped_scores_total"] > 0:
        causes.append(f"clipping_unexpectedly_active: n_clipped={reg_d['n_clipped_scores_total']}")

    return "FAIL", "; ".join(causes)


# ---------------------------------------------------------------------------
# PART 1: Path equivalence
# ---------------------------------------------------------------------------

def run_part1(
    tests: dict[str, Config],
    out_dir: Path,
) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("PART 1 — Raw vs eps=0 Regularized Path Equivalence")
    print("=" * 70)
    print("NOTE: Paths are expected to DIFFER structurally (interp order).")
    print("This pass quantifies the divergence magnitude.\n")

    rows = []

    for test_name, cfg in tests.items():
        x_grid = make_grid(cfg)
        true_u = compute_true_u0(x_grid, cfg)
        obs_final = compute_observed_final(x_grid, cfg)
        dx = x_grid[1] - x_grid[0]

        # Build eps=0 regularized config
        cfg_eps0 = patch_config(cfg)
        cfg_eps0.regularization = eps0_regularization_cfg()

        print(f"\n  Test {test_name} (T={cfg.heat.T}, alpha={cfg.heat.alpha})")
        print(f"  Grid: N={len(x_grid)}, n_steps={cfg.n_steps}")

        for variant, run_raw_fn, run_reg_fn, use_noise in [
            ("det", run_estimated_score_deterministic_raw,
             run_estimated_score_deterministic_regularized, False),
            ("stoc", run_estimated_score_stochastic_raw,
             run_estimated_score_stochastic_regularized, True),
        ]:
            # Same RNG seed for both — stochastic path uses noise, so seeds must match
            rng_raw = np.random.default_rng(PART1_SEED)
            rng_reg = np.random.default_rng(PART1_SEED)

            print(f"    [{variant}] raw  ...", end=" ", flush=True)
            raw_res = run_raw_fn(obs_final, x_grid, cfg, rng_raw)
            print("OK" if raw_res.completed else f"FAIL@{raw_res.failure_step}")

            print(f"    [{variant}] eps0 ...", end=" ", flush=True)
            reg_res = run_reg_fn(obs_final, x_grid, cfg_eps0, rng_reg)
            print("OK" if reg_res.completed else f"FAIL@{reg_res.failure_step}")

            raw_d = result_to_particle_metrics(raw_res, true_u, obs_final, x_grid, cfg)
            reg_d = result_to_particle_metrics(reg_res, true_u, obs_final, x_grid, cfg)
            diff_d = field_diff_metrics(raw_res.candidate, reg_res.candidate, x_grid)

            verdict, diagnosis = pass_fail_criterion(raw_d, reg_d, diff_d)
            print(f"    [{variant}] verdict: {verdict} — {diagnosis[:80]}")

            # Build CSV row for raw
            row_base = {
                "test": test_name, "variant": variant,
                "T": cfg.heat.T, "alpha": cfg.heat.alpha, "n_steps": cfg.n_steps,
            }
            for key, val in raw_d.items():
                rows.append({**row_base, "path": "raw", "metric": key, "value": val})
            for key, val in reg_d.items():
                rows.append({**row_base, "path": "eps0_reg", "metric": key, "value": val})
            for key, val in diff_d.items():
                rows.append({**row_base, "path": "diff", "metric": key, "value": val})
            rows.append({**row_base, "path": "verdict", "metric": "verdict", "value": verdict})
            rows.append({**row_base, "path": "verdict", "metric": "diagnosis", "value": diagnosis})

            # Plot for deterministic variant
            if variant == "det":
                _plot_field_diff(
                    test_name, x_grid, true_u, obs_final,
                    raw_res.candidate, reg_res.candidate,
                    raw_d, reg_d, diff_d, verdict,
                    out_dir / f"path_equivalence_diff_{test_name}.png",
                )

    df = pd.DataFrame(rows)
    csv_path = out_dir / "path_equivalence_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    _write_part1_summary(df, out_dir / "path_equivalence_summary.txt")
    return df


def _plot_field_diff(
    test_name: str,
    x_grid: np.ndarray,
    true_u: np.ndarray,
    obs_final: np.ndarray,
    raw_cand: np.ndarray,
    reg_cand: np.ndarray,
    raw_d: dict,
    reg_d: dict,
    diff_d: dict,
    verdict: str,
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Pre-compute status strings (avoid backslash in f-string for Python < 3.12)
    raw_status = "OK" if raw_d["completed"] else f"FAIL@{raw_d['failure_step']}"
    reg_status = "OK" if reg_d["completed"] else f"FAIL@{reg_d['failure_step']}"

    # Panel 1: raw candidate
    ax = axes[0]
    ax.plot(x_grid, true_u, "k--", lw=1.5, label="true u0")
    ax.plot(x_grid, obs_final, "gray", lw=1, alpha=0.6, label="observed (T)")
    ax.plot(x_grid, raw_cand, "b-", lw=1.5, label="raw candidate")
    ax.set_title(f"Test {test_name} — raw\nrel_L2={raw_d['rel_L2']:.4f}, {raw_status}")
    ax.legend(fontsize=7)
    ax.set_xlabel("x")

    # Panel 2: eps=0 regularized candidate
    ax = axes[1]
    ax.plot(x_grid, true_u, "k--", lw=1.5, label="true u0")
    ax.plot(x_grid, obs_final, "gray", lw=1, alpha=0.6, label="observed (T)")
    ax.plot(x_grid, reg_cand, "r-", lw=1.5, label="eps=0 reg candidate")
    ax.set_title(f"Test {test_name} — eps=0 reg\nrel_L2={reg_d['rel_L2']:.4f}, {reg_status}")
    ax.legend(fontsize=7)
    ax.set_xlabel("x")

    # Panel 3: difference raw - eps0
    diff = raw_cand - reg_cand
    ax = axes[2]
    ax.plot(x_grid, diff, "m-", lw=1.5)
    ax.axhline(0.0, color="k", lw=0.8, ls="--")
    ax.set_title(f"raw − eps0  [{verdict}]\n"
                 f"L2={diff_d['L2_field_diff']:.3e}, "
                 f"rel={diff_d['rel_field_diff']:.3e}, "
                 f"Linf={diff_d['Linf_field_diff']:.3e}")
    ax.set_xlabel("x")

    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {save_path.name}")


def _write_part1_summary(df: pd.DataFrame, save_path: Path) -> None:
    lines = [
        "PART 1 — Raw vs eps=0 Path Equivalence Summary",
        "=" * 70,
        "",
        "STRUCTURAL NOTE:",
        "  Raw path  : interp(u) → interp(u_x) → divide at particle positions",
        "  Reg path  : divide on grid (u_x / (u+0)) → interp score to positions",
        "  These operations do not commute.  Bit-identical results are impossible",
        "  unless the field is constant or u is identically zero everywhere.",
        "",
        "PASS criterion: both complete/fail at same step AND rel_field_diff < 1e-10",
        "FAIL criterion: any divergence (expected for structural interpolation order diff)",
        "",
    ]

    # Extract verdicts
    verdict_rows = df[df["metric"] == "verdict"]
    diag_rows = df[df["metric"] == "diagnosis"]

    header = f"{'Test':<6} {'Var':<6} {'Verdict':<8} {'rel_L2_raw':>12} {'rel_L2_eps0':>12} {'rel_field_diff':>16}"
    lines.append(header)
    lines.append("-" * len(header))

    for test in TEST_NAMES:
        for variant in ["det", "stoc"]:
            v_mask = (verdict_rows["test"] == test) & (verdict_rows["variant"] == variant)
            if not v_mask.any():
                continue
            verdict = str(verdict_rows[v_mask]["value"].values[0])

            def get_val(path, metric):
                m = df[(df["test"] == test) & (df["variant"] == variant) &
                       (df["path"] == path) & (df["metric"] == metric)]
                if m.empty:
                    return float("nan")
                v = m["value"].values[0]
                try:
                    return float(v)
                except (ValueError, TypeError):
                    return float("nan")

            rel_raw  = get_val("raw", "rel_L2")
            rel_reg  = get_val("eps0_reg", "rel_L2")
            rel_diff = get_val("diff", "rel_field_diff")

            lines.append(
                f"{test:<6} {variant:<6} {verdict:<8} "
                f"{rel_raw:>12.4f} {rel_reg:>12.4f} {rel_diff:>16.3e}"
            )

    lines.append("")
    lines.append("Divergence diagnoses:")
    lines.append("-" * 70)
    for test in TEST_NAMES:
        for variant in ["det", "stoc"]:
            d_mask = (diag_rows["test"] == test) & (diag_rows["variant"] == variant)
            if not d_mask.any():
                continue
            diag = str(diag_rows[d_mask]["value"].values[0])
            lines.append(f"  [{test}/{variant}]: {diag}")
    lines.append("")

    text = "\n".join(lines)
    save_path.write_text(text)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# PART 2: Tuned baseline sweep
# ---------------------------------------------------------------------------

def run_part2(
    tests: dict[str, Config],
    out_dir: Path,
) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("PART 2 — Tuned Baseline Sweep")
    print("=" * 70)

    sweep_rows = []

    # Step 1: Run particle methods for comparison
    print("\n  [Particle reference methods]")
    particle_results: dict[str, dict] = {}  # test -> {raw_rel_L2, raw_completed, ...}
    for test_name, cfg in tests.items():
        x_grid = make_grid(cfg)
        true_u = compute_true_u0(x_grid, cfg)
        obs_final = compute_observed_final(x_grid, cfg)

        rng = np.random.default_rng(PARTICLE_SEED)
        print(f"    [det_raw] test={test_name} ...", end=" ", flush=True)
        res = run_estimated_score_deterministic_raw(obs_final, x_grid, cfg, rng)
        print("OK" if res.completed else f"FAIL@{res.failure_step}")
        m = result_to_particle_metrics(res, true_u, obs_final, x_grid, cfg)
        w = compute_wasserstein(res.candidate, true_u, x_grid)
        particle_results[test_name] = {
            "det_raw_rel_L2": m["rel_L2"],
            "det_raw_Linf": m["Linf"],
            "det_raw_fwd_L2": m["forward_consistency_L2"],
            "det_raw_wasserstein": w,
            "det_raw_completed": res.completed,
            "det_raw_failure_step": res.failure_step,
        }

        # Add to sweep rows
        sweep_rows.append({
            "test": test_name, "method_type": "particle_det_raw",
            "param_type": "none", "param_value": float("nan"),
            "rel_L2": m["rel_L2"], "Linf": m["Linf"],
            "forward_consistency_L2": m["forward_consistency_L2"],
            "wasserstein": w,
            "completed": res.completed,
            "T": cfg.heat.T, "alpha": cfg.heat.alpha,
        })

    # Step 2: Spectral cutoff — noise_delta sweep
    print("\n  [Spectral cutoff — noise_delta sweep]")
    for test_name, cfg in tests.items():
        x_grid = make_grid(cfg)
        true_u = compute_true_u0(x_grid, cfg)
        obs_final = compute_observed_final(x_grid, cfg)

        for nd in NOISE_DELTA_VALUES:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                sc_res = spectral_cutoff_inverse(
                    obs_final, x_grid, cfg.heat.alpha, cfg.heat.T, noise_delta=nd)
            m = compute_baseline_metrics(
                sc_res.candidate, true_u, obs_final, x_grid, cfg,
                method_name=f"spectral_cutoff_nd={nd:.0e}",
                method_category="baseline_spectral",
            )
            w = compute_wasserstein(sc_res.candidate, true_u, x_grid)
            sweep_rows.append({
                "test": test_name, "method_type": "spectral_noise_delta",
                "param_type": "noise_delta", "param_value": nd,
                "rel_L2": m.relative_l2, "Linf": m.linf_error,
                "forward_consistency_L2": m.forward_consistency_l2,
                "wasserstein": w,
                "n_modes_kept": sc_res.n_modes_kept,
                "k_cut": sc_res.k_cut,
                "max_inv_mult": sc_res.max_inverse_multiplier,
                "completed": True,
                "T": cfg.heat.T, "alpha": cfg.heat.alpha,
            })

        print(f"    test={test_name}: noise_delta sweep done ({len(NOISE_DELTA_VALUES)} pts)")

    # Step 3: Spectral cutoff — k_cut_mult sweep (relative to noise_delta=1e-8 optimal)
    print("\n  [Spectral cutoff — k_cut_mult sweep]")
    for test_name, cfg in tests.items():
        x_grid = make_grid(cfg)
        true_u = compute_true_u0(x_grid, cfg)
        obs_final = compute_observed_final(x_grid, cfg)

        alpha = cfg.heat.alpha
        T_val = cfg.heat.T
        # Compute k_cut_opt from noise_delta=1e-8
        if alpha * T_val > 0:
            k_cut_opt = math.sqrt(math.log(1.0 / 1e-8) / (alpha * T_val))
        else:
            k_cut_opt = float("inf")

        for mult in K_CUT_MULT_VALUES:
            k_cut_now = mult * k_cut_opt
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                sc_res = spectral_cutoff_inverse(
                    obs_final, x_grid, alpha, T_val, k_cut=k_cut_now)
            m = compute_baseline_metrics(
                sc_res.candidate, true_u, obs_final, x_grid, cfg,
                method_name=f"spectral_cutoff_kcut_mult={mult}",
                method_category="baseline_spectral",
            )
            w = compute_wasserstein(sc_res.candidate, true_u, x_grid)
            sweep_rows.append({
                "test": test_name, "method_type": "spectral_kcut_mult",
                "param_type": "k_cut_mult", "param_value": mult,
                "rel_L2": m.relative_l2, "Linf": m.linf_error,
                "forward_consistency_L2": m.forward_consistency_l2,
                "wasserstein": w,
                "n_modes_kept": sc_res.n_modes_kept,
                "k_cut": sc_res.k_cut,
                "k_cut_opt": k_cut_opt,
                "max_inv_mult": sc_res.max_inverse_multiplier,
                "completed": True,
                "T": cfg.heat.T, "alpha": cfg.heat.alpha,
            })

        print(f"    test={test_name}: k_cut_mult sweep done ({len(K_CUT_MULT_VALUES)} pts)")

    # Step 4: Tikhonov — lambda sweep
    print("\n  [Tikhonov — lambda sweep]")
    for test_name, cfg in tests.items():
        x_grid = make_grid(cfg)
        true_u = compute_true_u0(x_grid, cfg)
        obs_final = compute_observed_final(x_grid, cfg)

        for lam in TIKHONOV_LAMBDAS:
            tik_res = tikhonov_inverse(obs_final, x_grid, cfg.heat.alpha, cfg.heat.T, lam)
            m = compute_baseline_metrics(
                tik_res.candidate, true_u, obs_final, x_grid, cfg,
                method_name=f"tikhonov_lam={lam:.0e}",
                method_category="baseline_tikhonov",
            )
            w = compute_wasserstein(tik_res.candidate, true_u, x_grid)
            sweep_rows.append({
                "test": test_name, "method_type": "tikhonov",
                "param_type": "lambda", "param_value": lam,
                "rel_L2": m.relative_l2, "Linf": m.linf_error,
                "forward_consistency_L2": m.forward_consistency_l2,
                "wasserstein": w,
                "completed": True,
                "T": cfg.heat.T, "alpha": cfg.heat.alpha,
            })

        print(f"    test={test_name}: Tikhonov sweep done ({len(TIKHONOV_LAMBDAS)} pts)")

    df = pd.DataFrame(sweep_rows)

    # Fill missing columns with NaN
    for col in ["n_modes_kept", "k_cut", "k_cut_opt", "max_inv_mult"]:
        if col not in df.columns:
            df[col] = float("nan")

    csv_path = out_dir / "baseline_sweep_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    # Build best-baseline comparison
    best_df = _build_best_comparison(df, particle_results, tests, out_dir)

    # Write summary
    _write_part2_summary(df, best_df, particle_results, out_dir / "baseline_sweep_summary.txt")

    # Plots
    _plot_spectral_noise_delta(df, tests, out_dir / "rel_L2_vs_noise_delta_ALL.png")
    _plot_tikhonov_lambda(df, tests, out_dir / "rel_L2_vs_tikhonov_lam_ALL.png")
    _plot_kcut_mult(df, tests, out_dir / "rel_L2_vs_kcut_mult_ALL.png")
    _plot_best_comparison(best_df, out_dir / "best_baseline_vs_particle_comparison.png")

    return df


# ---------------------------------------------------------------------------
# Best comparison builder
# ---------------------------------------------------------------------------

def _build_best_comparison(
    df: pd.DataFrame,
    particle_results: dict[str, dict],
    tests: dict[str, Config],
    out_dir: Path,
) -> pd.DataFrame:
    """Find best baseline params per test and compare to particle methods."""
    rows = []

    for test_name in TEST_NAMES:
        if test_name not in tests:
            continue
        cfg = tests[test_name]

        # Best spectral (noise_delta sweep)
        sc_nd = df[(df["test"] == test_name) & (df["method_type"] == "spectral_noise_delta")]
        if not sc_nd.empty:
            idx_best = sc_nd["rel_L2"].idxmin()
            best_nd_row = sc_nd.loc[idx_best]
            best_nd = best_nd_row["param_value"]
            best_sc_rel_L2 = best_nd_row["rel_L2"]
            best_sc_fwd_L2 = best_nd_row.get("forward_consistency_L2", float("nan"))
            best_sc_n_modes = best_nd_row.get("n_modes_kept", float("nan"))
        else:
            best_nd, best_sc_rel_L2, best_sc_fwd_L2, best_sc_n_modes = \
                float("nan"), float("nan"), float("nan"), float("nan")

        # Best Tikhonov
        tik = df[(df["test"] == test_name) & (df["method_type"] == "tikhonov")]
        if not tik.empty:
            idx_best = tik["rel_L2"].idxmin()
            best_tik_row = tik.loc[idx_best]
            best_lam = best_tik_row["param_value"]
            best_tik_rel_L2 = best_tik_row["rel_L2"]
            best_tik_fwd_L2 = best_tik_row.get("forward_consistency_L2", float("nan"))
        else:
            best_lam, best_tik_rel_L2, best_tik_fwd_L2 = float("nan"), float("nan"), float("nan")

        # Particle raw
        pr = particle_results.get(test_name, {})
        particle_rel_L2 = pr.get("det_raw_rel_L2", float("nan"))
        particle_fwd_L2 = pr.get("det_raw_fwd_L2", float("nan"))
        particle_completed = pr.get("det_raw_completed", False)
        particle_fail_step = pr.get("det_raw_failure_step", None)

        # Ratio: best_baseline / particle
        ratio_sc = best_sc_rel_L2 / particle_rel_L2 if (
            np.isfinite(best_sc_rel_L2) and np.isfinite(particle_rel_L2) and particle_rel_L2 > 0
        ) else float("nan")
        ratio_tik = best_tik_rel_L2 / particle_rel_L2 if (
            np.isfinite(best_tik_rel_L2) and np.isfinite(particle_rel_L2) and particle_rel_L2 > 0
        ) else float("nan")

        rows.append({
            "test": test_name,
            "T": cfg.heat.T,
            "alpha": cfg.heat.alpha,
            # Particle
            "particle_det_raw_rel_L2": particle_rel_L2,
            "particle_det_raw_fwd_L2": particle_fwd_L2,
            "particle_completed": particle_completed,
            "particle_failure_step": particle_fail_step,
            # Best spectral
            "best_spectral_noise_delta": best_nd,
            "best_spectral_rel_L2": best_sc_rel_L2,
            "best_spectral_fwd_L2": best_sc_fwd_L2,
            "best_spectral_n_modes": best_sc_n_modes,
            # Best Tikhonov
            "best_tikhonov_lambda": best_lam,
            "best_tikhonov_rel_L2": best_tik_rel_L2,
            "best_tikhonov_fwd_L2": best_tik_fwd_L2,
            # Ratios
            "ratio_best_spectral_over_particle": ratio_sc,
            "ratio_best_tikhonov_over_particle": ratio_tik,
        })

    best_df = pd.DataFrame(rows)
    best_df.to_csv(out_dir / "best_baseline_comparison.csv", index=False)
    print(f"  Saved: {out_dir / 'best_baseline_comparison.csv'}")
    return best_df


# ---------------------------------------------------------------------------
# Plots for PART 2
# ---------------------------------------------------------------------------

def _plot_spectral_noise_delta(df: pd.DataFrame, tests: dict, save_path: Path) -> None:
    test_names = [t for t in TEST_NAMES if t in tests]
    n = len(test_names)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for i, test_name in enumerate(test_names):
        ax = axes[i // ncols][i % ncols]
        sub = df[(df["test"] == test_name) & (df["method_type"] == "spectral_noise_delta")]
        if sub.empty:
            ax.set_title(f"Test {test_name} (no data)")
            continue
        sub_sorted = sub.sort_values("param_value")
        ax.semilogx(sub_sorted["param_value"], sub_sorted["rel_L2"], "b-o", ms=5, label="spectral")
        ax.set_xlabel("noise_delta")
        ax.set_ylabel("rel_L2")
        ax.set_title(f"Test {test_name} (T={tests[test_name].heat.T})")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # Hide unused subplots
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle("Spectral Cutoff rel_L2 vs noise_delta", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def _plot_tikhonov_lambda(df: pd.DataFrame, tests: dict, save_path: Path) -> None:
    test_names = [t for t in TEST_NAMES if t in tests]
    n = len(test_names)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for i, test_name in enumerate(test_names):
        ax = axes[i // ncols][i % ncols]
        sub = df[(df["test"] == test_name) & (df["method_type"] == "tikhonov")]
        if sub.empty:
            ax.set_title(f"Test {test_name} (no data)")
            continue
        sub_sorted = sub.sort_values("param_value")
        ax.loglog(sub_sorted["param_value"], sub_sorted["rel_L2"], "r-s", ms=5, label="Tikhonov")
        ax.set_xlabel("lambda")
        ax.set_ylabel("rel_L2")
        ax.set_title(f"Test {test_name} (T={tests[test_name].heat.T})")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle("Tikhonov rel_L2 vs lambda", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def _plot_kcut_mult(df: pd.DataFrame, tests: dict, save_path: Path) -> None:
    test_names = [t for t in TEST_NAMES if t in tests]
    n = len(test_names)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for i, test_name in enumerate(test_names):
        ax = axes[i // ncols][i % ncols]
        sub = df[(df["test"] == test_name) & (df["method_type"] == "spectral_kcut_mult")]
        if sub.empty:
            ax.set_title(f"Test {test_name} (no data)")
            continue
        sub_sorted = sub.sort_values("param_value")
        ax.plot(sub_sorted["param_value"], sub_sorted["rel_L2"], "g-^", ms=6, label="spectral")
        ax.set_xlabel("k_cut / k_cut_opt  (mult)")
        ax.set_ylabel("rel_L2")
        ax.set_title(f"Test {test_name} (T={tests[test_name].heat.T})")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle("Spectral Cutoff rel_L2 vs k_cut multiplier (rel. to noise_delta=1e-8 opt)", fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def _plot_best_comparison(best_df: pd.DataFrame, save_path: Path) -> None:
    if best_df.empty:
        return

    tests_present = list(best_df["test"])
    x = np.arange(len(tests_present))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))

    vals_particle = best_df["particle_det_raw_rel_L2"].values.astype(float)
    vals_spectral = best_df["best_spectral_rel_L2"].values.astype(float)
    vals_tik      = best_df["best_tikhonov_rel_L2"].values.astype(float)

    bars_p = ax.bar(x - width, vals_particle, width, label="particle det_raw", color="steelblue", alpha=0.85)  # type: ignore[arg-type]
    bars_s = ax.bar(x,         vals_spectral, width, label="best spectral",    color="darkorange", alpha=0.85)  # type: ignore[arg-type]
    bars_t = ax.bar(x + width, vals_tik,      width, label="best Tikhonov",    color="forestgreen", alpha=0.85)  # type: ignore[arg-type]

    # Annotate bars with values
    for bar in [*bars_p, *bars_s, *bars_t]:
        h = bar.get_height()
        if np.isfinite(h) and h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h * 1.02,
                f"{h:.3f}" if h >= 0.001 else f"{h:.1e}",
                ha="center", va="bottom", fontsize=7, rotation=60,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(tests_present)
    ax.set_xlabel("Test")
    ax.set_ylabel("rel_L2 (vs true_u0)")
    ax.set_title("Best Baseline vs Particle GRW — rel_L2 Comparison\n"
                 "(lower is better; particle fails = partial reconstruction)")
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", alpha=0.3)

    # Add failed-run markers
    for i, test_name in enumerate(tests_present):
        row = best_df[best_df["test"] == test_name]
        if not row.empty and not bool(row["particle_completed"].values[0]):
            ax.text(i - width, 0.02, "FAIL", ha="center", va="bottom",
                    color="red", fontsize=7, fontweight="bold")

    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


# ---------------------------------------------------------------------------
# Summary writer for PART 2
# ---------------------------------------------------------------------------

def _write_part2_summary(
    df: pd.DataFrame,
    best_df: pd.DataFrame,
    particle_results: dict,
    save_path: Path,
) -> None:
    lines = [
        "PART 2 — Tuned Baseline Sweep Summary",
        "=" * 70,
        "",
        f"Spectral noise_delta values swept: {NOISE_DELTA_VALUES}",
        f"Spectral k_cut_mult values swept:  {K_CUT_MULT_VALUES}",
        f"Tikhonov lambda values swept:      {TIKHONOV_LAMBDAS}",
        "",
        "Best parameters per test (minimizing rel_L2 vs true_u0):",
        "",
    ]

    if not best_df.empty:
        hdr = (f"{'Test':<6} {'T':>6} "
               f"{'Particle rel_L2':>17} {'Best Spec nd':>14} {'Spec rel_L2':>12} "
               f"{'Best Tik lam':>14} {'Tik rel_L2':>12} "
               f"{'Ratio Spec/Part':>16} {'Ratio Tik/Part':>15}")
        lines.append(hdr)
        lines.append("-" * len(hdr))
        for _, row in best_df.iterrows():
            lines.append(
                f"{row['test']:<6} {row['T']:>6.3f} "
                f"{row['particle_det_raw_rel_L2']:>17.4f} "
                f"{row['best_spectral_noise_delta']:>14.1e} "
                f"{row['best_spectral_rel_L2']:>12.4f} "
                f"{row['best_tikhonov_lambda']:>14.1e} "
                f"{row['best_tikhonov_rel_L2']:>12.6f} "
                f"{row['ratio_best_spectral_over_particle']:>16.3f} "
                f"{row['ratio_best_tikhonov_over_particle']:>15.3f}"
            )
    lines.append("")

    lines.append("Interpretation guide:")
    lines.append("  ratio < 1 : baseline BEATS particle (baseline is more accurate)")
    lines.append("  ratio > 1 : particle BEATS baseline  (particle is more accurate)")
    lines.append("  ratio = NaN: particle failed (no valid comparison possible)")
    lines.append("")
    lines.append("Particle completion status:")
    for test_name, pr in particle_results.items():
        status = "OK" if pr["det_raw_completed"] else f"FAIL@{pr['det_raw_failure_step']}"
        lines.append(
            f"  {test_name}: {status}  rel_L2={pr['det_raw_rel_L2']:.4f}"
        )
    lines.append("")

    # Forward consistency of best baselines
    lines.append("Forward consistency (L2 of forward_solve(candidate) - observed):")
    lines.append("  Smaller = better inverse quality (candidate produces observed when re-diffused)")
    if not best_df.empty:
        for _, row in best_df.iterrows():
            lines.append(
                f"  {row['test']}: spectral={row['best_spectral_fwd_L2']:.3e}  "
                f"tikhonov={row['best_tikhonov_fwd_L2']:.3e}  "
                f"particle={row['particle_det_raw_fwd_L2']:.3e}"
            )
    lines.append("")

    text = "\n".join(lines)
    save_path.write_text(text)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline verification pass for Phase 5.")
    p.add_argument(
        "--base-config", default="configs/gaussian_base.yaml",
        help="Path to base Gaussian config (default: configs/gaussian_base.yaml)",
    )
    p.add_argument(
        "--mixture-config", default="configs/gaussian_mixture.yaml",
        help="Path to Gaussian mixture config (default: configs/gaussian_mixture.yaml)",
    )
    p.add_argument(
        "--skip-part1", action="store_true",
        help="Skip PART 1 (path equivalence) and only run PART 2 (baseline sweep)",
    )
    p.add_argument(
        "--skip-part2", action="store_true",
        help="Skip PART 2 (baseline sweep) and only run PART 1 (path equivalence)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve config paths relative to the script's parent (project root)
    project_root = Path(__file__).resolve().parent.parent
    base_path    = project_root / args.base_config
    mixture_path = project_root / args.mixture_config

    if not base_path.exists():
        print(f"ERROR: base config not found: {base_path}", file=sys.stderr)
        sys.exit(1)
    if not mixture_path.exists():
        print(f"ERROR: mixture config not found: {mixture_path}", file=sys.stderr)
        sys.exit(1)

    base_cfg    = load_config(str(base_path))
    mixture_cfg = load_config(str(mixture_path))
    tests       = build_test_configs(base_cfg, mixture_cfg)

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = project_root / "outputs" / f"baseline_verification_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")
    print(f"Tests to run: {list(tests.keys())}")

    # PART 1
    if not args.skip_part1:
        run_part1(tests, out_dir)
    else:
        print("\n[PART 1 skipped via --skip-part1]")

    # PART 2
    if not args.skip_part2:
        run_part2(tests, out_dir)
    else:
        print("\n[PART 2 skipped via --skip-part2]")

    print(f"\nDone.  All outputs in: {out_dir}")


if __name__ == "__main__":
    main()
