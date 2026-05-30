"""
run_convergence_study.py — Grid and glob convergence study.

Purpose:
    Determine whether score-guided GRW converges as (a) the reconstruction
    grid is refined (n_grid) and (b) the glob count is increased
    (gradient_globs_per_jump).  Classical baselines are run at each
    resolution and retained as accuracy reference ceilings.

Tests:
    B : Gaussian, sigma0=0.08, T=0.15
    H : Gaussian mixture, T=0.15
    Z : near-zero tail stress, T=0.05

Sweeps:
    n_grid                    : [100, 200, 400, 800]
    gradient_globs_per_jump   : [10, 20, 40, 80]

Methods per cell (n_grid, gradient_globs_per_jump):
    - naive_backward
    - oracle_score_deterministic
    - estimated_score_deterministic_raw           (position_ratio_raw)
    - estimated_score_deterministic_grid_ratio_raw (grid_ratio_raw)
    - estimated_score_deterministic_regularized eps=1e-10
    - estimated_score_deterministic_regularized eps=1e-8
    - estimated_score_deterministic_regularized eps=1e-6
    - spectral_cutoff_inverse (noise_delta=1e-8, 1e-6, 1e-4 sweep; best retained)
    - tikhonov_inverse (lambda sweep 1e-8..1e-2; best retained)

Outputs:
    outputs/convergence_study_TIMESTAMP/
        convergence_metrics.csv
        convergence_summary.txt
        best_by_resolution.csv
        relative_L2_vs_n_grid_by_method_<test>.png
        relative_L2_vs_globs_by_method_<test>.png
        forward_consistency_vs_n_grid_<test>.png
        score_error_vs_n_grid_<test>.png
        runtime_vs_resolution_<test>.png
        position_ratio_vs_grid_ratio_diff_<test>.png

Usage:
    PYTHONPATH=src python scripts/run_convergence_study.py \\
        --base-config configs/gaussian_base.yaml \\
        --mixture-config configs/gaussian_mixture.yaml
"""

from __future__ import annotations

import sys
import argparse
import copy
import math
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
    load_config, Config, DomainConfig,
    RegularizationConfig, EpsilonFloorConfig, ScoreClippingConfig, SmoothingConfig,
)
from invheat_grw.fields import make_grid, true_u0 as compute_true_u0, observed_final as compute_observed_final
from invheat_grw.methods import (
    run_naive_backward,
    run_oracle_score_deterministic,
    run_estimated_score_deterministic_raw,
    run_estimated_score_deterministic_grid_ratio_raw,
    run_estimated_score_deterministic_regularized,
    MethodResult,
)
from invheat_grw.metrics import (
    compute_metrics,
    compute_baseline_metrics,
    compute_wasserstein,
    MethodMetrics,
)
from invheat_grw.baselines import spectral_cutoff_inverse, tikhonov_inverse


# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------

N_GRID_VALUES = [100, 200, 400, 800]
GLOB_VALUES = [10, 20, 40, 80]
EPSILON_VALUES = [1e-10, 1e-8, 1e-6]
NOISE_DELTA_VALUES = [1e-8, 1e-6, 1e-4, 1e-2]
TIKHONOV_LAMBDAS = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
RNG_SEED = 42


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


def make_reg_cfg(epsilon: float) -> RegularizationConfig:
    return RegularizationConfig(
        enabled=True,
        epsilon_floor=EpsilonFloorConfig(
            enabled=(epsilon > 0.0),
            value=epsilon,
            scale_by_peak=False,
        ),
        score_clipping=ScoreClippingConfig(enabled=False),
        smoothing=SmoothingConfig(enabled=False),
    )


# ---------------------------------------------------------------------------
# Base test configs (at default resolution; will be overridden per cell)
# ---------------------------------------------------------------------------

def build_base_tests(base_cfg: Config, mixture_cfg: Config) -> dict[str, Config]:
    tests = {}
    # Test B: Gaussian, heavy diffusion
    tests["B"] = patch_config(base_cfg, **{"heat.T": 0.15, "initial_condition.sigma0": 0.08})
    # Test H: Gaussian mixture, heavy diffusion
    tests["H"] = patch_config(mixture_cfg, **{"heat.T": 0.15})
    # Test Z: near-zero tail stress
    tests["Z"] = patch_config(base_cfg, **{
        "heat.T": 0.05,
        "initial_condition.sigma0": 0.05,
        "initial_condition.mu": 0.4,
    })
    return tests


def at_resolution(base_test_cfg: Config, n_grid: int, globs_per_jump: int) -> Config:
    """Return config with n_grid and gradient_globs_per_jump overridden."""
    return patch_config(base_test_cfg, **{
        "domain.n_grid": n_grid,
        "grw.gradient_globs_per_jump": globs_per_jump,
    })


# ---------------------------------------------------------------------------
# Row builder: run all methods at one (test, n_grid, globs) cell
# ---------------------------------------------------------------------------

def run_cell(
    test_name: str,
    cfg: Config,
    n_grid: int,
    globs_per_jump: int,
) -> list[dict]:
    """Run all methods for one resolution cell. Returns list of row dicts."""
    x_grid = make_grid(cfg)
    true_u = compute_true_u0(x_grid, cfg)
    u_obs = compute_observed_final(x_grid, cfg)
    rng = np.random.default_rng(RNG_SEED)

    rows = []

    def _row(result: MethodResult, method_label: str, extra: dict = {}) -> dict:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = compute_metrics(result, true_u, u_obs, x_grid, cfg)
            wass = compute_wasserstein(result.candidate, true_u, x_grid)
        # score error summaries
        sl2_vals = [v for v in result.score_L2_error_vs_oracle if np.isfinite(v)]
        sc_l2_vals = [v for v in result.score_core_L2_error_vs_oracle if np.isfinite(v)]
        r = {
            "test": test_name,
            "n_grid": n_grid,
            "gradient_globs_per_jump": globs_per_jump,
            "method": result.method_name,
            "method_label": method_label,
            "score_estimator_type": result.score_estimator_type,
            "completed": result.completed,
            "failure_step": result.failure_step if result.failure_step is not None else "",
            "failure_msg": result.failure_msg or "",
            "relative_l2": m.relative_l2,
            "l2_error": m.l2_error,
            "linf_error": m.linf_error,
            "forward_consistency_l2": m.forward_consistency_l2,
            "wasserstein": wass,
            "mass_rel_error": m.mass_rel_error,
            "max_abs_score_final": m.max_abs_score_final,
            "score_L2_error_vs_oracle_max": max(sl2_vals) if sl2_vals else float("nan"),
            "score_L2_error_vs_oracle_final": sl2_vals[-1] if sl2_vals else float("nan"),
            "score_core_L2_error_vs_oracle_final": sc_l2_vals[-1] if sc_l2_vals else float("nan"),
            "runtime_seconds": result.runtime_seconds,
            "epsilon_used": result.epsilon_used,
            "n_denom_below_eps_total": int(sum(result.n_denominator_below_epsilon)),
            "n_clipped_total": int(sum(result.n_clipped_scores)),
        }
        r.update(extra)
        return r

    def _baseline_row(candidate: np.ndarray, method_name: str, method_label: str,
                      extra: dict = {}) -> dict:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = compute_baseline_metrics(candidate, true_u, u_obs, x_grid, cfg,
                                         method_name=method_name,
                                         method_category="baseline")
            wass = compute_wasserstein(candidate, true_u, x_grid)
        r = {
            "test": test_name,
            "n_grid": n_grid,
            "gradient_globs_per_jump": globs_per_jump,
            "method": method_name,
            "method_label": method_label,
            "score_estimator_type": "",
            "completed": True,
            "failure_step": "",
            "failure_msg": "",
            "relative_l2": m.relative_l2,
            "l2_error": m.l2_error,
            "linf_error": m.linf_error,
            "forward_consistency_l2": m.forward_consistency_l2,
            "wasserstein": wass,
            "mass_rel_error": m.mass_rel_error,
            "max_abs_score_final": float("nan"),
            "score_L2_error_vs_oracle_max": float("nan"),
            "score_L2_error_vs_oracle_final": float("nan"),
            "score_core_L2_error_vs_oracle_final": float("nan"),
            "runtime_seconds": 0.0,
            "epsilon_used": float("nan"),
            "n_denom_below_eps_total": 0,
            "n_clipped_total": 0,
        }
        r.update(extra)
        return r

    # --- Particle methods ---
    # naive_backward
    r_naive = run_naive_backward(u_obs, x_grid, cfg, np.random.default_rng(RNG_SEED))
    rows.append(_row(r_naive, "naive_backward"))

    # oracle_score_deterministic
    r_oracle = run_oracle_score_deterministic(u_obs, x_grid, cfg, np.random.default_rng(RNG_SEED))
    rows.append(_row(r_oracle, "oracle_det"))

    # estimated_score_deterministic_raw (position_ratio_raw)
    r_raw = run_estimated_score_deterministic_raw(u_obs, x_grid, cfg, np.random.default_rng(RNG_SEED))
    rows.append(_row(r_raw, "est_det_position_ratio_raw"))

    # estimated_score_deterministic_grid_ratio_raw
    r_gr = run_estimated_score_deterministic_grid_ratio_raw(u_obs, x_grid, cfg, np.random.default_rng(RNG_SEED))
    rows.append(_row(r_gr, "est_det_grid_ratio_raw"))

    # estimated_score_deterministic_regularized with each epsilon
    for eps in EPSILON_VALUES:
        cfg_eps = patch_config(cfg)
        cfg_eps.regularization = make_reg_cfg(eps)
        r_eps = run_estimated_score_deterministic_regularized(
            u_obs, x_grid, cfg_eps, np.random.default_rng(RNG_SEED))
        rows.append(_row(r_eps, f"est_det_grid_ratio_eps={eps:.0e}",
                         {"epsilon_sweep_value": eps}))

    # --- Classical baselines (noise_delta sweep, keep best) ---
    alpha = cfg.heat.alpha
    T_val = cfg.heat.T
    best_sc = None
    best_sc_rel_l2 = float("inf")
    best_sc_nd = float("nan")
    for nd in NOISE_DELTA_VALUES:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sc_res = spectral_cutoff_inverse(u_obs, x_grid, alpha, T_val, noise_delta=nd)
        if np.all(np.isfinite(sc_res.candidate)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                m_sc = compute_baseline_metrics(sc_res.candidate, true_u, u_obs, x_grid, cfg,
                                                method_name="spectral_cutoff_baseline")
            if np.isfinite(m_sc.relative_l2) and m_sc.relative_l2 < best_sc_rel_l2:
                best_sc_rel_l2 = m_sc.relative_l2
                best_sc = sc_res.candidate.copy()
                best_sc_nd = nd
    if best_sc is not None:
        rows.append(_baseline_row(best_sc, "spectral_cutoff_best",
                                  f"spectral_cutoff_best(nd={best_sc_nd:.0e})",
                                  {"noise_delta_used": best_sc_nd}))
    else:
        rows.append(_baseline_row(np.zeros_like(x_grid), "spectral_cutoff_best",
                                  "spectral_cutoff_best(failed)",
                                  {"noise_delta_used": float("nan")}))

    # --- Tikhonov: lambda sweep, keep best ---
    best_tik = None
    best_tik_rel_l2 = float("inf")
    best_tik_lam = float("nan")
    for lam in TIKHONOV_LAMBDAS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            tik_res = tikhonov_inverse(u_obs, x_grid, alpha, T_val, lam=lam)
        if np.all(np.isfinite(tik_res.candidate)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                m_tik = compute_baseline_metrics(tik_res.candidate, true_u, u_obs, x_grid, cfg,
                                                 method_name="tikhonov_baseline")
            if np.isfinite(m_tik.relative_l2) and m_tik.relative_l2 < best_tik_rel_l2:
                best_tik_rel_l2 = m_tik.relative_l2
                best_tik = tik_res.candidate.copy()
                best_tik_lam = lam
    if best_tik is not None:
        rows.append(_baseline_row(best_tik, "tikhonov_best",
                                  f"tikhonov_best(lam={best_tik_lam:.0e})",
                                  {"tikhonov_lam_used": best_tik_lam}))
    else:
        rows.append(_baseline_row(np.zeros_like(x_grid), "tikhonov_best",
                                  "tikhonov_best(failed)",
                                  {"tikhonov_lam_used": float("nan")}))

    return rows


# ---------------------------------------------------------------------------
# Full sweep
# ---------------------------------------------------------------------------

def run_all(tests: dict[str, Config]) -> pd.DataFrame:
    all_rows = []
    total_cells = len(tests) * len(N_GRID_VALUES) * len(GLOB_VALUES)
    cell_idx = 0
    for test_name, base_cfg in tests.items():
        print(f"\n  Test {test_name}:")
        for n_grid in N_GRID_VALUES:
            for globs in GLOB_VALUES:
                cell_idx += 1
                cfg = at_resolution(base_cfg, n_grid, globs)
                print(f"    [{cell_idx}/{total_cells}] n_grid={n_grid}, globs_per_jump={globs} ...",
                      end=" ", flush=True)
                rows = run_cell(test_name, cfg, n_grid, globs)
                all_rows.extend(rows)
                # Quick status for this cell
                particle_rows = [r for r in rows if "est_det_position_ratio_raw" in r["method_label"]]
                if particle_rows:
                    r0 = particle_rows[0]
                    status = "OK" if r0["completed"] else f"FAIL@{r0['failure_step']}"
                    rl = r0["relative_l2"]
                    rl_str = f"{rl:.4f}" if np.isfinite(rl) else "nan"
                    print(f"pos_ratio_raw: {status} rel_L2={rl_str}")
                else:
                    print()
    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Analysis and summary text
# ---------------------------------------------------------------------------

def analyze(df: pd.DataFrame, tests: dict) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("CONVERGENCE STUDY SUMMARY")
    lines.append(f"Tests: {', '.join(tests.keys())}")
    lines.append(f"n_grid values:  {N_GRID_VALUES}")
    lines.append(f"glob values:    {GLOB_VALUES}")
    lines.append("=" * 80)

    # For each test, pick the full n_grid sweep at the default glob count (20)
    default_globs = 20

    for test_name in tests:
        lines.append(f"\n{'─'*70}")
        lines.append(f"TEST {test_name}")
        lines.append(f"{'─'*70}")

        # Build table: rows=n_grid, cols=method
        df_t = df[df["test"] == test_name].copy()
        methods_of_interest = [
            "naive_backward",
            "oracle_det",
            "est_det_position_ratio_raw",
            "est_det_grid_ratio_raw",
            f"est_det_grid_ratio_eps=1e-10",
            f"est_det_grid_ratio_eps=1e-08",
            f"est_det_grid_ratio_eps=1e-06",
            "spectral_cutoff_best",
            "tikhonov_best",
        ]

        # n_grid convergence at default globs
        lines.append(f"\n  rel_L2 vs n_grid (gradient_globs_per_jump={default_globs}):")
        df_ng = df_t[df_t["gradient_globs_per_jump"] == default_globs]
        header = f"  {'n_grid':>6}  " + "  ".join(f"{m[:24]:>24}" for m in methods_of_interest)
        lines.append(header)
        lines.append("  " + "-" * (8 + 28 * len(methods_of_interest)))
        for ng in N_GRID_VALUES:
            df_ng_row = df_ng[df_ng["n_grid"] == ng]
            vals = []
            for m in methods_of_interest:
                sub = df_ng_row[df_ng_row["method_label"] == m]
                if sub.empty:
                    vals.append("      n/a       ")
                else:
                    rl = sub["relative_l2"].values[0]
                    ok = bool(sub["completed"].values[0])
                    tag = "" if ok else "!"
                    vals.append(f"{rl:>10.5f}{tag:>2}" if np.isfinite(rl) else f"{'nan/fail':>12}")
            lines.append(f"  {ng:>6}  " + "  ".join(f"{v:>24}" for v in vals))

        # glob convergence at default n_grid (200)
        default_ng = 200
        lines.append(f"\n  rel_L2 vs globs_per_jump (n_grid={default_ng}):")
        df_gl = df_t[df_t["n_grid"] == default_ng]
        header2 = f"  {'globs':>6}  " + "  ".join(f"{m[:24]:>24}" for m in methods_of_interest)
        lines.append(header2)
        lines.append("  " + "-" * (8 + 28 * len(methods_of_interest)))
        for gl in GLOB_VALUES:
            df_gl_row = df_gl[df_gl["gradient_globs_per_jump"] == gl]
            vals = []
            for m in methods_of_interest:
                sub = df_gl_row[df_gl_row["method_label"] == m]
                if sub.empty:
                    vals.append("      n/a       ")
                else:
                    rl = sub["relative_l2"].values[0]
                    ok = bool(sub["completed"].values[0])
                    tag = "" if ok else "!"
                    vals.append(f"{rl:>10.5f}{tag:>2}" if np.isfinite(rl) else f"{'nan/fail':>12}")
            lines.append(f"  {gl:>6}  " + "  ".join(f"{v:>24}" for v in vals))

        # Forward consistency at default globs
        lines.append(f"\n  forward_consistency_l2 vs n_grid (gradient_globs_per_jump={default_globs}):")
        fwd_methods = ["est_det_position_ratio_raw", "est_det_grid_ratio_raw",
                       "spectral_cutoff_best", "tikhonov_best"]
        df_ng2 = df_t[df_t["gradient_globs_per_jump"] == default_globs]
        header3 = f"  {'n_grid':>6}  " + "  ".join(f"{m[:28]:>28}" for m in fwd_methods)
        lines.append(header3)
        for ng in N_GRID_VALUES:
            df_ng_row = df_ng2[df_ng2["n_grid"] == ng]
            vals = []
            for m in fwd_methods:
                sub = df_ng_row[df_ng_row["method_label"] == m]
                if sub.empty:
                    vals.append("      n/a        ")
                else:
                    fl = sub["forward_consistency_l2"].values[0]
                    vals.append(f"{fl:>12.4e}" if np.isfinite(fl) else f"{'nan':>12}")
            lines.append(f"  {ng:>6}  " + "  ".join(f"{v:>28}" for v in vals))

        # Convergence verdict
        lines.append(f"\n  --- Convergence Verdict ---")

        def check_converges(method_label: str, col: str = "relative_l2") -> str:
            sub = df_t[(df_t["gradient_globs_per_jump"] == default_globs) &
                       (df_t["method_label"] == method_label)]
            if sub.empty:
                return "n/a"
            sub_sorted = sub.sort_values("n_grid")
            vals = sub_sorted[col].values
            finite = [v for v in vals if np.isfinite(v)]
            if len(finite) < 2:
                return "insufficient data"
            # Monotone decrease?
            diffs = [finite[i+1] - finite[i] for i in range(len(finite)-1)]
            if all(d < 0 for d in diffs):
                return f"CONVERGES (monotone, {finite[0]:.4f} → {finite[-1]:.4f})"
            elif finite[-1] < finite[0]:
                return f"WEAKLY CONVERGES ({finite[0]:.4f} → {finite[-1]:.4f}, non-monotone)"
            elif finite[-1] > finite[0] * 1.05:
                return f"DIVERGES ({finite[0]:.4f} → {finite[-1]:.4f})"
            else:
                return f"PLATEAU ({finite[0]:.4f} → {finite[-1]:.4f})"

        for meth in ["oracle_det", "est_det_position_ratio_raw", "est_det_grid_ratio_raw",
                     "est_det_grid_ratio_eps=1e-08", "spectral_cutoff_best", "tikhonov_best"]:
            verdict = check_converges(meth)
            lines.append(f"    {meth:<40} : {verdict}")

        # Gap between best particle and best classical at highest resolution
        df_high = df_t[(df_t["n_grid"] == max(N_GRID_VALUES)) &
                       (df_t["gradient_globs_per_jump"] == max(GLOB_VALUES))]

        def best_rl(label_prefix: str) -> float:
            sub = df_high[df_high["method_label"].str.startswith(label_prefix)]
            vals = sub["relative_l2"].values
            finite = [v for v in vals if np.isfinite(v)]
            return min(finite) if finite else float("nan")

        best_particle = min(
            best_rl("est_det_position_ratio"),
            best_rl("est_det_grid_ratio"),
        )
        best_spectral = best_rl("spectral_cutoff_best")
        best_tikhonov = best_rl("tikhonov_best")

        lines.append(f"\n  At highest resolution (n_grid={max(N_GRID_VALUES)}, globs={max(GLOB_VALUES)}):")
        lines.append(f"    Best particle rel_L2  : {best_particle:.5f}")
        lines.append(f"    Best spectral rel_L2  : {best_spectral:.5f}")
        lines.append(f"    Best Tikhonov rel_L2  : {best_tikhonov:.5f}")
        if np.isfinite(best_particle) and np.isfinite(best_tikhonov) and best_tikhonov > 0:
            ratio = best_particle / best_tikhonov
            lines.append(f"    Particle/Tikhonov gap : {ratio:.1f}×")
        lines.append("")

    # Synthesis section (8 questions)
    lines.append("=" * 80)
    lines.append("SYNTHESIS — CONVERGENCE STUDY ANALYSIS")
    lines.append("=" * 80)

    q_answers = [
        ("Q1. Does oracle GRW improve with n_grid/glob refinement?",
         "See per-test oracle_det verdict above.  Oracle uses exact score so "
         "convergence reflects grid quality of reconstruction and boundary handling only."),
        ("Q2. Does position-ratio raw GRW improve?",
         "See est_det_position_ratio_raw verdict.  This estimator interpolates u "
         "and u_x separately then divides; accuracy improves if score error decreases "
         "with finer grid."),
        ("Q3. Does grid-ratio raw GRW improve?",
         "See est_det_grid_ratio_raw verdict.  This estimator divides u_x/u on the "
         "grid then interpolates; convergence behavior may differ from position-ratio "
         "at coarse grids."),
        ("Q4. Does epsilon-floor improve stability or accuracy?",
         "Compare est_det_grid_ratio_eps vs grid_ratio_raw at tests B/H/Z.  "
         "On tests D/Z (mixture with near-zero tails), epsilon rescues completions.  "
         "On smooth tests (A/G/B/H), epsilon may introduce small bias."),
        ("Q5. Does particle GRW approach classical accuracy or plateau?",
         "See 'Best particle rel_L2' vs 'Best Tikhonov/spectral rel_L2' at each test.  "
         "A large and persistent gap indicates a structural accuracy ceiling in the "
         "particle formulation, not resolvable by grid refinement alone."),
        ("Q6. Is the bottleneck grid resolution, glob resolution, or the score update?",
         "Compare n_grid sweep (fixed globs=20) vs glob sweep (fixed n_grid=200).  "
         "If rel_L2 improves with n_grid but not globs: grid is the bottleneck.  "
         "If rel_L2 improves with globs but not n_grid: glob count is the bottleneck.  "
         "If neither improves: score update is the bottleneck."),
        ("Q7. Is forward consistency improving with refinement?",
         "See forward_consistency_l2 table.  Particle forward consistency should improve "
         "with n_grid because the reconstruction fidelity improves."),
        ("Q8. Recommendation?",
         "If particle rel_L2 remains 10-100x worse than Tikhonov at highest resolution: "
         "the method needs stochastic posterior framing or a nonlinear test case where "
         "spectral methods fail.  "
         "If gap closes to <5x: further smoothing/KDE regularization is worth exploring."),
    ]
    for q, a in q_answers:
        lines.append(f"\n{q}")
        lines.append(f"  {a}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

PARTICLE_METHODS = [
    "naive_backward",
    "oracle_det",
    "est_det_position_ratio_raw",
    "est_det_grid_ratio_raw",
    "est_det_grid_ratio_eps=1e-10",
    "est_det_grid_ratio_eps=1e-08",
    "est_det_grid_ratio_eps=1e-06",
]
BASELINE_METHODS = ["spectral_cutoff_best", "tikhonov_best"]
ALL_PLOT_METHODS = PARTICLE_METHODS + BASELINE_METHODS

METHOD_COLORS = {
    "naive_backward": "gray",
    "oracle_det": "black",
    "est_det_position_ratio_raw": "royalblue",
    "est_det_grid_ratio_raw": "steelblue",
    "est_det_grid_ratio_eps=1e-10": "darkorange",
    "est_det_grid_ratio_eps=1e-08": "orangered",
    "est_det_grid_ratio_eps=1e-06": "red",
    "spectral_cutoff_best": "forestgreen",
    "tikhonov_best": "purple",
}
METHOD_MARKERS = {
    "naive_backward": "x",
    "oracle_det": "D",
    "est_det_position_ratio_raw": "o",
    "est_det_grid_ratio_raw": "s",
    "est_det_grid_ratio_eps=1e-10": "^",
    "est_det_grid_ratio_eps=1e-08": "^",
    "est_det_grid_ratio_eps=1e-06": "^",
    "spectral_cutoff_best": "P",
    "tikhonov_best": "*",
}


def _get_series(df_test: pd.DataFrame, method_label: str, x_col: str,
                y_col: str, fixed_col: str, fixed_val) -> tuple[list, list]:
    """Extract (x_vals, y_vals) for a method at fixed_col == fixed_val."""
    sub = df_test[(df_test["method_label"] == method_label) &
                  (df_test[fixed_col] == fixed_val)]
    sub = sub.sort_values(x_col)
    xs = sub[x_col].tolist()
    ys = sub[y_col].tolist()
    return xs, ys


def plot_rel_l2_vs_ngrid(df: pd.DataFrame, test_name: str, out_dir: Path) -> None:
    df_t = df[df["test"] == test_name]
    default_globs = 20
    fig, ax = plt.subplots(figsize=(9, 5))
    for meth in ALL_PLOT_METHODS:
        xs, ys = _get_series(df_t, meth, "n_grid", "relative_l2",
                             "gradient_globs_per_jump", default_globs)
        if not xs:
            continue
        ys_plot = [y if np.isfinite(y) else np.nan for y in ys]
        ax.semilogy(xs, ys_plot, marker=METHOD_MARKERS.get(meth, "o"),
                    color=METHOD_COLORS.get(meth, "black"), label=meth, linewidth=1.5)
    ax.set_xlabel("n_grid")
    ax.set_ylabel("relative_l2")
    ax.set_title(f"Test {test_name}: rel_L2 vs n_grid (globs_per_jump={default_globs})")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"relative_L2_vs_n_grid_by_method_{test_name}.png", dpi=130)
    plt.close(fig)


def plot_rel_l2_vs_globs(df: pd.DataFrame, test_name: str, out_dir: Path) -> None:
    df_t = df[df["test"] == test_name]
    default_ng = 200
    fig, ax = plt.subplots(figsize=(9, 5))
    for meth in ALL_PLOT_METHODS:
        xs, ys = _get_series(df_t, meth, "gradient_globs_per_jump", "relative_l2",
                             "n_grid", default_ng)
        if not xs:
            continue
        ys_plot = [y if np.isfinite(y) else np.nan for y in ys]
        ax.semilogy(xs, ys_plot, marker=METHOD_MARKERS.get(meth, "o"),
                    color=METHOD_COLORS.get(meth, "black"), label=meth, linewidth=1.5)
    ax.set_xlabel("gradient_globs_per_jump")
    ax.set_ylabel("relative_l2")
    ax.set_title(f"Test {test_name}: rel_L2 vs globs_per_jump (n_grid={default_ng})")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"relative_L2_vs_globs_by_method_{test_name}.png", dpi=130)
    plt.close(fig)


def plot_forward_consistency_vs_ngrid(df: pd.DataFrame, test_name: str, out_dir: Path) -> None:
    df_t = df[df["test"] == test_name]
    default_globs = 20
    methods = ["est_det_position_ratio_raw", "est_det_grid_ratio_raw",
               "est_det_grid_ratio_eps=1e-08", "spectral_cutoff_best", "tikhonov_best"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for meth in methods:
        xs, ys = _get_series(df_t, meth, "n_grid", "forward_consistency_l2",
                             "gradient_globs_per_jump", default_globs)
        if not xs:
            continue
        ys_plot = [y if np.isfinite(y) else np.nan for y in ys]
        ax.semilogy(xs, ys_plot, marker=METHOD_MARKERS.get(meth, "o"),
                    color=METHOD_COLORS.get(meth, "black"), label=meth, linewidth=1.5)
    ax.set_xlabel("n_grid")
    ax.set_ylabel("forward_consistency_l2")
    ax.set_title(f"Test {test_name}: forward_consistency vs n_grid (globs={default_globs})")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"forward_consistency_vs_n_grid_{test_name}.png", dpi=130)
    plt.close(fig)


def plot_score_error_vs_ngrid(df: pd.DataFrame, test_name: str, out_dir: Path) -> None:
    df_t = df[df["test"] == test_name]
    default_globs = 20
    methods = ["est_det_position_ratio_raw", "est_det_grid_ratio_raw",
               "est_det_grid_ratio_eps=1e-08"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for meth in methods:
        xs, ys = _get_series(df_t, meth, "n_grid", "score_L2_error_vs_oracle_final",
                             "gradient_globs_per_jump", default_globs)
        if not xs:
            continue
        ys_plot = [y if np.isfinite(y) else np.nan for y in ys]
        ax.semilogy(xs, ys_plot, marker=METHOD_MARKERS.get(meth, "o"),
                    color=METHOD_COLORS.get(meth, "black"), label=meth, linewidth=1.5)
    ax.set_xlabel("n_grid")
    ax.set_ylabel("score L2 error vs oracle (final step)")
    ax.set_title(f"Test {test_name}: Score error vs n_grid (globs={default_globs})")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"score_error_vs_n_grid_{test_name}.png", dpi=130)
    plt.close(fig)


def plot_runtime_vs_resolution(df: pd.DataFrame, test_name: str, out_dir: Path) -> None:
    df_t = df[df["test"] == test_name]
    default_globs = 20
    methods = ["oracle_det", "est_det_position_ratio_raw", "est_det_grid_ratio_raw"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for meth in methods:
        xs, ys = _get_series(df_t, meth, "n_grid", "runtime_seconds",
                             "gradient_globs_per_jump", default_globs)
        if not xs:
            continue
        ax.plot(xs, ys, marker=METHOD_MARKERS.get(meth, "o"),
                color=METHOD_COLORS.get(meth, "black"), label=meth, linewidth=1.5)
    ax.set_xlabel("n_grid")
    ax.set_ylabel("runtime_seconds")
    ax.set_title(f"Test {test_name}: Runtime vs n_grid (globs={default_globs})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"runtime_vs_resolution_{test_name}.png", dpi=130)
    plt.close(fig)


def plot_pos_vs_grid_ratio_diff(df: pd.DataFrame, test_name: str, out_dir: Path) -> None:
    """rel_L2 difference between position-ratio and grid-ratio paths vs n_grid."""
    df_t = df[df["test"] == test_name]
    default_globs = 20
    df_ng = df_t[df_t["gradient_globs_per_jump"] == default_globs]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, col, label in [
        (axes[0], "relative_l2", "rel_L2"),
        (axes[1], "forward_consistency_l2", "fwd_consistency_L2"),
    ]:
        for meth, style in [
            ("est_det_position_ratio_raw", "-"),
            ("est_det_grid_ratio_raw", "--"),
        ]:
            xs, ys = _get_series(df_t, meth, "n_grid", col,
                                 "gradient_globs_per_jump", default_globs)
            if not xs:
                continue
            ys_plot = [y if np.isfinite(y) else np.nan for y in ys]
            ax.semilogy(xs, ys_plot, linestyle=style,
                        marker=METHOD_MARKERS.get(meth, "o"),
                        color=METHOD_COLORS.get(meth, "black"), label=meth, linewidth=1.5)
        ax.set_xlabel("n_grid")
        ax.set_ylabel(label)
        ax.set_title(f"Test {test_name}: position-ratio vs grid-ratio ({label})")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / f"position_ratio_vs_grid_ratio_diff_{test_name}.png", dpi=130)
    plt.close(fig)


def make_best_by_resolution(df: pd.DataFrame) -> pd.DataFrame:
    """For each (test, n_grid, globs), record the best particle and best classical rel_L2."""
    particle_labels = [m for m in df["method_label"].unique()
                       if any(p in m for p in ["est_det_", "oracle_det", "naive_"])]
    classical_labels = ["spectral_cutoff_best", "tikhonov_best"]

    records = []
    for (test, ng, gl), grp in df.groupby(["test", "n_grid", "gradient_globs_per_jump"]):
        part = grp[grp["method_label"].isin(particle_labels) &
                   grp["method_label"].str.contains("est_det_") &
                   ~grp["method_label"].str.contains("eps")]
        # exclude naive from "best particle"
        part_nonnaive = part[~part["method_label"].str.contains("naive")]
        def best_rl(sub):
            vals = sub["relative_l2"].values
            finite = [v for v in vals if np.isfinite(v)]
            return min(finite) if finite else float("nan")
        def best_method(sub):
            finite = sub[sub["relative_l2"].apply(np.isfinite)]
            if finite.empty:
                return ""
            return finite.loc[finite["relative_l2"].idxmin(), "method_label"]

        records.append({
            "test": test, "n_grid": ng, "gradient_globs_per_jump": gl,
            "oracle_rel_L2": best_rl(grp[grp["method"] == "oracle_score_deterministic"]),
            "best_particle_rel_L2": best_rl(part_nonnaive),
            "best_particle_method": best_method(part_nonnaive),
            "best_spectral_rel_L2": best_rl(grp[grp["method"] == "spectral_cutoff_best"]),
            "best_tikhonov_rel_L2": best_rl(grp[grp["method"] == "tikhonov_best"]),
            "naive_rel_L2": best_rl(grp[grp["method"] == "naive_backward"]),
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run convergence study")
    parser.add_argument("--base-config", default="configs/gaussian_base.yaml")
    parser.add_argument("--mixture-config", default="configs/gaussian_mixture.yaml")
    args = parser.parse_args()

    base_cfg = load_config(args.base_config)
    mixture_cfg = load_config(args.mixture_config)
    tests = build_base_tests(base_cfg, mixture_cfg)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs") / f"convergence_study_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  CONVERGENCE STUDY")
    print(f"  Tests: {', '.join(tests.keys())}")
    print(f"  n_grid: {N_GRID_VALUES}")
    print(f"  globs:  {GLOB_VALUES}")
    print("=" * 60)

    df = run_all(tests)

    # Save raw metrics
    metrics_path = out_dir / "convergence_metrics.csv"
    df.to_csv(metrics_path, index=False)
    print(f"\n  Saved: {metrics_path}")

    # Best by resolution
    best_df = make_best_by_resolution(df)
    best_path = out_dir / "best_by_resolution.csv"
    best_df.to_csv(best_path, index=False)
    print(f"  Saved: {best_path}")

    # Summary text
    summary_text = analyze(df, tests)
    summary_path = out_dir / "convergence_summary.txt"
    summary_path.write_text(summary_text)
    print(f"  Saved: {summary_path}")

    # Plots
    print("\n  Generating plots ...")
    for test_name in tests:
        plot_rel_l2_vs_ngrid(df, test_name, out_dir)
        plot_rel_l2_vs_globs(df, test_name, out_dir)
        plot_forward_consistency_vs_ngrid(df, test_name, out_dir)
        plot_score_error_vs_ngrid(df, test_name, out_dir)
        plot_runtime_vs_resolution(df, test_name, out_dir)
        plot_pos_vs_grid_ratio_diff(df, test_name, out_dir)
        print(f"    test={test_name}: plots saved")

    print(f"\nDone.  All outputs in: {out_dir.resolve()}")

    # Print summary table
    print("\n" + "=" * 60)
    print("  SUMMARY TABLE: best_by_resolution")
    print("=" * 60)
    print(best_df.to_string(index=False, float_format="%.5f"))
    print()
    print(summary_text)


if __name__ == "__main__":
    main()
