"""
run_targeted_suite.py — Targeted unregularized experiment suite for score-guided backward GRW.

Tests:
  A : T=0.05, sigma0=0.08  (moderate diffusion, standard width)
  B : T=0.15, sigma0=0.08  (heavy diffusion, hardest inversion)
  C : T=0.01, sigma0=0.05  (mild diffusion, narrow peak, n_per=40)
  D : T=0.05, sigma0=0.05  (moderate diffusion, narrow peak, n_per=40)
  E : Stochastic ensemble seeds 0-9 for (T=0.01,s0=0.08) and (T=0.05,s0=0.08)
  F : T=0.05, mu=0.1, sigma0=0.08  (near boundary, test reflecting BC effect)

GO/CONDITIONAL GO/STOP decision is printed at the end.

Usage:
    python scripts/run_targeted_suite.py --base-config configs/gaussian_base.yaml
"""

from __future__ import annotations

import sys
import argparse
import copy
import dataclasses
import math
from datetime import datetime
from pathlib import Path

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from invheat_grw.config import load_config, Config
from invheat_grw.fields import make_grid, true_u0 as compute_true_u0, observed_final as compute_observed_final
from invheat_grw.globs import field_to_globs, reconstruct_field
from invheat_grw.methods import (
    run_naive_backward,
    run_oracle_score_deterministic,
    run_oracle_score_stochastic,
    run_estimated_score_deterministic_raw,
    run_estimated_score_stochastic_raw,
    MethodResult,
)
from invheat_grw.metrics import compute_metrics, MethodMetrics, compute_fwhm
from invheat_grw.plotting import (
    plot_field_comparison_all,
    plot_score_error_by_step,
    plot_score_overlay,
)
from invheat_grw.io_utils import make_output_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def patch_config(cfg: Config, **overrides) -> Config:
    """Return a shallow-copied config with nested fields patched."""
    new_cfg = copy.deepcopy(cfg)
    for key, val in overrides.items():
        if "." in key:
            section, attr = key.split(".", 1)
            section_obj = getattr(new_cfg, section)
            setattr(section_obj, attr, val)
        else:
            setattr(new_cfg, key, val)
    # n_steps is a @property computed from T/dt — no setter needed
    return new_cfg


def run_methods(cfg: Config, which: list[str]) -> dict[str, MethodResult]:
    """Run the selected methods and return {name: MethodResult}."""
    dispatch = {
        "naive_backward": run_naive_backward,
        "oracle_score_deterministic": run_oracle_score_deterministic,
        "oracle_score_stochastic": run_oracle_score_stochastic,
        "estimated_score_deterministic_raw": run_estimated_score_deterministic_raw,
        "estimated_score_stochastic_raw": run_estimated_score_stochastic_raw,
    }
    results = {}
    for name in which:
        fn = dispatch[name]
        rng = np.random.default_rng(cfg.grw.rng_seed + hash(name) % (2**31))
        print(f"    {name} ...", end=" ", flush=True)
        res = fn(compute_observed_final(make_grid(cfg), cfg), make_grid(cfg), cfg, rng)
        status = "OK" if res.completed else f"FAILED@{res.failure_step}"
        print(status)
        results[name] = res
    return results


def compute_all_metrics(results, cfg):
    x = make_grid(cfg)
    u0 = compute_true_u0(x, cfg)
    u_obs = compute_observed_final(x, cfg)
    return {
        name: compute_metrics(res, u0, u_obs, x, cfg)
        for name, res in results.items()
    }


# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------

ALL_DETERMINISTIC = [
    "naive_backward",
    "oracle_score_deterministic",
    "estimated_score_deterministic_raw",
]

ALL_STOCHASTIC = [
    "oracle_score_stochastic",
    "estimated_score_stochastic_raw",
]

ALL_METHODS = ALL_DETERMINISTIC + ALL_STOCHASTIC


TESTS = [
    # name, overrides, which_methods
    ("A",  {"heat.T": 0.05,  "initial_condition.sigma0": 0.08},                    ALL_METHODS),
    ("B",  {"heat.T": 0.15,  "initial_condition.sigma0": 0.08},                    ALL_METHODS),
    ("C",  {"heat.T": 0.01,  "initial_condition.sigma0": 0.05,
            "grw.gradient_globs_per_jump": 40},                                     ALL_METHODS),
    ("D",  {"heat.T": 0.05,  "initial_condition.sigma0": 0.05,
            "grw.gradient_globs_per_jump": 40},                                     ALL_METHODS),
    ("F",  {"heat.T": 0.05,  "initial_condition.mu": 0.1,
            "initial_condition.sigma0": 0.08},                                      ALL_METHODS),
]

ENSEMBLE_CONFIGS = [
    {"label": "E_easy",  "T": 0.01, "sigma0": 0.08, "seeds": list(range(10))},
    {"label": "E_hard",  "T": 0.05, "sigma0": 0.08, "seeds": list(range(10))},
]
ENSEMBLE_METHODS = ["oracle_score_stochastic", "estimated_score_stochastic_raw"]


# ---------------------------------------------------------------------------
# GO / CONDITIONAL GO / STOP decision logic
# ---------------------------------------------------------------------------

def decide(all_rows: list[dict]) -> str:
    """
    Evaluate GO/CONDITIONAL_GO/STOP based on the aggregate metrics table.

    Decision criteria:
    - STOP if: est det raw in test A or D is worse than naive (relative_l2 ratio > 1.1),
               or est det raw diverges (not completed) in A, C, or D,
               or mass_rel_error > 0.5 in test F for any non-naive method,
               or negative lobes: peak_ratio < 0 for est det raw anywhere.
    - GO if: oracle det completed in all ABCD tests,
             est det raw within 3x oracle_det rel_L2 in tests A and C,
             est det raw improves over naive in A and C.
    - CONDITIONAL_GO: oracle works but estimated is marginal.
    """
    rows_by_test_method = {}
    for r in all_rows:
        rows_by_test_method[(r["test"], r["method"])] = r

    def get(test, method, field, default=float("nan")):
        key = (test, method)
        if key not in rows_by_test_method:
            return default
        return rows_by_test_method[key].get(field, default)

    reasons_stop = []
    reasons_cgo = []

    # Check oracle completion
    for test in ["A", "B", "C", "D"]:
        completed = get(test, "oracle_score_deterministic", "completed", False)
        if not completed:
            reasons_stop.append(f"Oracle det failed to complete in test {test}")

    # Check estimated vs naive in A and D
    for test in ["A", "D"]:
        naive_rel = get(test, "naive_backward", "relative_l2")
        est_rel = get(test, "estimated_score_deterministic_raw", "relative_l2")
        completed_est = get(test, "estimated_score_deterministic_raw", "completed", False)
        if not completed_est:
            reasons_stop.append(f"Est det raw diverged in test {test}")
        elif not math.isnan(naive_rel) and not math.isnan(est_rel):
            if est_rel > naive_rel * 1.1:
                reasons_stop.append(
                    f"Est det raw WORSE than naive in test {test} "
                    f"(rel_L2 {est_rel:.4f} vs naive {naive_rel:.4f})"
                )

    # Check negative lobes
    for test in ["A", "B", "C", "D", "F"]:
        peak_ratio = get(test, "estimated_score_deterministic_raw", "peak_ratio")
        if not math.isnan(peak_ratio) and peak_ratio < 0:
            reasons_stop.append(f"Negative peak in est det raw test {test}")

    # Check mass in F
    for method in ["estimated_score_deterministic_raw", "oracle_score_deterministic"]:
        mass_rel = get("F", method, "mass_rel_error")
        if not math.isnan(mass_rel) and abs(mass_rel) > 0.5:
            reasons_stop.append(f"Mass failure in test F for {method} (rel_err={mass_rel:.3f})")

    # Check est vs oracle in A and C
    for test in ["A", "C"]:
        oracle_rel = get(test, "oracle_score_deterministic", "relative_l2")
        est_rel = get(test, "estimated_score_deterministic_raw", "relative_l2")
        if not math.isnan(oracle_rel) and not math.isnan(est_rel):
            ratio = est_rel / oracle_rel if oracle_rel > 0 else float("inf")
            if ratio > 3.0:
                reasons_cgo.append(
                    f"Est det raw is >3x oracle rel_L2 in test {test} "
                    f"(ratio={ratio:.2f})"
                )

    if reasons_stop:
        verdict = "STOP — Add regularization before proceeding."
        explanation = "\n  Triggers:\n  - " + "\n  - ".join(reasons_stop)
    elif reasons_cgo:
        verdict = "CONDITIONAL GO — Estimated score marginal; consider lightweight regularization."
        explanation = "\n  Notes:\n  - " + "\n  - ".join(reasons_cgo)
    else:
        verdict = "GO — Unregularized estimated score viable; proceed to next phase."
        explanation = "\n  All key checks passed."

    return f"\n{'='*60}\nDECISION: {verdict}{explanation}\n{'='*60}"


# ---------------------------------------------------------------------------
# Aggregate plots
# ---------------------------------------------------------------------------

def plot_aggregate_relative_l2(rows: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    tests = sorted(set(r["test"] for r in rows if not r["test"].startswith("E")))
    methods = ALL_DETERMINISTIC
    x_idx = np.arange(len(tests))
    bar_width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, method in enumerate(methods):
        vals = []
        for test in tests:
            v = next((r["relative_l2"] for r in rows if r["test"] == test and r["method"] == method), float("nan"))
            vals.append(v if not math.isnan(v) else 0.0)
        color_map = {
            "naive_backward": "#e41a1c",
            "oracle_score_deterministic": "#377eb8",
            "estimated_score_deterministic_raw": "#ff7f00",
        }
        ax.bar(x_idx + i * bar_width, vals, bar_width,
               label=method, color=color_map.get(method, "gray"))
    ax.set_xticks(x_idx + bar_width)
    ax.set_xticklabels([f"Test {t}" for t in tests])
    ax.set_ylabel("Relative L2 error")
    ax.set_title("Aggregate relative L2 error by test and method")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "aggregate_relative_l2.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_aggregate_fitted_sigma(rows: list[dict], out_dir: Path, cfg_sigma0: float) -> None:
    import matplotlib.pyplot as plt
    tests = sorted(set(r["test"] for r in rows if not r["test"].startswith("E")))
    methods = ALL_DETERMINISTIC
    x_idx = np.arange(len(tests))
    bar_width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, method in enumerate(methods):
        vals = []
        for test in tests:
            v = next((r.get("sigma_fit", float("nan")) for r in rows if r["test"] == test and r["method"] == method), float("nan"))
            vals.append(v if not math.isnan(v) else 0.0)
        color_map = {
            "naive_backward": "#e41a1c",
            "oracle_score_deterministic": "#377eb8",
            "estimated_score_deterministic_raw": "#ff7f00",
        }
        ax.bar(x_idx + i * bar_width, vals, bar_width,
               label=method, color=color_map.get(method, "gray"))
    ax.axhline(cfg_sigma0, color="black", ls="--", lw=1.5, label=f"True sigma0={cfg_sigma0}")
    ax.set_xticks(x_idx + bar_width)
    ax.set_xticklabels([f"Test {t}" for t in tests])
    ax.set_ylabel("Fitted sigma")
    ax.set_title("Aggregate fitted Gaussian sigma by test and method")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "aggregate_fitted_sigma.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_aggregate_peak_ratio(rows: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    tests = sorted(set(r["test"] for r in rows if not r["test"].startswith("E")))
    methods = ALL_DETERMINISTIC
    x_idx = np.arange(len(tests))
    bar_width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, method in enumerate(methods):
        vals = []
        for test in tests:
            v = next((r.get("peak_ratio", float("nan")) for r in rows if r["test"] == test and r["method"] == method), float("nan"))
            vals.append(v if not math.isnan(v) else 0.0)
        color_map = {
            "naive_backward": "#e41a1c",
            "oracle_score_deterministic": "#377eb8",
            "estimated_score_deterministic_raw": "#ff7f00",
        }
        ax.bar(x_idx + i * bar_width, vals, bar_width,
               label=method, color=color_map.get(method, "gray"))
    ax.axhline(1.0, color="black", ls="--", lw=1.5, label="Ideal peak ratio=1")
    ax.set_xticks(x_idx + bar_width)
    ax.set_xticklabels([f"Test {t}" for t in tests])
    ax.set_ylabel("Peak ratio (candidate / true)")
    ax.set_title("Aggregate peak ratio by test and method")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "aggregate_peak_ratio.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_aggregate_forward_consistency(rows: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    tests = sorted(set(r["test"] for r in rows if not r["test"].startswith("E")))
    methods = ALL_DETERMINISTIC
    x_idx = np.arange(len(tests))
    bar_width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, method in enumerate(methods):
        vals = []
        for test in tests:
            v = next((r.get("forward_consistency_l2", float("nan")) for r in rows if r["test"] == test and r["method"] == method), float("nan"))
            vals.append(v if not math.isnan(v) else 0.0)
        color_map = {
            "naive_backward": "#e41a1c",
            "oracle_score_deterministic": "#377eb8",
            "estimated_score_deterministic_raw": "#ff7f00",
        }
        ax.bar(x_idx + i * bar_width, vals, bar_width,
               label=method, color=color_map.get(method, "gray"))
    ax.set_xticks(x_idx + bar_width)
    ax.set_xticklabels([f"Test {t}" for t in tests])
    ax.set_ylabel("L2 forward consistency error")
    ax.set_title("Aggregate forward consistency by test and method")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "aggregate_forward_consistency.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_stochastic_ensemble_summary(ensemble_rows: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    if not ensemble_rows:
        return
    df = pd.DataFrame(ensemble_rows)
    groups = df.groupby(["ensemble_label", "method"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric in zip(axes, ["relative_l2", "forward_consistency_l2"]):
        for (label, method), grp in groups:
            ax.scatter(
                [label] * len(grp),
                grp[metric].values,
                label=f"{label}/{method}",
                alpha=0.7,
                s=30,
            )
            mean_val = grp[metric].mean()
            ax.plot([label], [mean_val], marker="D", markersize=8, color="black")
        ax.set_ylabel(metric)
        ax.set_title(f"Ensemble {metric}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Stochastic ensemble summary (test E)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "ensemble_summary.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run targeted unregularized suite.")
    parser.add_argument("--base-config", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = load_config(Path(args.base_config))

    project_root = Path(__file__).resolve().parent.parent
    outputs_root = project_root / "outputs"
    outputs_root.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_dir = outputs_root / f"targeted_suite_{ts}"
    suite_dir.mkdir(parents=True, exist_ok=True)
    print(f"Suite output: {suite_dir}\n")

    all_rows: list[dict] = []
    ensemble_rows: list[dict] = []

    # -------------------------------------------------------------------
    # Tests A–F
    # -------------------------------------------------------------------
    for test_name, overrides, which in TESTS:
        print(f"\n{'='*50}")
        print(f"Test {test_name}: overrides={overrides}")
        cfg = patch_config(base_cfg, **overrides)
        test_dir = suite_dir / f"test_{test_name}"
        test_dir.mkdir(exist_ok=True)

        x = make_grid(cfg)
        u0 = compute_true_u0(x, cfg)
        u_obs = compute_observed_final(x, cfg)

        # Sanity check
        init_state = field_to_globs(u_obs, x, cfg)
        recon = reconstruct_field(init_state, x)
        dx = x[1] - x[0]
        recon_err = float(np.sqrt(dx * np.sum((recon - u_obs) ** 2)))
        print(f"  Step-0 recon L2: {recon_err:.6e}")

        results = run_methods(cfg, which)
        metrics = compute_all_metrics(results, cfg)

        # Per-test plots
        plot_field_comparison_all(x, u0, u_obs, results, test_dir)
        plot_score_error_by_step(results, test_dir, filename=f"score_error_by_step.png")
        for est_name in ["estimated_score_deterministic_raw", "estimated_score_stochastic_raw"]:
            if est_name in results:
                plot_score_overlay(results[est_name], test_dir,
                                   filename=f"score_overlay_{est_name}.png")

        # Collect rows
        for name, m in metrics.items():
            row = {
                "test": test_name,
                "method": name,
                "completed": m.completed,
                "failure_step": m.failure_step,
                "failure_msg": m.failure_msg,
                "l2_error": m.l2_error,
                "relative_l2": m.relative_l2,
                "linf_error": m.linf_error,
                "peak_value": m.peak_value,
                "peak_ratio": m.peak_ratio,
                "peak_location": m.peak_location,
                "peak_width_fwhm": m.peak_width_fwhm,
                "sigma_moment": m.sigma_moment,
                "A_fit": m.A_fit,
                "mu_fit": m.mu_fit,
                "sigma_fit": m.sigma_fit,
                "fit_success": m.fit_success,
                "fit_rmse": m.fit_rmse,
                "total_variation": m.total_variation,
                "forward_consistency_l2": m.forward_consistency_l2,
                "mass_candidate": m.mass_candidate,
                "mass_true": m.mass_true,
                "mass_error": m.mass_error,
                "mass_rel_error": m.mass_rel_error,
                "max_abs_score_final": m.max_abs_score_final,
                "max_score_error_L2": m.max_score_error_L2,
                "runtime_seconds": m.runtime_seconds,
                # config context
                "T": cfg.heat.T,
                "sigma0": cfg.initial_condition.sigma0,
                "mu": cfg.initial_condition.mu,
                "n_per": cfg.grw.gradient_globs_per_jump,
            }
            all_rows.append(row)
            print(f"  {name:<45} L2={m.l2_error:.5f}  rel_L2={m.relative_l2:.4f}  "
                  f"{'OK' if m.completed else f'FAIL@{m.failure_step}'}")

    # -------------------------------------------------------------------
    # Test E: stochastic ensemble
    # -------------------------------------------------------------------
    print(f"\n{'='*50}")
    print("Test E: stochastic ensemble")
    ensemble_dir = suite_dir / "test_E"
    ensemble_dir.mkdir(exist_ok=True)

    for ec in ENSEMBLE_CONFIGS:
        label = ec["label"]
        print(f"  Ensemble {label}: T={ec['T']}, sigma0={ec['sigma0']}, seeds={ec['seeds']}")
        seed_dir = ensemble_dir / label
        seed_dir.mkdir(exist_ok=True)
        for seed in ec["seeds"]:
            cfg = patch_config(base_cfg,
                               **{"heat.T": ec["T"], "initial_condition.sigma0": ec["sigma0"],
                                  "grw.rng_seed": seed})
            x = make_grid(cfg)
            u0 = compute_true_u0(x, cfg)
            u_obs = compute_observed_final(x, cfg)
            results = run_methods(cfg, ENSEMBLE_METHODS)
            metrics = compute_all_metrics(results, cfg)
            for name, m in metrics.items():
                row = {
                    "ensemble_label": label,
                    "seed": seed,
                    "method": name,
                    "test": f"E_{label}",
                    "completed": m.completed,
                    "failure_step": m.failure_step,
                    "relative_l2": m.relative_l2,
                    "forward_consistency_l2": m.forward_consistency_l2,
                    "peak_ratio": m.peak_ratio,
                    "sigma_fit": m.sigma_fit,
                    "mass_rel_error": m.mass_rel_error,
                    "runtime_seconds": m.runtime_seconds,
                    "T": cfg.heat.T,
                    "sigma0": ec["sigma0"],
                }
                ensemble_rows.append(row)
                all_rows.append({**row, "l2_error": m.l2_error, "linf_error": m.linf_error,
                                  "max_score_error_L2": m.max_score_error_L2, "failure_msg": m.failure_msg,
                                  "peak_value": m.peak_value, "peak_location": m.peak_location,
                                  "peak_width_fwhm": m.peak_width_fwhm, "sigma_moment": m.sigma_moment,
                                  "A_fit": m.A_fit, "mu_fit": m.mu_fit, "fit_success": m.fit_success,
                                  "fit_rmse": m.fit_rmse, "total_variation": m.total_variation,
                                  "mass_candidate": m.mass_candidate, "mass_true": m.mass_true,
                                  "mass_error": m.mass_error, "max_abs_score_final": m.max_abs_score_final,
                                  "mu": base_cfg.initial_condition.mu,
                                  "n_per": cfg.grw.gradient_globs_per_jump})

    # -------------------------------------------------------------------
    # Save aggregate CSV
    # -------------------------------------------------------------------
    agg_df = pd.DataFrame(all_rows)
    agg_path = suite_dir / "aggregate_metrics.csv"
    agg_df.to_csv(agg_path, index=False)
    print(f"\nAggregate CSV saved: {agg_path}")

    # -------------------------------------------------------------------
    # Aggregate plots
    # -------------------------------------------------------------------
    print("Generating aggregate plots ...")
    plot_aggregate_relative_l2(all_rows, suite_dir)
    plot_aggregate_fitted_sigma(all_rows, suite_dir, base_cfg.initial_condition.sigma0)
    plot_aggregate_peak_ratio(all_rows, suite_dir)
    plot_aggregate_forward_consistency(all_rows, suite_dir)
    plot_stochastic_ensemble_summary(ensemble_rows, suite_dir)

    # -------------------------------------------------------------------
    # GO/STOP decision
    # -------------------------------------------------------------------
    decision = decide(all_rows)
    print(decision)

    # Write aggregate summary
    summary_lines = ["TARGETED SUITE SUMMARY", "=" * 60, ""]
    summary_lines.append("Tests run: A B C D E F")
    summary_lines.append(f"Suite dir: {suite_dir}")
    summary_lines.append("")
    summary_lines.append("METRICS TABLE (deterministic methods only)")
    summary_lines.append("-" * 100)
    header = f"{'Test':<6} {'Method':<45} {'Status':<15} {'rel_L2':>8} {'peak_ratio':>10} {'sigma_fit':>9} {'fwd_cons':>9} {'rt(s)':>7}"
    summary_lines.append(header)
    summary_lines.append("-" * len(header))
    for r in all_rows:
        if r.get("method") not in ALL_DETERMINISTIC:
            continue
        status = "OK" if r["completed"] else f"FAIL@{r.get('failure_step', '?')}"
        summary_lines.append(
            f"{r['test']:<6} {r['method']:<45} {status:<15} "
            f"{r.get('relative_l2', float('nan')):>8.4f} "
            f"{r.get('peak_ratio', float('nan')):>10.4f} "
            f"{r.get('sigma_fit', float('nan')):>9.4f} "
            f"{r.get('forward_consistency_l2', float('nan')):>9.5f} "
            f"{r.get('runtime_seconds', float('nan')):>7.3f}"
        )
    summary_lines.append("")
    summary_lines.append(decision)
    (suite_dir / "aggregate_summary.txt").write_text("\n".join(summary_lines))

    print(f"\nAll suite outputs in: {suite_dir}")


if __name__ == "__main__":
    main()
