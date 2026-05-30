"""
run_all.py — Run all backward GRW experiments defined in the config.

Usage:
    python scripts/run_all.py --config configs/gaussian_base.yaml

Creates a timestamped output directory under outputs/ containing:
  - all PNG plots
  - metrics_summary.csv
  - run_summary.txt
  - config_used.yaml
  - fields.csv, arrays.npz
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from invheat_grw.config import load_config
from invheat_grw.fields import make_grid, true_u0 as compute_true_u0, observed_final as compute_observed_final
from invheat_grw.globs import field_to_globs, reconstruct_field, GlobState
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
    plot_method_panels,
    plot_step_snapshots_oracle_det,
    plot_step_snapshots_estimated_det_raw,
    plot_metrics_bar_l2,
    plot_peak_width_tv,
    plot_score_diagnostics,
    plot_forward_consistency,
    plot_naive_vs_oracle,
    plot_residuals,
)
from invheat_grw.io_utils import (
    make_output_dir,
    save_config,
    save_metrics_csv,
    save_arrays,
    write_run_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all inverse heat score GRW experiments.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    cfg = load_config(config_path)

    # Locate outputs/ relative to project root (parent of scripts/)
    project_root = Path(__file__).resolve().parent.parent
    outputs_root = project_root / "outputs"
    outputs_root.mkdir(exist_ok=True)

    out_dir = make_output_dir(outputs_root)
    print(f"Output directory: {out_dir}")

    # -----------------------------------------------------------------------
    # Set up RNG (deterministic seed)
    # -----------------------------------------------------------------------
    rng = np.random.default_rng(cfg.grw.rng_seed)

    # -----------------------------------------------------------------------
    # Build fields
    # -----------------------------------------------------------------------
    x = make_grid(cfg)
    u0 = compute_true_u0(x, cfg)
    u_obs = compute_observed_final(x, cfg)

    # Verify step-0 reconstruction (sanity check for glob encoding)
    init_state = field_to_globs(u_obs, x, cfg)
    step0_recon = reconstruct_field(init_state, x)
    dx = x[1] - x[0]
    recon_err = float(np.sqrt(dx * np.sum((step0_recon - u_obs) ** 2)))
    print(f"Step-0 reconstruction L2 error: {recon_err:.6f}  (should be small)")

    # -----------------------------------------------------------------------
    # Run experiments
    # -----------------------------------------------------------------------
    exp = cfg.experiments
    results: dict[str, MethodResult] = {}

    def run_with_fresh_rng(fn, name):
        """Each method gets its own sub-RNG seeded from the global seed."""
        method_rng = np.random.default_rng(cfg.grw.rng_seed + hash(name) % (2**31))
        print(f"  Running: {name} ...", end=" ", flush=True)
        res = fn(u_obs, x, cfg, method_rng)
        status = "OK" if res.completed else f"FAILED at step {res.failure_step}: {res.failure_msg}"
        print(status)
        return res

    if exp.run_naive_backward:
        results["naive_backward"] = run_with_fresh_rng(run_naive_backward, "naive_backward")

    if exp.run_oracle_score_deterministic:
        results["oracle_score_deterministic"] = run_with_fresh_rng(
            run_oracle_score_deterministic, "oracle_score_deterministic")

    if exp.run_oracle_score_stochastic:
        results["oracle_score_stochastic"] = run_with_fresh_rng(
            run_oracle_score_stochastic, "oracle_score_stochastic")

    if exp.run_estimated_score_deterministic_raw:
        results["estimated_score_deterministic_raw"] = run_with_fresh_rng(
            run_estimated_score_deterministic_raw, "estimated_score_deterministic_raw")

    if exp.run_estimated_score_stochastic_raw:
        results["estimated_score_stochastic_raw"] = run_with_fresh_rng(
            run_estimated_score_stochastic_raw, "estimated_score_stochastic_raw")

    # -----------------------------------------------------------------------
    # Compute metrics
    # -----------------------------------------------------------------------
    metrics_list: list[MethodMetrics] = []
    for name, res in results.items():
        m = compute_metrics(res, u0, u_obs, x, cfg)
        metrics_list.append(m)

    # Reference metrics for true_u0 and observed_final
    from invheat_grw.metrics import forward_heat_solve_dct
    true_peak = float(np.max(u0))
    true_peak_loc = float(x[np.argmax(u0)])
    true_fwhm = compute_fwhm(u0, x)
    true_tv = float(np.sum(np.abs(np.diff(u0))))
    obs_peak = float(np.max(u_obs))
    obs_fwhm = compute_fwhm(u_obs, x)
    obs_tv = float(np.sum(np.abs(np.diff(u_obs))))

    # Wrap reference values as pseudo-MethodMetrics for summary text
    from invheat_grw.metrics import MethodMetrics as MM

    nan = float("nan")
    true_metrics = MM(
        method_name="true_u0", completed=True, failure_step=None, failure_msg="",
        l2_error=0.0, relative_l2=0.0, linf_error=0.0,
        peak_value=true_peak, peak_ratio=1.0, peak_location=true_peak_loc,
        peak_width_fwhm=true_fwhm, sigma_moment=nan, width_moment=nan,
        A_fit=nan, mu_fit=nan, sigma_fit=nan, fit_success=False, fit_rmse=nan,
        total_variation=true_tv, forward_consistency_l2=0.0,
        mass_candidate=nan, mass_true=nan, mass_error=0.0, mass_rel_error=0.0,
        max_abs_score_final=nan, max_score_error_L2=nan, runtime_seconds=0.0,
    )
    obs_metrics = MM(
        method_name="observed_final", completed=True, failure_step=None, failure_msg="",
        l2_error=nan, relative_l2=nan, linf_error=nan,
        peak_value=obs_peak, peak_ratio=nan, peak_location=true_peak_loc,
        peak_width_fwhm=obs_fwhm, sigma_moment=nan, width_moment=nan,
        A_fit=nan, mu_fit=nan, sigma_fit=nan, fit_success=False, fit_rmse=nan,
        total_variation=obs_tv, forward_consistency_l2=nan,
        mass_candidate=nan, mass_true=nan, mass_error=nan, mass_rel_error=nan,
        max_abs_score_final=nan, max_score_error_L2=nan, runtime_seconds=0.0,
    )

    # -----------------------------------------------------------------------
    # Save artifacts
    # -----------------------------------------------------------------------
    print("Saving artifacts ...")
    save_config(cfg, config_path, out_dir)
    save_metrics_csv(metrics_list, out_dir)
    save_arrays(x, u0, u_obs, results, out_dir)
    write_run_summary(cfg, metrics_list, true_metrics, obs_metrics, out_dir)

    # -----------------------------------------------------------------------
    # Save step-0 reconstruction info
    # -----------------------------------------------------------------------
    import pandas as pd
    recon_df = pd.DataFrame({"x": x, "observed_final": u_obs, "step0_recon": step0_recon})
    recon_df.to_csv(out_dir / "step0_recon.csv", index=False)

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    print("Generating plots ...")
    plot_field_comparison_all(x, u0, u_obs, results, out_dir)
    plot_method_panels(x, u0, u_obs, results, out_dir)
    plot_step_snapshots_oracle_det(x, u0, u_obs, results, out_dir)
    plot_step_snapshots_estimated_det_raw(x, u0, u_obs, results, out_dir)
    plot_metrics_bar_l2(metrics_list, out_dir)
    plot_peak_width_tv(metrics_list, true_peak, true_fwhm, true_tv, out_dir)
    plot_score_diagnostics(results, out_dir)
    plot_forward_consistency(metrics_list, out_dir)
    plot_naive_vs_oracle(x, u0, u_obs, results, out_dir)
    plot_residuals(x, u0, results, out_dir)

    # -----------------------------------------------------------------------
    # Print summary to console
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Method':<40} {'Status':<15} {'L2 error':>10} {'rel_L2':>8}")
    print("-" * 75)
    for m in metrics_list:
        status = "OK" if m.completed else f"FAIL@{m.failure_step}"
        print(f"{m.method_name:<40} {status:<15} {m.l2_error:>10.5f} {m.relative_l2:>8.4f}")

    print(f"\nStep-0 reconstruction L2 error: {recon_err:.6f}")
    print(f"\nAll outputs saved to: {out_dir}")
    print(f"  - metrics_summary.csv")
    print(f"  - run_summary.txt")
    print(f"  - config_used.yaml")
    print(f"  - *.png  (10 plots)")
    print(f"  - arrays.npz, fields.csv, step0_recon.csv")


if __name__ == "__main__":
    main()
