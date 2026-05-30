"""
Tasks 4 + 5 – Gaussian mixture tests (G & H) + verification summary.

Test G: gaussian_mixture, T=0.05, all 5 methods, n_per=40, seed=0
Test H: same mixture IC but T=0.15

Extra metrics (beyond standard targeted suite):
  - Wasserstein-1 distance between candidate and true u₀ (treated as densities)
  - Component peak recovery: height and location of each mixture peak in candidate

Outputs
-------
  outputs/mixture_tests_TIMESTAMP/
    mixture_metrics.csv
    mixture_summary.txt
    field comparison plots  (per method, tests G and H)
  outputs/verification_pass_TIMESTAMP/
    verification_summary.txt  (5 structured questions)
"""

import copy
import csv
import datetime
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from invheat_grw.config import (
    load_config,
    InitialConditionConfig,
    GaussianComponentConfig,
    HeatConfig,
)
from invheat_grw.fields import true_u0, exact_heat_solution
from invheat_grw.methods import (
    run_naive_backward,
    run_oracle_score_deterministic,
    run_oracle_score_stochastic,
    run_estimated_score_deterministic_raw,
    run_estimated_score_stochastic_raw,
)


def _rel_l2(candidate, u_true, x_grid):
    dx = x_grid[1] - x_grid[0]
    diff = candidate - u_true
    denom = float(np.sqrt(dx * np.sum(u_true ** 2)))
    return float(np.sqrt(dx * np.sum(diff ** 2))) / denom if denom > 0 else float("nan")


def _linf(candidate, u_true):
    return float(np.max(np.abs(candidate - u_true)))

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

MIXTURE_IC = InitialConditionConfig(
    type="gaussian_mixture",
    background=0.05,
    components=[
        GaussianComponentConfig(amplitude=0.75, mu=0.35, sigma0=0.05),
        GaussianComponentConfig(amplitude=0.45, mu=0.62, sigma0=0.08),
    ],
)


def make_mixture_cfg(base_cfg, T: float):
    cfg = copy.deepcopy(base_cfg)
    cfg.initial_condition = copy.deepcopy(MIXTURE_IC)
    cfg.heat.T = T
    cfg.domain.n_grid = 300
    cfg.grw.gradient_globs_per_jump = 40
    cfg.grw.rng_seed = 0
    return cfg


# ---------------------------------------------------------------------------
# Per-method run
# ---------------------------------------------------------------------------

def run_all_methods(cfg, x_grid):
    u_obs = exact_heat_solution(x_grid, cfg.heat.T, cfg)
    seed = cfg.grw.rng_seed
    results = {}
    results["naive"] = run_naive_backward(
        u_obs, x_grid, cfg, np.random.default_rng(seed))
    results["oracle_det"] = run_oracle_score_deterministic(
        u_obs, x_grid, cfg, np.random.default_rng(seed))
    results["oracle_stoch"] = run_oracle_score_stochastic(
        u_obs, x_grid, cfg, np.random.default_rng(seed))
    results["est_det"] = run_estimated_score_deterministic_raw(
        u_obs, x_grid, cfg, np.random.default_rng(seed))
    results["est_stoch"] = run_estimated_score_stochastic_raw(
        u_obs, x_grid, cfg, np.random.default_rng(seed))
    return results


# ---------------------------------------------------------------------------
# Mixture-specific metrics
# ---------------------------------------------------------------------------

def wasserstein_metric(candidate, u_true, x_grid):
    """
    Wasserstein-1 distance treating candidate and u_true as un-normalised
    1-D densities (normalise to mass-1 before computing).
    Returns nan if either density is non-positive.
    """
    dx = x_grid[1] - x_grid[0]
    mass_c = np.sum(np.maximum(candidate, 0.0)) * dx
    mass_t = np.sum(np.maximum(u_true,    0.0)) * dx
    if mass_c <= 0.0 or mass_t <= 0.0:
        return float("nan")
    p = np.maximum(candidate, 0.0) / mass_c
    q = np.maximum(u_true,    0.0) / mass_t
    return float(wasserstein_distance(x_grid, x_grid, p, q))


def peak_recovery(candidate, x_grid, components):
    """
    For each mixture component, find the nearest local maximum in candidate
    within ±3*sigma0 of mu.  Return list of (mu_true, peak_x, peak_val).
    """
    dx = x_grid[1] - x_grid[0]
    recovered = []
    for comp in components:
        lo = comp.mu - 3.0 * comp.sigma0
        hi = comp.mu + 3.0 * comp.sigma0
        mask = (x_grid >= lo) & (x_grid <= hi)
        if not np.any(mask):
            recovered.append((comp.mu, float("nan"), float("nan")))
            continue
        idx = np.argmax(candidate[mask])
        x_sub = x_grid[mask]
        recovered.append((comp.mu, float(x_sub[idx]), float(candidate[mask][idx])))
    return recovered


def compute_all_metrics(candidate, u_true0, x_grid, cfg):
    rl2 = _rel_l2(candidate, u_true0, x_grid)
    linf = _linf(candidate, u_true0)
    wd = wasserstein_metric(candidate, u_true0, x_grid)
    peaks = peak_recovery(candidate, x_grid, cfg.initial_condition.components)
    return rl2, linf, wd, peaks


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(out_dir, test_name, method_name, x_grid, u_true0, u_obs, candidate, cfg):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x_grid, u_true0,  "k--", lw=2, label="true u₀")
    ax.plot(x_grid, u_obs,    "r:",  lw=1.5, label=f"u_obs (T={cfg.heat.T})")
    if candidate is not None:
        ax.plot(x_grid, candidate, "b-", lw=2, label="candidate")
    # Mark true mixture peaks
    for comp in cfg.initial_condition.components:
        ax.axvline(comp.mu, color="grey", ls="--", lw=0.8, alpha=0.5)
    ax.set_title(f"{test_name} | {method_name}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, f"field_{test_name}_{method_name}.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    config_dir = os.path.join(base_dir, "configs")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base_dir, "outputs", f"mixture_tests_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    vp_dir  = os.path.join(base_dir, "outputs", f"verification_pass_{ts}")
    os.makedirs(vp_dir, exist_ok=True)

    # Load base config just for infrastructure (domain, alpha, dt, grw defaults)
    cfg_base = load_config(os.path.join(config_dir, "gaussian_base.yaml"))

    TESTS = [
        ("G", 0.05),
        ("H", 0.15),
    ]
    METHODS = ["naive", "oracle_det", "oracle_stoch", "est_det", "est_stoch"]

    rows = []
    peak_rows = []
    all_results = {}  # test_name -> method -> result

    for test_name, T in TESTS:
        cfg = make_mixture_cfg(cfg_base, T)
        x_grid = np.linspace(cfg.domain.x_min, cfg.domain.x_max, cfg.domain.n_grid)
        u_true0 = true_u0(x_grid, cfg)
        u_obs   = exact_heat_solution(x_grid, cfg.heat.T, cfg)

        results = run_all_methods(cfg, x_grid)
        all_results[test_name] = results

        for method_name, result in results.items():
            if result.completed:
                rl2, linf, wd, peaks = compute_all_metrics(result.candidate, u_true0, x_grid, cfg)
                candidate = result.candidate
            else:
                rl2 = linf = wd = float("nan")
                peaks = [(c.mu, float("nan"), float("nan"))
                         for c in cfg.initial_condition.components]
                candidate = None

            rows.append({
                "test":          test_name,
                "T":             T,
                "method":        method_name,
                "completed":     result.completed,
                "failure_step":  result.failure_step,
                "rel_L2":        rl2,
                "Linf":          linf,
                "wasserstein":   wd,
                "suspicious_score_steps": len(result.score_suspicious_steps),
            })

            for mu_true, x_peak, v_peak in peaks:
                peak_rows.append({
                    "test": test_name, "method": method_name,
                    "mu_true": mu_true, "peak_x": x_peak, "peak_val": v_peak,
                })

            plot_comparison(out_dir, test_name, method_name,
                            x_grid, u_true0, u_obs, candidate, cfg)

    # -----------------------------------------------------------------------
    # Write CSV
    # -----------------------------------------------------------------------
    csv_path = os.path.join(out_dir, "mixture_metrics.csv")
    fieldnames = ["test", "T", "method", "completed", "failure_step",
                  "rel_L2", "Linf", "wasserstein", "suspicious_score_steps"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    peaks_csv = os.path.join(out_dir, "mixture_peaks.csv")
    with open(peaks_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["test","method","mu_true","peak_x","peak_val"])
        writer.writeheader()
        writer.writerows(peak_rows)

    # -----------------------------------------------------------------------
    # Mixture summary
    # -----------------------------------------------------------------------
    summary_path = os.path.join(out_dir, "mixture_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("GAUSSIAN MIXTURE TESTS  (G = T=0.05, H = T=0.15)\n")
        f.write(f"Timestamp: {ts}\n")
        f.write("=" * 70 + "\n\n")
        for test_name, T in TESTS:
            f.write(f"\n--- Test {test_name}  (T={T}) ---\n")
            f.write(f"{'Method':<24}  {'Cmplt':5}  {'rel_L2':>9}  {'Linf':>9}  {'Wass':>9}\n")
            f.write("-" * 62 + "\n")
            for row in rows:
                if row["test"] == test_name:
                    ok = "YES" if row["completed"] else "NO"
                    f.write(f"  {row['method']:<22}  {ok:5}  "
                            f"{row['rel_L2']:9.4f}  {row['Linf']:9.4f}  "
                            f"{row['wasserstein']:9.4f}\n")

    # -----------------------------------------------------------------------
    # Verification summary (5 structured questions)
    # -----------------------------------------------------------------------

    # Gather key numbers for Q answers
    def lookup(test, method, field):
        for row in rows:
            if row["test"] == test and row["method"] == method:
                return row.get(field, float("nan"))
        return float("nan")

    # Q1: Did estimated raw accidentally depend on oracle?
    #   → Already answered by sanity checks (run_sanity_checks.py).
    #   Here we just note the isolation: oracle_mode and estimated_mode are applied
    #   after error recording, so est score error is computed from unmodified s_est.
    q1_answer = (
        "The ablation modes (oracle_zero/flip) are applied inside _run_integration "
        "AFTER the score-error comparison block.  Therefore the estimated score error "
        "metrics are always computed against the natural (unmodified) oracle, regardless "
        "of the oracle_mode override.  The sanity check script (run_sanity_checks.py) "
        "quantitatively verifies that est_det_raw rel_L2 is unchanged (≤0.5%) when "
        "oracle_mode is zeroed or flipped."
    )

    # Q2: Did ablations behave as expected?
    #   → Summarised per pass/fail in sanity_summary.txt; here give the logic.
    q2_answer = (
        "Expected behaviour:\n"
        "  oracle_zero/flip   → oracle_det  should worsen (no drift or reversed drift);\n"
        "                        est_det_raw must be unchanged (mode-independent).\n"
        "  estimated_zero/flip → est_det_raw should worsen (score forced wrong);\n"
        "                         oracle_det must be unchanged.\n"
        "See sanity_checks_*/sanity_summary.txt for per-config PASS/FAIL results."
    )

    # Q3: Does estimated raw work on mixture?
    g_est_rl2  = lookup("G", "est_det",  "rel_L2")
    g_or_rl2   = lookup("G", "oracle_det","rel_L2")
    h_est_rl2  = lookup("H", "est_det",  "rel_L2")
    h_or_rl2   = lookup("H", "oracle_det","rel_L2")
    g_est_wd   = lookup("G", "est_det",  "wasserstein")
    h_est_wd   = lookup("H", "est_det",  "wasserstein")
    q3_answer = (
        f"Test G (T=0.05): est_det_raw rel_L2={g_est_rl2:.4f}  vs  oracle_det={g_or_rl2:.4f}  "
        f"| Wasserstein={g_est_wd:.4f}\n"
        f"Test H (T=0.15): est_det_raw rel_L2={h_est_rl2:.4f}  vs  oracle_det={h_or_rl2:.4f}  "
        f"| Wasserstein={h_est_wd:.4f}\n"
        "A finite rel_L2 below ~1.0 and low Wasserstein indicate the raw score correctly "
        "guides backward diffusion on the mixture without regularization."
    )

    # Q4: Is regularization a refinement or a rescue?
    # Rescue = est_det_raw already fails (rel_L2 > 1 or not completed)
    # Refinement = est_det_raw works but is worse than oracle
    def verdict(rl2, completed):
        if not completed:
            return "rescue needed (method failed)"
        if rl2 > 0.5:
            return "rescue likely needed (rel_L2 > 0.5)"
        if rl2 > 0.1:
            return "borderline – regularization would improve robustness"
        return "refinement (raw score already works)"

    q4_answer = (
        f"Test G: {verdict(g_est_rl2, lookup('G','est_det','completed'))}\n"
        f"Test H: {verdict(h_est_rl2, lookup('H','est_det','completed'))}\n"
        "If raw score already achieves low rel_L2 and low Wasserstein, regularization "
        "(e.g., epsilon-clipping, KDE, Tikhonov) would be a refinement to improve "
        "stability in tails.  If the raw method fails or degrades badly, it is a rescue."
    )

    # Q5: What to add next?
    q5_answer = (
        "Recommended next steps (Phase 5):\n"
        "  1. Add epsilon-floor regularization to estimated score: s = u_x / (u + eps).\n"
        "     Sweep eps ∈ {1e-4, 1e-3, 1e-2} on Tests B, G, H and compare rel_L2.\n"
        "  2. Add KDE smoothing of the field before score estimation.\n"
        "  3. Evaluate sensitivity to n_grid (100, 200, 300, 500) for mixture IC.\n"
        "  4. Extend to non-reflecting (periodic or absorbing) boundaries.\n"
        "  5. Add Wasserstein to the targeted suite as a standard metric.\n"
        "Blocking criterion: proceed to Phase 5 only if both Test G and H pass "
        "(est_det_raw rel_L2 < oracle_det rel_L2 × 2.0 and Wasserstein finite)."
    )

    vp_path = os.path.join(vp_dir, "verification_summary.txt")
    with open(vp_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("VERIFICATION PASS SUMMARY\n")
        f.write(f"Timestamp: {ts}\n")
        f.write("=" * 70 + "\n\n")

        f.write("Q1: Did the estimated raw score accidentally depend on the oracle?\n")
        f.write(f"A1: {q1_answer}\n\n")

        f.write("Q2: Did the score ablations behave as expected?\n")
        f.write(f"A2: {q2_answer}\n\n")

        f.write("Q3: Does the estimated raw score method work on a Gaussian mixture IC?\n")
        f.write(f"A3: {q3_answer}\n\n")

        f.write("Q4: Is regularization a refinement or a rescue at this stage?\n")
        f.write(f"A4: {q4_answer}\n\n")

        f.write("Q5: What should be added in the next phase?\n")
        f.write(f"A5: {q5_answer}\n")

    # Console output
    print(f"\nMixture tests complete → {out_dir}")
    print(f"Verification summary  → {vp_path}\n")
    print(f"{'Test':<6}  {'Method':<24}  {'Cmplt':5}  {'rel_L2':>9}  {'Wass':>9}")
    print("-" * 60)
    for row in rows:
        ok = "YES" if row["completed"] else "NO"
        print(f"  {row['test']:<5}  {row['method']:<24}  {ok:5}  "
              f"{row['rel_L2']:9.4f}  {row['wasserstein']:9.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
