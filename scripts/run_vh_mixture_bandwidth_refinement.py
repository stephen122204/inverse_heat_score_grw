"""
run_vh_mixture_bandwidth_refinement.py

Manuscript labels (draft): `fig:vh_mixture` (`sec:mixture_analysis`).

Final pre-writeup bandwidth refinement for the VH (variable-coefficient heat,
Gaussian mixture IC) case.

Sweeps smoothed_log bandwidth factors [2, 3, 4, 5, 6] for:
  - beta = 0.5  (required)
  - beta = 0.9  (run if --also-beta09 passed or if beta=0.5 finishes quickly)

PDE:  u_t = ∂_x( a(x) u_x )    with   a(x) = alpha0 * (1 + beta * sin(2πx))
Reverse velocity:  Δx = a(x) * ∂_x log u(x, t) * dt   (NO a'(x) term)
Parameters:  alpha0=0.01, T=0.15, dt=0.001, n_grid=400, n_particles=10000

Reference baselines (from variable_coefficient_audit, 2026-05-30):
  Oracle           VH beta=0.5 : 0.0201    beta=0.9 : 0.0204
  Tikhonov         VH beta=0.5 : 0.0055    beta=0.9 : 0.0055
  sl_bw4 (audit)   VH beta=0.5 : 0.0794    beta=0.9 : 0.0877

STOP CONDITION: This is the final pre-writeup experiment.
No adaptive bandwidth, no N=20000, no new methods after this run.

Outputs
-------
outputs/vh_mixture_bandwidth_refinement_<TIMESTAMP>/
    vh_mixture_bandwidth_metrics.csv
    vh_mixture_bandwidth_summary.txt
    rel_L2_vs_bandwidth_VH_beta05.png
    field_comparison_best_VH_beta05.png
    residual_best_VH_beta05.png
    rel_L2_vs_bandwidth_VH_beta09.png          (if beta=0.9 run)
    field_comparison_best_VH_beta09.png        (if beta=0.9 run)
    residual_best_VH_beta09.png                (if beta=0.9 run)
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — import helpers from the audit script (no code duplication)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))   # so we can import the audit module

import run_variable_coefficient_audit as vc  # noqa: E402  (script import)

from invheat_grw.config import load_config
from invheat_grw.fields import make_grid, true_u0 as compute_true_u0

# ---------------------------------------------------------------------------
# Constants matching the audit
# ---------------------------------------------------------------------------
ALPHA0       = 0.01
T            = 0.15
DT           = 0.001
N_GRID       = 400
N_PARTICLES  = 10_000
EPSILON      = 1e-8

BANDWIDTH_FACTORS = [2, 3, 4, 5, 6]

# Reference baselines from the prior audit run (hard-coded for display)
PRIOR_AUDIT = {
    "VH_beta05": {"oracle": 0.0201, "tikhonov": 0.0055, "sl_bw4_audit": 0.0794},
    "VH_beta09": {"oracle": 0.0204, "tikhonov": 0.0055, "sl_bw4_audit": 0.0877},
}

MIXTURE_CONFIG = REPO_ROOT / "configs" / "gaussian_mixture.yaml"


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------
def _rel_l2(cand: np.ndarray, ref: np.ndarray, x_grid: np.ndarray) -> float:
    dx = x_grid[1] - x_grid[0]
    norm = float(np.sqrt(dx * np.sum(ref ** 2)))
    return float(np.sqrt(dx * np.sum((cand - ref) ** 2))) / (norm + 1e-300)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_rel_l2_vs_bw(
    bw_factors: list[int],
    rel_l2_values: list[float],
    oracle_rel_l2: float,
    tikhonov_ref: float,
    audit_bw4_ref: float,
    case_label: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(bw_factors, rel_l2_values, "o-", color="tab:blue", lw=2,
            markersize=7, label="smoothed_log (this run)")

    # Mark the prior bw=4 audit value
    ax.axhline(audit_bw4_ref, color="tab:blue", lw=1, ls=":", alpha=0.6,
               label=f"audit sl_bw4 = {audit_bw4_ref:.4f}")

    ax.axhline(oracle_rel_l2, color="k", lw=1.5, ls="--",
               label=f"oracle = {oracle_rel_l2:.4f}")
    ax.axhline(tikhonov_ref, color="tab:red", lw=1.5, ls="--",
               label=f"Tikhonov = {tikhonov_ref:.4f}")
    ax.axhline(0.060, color="tab:green", lw=1, ls=":", alpha=0.7,
               label="target 0.060")
    ax.axhline(0.070, color="gray", lw=1, ls=":", alpha=0.7,
               label="threshold 0.070")

    ax.set_xticks(bw_factors)
    ax.set_xlabel("bandwidth factor")
    ax.set_ylabel("rel L2 vs true u₀")
    ax.set_title(f"Bandwidth sweep — {case_label}")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_field_comparison_best(
    x_grid: np.ndarray,
    true_u0: np.ndarray,
    u_obs: np.ndarray,
    best_candidate: np.ndarray,
    oracle_candidate: np.ndarray,
    best_bw: int,
    best_rel_l2: float,
    oracle_rel_l2: float,
    case_label: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_grid, true_u0, "k-", lw=2, label="True u₀")
    ax.plot(x_grid, u_obs, "k--", lw=1.2, alpha=0.5, label="Observed u(T)")
    ax.plot(x_grid, oracle_candidate, color="tab:orange", lw=1.5,
            label=f"Oracle (rel_L2={oracle_rel_l2:.4f})", alpha=0.85)
    ax.plot(x_grid, best_candidate, color="tab:blue", lw=1.5,
            label=f"sl_bw{best_bw} (rel_L2={best_rel_l2:.4f})")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.set_title(f"Best reconstruction — {case_label}")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_residual(
    x_grid: np.ndarray,
    true_u0: np.ndarray,
    best_candidate: np.ndarray,
    oracle_candidate: np.ndarray,
    best_bw: int,
    case_label: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x_grid, best_candidate - true_u0, color="tab:blue",
            lw=1.5, label=f"sl_bw{best_bw} residual")
    ax.plot(x_grid, oracle_candidate - true_u0, color="tab:orange",
            lw=1.5, label="oracle residual", alpha=0.7)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("x")
    ax.set_ylabel("candidate − true u₀")
    ax.set_title(f"Residuals — {case_label}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Run a single beta case
# ---------------------------------------------------------------------------
def run_case(
    beta: float,
    cfg,
    x_grid: np.ndarray,
    u0_true: np.ndarray,
    out_dir: Path,
    n_particles: int,
) -> tuple[list[dict], list[float], int, float, float]:
    """
    Run oracle + smoothed_log bw sweep for one beta value.
    Returns (rows, bw_rel_l2, best_bw, best_rel_l2, oracle_rel_l2).
    """
    case_id = f"VH_beta{'05' if abs(beta - 0.5) < 0.01 else '09'}"
    beta_tag = "beta05" if abs(beta - 0.5) < 0.01 else "beta09"
    alpha0 = ALPHA0
    dt = DT
    n_steps = round(T / dt)

    print(f"\n{'='*60}")
    print(f"[CASE] {case_id}  beta={beta}  IC=mixture  alpha0={alpha0}"
          f"  T={T}  dt={dt}  n_steps={n_steps}")

    # Forward solve
    print("[FORWARD] Solving variable-coeff heat equation...")
    t0 = time.perf_counter()
    u_obs, snapshots = vc.solve_varcoeff_forward(u0_true, x_grid, alpha0, beta, dt, n_steps)
    print(f"  Done in {time.perf_counter()-t0:.2f}s  "
          f"u_obs=[{u_obs.min():.4f}, {u_obs.max():.4f}]")

    rows: list[dict] = []
    bw_rel_l2: list[float] = []
    bw_candidates: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Oracle baseline (single run for this beta)
    # ------------------------------------------------------------------
    print("[METHOD] oracle")
    r_oracle = vc.run_varcoeff_oracle(
        u_obs, x_grid, snapshots, alpha0, beta, dt, n_steps,
        n_particles=n_particles,
        x_min=float(cfg.domain.x_min), x_max=float(cfg.domain.x_max)
    )
    oracle_rel_l2 = _rel_l2(r_oracle["candidate"], u0_true, x_grid)
    oracle_candidate = r_oracle["candidate"].copy()
    print(f"  rel_L2={oracle_rel_l2:.4f}  t={r_oracle['runtime_seconds']:.1f}s")

    rows.append({
        "case": case_id,
        "beta": beta,
        "method": "oracle",
        "bandwidth_factor": None,
        "epsilon": None,
        "relative_l2": oracle_rel_l2,
        "completed": r_oracle["completed"],
        "runtime_seconds": r_oracle["runtime_seconds"],
    })

    # ------------------------------------------------------------------
    # smoothed_log bandwidth sweep
    # ------------------------------------------------------------------
    for bw in BANDWIDTH_FACTORS:
        print(f"[METHOD] smoothed_log  bw={bw}  eps={EPSILON:.0e}")
        r = vc.run_varcoeff_estimated(
            u_obs, x_grid, snapshots, alpha0, beta, dt, n_steps,
            score_method="smoothed_log",
            bandwidth_factor=float(bw),
            n_particles=n_particles,
            epsilon=EPSILON,
            x_min=float(cfg.domain.x_min), x_max=float(cfg.domain.x_max),
        )
        rel_l2 = _rel_l2(r["candidate"], u0_true, x_grid)
        bw_rel_l2.append(rel_l2)
        bw_candidates[bw] = r["candidate"].copy()

        print(f"  rel_L2={rel_l2:.4f}  completed={r['completed']}"
              f"  t={r['runtime_seconds']:.1f}s")

        rows.append({
            "case": case_id,
            "beta": beta,
            "method": "smoothed_log",
            "bandwidth_factor": bw,
            "epsilon": EPSILON,
            "relative_l2": rel_l2,
            "completed": r["completed"],
            "runtime_seconds": r["runtime_seconds"],
        })

    # ------------------------------------------------------------------
    # Identify best bandwidth
    # ------------------------------------------------------------------
    best_idx = int(np.argmin(bw_rel_l2))
    best_bw = BANDWIDTH_FACTORS[best_idx]
    best_rel_l2 = bw_rel_l2[best_idx]

    print(f"\n  [RESULT] best bw={best_bw}  rel_L2={best_rel_l2:.4f}"
          f"  (oracle={oracle_rel_l2:.4f})")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    prior = PRIOR_AUDIT[case_id]

    plot_rel_l2_vs_bw(
        BANDWIDTH_FACTORS, bw_rel_l2,
        oracle_rel_l2, prior["tikhonov"], prior["sl_bw4_audit"],
        case_id,
        out_dir / f"rel_L2_vs_bandwidth_VH_{beta_tag}.png",
    )

    plot_field_comparison_best(
        x_grid, u0_true, u_obs,
        bw_candidates[best_bw], oracle_candidate,
        best_bw, best_rel_l2, oracle_rel_l2,
        case_id,
        out_dir / f"field_comparison_best_VH_{beta_tag}.png",
    )

    plot_residual(
        x_grid, u0_true,
        bw_candidates[best_bw], oracle_candidate,
        best_bw,
        case_id,
        out_dir / f"residual_best_VH_{beta_tag}.png",
    )

    return rows, bw_rel_l2, best_bw, best_rel_l2, oracle_rel_l2


# ---------------------------------------------------------------------------
# Summary analysis
# ---------------------------------------------------------------------------
def analyze_results(
    case_id: str,
    bw_factors: list[int],
    bw_rel_l2: list[float],
    best_bw: int,
    best_rel_l2: float,
    oracle_rel_l2: float,
) -> list[str]:
    """Return formatted summary lines for one beta case."""
    prior = PRIOR_AUDIT[case_id]
    audit_bw4 = prior["sl_bw4_audit"]
    tikhonov = prior["tikhonov"]

    lines = [f"  Case: {case_id}"]
    lines.append(f"    Oracle (this run) : {oracle_rel_l2:.4f}")
    lines.append(f"    Oracle (audit)    : {prior['oracle']:.4f}")
    lines.append(f"    Tikhonov (audit)  : {tikhonov:.4f}")
    lines.append(f"    Audit sl_bw4      : {audit_bw4:.4f}")
    lines.append("")
    lines.append("    Bandwidth sweep (smoothed_log, epsilon=1e-8):")
    for bw, rl2 in zip(bw_factors, bw_rel_l2):
        marker = "  <-- BEST" if bw == best_bw else ""
        lines.append(f"      bw={bw}  rel_L2={rl2:.4f}{marker}")

    lines.append("")
    improved = best_rel_l2 < audit_bw4
    pct = 100.0 * (audit_bw4 - best_rel_l2) / audit_bw4
    lines.append(f"  1. Best bandwidth factor     : {best_bw}")
    lines.append(f"  2. Best rel_L2               : {best_rel_l2:.4f}")
    lines.append(f"  3. Improves over audit bw=4  : {'YES' if improved else 'NO'}"
                 f"  ({'+' if improved else ''}{pct:.1f}%)")
    lines.append(f"  4. Any bw < 0.060            : {'YES' if any(v < 0.060 for v in bw_rel_l2) else 'NO'}")
    lines.append(f"  5. All bw >= 0.070           : {'YES' if all(v >= 0.070 for v in bw_rel_l2) else 'NO'}")

    # Curve shape: U-shaped vs flat
    min_i = int(np.argmin(bw_rel_l2))
    if min_i == 0 or min_i == len(bw_rel_l2) - 1:
        shape = "MONOTONE (optimum at endpoint)"
    else:
        left_rise  = bw_rel_l2[min_i] < bw_rel_l2[0]
        right_rise = bw_rel_l2[min_i] < bw_rel_l2[-1]
        spread = max(bw_rel_l2) - min(bw_rel_l2)
        if left_rise and right_rise and spread > 0.002:
            shape = "U-SHAPED (clear interior minimum)"
        elif spread < 0.002:
            shape = "FLAT (variation < 0.002)"
        else:
            shape = "ASYMMETRIC (partially U-shaped)"
    lines.append(f"  6. Curve shape               : {shape}")

    # Mixture limitation
    ratio_oracle = best_rel_l2 / (oracle_rel_l2 + 1e-300)
    ratio_tik    = best_rel_l2 / (tikhonov + 1e-300)
    if ratio_oracle > 2.0 or ratio_tik > 10.0:
        limitation = f"YES — mixture gap remains ({ratio_oracle:.1f}× oracle, {ratio_tik:.1f}× Tikhonov)"
    else:
        limitation = f"NO — gap closed ({ratio_oracle:.1f}× oracle)"
    lines.append(f"  7. Mixture limitation remains: {limitation}")

    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="VH mixture bandwidth refinement — final pre-writeup experiment"
    )
    parser.add_argument("--n-particles", type=int, default=N_PARTICLES)
    parser.add_argument("--n-grid", type=int, default=N_GRID)
    parser.add_argument("--also-beta09", action="store_true",
                        help="Also run beta=0.9 (default: beta=0.5 only)")
    parser.add_argument("--base-config", type=str,
                        default=str(REPO_ROOT / "configs" / "gaussian_base.yaml"),
                        help="Unused (kept for CLI parity); mixture config used only")
    parser.add_argument("--mixture-config", type=str,
                        default=str(MIXTURE_CONFIG))
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Output directory
    # -----------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "outputs" / f"vh_mixture_bandwidth_refinement_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory: {out_dir}")
    print(f"[INFO] n_grid={args.n_grid}  N_particles={args.n_particles}"
          f"  bandwidth_factors={BANDWIDTH_FACTORS}")

    # -----------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------
    mix_cfg = load_config(args.mixture_config)
    mix_cfg = vc.patch_config(mix_cfg, **{
        "domain.n_grid": args.n_grid,
        "heat.T": T,
        "heat.dt": DT,
    })
    x_grid = make_grid(mix_cfg)
    u0_true = compute_true_u0(x_grid, mix_cfg)

    # -----------------------------------------------------------------------
    # Run beta=0.5 (required)
    # -----------------------------------------------------------------------
    t_case_start = time.perf_counter()
    all_rows: list[dict] = []

    rows05, bw_rl2_05, best_bw_05, best_rl2_05, oracle_rl2_05 = run_case(
        beta=0.5, cfg=mix_cfg, x_grid=x_grid, u0_true=u0_true,
        out_dir=out_dir, n_particles=args.n_particles,
    )
    all_rows.extend(rows05)
    t_beta05 = time.perf_counter() - t_case_start
    print(f"\n[TIMING] beta=0.5 case finished in {t_beta05:.1f}s")

    # -----------------------------------------------------------------------
    # Run beta=0.9 (optional)
    # -----------------------------------------------------------------------
    bw_rl2_09, best_bw_09, best_rl2_09, oracle_rl2_09 = None, None, None, None
    if args.also_beta09:
        rows09, bw_rl2_09, best_bw_09, best_rl2_09, oracle_rl2_09 = run_case(
            beta=0.9, cfg=mix_cfg, x_grid=x_grid, u0_true=u0_true,
            out_dir=out_dir, n_particles=args.n_particles,
        )
        all_rows.extend(rows09)
    else:
        print("\n[INFO] Skipping beta=0.9 (pass --also-beta09 to include)")

    # -----------------------------------------------------------------------
    # CSV
    # -----------------------------------------------------------------------
    df = pd.DataFrame(all_rows)
    csv_path = out_dir / "vh_mixture_bandwidth_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[CSV] Saved: {csv_path}")

    # -----------------------------------------------------------------------
    # Summary text
    # -----------------------------------------------------------------------
    lines = [
        "=" * 70,
        "VH MIXTURE BANDWIDTH REFINEMENT — FINAL PRE-WRITEUP EXPERIMENT",
        f"Timestamp     : {timestamp}",
        f"n_grid        : {args.n_grid}",
        f"N_particles   : {args.n_particles}",
        f"bandwidth_factors: {BANDWIDTH_FACTORS}",
        f"epsilon       : {EPSILON:.0e}",
        f"score_method  : smoothed_log",
        f"alpha0        : {ALPHA0}  T={T}  dt={DT}",
        "=" * 70,
        "",
    ]

    lines.extend(analyze_results(
        "VH_beta05", BANDWIDTH_FACTORS, bw_rl2_05,
        best_bw_05, best_rl2_05, oracle_rl2_05,
    ))

    if bw_rl2_09 is not None:
        lines.append("")
        lines.extend(analyze_results(
            "VH_beta09", BANDWIDTH_FACTORS, bw_rl2_09,
            best_bw_09, best_rl2_09, oracle_rl2_09,  # type: ignore[arg-type]
        ))

    # Overall verdict
    lines += ["", "=" * 70, "DECISION"]
    all_bw_vals = bw_rl2_05 + (bw_rl2_09 if bw_rl2_09 is not None else [])
    any_below_060 = any(v < 0.060 for v in all_bw_vals)
    any_below_070 = any(v < 0.070 for v in all_bw_vals)
    best_overall = min(all_bw_vals)

    if any_below_060:
        decision = (
            "SUCCESS — at least one bandwidth achieves rel_L2 < 0.060.\n"
            "  Use that bandwidth factor as the reported production value."
        )
    elif any_below_070:
        decision = (
            "PARTIAL IMPROVEMENT — no bandwidth reaches 0.060 but at least one\n"
            "  is below 0.070. Report best bandwidth; acknowledge mixture gap."
        )
    else:
        decision = (
            "NO BREAKTHROUGH — all bandwidths remain >= 0.070.\n"
            "  Keep existing bw=4 result. Explicitly label multimodal score\n"
            "  estimation as a limitation in the paper. Do NOT continue tuning."
        )

    lines.append(f"  Best rel_L2 across all runs : {best_overall:.4f}")
    lines.append(f"  {decision}")
    lines += [
        "",
        "STOP CONDITION: No further tuning. Proceed to paper writeup.",
        "  Future work: adaptive bandwidth, N=50k+, mixture-aware KDE.",
        "=" * 70,
    ]

    summary_text = "\n".join(lines)
    print("\n" + summary_text)

    summary_path = out_dir / "vh_mixture_bandwidth_summary.txt"
    summary_path.write_text(summary_text)
    print(f"\n[SUMMARY] Saved: {summary_path}")


if __name__ == "__main__":
    main()
