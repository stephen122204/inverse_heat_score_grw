"""
reproduce.py — one-command regeneration of every paper artifact.

Runs every experiment script behind the paper's tables and figures with the
exact frozen arguments, then regenerates the twelve shipped figures from the
resulting CSVs.  This is the single source of truth for
"clone -> install -> reproduce".

    python reproduce.py            # full pipeline (CSVs + figures)
    python reproduce.py --figures-only   # just rebuild figures from existing CSVs
    python reproduce.py verify     # check paper numbers against archived outputs
                                   # (runs in seconds, no simulation)

The core density-particle method is deterministic (quantile initialization,
analytic/KDE scores, no RNG).  The studies that draw random observation noise
use fixed realizations: 0-24 for the reported noise and discrepancy tables,
and 0-2 for the superseded validation-stage reconciliation task.  Re-running
therefore yields identical CSVs for the deterministic experiments and
identical realization means for the noise-driven studies.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
PY = sys.executable
BASE = "configs/gaussian_base.yaml"
MIX = "configs/gaussian_mixture.yaml"

# (label, argv) — argv is passed to the experiment script.
STEPS = [
    ("representation and grid-by-count convergence sweep  -> Tables 1 and 6, Figures 2-3",
     ["scripts/run_representation_audit.py", "--base-config", BASE, "--mixture-config", MIX,
      "--n-grid", "100", "200", "400", "800",
      "--n-particles", "1000", "5000", "10000", "20000"]),
    ("time-step sweep for the exact-score gradient globs  -> Section 3.3",
     ["scripts/run_dt_sweep_glob.py"]),
    ("score-estimation and bandwidth audit  -> Table 2, Figure 5",
     ["scripts/run_score_estimation_audit.py", "--base-config", BASE, "--mixture-config", MIX,
      "--bandwidth-factor", "1", "2", "4", "6", "8", "12", "16",
      "--n-particles", "5000",
      "--score-methods", "smoothed_log", "fd_grid_ratio", "direct_kde",
      "--skip-field-comparison", "--skip-tikhonov"]),
    ("validation stage, particle-count convergence  -> Figure 6",
     ["scripts/run_validation_stage.py", "--base-config", BASE, "--mixture-config", MIX,
      "--skip-n20000"]),
    ("noise study over 25 realizations  -> Table 3, Figure 7",
     ["scripts/run_noise_study_25seeds.py"]),
    ("discrepancy-principle residual curves  -> feeds Table 4",
     ["scripts/run_discrepancy_principle.py"]),
    ("discrepancy-principle selection at tau = 1.2  -> Table 4, Figure 8",
     ["scripts/reselect_discrepancy.py"]),
    ("non-smooth initial data  -> Table 5, Figure 9",
     ["scripts/run_nonsmooth_case.py"]),
    ("variable-coefficient audit  -> Table 7, Figures 10-11",
     ["scripts/run_variable_coefficient_audit.py", "--n-grid", "400", "--n-particles", "10000"]),
    ("variable-coefficient mixture bandwidth sweep  -> Figure 12",
     ["scripts/run_vh_mixture_bandwidth_refinement.py", "--n-particles", "10000", "--n-grid", "400"]),
]


def run(argv, label):
    print(f"\n{'='*70}\n>>> {label}\n>>> {PY} {' '.join(argv)}\n{'='*70}", flush=True)
    t0 = time.perf_counter()
    subprocess.run([PY] + argv, cwd=str(REPO), check=True)
    print(f"--- done in {time.perf_counter()-t0:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?", choices=["all", "verify"], default="all",
                    help="'verify' checks the paper numbers against the archived outputs and exits.")
    ap.add_argument("--figures-only", action="store_true",
                    help="Skip experiments; only regenerate figures from existing CSVs.")
    args = ap.parse_args()

    if args.target == "verify":
        sys.exit(subprocess.run([PY, "scripts/verify_numbers.py"], cwd=str(REPO)).returncode)

    if not args.figures_only:
        for label, argv in STEPS:
            run(argv, label)

    run(["make_figures.py", "--only", "naive", "convergence", "loop", "bandwidth",
         "particle_count", "variable_field", "variable_results", "vh_mixture"],
        "core figures from CSVs")
    run(["make_new_figures.py"], "noise-band, representation-failure, and nonsmooth figures")
    run(["make_discrepancy_figure.py"], "discrepancy figure")
    print("\nAll artifacts regenerated. CSVs in outputs/, figures in figures/.")


if __name__ == "__main__":
    main()
