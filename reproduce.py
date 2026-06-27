"""
reproduce.py — one-command regeneration of every paper artifact.

Runs the five experiment scripts (plus the grid x N representation/convergence
sweep that backs Fig 1) with the exact frozen arguments, then regenerates all
figures from the resulting CSVs.  This is the single source of truth for
"clone -> install -> reproduce".

    python reproduce.py            # full pipeline (CSVs + figures)
    python reproduce.py --figures-only   # just rebuild figures from existing CSVs

The core density-particle method is deterministic (quantile initialisation,
analytic/KDE scores, no RNG); only the noise study uses fixed seeds {0,1,2}.
Re-running therefore yields identical CSVs for the deterministic experiments
and identical seed-averaged means for the noise study.
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
    ("representation + grid×N convergence sweep (Fig 1 + Table 1)",
     ["scripts/run_representation_audit.py", "--base-config", BASE, "--mixture-config", MIX,
      "--n-grid", "100", "200", "400", "800",
      "--n-particles", "1000", "5000", "10000", "20000"]),
    ("score-estimation / bandwidth audit (incl. direct_kde)",
     ["scripts/run_score_estimation_audit.py", "--base-config", BASE, "--mixture-config", MIX,
      "--bandwidth-factor", "1", "2", "4", "6", "8", "12", "16",
      "--n-particles", "5000",
      "--score-methods", "smoothed_log", "fd_grid_ratio", "direct_kde",
      "--skip-field-comparison", "--skip-tikhonov"]),
    ("validation stage (N-convergence + noise robustness)",
     ["scripts/run_validation_stage.py", "--base-config", BASE, "--mixture-config", MIX,
      "--skip-n20000"]),
    ("variable-coefficient audit",
     ["scripts/run_variable_coefficient_audit.py", "--n-grid", "400", "--n-particles", "10000"]),
    ("VH-mixture bandwidth refinement",
     ["scripts/run_vh_mixture_bandwidth_refinement.py", "--n-particles", "10000", "--n-grid", "400"]),
]


def run(argv, label):
    print(f"\n{'='*70}\n>>> {label}\n>>> {PY} {' '.join(argv)}\n{'='*70}", flush=True)
    t0 = time.perf_counter()
    subprocess.run([PY] + argv, cwd=str(REPO), check=True)
    print(f"--- done in {time.perf_counter()-t0:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures-only", action="store_true",
                    help="Skip experiments; only regenerate figures from existing CSVs.")
    args = ap.parse_args()

    if not args.figures_only:
        for label, argv in STEPS:
            run(argv, label)

    run(["make_figures.py"], "figures from CSVs")
    print("\nAll artifacts regenerated. CSVs in outputs/, figures in figures/.")


if __name__ == "__main__":
    main()
