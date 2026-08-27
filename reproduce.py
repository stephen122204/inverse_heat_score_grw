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
from datetime import datetime, timezone
from pathlib import Path

from provenance import DEFAULT_MANIFEST, write_manifest

REPO = Path(__file__).resolve().parent
PY = sys.executable
BASE = "configs/gaussian_base.yaml"
MIX = "configs/gaussian_mixture.yaml"

# (study key, output-directory pattern, label, argv).  A full reproduction
# records the one directory created by each command in a run manifest.
STEPS = [
    ("representation_audit", "representation_audit_*",
     "representation and grid-by-count convergence sweep  -> Tables 1 and 6, Figures 2-3",
     ["scripts/run_representation_audit.py", "--base-config", BASE, "--mixture-config", MIX,
      "--n-grid", "100", "200", "400", "800",
      "--n-particles", "1000", "5000", "10000", "20000"]),
    ("dt_sweep_glob", "dt_sweep_glob_*",
     "time-step sweep for the exact-score gradient globs  -> Section 3.3",
     ["scripts/run_dt_sweep_glob.py"]),
    ("score_estimation_audit", "score_estimation_audit_*",
     "score-estimation and bandwidth audit  -> Table 2, Figure 5",
     ["scripts/run_score_estimation_audit.py", "--base-config", BASE, "--mixture-config", MIX,
      "--bandwidth-factor", "1", "2", "4", "6", "8", "12", "16",
      "--n-particles", "5000",
      "--score-methods", "smoothed_log", "fd_grid_ratio", "direct_kde",
      "--skip-field-comparison", "--skip-tikhonov"]),
    ("validation_stage", "validation_stage_*",
     "validation stage, particle-count convergence  -> Figure 6",
     ["scripts/run_validation_stage.py", "--base-config", BASE, "--mixture-config", MIX,
      "--skip-n20000"]),
    ("noise_study", "noise_study_25seeds_*",
     "noise study over 25 realizations  -> Table 3, Figure 7",
     ["scripts/run_noise_study_25seeds.py"]),
    ("discrepancy_sweep", "discrepancy_principle_[0-9]*",
     "discrepancy-principle residual curves  -> feeds Table 4",
     ["scripts/run_discrepancy_principle.py"]),
    ("discrepancy_final", "discrepancy_principle_final_*",
     "discrepancy-principle selection at tau = 1.2  -> Table 4, Figure 8",
     ["scripts/reselect_discrepancy.py"]),
    ("nonsmooth_case", "nonsmooth_case_*",
     "non-smooth initial data  -> Table 5, Figure 9",
     ["scripts/run_nonsmooth_case.py"]),
    ("variable_coefficient_audit", "variable_coefficient_audit_*",
     "variable-coefficient audit  -> Table 7, Figures 10-11",
     ["scripts/run_variable_coefficient_audit.py", "--n-grid", "400", "--n-particles", "10000"]),
    ("vh_mixture_bandwidth", "vh_mixture_bandwidth_refinement_*",
     "variable-coefficient mixture bandwidth sweep  -> Figure 12",
     ["scripts/run_vh_mixture_bandwidth_refinement.py", "--n-particles", "10000", "--n-grid", "400"]),
]


def run(argv, label):
    print(f"\n{'='*70}\n>>> {label}\n>>> {PY} {' '.join(argv)}\n{'='*70}", flush=True)
    t0 = time.perf_counter()
    subprocess.run([PY] + argv, cwd=str(REPO), check=True)
    elapsed = time.perf_counter() - t0
    print(f"--- done in {elapsed:.1f}s")
    return elapsed


def output_dirs(pattern: str) -> set[Path]:
    return {path.resolve() for path in (REPO / "outputs").glob(pattern) if path.is_dir()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?", choices=["all", "verify"], default="all",
                    help="'verify' checks the paper numbers against the archived outputs and exits.")
    ap.add_argument("--figures-only", action="store_true",
                    help="Skip experiments; only regenerate figures from existing CSVs.")
    ap.add_argument("--manifest", default=None,
                    help=("Manifest to read for --figures-only/verify, or path to write for a "
                          "full run. Archived commands default to manifests/paper_v5_1.json."))
    args = ap.parse_args()

    if args.target == "verify":
        manifest = Path(args.manifest) if args.manifest else DEFAULT_MANIFEST
        sys.exit(subprocess.run(
            [PY, "scripts/verify_numbers.py", "--manifest", str(manifest)], cwd=str(REPO)
        ).returncode)

    studies: dict[str, Path] = {}
    commands: list[dict] = []
    if not args.figures_only:
        for key, pattern, label, argv in STEPS:
            run_argv = list(argv)
            if key == "discrepancy_final":
                sweep = studies.get("discrepancy_sweep")
                if sweep is None:
                    raise RuntimeError("discrepancy_final requires the current run's discrepancy_sweep")
                run_argv.extend(["--source-dir", str(sweep)])
            before = output_dirs(pattern)
            elapsed = run(run_argv, label)
            created = output_dirs(pattern) - before
            if len(created) != 1:
                raise RuntimeError(
                    f"Study {key!r} created {len(created)} matching directories; "
                    f"expected exactly one for pattern {pattern!r}: {sorted(created)}"
                )
            studies[key] = next(iter(created))
            commands.append({
                "study": key,
                "argv": [PY] + run_argv,
                "elapsed_seconds": elapsed,
            })

        if args.manifest:
            manifest = Path(args.manifest)
        else:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            manifest = REPO / "outputs" / f"run_manifest_{stamp}.json"
        manifest = write_manifest(manifest, studies, commands, REPO)
        print(f"[provenance] wrote {manifest}")
    else:
        manifest = Path(args.manifest) if args.manifest else DEFAULT_MANIFEST

    run(["make_figures.py", "--only", "naive", "convergence", "loop", "bandwidth",
         "particle_count", "variable_field", "variable_results", "vh_mixture",
         "--manifest", str(manifest)],
        "core figures from CSVs")
    run(["make_new_figures.py", "--manifest", str(manifest)],
        "noise-band, representation-failure, and nonsmooth figures")
    run(["make_discrepancy_figure.py", "--manifest", str(manifest)], "discrepancy figure")
    print(f"\nAll artifacts regenerated from manifest {manifest}. ")


if __name__ == "__main__":
    main()
