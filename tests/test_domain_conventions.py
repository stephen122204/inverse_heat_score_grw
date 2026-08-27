"""Forbid inferring the physical domain from sample coordinates.

Computing a domain length as the span between the first and last sample is
correct only on endpoint grids; on a cell-centered grid it silently shortens
L by one cell and shifts every spectral eigenvalue.  The physical walls and
length must come from the configuration.  This test scans the production
sources for the forbidden length patterns.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

SCANNED_FILES = [
    "figure_data.py",
    "make_discrepancy_figure.py",
    "make_figures.py",
    "make_new_figures.py",
    "provenance.py",
    "reproduce.py",
    "scripts/generate_figure_data.py",
    "scripts/reselect_discrepancy.py",
    "scripts/run_all.py",
    "scripts/run_discrepancy_principle.py",
    "scripts/run_dt_sweep_glob.py",
    "scripts/run_noise_study_25seeds.py",
    "scripts/run_nonsmooth_case.py",
    "scripts/run_representation_audit.py",
    "scripts/run_score_estimation_audit.py",
    "scripts/run_validation_stage.py",
    "scripts/run_variable_coefficient_audit.py",
    "scripts/run_vh_mixture_bandwidth_refinement.py",
    "scripts/verify_figure_freeze.py",
    "scripts/verify_numbers.py",
    "src/invheat_grw/baselines.py",
    "src/invheat_grw/config.py",
    "src/invheat_grw/fields.py",
    "src/invheat_grw/globs.py",
    "src/invheat_grw/io_utils.py",
    "src/invheat_grw/methods.py",
    "src/invheat_grw/metrics.py",
    "src/invheat_grw/neumann_kernels.py",
    "src/invheat_grw/plotting.py",
    "src/invheat_grw/scores.py",
]

FORBIDDEN = re.compile(r"x(?:_grid)?\s*\[\s*-1\s*\]\s*-\s*x(?:_grid)?\s*\[\s*0\s*\]")


class DomainConventionTests(unittest.TestCase):
    def test_scanned_files_exist(self):
        missing = [f for f in SCANNED_FILES if not (REPO / f).is_file()]
        self.assertEqual(missing, [])

    def test_no_domain_length_inferred_from_sample_span(self):
        offenders = []
        for rel in SCANNED_FILES:
            text = (REPO / rel).read_text(encoding="utf-8")
            for i, line in enumerate(text.splitlines(), start=1):
                if FORBIDDEN.search(line):
                    offenders.append(f"{rel}:{i}: {line.strip()}")
        self.assertEqual(offenders, [],
                         "domain length inferred from sample coordinates:\n"
                         + "\n".join(offenders))


if __name__ == "__main__":
    unittest.main()
