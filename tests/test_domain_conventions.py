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

FORBIDDEN = [
    # Domain length inferred from the sample span.
    re.compile(r"x(?:_grid)?\s*\[\s*-1\s*\]\s*-\s*x(?:_grid)?\s*\[\s*0\s*\]"),
    # Physical walls assigned from the first/last sample coordinate.
    re.compile(r"x_min\s*=\s*(?:float\(\s*)?x(?:_grid)?\s*\[\s*0\s*\]"),
    re.compile(r"x_max\s*=\s*(?:float\(\s*)?x(?:_grid)?\s*\[\s*-1\s*\]"),
    re.compile(r"x_min\s*,\s*x_max\s*=\s*(?:float\(\s*)?x(?:_grid)?\s*\[\s*0\s*\]"),
]


class DomainConventionTests(unittest.TestCase):
    def test_scanned_files_exist(self):
        missing = [f for f in SCANNED_FILES if not (REPO / f).is_file()]
        self.assertEqual(missing, [])

    def test_no_domain_convention_inferred_from_samples(self):
        offenders = []
        for rel in SCANNED_FILES:
            text = (REPO / rel).read_text(encoding="utf-8")
            for i, line in enumerate(text.splitlines(), start=1):
                if any(pattern.search(line) for pattern in FORBIDDEN):
                    offenders.append(f"{rel}:{i}: {line.strip()}")
        self.assertEqual(offenders, [],
                         "domain convention inferred from sample coordinates:\n"
                         + "\n".join(offenders))

    def test_no_trapezoidal_quadrature_in_production(self):
        """The cell-centered contract integrates by the midpoint rule; any
        trapezoidal call in a production path is an endpoint-era remnant."""
        pattern = re.compile(r"trapz|trapezoid", re.IGNORECASE)
        offenders = []
        for rel in SCANNED_FILES:
            text = (REPO / rel).read_text(encoding="utf-8")
            for i, line in enumerate(text.splitlines(), start=1):
                if pattern.search(line):
                    offenders.append(f"{rel}:{i}: {line.strip()}")
        self.assertEqual(offenders, [],
                         "trapezoidal quadrature in production paths:\n"
                         + "\n".join(offenders))


class GlobWallTests(unittest.TestCase):
    def test_glob_state_keeps_physical_walls_on_cell_centered_samples(self):
        import copy
        import sys as _sys
        _sys.path.insert(0, str(REPO / "src"))
        import numpy as np
        from invheat_grw.cell_grid import cell_centers
        from invheat_grw.config import load_config
        from invheat_grw.globs import field_to_globs

        cfg = copy.deepcopy(load_config(str(REPO / "configs" / "gaussian_base.yaml")))
        x = cell_centers(cfg.domain.x_min, cfg.domain.x_max, 32)
        u = 1.0 + 0.5 * np.cos(np.pi * (x - cfg.domain.x_min))
        state = field_to_globs(u, x, cfg)
        self.assertEqual(state.x_min, cfg.domain.x_min)
        self.assertEqual(state.x_max, cfg.domain.x_max)
        # The samples themselves sit strictly inside the walls.
        self.assertGreater(float(x[0]), state.x_min)
        self.assertLess(float(x[-1]), state.x_max)


if __name__ == "__main__":
    unittest.main()
