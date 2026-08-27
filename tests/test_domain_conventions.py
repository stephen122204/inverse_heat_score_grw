"""Forbid inferring the physical domain from sample coordinates.

Computing a domain length as the span between the first and last sample is
correct only on endpoint grids; on a cell-centered grid it silently shortens
L by one cell and shifts every spectral eigenvalue.  The physical walls and
length must come from the configuration.  This test scans the production
sources for the forbidden length patterns.
"""

from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def production_files() -> list[str]:
    """Every tracked Python production file, discovered from git.

    Discovery keeps these bans current automatically: a new production
    script is scanned the moment it is tracked, with no static inventory to
    forget.  Test files are exempt (they may legitimately use, e.g.,
    trapezoidal quadrature as an independent check), and untracked local
    scripts are ignored.
    """
    try:
        out = subprocess.check_output(["git", "ls-files", "*.py"],
                                      cwd=REPO, text=True)
    except (OSError, subprocess.CalledProcessError):
        return []
    return sorted(f for f in out.splitlines()
                  if f and not f.startswith("tests/"))

FORBIDDEN = [
    # Domain length inferred from the sample span.
    re.compile(r"x(?:_grid)?\s*\[\s*-1\s*\]\s*-\s*x(?:_grid)?\s*\[\s*0\s*\]"),
    # Physical walls assigned from the first/last sample coordinate.
    re.compile(r"x_min\s*=\s*(?:float\(\s*)?x(?:_grid)?\s*\[\s*0\s*\]"),
    re.compile(r"x_max\s*=\s*(?:float\(\s*)?x(?:_grid)?\s*\[\s*-1\s*\]"),
    re.compile(r"x_min\s*,\s*x_max\s*=\s*(?:float\(\s*)?x(?:_grid)?\s*\[\s*0\s*\]"),
]


class DomainConventionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = production_files()

    def _require_discovery(self):
        if not self.files:
            self.skipTest("git file discovery unavailable")

    def test_discovery_finds_the_production_tree(self):
        self._require_discovery()
        self.assertGreaterEqual(len(self.files), 25)
        for sentinel in ("figure_data.py",
                         "scripts/run_variable_coefficient_audit.py",
                         "src/invheat_grw/methods.py"):
            self.assertIn(sentinel, self.files)

    def test_no_domain_convention_inferred_from_samples(self):
        self._require_discovery()
        offenders = []
        for rel in self.files:
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
        self._require_discovery()
        pattern = re.compile(r"trapz|trapezoid", re.IGNORECASE)
        offenders = []
        for rel in self.files:
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
