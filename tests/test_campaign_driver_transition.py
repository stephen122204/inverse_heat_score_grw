"""Contract tests for the isolated transition-table driver."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import campaign_driver_transition as transition  # noqa: E402
import campaign_results as results  # noqa: E402


class TestVendoredLegacyPieces(unittest.TestCase):
    def test_endpoint_initializer_inverts_a_uniform_density_exactly(self):
        x = np.linspace(0.0, 1.0, 101)
        positions, mass = transition._legacy_quantile_init(np.ones(101), x, 4)
        self.assertAlmostEqual(mass, 1.0, places=12)
        np.testing.assert_allclose(positions,
                                   [0.125, 0.375, 0.625, 0.875], atol=1e-12)

    def test_legacy_reflection_mirrors_and_clips(self):
        pos = np.array([-0.25, 0.5, 1.1])
        np.testing.assert_allclose(transition._legacy_reflect(pos),
                                   [0.25, 0.5, 0.9], atol=1e-15)

    def test_free_space_kde_loses_boundary_mass(self):
        x = np.linspace(0.0, 1.0, 201)
        u = transition._legacy_free_space_kde(np.array([0.0]), 1.0, x, 0.05)
        dx = x[1] - x[0]
        self.assertLess(float(np.sum(0.5 * (u[:-1] + u[1:]) * dx)), 0.55)


class TestTransitionDriver(unittest.TestCase):
    def test_era_registry_matches_the_preregistered_rows(self):
        eras = {r["era"] for r in results.enumerate_rows("transition_table")}
        self.assertEqual(eras, set(transition.ERA_RUNNERS))
        self.assertEqual(eras, set(transition.ERA_METADATA))

    def test_canonical_era_row_completes_with_campaign_accuracy(self):
        rows = [r for r in results.enumerate_rows("transition_table")
                if r["era"] == "cell_centered_neumann_canonical"]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            accounting = transition.drive_transition_table(out, rows=rows)
            self.assertEqual(accounting.completed, 1)
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["legacy_source_commit"],
                             transition.LEGACY_SOURCE_COMMIT)
            import csv
            with (out / "transition_table_rows.csv").open(newline="",
                                                          encoding="utf-8") as f:
                row = list(csv.DictReader(f))[0]
            self.assertLess(float(row["E2"]), 0.05)
            self.assertEqual(row["h_nominal"], row["h_effective"])
            self.assertEqual(row["smoothing_operations"], "1")


if __name__ == "__main__":
    unittest.main()
