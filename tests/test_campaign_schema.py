"""The campaign schema, the protocol document, and the freeze gate must
agree.  These tests hold PHASE2C_PROTOCOL.md and campaign_schema.py together:
amending one without the other fails the suite, which is the enforcement arm
of the preregistration rules."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import campaign_schema as schema


class ScheduleLiteralsTests(unittest.TestCase):
    def test_grids_and_defaults(self):
        self.assertEqual(schema.BANDWIDTHS,
                         (0.005, 0.007, 0.010, 0.014, 0.020, 0.028, 0.040))
        self.assertEqual(schema.HEADLINE_H, 0.014)
        self.assertEqual(schema.EPS_REL_GRID, (1e-10, 1e-8, 1e-6, 1e-4))
        self.assertEqual(schema.HEADLINE_EPS_REL, 1e-8)
        self.assertEqual(schema.ETAS, (0.0, 0.001, 0.005, 0.01))
        self.assertEqual(schema.SEEDS, tuple(range(25)))
        self.assertEqual((schema.DEFAULT_M, schema.DEFAULT_DT, schema.DEFAULT_N),
                         (400, 1e-3, 4000))
        self.assertEqual(schema.ADEQUACY_CASES, ("C1", "H"))
        self.assertEqual(schema.ADEQUACY_N, (1000, 2000, 4000, 8000, 10000))
        self.assertEqual(schema.LAMBDA_LOG_BOUNDS, (-12.0, -1.0))
        self.assertEqual(schema.LAMBDA_SCAN_POINTS, 65)
        self.assertEqual(schema.CLOSURE_H_BRIDGE, (0.020, 0.014, 0.010, 0.007))
        self.assertEqual(schema.CLOSURE_REFINEMENTS,
                         ((200, 2e-3), (400, 1e-3), (800, 5e-4)))

    def test_case_registry_roles_and_anchors(self):
        roles = {name: c["role"] for name, c in schema.CASES.items()}
        self.assertEqual(roles["C1"], "primary")
        self.assertEqual(roles["C2"], "primary")
        for s in ("B", "H", "Z"):
            self.assertEqual(roles[s], "secondary")
        for v in ("VB05", "VB09", "VH05", "VH09"):
            self.assertEqual(roles[v], "variable")
        g1 = schema.CASES["G1"]
        self.assertGreater(g1["background"], abs(g1["modes"][0][1]))
        self.assertTrue(schema.CASES["GB"].get("approximate_neumann"))

    def test_frozen_closure_offset_matches_the_analytic_anchor(self):
        a, alpha, T = 1.0, 0.01, 1.0
        expected = -a * (1.0 - math.exp(-alpha * math.pi ** 2 * T))
        self.assertAlmostEqual(schema.frozen_closure_offset(a, alpha, T),
                               expected, places=15)
        self.assertLess(schema.frozen_closure_offset(a, alpha, T), 0.0)


class RowAccountingTests(unittest.TestCase):
    def test_expected_counts_reproduce_independent_arithmetic(self):
        counts = schema.expected_counts()
        self.assertEqual(counts["bandwidth_clean"], 9 * 7)
        self.assertEqual(counts["epsilon_sensitivity"], 2 * 4)
        self.assertEqual(counts["adequacy_N"], 2 * 5)
        self.assertEqual(counts["noise_paired"],
                         2 * (7 + 3 * 25 * 2 * 7))
        self.assertEqual(counts["lambda_oracle_clean"], 9)
        self.assertEqual(counts["lambda_noise"], 2 * 3 * 25 * 2 * 2)
        per_closure_case = (2 * 3) + 2 + (2 * 2 * 2) + (4 * 2)
        self.assertEqual(counts["closure"], 3 * per_closure_case)
        self.assertEqual(counts["transition_table"], 3)
        self.assertEqual(sum(counts.values()), 2879)

    def test_rows_are_well_formed(self):
        for name, fn in schema.STUDIES.items():
            for row in fn():
                self.assertEqual(row["study"], name)
                if "h" in row and row.get("study") != "closure":
                    self.assertIn(row["h"], schema.BANDWIDTHS)
                if "seed" in row and row["seed"] is not None:
                    self.assertIn(row["seed"], schema.SEEDS)
                if "arm" in row and row["arm"] != "shared":
                    self.assertIn(row["arm"], schema.ARMS)


class ProtocolBindingTests(unittest.TestCase):
    def setUp(self):
        self.text = schema.PROTOCOL_FILE.read_text(encoding="utf-8")

    def test_document_carries_the_schema_literals(self):
        for needle in (
            "0.005, 0.007, 0.010, 0.014, 0.020, 0.028, 0.040",
            "1000, 2000, 4000, 8000, 10000",
            "1e-10, 1e-8, 1e-6, 1e-4",
            "65-point",
            "N = 4000",
            "**C1**, the exact bounded-domain primary, and **H**",
            "(200, 2e-3), (400, 1e-3), (800, 5e-4)",
            "{0.020, 0.014, 0.010, 0.007}",
        ):
            self.assertIn(needle, self.text, needle)

    def test_status_line_and_freeze_gate(self):
        status = schema.protocol_status()
        self.assertIn(status, ("PROPOSED", "FROZEN"))
        if status == "PROPOSED":
            with self.assertRaises(schema.ProtocolNotFrozen):
                schema.assert_frozen()
        else:
            schema.assert_frozen()


if __name__ == "__main__":
    unittest.main()
