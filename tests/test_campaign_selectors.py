"""Contract tests for the Phase 2C selection rules."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from invheat_grw.baselines import tikhonov_inverse  # noqa: E402
from invheat_grw.campaign_selectors import (  # noqa: E402
    morozov_tikhonov,
    oracle_continuous,
    residual_matched_bandwidth,
)
from invheat_grw.cell_grid import cell_centers, cell_spacing, midpoint_norm  # noqa: E402


class TestOracleContinuous(unittest.TestCase):
    def test_recovers_interior_minimizer(self):
        record = oracle_continuous(lambda z: (z + 5.3) ** 2 + 1.0)
        self.assertEqual(record.label, "oracle_continuous")
        self.assertAlmostEqual(record.z, -5.3, places=4)
        self.assertFalse(record.endpoint)
        self.assertGreaterEqual(record.evaluations, 65)
        self.assertAlmostEqual(record.extra["coarse_winner_z"], -5.3, delta=0.2)

    def test_flags_endpoint_minimizer(self):
        record = oracle_continuous(lambda z: z)
        self.assertTrue(record.endpoint)
        self.assertAlmostEqual(record.z, -12.0, places=6)


class TestMorozov(unittest.TestCase):
    def test_interior_root_is_labeled_morozov(self):
        record = morozov_tikhonov(
            lambda z: 1.0 / (1.0 + math.exp(-(z + 6.0))), 0.5)
        self.assertEqual(record.label, "morozov")
        self.assertFalse(record.endpoint)
        self.assertAlmostEqual(record.z, -6.0, places=4)

    def test_unattainable_target_selects_endpoint_not_morozov(self):
        record = morozov_tikhonov(
            lambda z: 1.0 / (1.0 + math.exp(-(z + 6.0))), -0.5)
        self.assertEqual(record.label, "endpoint")
        self.assertTrue(record.endpoint)
        self.assertAlmostEqual(record.z, -12.0, places=6)
        high = morozov_tikhonov(
            lambda z: 1.0 / (1.0 + math.exp(-(z + 6.0))), 2.0)
        self.assertEqual(high.label, "endpoint")
        self.assertAlmostEqual(high.z, -1.0, places=6)

    def test_nonmonotone_residual_is_a_failed_selection(self):
        record = morozov_tikhonov(
            lambda z: 1.0 / (1.0 + math.exp(-(z + 6.0)))
            + 0.2 * math.exp(-((z + 4.0) ** 2)), 0.5)
        self.assertEqual(record.label, "failed_monotonicity")
        self.assertIsNone(record.lam)


class TestResidualMatched(unittest.TestCase):
    def test_tie_goes_to_the_larger_bandwidth(self):
        record = residual_matched_bandwidth(
            (0.01, 0.014, 0.02, 0.028), (3.0, 1.0, 1.0, 2.0), 1.0)
        self.assertEqual(record.extra["selected_h"], 0.02)
        self.assertFalse(record.endpoint)
        self.assertEqual(record.extra["absolute_mismatch"], 0.0)

    def test_endpoint_selection_is_flagged(self):
        record = residual_matched_bandwidth(
            (0.01, 0.014, 0.02), (3.0, 4.0, 5.0), 10.0)
        self.assertEqual(record.extra["selected_h"], 0.02)
        self.assertTrue(record.endpoint)


class TestTikhonovIntegration(unittest.TestCase):
    def test_clean_c1_oracle_lambda_is_a_censored_tiny_error(self):
        m = 200
        x = cell_centers(0.0, 1.0, m)
        dx = cell_spacing(0.0, 1.0, m)
        alpha, big_t = 0.01, 1.0
        u0 = 1.0 + 0.5 * np.cos(3 * np.pi * x)
        g = 1.0 + 0.5 * math.exp(-alpha * (3 * np.pi) ** 2 * big_t) \
            * np.cos(3 * np.pi * x)

        def objective(z: float) -> float:
            candidate = tikhonov_inverse(
                g, x, alpha, big_t, 10.0 ** z, length=1.0).candidate
            return midpoint_norm(candidate - u0, dx) / midpoint_norm(u0, dx)

        record = oracle_continuous(objective)
        self.assertTrue(record.endpoint)   # clean data favor the smallest lambda
        self.assertLess(record.value, 1e-9)


if __name__ == "__main__":
    unittest.main()
