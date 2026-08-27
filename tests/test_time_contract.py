"""The shared step-count contract: n * dt = T exactly, or fail loudly."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from invheat_grw.config import exact_step_count, load_config


class TimeContractTests(unittest.TestCase):
    def test_divisible_horizons_give_the_expected_counts(self):
        self.assertEqual(exact_step_count(0.15, 0.001), 150)
        self.assertEqual(exact_step_count(0.05, 0.001), 50)
        self.assertEqual(exact_step_count(0.15, 0.00025), 600)

    def test_representable_products_pass_within_tolerance(self):
        # 0.15 = 150 * 0.001 only up to floating-point representation;
        # the documented tolerance absorbs exactly that and nothing more.
        n = exact_step_count(0.3, 0.1)
        self.assertEqual(n, 3)

    def test_nondivisible_horizon_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "not an integer multiple"):
            exact_step_count(0.1, 0.03)

    def test_nonpositive_inputs_fail(self):
        with self.assertRaises(ValueError):
            exact_step_count(0.0, 0.001)
        with self.assertRaises(ValueError):
            exact_step_count(0.1, -0.001)

    def test_config_property_uses_the_contract(self):
        cfg = copy.deepcopy(load_config(str(REPO / "configs" / "gaussian_base.yaml")))
        cfg.heat.T, cfg.heat.dt = 0.15, 0.001
        self.assertEqual(cfg.n_steps, 150)
        cfg.heat.dt = 0.04
        with self.assertRaises(ValueError):
            _ = cfg.n_steps


if __name__ == "__main__":
    unittest.main()
