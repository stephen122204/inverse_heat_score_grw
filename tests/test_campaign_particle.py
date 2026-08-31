"""Contract tests for the canonical Phase 2C density-particle runner."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from invheat_grw.campaign_particle import (  # noqa: E402
    CampaignContractError,
    certified_b0,
    certified_b1,
    exact_step_count,
    run_campaign_density,
    score_check_message,
    tightened_mode_count,
    tolerance_for_modes,
)
from invheat_grw.cell_grid import cell_centers, propagate_heat  # noqa: E402
from invheat_grw.neumann_kernels import neumann_mode_count  # noqa: E402


def brute_tail(n_modes: int, bandwidth: float, length: float,
               total_mass: float, weight_power: int) -> float:
    theta = 0.5 * (math.pi * bandwidth / length) ** 2
    n = np.arange(n_modes + 1, n_modes + 20000)
    series = float(np.sum(n ** weight_power * np.exp(-theta * n ** 2)))
    if weight_power == 0:
        return (2.0 * total_mass / length) * series
    return (2.0 * math.pi * total_mass / length ** 2) * series


class TestCertifiedEnvelopes(unittest.TestCase):
    def test_bounds_dominate_and_stay_tight(self):
        for n_modes, h in ((50, 0.028), (120, 0.014), (300, 0.007)):
            for fn, power in ((certified_b0, 0), (certified_b1, 1)):
                bound = fn(n_modes, h, 1.0, 1.3)
                true = brute_tail(n_modes, h, 1.0, 1.3, power)
                self.assertGreaterEqual(bound, true)
                self.assertLessEqual(bound, 5.0 * true + 1e-300)

    def test_tightened_mode_count_reduces_both_envelopes(self):
        n_modes = neumann_mode_count(0.028, 1.0, 1e-14)
        k = tightened_mode_count(n_modes, 0.028, 1.0, 1.0)
        self.assertGreater(k, n_modes)
        self.assertLessEqual(certified_b0(k, 0.028, 1.0, 1.0),
                             certified_b0(n_modes, 0.028, 1.0, 1.0) / 100.0)
        self.assertLessEqual(certified_b1(k, 0.028, 1.0, 1.0),
                             certified_b1(n_modes, 0.028, 1.0, 1.0) / 100.0)

    def test_tolerance_for_modes_round_trips(self):
        for h in (0.014, 0.028, 0.040):
            for k in (25, 90, 200):
                tol = tolerance_for_modes(k, h, 1.0)
                self.assertEqual(neumann_mode_count(h, 1.0, tol), k)


class TestContracts(unittest.TestCase):
    def test_exact_step_count(self):
        self.assertEqual(exact_step_count(1.0, 1e-3), 1000)
        with self.assertRaises(CampaignContractError):
            exact_step_count(1.0, 3e-4)

    def test_score_check_messages(self):
        good = np.zeros(4)
        self.assertEqual(score_check_message(good, np.ones(4)), "")
        self.assertIn("denominator",
                      score_check_message(good, np.array([1.0, 0.0, 1, 1])))
        self.assertIn("nonfinite",
                      score_check_message(np.array([np.nan, 0, 0, 0]),
                                          np.ones(4)))
        self.assertIn("ceiling",
                      score_check_message(np.array([2e6, 0, 0, 0]),
                                          np.ones(4)))

    def test_exactly_one_diffusivity_spec(self):
        with self.assertRaises(CampaignContractError):
            run_campaign_density(np.ones(16), x_min=0.0, x_max=1.0, T=0.01,
                                 dt=1e-3, n_particles=8, bandwidth=0.028,
                                 eps_rel=1e-8)


class TestQuickRuns(unittest.TestCase):
    def _c1(self, m):
        x = cell_centers(0.0, 1.0, m)
        alpha, big_t = 0.01, 0.02
        u0 = 1.0 + 0.5 * np.cos(3 * np.pi * x)
        g = 1.0 + 0.5 * math.exp(-alpha * (3 * np.pi) ** 2 * big_t) \
            * np.cos(3 * np.pi * x)
        return x, u0, g, alpha, big_t

    def test_constant_coefficient_run_completes_with_gates(self):
        m = 200
        x, u0, g, alpha, big_t = self._c1(m)

        def residual(uhat):
            fwd = propagate_heat(uhat, 1.0, alpha, big_t)
            return float(np.linalg.norm(fwd - g) / np.linalg.norm(g))

        res = run_campaign_density(
            g, x_min=0.0, x_max=1.0, T=big_t, dt=1e-3, n_particles=800,
            bandwidth=0.028, eps_rel=1e-8, alpha=alpha, u0_reference=u0,
            forward_residual=residual)
        self.assertEqual(res.status, "completed")
        self.assertTrue(res.gates["eps_floor"])
        self.assertTrue(res.gates["tightened_score"])
        self.assertEqual(sorted(res.snapshots), [0, 10, 19])
        self.assertLess(res.metrics["mass_error_rel"], 1e-12)
        self.assertLess(res.metrics["E2"], 0.2)
        self.assertLess(res.metrics["forward_residual"], 0.2)
        self.assertGreater(res.diagnostics["n_modes_tightened"],
                           res.diagnostics["n_modes"])
        for change in res.diagnostics["tightened_score_change"].values():
            self.assertLess(change, 1e-8)

    def test_variable_coefficient_run_completes(self):
        m = 200
        x = cell_centers(0.0, 1.0, m)
        g = 0.2 + np.exp(-(x - 0.4) ** 2 / (2 * 0.08 ** 2))

        def a_of_x(z):
            return 0.01 * (1.0 + 0.9 * np.sin(2 * np.pi * z))

        res = run_campaign_density(
            g, x_min=0.0, x_max=1.0, T=0.01, dt=1e-3, n_particles=600,
            bandwidth=0.014, eps_rel=1e-8, a_of_x=a_of_x)
        self.assertEqual(res.status, "completed")
        self.assertTrue(res.gates["tightened_score"])
        self.assertIsNotNone(res.reconstruction)

    def test_eps_floor_gate_fails_for_diagnostic_zero_epsilon(self):
        m = 100
        x, u0, g, alpha, big_t = self._c1(m)
        res = run_campaign_density(
            g, x_min=0.0, x_max=1.0, T=0.005, dt=1e-3, n_particles=200,
            bandwidth=0.028, eps_rel=0.0, alpha=alpha)
        self.assertFalse(res.gates["eps_floor"])


if __name__ == "__main__":
    unittest.main()
