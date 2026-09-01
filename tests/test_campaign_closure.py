"""Contract and pre-use verification tests for the Phase 2C closure machinery."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from invheat_grw.campaign_closure import (  # noqa: E402
    carrier_score,
    closure_constant,
    exact_decomposition,
    frozen_left_offset,
    mc_slopes,
    reconstruct_centers,
    run_gradient_carriers,
    run_reference,
    smooth_field,
    smoothed_center_values_and_derivative,
)
from invheat_grw.cell_grid import cell_centers, cell_spacing  # noqa: E402

ALPHA, BIG_T = 0.01, 1.0
DECAY = math.exp(-ALPHA * math.pi ** 2 * BIG_T)


def g1_fields(m):
    x = cell_centers(0.0, 1.0, m)
    u0 = 2.0 + np.cos(np.pi * x)
    g = 2.0 + DECAY * np.cos(np.pi * x)
    return x, u0, g


class TestSmoothing(unittest.TestCase):
    def test_fast_center_path_matches_dense_evaluation(self):
        rng = np.random.default_rng(1)
        u = 1.0 + 0.3 * rng.random(64)
        x = cell_centers(0.0, 1.0, 64)
        smooth = smooth_field(u, 0.02, x_min=0.0, x_max=1.0)
        values, derivative = smoothed_center_values_and_derivative(
            u, 0.02, length=1.0)
        np.testing.assert_allclose(values, smooth.values(x), atol=1e-11)
        np.testing.assert_allclose(derivative, smooth.derivative(x), atol=1e-9)

    def test_carrier_score_matches_the_analytic_cosine_formula(self):
        m, h, eps = 200, 0.014, 1e-6
        x = cell_centers(0.0, 1.0, m)
        a = 0.4
        u = 1.0 + a * np.cos(np.pi * x)
        damping = math.exp(-0.5 * (math.pi * h) ** 2)
        score = carrier_score(u, h, eps, x_min=0.0, x_max=1.0)
        probe = np.array([0.21, 0.44, 0.73])
        expected = (-a * damping * math.pi * np.sin(np.pi * probe)
                    / (1.0 + a * damping * np.cos(np.pi * probe) + eps))
        np.testing.assert_allclose(score(probe), expected, atol=1e-10)


class TestClosures(unittest.TestCase):
    def test_frozen_left_reproduces_the_analytic_anchor(self):
        m = 200
        x, u0, g = g1_fields(m)
        dx = cell_spacing(0.0, 1.0, m)
        edges = np.arange(m + 1) * dx
        q_avg = np.diff(2.0 + np.cos(np.pi * edges)) / dx  # exact cell averages
        c = closure_constant(q_avg, "frozen_left", dx=dx,
                             anchor_value=float(g[0]),
                             total_mass=float(dx * np.sum(g)), length=1.0)
        u_frozen = reconstruct_centers(q_avg, c, dx)
        offset = u_frozen - u0
        self.assertLess(abs(float(np.mean(offset))
                            - frozen_left_offset(1.0, ALPHA, BIG_T)), 1e-3)
        self.assertLess(float(np.std(offset)), 1e-3)

    def test_mass_closure_matches_the_target_mass_exactly(self):
        m = 150
        x, u0, g = g1_fields(m)
        dx = cell_spacing(0.0, 1.0, m)
        q = np.gradient(g, dx)
        total = float(dx * np.sum(g))
        c = closure_constant(q, "mass", dx=dx, anchor_value=0.0,
                             total_mass=total, length=1.0)
        u = reconstruct_centers(q, c, dx)
        self.assertAlmostEqual(float(dx * np.sum(u)), total, places=12)


class TestCarriers(unittest.TestCase):
    def test_split_invariance(self):
        m = 100
        _, _, g = g1_fields(m)
        kwargs = dict(x_min=0.0, x_max=1.0, T=0.01, dt=1e-3, bandwidth=0.014,
                      eps_rel=1e-8, alpha=ALPHA, closure="frozen_left")
        one = run_gradient_carriers(g, subparticles=1, **kwargs)
        three = run_gradient_carriers(g, subparticles=3, **kwargs)
        self.assertEqual(one.status, "completed")
        self.assertEqual(three.status, "completed")
        rel = (float(np.max(np.abs(one.u_final - three.u_final)))
               / float(np.max(np.abs(one.u_final))))
        self.assertLess(rel, 1e-13)
        np.testing.assert_allclose(one.q_final, three.q_final, atol=1e-10)


class TestReferenceSolver(unittest.TestCase):
    def test_zero_state_is_preserved_and_walls_carry_no_flux(self):
        m = 80
        result = run_reference(
            np.zeros(m), kind="unregularized", closure="frozen_left",
            anchor_value=2.0, total_mass=2.0, x_min=0.0, x_max=1.0,
            T=0.01, dt=1e-3, alpha=ALPHA)
        self.assertEqual(result.status, "completed")
        self.assertLess(float(np.max(np.abs(result.q))), 1e-15)
        np.testing.assert_allclose(result.u, np.full(m, 2.0), atol=1e-14)

    def test_conservation_of_the_signed_carrier_mass(self):
        m = 200
        x, u0, g = g1_fields(m)
        dx = cell_spacing(0.0, 1.0, m)
        q0 = -math.pi * DECAY * np.sin(np.pi * x)
        for kind, extra in (("unregularized", {}),
                            ("regularized",
                             {"bandwidth": 0.014, "eps_rel": 1e-8})):
            result = run_reference(
                q0, kind=kind, closure="frozen_left",
                anchor_value=float(g[0]), total_mass=float(dx * np.sum(g)),
                x_min=0.0, x_max=1.0, T=0.01, dt=1e-4, alpha=ALPHA, **extra)
            self.assertEqual(result.status, "completed", kind)
            self.assertAlmostEqual(float(np.sum(result.q) * dx),
                                   float(np.sum(q0) * dx), places=12)
            self.assertGreater(result.max_cfl, 0.0)
            self.assertGreater(result.speed_bound, 0.0)
            self.assertGreater(result.min_u, 0.0)

    def test_one_step_operator_converges_at_second_order(self):
        a = DECAY
        errors = {}
        for m in (100, 200):
            x = cell_centers(0.0, 1.0, m)
            dx = cell_spacing(0.0, 1.0, m)
            q0 = -math.pi * a * np.sin(np.pi * x)
            g = 2.0 + a * np.cos(np.pi * x)
            dt = 1e-6
            result = run_reference(
                q0, kind="unregularized", closure="frozen_left",
                anchor_value=float(g[0]), total_mass=float(dx * np.sum(g)),
                x_min=0.0, x_max=1.0, T=dt, dt=dt, alpha=ALPHA)
            rhs_numeric = (result.q - q0) / dt
            u = 2.0 + a * np.cos(np.pi * x)
            s, c = np.sin(np.pi * x), np.cos(np.pi * x)
            exact = -(ALPHA * math.pi ** 3 * a ** 2
                      * (2.0 * s * c * (2.0 + a * c) + a * s ** 3)
                      / (2.0 + a * c) ** 2)
            interior = slice(m // 10, m - m // 10)
            keep = np.abs(np.abs(x - 0.5) - 0.0) > 5.0 / m
            mask = np.zeros(m, dtype=bool)
            mask[interior] = True
            mask &= keep
            errors[m] = float(np.max(np.abs((rhs_numeric - exact)[mask])))
        self.assertGreater(errors[100] / errors[200], 3.0)

    def test_positivity_loss_fails_loudly(self):
        m = 80
        x = cell_centers(0.0, 1.0, m)
        q0 = -50.0 * np.sin(np.pi * x)   # forces U through zero
        result = run_reference(
            q0, kind="unregularized", closure="mass", anchor_value=0.0,
            total_mass=0.05, x_min=0.0, x_max=1.0, T=0.01, dt=1e-3,
            alpha=ALPHA)
        self.assertEqual(result.status, "failed")
        self.assertIn("positivity", result.failure_message)


class TestSnapshots(unittest.TestCase):
    def test_reference_snapshot_equals_a_run_stopped_at_that_time(self):
        m = 120
        x, u0, g = g1_fields(m)
        dx = cell_spacing(0.0, 1.0, m)
        q0 = -math.pi * DECAY * np.sin(np.pi * x)
        kwargs = dict(kind="regularized", closure="frozen_left",
                      anchor_value=float(g[0]),
                      total_mass=float(dx * np.sum(g)), x_min=0.0, x_max=1.0,
                      dt=1e-3, alpha=ALPHA, bandwidth=0.014, eps_rel=1e-8)
        full = run_reference(q0, T=0.01, snapshot_times=(0.005,), **kwargs)
        short = run_reference(q0, T=0.005, **kwargs)
        u_snap, q_snap = full.snapshots[0.005]
        np.testing.assert_array_equal(q_snap, short.q)
        np.testing.assert_array_equal(u_snap, short.u)

    def test_carrier_snapshot_equals_a_run_stopped_at_that_time(self):
        m = 100
        _, _, g = g1_fields(m)
        kwargs = dict(x_min=0.0, x_max=1.0, dt=1e-3, bandwidth=0.014,
                      eps_rel=1e-8, alpha=ALPHA, closure="frozen_left")
        full = run_gradient_carriers(g, T=0.01, snapshot_times=(0.005,),
                                     **kwargs)
        short = run_gradient_carriers(g, T=0.005, **kwargs)
        u_snap, q_snap = full.snapshots[0.005]
        np.testing.assert_array_equal(u_snap, short.u_final)
        np.testing.assert_array_equal(q_snap, short.q_final)

    def test_misaligned_snapshot_time_fails_loudly(self):
        m = 60
        _, _, g = g1_fields(m)
        with self.assertRaises(Exception):
            run_gradient_carriers(
                g, x_min=0.0, x_max=1.0, T=0.01, dt=1e-3, bandwidth=0.014,
                eps_rel=1e-8, alpha=ALPHA, closure="frozen_left",
                snapshot_times=(0.0042,))


class TestDecomposition(unittest.TestCase):
    def test_identity_and_reconciliation_are_exact(self):
        rng = np.random.default_rng(7)
        fields = [rng.random(50) for _ in range(5)]
        out = exact_decomposition(*fields, dx=0.02)
        self.assertLess(out["identity_max_abs"], 1e-12)
        self.assertLess(out["reconciliation_residual"], 1e-12)
        self.assertEqual(len(out["inner_products"]), 6)


if __name__ == "__main__":
    unittest.main()
