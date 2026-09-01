"""Contract tests for the Phase 2C case fields and data builders."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from invheat_grw.campaign_data import (  # noqa: E402
    apply_noise,
    cell_edges,
    cn_forward_solve,
    cosine_field,
    decrimed_variable_data,
    flux_operator_diagonals,
    gaussian_cell_averages,
    gaussian_free_space,
    mixture_cell_averages,
    nominal_delta,
    project_mean,
    projection_stats,
    variable_diffusivity,
)
from invheat_grw.cell_grid import cell_centers, cell_spacing, propagate_heat  # noqa: E402


def apply_operator(main, off, u):
    out = main * u
    out[:-1] += off * u[1:]
    out[1:] += off * u[:-1]
    return out


class TestAnalyticFields(unittest.TestCase):
    def test_cosine_terminal_matches_the_exact_dct_propagation(self):
        m, length, alpha, big_t = 400, 1.0, 0.01, 1.0
        x = cell_centers(0.0, length, m)
        u0 = cosine_field(x, 1.0, ((3, 0.5),), length=length)
        g_analytic = cosine_field(x, 1.0, ((3, 0.5),), length=length,
                                  alpha=alpha, t=big_t)
        g_dct = propagate_heat(u0, length, alpha, big_t)
        self.assertLess(float(np.max(np.abs(g_dct - g_analytic))), 1e-13)

    def test_gaussian_free_space_conserves_mass_and_shrinks_peak(self):
        x = np.linspace(-3.0, 4.0, 70001)
        dx = x[1] - x[0]
        u0 = gaussian_free_space(x, 0.4, 0.08, 1.0)
        ut = gaussian_free_space(x, 0.4, 0.08, 1.0, alpha=0.01, t=0.15)
        self.assertAlmostEqual(float(np.sum(u0) * dx), float(np.sum(ut) * dx),
                               places=8)
        expected_peak = 0.08 / math.sqrt(0.08 ** 2 + 2 * 0.01 * 0.15)
        self.assertAlmostEqual(float(np.max(ut)), expected_peak, places=6)

    def test_cell_averages_match_dense_quadrature(self):
        edges = cell_edges(0.0, 1.0, 25)
        exact = gaussian_cell_averages(edges, 0.4, 0.08, 0.75)
        dense = []
        for a, b in zip(edges[:-1], edges[1:]):
            width = (b - a) / 4000
            z = a + (np.arange(4000) + 0.5) * width
            f = 0.75 * np.exp(-(z - 0.4) ** 2 / (2 * 0.08 ** 2))
            dense.append(float(np.mean(f)))
        self.assertLess(float(np.max(np.abs(exact - np.array(dense)))), 1e-9)
        mixture = mixture_cell_averages(edges, 0.05,
                                        ((0.75, 0.35, 0.05), (0.45, 0.62, 0.08)))
        self.assertGreater(float(np.min(mixture)), 0.05 - 1e-12)


class TestForwardSolver(unittest.TestCase):
    def setUp(self):
        self.a_of_x = variable_diffusivity(0.01, 0.9)

    def test_operator_is_conservative_and_symmetric(self):
        m = 60
        dx = cell_spacing(0.0, 1.0, m)
        faces = 0.0 + np.arange(1, m) * dx
        main, off = flux_operator_diagonals(self.a_of_x(faces), dx)
        rng = np.random.default_rng(0)
        u = rng.random(m)
        self.assertLess(abs(float(np.sum(apply_operator(main, off, u)))), 1e-10)
        # symmetry: <Au, v> == <u, Av>
        v = rng.random(m)
        left = float(np.dot(apply_operator(main, off, u), v))
        right = float(np.dot(u, apply_operator(main, off, v)))
        self.assertAlmostEqual(left, right, places=10)

    def test_cn_preserves_constants_and_mass(self):
        m = 100
        u = np.full(m, 2.5)
        out = cn_forward_solve(u, self.a_of_x, 0.15, 1e-3, x_min=0.0, x_max=1.0)
        self.assertLess(float(np.max(np.abs(out - 2.5))), 1e-12)
        edges = cell_edges(0.0, 1.0, m)
        u0 = gaussian_cell_averages(edges, 0.4, 0.08, 1.0)
        out = cn_forward_solve(u0, self.a_of_x, 0.15, 1e-3, x_min=0.0, x_max=1.0)
        self.assertAlmostEqual(float(np.sum(out)), float(np.sum(u0)), places=10)

    def test_cn_matches_the_spectral_solution_for_constant_diffusivity(self):
        m, alpha, big_t = 200, 0.01, 0.1
        x = cell_centers(0.0, 1.0, m)
        u0 = 1.0 + 0.4 * np.cos(2 * np.pi * x)
        constant_a = variable_diffusivity(alpha, 0.0)
        cn = cn_forward_solve(u0, constant_a, big_t, 1e-3, x_min=0.0, x_max=1.0)
        exact = propagate_heat(u0, 1.0, alpha, big_t)
        rel = float(np.max(np.abs(cn - exact)) / np.max(np.abs(exact)))
        self.assertLess(rel, 5e-3)

    def test_projection_is_conservative_and_decrime_runs(self):
        fine = np.arange(24.0)
        coarse = project_mean(fine, 4)
        self.assertAlmostEqual(float(np.mean(fine)), float(np.mean(coarse)),
                               places=14)
        edges_fn = lambda edges: gaussian_cell_averages(edges, 0.4, 0.08, 1.0)
        data = decrimed_variable_data(
            edges_fn, self.a_of_x, 0.15, x_min=0.0, x_max=1.0, m_inverse=100,
            fine=(400, 1e-3), finer=(800, 5e-4), gate=1e-3)
        self.assertEqual(data.terminal.shape, (100,))
        self.assertLess(data.gate_difference, 1e-3)
        self.assertLess(data.mass_discrepancy, 1e-10)


class TestNoiseBookkeeping(unittest.TestCase):
    def test_noise_projection_and_delta_formulas(self):
        g = np.array([2.0, 1.0, 0.5, 1.5])
        xi = np.array([1.0, -1.0, 0.5, 0.0])
        d = apply_noise(g, 0.1, xi)
        np.testing.assert_allclose(d, g + 0.2 * xi)

        stats = projection_stats(np.array([2.0, -1.0, 3.0, -0.5]), 0.5)
        self.assertAlmostEqual(stats["negative_fraction"], 0.5)
        self.assertAlmostEqual(stats["negative_mass_removed"], 0.75)
        np.testing.assert_allclose(stats["projected"], [2.0, 0.0, 3.0, 0.0])
        self.assertAlmostEqual(stats["mass_change_rel"], 0.75 / 1.75)

        dx = 0.25
        delta = nominal_delta(0.01, g, g, dx, 1.0)
        expected = 0.01 * 2.0 * 1.0 / math.sqrt(dx * float(np.sum(g ** 2)))
        self.assertAlmostEqual(delta, expected, places=14)


if __name__ == "__main__":
    unittest.main()
