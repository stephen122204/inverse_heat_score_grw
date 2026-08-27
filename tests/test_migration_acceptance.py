"""Acceptance tests for the cell-centered migration.

Everything here checks operator correctness of the migrated production
paths: the grid itself, exact modal behavior of the forward and inverse
spectral operators through the public APIs, the exact folded reflection
under arbitrarily large overshoots, and quantile/histogram consistency with
the cell contract.
"""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from invheat_grw.baselines import spectral_cutoff_inverse, tikhonov_inverse
from invheat_grw.cell_grid import (
    cell_centers,
    cell_edge_quantile_positions,
    cell_spacing,
    wave_numbers,
)
from invheat_grw.config import load_config
from invheat_grw.fields import GRID_CONVENTION, make_grid
from invheat_grw.globs import apply_reflecting_boundary
from invheat_grw.methods import _quantile_init_particles, _reconstruct_density_particles
from invheat_grw.metrics import forward_heat_solve_dct


def _cfg(n_grid: int = 128):
    cfg = copy.deepcopy(load_config(str(REPO / "configs" / "gaussian_base.yaml")))
    cfg.domain.n_grid = n_grid
    return cfg


class GridMigrationTests(unittest.TestCase):
    def test_make_grid_is_the_cell_contract(self):
        self.assertEqual(GRID_CONVENTION, "cell-centered")
        cfg = _cfg(37)
        x = make_grid(cfg)
        expected = cell_centers(cfg.domain.x_min, cfg.domain.x_max, 37)
        self.assertTrue(np.array_equal(x, expected))
        dx = cell_spacing(cfg.domain.x_min, cfg.domain.x_max, 37)
        self.assertAlmostEqual(float(x[1] - x[0]), dx, places=15)
        self.assertGreater(float(x[0]), cfg.domain.x_min)
        self.assertLess(float(x[-1]), cfg.domain.x_max)

    def test_forward_solve_applies_exact_multiplier(self):
        cfg = _cfg(128)
        cfg.heat.alpha, cfg.heat.T = 0.01, 0.15
        x = make_grid(cfg)
        L = cfg.domain.x_max - cfg.domain.x_min
        k = float(wave_numbers(128, L)[9])
        u = 1.0 + 0.3 * np.cos(k * (x - cfg.domain.x_min))
        out = forward_heat_solve_dct(u, x, cfg)
        exact = 1.0 + 0.3 * np.exp(-cfg.heat.alpha * k ** 2 * cfg.heat.T) \
            * np.cos(k * (x - cfg.domain.x_min))
        self.assertLess(float(np.max(np.abs(out - exact))), 1e-13)

    def test_tikhonov_filter_is_exact_per_mode(self):
        cfg = _cfg(128)
        alpha, T, lam = 0.01, 0.15, 1e-4
        x = make_grid(cfg)
        L = cfg.domain.x_max - cfg.domain.x_min
        k = float(wave_numbers(128, L)[11])
        mode = np.cos(k * (x - cfg.domain.x_min))
        A = np.exp(-alpha * k ** 2 * T)
        out = tikhonov_inverse(0.7 * mode, x, alpha, T, lam, length=L).candidate
        exact = 0.7 * (A / (A ** 2 + lam)) * mode
        self.assertLess(float(np.max(np.abs(out - exact))), 1e-12)

    def test_spectral_cutoff_keeps_and_zeroes_exact_modes(self):
        cfg = _cfg(128)
        alpha, T = 0.01, 0.15
        x = make_grid(cfg)
        L = cfg.domain.x_max - cfg.domain.x_min
        ks = wave_numbers(128, L)
        k_low, k_high = float(ks[3]), float(ks[40])
        u = (0.5 * np.cos(k_low * (x - cfg.domain.x_min))
             + 0.5 * np.cos(k_high * (x - cfg.domain.x_min)))
        k_cut = 0.5 * (k_low + k_high)
        out = spectral_cutoff_inverse(u, x, alpha, T, k_cut=k_cut, length=L).candidate
        exact = 0.5 * np.exp(alpha * k_low ** 2 * T) * np.cos(k_low * (x - cfg.domain.x_min))
        self.assertLess(float(np.max(np.abs(out - exact))), 1e-10)


class ReflectionTests(unittest.TestCase):
    def _reference(self, p, a, b):
        p = np.array(p, dtype=float)
        for _ in range(400):
            p = np.where(p < a, 2.0 * a - p, p)
            p = np.where(p > b, 2.0 * b - p, p)
        return p

    def test_arbitrary_overshoots_match_repeated_reflection(self):
        rng = np.random.default_rng(3)
        a, b = 0.2, 1.4
        inside = a + (b - a) * rng.random(200)
        p = inside.copy()
        p[:150] += (b - a) * rng.uniform(-15.0, 15.0, 150)
        folded = apply_reflecting_boundary(p, a, b)
        self.assertTrue(np.all((folded >= a) & (folded <= b)))
        self.assertTrue(np.allclose(folded, self._reference(p, a, b), atol=1e-10))

    def test_interior_points_and_walls_are_fixed(self):
        a, b = 0.0, 1.0
        p = np.array([a, 0.25, 0.5, 0.75, b])
        self.assertTrue(np.allclose(apply_reflecting_boundary(p, a, b), p, atol=1e-15))


class QuantileHistogramTests(unittest.TestCase):
    def test_quantile_initializer_matches_the_cell_contract(self):
        cfg = _cfg(64)
        x = make_grid(cfg)
        u = 0.2 + np.exp(-0.5 * ((x - 0.4) / 0.1) ** 2)
        pos_a, mass_a = _quantile_init_particles(
            u, x, 500, x_min=cfg.domain.x_min, x_max=cfg.domain.x_max)
        pos_b, mass_b = cell_edge_quantile_positions(
            u, cfg.domain.x_min, cfg.domain.x_max, 500)
        self.assertTrue(np.array_equal(pos_a, pos_b))
        self.assertEqual(mass_a, mass_b)

    def test_histogram_bins_span_the_physical_domain(self):
        cfg = _cfg(32)
        x = make_grid(cfg)
        dx = float(x[1] - x[0])
        # One particle at every cell center reconstructs a constant density.
        u = _reconstruct_density_particles(
            x.copy(), total_mass=1.0, x_grid=x, recon_method="histogram",
            bandwidth=0.0, x_min=cfg.domain.x_min, x_max=cfg.domain.x_max)
        self.assertTrue(np.allclose(u, 1.0, atol=1e-12))
        self.assertAlmostEqual(float(dx * np.sum(u)), 1.0, places=13)


if __name__ == "__main__":
    unittest.main()
