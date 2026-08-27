"""Contract tests for the prospective cell-centered discretization.

These pin down the grid, transform, norm, quantile-initializer, and kernel
behavior the migrated method will rely on, before any production code is
rewired.  The two exactness tests matter most: on cell centers the DCT-II
modes are exactly the Neumann cosine eigenfunctions, so the heat multiplier
and the Neumann-kernel damping are exact spectral statements rather than
approximations.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from invheat_grw.cell_grid import (
    cell_centers,
    cell_edge_quantile_positions,
    cell_spacing,
    dct2_forward,
    dct2_inverse,
    midpoint_mass,
    midpoint_norm,
    propagate_heat,
    wave_numbers,
)
from invheat_grw.neumann_kernels import neumann_kde_density_derivative, neumann_kde_score


class CellGridContractTests(unittest.TestCase):
    def test_cell_coordinates_and_wall_independence(self):
        x_min, x_max, m = 0.2, 1.2, 5
        dx = cell_spacing(x_min, x_max, m)
        x = cell_centers(x_min, x_max, m)
        self.assertAlmostEqual(dx, 0.2, places=15)
        self.assertAlmostEqual(x[0] - x_min, dx / 2, places=15)
        self.assertAlmostEqual(x_max - x[-1], dx / 2, places=15)
        self.assertTrue(np.allclose(np.diff(x), dx, atol=1e-15))
        # The physical walls are the configuration bounds, never the samples.
        self.assertNotEqual(x[0], x_min)
        self.assertNotEqual(x[-1], x_max)

    def test_dct2_roundtrip(self):
        u = np.random.default_rng(1).standard_normal(257)
        back = dct2_inverse(dct2_forward(u))
        self.assertLess(float(np.max(np.abs(back - u))), 1e-13)

    def test_constant_field_is_preserved_exactly(self):
        u = np.ones(200)
        out = propagate_heat(u, length=1.0, alpha=0.01, t=0.15)
        self.assertLess(float(np.max(np.abs(out - 1.0))), 1e-14)

    def test_single_cosine_mode_gets_exact_heat_multiplier(self):
        x_min, x_max, m = 0.0, 1.0, 400
        length = x_max - x_min
        alpha, t, n, a = 0.01, 0.15, 7, 0.4
        x = cell_centers(x_min, x_max, m)
        k = float(wave_numbers(m, length)[n])
        u = 1.0 + a * np.cos(k * (x - x_min))
        out = propagate_heat(u, length, alpha, t)
        exact = 1.0 + a * np.exp(-alpha * k ** 2 * t) * np.cos(k * (x - x_min))
        self.assertLess(float(np.max(np.abs(out - exact))), 1e-14)

    def test_neumann_kernel_damps_one_mode_exactly(self):
        x_min, x_max, m = 0.0, 1.0, 256
        length = x_max - x_min
        n, a, h = 12, 0.3, 0.02
        x = cell_centers(x_min, x_max, m)
        dx = cell_spacing(x_min, x_max, m)
        k = float(wave_numbers(m, length)[n])
        u = 1.0 + a * np.cos(k * (x - x_min))
        # Particles at the cell centers carrying the midpoint cell masses:
        # discrete DCT-II orthogonality makes the damped mode amplitude exact.
        density, _, _ = neumann_kde_density_derivative(
            x, x, u * dx, x_min, x_max, h)
        amp = float(2.0 / m * np.sum(density * np.cos(k * (x - x_min))))
        expected = a * np.exp(-0.5 * (k * h) ** 2)
        self.assertLess(abs(amp - expected), 1e-12)
        self.assertAlmostEqual(float(dx * np.sum(density)),
                               midpoint_mass(u, dx), places=13)

    def test_midpoint_mass_and_norm_hand_values(self):
        u = np.array([2.0, 4.0])
        self.assertAlmostEqual(midpoint_mass(u, 0.5), 3.0, places=15)
        self.assertAlmostEqual(midpoint_norm(u, 0.5), np.sqrt(10.0), places=15)

    def test_quantile_positions_for_uniform_density_are_exact(self):
        m, n_particles = 50, 10
        u = np.ones(m)
        pos, total = cell_edge_quantile_positions(u, 0.0, 1.0, n_particles)
        expected = (np.arange(n_particles) + 0.5) / n_particles
        self.assertTrue(np.allclose(pos, expected, atol=1e-14))
        self.assertAlmostEqual(total, 1.0, places=15)

    def test_quantile_positions_respect_cell_masses(self):
        m, n_particles = 64, 1000
        u = np.concatenate([np.ones(m // 2), 3.0 * np.ones(m // 2)])
        pos, total = cell_edge_quantile_positions(u, 0.0, 1.0, n_particles)
        self.assertAlmostEqual(total, 2.0, places=14)
        left = int(np.sum(pos < 0.5))
        self.assertLessEqual(abs(left - n_particles // 4), 1)
        self.assertTrue(np.all(np.diff(pos) >= 0.0))

    def test_quantile_positions_clip_negative_cells(self):
        u = np.array([-1.0, 2.0, -3.0, 2.0])
        pos, total = cell_edge_quantile_positions(u, 0.0, 1.0, 8)
        self.assertAlmostEqual(total, 1.0, places=15)
        inside_positive_cells = ((pos >= 0.25) & (pos <= 0.5)) | (pos >= 0.75)
        self.assertTrue(np.all(inside_positive_cells))

    def test_epsilon_zero_is_deliberately_unusable_in_deep_tails(self):
        """With eps=0 the truncated far-tail score is meaningless (junk or
        NaN); a positive eps above the truncation floor flattens it to ~0.
        The canonical method must therefore assume eps > 0."""
        h = 0.02
        positions = np.full(64, 0.1)
        weights = np.full(64, 1.0 / 64.0)
        far = np.array([0.9])
        # True far-tail score is dominated by the direct image: -(0.9-0.1)/h^2.
        true_score = -(0.9 - 0.1) / h ** 2
        _, _, s0, _ = neumann_kde_score(far, positions, weights, 0.0, 1.0, h,
                                        epsilon=0.0)
        bad = (not np.all(np.isfinite(s0))) or abs(float(s0[0]) - true_score) > 1e3
        self.assertTrue(bad, f"eps=0 far-tail score unexpectedly usable: {s0}")
        _, _, s_eps, _ = neumann_kde_score(far, positions, weights, 0.0, 1.0, h,
                                           epsilon=1e-8)
        self.assertTrue(np.all(np.isfinite(s_eps)))
        self.assertLess(abs(float(s_eps[0])), 1.0)


if __name__ == "__main__":
    unittest.main()
