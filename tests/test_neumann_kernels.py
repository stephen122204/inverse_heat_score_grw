from __future__ import annotations

import copy
import unittest
from pathlib import Path

import numpy as np

from invheat_grw.config import load_config
from invheat_grw.fields import make_grid
from invheat_grw.methods import run_density_particle_estimated_score_deterministic
from invheat_grw.neumann_kernels import (
    neumann_kde_density_derivative,
    neumann_kde_score,
    neumann_mode_count,
)


REPO = Path(__file__).resolve().parent.parent
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


class NeumannKernelTests(unittest.TestCase):
    def test_mode_count_controls_first_omitted_multiplier(self):
        h, length, tol = 0.04, 1.0, 1e-14
        count = neumann_mode_count(h, length, tol)
        omitted = np.exp(-0.5 * (((count + 1) * np.pi * h / length) ** 2))
        self.assertLessEqual(omitted, tol)
        if count:
            retained = np.exp(-0.5 * ((count * np.pi * h / length) ** 2))
            self.assertGreater(retained, tol)

    def test_mass_and_wall_derivative_for_wall_near_particles(self):
        x = np.linspace(0.0, 1.0, 20_001)
        positions = np.array([0.0, 0.03, 0.51, 0.97, 1.0])
        weights = np.array([0.2, 0.4, 0.7, 0.5, 0.3])
        density, derivative, diagnostics = neumann_kde_density_derivative(
            x, positions, weights, 0.0, 1.0, 0.04
        )
        self.assertAlmostEqual(float(_trapz(density, x)), float(weights.sum()), places=11)
        self.assertLess(abs(float(derivative[0])), 1e-12)
        self.assertLess(abs(float(derivative[-1])), 1e-12)
        self.assertLessEqual(diagnostics.omitted_multiplier_bound, 1e-14)
        self.assertGreater(float(density.min()), -1e-12)

    def test_analytic_derivative_matches_centered_difference(self):
        positions = np.array([0.08, 0.31, 0.55, 0.91])
        weights = np.array([0.2, 0.3, 0.1, 0.4])
        x = np.linspace(0.05, 0.95, 301)
        step = 1e-6
        density, derivative, _ = neumann_kde_density_derivative(
            x, positions, weights, 0.0, 1.0, 0.055
        )
        plus, _, _ = neumann_kde_density_derivative(
            x + step, positions, weights, 0.0, 1.0, 0.055
        )
        minus, _, _ = neumann_kde_density_derivative(
            x - step, positions, weights, 0.0, 1.0, 0.055
        )
        finite_difference = (plus - minus) / (2.0 * step)
        scale = max(1.0, float(np.max(np.abs(derivative))))
        self.assertLess(float(np.max(np.abs(finite_difference - derivative))) / scale, 2e-8)
        self.assertTrue(np.all(np.isfinite(density)))

    def test_equal_cell_mass_reconstructs_constant_density(self):
        n_particles = 512
        positions = (np.arange(n_particles) + 0.5) / n_particles
        weights = np.full(n_particles, 1.0 / n_particles)
        x = np.linspace(0.0, 1.0, 1001)
        density, derivative, score, _ = neumann_kde_score(
            x, positions, weights, 0.0, 1.0, 0.04, epsilon=1e-8
        )
        self.assertLess(float(np.max(np.abs(density - 1.0))), 2e-13)
        self.assertLess(float(np.max(np.abs(derivative))), 2e-12)
        self.assertLess(float(np.max(np.abs(score))), 2e-12)

    def test_canonical_integration_preserves_constant_field(self):
        cfg = copy.deepcopy(load_config(str(REPO / "configs" / "gaussian_base.yaml")))
        cfg.domain.n_grid = 101
        cfg.heat.T = 0.002
        cfg.heat.dt = 0.001
        x = make_grid(cfg)
        result = run_density_particle_estimated_score_deterministic(
            np.ones_like(x),
            x,
            cfg,
            n_particles=256,
            recon_method="neumann_kde",
            bandwidth=0.04,
            epsilon=1e-8,
            score_method="neumann_kde",
            save_snapshots=False,
        )
        self.assertTrue(result.completed, result.failure_msg)
        self.assertLess(float(np.max(np.abs(result.candidate - 1.0))), 5e-13)
        self.assertAlmostEqual(result.mass_initial, 1.0, places=13)

    def test_canonical_score_rejects_mismatched_reconstruction(self):
        cfg = copy.deepcopy(load_config(str(REPO / "configs" / "gaussian_base.yaml")))
        cfg.domain.n_grid = 51
        cfg.heat.T = cfg.heat.dt
        x = make_grid(cfg)
        with self.assertRaisesRegex(ValueError, "requires recon_method='neumann_kde'"):
            run_density_particle_estimated_score_deterministic(
                np.ones_like(x),
                x,
                cfg,
                n_particles=32,
                recon_method="kde",
                bandwidth=0.05,
                epsilon=1e-8,
                score_method="neumann_kde",
                save_snapshots=False,
            )

    def test_canonical_score_requires_physical_bandwidth(self):
        cfg = copy.deepcopy(load_config(str(REPO / "configs" / "gaussian_base.yaml")))
        cfg.domain.n_grid = 51
        cfg.heat.T = cfg.heat.dt
        x = make_grid(cfg)
        with self.assertRaisesRegex(ValueError, "explicit physical bandwidth"):
            run_density_particle_estimated_score_deterministic(
                np.ones_like(x),
                x,
                cfg,
                n_particles=32,
                recon_method="neumann_kde",
                bandwidth=None,
                epsilon=1e-8,
                score_method="neumann_kde",
                save_snapshots=False,
            )


if __name__ == "__main__":
    unittest.main()
