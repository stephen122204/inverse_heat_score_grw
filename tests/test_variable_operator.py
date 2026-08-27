"""Acceptance tests for the conservative cell-centered variable operator.

Structure (symmetry, nullspace, mass conservation, dissipation, adjoint
identity) plus a forced manufactured-solution convergence study at second
order in space and time.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from invheat_grw.cell_grid import cell_centers, midpoint_norm

_spec = importlib.util.spec_from_file_location(
    "vc_audit", str(REPO / "scripts" / "run_variable_coefficient_audit.py"))
vc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vc)  # type: ignore[union-attr]

ALPHA0, BETA = 0.01, 0.9


def _operator(m: int):
    x = cell_centers(0.0, 1.0, m)
    return x, vc.build_varcoeff_operator(x, ALPHA0, BETA).toarray()


class VariableOperatorStructureTests(unittest.TestCase):
    def test_symmetric(self):
        _, L = _operator(100)
        self.assertLess(float(np.max(np.abs(L - L.T))), 1e-12 * np.max(np.abs(L)))

    def test_constants_in_nullspace(self):
        _, L = _operator(100)
        self.assertLess(float(np.max(np.abs(L @ np.ones(100)))), 1e-9)

    def test_mass_conservation_zero_column_sums(self):
        _, L = _operator(100)
        self.assertLess(float(np.max(np.abs(np.ones(100) @ L))), 1e-9)

    def test_negative_semidefinite(self):
        _, L = _operator(100)
        eigs = np.linalg.eigvalsh(0.5 * (L + L.T))
        self.assertLessEqual(float(eigs.max()), 1e-8)

    def test_adjoint_identity(self):
        _, L = _operator(100)
        rng = np.random.default_rng(5)
        z, y = rng.standard_normal(100), rng.standard_normal(100)
        self.assertAlmostEqual(float(z @ (L @ y)), float((L @ z) @ y), places=8)


class NonfiniteScoreControlFlowTests(unittest.TestCase):
    def test_estimated_path_fails_loudly_on_nonfinite_scores(self):
        """A poisoned score estimator must abort the run with a failure
        record, never be silently zeroed (the endpoint-era fallback)."""
        m = 32
        x = cell_centers(0.0, 1.0, m)
        u_obs = 1.0 + 0.5 * np.cos(np.pi * x)
        snapshots = {k: u_obs.copy() for k in range(3)}

        def poisoned(x_eval, u_grid, x_grid, smooth_sigma, epsilon=0.0):
            return np.full(np.shape(x_eval), np.nan), {}

        original = vc.smoothed_log_score
        vc.smoothed_log_score = poisoned
        self.addCleanup(setattr, vc, "smoothed_log_score", original)

        r = vc.run_varcoeff_estimated(
            u_obs, x, snapshots, 0.01, 0.5, 0.001, 2,
            score_method="smoothed_log", bandwidth_factor=4.0, n_particles=64,
            x_min=0.0, x_max=1.0)
        self.assertFalse(r["completed"])
        self.assertEqual(r["failure_step"], 0)
        self.assertIn("NaN/Inf in estimated score", r["failure_msg"])


class ForcedManufacturedSolutionTests(unittest.TestCase):
    """u(x,t) = e^{-t} (1 + 0.5 cos(pi x)) satisfies Neumann walls exactly;
    the forcing f = u_t - d/dx(a du/dx) is analytic."""

    @staticmethod
    def _exact(x, t):
        return np.exp(-t) * (1.0 + 0.5 * np.cos(np.pi * x))

    @staticmethod
    def _forcing(x, t):
        v = 1.0 + 0.5 * np.cos(np.pi * x)
        vp = -0.5 * np.pi * np.sin(np.pi * x)
        vpp = -0.5 * np.pi ** 2 * np.cos(np.pi * x)
        a = ALPHA0 * (1.0 + BETA * np.sin(2.0 * np.pi * x))
        ap = 2.0 * np.pi * ALPHA0 * BETA * np.cos(2.0 * np.pi * x)
        return np.exp(-t) * (-v - (ap * vp + a * vpp))

    def test_second_order_convergence_in_space_and_time(self):
        T = 0.1
        errors = []
        for m in (50, 100, 200):
            x = cell_centers(0.0, 1.0, m)
            dx = float(x[1] - x[0])
            n_steps = 2 * m                 # dt proportional to dx, exact horizon
            dt = T / n_steps
            u_T, _ = vc.solve_varcoeff_forward(
                self._exact(x, 0.0), x, ALPHA0, BETA, dt, n_steps,
                forcing=self._forcing)
            err = midpoint_norm(u_T - self._exact(x, T), dx) \
                / midpoint_norm(self._exact(x, T), dx)
            errors.append(err)
        orders = [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
        for order in orders:
            self.assertGreater(order, 1.8,
                               f"errors={errors}, observed orders={orders}")
        self.assertLess(errors[-1], 1e-4)


if __name__ == "__main__":
    unittest.main()
