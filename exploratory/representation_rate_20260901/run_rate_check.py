"""Initial-rate check: does the wrong-limit separation grow at ||D[g]||?

For G1 (u0 = 2 + cos(pi x), alpha = 0.01, T = 1) the representation defect
for the gradient representation at the terminal datum g is

  D(x) = [wrong-limit RHS] - [true carrier RHS] at reverse time 0,

with the wrong-limit RHS = -alpha * d/dx (q^2/U) and the true RHS
= d/dtau q_true(0). Both are closed-form. The unregularized reference
solver (grid-based, no carrier binning) measures the actual separation
e(tau) = ||q_ref(tau) - q_true(tau)||_2, which the short-time theorem
predicts to be tau * ||D||_2 + O(tau^2). EXPLORATORY.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from invheat_grw.campaign_closure import run_reference
from invheat_grw.cell_grid import cell_centers, cell_spacing, midpoint_norm

ALPHA, BIG_T, M, DT = 0.01, 1.0, 1600, 2.5e-4
b = math.exp(-ALPHA * math.pi ** 2 * BIG_T)     # terminal mode amplitude
x = cell_centers(0.0, 1.0, M)
dx = cell_spacing(0.0, 1.0, M)
edges = np.arange(M + 1) * dx

g_centers = 2.0 + b * np.cos(np.pi * x)
q0 = np.diff(2.0 + b * np.cos(np.pi * edges)) / dx   # exact cell averages

s, c = np.sin(np.pi * x), np.cos(np.pi * x)
u = 2.0 + b * c
wrong_rhs = -(ALPHA * math.pi ** 3 * b ** 2
              * (2.0 * s * c * (2.0 + b * c) + b * s ** 3) / (2.0 + b * c) ** 2)
true_rhs = -ALPHA * math.pi ** 3 * b * s        # d/dtau of q_true at tau = 0
defect = wrong_rhs - true_rhs
d_norm = midpoint_norm(defect, dx)
print(f"||D[g]||_2 = {d_norm:.6e}")

print(f"{'tau':>8} {'e(tau)':>12} {'e/tau':>12} {'e/(tau*||D||)':>14}")
for tau in (0.0125, 0.025, 0.05, 0.1, 0.2, 0.4):
    ref = run_reference(q0, kind="unregularized", closure="mass",
                        anchor_value=float(g_centers[0]),
                        total_mass=float(dx * np.sum(g_centers)),
                        x_min=0.0, x_max=1.0, T=tau, dt=DT, alpha=ALPHA)
    assert ref.status == "completed", ref.failure_message
    q_true = -math.pi * b * math.exp(ALPHA * math.pi ** 2 * tau) * s
    e = midpoint_norm(ref.q - q_true, dx)
    print(f"{tau:8.4f} {e:12.5e} {e / tau:12.5e} {e / (tau * d_norm):14.4f}")
