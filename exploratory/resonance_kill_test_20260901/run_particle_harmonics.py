"""Gate item (iv), reproducible: harmonic signature in the particle
reconstruction. Runs the production path on the C1 case and writes the
error's cosine coefficients machine-readably. EXPLORATORY."""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import numpy as np
from invheat_grw.campaign_particle import run_campaign_density
from invheat_grw.cell_grid import cell_centers, cell_spacing
from provenance import git_commit

ALPHA, T, M = 0.01, 1.0, 400
A, N_MODE, H = 0.5, 3, 0.028
EPS_REL, N_PART, DT = 1e-8, 4000, 1e-3

x = cell_centers(0.0, 1.0, M)
dx = cell_spacing(0.0, 1.0, M)
k = N_MODE * math.pi
u0 = 1.0 + A * np.cos(k * x)
g = 1.0 + A * math.exp(-ALPHA * k * k * T) * np.cos(k * x)

res = run_campaign_density(g, x_min=0.0, x_max=1.0, T=T, dt=DT,
                           n_particles=N_PART, bandwidth=H, eps_rel=EPS_REL,
                           alpha=ALPHA, u0_reference=u0)
assert res.status == "completed", res.failure_message
err = res.reconstruction - u0
E2 = float(math.sqrt(dx * np.sum(err ** 2)) / math.sqrt(dx * np.sum(u0 ** 2)))

m_ = 1.0 / (1.0 + EPS_REL)
p1 = math.exp(-0.5 * (k * H) ** 2)
p2 = math.exp(-0.5 * (2 * k * H) ** 2)
g1 = ALPHA * m_ * k * k * p1
g2 = ALPHA * m_ * 4 * k * k * p2
b = (ALPHA * m_ * k * k * p1 * (1 - m_ * p1) * math.exp(-2 * ALPHA * k * k * T)
     * (math.exp(g2 * T) - math.exp(2 * g1 * T)) / (g2 - 2 * g1))

out = {
    "commit": git_commit(REPO),
    "params": {"alpha": ALPHA, "T": T, "M": M, "a": A, "n": N_MODE, "h": H,
               "eps_rel": EPS_REL, "n_particles": N_PART, "dt": DT},
    "E2": E2,
    "error_cosine_coefficients": {
        str(m2): float(2.0 * dx * np.sum(err * np.cos(m2 * math.pi * x)))
        for m2 in (3, 6, 9, 12, 15)},
    "formula_second_harmonic": A * A * abs(b),
    "linear_signal_deficit": A * (1 - math.exp(-ALPHA * k * k * T
                                               * (1 - m_ * p1))),
}
path = HERE / "particle_harmonics.json"
path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
print(json.dumps(out, indent=1))
