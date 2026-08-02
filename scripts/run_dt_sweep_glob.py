"""
run_dt_sweep_glob.py — time-step refinement sweep for the exact-score
gradient-glob reconstruction (Test B), with a density-particle control.

Regenerates the round-3 sweep pinned in FROZEN_NUMBERS.md: Test B, alpha=0.01,
n_grid=400, globs_per_jump=80, dt in {0.002, 0.001, 0.0005, 0.00025}; density
control at N=5000, KDE reconstruction, bandwidth factor 4.  Backs the
time-step sentence in the representation-evidence section of the paper
(dt-invariance of the gradient-glob error).

    python scripts/run_dt_sweep_glob.py

Output: outputs/dt_sweep_glob_<timestamp>/dt_sweep_glob.csv
Deterministic (exact analytic score, no RNG in the drift); reruns are
bit-identical.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import pandas as pd

from run_representation_audit import (
    build_test_configs,
    patch_config,
    run_gradient_glob_cell,
    run_density_particle_cell,
)
from invheat_grw.config import load_config

DT_VALUES = [0.002, 0.001, 0.0005, 0.00025]
N_GRID = 400
GLOBS_PER_JUMP = 80
N_PARTICLES = 5000
BW_FACTOR = 4.0


def main() -> None:
    base = load_config(str(REPO / "configs" / "gaussian_base.yaml"))
    mix = load_config(str(REPO / "configs" / "gaussian_mixture.yaml"))
    cfg_b = build_test_configs(base, mix)["B"]

    rows = []
    for dt in DT_VALUES:
        cfg_dt = patch_config(cfg_b, **{"heat.dt": dt})
        t0 = time.perf_counter()
        g = run_gradient_glob_cell("B", cfg_dt, N_GRID, GLOBS_PER_JUMP)
        g.update({"dt": dt, "sweep_method": "gradient_glob_oracle"})
        rows.append(g)
        d = run_density_particle_cell("B", cfg_dt, N_GRID, N_PARTICLES, "kde", BW_FACTOR)
        d.update({"dt": dt, "sweep_method": "density_particle_oracle"})
        rows.append(d)
        print(f"dt={dt}: glob rel_l2={g['relative_l2']:.6f}  "
              f"density rel_l2={d['relative_l2']:.6f}  ({time.perf_counter()-t0:.1f}s)",
              flush=True)

    out_dir = REPO / "outputs" / f"dt_sweep_glob_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "dt_sweep_glob.csv", index=False)
    print(f"wrote {out_dir / 'dt_sweep_glob.csv'}")


if __name__ == "__main__":
    main()
