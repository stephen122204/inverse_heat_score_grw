"""Window-instability seed isolation at (n=6, h=0.014) (EXPLORATORY).

Discriminates the seed of the O(1) failure at the double-precision
boundary: prescribed signal amplitude (a-sweep), exact constant state
(a=0), injected seeds of known size at a window mode, float32 floor,
and resolution/time-step variation. Writes JSON.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ALPHA, T, EPS_REL, H = 0.01, 1.0, 1e-8, 0.014
N_SIG = 6
WINDOW_MODE = 32          # k ~ sqrt(2)/h ~ 101 ~ 32*pi


def run(a, M=2048, NMODES=220, dt=2.5e-4, dtype=np.float64,
        seed_mode=None, seed_amp=0.0):
    x = ((np.arange(M) + 0.5) / M).astype(dtype)
    dx = dtype(1.0 / M)
    K = (np.arange(1, NMODES + 1) * math.pi).astype(dtype)
    COS = np.cos(np.outer(x, K)).astype(dtype)
    SIN = np.sin(np.outer(x, K)).astype(dtype)
    phi = np.exp(-0.5 * (K * dtype(H)) ** 2).astype(dtype)

    def rhs(rho):
        c = 2.0 * dx * (COS.T @ rho)
        w = rho.mean() + COS @ (c * phi)
        wx = -SIN @ (c * phi * K)
        F = rho * wx / (w + dtype(EPS_REL))
        f_sin = 2.0 * dx * (SIN.T @ F)
        return -dtype(ALPHA) * (COS @ (f_sin * K))

    k = N_SIG * math.pi
    rho = 1.0 + dtype(a * math.exp(-ALPHA * k * k * T)) * np.cos(dtype(k) * x)
    if seed_mode is not None:
        rho = rho + dtype(seed_amp) * np.cos(dtype(seed_mode * math.pi) * x)
    steps = round(T / dt)
    for step in range(steps):
        k1 = rhs(rho); k2 = rhs(rho + dtype(0.5 * dt) * k1)
        k3 = rhs(rho + dtype(0.5 * dt) * k2); k4 = rhs(rho + dtype(dt) * k3)
        rho = rho + dtype(dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        if not np.all(np.isfinite(rho)) or np.max(np.abs(rho)) > 1e6:
            return {"blowup_tau": step * dt}
    c_end = 2.0 * float(dx) * (COS.T.astype(np.float64) @ rho.astype(np.float64))
    return {"blowup_tau": None,
            "max_dev": float(np.max(np.abs(rho - 1.0))) if a == 0.0 else None,
            "high_mode_max": float(np.max(np.abs(c_end[25:])))}

cases = {
    "a0_float64": run(0.0),
    "a0_float32": run(0.0, dtype=np.float32),
    "a0_seed_1e-12": run(0.0, seed_mode=WINDOW_MODE, seed_amp=1e-12),
    "a0_seed_1e-10": run(0.0, seed_mode=WINDOW_MODE, seed_amp=1e-10),
    "a0_seed_1e-8": run(0.0, seed_mode=WINDOW_MODE, seed_amp=1e-8),
    "a0_M1024_N150": run(0.0, M=1024, NMODES=150),
    "a0_dt5e-4": run(0.0, dt=5e-4),
    "a0.1_float64": run(0.1),
}
out = {"params": {"alpha": ALPHA, "T": T, "h": H, "eps_rel": EPS_REL,
                  "window_mode": WINDOW_MODE},
       "amplification_at_window_e^{gamma T}": math.exp(
           ALPHA * (WINDOW_MODE * math.pi) ** 2
           * math.exp(-0.5 * (WINDOW_MODE * math.pi * H) ** 2) * T),
       "cases": cases}
(HERE / "window_seed_experiment.json").write_text(
    json.dumps(out, indent=1) + "\n", encoding="utf-8")
print(json.dumps(out, indent=1))
