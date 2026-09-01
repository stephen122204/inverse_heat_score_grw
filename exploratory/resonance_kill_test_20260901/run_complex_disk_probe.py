"""Provisional complex-disk majorant probe (Route 1 front-load, EXPLORATORY).

Runs the collocation solver at COMPLEX amplitudes a = rho e^{i theta} on
the upper semicircle (real-analyticity gives the lower half by
conjugation), tracking min |K_h rho + eps| throughout (the complex
denominator condition real positivity does not supply). From the circle
maxima of the RESIDUAL functions
  g_k(a)  := c_k(a)  - (a(1-d) ... linear part) + a d - a^3 r1   [vanishes to O(a^5)]
  g_2k(a) := c_2k(a) - a^2 b - a^4 r2                            [vanishes to O(a^6)]
provisional Cauchy tail bounds T5(a) = max|g_k| (a/rho)^5/(1-a/rho),
T6(a) analogously, evaluated at a = 0.55, 0.6 against the measured
crossover margins. Floating-point preview, not certification.
"""
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ALPHA, T, EPS_REL = 0.01, 1.0, 1e-8
M, NMODES, DT, H = 1024, 150, 5e-4, 0.028   # coarse grid; clean at h=0.028 per seed experiment
NMODE = 3
K = np.arange(1, NMODES + 1) * math.pi
x = (np.arange(M) + 0.5) / M
dx = 1.0 / M
COS = np.cos(np.outer(x, K)); SIN = np.sin(np.outer(x, K))
phi = np.exp(-0.5 * (K * H) ** 2)
k0 = NMODE * math.pi
d_lin, r1 = 2.993971e-02, -3.517922e-03
b2, r2 = 5.938425e-02, 1.442372e-02


def run_complex(a):
    rho = (1.0 + a * math.exp(-ALPHA * k0 * k0 * T)
           * np.cos(k0 * x)).astype(complex)
    min_den = np.inf

    def rhs(r):
        nonlocal min_den
        c = 2.0 * dx * (COS.T @ r)
        w = r.mean() + COS @ (c * phi)
        den = w + EPS_REL
        min_den = min(min_den, float(np.min(np.abs(den))))
        wx = -SIN @ (c * phi * K)
        F = r * wx / den
        f_sin = 2.0 * dx * (SIN.T @ F)
        return -ALPHA * (COS @ (f_sin * K))

    for _ in range(round(T / DT)):
        k1 = rhs(rho); k2 = rhs(rho + 0.5 * DT * k1)
        k3 = rhs(rho + 0.5 * DT * k2); k4 = rhs(rho + DT * k3)
        rho = rho + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if not np.all(np.isfinite(rho.real)) or np.max(np.abs(rho)) > 1e6:
            return None, min_den
    ce = 2.0 * dx * (COS.T @ rho)          # complex mode coefficients
    return ce, min_den


results = {}
for rho_r in (0.75, 0.9, 1.0):
    maxg1 = maxg2 = 0.0
    min_den_all = np.inf
    ok = True
    for deg in (0, 30, 60, 90, 120, 150, 180):
        a = rho_r * np.exp(1j * math.radians(deg))
        ce, mind = run_complex(a)
        min_den_all = min(min_den_all, mind)
        if ce is None:
            ok = False
            results[str(rho_r)] = {"stable": False, "blowup_at_deg": deg,
                                   "min_denominator": min_den_all}
            break
        c1 = ce[NMODE - 1] - a * (1 - d_lin) - a**3 * r1
        c2 = ce[2 * NMODE - 1] - a**2 * b2 - a**4 * r2
        maxg1 = max(maxg1, abs(c1)); maxg2 = max(maxg2, abs(c2))
    if ok:
        entry = {"stable": True, "min_denominator": min_den_all,
                 "max_residual_k": maxg1, "max_residual_2k": maxg2,
                 "tails": {}}
        for a in (0.55, 0.6):
            q = a / rho_r
            entry["tails"][str(a)] = {
                "T5_mode_k": maxg1 * q**5 / (1 - q),
                "T6_mode_2k": maxg2 * q**6 / (1 - q)}
        results[str(rho_r)] = entry
    print(rho_r, json.dumps(results[str(rho_r)], indent=1))

margins = {"0.55": 2.343e-3, "0.6": 4.727e-3}
verdict = {}
for rho_r, e in results.items():
    if e.get("stable"):
        for a, tl in e["tails"].items():
            tot = tl["T5_mode_k"] + tl["T6_mode_2k"]
            verdict[f"rho={rho_r},a={a}"] = {
                "tail_total": tot, "margin": margins[a],
                "fits": bool(tot < margins[a])}
print(json.dumps(verdict, indent=1))
(HERE / "complex_disk_probe.json").write_text(json.dumps(
    {"results": results, "margins": margins, "verdict": verdict,
     "note": "floating-point preview, not certification"}, indent=1) + "\n",
    encoding="utf-8")
