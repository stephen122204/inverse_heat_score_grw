"""Provisional C5, C6 preview (gate step 5, front-loaded; EXPLORATORY).

Jets in (a - xi) around nonzero base amplitudes xi in [0, 0.6]: the
order-0 slot carries the full nonlinear truncated-lattice base solution
(with its cascade), orders 1..6 are the a-derivatives along the family.
Provisional Lagrange constants C5 = max_xi |d^5_a c_k(T)|/5!,
C6 = max_xi |d^6_a c_{2k}(T)|/6!; then the provisional crossover
inequality on a = 0.5..0.6. Floating-point preview, NOT certification.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

HERE = Path(__file__).resolve().parent
ALPHA, T, EPS_REL, H, NMODE = 0.01, 1.0, 1e-8, 0.028, 3
K = NMODE * math.pi
N, ORD, DT = 12, 7, 2.5e-4
idx = lambda n: n + N
phi = np.array([math.exp(-0.5 * (n * K * H) ** 2) for n in range(-N, N + 1)])
m_eps = 1.0 / (1.0 + EPS_REL)
delta = np.zeros((2 * N + 1, ORD), dtype=complex); delta[idx(0), 0] = 1.0


def jmul(A, B):
    full = fftconvolve(A, B)                     # 2D convolution
    return full[N:3 * N + 1, :ORD]               # crop modes -N..N, orders


def solve_v(c):
    w_off = (phi[:, None] * c).copy(); w_off[idx(0)] = 0.0
    v = m_eps * delta.copy()
    for _ in range(ORD + 3):
        v = m_eps * (delta - jmul(w_off, v))
    return v


DK = np.array([(1j * n * K) for n in range(-N, N + 1)])


def rhs(c):
    v = solve_v(c)
    dW = DK[:, None] * phi[:, None] * c
    G = jmul(jmul(c, dW), v)
    return (-ALPHA) * DK[:, None] * G


def run(xi):
    c = np.zeros((2 * N + 1, ORD), dtype=complex)
    c[idx(0), 0] = 1.0
    A0 = math.exp(-ALPHA * K * K * T) / 2.0
    for s in (1, -1):
        c[idx(s), 0] = xi * A0
        c[idx(s), 1] = A0
    for _ in range(round(T / DT)):
        k1 = rhs(c); k2 = rhs(c + 0.5 * DT * k1)
        k3 = rhs(c + 0.5 * DT * k2); k4 = rhs(c + DT * k3)
        c = c + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    f = math.factorial
    return {"d5_ck_over_5!": 2 * abs(c[idx(1), 5].real) * f(5) / f(5) / f(5) * f(5),
            "raw_d5_ck/5!": 2 * c[idx(1), 5].real,
            "raw_d6_c2k/6!": 2 * c[idx(2), 6].real}


# jets give Taylor coefficients directly: coefficient of (a-xi)^j IS d^j/j!
results = {}
for xi in (0.0, 0.15, 0.3, 0.45, 0.55, 0.6):
    r = run(xi)
    results[str(xi)] = {"taylor5_mode_k": r["raw_d5_ck/5!"],
                        "taylor6_mode_2k": r["raw_d6_c2k/6!"]}
    print(f"xi={xi:4.2f}: [d^5 c_k]/5! = {r['raw_d5_ck/5!']:+.5e}   "
          f"[d^6 c_2k]/6! = {r['raw_d6_c2k/6!']:+.5e}")

C5 = max(abs(v["taylor5_mode_k"]) for v in results.values())
C6 = max(abs(v["taylor6_mode_2k"]) for v in results.values())
d, r1 = 2.993971e-02, -3.517922e-03
b, r2 = 5.938425e-02, 1.442372e-02
print(f"\nprovisional C5 = {C5:.4e}, C6 = {C6:.4e}")
print(f"{'a':>5} {'harmonic lower':>15} {'linear upper':>13} {'margin':>10}")
rows = {}
for a in (0.5, 0.55, 0.6):
    harm = a*a*b + a**4*r2 - C6*a**6
    lin = a*d + a**3*abs(r1) + C5*a**5
    rows[str(a)] = {"harmonic_lower": harm, "linear_upper": lin,
                    "margin": harm - lin}
    print(f"{a:5.2f} {harm:15.5e} {lin:13.5e} {harm-lin:+10.3e}")
(HERE / "provisional_constants.json").write_text(json.dumps(
    {"C5": C5, "C6": C6, "per_xi": results, "crossover_check": rows,
     "note": "floating-point preview, not certification"},
    indent=1) + "\n", encoding="utf-8")
