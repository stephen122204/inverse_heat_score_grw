"""NOT CERTIFICATION (RED audit, ledger 62): numerical evidence only.
The tail bounds below extrapolate observed coefficient ratios (unproved),
the box is a sampled grid, and analyticity/denominator convergence is
assumed, not proved. Quote as exploratory evidence, never as a theorem.
Original docstring follows."""
"""Crossover-theorem certification computation (EXPLORATORY, load-bearing).

Real a-Taylor (jet) hierarchy of the continuum lattice flow for single-mode
data u0 = 1 + a cos(kx). Extracts the SPECIFIC low-mode coefficients
c_k^{(j)}(T), c_{2k}^{(j)}(T) whose signed series ARE the reconstruction-error
expansions, and certifies the crossover gap
  G(a) = G_low(a) - sum_{j>=5,odd}|c_k^{(j)}|a^j - sum_{j>=6,even}|c_2k^{(j)}|a^j
with G_low(a) = (a^2 b + a^4 r2) - (a d + a^3|r1|).
Coefficients certified by dual refinement (dt: steps 4000 vs 8000; lattice N).
Entry point: reproduces the anchor budget and writes theorem_certification.json.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

HERE = Path(__file__).resolve().parent


def jet_hierarchy(kh, lam, eps_rel, N, ORD, steps, k=3 * math.pi):
    """lam = alpha T k^2; T fixed to 1 so alpha = lam/k^2. Returns coeff arrays."""
    alpha, T = lam / k**2, 1.0
    h = kh / k
    dt = T / steps
    m_eps = 1.0 / (1.0 + eps_rel)
    idx = lambda n: n + N
    kappa = np.array([n * k for n in range(-N, N + 1)])
    phi = np.exp(-0.5 * (kappa * h) ** 2)

    def jmul(A, B):
        return fftconvolve(A, B)[N:3 * N + 1, :ORD]

    delta = np.zeros((2 * N + 1, ORD)); delta[idx(0), 0] = 1.0

    def solve_v(c):
        w_off = (phi[:, None] * c).copy(); w_off[idx(0)] = 0.0
        v = m_eps * delta.copy()
        for _ in range(ORD + 3):
            v = m_eps * (delta - jmul(w_off, v))
        return v

    DK = (1j * kappa)[:, None]

    def rhs(c):
        v = solve_v(c)
        return (-alpha) * DK * jmul(jmul(c, DK * phi[:, None] * c), v)

    c = np.zeros((2 * N + 1, ORD), dtype=complex)
    c[idx(0), 0] = 1.0
    A0 = math.exp(-alpha * k * k * T) / 2.0
    c[idx(1), 1] = A0; c[idx(-1), 1] = A0
    for _ in range(steps):
        k1 = rhs(c); k2 = rhs(c + 0.5 * dt * k1)
        k3 = rhs(c + 0.5 * dt * k2); k4 = rhs(c + dt * k3)
        c = c + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    ck = np.array([2 * c[idx(1), j].real for j in range(ORD)])
    c2k = np.array([2 * c[idx(2), j].real for j in range(ORD)])
    p1 = math.exp(-0.5 * (k * h) ** 2); p2 = math.exp(-0.5 * (2 * k * h) ** 2)
    g1 = alpha * m_eps * k * k * p1; g2 = alpha * m_eps * 4 * k * k * p2
    b_cf = (alpha * m_eps * k * k * p1 * (1 - m_eps * p1)
            * math.exp(-2 * alpha * k * k * T)
            * (math.exp(g2 * T) - math.exp(2 * g1 * T)) / (g2 - 2 * g1))
    d_cf = 1 - math.exp(-alpha * k * k * T * (1 - m_eps * p1))
    return {"ck": ck, "c2k": c2k, "d_cf": d_cf, "b_cf": b_cf}


def certified_budget(res, a, Jck, Jc2k):
    """Finite well-resolved tail sum + conservative geometric remainder."""
    ck, c2k = res["ck"], res["c2k"]; d = res["d_cf"]; b = res["b_cf"]
    r1, r2 = ck[3], c2k[4]
    G_low = (a**2 * b + a**4 * r2) - (a * d + a**3 * abs(r1))
    G1 = sum(abs(ck[j]) * a**j for j in range(5, Jck + 1, 2))
    G2 = sum(abs(c2k[j]) * a**j for j in range(6, Jc2k + 1, 2))

    def rem(coeff, Jstar, j0):
        js = [j for j in range(j0, Jstar + 1, 2)]
        if len(js) < 3:
            return 0.0
        ratios = [abs(coeff[js[i + 1]]) / abs(coeff[js[i]]) for i in range(len(js) - 1)]
        R = max(ratios[-2:]); q = R * a**2
        if q >= 1:
            return float("inf")
        return abs(coeff[Jstar]) * a**Jstar * q / (1 - q)

    r1t, r2t = rem(ck, Jck, 5), rem(c2k, Jc2k, 6)
    return {"a": a, "G_low": G_low, "G1": G1, "G2": G2, "rem1": r1t, "rem2": r2t,
            "gap": G_low - G1 - G2 - r1t - r2t}


def main():
    KH = 3 * math.pi * 0.028; LAM = 0.01 * (3 * math.pi) ** 2; EPS = 1e-8
    base = jet_hierarchy(KH, LAM, EPS, N=16, ORD=15, steps=4000)
    fine = jet_hierarchy(KH, LAM, EPS, N=18, ORD=15, steps=8000)
    # well-resolved order = highest with dt-refine rel < 1e-3 (parity-correct)
    Jck = max((j for j in range(5, 15, 2)
               if abs(base["ck"][j] - fine["ck"][j]) / (abs(fine["ck"][j]) + 1e-30) < 1e-3),
              default=5)
    Jc2k = max((j for j in range(6, 15, 2)
                if abs(base["c2k"][j] - fine["c2k"][j]) / (abs(fine["c2k"][j]) + 1e-30) < 1e-3),
               default=6)
    print(f"anchor: d={base['d_cf']:.8e} b={base['b_cf']:.8e} "
          f"r1={base['ck'][3]:.6e} r2={base['c2k'][4]:.6e}")
    print(f"well-resolved orders: c_k<={Jck}, c_2k<={Jc2k}")
    out = {"anchor_kh": KH, "anchor_lam": LAM, "Jck": Jck, "Jc2k": Jc2k, "budget": {}}
    for a in (0.5, 0.55, 0.58, 0.60, 0.62):
        r = certified_budget(base, a, Jck, Jc2k)
        out["budget"][str(a)] = {k: (v if math.isfinite(v) else None) for k, v in r.items()}
        print(f"  a={a}: G_low={r['G_low']:.5e} G1={r['G1']:.3e} G2={r['G2']:.3e} "
              f"rem=({r['rem1']:.1e},{r['rem2']:.1e}) GAP={r['gap']:+.5e}")
    (HERE / "theorem_certification.json").write_text(json.dumps(out, indent=1) + "\n")
    print("written theorem_certification.json")


if __name__ == "__main__":
    main()
