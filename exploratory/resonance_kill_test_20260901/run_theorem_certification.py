"""Crossover-theorem certification computation (EXPLORATORY, load-bearing).

Real a-Taylor (jet) hierarchy of the continuum lattice flow for single-mode
data u0 = 1 + a cos(kx). Extracts the SPECIFIC low-mode coefficients
c_k^{(j)}(T) and c_{2k}^{(j)}(T) (mode indices 1 and 2 on the lattice),
whose signed series ARE the reconstruction-error expansions. The residual
tails are bounded directly by their own (real) coefficient series -
tighter than complex circle-max/Schwarz and needing no complex work:
  |e_k(a)|  <= a d + a^3 |r1| + G1tail(a),  G1tail = sum_{j>=5} |c_k^{(j)}| a^j
  |e_2k(a)| >= a^2 b + a^4 r2 - G2tail(a),  G2tail = sum_{j>=6} |c_2k^{(j)}| a^j
Crossover gap budget:  G_low(a) - G1tail(a) - G2tail(a) > 0.
Also computes A_j = full-lattice order norm (all-orders majorant diagnostic),
closed-form d and b for cross-check, and a dt-refinement certification.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

HERE = Path(__file__).resolve().parent


def jet_hierarchy(kh, lam, eps_rel, N, ORD, steps, k=3 * math.pi):
    """lam = alpha T k^2 ; returns dict of coefficient arrays c[mode,order]."""
    alpha_T = lam / k**2                 # alpha*T ; set T=1 => alpha=alpha_T
    alpha, T = alpha_T, 1.0
    h = kh / k
    dt = T / steps
    m_eps = 1.0 / (1.0 + eps_rel)
    idx = lambda n: n + N
    kappa = np.array([n * k for n in range(-N, N + 1)])
    phi = np.exp(-0.5 * (kappa * h) ** 2)

    def jmul(A, B):
        full = fftconvolve(A, B)
        return full[N:3 * N + 1, :ORD]

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
        dW = DK * phi[:, None] * c
        G = jmul(jmul(c, dW), v)
        return (-alpha) * DK * G

    c = np.zeros((2 * N + 1, ORD), dtype=complex)
    c[idx(0), 0] = 1.0
    A0 = math.exp(-alpha * k * k * T) / 2.0
    c[idx(1), 1] = A0; c[idx(-1), 1] = A0
    for _ in range(steps):
        k1 = rhs(c); k2 = rhs(c + 0.5 * dt * k1)
        k3 = rhs(c + 0.5 * dt * k2); k4 = rhs(c + dt * k3)
        c = c + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    # cosine-amplitude coefficient of lattice mode n>0 is 2*Re(c[n])
    ck = np.array([2 * c[idx(1), j].real for j in range(ORD)])   # mode k
    c2k = np.array([2 * c[idx(2), j].real for j in range(ORD)])  # mode 2k
    Aj = np.array([float(np.sum(np.abs(c[:, j]))) for j in range(ORD)])
    # closed forms
    p1 = math.exp(-0.5 * (k * h) ** 2); p2 = math.exp(-0.5 * (2 * k * h) ** 2)
    g1 = alpha * m_eps * k * k * p1; g2 = alpha * m_eps * 4 * k * k * p2
    b_cf = (alpha * m_eps * k * k * p1 * (1 - m_eps * p1)
            * math.exp(-2 * alpha * k * k * T)
            * (math.exp(g2 * T) - math.exp(2 * g1 * T)) / (g2 - 2 * g1))
    d_cf = 1 - math.exp(-alpha * k * k * T * (1 - m_eps * p1))
    return {"ck": ck, "c2k": c2k, "Aj": Aj, "d_cf": d_cf, "b_cf": b_cf}


def budget(res, a):
    ck, c2k = res["ck"], res["c2k"]
    d = res["d_cf"]; b = res["b_cf"]
    r1 = ck[3]; r2 = c2k[4]
    ORD = len(ck)
    # low-order gap
    G_low = (a**2 * b + a**4 * r2) - (a * d + a**3 * abs(r1))
    # residual tails from computed coefficients (orders 5.. and 6..)
    G1 = sum(abs(ck[j]) * a**j for j in range(5, ORD))
    G2 = sum(abs(c2k[j]) * a**j for j in range(6, ORD))
    # geometric remainder beyond ORD, from last two nonzero ratios (diagnostic)
    def geo_tail(coeff, jstart):
        js = [j for j in range(jstart, ORD) if abs(coeff[j]) > 0]
        if len(js) < 2:
            return 0.0
        jl, jp = js[-1], js[-2]
        ratio = (abs(coeff[jl]) / abs(coeff[jp])) ** (1.0 / (jl - jp)) * a
        if ratio >= 1:
            return float("inf")
        last = abs(coeff[jl]) * a**jl
        return last * ratio**2 / (1 - ratio)  # next order onward
    G1 += geo_tail(ck, 5); G2 += geo_tail(c2k, 6)
    return {"a": a, "G_low": G_low, "G1tail": G1, "G2tail": G2,
            "gap": G_low - G1 - G2, "r1": r1, "r2": r2}


# --- anchor certification with dt-refinement -------------------------------
KH = 3 * math.pi * 0.028; LAM = 0.01 * 1.0 * (3 * math.pi) ** 2; EPS = 1e-8
base = jet_hierarchy(KH, LAM, EPS, N=16, ORD=15, steps=4000)
fine = jet_hierarchy(KH, LAM, EPS, N=16, ORD=15, steps=8000)
print("=== coefficient cross-check (base vs closed form) ===")
print(f"d: jet {base['ck'][1]:.6e} (=a-coeff, should be 1-d={1-base['d_cf']:.6e})")
print(f"d_cf = {base['d_cf']:.8e},  b_cf = {base['b_cf']:.8e}")
print(f"r1 = {base['ck'][3]:.8e},  r2 = {base['c2k'][4]:.8e}")
print("\n=== low-mode residual coefficients c_k^(j), c_2k^(j) at T ===")
for j in range(5, 15):
    if abs(base['ck'][j]) > 1e-14 or abs(base['c2k'][j]) > 1e-14:
        print(f"  j={j:2d}: c_k^(j)={base['ck'][j]:+.4e}  c_2k^(j)={base['c2k'][j]:+.4e}"
              f"   dt-refine rel: ck {abs(base['ck'][j]-fine['ck'][j])/(abs(fine['ck'][j])+1e-30):.1e}")
print("\n=== all-orders majorant diagnostic A_j and ratios ===")
for j in range(1, 15):
    rr = base['Aj'][j] / base['Aj'][j-1] if j > 0 and base['Aj'][j-1] > 0 else 0
    print(f"  A_{j:2d} = {base['Aj'][j]:.4e}   ratio {rr:.4f}")
print("\n=== BUDGET at anchor ===")
out = {"anchor": {"kh": KH, "lam": LAM, "eps": EPS,
                  "d": base['d_cf'], "b": base['b_cf'],
                  "r1": base['ck'][3], "r2": base['c2k'][4]}, "budget": {}}
for a in (0.5, 0.55, 0.58, 0.60, 0.62):
    bd = budget(base, a)
    bdf = budget(fine, a)
    print(f"  a={a}: G_low={bd['G_low']:.5e}  G1tail={bd['G1tail']:.3e}  "
          f"G2tail={bd['G2tail']:.3e}  GAP={bd['gap']:+.5e}  "
          f"(dt-refine gap {bdf['gap']:+.5e})")
    out["budget"][str(a)] = {k: (v if math.isfinite(v) else None) for k, v in bd.items()}
(HERE / "theorem_certification.json").write_text(json.dumps(out, indent=1) + "\n")
