"""Exact perturbation hierarchy via jet transport, WITH built-in
full-solver validation and refinement certification (EXPLORATORY).

Signed lattice modes n = -N..N, Taylor jets in the amplitude a to order
ORD-1, propagated through the exact trilinear system with the
auxiliary-variable constraint solved by Neumann iteration in jets.
Checks: order 1 = linear deficit d; order 2 = closed-form b(T);
time-step and mode-cutoff refinement of the reported coefficients;
then full collocation-solver comparison at a = 0.35 and a = 0.5.
Writes jet_hierarchy_validation.json.
"""
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ALPHA, T, EPS_REL, H, NMODE = 0.01, 1.0, 1e-8, 0.028, 3
K = NMODE * math.pi
ORD = 6


def jet_run(N, dt):
    idx = lambda n: n + N
    phi = np.array([math.exp(-0.5 * (n * K * H) ** 2)
                    for n in range(-N, N + 1)])
    m_eps = 1.0 / (1.0 + EPS_REL)

    def jmul(A, B):
        out = np.zeros((2 * N + 1, ORD), dtype=complex)
        for p in range(-N, N + 1):
            for q in range(-N, N + 1):
                n = p + q
                if -N <= n <= N:
                    out[idx(n)] += np.convolve(A[idx(p)], B[idx(q)])[:ORD]
        return out

    delta = np.zeros((2 * N + 1, ORD), dtype=complex)
    delta[idx(0), 0] = 1.0

    def solve_v(c):
        w_off = (phi[:, None] * c).astype(complex).copy()
        w_off[idx(0)] = 0.0
        v = m_eps * delta.copy()
        for _ in range(ORD + 2):
            v = m_eps * (delta - jmul(w_off, v))
        return v

    def rhs(c):
        v = solve_v(c)
        dW = np.array([(1j * n * K) * phi[idx(n)] * c[idx(n)]
                       for n in range(-N, N + 1)])
        G = jmul(jmul(c, dW), v)
        return np.array([(-ALPHA) * (1j * n * K) * G[idx(n)]
                         for n in range(-N, N + 1)])

    c = np.zeros((2 * N + 1, ORD), dtype=complex)
    c[idx(0), 0] = 1.0
    A0 = math.exp(-ALPHA * K * K * T) / 2.0
    c[idx(1), 1] = A0
    c[idx(-1), 1] = A0
    for _ in range(round(T / dt)):
        k1 = rhs(c); k2 = rhs(c + 0.5 * dt * k1)
        k3 = rhs(c + 0.5 * dt * k2); k4 = rhs(c + dt * k3)
        c = c + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    amp = lambda n: (2 * c[idx(n)].real).tolist()
    return {"lin1": amp(1)[1], "r1": amp(1)[3], "b": amp(2)[2],
            "r2": amp(2)[4], "c3": amp(3)[3], "c4": amp(4)[4]}


def full_solver_coeffs(a, M=2048, NMODES=220, dt=2.5e-4):
    x = (np.arange(M) + 0.5) / M
    dx = 1.0 / M
    Kv = np.arange(1, NMODES + 1) * math.pi
    COS = np.cos(np.outer(x, Kv)); SIN = np.sin(np.outer(x, Kv))
    phi = np.exp(-0.5 * (Kv * H) ** 2)

    def rhs(rho):
        cc = 2.0 * dx * (COS.T @ rho)
        w = rho.mean() + COS @ (cc * phi)
        wx = -SIN @ (cc * phi * Kv)
        F = rho * wx / (w + EPS_REL)
        f_sin = 2.0 * dx * (SIN.T @ F)
        return -ALPHA * (COS @ (f_sin * Kv))

    rho = 1.0 + a * math.exp(-ALPHA * K * K * T) * np.cos(K * x)
    for _ in range(round(T / dt)):
        k1 = rhs(rho); k2 = rhs(rho + 0.5 * dt * k1)
        k3 = rhs(rho + 0.5 * dt * k2); k4 = rhs(rho + dt * k3)
        rho = rho + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    err = rho - (1.0 + a * np.cos(K * x))
    ce = 2.0 * dx * (COS.T @ err)
    return {str(m): float(ce[m * NMODE - 1]) for m in (1, 2, 3, 4)}


# --- refinement certification of the jet coefficients ------------------
grids = {"base": jet_run(6, 2.5e-4), "dt_half": jet_run(6, 1.25e-4),
         "modes_plus": jet_run(8, 2.5e-4)}
certified = {k: grids["base"][k] for k in grids["base"]}
digits = {k: min(
    -math.log10(abs(grids[g][k] - grids["base"][k])
                / max(abs(grids["base"][k]), 1e-300) + 1e-300)
    for g in ("dt_half", "modes_plus")) for k in certified}

mm = 1.0 / (1.0 + EPS_REL)
p1 = math.exp(-0.5 * (K * H) ** 2); p2 = math.exp(-0.5 * (2 * K * H) ** 2)
g1 = ALPHA * mm * K * K * p1; g2 = ALPHA * mm * 4 * K * K * p2
b_formula = (ALPHA * mm * K * K * p1 * (1 - mm * p1)
             * math.exp(-2 * ALPHA * K * K * T)
             * (math.exp(g2 * T) - math.exp(2 * g1 * T)) / (g2 - 2 * g1))
d_lin = 1 - math.exp(-ALPHA * K * K * T * (1 - mm * p1))

# --- full-solver validation -------------------------------------------
validation = {}
for a in (0.35, 0.5):
    meas = full_solver_coeffs(a)
    pred = {"1": -a * d_lin + a**3 * certified["r1"],
            "2": a**2 * certified["b"] + a**4 * certified["r2"],
            "3": a**3 * certified["c3"],
            "4": a**4 * certified["c4"]}
    validation[str(a)] = {
        m: {"measured": meas[m], "predicted": pred[m],
            "rel_diff": abs(meas[m] - pred[m]) / abs(meas[m])}
        for m in ("1", "2", "3", "4")}

out = {"params": {"alpha": ALPHA, "T": T, "h": H, "n": NMODE,
                  "eps_rel": EPS_REL, "ord": ORD},
       "checks": {"one_minus_lin1_vs_d": [1 - certified["lin1"], d_lin],
                  "b_jet_vs_formula": [certified["b"], b_formula]},
       "coefficients": certified,
       "certified_digits_vs_refinement": digits,
       "full_solver_validation": validation}
(HERE / "jet_hierarchy_validation.json").write_text(
    json.dumps(out, indent=1) + "\n", encoding="utf-8")
print(json.dumps({"coefficients": certified,
                  "digits": {k: round(v, 1) for k, v in digits.items()},
                  "validation_rel_diffs": {
                      a: {m: round(v["rel_diff"], 4)
                          for m, v in validation[a].items()}
                      for a in validation}}, indent=1))
