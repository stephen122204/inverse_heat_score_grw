"""R1 validation (evidence only): solve the mass-closed wrong flow
U_tau = -alpha U_x^2/U + alpha <U_x^2/U> from g, tabulate the separation
||U - u_true||_{L2} against c_rep*tau, and the empirical second-derivative
scale. Cosine-collocation, RK4. Writes wrong_flow_validation.json."""
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
alpha, a, T = 0.01, 2.0, 1.0
b = math.exp(-alpha * math.pi ** 2 * T)
R = math.sqrt(a * a - b * b)
crep = alpha * math.pi ** 2 * math.sqrt(R * (a - R))

M, NM, DT = 1024, 60, 1e-4
x = (np.arange(M) + 0.5) / M
dx = 1.0 / M
n = np.arange(1, NM + 1)
COS = np.cos(np.outer(x, n * math.pi))
SIN = np.sin(np.outer(x, n * math.pi))

def rhs(U):
    c = 2 * dx * (COS.T @ (U - U.mean()))
    Ux = -SIN @ (c * n * math.pi)
    Phi = Ux * Ux / U
    return -alpha * (Phi - Phi.mean())

U = a + b * np.cos(math.pi * x)
rows = {}
targets = [0.0125, 0.025, 0.05, 0.1, 0.2]
steps = int(round(max(targets) / DT))
for step in range(1, steps + 1):
    k1 = rhs(U); k2 = rhs(U + 0.5 * DT * k1)
    k3 = rhs(U + 0.5 * DT * k2); k4 = rhs(U + DT * k3)
    U = U + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    tau = step * DT
    for tt in targets:
        if abs(tau - tt) < DT / 2:
            ut = a + b * math.exp(alpha * math.pi ** 2 * tau) * np.cos(math.pi * x)
            e = math.sqrt(dx * float(np.sum((U - ut) ** 2)))
            rows[str(tt)] = {"e": e, "crep_tau": crep * tt,
                             "ratio": e / (crep * tt),
                             "M_emp": 2 * abs(e - crep * tt) / tt ** 2,
                             "minU": float(U.min())}
for k, v in rows.items():
    print(f"tau={k}: e={v['e']:.6e} crep*tau={v['crep_tau']:.6e} "
          f"ratio={v['ratio']:.4f} M_emp={v['M_emp']:.4f} minU={v['minU']:.4f}")
(HERE / "wrong_flow_validation.json").write_text(
    json.dumps({"crep": crep, "rows": rows}, indent=1) + "\n")
