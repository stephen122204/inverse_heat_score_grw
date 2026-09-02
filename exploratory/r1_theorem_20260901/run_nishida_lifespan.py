"""R1 repair: explicit lifespan from Nishida's abstract Cauchy-Kowalewski
theorem (Orsay notes 78.02, Ch. III Thm 2.1) applied to the AUGMENTED
system (U, q), q = U_x invariant:
  U_tau = -alpha q^2/U + alpha <q^2/U>     (order zero, no loss)
  q_tau = -alpha d/dx (q^2/U)              (one derivative)
Scale: B_rho = A_{e^rho} (weighted Wiener algebra), rho in [0, rho0);
derivative loss ||f_x||_{rho'} <= (pi/e) ||f||_rho / (rho - rho').
Nishida constants (from the proof): a = a0/2, a0 = min(1/(20.25 C),
R/(20.64 K)); existence for |t| < a (rho0 - rho) with ||u||_rho <= R/2.
Optimizes over (sigma0, R, rho) subject to the positivity/Neumann-series
constraint B sigma0 + R < c. Writes nishida_lifespan.json.
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
alpha, c, T = 0.01, 2.0, 1.0
B = math.exp(-alpha * math.pi ** 2 * T)
R_ = math.sqrt(c * c - B * B)
crep = alpha * math.pi ** 2 * math.sqrt(R_ * (c - R_))
PIE = math.pi / math.e
SUM2 = math.pi ** 2 / 6 - 1          # sum_{n>=2} n^-2 = 0.6449
A0_C = 1 / (4 * (3 / 2) ** 4)        # 1/20.25 : lambda-recursion
A0_K = 1 / (16 * SUM2 * 2)           # 1/20.64 : (2.17) with R/2

def lifespan(sigma0, R, rho):
    rho0 = math.log(sigma0)
    if not (0 < rho < rho0) or B * sigma0 + R >= c:
        return None
    Q = B * math.pi * sigma0 + R                       # sup ||q||_rho0
    Uma = B * sigma0 + R                               # ||U - c||_rho0
    V = (1 / c) / (1 - Uma / c)                        # sup ||1/U||
    L1 = 2 * Q * V + Q * Q * V * V                     # Lipschitz of q^2/U
    C = alpha * L1 * max(2 * rho0, PIE)               # Nishida (ii)
    Vg = (1 / c) / (1 - B * sigma0 / c)
    Kq = alpha * PIE * (B * math.pi * sigma0) ** 2 * Vg     # F2(g): loses 1
    KU = 2 * alpha * (B * math.pi * sigma0) ** 2 * Vg * rho0  # F1(g)*rho0
    K = max(Kq, KU)
    a0 = min(A0_C / C, A0_K * R / K)
    a = a0 / 2
    T_N = a * (rho0 - rho)
    # M on the Nishida ball (||zeta||,||eta|| <= R/2 in B_rho):
    X1 = B * math.pi + R / 2                           # ||U_x||_inf = ||q||_inf
    X2 = B * math.pi ** 2 + (math.pi / (math.e * rho)) * R / 2   # ||q_x||_inf
    mu = (c - B) - R / 2
    M = 8 * alpha ** 2 * (X1 ** 2 * X2 / mu ** 2 + X1 ** 4 / mu ** 3) \
        + alpha ** 2 * math.pi ** 4 * B * math.exp(alpha * math.pi ** 2 * T_N) / math.sqrt(2)
    return dict(sigma0=sigma0, R=R, rho=rho, rho0=rho0, Q=Q, V=V, L1=L1,
                C=C, K=K, a0=a0, a=a, T_N=T_N, X1=X1, X2=X2, mu=mu, M=M,
                crep_over_M=crep / M, T0=min(T_N, crep / M))

best = None
for s0 in [1.1 + 0.02 * i for i in range(60)]:
    for Rr in [0.05 * j for j in range(1, 16)]:
        for frac in [0.1 * k for k in range(1, 10)]:
            r = lifespan(s0, Rr, frac * math.log(s0))
            if r and (best is None or r["T0"] > best["T0"]):
                best = r
best["crep"] = crep
best["corollary"] = f"||w(tau)|| >= {crep/2:.6f} tau on [0, {best['T0']:.5f}]"
best["nishida_constants"] = {"a0_from_C": "1/(20.25 C)", "a0_from_K": "R/(20.64 K)",
                             "a": "a0/2 (prod (1-(k+2)^-2) = 1/2)"}
(HERE / "nishida_lifespan.json").write_text(json.dumps(best, indent=1) + "\n")
print(json.dumps(best, indent=1))
