"""Recovery-session constants: exact closed-form evaluations supporting the
Stage 2/4 theorem and failed inequality in rate_program.tex (ledger 63).
These are evaluations of displayed analytic formulas (outward-rounded in
the note by hand), plus a numerical validation of the Stage 1 parity lemma.
Writes recovery_constants.json.
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
k = 3 * math.pi
h, alpha, T, eps_rel = 0.028, 0.01, 1.0, 1e-8
m = 1.0 / (1.0 + eps_rel)
lam = alpha * T * k * k

D1 = 1.0 / (h * math.sqrt(math.e))
D2 = 2.0 / (math.e * h * h)
gamma_star = alpha * m * D2            # sup of the continuous envelope
R0 = 0.25
# AUDIT CORRECTION (ledger 64): absorption needs -sigma'/sigma >= k ||v||,
# so beta carries the factor k; exponential shrink sigma = sigma0 e^{-beta t},
# sigma0 = e^{beta T} (so sigma(T) = 1 exactly).
beta = 2 * alpha * k * D1 * R0
sigma0 = math.exp(beta * T)
CQ = 8 * alpha * D2
r0 = gamma_star / (4 * CQ * math.exp(gamma_star * T)) / (math.exp(-lam) * sigma0)
# identical to m/(32 e^{g*T} e^{-lam} sigma0):
r0_alt = m / (32 * math.exp(gamma_star * T) * math.exp(-lam) * sigma0)
miss = 0.55 / r0
M0 = m / 16.0
out = {
    "anchor": {"k": k, "h": h, "alpha": alpha, "T": T, "eps_rel": eps_rel,
               "kh": k * h, "lambda": lam},
    "constants": {"D1": D1, "D2": D2, "gamma_star": gamma_star,
                  "exp_gamma_T": math.exp(gamma_star * T),
                  "beta": beta, "sigma0": sigma0, "C_Q": CQ},
    "theorem_A": {"radius_r0": r0, "radius_check": r0_alt,
                  "denominator_lower_bound": 0.75,
                  "cauchy_M0": M0},
    "stage4_failed_inequality": {
        "required": "a <= r0", "a": 0.55,
        "miss_factor": miss,
        "vacuous_bound": "exp(gamma* T) = %.4e" % math.exp(gamma_star * T)},
}
(HERE / "recovery_constants.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out["theorem_A"], indent=1))
print("miss factor a=0.55:", f"{miss:.3e}")

# Stage 1 numerical validation (evidence only): parity zeros
import numpy as np
from scipy.signal import fftconvolve
def jet(N, ORD, steps):
    dt = T / steps
    idx = lambda n: n + N
    kappa = np.array([n * k for n in range(-N, N + 1)])
    phi = np.exp(-0.5 * (kappa * h) ** 2)
    def jmul(A, B):
        return fftconvolve(A, B)[N:3 * N + 1, :ORD]
    delta = np.zeros((2 * N + 1, ORD)); delta[idx(0), 0] = 1.0
    def solve_v(c):
        w_off = (phi[:, None] * c).copy(); w_off[idx(0)] = 0.0
        v = m * delta.copy()
        for _ in range(ORD + 3):
            v = m * (delta - jmul(w_off, v))
        return v
    DK = (1j * kappa)[:, None]
    def rhs(c):
        return (-alpha) * DK * jmul(jmul(c, DK * phi[:, None] * c), solve_v(c))
    c = np.zeros((2 * N + 1, ORD), dtype=complex)
    c[idx(0), 0] = 1.0
    A0 = math.exp(-lam) / 2.0
    c[idx(1), 1] = A0; c[idx(-1), 1] = A0
    for _ in range(steps):
        k1 = rhs(c); k2 = rhs(c + 0.5 * dt * k1)
        k3 = rhs(c + 0.5 * dt * k2); k4 = rhs(c + dt * k3)
        c = c + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return c, idx
c, idx = jet(10, 9, 1500)
viol_parity = max(abs(c[idx(n), j]) for n in range(-10, 11) for j in range(9)
                  if (n - j) % 2 != 0)
viol_support = max((abs(c[idx(n), j]) for n in range(-10, 11) for j in range(9)
                    if abs(n) > j), default=0.0)
viol_real = max(abs(c[idx(n), j].imag) for n in range(-10, 11) for j in range(9))
print(f"validation: parity violation {viol_parity:.1e}, support {viol_support:.1e}, imag {viol_real:.1e}")
