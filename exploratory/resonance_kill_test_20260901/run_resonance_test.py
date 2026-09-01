"""Second-harmonic resonance kill test for the U2 uniform-ball claim.

Continuum pseudospectral solve of the regularized density flow
  rho_tau = -alpha d_x( rho d_x K_h rho / (K_h rho + eps) )
from EXACT data g = 1 + a e^{-alpha k^2 T} cos(kx), reverse horizon T.
Measures the generated mode-2k amplitude against the exact second-order
formula
  a^2 b(T), b(T) = alpha m k^2 phi (1 - m phi) e^{-2 alpha k^2 T}
                   (e^{gamma_{2k} T} - e^{2 gamma_k T})/(gamma_{2k} - 2 gamma_k),
gamma_k = alpha m k^2 phi_k, phi_k = e^{-k^2 h^2/2}. EXPLORATORY.
"""
import math
import numpy as np

ALPHA, T = 0.01, 1.0
EPS_REL = 1e-8
M = 2048                      # collocation cells
NMODES = 220                  # retained cosine modes

x = (np.arange(M) + 0.5) / M
dx = 1.0 / M
n_arr = np.arange(1, NMODES + 1)
K = n_arr * math.pi
COS = np.cos(np.outer(x, K))          # M x N
SIN = np.sin(np.outer(x, K))


def coeffs(f):
    return 2.0 * dx * (COS.T @ f)      # cosine coefficients c_n


def rhs(rho, h, eps_abs):
    c = coeffs(rho)
    phi = np.exp(-0.5 * (K * h) ** 2)
    w = rho.mean() + COS @ (c * phi)          # K_h rho
    wx = -SIN @ (c * phi * K)                 # d_x K_h rho
    F = rho * wx / (w + eps_abs)
    dF = np.empty(M)                          # central differences, F=0 at walls
    dF[1:-1] = (F[2:] - F[:-2]) / (2 * dx)
    dF[0] = (F[1] - (-F[0])) / (2 * dx)       # odd reflection (F ~ sine series)
    dF[-1] = ((-F[-1]) - F[-2]) / (2 * dx)
    return -ALPHA * dF


def solve(a, n, h, dt=2.5e-4):
    k = n * math.pi
    eps_abs = EPS_REL * 1.0
    rho = 1.0 + a * math.exp(-ALPHA * k * k * T) * np.cos(k * x)
    steps = round(T / dt)
    for _ in range(steps):
        k1 = rhs(rho, h, eps_abs)
        k2 = rhs(rho + 0.5 * dt * k1, h, eps_abs)
        k3 = rhs(rho + 0.5 * dt * k2, h, eps_abs)
        k4 = rhs(rho + dt * k3, h, eps_abs)
        rho = rho + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return rho


def predicted_b(n, h):
    k = n * math.pi
    m = 1.0 / (1.0 + EPS_REL)
    phi1 = math.exp(-0.5 * (k * h) ** 2)
    phi2 = math.exp(-0.5 * (2 * k * h) ** 2)
    g1 = ALPHA * m * k * k * phi1
    g2 = ALPHA * m * 4 * k * k * phi2
    num = math.exp(g2 * T) - math.exp(2 * g1 * T)
    return (ALPHA * m * k * k * phi1 * (1 - m * phi1)
            * math.exp(-2 * ALPHA * k * k * T) * num / (g2 - 2 * g1))


print(f"{'a':>6} {'n':>3} {'h':>6} {'meas 2k amp':>12} {'pred a^2 b':>12} "
      f"{'ratio':>7} {'mode-k err':>11}")
for a, n, h in ((0.05, 3, 0.028), (0.1, 3, 0.028), (0.2, 3, 0.028),
                (0.5, 3, 0.028), (0.1, 5, 0.020), (0.1, 6, 0.014)):
    rho_T = solve(a, n, h)
    u0 = 1.0 + a * np.cos(n * math.pi * x)
    err = rho_T - u0
    c_err = coeffs(err)
    meas2k = abs(c_err[2 * n - 1])
    pred = a * a * abs(predicted_b(n, h))
    print(f"{a:6.2f} {n:3d} {h:6.3f} {meas2k:12.4e} {pred:12.4e} "
          f"{meas2k / pred:7.3f} {abs(c_err[n - 1]):11.4e}")

# The C1 anchor case: where does the 2.5 percent error live?
a, n, h = 0.5, 3, 0.028
rho_T = solve(a, n, h)
u0 = 1.0 + a * np.cos(n * math.pi * x)
err = rho_T - u0
c_err = coeffs(err)
E2 = math.sqrt(dx * np.sum(err ** 2)) / math.sqrt(dx * np.sum(u0 ** 2))
print(f"\nC1-like continuum: E2 = {E2:.4f}")
top = np.argsort(-np.abs(c_err))[:4]
for i in top:
    print(f"  mode {i+1} (k = {i+1}pi): |c_err| = {abs(c_err[i]):.4e}")
