"""Purified pseudospectral version of the resonance test (gate item iii).

Fully spectral: the flux F = rho * s (which vanishes at the walls) is
projected onto the sine series and differentiated exactly; no finite
differences. Convergence is checked in dt, collocation size, and mode
cutoff. EXPLORATORY.
"""
import math
import numpy as np

ALPHA, T, EPS_REL = 0.01, 1.0, 1e-8


def make_solver(M, NMODES):
    x = (np.arange(M) + 0.5) / M
    dx = 1.0 / M
    n_arr = np.arange(1, NMODES + 1)
    K = n_arr * math.pi
    COS = np.cos(np.outer(x, K))
    SIN = np.sin(np.outer(x, K))

    def rhs(rho, h, eps_abs):
        c = 2.0 * dx * (COS.T @ rho)
        phi = np.exp(-0.5 * (K * h) ** 2)
        w = rho.mean() + COS @ (c * phi)
        wx = -SIN @ (c * phi * K)
        F = rho * wx / (w + eps_abs)
        f_sin = 2.0 * dx * (SIN.T @ F)          # sine coefficients of F
        return -ALPHA * (COS @ (f_sin * K))     # exact d/dx of the series

    def solve(a, n, h, dt):
        k = n * math.pi
        rho = 1.0 + a * math.exp(-ALPHA * k * k * T) * np.cos(k * x)
        for _ in range(round(T / dt)):
            k1 = rhs(rho, h, EPS_REL)
            k2 = rhs(rho + 0.5 * dt * k1, h, EPS_REL)
            k3 = rhs(rho + 0.5 * dt * k2, h, EPS_REL)
            k4 = rhs(rho + dt * k3, h, EPS_REL)
            rho = rho + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return rho, x, dx

    return solve


def predicted_b(n, h):
    k = n * math.pi
    m = 1.0 / (1.0 + EPS_REL)
    p1 = math.exp(-0.5 * (k * h) ** 2)
    p2 = math.exp(-0.5 * (2 * k * h) ** 2)
    g1 = ALPHA * m * k * k * p1
    g2 = ALPHA * m * 4 * k * k * p2
    return (ALPHA * m * k * k * p1 * (1 - m * p1)
            * math.exp(-2 * ALPHA * k * k * T)
            * (math.exp(g2 * T) - math.exp(2 * g1 * T)) / (g2 - 2 * g1))


CASES = ((0.1, 3, 0.028), (0.5, 3, 0.028), (0.1, 6, 0.014))
GRIDS = ((2048, 220, 2.5e-4), (4096, 300, 1.25e-4))
for (M, NM, dt) in GRIDS:
    solve = make_solver(M, NM)
    print(f"M={M} modes={NM} dt={dt}:")
    for a, n, h in CASES:
        rho_T, x, dx = solve(a, n, h, dt)
        u0 = 1.0 + a * np.cos(n * math.pi * x)
        err = rho_T - u0
        c2k = 2.0 * dx * np.sum(err * np.cos(2 * n * math.pi * x))
        pred = a * a * abs(predicted_b(n, h))
        print(f"  a={a:4.2f} n={n} h={h:5.3f}: |c_2k|={abs(c2k):.4e} "
              f"pred={pred:.4e} ratio={abs(c2k)/pred:.4f}")
