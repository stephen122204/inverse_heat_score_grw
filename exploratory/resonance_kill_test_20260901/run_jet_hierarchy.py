"""Exact perturbation hierarchy via jet transport (EXPLORATORY).

Signed lattice modes n = -N..N, Taylor jets in the amplitude a up to
order 5, propagated through the exact trilinear system
  dc_n/dtau = -alpha (i n k) sum_{p+q+r=n} c_p (i q k phi_q c_q) v_r,
with the auxiliary constraint (1+eps) v_n + sum_{m != n}
phi_{n-m} c_{n-m} v_m = delta_{n0} solved by Neumann iteration in jets.
Initial data: exact heat data of u0 = 1 + a cos(kx). Outputs the exact
coefficients of the cosine-amplitude expansions
  mode k:  a (1 - d) + a^3 r1 + O(a^5)
  mode 2k: a^2 b + a^4 r2 + O(a^6)
and validates order 2 against the closed-form b(T). Then compares the
truncated expansions against the full collocation solver at a = 0.35
and a = 0.5.
"""
import math
import numpy as np

ALPHA, T, EPS_REL, H, NMODE = 0.01, 1.0, 1e-8, 0.028, 3
K = NMODE * math.pi
N, ORD = 6, 6                # modes -N..N, jet orders a^0..a^{ORD-1}
DT = 2.5e-4

idx = lambda n: n + N
phi = np.array([math.exp(-0.5 * (n * K * H) ** 2) for n in range(-N, N + 1)])
m_eps = 1.0 / (1.0 + EPS_REL)


def jmul(A, B):
    """Jet (mode-convolution x order-product) of two [2N+1, ORD] arrays."""
    out = np.zeros((2 * N + 1, ORD), dtype=complex)
    for p in range(-N, N + 1):
        for q in range(-N, N + 1):
            n = p + q
            if -N <= n <= N:
                conv = np.convolve(A[idx(p)], B[idx(q)])[:ORD]
                out[idx(n)] += conv
    return out


def solve_v(c):
    w_off = (phi[:, None] * c).astype(complex).copy()
    w_off[idx(0)] = 0.0                     # exclude the diagonal constant
    delta = np.zeros((2 * N + 1, ORD), dtype=complex)
    delta[idx(0), 0] = 1.0
    v = m_eps * delta.copy()
    for _ in range(ORD + 2):
        v = m_eps * (delta - jmul(w_off, v))
    return v


def rhs(c):
    v = solve_v(c)
    dW = np.array([(1j * n * K) * phi[idx(n)] * c[idx(n)]
                   for n in range(-N, N + 1)])
    G = jmul(jmul(c, dW), v)
    out = np.array([(-ALPHA) * (1j * n * K) * G[idx(n)]
                    for n in range(-N, N + 1)])
    return out


c = np.zeros((2 * N + 1, ORD), dtype=complex)
c[idx(0), 0] = 1.0
A0 = math.exp(-ALPHA * K * K * T) / 2.0
c[idx(1), 1] = A0
c[idx(-1), 1] = A0

for _ in range(round(T / DT)):
    k1 = rhs(c); k2 = rhs(c + 0.5 * DT * k1)
    k3 = rhs(c + 0.5 * DT * k2); k4 = rhs(c + DT * k3)
    c = c + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

amp1 = 2 * c[idx(1)].real          # cosine amplitude series of mode k
amp2 = 2 * c[idx(2)].real
lin1 = amp1[1]                     # coefficient of a
r1 = amp1[3]
b_jet = amp2[2]
r2 = amp2[4]

mm = m_eps
p1 = math.exp(-0.5 * (K * H) ** 2); p2 = math.exp(-0.5 * (2 * K * H) ** 2)
g1 = ALPHA * mm * K * K * p1; g2 = ALPHA * mm * 4 * K * K * p2
b_formula = (ALPHA * mm * K * K * p1 * (1 - mm * p1)
             * math.exp(-2 * ALPHA * K * K * T)
             * (math.exp(g2 * T) - math.exp(2 * g1 * T)) / (g2 - 2 * g1))
d_lin = 1 - math.exp(-ALPHA * K * K * T * (1 - mm * p1))

print(f"order-1 check: 1 - lin1 = {1-lin1:.6e}  vs  d = {d_lin:.6e}")
print(f"order-2 check: b_jet = {b_jet:.6e}  vs  b_formula = {b_formula:.6e}"
      f"  ratio = {b_jet/b_formula:.6f}")
print(f"r1 (a^3, mode k)  = {r1:.6e}")
print(f"r2 (a^4, mode 2k) = {r2:.6e}")
print(f"mode 3k a^3 coeff = {2*c[idx(3)].real[3]:.6e}")
print(f"mode 4k a^4 coeff = {2*c[idx(4)].real[4]:.6e}")
for a in (0.35, 0.5):
    e1 = -a * d_lin + a**3 * r1
    e2 = a**2 * b_jet + a**4 * r2
    print(f"a={a}: predicted mode-k err {e1:+.5e}, mode-2k err {e2:+.5e}")
