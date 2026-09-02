"""R1: closed-form c_rep = ||D[g]||_{L2(0,1)} with outward-rounded enclosure.

D[g] = F_wrong(g) - F_true(g)
     = -alpha g_x^2/g + alpha <g_x^2/g> + alpha g_xx,
g = a + b cos(pi x) on [0,1] (a=2, b=e^{-alpha pi^2 T}), F_true = -alpha u_xx.
With P(x) = b^2 pi^2 sin^2(pi x)/(a + b cos(pi x)):
  D[g] = alpha [ <P> - P - b pi^2 cos(pi x) ].
Closed forms over the period average (theta = pi x, even integrand):
  <sin^2/u>      = (a - r)/b^2,                 r = sqrt(a^2-b^2)
  <sin^2/u^2>    = (a/r - 1)/b^2
  <sin^4/u^2>    = [ -1/2 + 2a<sin^2/u> - (a^2-b^2)<sin^2/u^2> ] / b^2
  <sin^2 cos/u>  = [ 1/2 - a<sin^2/u> ] / b
c_rep^2 = alpha^2 [ <P^2> - <P>^2 + 2 b pi^2 <P cos> + b^2 pi^4 / 2 ].
Validation: numerical quadrature; and ||d/dx D[g]|| vs committed q-level 0.21038.
Writes crep_enclosure.json.
"""
import json
from pathlib import Path
from fractions import Fraction

import mpmath as mp

HERE = Path(__file__).resolve().parent
mp.mp.dps = 60

alpha = mp.mpf('0.01'); T = mp.mpf(1); a = mp.mpf(2)
b = mp.e ** (-alpha * mp.pi ** 2 * T)
r = mp.sqrt(a * a - b * b)

S1 = (a - r) / b ** 2                     # <sin^2/u>
S2 = (a / r - 1) / b ** 2                 # <sin^2/u^2>
S4 = (-mp.mpf(1) / 2 + 2 * a * S1 - (a * a - b * b) * S2) / b ** 2  # <sin^4/u^2>
SC = (mp.mpf(1) / 2 - a * S1) / b         # <sin^2 cos/u>

bp2 = b ** 2 * mp.pi ** 2
P_mean = bp2 * S1
P2_mean = bp2 ** 2 * S4
Pcos_mean = bp2 * SC
crep2 = alpha ** 2 * (P2_mean - P_mean ** 2 + 2 * b * mp.pi ** 2 * Pcos_mean
                      + b ** 2 * mp.pi ** 4 / 2)
crep = mp.sqrt(crep2)

# quadrature cross-check
f = lambda x: (-bp2 * mp.sin(mp.pi * x) ** 2 / (a + b * mp.cos(mp.pi * x))
               + P_mean - b * mp.pi ** 2 * mp.cos(mp.pi * x))
crep_quad = mp.sqrt(mp.quad(lambda x: (alpha * f(x)) ** 2, [0, 1]))

# q-level check: || d/dx D[g] ||_{L2} vs committed 0.21038
Dq = lambda x: alpha * mp.diff(f, x)
crep_q = mp.sqrt(mp.quad(lambda x: Dq(x) ** 2, [0, 1]))

# mean-zero check
meanD = mp.quad(lambda x: alpha * f(x), [0, 1])

# outward-rounded rational enclosure (12 significant digits guard)
lo = Fraction(int(mp.floor(crep * 10 ** 12)), 10 ** 12)
hi = Fraction(int(mp.ceil(crep * 10 ** 12)), 10 ** 12)

out = {
    "params": {"alpha": 0.01, "T": 1.0, "a": 2.0, "b": float(b)},
    "closed_form": {"sqrt_a2_b2": float(r), "S1": float(S1), "S2": float(S2),
                    "S4": float(S4), "SC": float(SC)},
    "crep_closed_form": mp.nstr(crep, 20),
    "crep_quadrature": mp.nstr(crep_quad, 20),
    "agreement": mp.nstr(abs(crep - crep_quad), 3),
    "rational_enclosure": [str(lo), str(hi)],
    "mean_of_D": mp.nstr(meanD, 3),
    "q_level_norm_check": mp.nstr(crep_q, 10),
    "q_level_committed": 0.21038,
}
(HERE / "crep_enclosure.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
