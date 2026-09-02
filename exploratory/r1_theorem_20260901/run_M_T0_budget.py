"""R1: explicit second-derivative bound M and interval T0 (working-note
constants, anchored at g, paying constants only on the delta-ball).

Ball: ||U - g||_{A_sigma} <= delta on [0, T_ball], sigma in [1.5, 1.8],
delta = 0.1. Sup-norm conversion constants are DISCRETE suprema over
integer modes. U_tautau assembly:
  Phi = U_x^2/U;  U_tau = -alpha(Phi - <Phi>);
  U_xtau = -alpha(2 U_x U_xx/U - U_x^3/U^2);
  Phi_tau = -alpha[4U_x^2U_xx/U^2 - 2U_x^4/U^3] + alpha U_x^2(Phi-<Phi>)/U^2;
  ||U_tautau||_inf <= 2 alpha sup|Phi_tau|
                   <= 8 alpha^2 [X1^2 X2/mu^2 + X1^4/mu^3].
c_rep closed form: alpha*pi^2*sqrt(R(c-R)), R = sqrt(c^2-B^2)
(two independent derivations agree). Writes M_T0_budget.json.
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
alpha, a, T = 0.01, 2.0, 1.0
b = math.exp(-alpha * math.pi ** 2 * T)
R = math.sqrt(a * a - b * b)
crep = alpha * math.pi ** 2 * math.sqrt(R * (a - R))
delta, s_min, s0 = 0.1, 1.5, 1.8

K1 = math.pi * max(n * s_min ** (-n) for n in range(1, 60))
K2 = math.pi ** 2 * max(n * n * s_min ** (-n) for n in range(1, 60))
X1 = b * math.pi + K1 * delta        # ||U_x||_inf on the ball
X2 = b * math.pi ** 2 + K2 * delta   # ||U_xx||_inf on the ball
mu = (a - b) - delta                 # inf U on the ball

M_wrong = 8 * alpha ** 2 * (X1 ** 2 * X2 / mu ** 2 + X1 ** 4 / mu ** 3)
M_true = alpha ** 2 * math.pi ** 4 * b * math.exp(alpha * math.pi ** 2 * 0.35) / math.sqrt(2)
M = M_wrong + M_true

gx_norm = b * math.pi * s_min                       # ||g_x||_{A_1.5}
dx_loss = math.pi * max(n * (s_min / s0) ** n for n in range(1, 80))
Ux_norm = gx_norm + dx_loss * delta                 # ||U_x||_{A_1.5}
Uma = b * s_min + delta                             # ||U - a||_{A_1.5}
inv_norm = (1 / a) / (1 - Uma / a)                  # ||1/U||_{A_1.5}
Phi_norm = Ux_norm ** 2 * inv_norm
F_norm = 2 * alpha * Phi_norm
T_ball = delta / F_norm
T0 = min(T_ball, crep / M)

out = {"crep_closed_form": crep, "delta": delta, "sigma": [s_min, s0],
       "K1": K1, "K2": K2, "X1": X1, "X2": X2, "mu": mu,
       "M_wrong": M_wrong, "M_true_flow": M_true, "M": M,
       "F_wrong_ball_norm": F_norm, "T_ball": T_ball,
       "crep_over_M": crep / M, "T0": T0,
       "corollary": f"||w(tau)|| >= {crep/2:.6f}*tau on [0, {T0:.4f}]"}
(HERE / "M_T0_budget.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
