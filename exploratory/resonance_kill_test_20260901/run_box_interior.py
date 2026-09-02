"""Interior densification + monotonicity of the certified box (EXPLORATORY)."""
import math, json, itertools
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
src = (HERE / "run_theorem_certification.py").read_text()
ns = {"__file__": str(HERE / "run_theorem_certification.py")}
exec(src.split("# --- anchor")[0], ns); jet = ns["jet_hierarchy"]
_c = {}
def coeffs(kh, lam, eps):
    key = (round(kh, 4), round(lam, 4), eps)
    if key not in _c: _c[key] = jet(kh, lam, eps, N=14, ORD=13, steps=3000)
    return _c[key]
def gap(kh, lam, eps, a):
    r = coeffs(kh, lam, eps); ck, c2k = r['ck'], r['c2k']; d = r['d_cf']; b = r['b_cf']
    G_low = (a**2*b + a**4*c2k[4]) - (a*d + a**3*abs(ck[3]))
    G1 = sum(abs(ck[j])*a**j for j in range(5, 13, 2)) + 2*abs(ck[11])*a**11
    G2 = sum(abs(c2k[j])*a**j for j in range(6, 13, 2)) + 2*abs(c2k[12])*a**12
    return G_low - G1 - G2
# 5-point-per-axis interior grid
A = np.linspace(0.55, 0.60, 5); KH = np.linspace(0.23, 0.29, 5)
LAM = np.linspace(0.88, 1.15, 5); E = [1e-9, 1e-7]
mn = 1e9; loc = None
for kh, lam, eps in itertools.product(KH, LAM, E):
    for a in A:
        g = gap(kh, lam, eps, a)
        if g < mn: mn = g; loc = (float(a), float(kh), float(lam), eps)
# monotonicity spot-checks at the worst corner neighborhood
worst_kh, worst_lam = 0.29, 0.88
mono_a = [gap(worst_kh, worst_lam, 1e-7, a) for a in A]
mono_lam = [gap(worst_kh, l, 1e-7, 0.55) for l in LAM]
out = {"interior_min_gap": mn, "interior_min_loc": loc,
       "monotone_in_a_at_worst": mono_a, "monotone_in_lam_at_worst": mono_lam,
       "grid": "5x5x5x2 = 250 interior points"}
(HERE / "box_interior.json").write_text(json.dumps(out, indent=1) + "\n")
print("INTERIOR MIN GAP =", mn, "at", loc)
print("gap increasing in a? ", all(x < y for x, y in zip(mono_a, mono_a[1:])), mono_a)
print("gap increasing in lam?", all(x < y for x, y in zip(mono_lam, mono_lam[1:])), mono_lam)
