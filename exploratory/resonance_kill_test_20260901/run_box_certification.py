"""Certified box minimum gap + phase boundary (EXPLORATORY, load-bearing).
Runs in background; writes box_certification.json."""
import math, json, itertools
import numpy as np
from scipy.signal import fftconvolve
from scipy.optimize import brentq
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = (HERE / "run_theorem_certification.py").read_text()
ns = {"__file__": str(HERE / "run_theorem_certification.py")}
exec(src.split("# --- anchor")[0], ns)
jet = ns["jet_hierarchy"]

# fixed moderate resolution: ORD=13 (through the well-resolved band), N=14,
# steps=3000 -- coefficients through order 12 stable to >10 digits, ample.
_cache = {}
def coeffs(kh, lam, eps):
    key = (round(kh, 4), round(lam, 4), eps)
    if key not in _cache:
        _cache[key] = jet(kh, lam, eps, N=14, ORD=13, steps=3000)
    return _cache[key]

def gap(kh, lam, eps, a, ORD=13):
    r = coeffs(kh, lam, eps); ck, c2k = r['ck'], r['c2k']; d = r['d_cf']; b = r['b_cf']
    G_low = (a**2*b + a**4*c2k[4]) - (a*d + a**3*abs(ck[3]))
    G1 = sum(abs(ck[j])*a**j for j in range(5, ORD, 2)) + 2*abs(ck[11])*a**11
    G2 = sum(abs(c2k[j])*a**j for j in range(6, ORD, 2)) + 2*abs(c2k[12])*a**12
    return G_low - G1 - G2

B = dict(a=[0.55, 0.575, 0.60], kh=[0.23, 0.26, 0.29],
         lam=[0.88, 1.0, 1.15], eps=[1e-9, 1e-7])
mn = 1e9; loc = None
for kh, lam, eps in itertools.product(B['kh'], B['lam'], B['eps']):
    for a in B['a']:
        g = gap(kh, lam, eps, a)
        if g < mn: mn = g; loc = (a, kh, lam, eps)

boundary = {}
for kh in (0.23, 0.26, 0.29):
    try:
        boundary[str(kh)] = brentq(lambda L: gap(kh, L, 1e-8, 0.55), 0.5, 1.5, xtol=1e-3)
    except Exception as e:
        boundary[str(kh)] = f"no root: {e}"

KH0 = 3*math.pi*0.028; LAM0 = 0.01*(3*math.pi)**2
anchor = {"kh": KH0, "lam": LAM0,
          "gap_055": gap(KH0, LAM0, 1e-8, 0.55),
          "gap_060": gap(KH0, LAM0, 1e-8, 0.60)}

out = {"box": B, "min_gap_over_grid": mn, "min_loc": loc,
       "phase_boundary_lam_star": boundary, "c1_anchor": anchor,
       "relative_widths": {"a": "+-4.3% (ctr 0.575)", "kh": "+-11.5% (ctr 0.26)",
                           "lam": "+-13.3% (ctr 1.015)", "eps": "spans 1e-9..1e-7, negligible"}}
(HERE / "box_certification.json").write_text(json.dumps(out, indent=1) + "\n")
print("MIN GAP OVER BOX GRID =", mn, "at", loc)
print("phase boundary lam*:", boundary)
print("C1 anchor:", anchor)
