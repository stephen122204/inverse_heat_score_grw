"""
run_nonsmooth_case.py — non-smooth test cases (tent and top-hat).

Manuscript labels (draft): `tab:nonsmooth`, `fig:nonsmooth` (`sec:nonsmooth`).

Adds a non-smooth initial condition to the constant-coefficient backward-heat
study, run through the IDENTICAL production pipeline (density-particle with
smoothed-log estimator), to show the numbers are not an artifact of smoothness.

Frozen configuration (held fixed):
    alpha = 0.01, domain [0,1], Neumann BC, n_grid = 400, dt = 0.001, epsilon=1e-8.
    Forward model: constant-coefficient spectral (cosine-transform / DCT-II)
        u(.,t) = idct( dct(u0) * exp(-alpha k_n^2 t) ),  k_n = n*pi/L.
    This is exactly the forward operator the cosine-transform Tikhonov reference
    inverts, so Tikhonov is matched to the forward field (no model mismatch),
    and it is the same spectral heat propagation used to define the Gaussian
    tests' observed fields.

Test functions (positive, so the score s = d/dx log u is defined):
    tent    : u0(x) = 0.1 + 0.9 * max(0, 1 - |x-0.4|/0.15)   (kinks; primary)
    top-hat : u0(x) = 0.1 + 0.9 * 1_{[0.30,0.60]}(x)         (jump; optional)

Methods (identical pipeline, N=10000 density particles):
    exact-score  : numerical oracle score read from the true forward snapshots
                   (diagnostic lower bound) — reuse run_varcoeff_oracle, beta=0.
    smoothed-log : run_density_particle_estimated_score_deterministic, sweep
                   bandwidth factors {2,4,6}; report the best.
    Tikhonov     : cosine-transform Tikhonov, optimal lambda (best-case reference).

Reported per case: E2 = ||u0_hat - u0||_2/||u0||_2 for each method, and
forward-consistency E_fwd = ||H_T[u0_hat] - g||_2/||g||_2 for the best
smoothed-log reconstruction.

Outputs: outputs/nonsmooth_case_TIMESTAMP/
    nonsmooth_metrics.csv, nonsmooth_report.txt, nonsmooth_arrays.npz
"""

from __future__ import annotations

import sys
import warnings
import importlib.util
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd
import copy
from scipy.fft import dct, idct

from invheat_grw.config import load_config, Config
from invheat_grw.fields import make_grid
from invheat_grw.methods import run_density_particle_estimated_score_deterministic
from invheat_grw.baselines import tikhonov_inverse

# import the variable-coefficient module to reuse the numerical-oracle reconstruction
_spec = importlib.util.spec_from_file_location(
    "vc_audit", str(REPO / "scripts" / "run_variable_coefficient_audit.py"))
vc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vc)  # type: ignore[union-attr]

# --- frozen constants ---
ALPHA = 0.01
N_GRID = 400
DT = 0.001
EPSILON = 1e-8
N_PARTICLES = 10000
BW_FACTORS = [2.0, 4.0, 6.0]
TIKHONOV_LAMBDAS = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]


def patch(cfg: Config, **ov) -> Config:
    cfg = copy.deepcopy(cfg)
    for k, v in ov.items():
        obj = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], v)
    return cfg


def tent_u0(x):
    return 0.1 + 0.9 * np.maximum(0.0, 1.0 - np.abs(x - 0.4) / 0.15)


def tophat_u0(x):
    return 0.1 + 0.9 * ((x >= 0.30) & (x <= 0.60)).astype(float)


def dct_propagate(u0, x, t, alpha=ALPHA, *, length: float):
    """Spectral (cosine-transform) heat propagation to time t on Neumann [0,L].

    length is the physical domain extent from the configuration (required
    keyword; the sample span is not a valid substitute on a cell-centered
    grid)."""
    N = len(u0)
    L = float(length)
    c = dct(u0, type=2, norm="ortho")
    k_n = np.arange(N) * np.pi / L
    c_t = c * np.exp(-alpha * k_n ** 2 * t)
    return idct(c_t, type=2, norm="ortho")


def run_case(name, u0_fn, T, x, cfg):
    dx = float(x[1] - x[0])
    length = float(cfg.domain.x_max - cfg.domain.x_min)
    n_steps = round(T / DT)
    u0 = u0_fn(x)
    u0_norm = float(np.sqrt(dx * np.sum(u0 ** 2)))

    def rel_l2(c):
        return float(np.sqrt(dx * np.sum((c - u0) ** 2))) / u0_norm

    # forward field + true snapshots (for exact-score)
    g = dct_propagate(u0, x, T, length=length)
    snapshots = {k: dct_propagate(u0, x, k * DT, length=length) for k in range(n_steps + 1)}
    # forward structure diagnostic
    contrast = float((g.max() - g.min()) / (u0.max() - u0.min()))

    results = {}
    cands = {}

    # --- exact-score (numerical oracle from true snapshots), beta=0 ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r_or = vc.run_varcoeff_oracle(
            g, x, snapshots, ALPHA, 0.0, DT, n_steps,
            n_particles=N_PARTICLES, recon_method="kde", bandwidth_factor=4.0,
            x_min=float(cfg.domain.x_min), x_max=float(cfg.domain.x_max))
    cands["exact_score"] = r_or["candidate"]
    results["exact_score"] = {"E2": rel_l2(r_or["candidate"]), "bw": 4.0,
                              "completed": r_or["completed"]}

    # --- smoothed-log sweep over bandwidth factors ---
    best_bw, best_e2, best_cand = None, np.inf, None
    for bw in BW_FACTORS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            r = run_density_particle_estimated_score_deterministic(
                g, x, cfg, N_PARTICLES,
                recon_method="kde", bandwidth_factor=bw,
                epsilon=EPSILON, scale_epsilon_by_peak=False,
                score_clipping=None, save_snapshots=False,
                score_method="smoothed_log", smooth_sigma_factor=1.0)
        e2 = rel_l2(r.candidate)
        results[f"smoothed_log_bw{int(bw)}"] = {"E2": e2, "bw": bw,
                                                "completed": r.completed}
        cands[f"smoothed_log_bw{int(bw)}"] = r.candidate
        if e2 < best_e2:
            best_bw, best_e2, best_cand = bw, e2, r.candidate

    # --- Tikhonov optimal (cosine-transform) ---
    best_tik_e2, best_tik_cand, best_lam = np.inf, None, None
    for lam in TIKHONOV_LAMBDAS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            tr = tikhonov_inverse(g, x, ALPHA, T, lam, length=length)
        if np.all(np.isfinite(tr.candidate)):
            e2 = rel_l2(tr.candidate)
            if e2 < best_tik_e2:
                best_tik_e2, best_tik_cand, best_lam = e2, tr.candidate, lam
    results["tikhonov_optimal"] = {"E2": best_tik_e2, "lam": best_lam}
    cands["tikhonov_optimal"] = best_tik_cand

    # --- forward consistency of the best smoothed-log reconstruction ---
    g_norm = float(np.sqrt(dx * np.sum(g ** 2)))
    fwd_best = dct_propagate(best_cand, x, T, length=length)
    e_fwd_best = float(np.sqrt(dx * np.sum((fwd_best - g) ** 2))) / g_norm

    return {
        "name": name, "T": T, "n_steps": n_steps, "contrast": contrast,
        "best_bw": best_bw, "E_fwd_best_sl": e_fwd_best,
        "results": results, "cands": cands,
        "u0": u0, "g": g, "x": x,
    }


def main():
    base = load_config(str(REPO / "configs" / "gaussian_base.yaml"))
    cfg = patch(base, **{"heat.alpha": ALPHA, "heat.dt": DT,
                         "domain.n_grid": N_GRID})
    x = make_grid(cfg)

    T = 0.05
    cfgT = patch(cfg, **{"heat.T": T})

    cases = [("tent", tent_u0), ("tophat", tophat_u0)]
    all_rows = []
    out_arrays = {}
    case_outputs = []

    for cname, fn in cases:
        co = run_case(cname, fn, T, x, cfgT)
        case_outputs.append(co)
        print(f"\n=== {cname}  T={T}  n_steps={co['n_steps']}  "
              f"forward-contrast={co['contrast']:.3f}  best_bw={co['best_bw']} ===")
        for m, r in co["results"].items():
            print(f"  {m:<20} E2={r['E2']:.4f}"
                  + (f"  bw={r.get('bw')}" if 'bw' in r else "")
                  + (f"  lam={r.get('lam')}" if 'lam' in r else ""))
            all_rows.append({"case": cname, "T": T, "method": m,
                             "E2": r["E2"], "bw": r.get("bw"),
                             "lam": r.get("lam"),
                             "forward_contrast": co["contrast"]})
        print(f"  E_fwd (best smoothed-log, bw={co['best_bw']}): {co['E_fwd_best_sl']:.4f}")
        all_rows.append({"case": cname, "T": T, "method": "E_fwd_best_smoothed_log",
                         "E2": np.nan, "bw": co["best_bw"], "lam": None,
                         "forward_contrast": co["contrast"],
                         "E_fwd": co["E_fwd_best_sl"]})
        out_arrays[f"{cname}__x"] = co["x"]
        out_arrays[f"{cname}__u0"] = co["u0"]
        out_arrays[f"{cname}__g"] = co["g"]
        for m, c in co["cands"].items():
            if c is not None:
                out_arrays[f"{cname}__{m}"] = c

    df = pd.DataFrame(all_rows)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPO / "outputs" / f"nonsmooth_case_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "nonsmooth_metrics.csv", index=False)
    np.savez(out / "nonsmooth_arrays.npz", **out_arrays)

    lines = ["NON-SMOOTH TEST CASES",
             f"alpha={ALPHA} n_grid={N_GRID} dt={DT} domain[0,1] Neumann epsilon={EPSILON}",
             "forward model: constant-coefficient spectral (cosine-transform/DCT)",
             f"N_particles={N_PARTICLES}  bandwidth sweep {BW_FACTORS}", ""]
    for co in case_outputs:
        lines.append(f"Case {co['name']}  T={co['T']}  n_steps={co['n_steps']}  "
                     f"forward-contrast(max-min ratio vs u0)={co['contrast']:.3f}")
        for m, r in co["results"].items():
            extra = (f"bw={r.get('bw')}" if 'bw' in r else
                     (f"lam={r.get('lam')}" if 'lam' in r else ""))
            lines.append(f"    {m:<20} E2={r['E2']:.4f}   {extra}")
        lines.append(f"    best smoothed-log: bw={co['best_bw']}  "
                     f"E2={co['results']['smoothed_log_bw'+str(int(co['best_bw']))]['E2']:.4f}  "
                     f"E_fwd={co['E_fwd_best_sl']:.4f}")
        lines.append("")
    report = "\n".join(lines)
    (out / "nonsmooth_report.txt").write_text(report)
    print("\n" + report)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
