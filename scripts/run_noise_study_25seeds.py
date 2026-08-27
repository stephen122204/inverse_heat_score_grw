"""
run_noise_study_25seeds.py — Noise study with 25 realizations and reported spread.

Manuscript labels (draft): `tab:noise`, `fig:noise` (`sec:noise`).

Reuses the EXISTING pipeline
(run_density_particle_estimated_score_deterministic + tikhonov_inverse) with the
frozen configuration; the ONLY change vs. the original 3-seed validation noise
study is the number of realizations (3 -> 25, seeds 0..24).

Frozen configuration (held fixed):
    Test B  : Gaussian u0 = exp(-(x-0.4)^2/(2*0.08^2)),  T = 0.15
    alpha = 0.01, domain [0,1], Neumann, n_grid = 400, dt = 0.001
    N = 10000 density particles
    smoothed-log estimator at bandwidth factors 4 and 6 (h/dx convention)
    fd_grid_ratio at factor 4 (kept for reconciliation with the 3-seed table)
    Tikhonov (optimal lambda) reference, cosine-transform form
    epsilon = 1e-8
    noise levels eta in {0, 0.001, 0.005, 0.01}
    noise model: u_obs_noisy = u_obs + eta * max(u_obs) * N(0,1),
                 rng = np.random.default_rng(seed)   (identical to validation stage)

eta = 0 is deterministic -> one value, std = 0 (no noise added).

Outputs (outputs/noise_study_25seeds_TIMESTAMP/):
    noise_study_raw.csv        — one row per (method, eta, seed)
    noise_study_summary.csv    — per (method, eta): mean, std, p10, p90, n
    noise_study_arrays.npz     — raw E2 arrays keyed by method+eta (for the figure)
    noise_study_report.txt     — human-readable table
"""

from __future__ import annotations

import sys
import warnings
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd
import copy

from invheat_grw.config import load_config, Config
from invheat_grw.fields import make_grid, true_u0, observed_final
from invheat_grw.methods import run_density_particle_estimated_score_deterministic
from invheat_grw.baselines import tikhonov_inverse

# --- frozen constants -------------------------------------------------------
N_PARTICLES = 10000
N_GRID = 400
EPSILON = 1e-8
SEEDS = list(range(25))                       # 0..24
ETAS = [0.0, 0.001, 0.005, 0.01]
TIKHONOV_LAMBDAS = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
RECON_METHOD = "kde"


def patch(cfg: Config, **ov) -> Config:
    cfg = copy.deepcopy(cfg)
    for k, v in ov.items():
        obj = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], v)
    return cfg


def make_noisy_obs(u_obs: np.ndarray, eta: float, seed: int) -> np.ndarray:
    """u_obs_noisy = u_obs + eta * max(u_obs) * N(0,1) — same as validation stage."""
    rng = np.random.default_rng(seed)
    return u_obs + eta * float(np.max(u_obs)) * rng.standard_normal(u_obs.shape)


def main() -> None:
    base = load_config(str(REPO / "configs" / "gaussian_base.yaml"))
    cfg = patch(base, **{"heat.T": 0.15,
                         "initial_condition.sigma0": 0.08,
                         "domain.n_grid": N_GRID})
    x = make_grid(cfg)
    u0 = true_u0(x, cfg)
    uT = observed_final(x, cfg)
    dx = float(x[1] - x[0])
    u0_norm = float(np.sqrt(dx * np.sum(u0 ** 2)))

    def rel_l2(cand: np.ndarray) -> float:
        return float(np.sqrt(dx * np.sum((cand - u0) ** 2))) / u0_norm

    def run_sl(u_obs, bw):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            r = run_density_particle_estimated_score_deterministic(
                u_obs, x, cfg, N_PARTICLES,
                recon_method=RECON_METHOD, bandwidth_factor=bw,
                epsilon=EPSILON, scale_epsilon_by_peak=False,
                score_clipping=None, save_snapshots=False,
                score_method="smoothed_log", smooth_sigma_factor=1.0)
        return rel_l2(r.candidate)

    def run_fd(u_obs, bw):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            r = run_density_particle_estimated_score_deterministic(
                u_obs, x, cfg, N_PARTICLES,
                recon_method=RECON_METHOD, bandwidth_factor=bw,
                epsilon=EPSILON, scale_epsilon_by_peak=False,
                score_clipping=None, save_snapshots=False,
                score_method="fd_grid_ratio", smooth_sigma_factor=1.0)
        return rel_l2(r.candidate)

    def run_tik(u_obs):
        best = np.inf
        for lam in TIKHONOV_LAMBDAS:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                tr = tikhonov_inverse(u_obs, x, cfg.heat.alpha, cfg.heat.T, lam,
                                      length=float(cfg.domain.x_max - cfg.domain.x_min))
            if np.all(np.isfinite(tr.candidate)):
                r = rel_l2(tr.candidate)
                if r < best:
                    best = r
        return float(best)

    methods = {
        "smoothed_log_bw4": lambda uo: run_sl(uo, 4.0),
        "smoothed_log_bw6": lambda uo: run_sl(uo, 6.0),
        "fd_grid_ratio_bw4": lambda uo: run_fd(uo, 4.0),
        "tikhonov_optimal": run_tik,
    }

    rows = []
    print(f"Noise study: {len(SEEDS)} seeds, etas={ETAS}, N={N_PARTICLES}, n_grid={N_GRID}")
    for eta in ETAS:
        seeds = [0] if eta == 0.0 else SEEDS   # eta=0 is deterministic
        for seed in seeds:
            u_obs = uT.copy() if eta == 0.0 else make_noisy_obs(uT, eta, seed)
            for mname, fn in methods.items():
                e2 = fn(u_obs)
                rows.append({"method": mname, "eta": eta, "seed": seed, "rel_L2": e2})
            print(f"  eta={eta:.3f} seed={seed:2d}  "
                  + "  ".join(f"{m}={[r['rel_L2'] for r in rows if r['method']==m and r['eta']==eta and r['seed']==seed][0]:.4f}"
                              for m in methods), flush=True)

    raw = pd.DataFrame(rows)

    # --- aggregate ---
    agg_rows = []
    arrays = {}
    for mname in methods:
        for eta in ETAS:
            vals = raw[(raw.method == mname) & (raw.eta == eta)].rel_L2.to_numpy()
            arrays[f"{mname}__eta{eta:g}"] = vals
            agg_rows.append({
                "method": mname,
                "eta": eta,
                "n": int(vals.size),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
                "p10": float(np.percentile(vals, 10)),
                "p90": float(np.percentile(vals, 90)),
            })
    agg = pd.DataFrame(agg_rows)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPO / "outputs" / f"noise_study_25seeds_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out / "noise_study_raw.csv", index=False)
    agg.to_csv(out / "noise_study_summary.csv", index=False)
    np.savez(out / "noise_study_arrays.npz", **arrays)

    # --- report ---
    lines = []
    lines.append("NOISE STUDY — 25 realizations (seeds 0..24), Test B, N=10000")
    lines.append("alpha=0.01 n_grid=400 dt=0.001 domain[0,1] Neumann epsilon=1e-8")
    lines.append("noise: u_obs + eta*max(u_obs)*N(0,1), rng=default_rng(seed)")
    lines.append("eta=0 is deterministic (1 value, std=0).")
    lines.append("")
    hdr = f"{'method':<20}{'eta':>7}{'mean':>10}{'std':>10}{'p10':>10}{'p90':>10}{'n':>5}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for mname in methods:
        for eta in ETAS:
            r = agg[(agg.method == mname) & (agg.eta == eta)].iloc[0]
            lines.append(f"{mname:<20}{eta:>7.3f}{r['mean']:>10.4f}{r['std']:>10.4f}"
                         f"{r['p10']:>10.4f}{r['p90']:>10.4f}{int(r['n']):>5}")
        lines.append("")
    report = "\n".join(lines)
    (out / "noise_study_report.txt").write_text(report)
    print("\n" + report)
    print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()
