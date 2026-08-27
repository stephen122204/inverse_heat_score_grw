"""
reselect_discrepancy.py — re-do the Morozov parameter selection from the already
computed sweeps, adding the standard safety factor tau, and report both tau=1.0
(literal expected-noise) and tau=1.2 (conventional safety factor).

Manuscript labels (draft): `tab:discrepancy`, `fig:discrepancy`
(`sec:discrepancy`).

Why: with delta set EXACTLY to the expected noise norm (tau=1), the Tikhonov
forward residual is flat at the noise floor across lambda in [1e-8,1e-3], sitting
right at delta. Per-seed noise fluctuations then push "smallest lambda with
r>=delta" to lambda=1e-8 (no regularization) for some seeds -> blow-up. The
conventional Morozov safety factor tau>1 moves the target above the flat floor
and restores the well-behaved Tikhonov selection that Section 7 expects. The
particle selection (argmin|r - tau*delta| on the non-monotone U-shaped residual)
is unchanged in character by tau.

Heavy particle sweeps are reused from discrepancy_bw_curve.csv (deterministic);
only the cheap Tikhonov sweep is recomputed. The script validates that the
particle oracle/disc and Tikhonov oracle reproduce the original raw.csv.
"""

from __future__ import annotations

import argparse
import sys, math, warnings
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd
import copy

from invheat_grw.config import load_config, Config
from invheat_grw.fields import make_grid, true_u0, observed_final
from invheat_grw.metrics import forward_heat_solve_dct
from invheat_grw.methods import run_density_particle_estimated_score_deterministic
from invheat_grw.baselines import tikhonov_inverse

N_GRID = 400
N_PARTICLES = 10000
EPSILON = 1e-8
SEEDS = list(range(25))
ETAS = [0.001, 0.005, 0.01]
TIK_LAMBDAS = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
TAUS = [1.0, 1.2]


def patch(cfg: Config, **ov) -> Config:
    cfg = copy.deepcopy(cfg)
    for k, v in ov.items():
        o = cfg; p = k.split(".")
        for q in p[:-1]:
            o = getattr(o, q)
        setattr(o, p[-1], v)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source-dir",
        required=True,
        help="explicit discrepancy sweep directory containing discrepancy_bw_curve.csv",
    )
    args = ap.parse_args()
    src = Path(args.source_dir)
    if not src.is_absolute():
        src = REPO / src
    src = src.resolve()
    outputs = (REPO / "outputs").resolve()
    try:
        src.relative_to(outputs)
    except ValueError as exc:
        raise ValueError(f"--source-dir must be inside {outputs}: {src}") from exc
    for required in ("discrepancy_bw_curve.csv", "discrepancy_raw.csv"):
        if not (src / required).is_file():
            raise FileNotFoundError(f"Incomplete discrepancy sweep: missing {src / required}")
    bw = pd.read_csv(src / "discrepancy_bw_curve.csv")
    raw_orig = pd.read_csv(src / "discrepancy_raw.csv")

    base = load_config(str(REPO / "configs" / "gaussian_base.yaml"))
    cfg = patch(base, **{"heat.T": 0.15, "initial_condition.sigma0": 0.08,
                         "domain.n_grid": N_GRID})
    x = make_grid(cfg); u0 = true_u0(x, cfg); g_clean = observed_final(x, cfg)
    dx = float(x[1] - x[0])
    u0n = float(np.sqrt(dx * np.sum(u0 ** 2)))
    gcn = float(np.sqrt(dx * np.sum(g_clean ** 2)))
    gmax = float(g_clean.max())

    def E2(c): return float(np.sqrt(dx * np.sum((c - u0) ** 2)) / u0n)

    def resid(c, gn): return float(np.sqrt(dx * np.sum((forward_heat_solve_dct(c, x, cfg) - gn) ** 2))
                                   / np.sqrt(dx * np.sum(gn ** 2)))

    def g_noisy_of(eta, seed):
        return g_clean + eta * gmax * np.random.default_rng(seed).standard_normal(g_clean.shape)

    def tik(gn, lam):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return tikhonov_inverse(gn, x, cfg.heat.alpha, cfg.heat.T, lam).candidate

    def particle(gn, f):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return run_density_particle_estimated_score_deterministic(
                gn, x, cfg, N_PARTICLES, recon_method="kde", bandwidth_factor=float(f),
                epsilon=EPSILON, score_method="smoothed_log", smooth_sigma_factor=1.0,
                save_snapshots=False).candidate

    def delta_nom(eta): return eta * gmax * np.sqrt(dx * N_GRID) / gcn

    # particle selection from stored (factor->E2,residual)
    def particle_select(d_by_f, target):
        facs = sorted(d_by_f)
        rs = [d_by_f[f][1] for f in facs]
        monotone = all(rs[i + 1] >= rs[i] - 1e-9 for i in range(len(rs) - 1))
        if monotone:
            ge = [f for f in facs if d_by_f[f][1] >= target]
            if ge and any(d_by_f[f][1] < target for f in facs):
                f_sel = min(ge)
            elif ge:
                f_sel = min(facs)
            else:
                f_sel = max(facs)
            note = ""
        else:
            f_sel = min(facs, key=lambda f: abs(d_by_f[f][1] - target))
            note = "nonmonotonic"
        return f_sel, note

    # tikhonov selection: need per-lambda residual; recompute residual & E2
    def tik_sweep(gn):
        out = {}
        for lam in TIK_LAMBDAS:
            c = tik(gn, lam)
            out[lam] = (E2(c), resid(c, gn))
        return out

    def tik_select(gn, tsweep, target):
        lams = TIK_LAMBDAS
        ge = [l for l in lams if tsweep[l][1] >= target]
        if ge and any(tsweep[l][1] < target for l in lams):
            l_hi = min(ge); l_lo = max(l for l in lams if l < l_hi and tsweep[l][1] < target)
            r_lo, r_hi = tsweep[l_lo][1], tsweep[l_hi][1]
            t = (target - r_lo) / (r_hi - r_lo) if r_hi > r_lo else 1.0
            lam_sel = math.exp(math.log(l_lo) + t * (math.log(l_hi) - math.log(l_lo)))
            c = tik(gn, lam_sel)
            return lam_sel, E2(c), resid(c, gn), c
        elif ge:   # all residuals >= target -> least regularization
            lam_sel = min(lams); c = tik(gn, lam_sel)
            return lam_sel, E2(c), resid(c, gn), c
        else:      # all < target -> most regularization
            lam_sel = max(lams); c = tik(gn, lam_sel)
            return lam_sel, E2(c), resid(c, gn), c

    rows = []
    tik_cache = {}   # (eta,seed)->sweep
    for eta in ETAS:
        dnom = delta_nom(eta)
        for seed in SEEDS:
            sub = bw[(bw.eta == eta) & (bw.seed == seed)]
            d_by_f = {float(r.factor): (float(r.E2), float(r.residual)) for r in sub.itertuples()}
            f_oracle = min(d_by_f, key=lambda f: d_by_f[f][0])
            E2_p_oracle = d_by_f[f_oracle][0]

            gn = g_noisy_of(eta, seed)
            tsw = tik_sweep(gn); tik_cache[(eta, seed)] = tsw
            lam_oracle = min(TIK_LAMBDAS, key=lambda l: tsw[l][0])
            E2_t_oracle = tsw[lam_oracle][0]

            row = {"eta": eta, "seed": seed, "delta_nominal": dnom,
                   "particle_oracle_factor": f_oracle, "E2_particle_oracle": E2_p_oracle,
                   "tik_oracle_lambda": lam_oracle, "E2_tik_oracle": E2_t_oracle}
            for tau in TAUS:
                target = tau * dnom
                f_sel, note = particle_select(d_by_f, target)
                row[f"particle_disc_factor_tau{tau}"] = f_sel
                row[f"E2_particle_disc_tau{tau}"] = d_by_f[f_sel][0]
                row[f"particle_note_tau{tau}"] = note
                lam_sel, e2t, rt, _ = tik_select(gn, tsw, target)
                row[f"tik_disc_lambda_tau{tau}"] = lam_sel
                row[f"E2_tik_disc_tau{tau}"] = e2t
            rows.append(row)
    raw = pd.DataFrame(rows)

    # ---- validation vs original run ----
    val = raw.merge(raw_orig[["eta", "seed", "E2_particle_oracle", "E2_tik_oracle",
                              "particle_disc_factor", "E2_particle_disc"]],
                    on=["eta", "seed"], suffixes=("", "_orig"))
    chk_po = float((val.E2_particle_oracle - val.E2_particle_oracle_orig).abs().max())
    chk_to = float((val.E2_tik_oracle - val.E2_tik_oracle_orig).abs().max())
    chk_pf = float((val["particle_disc_factor_tau1.0"] - val.particle_disc_factor).abs().max())
    chk_pe = float((val["E2_particle_disc_tau1.0"] - val.E2_particle_disc).abs().max())
    print(f"VALIDATION vs original run (should be ~0):")
    print(f"  particle oracle E2 max|d|={chk_po:.2e}  tik oracle E2 max|d|={chk_to:.2e}")
    print(f"  particle disc factor (tau1) max|d|={chk_pf:.2e}  particle disc E2 (tau1) max|d|={chk_pe:.2e}")

    # ---- aggregate ----
    def agg_series(vals):
        v = np.asarray(vals, float)
        return (float(np.mean(v)), float(np.std(v, ddof=1)),
                float(np.percentile(v, 10)), float(np.percentile(v, 90)))

    arows = []
    for eta in ETAS:
        s = raw[raw.eta == eta]
        d = {"eta": eta, "delta_nominal": float(delta_nom(eta))}
        for tag, col in [("particle_oracle", "E2_particle_oracle"),
                         ("tik_oracle", "E2_tik_oracle")]:
            m, sd, p10, p90 = agg_series(s[col]); d.update({
                f"{tag}_mean": m, f"{tag}_std": sd, f"{tag}_p10": p10, f"{tag}_p90": p90})
        for tau in TAUS:
            for tag in ["particle_disc", "tik_disc"]:
                m, sd, p10, p90 = agg_series(s[f"E2_{tag}_tau{tau}"]); d.update({
                    f"{tag}_tau{tau}_mean": m, f"{tag}_tau{tau}_std": sd,
                    f"{tag}_tau{tau}_p10": p10, f"{tag}_tau{tau}_p90": p90})
            d[f"particle_disc_factor_tau{tau}_mean"] = float(s[f"particle_disc_factor_tau{tau}"].mean())
            d[f"tik_disc_lambda_tau{tau}_gmean"] = float(np.exp(np.mean(np.log(s[f"tik_disc_lambda_tau{tau}"]))))
        d["particle_oracle_factor_mean"] = float(s.particle_oracle_factor.mean())
        d["tik_oracle_lambda_gmean"] = float(np.exp(np.mean(np.log(s.tik_oracle_lambda))))
        arows.append(d)
    agg = pd.DataFrame(arows)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPO / "outputs" / f"discrepancy_principle_final_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out / "discrepancy_raw.csv", index=False)
    agg.to_csv(out / "discrepancy_summary.csv", index=False)

    # ---- figure data (tau=1.2, eta=0.005) ----
    TAU_FIG = 1.2; eta_fig = 0.005
    sf = raw[raw.eta == eta_fig]
    mean_disc = float(sf[f"E2_particle_disc_tau{TAU_FIG}"].mean())
    rep_seed = int(sf.iloc[(sf[f"E2_particle_disc_tau{TAU_FIG}"] - mean_disc).abs().argmin()].seed)
    gn = g_noisy_of(eta_fig, rep_seed)
    # rep particle disc reconstruction
    f_disc_rep = float(sf[sf.seed == rep_seed][f"particle_disc_factor_tau{TAU_FIG}"].iloc[0])
    c_part = particle(gn, f_disc_rep)
    lam_disc_rep, e2_tik_rep, _, c_tik = tik_select(gn, tik_cache[(eta_fig, rep_seed)], TAU_FIG * delta_nom(eta_fig))
    e2_part_rep = E2(c_part)
    bwcurve = bw[bw.eta == eta_fig].groupby("factor").E2.mean().sort_index()
    f_min_mean = float(bwcurve.idxmin())
    f_disc_mean = float(sf[f"particle_disc_factor_tau{TAU_FIG}"].mean())
    np.savez(out / "discrepancy_arrays.npz",
             x=x, u0=u0, rep_seed=rep_seed, eta_fig=eta_fig, tau_fig=TAU_FIG,
             rep_part_disc=c_part, rep_tik_disc=c_tik,
             rep_E2_part=e2_part_rep, rep_E2_tik=e2_tik_rep,
             bw_factors=bwcurve.index.to_numpy(), bw_E2_mean=bwcurve.to_numpy(),
             f_disc_mean=f_disc_mean, f_min_mean=f_min_mean)

    # ---- report ----
    L = []
    L.append("DISCREPANCY-PRINCIPLE EXPERIMENT — final selection (with safety factor)")
    L.append("Test B, N=10000, smoothed-log; alpha=0.01 n_grid=400 dt=0.001 [0,1] Neumann eps=1e-8")
    L.append("noise: g+eta*max(g)*N(0,1), rng=default_rng(seed), seeds 0..24")
    L.append(f"delta_nominal(eta) = eta*max(g_clean)*sqrt(dx*n_grid)/||g_clean||_2 (same per seed, no u0)")
    L.append(f"  max(g_clean)={gmax:.5f} ||g_clean||={gcn:.5f}  delta_nominal = "
             + ", ".join(f"{delta_nom(e):.4f}@eta={e}" for e in ETAS))
    L.append("Morozov target = tau * delta_nominal. tau=1.0 is the literal expected-noise level;")
    L.append("tau=1.2 is the conventional safety factor (target above the flat residual floor).")
    L.append("u0 used ONLY for E2 scoring; never in delta or selection.")
    L.append(f"VALIDATION vs first run: particle-oracle max|dE2|={chk_po:.1e}, "
             f"tik-oracle max|dE2|={chk_to:.1e}, particle-disc(tau1) max|dE2|={chk_pe:.1e}")
    L.append("")
    L.append("Headline E2 (mean +/- std, 25 seeds). 'disc' = discrepancy-tuned (realistic).")
    L.append(f"{'eta':>6} | {'part oracle bw':>16} {'part disc bw':>16} | "
             f"{'Tik oracle l':>14} {'Tik disc l':>14}   (tau)")
    for eta in ETAS:
        r = agg[agg.eta == eta].iloc[0]
        for tau in TAUS:
            L.append(f"{eta:>6.3f} | {r['particle_oracle_mean']:.4f}+/-{r['particle_oracle_std']:.4f}  "
                     f"{r[f'particle_disc_tau{tau}_mean']:.4f}+/-{r[f'particle_disc_tau{tau}_std']:.4f} | "
                     f"{r['tik_oracle_mean']:.4f}+/-{r['tik_oracle_std']:.4f}  "
                     f"{r[f'tik_disc_tau{tau}_mean']:.4f}+/-{r[f'tik_disc_tau{tau}_std']:.4f}   (tau={tau})")
        L.append("")
    L.append("Chosen parameters (mean over seeds):")
    L.append(f"{'eta':>6} | part oracle bw / disc bw (tau1.0, tau1.2) | "
             f"Tik oracle l / disc l (tau1.0, tau1.2)")
    for eta in ETAS:
        r = agg[agg.eta == eta].iloc[0]
        L.append(f"{eta:>6.3f} | {r['particle_oracle_factor_mean']:.2f} / "
                 f"{r['particle_disc_factor_tau1.0_mean']:.2f}, {r['particle_disc_factor_tau1.2_mean']:.2f} | "
                 f"{r['tik_oracle_lambda_gmean']:.2e} / "
                 f"{r['tik_disc_lambda_tau1.0_gmean']:.2e}, {r['tik_disc_lambda_tau1.2_gmean']:.2e}")
    L.append("")
    L.append("RECONCILIATION: oracle-lambda Tikhonov 25-seed means must be 0.004/0.016/0.021:")
    for eta in ETAS:
        r = agg[agg.eta == eta].iloc[0]
        L.append(f"  eta={eta}: oracle-l Tikhonov mean E2 = {r['tik_oracle_mean']:.4f}")
    L.append("")
    L.append(f"Figure (tau={TAU_FIG}, eta={eta_fig}): rep seed {rep_seed} "
             f"(disc-particle E2={e2_part_rep:.4f} ~ mean {mean_disc:.4f}); "
             f"mean disc bw={f_disc_mean:.2f} vs min-error bw={f_min_mean:.0f}")
    report = "\n".join(L)
    (out / "discrepancy_report.txt").write_text(report)
    print("\n" + report)
    print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()
