"""
run_discrepancy_principle.py — Realistic parameter choice via the discrepancy
principle (Morozov), for both Tikhonov's lambda and the particle bandwidth.

Manuscript labels (draft): `tab:discrepancy`, `fig:discrepancy`
(`sec:discrepancy`); the tau reselection lives in reselect_discrepancy.py.

The paper tunes both lambda and bandwidth against the known true solution
(best-case vs best-case). This experiment instead chooses BOTH parameters by
matching the reconstruction's forward residual to the noise level delta, using
ONLY the noisy data and the (statistically known) noise level — never u0. u0 is
read for SCORING ONLY (E2), never for parameter selection. That makes the
comparison realistic-vs-realistic, with oracle-tuned columns kept for contrast.

Frozen config (held fixed): Test B, alpha=0.01, [0,1], Neumann, n_grid=400,
dt=0.001, epsilon=1e-8, N=10000, smoothed-log estimator.
Noise: u_obs + eta*max(u_obs)*N(0,1), rng=default_rng(seed), seeds 0..24,
eta in {0.001,0.005,0.01}.
Sweeps: Tikhonov lambda in the frozen grid [1e-8..1e-2]; particle bandwidth
factor in {1,2,4,6,8,12,16} (discrepancy crossing refined by bisection between
the bracketing grid factors).

Forward operator H_T: forward_heat_solve_dct (the exact operator used for the
paper's forward-consistency error and inverted by tikhonov_inverse).

Noise-level estimate delta (same for every seed at a given eta, from the noise
model and the noiseless observation g_clean = u(.,T), NOT from u0):
    delta(eta) = eta * max(g_clean) * sqrt(dx * n_grid) / ||g_clean||_2
which is the expected relative L2 norm of the added perturbation
(E||noise||_2 = eta*max(g_clean)*sqrt(dx*sum E[z_i^2]) = eta*max(g_clean)*sqrt(dx*n_grid)),
in the same discrete L2 norm ||v||_2 = sqrt(dx*sum v^2) used everywhere else.
g_clean is the data-side forward field (what noise is added to), not the
reconstruction target u0.

Outputs: outputs/discrepancy_principle_TIMESTAMP/
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
from invheat_grw.metrics import forward_heat_solve_dct
from invheat_grw.methods import run_density_particle_estimated_score_deterministic
from invheat_grw.baselines import tikhonov_inverse

# --- frozen constants ---
N_PARTICLES = 10000
N_GRID = 400
EPSILON = 1e-8
SEEDS = list(range(25))
ETAS = [0.001, 0.005, 0.01]
TIKHONOV_LAMBDAS = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
BW_FACTORS = [1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0]
RECON_METHOD = "kde"
BISECT_ITERS = 6   # refine particle bandwidth crossing between bracketing grid factors


def patch(cfg: Config, **ov) -> Config:
    cfg = copy.deepcopy(cfg)
    for k, v in ov.items():
        obj = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], v)
    return cfg


def main():
    base = load_config(str(REPO / "configs" / "gaussian_base.yaml"))
    cfg = patch(base, **{"heat.T": 0.15, "initial_condition.sigma0": 0.08,
                         "domain.n_grid": N_GRID})
    x = make_grid(cfg)
    u0 = true_u0(x, cfg)               # used ONLY for scoring E2
    g_clean = observed_final(x, cfg)   # data-side forward field; noise added to this
    dx = float(x[1] - x[0])
    u0_norm = float(np.sqrt(dx * np.sum(u0 ** 2)))
    g_clean_norm = float(np.sqrt(dx * np.sum(g_clean ** 2)))
    gmax = float(np.max(g_clean))

    # sanity: DCT forward of u0 vs analytic observed_final (should be tiny for Test B)
    fwd_u0 = forward_heat_solve_dct(u0, x, cfg)
    dct_vs_analytic = float(np.sqrt(dx * np.sum((fwd_u0 - g_clean) ** 2)) / g_clean_norm)

    def E2(cand):  # SCORING ONLY
        return float(np.sqrt(dx * np.sum((cand - u0) ** 2)) / u0_norm)

    def fwd_resid(cand, g_noisy):
        gn = float(np.sqrt(dx * np.sum(g_noisy ** 2)))
        f = forward_heat_solve_dct(cand, x, cfg)
        return float(np.sqrt(dx * np.sum((f - g_noisy) ** 2)) / gn)

    def make_noisy(eta, seed):
        rng = np.random.default_rng(seed)
        return g_clean + eta * gmax * rng.standard_normal(g_clean.shape)

    def particle(g_noisy, factor):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            r = run_density_particle_estimated_score_deterministic(
                g_noisy, x, cfg, N_PARTICLES,
                recon_method=RECON_METHOD, bandwidth_factor=factor,
                epsilon=EPSILON, scale_epsilon_by_peak=False,
                score_clipping=None, save_snapshots=False,
                score_method="smoothed_log", smooth_sigma_factor=1.0)
        return r.candidate

    def tikhonov(g_noisy, lam):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return tikhonov_inverse(g_noisy, x, cfg.heat.alpha, cfg.heat.T, lam).candidate

    # delta(eta): nominal expected relative perturbation norm (same for all seeds)
    def delta_of(eta):
        return eta * gmax * np.sqrt(dx * N_GRID) / g_clean_norm

    raw_rows = []        # one per (eta, seed)
    bw_curve_rows = []   # one per (eta, seed, factor): E2 and residual
    rep_candidates = {}  # representative-seed arrays for the figure, per eta

    for eta in ETAS:
        delta = delta_of(eta)
        print(f"\n=== eta={eta}  delta={delta:.5f} ===", flush=True)
        for seed in SEEDS:
            g_noisy = make_noisy(eta, seed)

            # ---- particle sweep over grid factors: store E2 and residual ----
            cand_by_f = {}
            E2_by_f = {}
            r_by_f = {}
            for f in BW_FACTORS:
                c = particle(g_noisy, f)
                cand_by_f[f] = c
                E2_by_f[f] = E2(c)
                r_by_f[f] = fwd_resid(c, g_noisy)
                bw_curve_rows.append({"eta": eta, "seed": seed, "factor": f,
                                      "E2": E2_by_f[f], "residual": r_by_f[f]})

            # particle oracle bandwidth: min true E2 (uses u0 — labelled oracle)
            f_oracle = min(BW_FACTORS, key=lambda f: E2_by_f[f])

            # particle discrepancy bandwidth.
            # The smoothed-log forward residual is U-shaped in bandwidth (noisy
            # under-smoothing -> high residual; over-smoothing -> high residual),
            # so it is generally NOT monotone. Selection rule:
            #   - monotone residual: standard Morozov (smallest bw with r>=delta,
            #     bisection-refined between bracketing grid factors);
            #   - non-monotone residual: pick the bw minimizing |r-delta| on the
            #     grid, and flag it.
            order = BW_FACTORS  # ascending
            rs = [r_by_f[f] for f in order]
            monotone = all(rs[i + 1] >= rs[i] - 1e-9 for i in range(len(rs) - 1))
            monotonic_note = "" if monotone else "nonmonotonic"

            if monotone:
                ge = [f for f in order if r_by_f[f] >= delta]
                has_lo = any(r_by_f[f] < delta for f in order)
                if ge and has_lo:
                    f_hi = min(ge)
                    f_lo = max(f for f in order if f < f_hi and r_by_f[f] < delta)
                    a, b = f_lo, f_hi
                    f_disc, c_disc, r_disc = f_hi, cand_by_f[f_hi], r_by_f[f_hi]
                    for _ in range(BISECT_ITERS):
                        m = 0.5 * (a + b)
                        cm = particle(g_noisy, m)
                        rm = fwd_resid(cm, g_noisy)
                        if rm >= delta:
                            b, f_disc, c_disc, r_disc = m, m, cm, rm
                        else:
                            a = m
                elif ge:   # all residuals >= delta -> least-regularized (smallest bw)
                    f_disc = min(order); monotonic_note = "all_r>=delta"
                    c_disc, r_disc = cand_by_f[f_disc], r_by_f[f_disc]
                else:      # all residuals < delta -> most-regularized (largest bw)
                    f_disc = max(order); monotonic_note = "all_r<delta"
                    c_disc, r_disc = cand_by_f[f_disc], r_by_f[f_disc]
            else:
                # non-monotone: bandwidth on the grid whose residual is nearest delta
                f_disc = min(order, key=lambda f: abs(r_by_f[f] - delta))
                c_disc, r_disc = cand_by_f[f_disc], r_by_f[f_disc]
            E2_part_disc = E2(c_disc)
            E2_part_oracle = E2_by_f[f_oracle]

            # ---- Tikhonov sweep ----
            tcand = {lam: tikhonov(g_noisy, lam) for lam in TIKHONOV_LAMBDAS}
            tE2 = {lam: E2(tcand[lam]) for lam in TIKHONOV_LAMBDAS}
            tr = {lam: fwd_resid(tcand[lam], g_noisy) for lam in TIKHONOV_LAMBDAS}
            lam_oracle = min(TIKHONOV_LAMBDAS, key=lambda l: tE2[l])
            E2_tik_oracle = tE2[lam_oracle]
            # discrepancy lambda: smallest lambda with r >= delta; log-interp + resolve
            lams = TIKHONOV_LAMBDAS
            ge_l = [l for l in lams if tr[l] >= delta]
            if ge_l and any(tr[l] < delta for l in lams):
                l_hi = min(ge_l)
                lowers = [l for l in lams if l < l_hi and tr[l] < delta]
                l_lo = max(lowers)
                # log-linear interpolate lambda* where residual == delta
                import math
                r_lo, r_hi = tr[l_lo], tr[l_hi]
                if r_hi > r_lo:
                    t = (delta - r_lo) / (r_hi - r_lo)
                else:
                    t = 1.0
                log_lam = math.log(l_lo) + t * (math.log(l_hi) - math.log(l_lo))
                lam_disc = math.exp(log_lam)
                c_tik_disc = tikhonov(g_noisy, lam_disc)
            elif ge_l:
                lam_disc = min(lams); c_tik_disc = tcand[lam_disc]
            else:
                lam_disc = max(lams); c_tik_disc = tcand[lam_disc]
            E2_tik_disc = E2(c_tik_disc)
            r_tik_disc = fwd_resid(c_tik_disc, g_noisy)

            raw_rows.append({
                "eta": eta, "seed": seed, "delta": delta,
                "particle_oracle_factor": f_oracle, "E2_particle_oracle": E2_part_oracle,
                "particle_disc_factor": f_disc, "E2_particle_disc": E2_part_disc,
                "particle_disc_residual": r_disc,
                "tik_oracle_lambda": lam_oracle, "E2_tik_oracle": E2_tik_oracle,
                "tik_disc_lambda": lam_disc, "E2_tik_disc": E2_tik_disc,
                "tik_disc_residual": r_tik_disc,
                "note": monotonic_note,
                # keep candidates needed for the figure (particle disc / tik disc)
                "_c_part_disc": c_disc, "_c_tik_disc": c_tik_disc,
            })
            print(f"  seed={seed:2d}  part: disc_f={f_disc:.2f} E2={E2_part_disc:.4f} | "
                  f"oracle_f={f_oracle:.0f} E2={E2_part_oracle:.4f}   "
                  f"tik: disc_lam={lam_disc:.1e} E2={E2_tik_disc:.4f} | "
                  f"oracle E2={E2_tik_oracle:.4f}  {monotonic_note}", flush=True)

    raw = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_c_")}
                        for r in raw_rows])
    bw_curve = pd.DataFrame(bw_curve_rows)

    # --- aggregate ---
    agg_rows = []
    for eta in ETAS:
        sub = raw[raw.eta == eta]
        rowd = {"eta": eta, "delta": float(delta_of(eta))}
        for col in ["E2_particle_oracle", "E2_particle_disc",
                    "E2_tik_oracle", "E2_tik_disc"]:
            v = sub[col].to_numpy()
            rowd[col + "_mean"] = float(np.mean(v))
            rowd[col + "_std"] = float(np.std(v, ddof=1))
            rowd[col + "_p10"] = float(np.percentile(v, 10))
            rowd[col + "_p90"] = float(np.percentile(v, 90))
        rowd["particle_oracle_factor_mean"] = float(sub.particle_oracle_factor.mean())
        rowd["particle_disc_factor_mean"] = float(sub.particle_disc_factor.mean())
        rowd["tik_oracle_lambda_gmean"] = float(np.exp(np.mean(np.log(sub.tik_oracle_lambda))))
        rowd["tik_disc_lambda_gmean"] = float(np.exp(np.mean(np.log(sub.tik_disc_lambda))))
        agg_rows.append(rowd)
    agg = pd.DataFrame(agg_rows)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPO / "outputs" / f"discrepancy_principle_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out / "discrepancy_raw.csv", index=False)
    bw_curve.to_csv(out / "discrepancy_bw_curve.csv", index=False)
    agg.to_csv(out / "discrepancy_summary.csv", index=False)

    # --- figure data: representative seed at eta=0.005 (disc-particle E2 nearest mean) ---
    eta_fig = 0.005
    sub_fig = raw[raw.eta == eta_fig]
    mean_disc = float(sub_fig.E2_particle_disc.mean())
    rep_seed = int(sub_fig.iloc[(sub_fig.E2_particle_disc - mean_disc).abs().argmin()].seed)
    rep_row = next(r for r in raw_rows if r["eta"] == eta_fig and r["seed"] == rep_seed)
    # seed-mean E2 vs factor at eta_fig
    bw_fig = bw_curve[bw_curve.eta == eta_fig].groupby("factor").E2.mean().sort_index()
    f_min_mean = float(bw_fig.idxmin())
    np.savez(out / "discrepancy_arrays.npz",
             x=x, u0=u0,
             rep_seed=rep_seed,
             rep_part_disc=rep_row["_c_part_disc"],
             rep_tik_disc=rep_row["_c_tik_disc"],
             rep_E2_part=rep_row["E2_particle_disc"],
             rep_E2_tik=rep_row["E2_tik_disc"],
             bw_factors=bw_fig.index.to_numpy(),
             bw_E2_mean=bw_fig.to_numpy(),
             f_disc_mean=float(sub_fig.particle_disc_factor.mean()),
             f_min_mean=f_min_mean,
             eta_fig=eta_fig)

    # --- report ---
    lines = []
    lines.append("DISCREPANCY-PRINCIPLE EXPERIMENT (realistic parameter choice)")
    lines.append("Test B, N=10000, smoothed-log; alpha=0.01 n_grid=400 dt=0.001 [0,1] Neumann eps=1e-8")
    lines.append("noise: g + eta*max(g)*N(0,1), rng=default_rng(seed), seeds 0..24")
    lines.append(f"delta(eta) = eta*max(g_clean)*sqrt(dx*n_grid)/||g_clean||_2  (same per seed, no u0)")
    lines.append(f"  max(g_clean)={gmax:.5f}  ||g_clean||_2={g_clean_norm:.5f}  sqrt(dx*n_grid)={np.sqrt(dx*N_GRID):.5f}")
    lines.append(f"  DCT-forward(u0) vs analytic g_clean rel-L2 = {dct_vs_analytic:.2e} (negligible -> H_T consistent)")
    lines.append("u0 is used ONLY in E2(...) scoring; never in delta or parameter selection.")
    lines.append("")
    lines.append("Headline E2 (mean +/- std over 25 seeds):")
    hdr = f"{'eta':>6}{'delta':>9}  {'part(oracle bw)':>18}{'part(disc bw)':>16}{'Tik(oracle l)':>16}{'Tik(disc l)':>15}"
    lines.append(hdr)
    for eta in ETAS:
        r = agg[agg.eta == eta].iloc[0]
        def ms(c): return f"{r[c+'_mean']:.4f}+/-{r[c+'_std']:.4f}"
        lines.append(f"{eta:>6.3f}{r['delta']:>9.4f}  "
                     f"{ms('E2_particle_oracle'):>18}{ms('E2_particle_disc'):>16}"
                     f"{ms('E2_tik_oracle'):>16}{ms('E2_tik_disc'):>15}")
    lines.append("")
    lines.append("Parameter choice (mean over 25 seeds):")
    lines.append(f"{'eta':>6}  {'part oracle bw':>14}{'part disc bw':>14}  {'Tik oracle l(gmean)':>20}{'Tik disc l(gmean)':>18}")
    for eta in ETAS:
        r = agg[agg.eta == eta].iloc[0]
        lines.append(f"{eta:>6.3f}  {r['particle_oracle_factor_mean']:>14.2f}{r['particle_disc_factor_mean']:>14.2f}  "
                     f"{r['tik_oracle_lambda_gmean']:>20.2e}{r['tik_disc_lambda_gmean']:>18.2e}")
    lines.append("")
    lines.append(f"Figure representative seed at eta={eta_fig}: seed {rep_seed} "
                 f"(disc-particle E2={rep_row['E2_particle_disc']:.4f} nearest mean {mean_disc:.4f})")
    lines.append(f"Panel(b) eta={eta_fig}: mean disc bw={float(sub_fig.particle_disc_factor.mean()):.2f}  "
                 f"vs bw minimizing mean true E2={f_min_mean:.2f}")
    lines.append("")
    lines.append("RECONCILIATION: oracle-lambda Tikhonov 25-seed means must be 0.004/0.016/0.021:")
    for eta in ETAS:
        r = agg[agg.eta == eta].iloc[0]
        lines.append(f"  eta={eta}: oracle-l Tikhonov mean E2 = {r['E2_tik_oracle_mean']:.4f}")
    report = "\n".join(lines)
    (out / "discrepancy_report.txt").write_text(report)
    print("\n" + report)
    print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()
