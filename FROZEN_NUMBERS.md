# Frozen numbers — single source of truth (refreshed 2026-07-03, round-4 close-out)

Every headline number in the paper, traced to committed data. Fixed throughout:
`alpha=0.01`, `dt=0.001`, domain `[0,1]`, Neumann BC, `n_grid=400`, `dx=1/399`,
positivity floor `epsilon=1e-8`, density reconstruction bandwidth factor 4
(production), smoothed-log smoothing sigma = kernel bandwidth.
Environment: Python 3.11, numpy pin 2.4.4 (results verified identical on 1.26.4),
scipy 1.17.1. Rounding convention: every printed value is the direct round of the
stored value at the printed precision (round-4 audit, 163 values checked).

Determinism: the density-particle method has no RNG (deterministic quantile init,
analytic/KDE scores); independent reruns are bit-identical. Only the noise,
discrepancy, and (via noise) summary rows draw random noise
(`g + eta*max(g)*N(0,1)`, `rng=default_rng(realization)`, realizations 0..24).
Figure regeneration is byte-stable across consecutive runs (verified with
`SOURCE_DATE_EPOCH` pinned).

## 1. Representation theorem (tab:representation, fig_representation_convergence, fig_representation_failure_visual)
Gradient-glob exact-score rel_L2 across the grid×globs sweep (n_grid ∈ {100,200,400,800} × globs/jump ∈ {20,80}):
| Test | mean | CoV (population) |
|---|---|---|
| B | 0.1755 | 0.52 % |
| H | 0.2400 | 0.53 % |
| Z | 0.1549 | 0.14 % |
Also flat under dt refinement: B glob E2 = 0.1748/0.1749/0.1749/0.1750 at dt = 0.002/0.001/0.0005/0.00025 (round-3 sweep; regenerated and archived 2026-07-11 by `scripts/run_dt_sweep_glob.py` → `outputs/dt_sweep_glob_20260711_002100/dt_sweep_glob.csv`, stored glob values 0.174813/0.174866/0.174866/0.174989, density control 0.0070/0.0069/0.0069/0.0069; now reported in Sec 3.3 as "moved only between 0.1748 and 0.1750").

Canonical cell (n_grid=400, N=5000, bw 4): B 0.175 / 0.0069 (25×); H 0.239 / 0.0160 (15×); Z 0.155 / 0.0175 (8.9×).
Density exact-score falls with grid: B 0.097 (n=100) → 0.007 (n=400); N-invariant.
Source: `outputs/paper_run_20260627/representation_audit_20260627_094153/`.

## 2. Bandwidth study (tab:bandwidth, fig_bandwidth_sweep) — at epsilon=1e-8
| Test | best estimator | bw | best E2 | exact-score E2 | ratio |
|---|---|---|---|---|---|
| B | smoothed-log | 4 | 0.0122 | 0.0069 | 1.8× |
| H | grid-ratio | 4 | 0.0657 | 0.0160 | 4.1× |
| Z | smoothed-log | 2 | 0.0103 | 0.0175 | 0.59× |
Test B curve: 1.11 (bw 1) → 0.0122 (bw 4) → 0.134 (bw 16). (0.134 is the direct
round of 0.134448; an earlier record said 0.135 via double rounding.)
Cross-check at bw 4 (direct-kde vs grid-ratio): B 0.0257/0.0231; H 0.0650/0.0657; Z 0.0229/0.0230.
Source: `outputs/paper_run_20260627/score_estimation_audit_20260627_084924/`.

## 3. Particle count (Test B, bw 4, clean)
smoothed-log 0.0122 (N=5000) → 0.0120 (N=10000); grid-ratio 0.0231 → 0.0133.
Source: `outputs/paper_run_20260627/validation_stage_20260627_082621/`.

## 4. Noise study — 25 realizations (tab:noise, fig_noise_robustness_bands)
mean ± std, eta = 0 / 0.001 / 0.005 / 0.01:
- smoothed-log bw4: 0.012 / 0.031±0.006 / **0.123±0.026** / 0.222±0.047
- smoothed-log bw6: 0.026 / 0.026±0.000 / 0.027±0.001 / 0.032±0.002
- grid-ratio bw4: 0.013 / 0.463±0.060 / 0.525±0.065 / 0.552±0.076
- Tikhonov (optimal λ): 0.001 / 0.004±0.001 / 0.016±0.002 / **0.021±0.002**
(bold cells: earlier records printed 0.124±0.027 and ±0.003 by double rounding;
stored values 0.123474±0.026488 and 0.002493 round directly to the values above.)
Reconciliation anchors: realizations {0,1,2} reproduce the archived 3-realization
means exactly; the full study re-executed from scratch 2026-07-03 is bit-identical.
Source: `outputs/noise_study_25seeds_20260629_153744/` (re-run: `noise_study_25seeds_20260703_031324/`).

## 5. Data-driven selection, Morozov tau=1.2 (tab:discrepancy, fig_discrepancy_comparison)
mean ± std over 25 realizations; delta_nominal = 0.0024 / 0.0121 / 0.0242:
| eta | particle optimal bw | particle discrepancy bw | Tik optimal λ | Tik discrepancy λ |
|---|---|---|---|---|
| 0.001 | 0.025±0.002 | 0.031±0.006 | 0.004±0.001 | 0.004±0.001 |
| 0.005 | 0.027±0.001 | 0.123±0.026 | 0.016±0.002 | 0.013±0.003 |
| 0.010 | 0.032±0.002 | 0.111±0.111 | 0.021±0.002 | 0.021±0.002 |
tau=1.0 destabilizes only the Tikhonov selection (residual floor); particle
discrepancy bw ≈ 4 vs error-optimal ≈ 6 (systematic under-smoothing).
Representative reconstruction (eta=0.005): particle 0.122, Tikhonov 0.012.
Reconciliation anchor: optimal-λ Tikhonov 25-realization means = 0.0039/0.0160/0.0209.
Source: `outputs/discrepancy_principle_final_20260629_165900/` (eta=0.005 row
re-executed from scratch 2026-07-03, identical: `discrepancy_principle_final_20260703_032345/`).

## 6. Non-smooth cases, T=0.05, N=10000 (tab:nonsmooth, fig_nonsmooth_reconstruction)
tent: exact 0.025, smoothed-log best (bw6) 0.053, Tikhonov 0.009, E_fwd(sl bw6) 0.023.
top-hat: exact 0.125, smoothed-log best (bw2) 0.143, Tikhonov 0.109.
Snapshot solver note (v5.0 audit): the nonsmooth exact-score snapshots come from the
spectral cosine-transform (DCT) propagator in run_nonsmooth_case.py, NOT Crank-Nicolson
(CN is the variable-coefficient study only); the manuscript now says "cosine-transform
forward snapshots". The `numerical_oracle_score` smoothed log-derivative routine is
shared by both studies.
Source: `outputs/nonsmooth_case_20260629_152746/` (re-run bit-identical 2026-07-03).

## 7. Variable coefficient (tab:variable, fig_variable_coefficient_*)
| Case | exact | sl bw4 | sl bw6 | fd bw4 | Tikhonov |
|---|---|---|---|---|---|
| VB β=0.5 | 0.0083 | 0.0127 | 0.0270 | 0.0568 | <1e-3 |
| VB β=0.9 | 0.0084 | 0.0145 | 0.0277 | 0.1469 | <1e-3 |
| VH β=0.5 | 0.0201 | 0.0794 | 0.0870 | 0.0917 | 0.0054 |
| VH β=0.9 | 0.0204 | 0.0877 | 0.0897 | 0.2508 | 0.0055 |
Velocity v = a(x)∂ₓlog u (no a′ term). Forward: Crank-Nicolson, dt_fwd = 1e-3 (= reverse dt).
Column-label note (round 6): the "fd bw4" column here is the variable-coefficient
script's `fd_grid_ratio` branch, which routes to `log_density_fd_score` (central
differences of log(u+eps), i.e. the smoothed-log construction WITHOUT smoothing) —
not the constant-coefficient gradient(u)/(u+eps) grid-ratio path. The manuscript
therefore labels this column "ul factor 4" (unsmoothed log-gradient). Values unchanged.
Exact-score column: from CN snapshots via `numerical_oracle_score` (Gaussian
pre-smoothing sigma = 3*dx, then central differences of the log); same routine backs
the tab:nonsmooth exact-score column at beta=0 — now stated in the manuscript.
VH-mixture sweep (β=0.5, bw 2–6): 0.734 / 0.106 / 0.079 / 0.078 / 0.087; exact floor 0.0201.
Source: `outputs/paper_run_20260627/variable_coefficient_audit_20260627_082333/`,
`vh_mixture_bandwidth_refinement_20260627_084940/`.

## 8. Other paper-level facts
- E_fwd (smoothed-log bw4, relative) across all cases: 0.008–0.021; always < E2.
- tab:summary Tikhonov clean values: B 0.001 (0.001033), H 0.003 (0.003284), Z 0.001 (0.000750).
- tab:summary Z exact-score cell: stored 0.017466 → direct round **0.017**
  (manuscript currently prints 0.018 via double rounding — round-4 finding, pending triage).
- Universal claim audit (round 4): Tikhonov smaller than every coexisting particle
  value in all 32 comparisons; closest margins: top-hat 1.31×, noise eta=0.01 1.53×.
- Terminal-gradient sign changes (log-q obstruction): B 1, H 3, Z 1.
- Figure set: exactly 12 shipped figures (see `paper_draft/main.tex` includegraphics list);
  no "oracle", no "seed" in any package file.

Regenerate figures: `python make_figures.py --only naive convergence loop bandwidth
particle_count variable_field variable_results vh_mixture && python make_new_figures.py
&& python make_discrepancy_figure.py`.
