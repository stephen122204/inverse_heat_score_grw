# Science-campaign protocol (Phase 2C)

**Status: PROPOSED — no 2C experiment runs until this document is reviewed
and marked FROZEN.**  Baseline: tag `sinum-phase2b-v1` (verified cell-centered
core; pinned dossier `manifests/phase2b_verification.json`).  Every choice
below is fixed before any result is seen; deviations require amending this
document first, in a commit of its own.

## 1. Test cases

Primary (exact Neumann manufactured solutions; analytic score available):

- **C1** single mode: u0 = 1 + 0.5 cos(3 pi x), alpha = 0.01, T = 1.0.
- **C2** two modes: u0 = 1 + 0.4 cos(pi x) + 0.2 cos(2 pi x), alpha = 0.01,
  T = 1.0.

Secondary (continuity with the archived paper; approximate-Neumann,
wall-flux ratio disclosed): Gaussian tests B (T = 0.15, sigma0 = 0.08),
H (mixture, T = 0.15), Z (T = 0.05, sigma0 = 0.05), all as configured in the
archived studies but on the cell-centered grid.

Variable-coefficient: a(x) = alpha0 (1 + beta sin(2 pi x)), alpha0 = 0.01,
beta in {0.5, 0.9}, Gaussian and mixture initial data, T = 0.15.

## 2. Fixed resolutions during parameter studies

M = 400 cells, dt = 1e-3, N = 10000 particles, on [0, 1] with walls from the
configuration.  These change only inside dedicated one-factor refinement
studies, never inside a parameter (h, epsilon, lambda, noise) study.

## 3. Physical bandwidth

- Grid: h in {0.005, 0.007, 0.010, 0.014, 0.020, 0.028, 0.040}
  (approximately factor sqrt(2); brackets the sqrt(2) x archived-effective
  hypothesis h ~ 0.014).  Bandwidths are physical lengths; the h = c dx
  coupling is forbidden by the API and by the pinned dossier's control study.
- **Oracle selection**: minimizer of the truth error over the grid, labeled
  "oracle-best (grid)".
- **Data-driven selection**: the two-branch residual rule of Section 6,
  labeled "residual-matching selection" (the label "Morozov" is reserved for
  the monotone branch only).

## 4. Epsilon and positivity

- Epsilon is relative to the mean density scale: epsilon = eps_rel * M0 / L.
  Grid: eps_rel in {1e-10, 1e-8, 1e-6, 1e-4}; production default 1e-8.
  eps_rel = 0 is a diagnostic-only setting (deep-tail scores are then
  meaningless by the pinned contract test) and never a production run.
- Every epsilon must exceed the kernel truncation floor (series tolerance
  1e-14) by at least a factor of 100.
- Noisy-data positivity, two arms, both reported:
  - **Arm R (archived-comparable)**: raw noisy datum to Tikhonov; particles
    initialize from the clipped datum (clipping is intrinsic to the mass
    representation).
  - **Arm P (matched)**: one common positivity-projected datum for both
    methods, with a projected-Tikhonov baseline.
  Per realization, record: negative fraction, clip norm relative to ||g||,
  mass change, and which arm each summary row belongs to.

## 5. Tikhonov reference (one shared helper)

- Continuous optimization of log10(lambda) on [-12, -1], bounded scalar
  minimization, xatol 1e-6 in log10(lambda); report lambda*, the achieved
  error, and a boundary-hit flag.  Identical helper for the spectral
  (constant) and matrix (variable) forms.
- Labels: "oracle-tuned (continuous)" when minimizing truth error;
  "discrepancy-selected (interpolated)" for the data-driven rule.
  "Best on the tested grid" is retired.

## 6. Noise and the selection rule

- Model: g_noisy = g + eta * max(g_clean) * N(0, 1) per grid value;
  eta in {0, 0.001, 0.005, 0.01}; seeds 0..24, paired across every method,
  arm, and parameter setting.
- delta(eta) = eta * max(g_clean) * sqrt(dx * M) / ||g_clean|| (the RMS norm
  of the perturbation; the RMS-vs-mean substitution is documented and is a
  0.06 percent effect at M = 400).  The residual and delta always use the
  same discrete norm and the same datum as the arm under test.
- Selection: safety factor tau = 1.2 headline (tau = 1.0 recorded);
  monotone residual: smallest parameter with r >= tau delta, bisection- or
  log-interpolation-refined ("Morozov branch"); nonmonotone: closest match
  |r - tau delta| on the grid, flagged "residual-matching".
- Reported statistics: mean, sample std, median, p10, p90, failure count.

## 7. De-crimed variable data

- Truth solve: 4x finer in space and time (M = 1600, dt = 2.5e-4), verified
  against one further refinement (M = 3200, dt = 1.25e-4; relative
  difference must be below 1e-6).
- Projection to the inverse grid: conservative cell averaging — coarse cell
  j receives the mean of its four fine cells (4j..4j+3).
- One same-operator run is retained, labeled "inverse-crime demonstration",
  to display the effect's size; every headline comparison uses de-crimed
  data.

## 8. Gradient-carrier closure experiments

Two closures of the glob reconstruction U from transported jumps
(U_tilde(x) = sum of weights w_i with X_i < x):

- **Frozen-left**: U(x, tau) = g(x_c1) + U_tilde(x, tau), where g(x_c1) is
  the observed terminal value at the first cell center, frozen for all tau
  (the archived code's closure).
- **Mass-preserving**: U(x, tau) = c(tau) + U_tilde(x, tau) with c(tau)
  chosen so the midpoint mass of U equals M0 = midpoint mass of g at every
  step: c(tau) = (M0 - dx * sum_j U_tilde(x_j, tau)) / L.

Each closure runs against two references: the true derivative solution
u_x(., T - tau), and the wrong-limit reference PDE
q_tau = -alpha (q^2 / U)_x solved with the SAME closure on the fine grid
(M = 1600, dt = 2.5e-4).  The reported decomposition is
total error = wrong-transport error + closure error + discretization error,
anchored by the analytic frozen-closure error a (1 - e^{-alpha pi^2 T}) on
the cosine counterexample.  Runs abort (fail-loud) if U <= 0 anywhere.

## 9. Estimated-score transport (deferred from 2B, runs here)

The self-consistent Neumann-KDE estimated-score method (score_method
"neumann_kde", single kernel, analytic derivative, explicit physical h,
epsilon per Section 4) is the canonical practical method for every 2C
performance table.  The legacy estimators (smoothed_log, fd_grid_ratio,
direct_kde free-space) appear only in one labeled comparison table on C1 to
document the transition from the archived method.

## 10. Provenance

Every 2C study self-manifests through provenance.write_manifest with the
grid convention recorded.  At 2C close, one full campaign run is pinned
under manifests/ (as `phase2b_verification.json` was) and becomes the sole
source for the rewritten manuscript's tables and figures.  No table or
figure may mix studies from different manifests.

## Freeze checklist (all boxes required before the first run)

- [ ] Reviewed: test cases and T values (Section 1)
- [ ] Reviewed: default resolutions (Section 2)
- [ ] Reviewed: bandwidth grid and labels (Section 3)
- [ ] Reviewed: epsilon scale and two-arm positivity design (Section 4)
- [ ] Reviewed: Tikhonov bounds and tolerance (Section 5)
- [ ] Reviewed: noise model, delta, tau, selection naming (Section 6)
- [ ] Reviewed: de-crime construction and verification tolerance (Section 7)
- [ ] Reviewed: closure definitions and reference solves (Section 8)
- [ ] Reviewed: canonical-estimator policy (Section 9)
- [ ] Status line above changed from PROPOSED to FROZEN in its own commit
