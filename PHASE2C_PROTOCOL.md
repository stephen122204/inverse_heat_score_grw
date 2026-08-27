# Phase 2C science-campaign protocol

**Status: PROPOSED — amended after independent review; no campaign runs are authorized until this line reads FROZEN in its own commit.**

**Baseline:** nested-repository tag `sinum-phase2b-v1`, with the verified cell-centered dossier pinned by `manifests/phase2b_verification.json`. The root tag of the same name includes only the inert proposed-protocol commit beyond the exact nested Phase 2B boundary; the nested tag is the authoritative code baseline.

This protocol pre-registers the numerical science campaign that will replace the archived paper tables. Phase 2B established that the corrected operators, grid convention, quadrature, score kernel, and variable-coefficient solver behave as designed. Phase 2C asks the scientific questions on that verified foundation. Its purpose is not to recover the archived numbers.

The campaign follows four global rules.

1. Parameter grids, primary outcomes, acceptance criteria, failure accounting, and selection rules are fixed before the first scientific run.
2. Exploratory calculations are saved and labeled exploratory. They cannot silently replace a failed, boundary-censored, or unfavorable preregistered result.
3. Every attempted row and seed remains in the output, including failures. No failed run is dropped from a denominator or summary.
4. A code defect or protocol change requires a committed amendment identifying every affected study; those studies are then rerun in full. A favorable subset may not be retained.

## 1. Test cases and roles

### Primary bounded-domain cases

The primary constant-coefficient tests are exact Neumann cosine solutions on `[0,1]`:

- **C1 (single mode):** `u0(x) = 1 + 0.5 cos(3 pi x)`, `alpha = 0.01`, `T = 1`.
- **C2 (two modes):** `u0(x) = 1 + 0.4 cos(pi x) + 0.2 cos(2 pi x)`, `alpha = 0.01`, `T = 1`.

Their exact forward solutions, terminal data, scores, inverse solutions, and modal damping factors are analytic. Their initial minima are `0.5` and `0.7`, respectively, so positivity is not a near-zero artifact.

### Secondary archived-shape cases

Tests B, H, and Z are retained as secondary cases for continuity with the archived paper. They are free-space Gaussian constructions restricted to `[0,1]`, not exact Neumann solutions:

- **B:** `u0 = exp(-(x-0.4)^2/(2*0.08^2))`, `T = 0.15`.
- **H:** `u0 = 0.05 + 0.75 exp(-(x-0.35)^2/(2*0.05^2)) + 0.45 exp(-(x-0.62)^2/(2*0.08^2))`, `T = 0.15`.
- **Z:** `u0 = exp(-(x-0.4)^2/(2*0.05^2))`, `T = 0.05`.

All use `alpha = 0.01` and unit Gaussian amplitudes where an amplitude is not displayed. These formulas, rather than mutable default values, define the cases. Every table and caption must label them **approximate-Neumann secondary tests**. For each case, archive:

- the analytic wall fluxes and their ratios to an interior flux scale;
- the initial and terminal mass in `[0,1]`;
- the fraction of mass in the fixed wall strips `[0,0.05]` and `[0.95,1]` at initial and terminal time; and
- the corrected-grid discrepancy from the exact Neumann forward solve.

Here wall flux means `J = -alpha u_x`, evaluated analytically at both walls at initial and terminal time; its disclosed ratio is `max(|J(0)|,|J(1)|)/max_x |J(x)|` at the same time. The Neumann discrepancy is the relative cell-centered `L2` difference at terminal time between the free-space formula and the corrected Neumann solve initialized from the same restricted field. These diagnostics determine how much evidentiary weight the secondary cases can carry; they are not used to redefine the cases after seeing results.

### Variable-coefficient cases

Use

`a(x) = 0.01 (1 + beta sin(2 pi x))`, with `beta in {0.5, 0.9}`,

at `T = 0.15`, with the B Gaussian and H mixture initial conditions just defined. Thus the four cases are `(B, beta=0.5)`, `(B, beta=0.9)`, `(H, beta=0.5)`, and `(H, beta=0.9)`. The `beta = 0.9` cases are the principal stress tests. All coefficient values used by the finite-volume operator are analytic face values.

The headline scientific conclusions must be supported by C1/C2 or by the de-crimed variable-coefficient cases. B/H/Z provide historical continuity and stress tests only.

### Preregistered study matrix

- **Clean physical-bandwidth curves:** C1, C2, B, H, Z, and all four de-crimed variable cases.
- **Epsilon sensitivity at `h = 0.014`:** C1 and Z. C1 is the exact bounded-domain primary; Z is the near-zero-tail stress case.
- **Paired 25-seed noise campaign:** C1 and B. This supplies one exact bounded-domain primary and a direct bridge to the archived noise study. No other case is silently included in the headline noise aggregate.
- **Continuous-lambda oracle curves:** every clean case in the bandwidth study.
- **Noisy-lambda selections:** for every nonzero-noise `(case, eta, seed, arm)` block, three preregistered noisy-lambda selections: **oracle continuous lambda** as the attainable-performance diagnostic (truth access; never presented as data-driven), **residual-selected at `tau = 1.2`** (headline), and **residual-selected at `tau = 1.0`** (labeled sensitivity).
- **Gradient-carrier closures:** the G1, G2, and GB cases defined in Section 8.
- **Legacy transition table:** C1 only.

An unlisted case may be run only as separately manifested exploratory work and cannot enter a headline aggregate.

## 2. Fixed discretization and particle sample size

Unless a named refinement study varies one of them, use:

- cell-centered grid with physical walls `[0,1]`;
- `M = 400`;
- `dt = 1e-3`;
- `N = 4000` density particles; and
- the exact time-step contract from Phase 2A.

The change from the initially proposed `N = 10000` is deliberate and made before any Phase 2C run. The pinned Phase 2B dossier found the fixed-bandwidth error flat from `N = 500` through `N = 8000`, with relative spread about `7e-6`; carrying `N = 10000` through every bandwidth and noise replicate would add large score-evaluation cost without resolving a demonstrated error source.

Particle adequacy is nevertheless checked with the canonical self-consistent estimated score in two preregistered refinements — **C1**, the exact bounded-domain primary, and **H**, the mixture-saddle stress case that has historically been the score-estimation bottleneck — each at fixed `M`, `dt`, `h = 0.014`, and `epsilon_rel = 1e-8`, using

`N in {1000, 2000, 4000, 8000, 10000}`.

Certifying the default only on the easiest case and applying it to the saddle-limited cases would leave the choice open to exactly the retrospective-rerun risk the global rules forbid. Writing the principal relative `L2` errors of each case as `E_N`, acceptance requires `|E_4000-E_10000|/E_10000 <= 0.005` in **both** cases; the final reconstruction-mass difference between the two runs of each case must also be at most `1e-12` relative to `M0`. If either case fails, the campaign stops for a protocol amendment; the default `N` is not selected retrospectively from the same tables.

A performance-only preflight may time a small number of steps after the protocol is frozen. It may motivate an algebraically equivalent optimization, but it may not change any scientific parameter. Any optimization must pass the Phase 2B suite and an exact or explicitly tolerance-bounded output comparison before campaign runs begin.

The `N` contract applies to the density-particle method. The signed gradient-carrier diagnostic has its own nonredundant resolution rule in Section 8.

## 3. Physical bandwidth study

The bandwidth is a physical length, never a multiple of `dx`. Use the preregistered grid

`h in {0.005, 0.007, 0.010, 0.014, 0.020, 0.028, 0.040}`.

The non-rounded values are the displayed protocol values and therefore the values to execute; do not replace them by exact powers of `sqrt(2)`.

For every clean case in the preregistered bandwidth matrix, archive the full curve of the prespecified reconstruction error, residual, mass error, minimum reconstructed density, and failure status. The principal error metric is the relative cell-centered `L2` error. Report `Linf`, reconstructed total variation, and truth total variation as secondary diagnostics.

Two bandwidth labels are permitted:

- **Oracle-grid bandwidth:** the minimizer of the full preregistered grid against known truth, used only to quantify attainable performance in a synthetic study.
- **Data-selected bandwidth:** selected from the normalized forward residual by the rule in Section 6, without truth access.

The oracle-grid label must never be shortened to “optimal bandwidth.” If a minimum or residual match occurs at an endpoint, mark it as **boundary-censored**. The base grid is not silently extended. A separately manifested exploratory extension may diagnose the censoring, but it cannot replace the preregistered headline result.

For noiseless performance summaries, use the preregistered reference `h = 0.014` and show the complete bandwidth curve. A zero-noise residual target is degenerate, so no “data-selected” bandwidth is reported at `eta = 0`.

## 4. Score floor and positivity arms

Define

`epsilon_abs = epsilon_rel * (M0 / L)`,

where `M0` is the initial mass and `L = x_right - x_left`. Use

`epsilon_rel in {1e-10, 1e-8, 1e-6, 1e-4}`

for the C1/Z sensitivity study named in Section 1. The preregistered headline value is `epsilon_rel = 1e-8`; the sweep is a robustness check and must not be used to choose a different headline value from truth error. `epsilon = 0` is diagnostic-only and may fail loudly.

The implemented score is the additive regularization

`s_K = (partial_x rho_K) / (rho_K + epsilon_abs)`,

not a pointwise `max(rho_K, epsilon_abs)` clamp. A nonpositive denominator is a failed run.

For positive particle weights with total mass `M0`, define the rigorous omitted-tail envelopes

`B0(K) = (2 M0/L) sum_{k>K} exp(-0.5 (k pi h/L)^2)`

and

`B1(K) = (2 pi M0/L^2) sum_{k>K} k exp(-0.5 (k pi h/L)^2)`.

They bound the absolute density and density-derivative truncation errors, respectively. Evaluate the sums to a certified geometric/integral tail bound rather than truncating them at an arbitrary second cutoff.

For every score run record `K`, the first omitted multiplier, `B0`, `B1`, `epsilon_abs`, minimum raw kernel density, minimum regularized denominator, maximum absolute score, and failure status. Require `epsilon_abs >= 100 B0`. A dimensionless multiplier tolerance alone is not a density bound and does not satisfy this gate. In addition, every completed particle row retains its states immediately before reverse steps `0`, `floor(n_steps/2)`, and `n_steps-1`. On each state, compare the score produced by the production `K` with one whose `B0` and `B1` are each at least 100 times smaller. Require relative score `L2` change at most `1e-8` wherever the tightened score norm is nonzero and absolute change at most `1e-10/L` otherwise. Failure blocks that row rather than permitting a post hoc tolerance choice.

For noisy data, define the input projection

`P+(d)_j = max(d_j, 0)`.

Run two matched arms:

- **Arm R (raw/archived-comparable):** both methods receive the raw noisy datum `d_R = d`. The particle initializer performs its documented internal nonnegative clipping; Tikhonov remains the linear unconstrained reconstruction from `d_R`.
- **Arm P (common projected input):** both methods receive `d_P = P+(d)`. Tikhonov is the same linear method applied to `d_P`; its output is not projected. The particle method receives the same `d_P`.

This design separates unequal preprocessing from a common-data comparison without falsely calling an output-clipped solution “Tikhonov.” Record the projected mass change, negative-entry fraction, and negative mass removed in every replicate.

## 5. Continuous Tikhonov parameter selection

Use one shared implementation for the constant- and variable-coefficient studies. Let `z = log10(lambda)` on `[-12, -1]`.

### Oracle continuous lambda

1. Evaluate the truth error on a fixed 65-point uniform grid in `z`, including both endpoints.
2. Refine inside the two neighboring grid intervals around the best grid point with bounded scalar minimization at `xatol = 1e-6` in `z`.
3. Re-evaluate the endpoints and retain the best of all candidates.
4. Record the coarse-grid winner, refined value, objective value, evaluation count, and boundary-censor flag.

This scan-then-refine construction does not assume that a single blind bounded search found the global minimum. Its output is labeled **oracle continuous lambda** and is not a data-driven result.

### Residual-selected lambda

For the linear Tikhonov family, solve the monotone residual equation on `[-12,-1]` by a bracketed root method. If the target lies outside the attainable residual range, select the nearest endpoint and mark the result boundary-censored; do not call it a Morozov solution.

All tables retire “best on the tested grid.” They report the selection label and boundary status explicitly.

## 6. Noise model, residual normalization, and paired summaries

Use the paired additive-noise model

`d = g_clean + eta * max(g_clean) * xi`, with `xi_j ~ N(0,1)`,

for

`eta in {0, 0.001, 0.005, 0.01}`.

Use one noiseless run at `eta = 0` and seeds `0,...,24` for each nonzero level. The same standard-normal vector is used for every method, positivity arm, and parameter candidate within a `(case, eta, seed)` block.

For arm `A in {R,P}`, define the normalized residual

`r_A(c) = ||F c - d_A||_L2 / ||d_A||_L2`

and the nominal relative noise target

`delta_nom,A = eta * max(g_clean) * sqrt(L) / ||d_A||_L2`.

The numerator is the known synthetic noise RMS scale before projection; the denominator matches the datum used in that arm. Also record

`delta_real,A = ||d_A - g_clean||_L2 / ||d_A||_L2`

as a simulation diagnostic only. Because it uses clean truth, `delta_real,A` is never used for selection.

Use `tau = 1.2` for headline residual selection and `tau = 1.0` as a labeled sensitivity analysis.

- For Tikhonov, select the bracketed monotone solution of `r_A(lambda) = tau delta_nom,A` when it exists. Only this interior monotone-branch solution is labeled **Morozov**.
- For the particle bandwidth, whose residual need not be monotone, select the grid point minimizing `|r_A(h) - tau delta_nom,A|`. Label it **residual-matched bandwidth**, never Morozov. Ties go to the larger `h`. Record the full residual curve, selected index, absolute mismatch, and boundary-censor flag.

The monotonicity of the Tikhonov residual is checked numerically with a relative tolerance of `1e-10`; any violation is a failed row, not a switch to the particle rule.

For each method and arm report mean, standard deviation, median, 10th/90th percentiles, failure fraction, and boundary-censor fraction. Because seeds are paired, also report the per-seed difference in the principal `L2` error, its mean, a two-sided 95% paired Student-`t` confidence interval, median difference, and win fraction. Confidence intervals quantify Monte Carlo uncertainty; they are not used as a binary scientific-significance gate.

## 7. De-crimed variable-coefficient data

Production variable-coefficient terminal data must be generated independently of the inverse grid:

1. Form fine-grid initial cell averages from analytic integrals where available, otherwise from fixed high-order cell quadrature. Do not sample the initial field only at fine cell centers.
2. Solve with the conservative forward operator at `4x` resolution: `M = 1600`, `dt = 2.5e-4`.
3. Independently solve at `8x`: `M = 3200`, `dt = 1.25e-4`.
4. Conservatively average both terminal solutions onto the common `M = 400` inverse grid before comparing them.
5. Require the relative coarse-grid `L2` difference between the `4x` and `8x` projected data to be at most `1e-6`, and record the mass discrepancy. If the gate fails, increase truth resolution by a committed protocol amendment before inversion.
6. Use the projected `4x` result as clean data. Add noise only after projection to the inverse grid.

Truth-error calculations compare the inverse result with analytic or high-order coarse-cell averages of the initial condition, not point samples. A same-grid/same-operator inversion is retained only as a labeled inverse-crime demonstration.

## 8. Gradient-carrier closure experiment

Write the gradient carrier as `q = U_x` and reconstruct `U` from `q` with the physical walls held fixed. Test two closures:

- **Frozen-left closure:** the reconstruction constant is the observed terminal first-cell reconstruction anchor `g(x_c1)` at reverse time zero and is held fixed in time.
- **Mass closure:** choose the additive constant at every time so that `dx * sum(U) = M0` on the cell-centered grid.

The discrete formulas, including the location of the reconstruction anchor relative to the first cell center, must be written in the campaign dossier before results are interpreted. Both closures use identical transport, score, reflection, time stepping, and carrier realization; only the additive-constant rule changes.

The preregistered closure cases are:

- **G1 (analytic obstruction):** `u0 = 2 + cos(pi x)`, `alpha = 0.01`, `T = 1`;
- **G2 (multimode):** the C2 case from Section 1; and
- **GB (historical secondary):** Test B, labeled approximate-Neumann.

G1 and G2 support the mathematical conclusion. GB only shows how the obstruction appears in the archived test family.

### Gradient-carrier score and refinement

Gradient globs carry signed jumps, so the positive-weight density-particle `neumann_kde_score` is not a valid API for this experiment. Let `K_h` denote the same bounded Neumann heat-kernel smoothing operator and define the closure-aware score

`S_h,eps[U] = partial_x(K_h U) / (K_h U + epsilon_abs)`.

Evaluate `K_h U` and its derivative through the cell-centered cosine representation, then interpolate the score to carrier positions. The production closure comparison uses `h = 0.014`, `epsilon_rel = 1e-8`, no score clipping, and deterministic carrier motion `dX/dtau = alpha S_h,eps[U](X)`.

Use one carrier per initial grid jump. Splitting a jump into coincident equal-weight subcarriers is algebraically redundant under deterministic motion; verify once that `1` and `20` subcarriers per jump produce the same trajectory and reconstruction to `1e-13` relative, then use one. This split-invariance test replaces particle-count theater in the closure study.

At fixed physical `h` and `epsilon_rel`, run the coupled refinement

`(M, dt) in {(200, 2e-3), (400, 1e-3), (800, 5e-4)}`.

The final `q` and `U` discrepancies from the matching regularized reference defined below must decrease on both refinements; the last reduction factor must be at least `1.5`. Otherwise the particle/discretization term is reported as unresolved and no convergence-to-reference claim is made.

Run both closures against:

1. the true heat-equation carrier `q_true = u_x`; and
2. the same-closure regularized reference solving

   `q_t = -partial_x(alpha q S_h,eps[U])`, with `U_x = q`; and
3. the same-closure unregularized wrong-limit reference solving

   `q_t = -alpha * partial_x(q^2 / U)`, with `U_x = q`,

   using the corresponding closure.

The regularized reference is the continuum equation matched to the finite-bandwidth score actually used by the carriers. The unregularized equation is the carrier-consistency obstruction. Comparing the carrier directly only with the latter would confound score regularization with representation error.

Reverse time begins at the terminal datum, so both references start from the analytic terminal carrier `q(0,x) = g_x(x)`, with `U(0,x) = g(x)`. Starting either reference from `u0_x` is forbidden. At every Runge--Kutta stage, reconstruct `U` from `q` with the arm's closure. Discretize each conservation law by a cell-centered finite-volume method with piecewise-linear monotonized-central reconstruction, local Rusanov flux, zero numerical flux at both physical walls, and SSPRK3 time stepping. Use speed `|2 alpha q/U|` for the unregularized flux. For the regularized flux, use and record a global bound no smaller than `max_x alpha (|S_h,eps[U]| + 2 |q|/(K_h U + epsilon_abs))`. Record the maximum CFL number.

Compute each reference at `M = 1600`, `dt = 2.5e-4` and independently at `M = 3200`, `dt = 1.25e-4`. Conservatively project both solutions to the `M = 400` comparison grid. The relative differences of both `q` and reconstructed `U` must be at most `1e-4` at every archived comparison time; otherwise no closure conclusion is reported. The reference operator must pass constant-state, conservation, boundary-flux, and one-step refinement tests before use.

To connect the regularized reference to the theorem, repeat only the Eulerian reference comparison at `h in {0.020, 0.014, 0.010, 0.007}` with `epsilon_rel = 1e-8`. Archive its distance from the unregularized reference. The discrepancy at `h = 0.007` must be smaller than at `h = 0.014`; otherwise describe finite-bandwidth regularization without claiming observed approach to the wrong limit.

Require `U > 0` at every reference and particle step. Record the minimum `U`; loss of positivity is a failed run and is not repaired by clipping.

### Exact error decomposition

Scalar error norms are not additive. Let `U_0^c` be the unregularized wrong-limit solution, `U_h,eps^c` its closure-matched regularized counterpart, and `U_P^c` the transported gradient-carrier reconstruction. Use the mass-closure unregularized solution as the common model reference. For closure `c`, archive the exact field identity

`U_P^c - u = (U_0^mass - u) + (U_0^c - U_0^mass) + (U_h,eps^c - U_0^c) + (U_P^c - U_h,eps^c)`.

The four fields are, respectively:

1. wrong-transport/model error;
2. additive-closure error; and
3. finite-bandwidth/epsilon score-regularization error; and
4. carrier/discretization error.

All fields in this identity are evaluated on the common `M = 400` grid: fine reference fields are conservatively projected first, while analytic truth is evaluated under the declared cell-centered convention. Archive the identity at every recorded reverse time and use the final reverse time for the headline table. Archive the analogous identity for `q`. Report the norm of the total error and each component, plus all six pairwise `L2` inner products, so the squared-norm reconciliation can be checked exactly. Do not state or imply that the four component norms sum to the total norm.

### Analytic anchor

For the counterexample

`u0(x) = c + a cos(pi x)`, with `c > |a|`,

the frozen-left reconstruction at reverse time `T` satisfies

`U_frozen - u0 = -a (1 - exp(-alpha pi^2 T))`.

Thus the absolute offset magnitude is `|a| (1 - exp(-alpha pi^2 T))`. Verify both the signed error and its magnitude by reconstructing the analytic true heat carrier with the frozen closure; do not substitute the wrong-limit trajectory into this check. Report the magnitude and its relative `L2` contribution under the paper's normalization. This is the analytic anchor for the closure obstruction, not a fitted numerical claim.

## 9. Estimator convention and transition table

Every production Phase 2C **density-particle performance** table uses the canonical bounded-domain `neumann_kde_score` with:

- exact folded reflection at the physical walls;
- self-interaction included;
- no leave-one-out correction;
- one Neumann heat-kernel smoothing with the recorded physical `h`; and
- the score-floor diagnostics from Section 4.

Production score clipping is disabled. A nonfinite score or `max |s| > 1e6` fails loudly with its step recorded; it is not zeroed or clipped. The `1e6` ceiling is the preexisting safety-abort threshold, not a regularization parameter or a value that can be tuned from outcomes.

The signed gradient-carrier structural diagnostic is the sole exception and uses the explicitly defined `S_h,eps` construction in Section 8. It must never be pooled with the density-particle performance rows.

Legacy estimators are confined to one C1 transition table. That table reports, in sequence, endpoint/free-space legacy, cell-centered/free-space legacy, and cell-centered/Neumann canonical results at explicitly stated nominal and effective smoothing. Because the transition changes boundary treatment, grid convention, reflection, and—in the smoothed-log path—the number of smoothing operations, it is a **bundled historical transition**, not a pure one-factor estimator ablation.

No primary novelty or accuracy claim may rely on the legacy rows.

## 10. Acceptance gates and row accounting

Each study declares its expected case/parameter/seed Cartesian product before execution. The summary must state expected rows, attempted rows, completed rows, failed rows, and censored rows, and verify that attempted equals expected. A failed numerical state remains a row with its seed, parameters, failure step, and message.

At minimum, the campaign gate checks:

- all Phase 2A tests and the pinned Phase 2B dossier remain green;
- the `N = 4000` adequacy criterion in Section 2 passes;
- the score truncation/floor gate in Section 4 passes for every completed particle run;
- all requested bandwidth and lambda endpoints carry censor flags;
- the Tikhonov monotonicity check passes for every row labeled Morozov;
- the `4x`/`8x` variable-data gate in Section 7 passes;
- the split-invariance and carrier-refinement gates in Section 8 pass;
- both closure-reference refinement gates in Section 8 pass;
- the `h = 0.007` regularized-to-unregularized discrepancy is smaller than the `h = 0.014` discrepancy;
- all exact decomposition identities reconcile to `1e-10` relative or better; and
- every manifest hash validates before a study is called complete.

Failure of a gate blocks the associated scientific conclusion. It does not authorize a parameter change inside the same campaign.

## 11. Provenance, pinning, and manuscript boundary

Each study writes:

- complete row-level CSV data;
- a machine-readable summary with verdicts and expected/attempted/completed counts;
- environment and seed metadata;
- the exact parameter grid and selection labels;
- the generating commit and grid convention; and
- a hash-validated run manifest.

At Phase 2C close, one campaign manifest pins every accepted study artifact and the exact protocol commit. That pinned manifest is the sole numerical source for the rewritten manuscript. Exploratory outputs remain outside it and are never discovered by “latest directory” logic.

No manuscript accuracy, novelty, or comparison prose is finalized until the pinned campaign passes an independent claim-to-artifact audit. Prose edits made earlier remain in the delta ledger only.

## Freeze checklist

- [ ] Primary/secondary case roles and analytic formulas independently checked.
- [ ] Fixed discretization and `N = 4000` adequacy gate accepted.
- [ ] Physical bandwidth grid, endpoint censoring, and noiseless rule accepted.
- [ ] Score-floor diagnostics and fixed headline epsilon accepted.
- [ ] Input-only positivity arms and residual normalization accepted.
- [ ] Continuous-lambda scan/refinement and boundary rules accepted.
- [ ] Paired noise seeds, selection labels, and paired summaries accepted.
- [ ] De-crimed variable-data projection and convergence gate accepted.
- [ ] Closure definitions, wrong-limit reference, and exact field decomposition accepted.
- [ ] Canonical estimator/self-interaction and transition-table scope accepted.
- [ ] Row accounting, acceptance gates, provenance, and amendment rules accepted.
- [ ] Status changed from PROPOSED to FROZEN in a dedicated commit.

Scientific experiment runs begin only after all boxes are checked and the status line is FROZEN.
