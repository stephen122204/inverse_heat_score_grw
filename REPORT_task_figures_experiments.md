# Task report — figures, fixes, and experiments

Branch: `experiments-noise-nonsmooth-figs`. Environment: Python 3.11, numpy 1.26.4,
scipy 1.10.1 (the frozen pin is numpy 2.4.4; all frozen headline numbers
nonetheless reproduce — see reconciliation §4). Frozen config held fixed for
every run: `alpha=0.01`, domain `[0,1]`, Neumann, `n_grid=400`, `dt=0.001`,
`dx=1/399≈0.002506`, bandwidth factor = `h/dx`, `epsilon=1e-8`.

**These numbers are for a consistency check before they go into any table or
caption. Nothing in the `.tex` was edited.**

---

## 1. Noise study (Task 3.1) — 25 realizations

- Test B, `N=10000`, smoothed-log at factors 4 and 6, fd-grid-ratio factor 4
  (kept only for reconciliation), Tikhonov optimal-λ (cosine-transform).
- Seed list: **`0..24`** (25 seeds), `rng = np.random.default_rng(seed)` per
  realization. Noise model (identical to the original validation stage):
  `u_obs_noisy = u_obs + eta * max(u_obs) * N(0,1)`.
- `eta = 0` is deterministic → single value, std = 0.
- Figure band definition (fig 4.3): **mean line with shaded ±1 std band**
  (lower edge clamped to >0 for the log axis); zero width at `eta=0`.

| method | eta | mean | std | p10 | p90 | n |
|---|---|---|---|---|---|---|
| smoothed_log_bw4 | 0.000 | 0.0120 | 0.0000 | 0.0120 | 0.0120 | 1 |
| smoothed_log_bw4 | 0.001 | 0.0306 | 0.0059 | 0.0243 | 0.0370 | 25 |
| smoothed_log_bw4 | 0.005 | 0.1235 | 0.0265 | 0.0944 | 0.1486 | 25 |
| smoothed_log_bw4 | 0.010 | 0.2221 | 0.0473 | 0.1720 | 0.2828 | 25 |
| smoothed_log_bw6 | 0.000 | 0.0256 | 0.0000 | 0.0256 | 0.0256 | 1 |
| smoothed_log_bw6 | 0.001 | 0.0256 | 0.0002 | 0.0254 | 0.0258 | 25 |
| smoothed_log_bw6 | 0.005 | 0.0273 | 0.0009 | 0.0262 | 0.0280 | 25 |
| smoothed_log_bw6 | 0.010 | 0.0319 | 0.0023 | 0.0292 | 0.0343 | 25 |
| fd_grid_ratio_bw4 | 0.000 | 0.0133 | 0.0000 | 0.0133 | 0.0133 | 1 |
| fd_grid_ratio_bw4 | 0.001 | 0.4628 | 0.0595 | 0.3882 | 0.5353 | 25 |
| fd_grid_ratio_bw4 | 0.005 | 0.5254 | 0.0650 | 0.4570 | 0.5849 | 25 |
| fd_grid_ratio_bw4 | 0.010 | 0.5524 | 0.0764 | 0.4794 | 0.6734 | 25 |
| tikhonov_optimal | 0.000 | 0.0010 | 0.0000 | 0.0010 | 0.0010 | 1 |
| tikhonov_optimal | 0.001 | 0.0039 | 0.0009 | 0.0026 | 0.0051 | 25 |
| tikhonov_optimal | 0.005 | 0.0160 | 0.0024 | 0.0129 | 0.0185 | 25 |
| tikhonov_optimal | 0.010 | 0.0209 | 0.0025 | 0.0176 | 0.0245 | 25 |

Message is intact: factor 4 best on clean data (0.012) then rises steeply
(0.222 at η=0.01); factor 6 nearly flat (0.026→0.032); Tikhonov lowest throughout;
fd-grid-ratio collapses immediately under noise.

Raw + aggregated CSVs: `outputs/noise_study_25seeds_20260629_153744/`
(`noise_study_raw.csv`, `noise_study_summary.csv`, `noise_study_arrays.npz`).
Script: `scripts/run_noise_study_25seeds.py`.

---

## 2. Non-smooth case (Task 3.2)

Forward model: **constant-coefficient spectral (cosine-transform / DCT-II)**
heat propagation, `u(·,t)=idct(dct(u0)·exp(-α k_n² t))`, `k_n=nπ/L` — i.e. the
exact operator the cosine-transform Tikhonov reference inverts (so Tikhonov is
matched to the field), and the same spectral heat propagation behind the
Gaussian tests' observed fields. Pipeline is the production density-particle
smoothed-log method; `N=10000`; exact-score = numerical score read from the true
forward snapshots (diagnostic lower bound). All of `alpha=0.01, n_grid=400,
dt=0.001` held fixed.

### Primary: tent `u0(x) = 0.1 + 0.9·max(0, 1 − |x−0.4|/0.15)`, **T = 0.05**
Forward field retains structure (max−min ratio vs `u0` = **0.837**, far from flat),
so `T=0.05` is recoverable and used as-is.

| method | E2 |
|---|---|
| exact-score (numerical, bw=4) | **0.0246** |
| smoothed-log bw=2 | 0.1052 |
| smoothed-log bw=4 | 0.0587 |
| **smoothed-log bw=6 (best)** | **0.0530** |
| Tikhonov (optimal λ=1e-8) | 0.0090 |

Forward-consistency of the best smoothed-log reconstruction (bw=6):
**`E_fwd = 0.0229`**.

### Optional harder variant: top-hat `u0(x)=0.1+0.9·1_{[0.30,0.60]}(x)`, T = 0.05
(jump discontinuity; ran since cheap — report honestly)

| method | E2 |
|---|---|
| exact-score (numerical, bw=4) | 0.1254 |
| **smoothed-log bw=2 (best)** | **0.1426** |
| smoothed-log bw=4 | 0.1605 |
| smoothed-log bw=6 | 0.1775 |
| Tikhonov (optimal λ=1e-8) | 0.1091 |

Best smoothed-log (bw=2) `E_fwd = 0.0076`.

**Honest reading:** the non-smooth tent is recoverable (best E2≈0.053, ~2× the
exact-score floor, ~6× Tikhonov) with visible rounding of the apex kink and mild
baseline ringing — it is *harder* than the smooth Gaussians (Test B best ≈0.012)
but not qualitatively different, so the method's success is not an artifact of
smoothness. The jump top-hat is genuinely stiff: every method sits at E2≈0.11–0.18,
the particle method only ~1.3× the exact-score floor. Note the optimal Tikhonov λ
sits at the lower edge of the standard sweep (1e-8) for both — expected for
noiseless spectral-forward data; the same λ grid as all frozen studies was used,
so the reported Tikhonov is a (very slightly conservative) best case.

Data: `outputs/nonsmooth_case_20260629_152746/`
(`nonsmooth_metrics.csv`, `nonsmooth_arrays.npz`). Script: `scripts/run_nonsmooth_case.py`.

---

## 3. Figure files

**New (Task 4):**
- `fig_representation_failure_visual.{pdf,png}` — 4.2, gradient-representation
  failure shown visually (Test B, exact score). Reproduces the Table-1 run:
  density E2 = **0.0069**, gradient-glob E2 = **0.1749** (≈0.175). Density curve
  overlays truth; gradient-glob curve peaks at ~0.82 vs 1.0 — visibly wrong.
- `fig_noise_robustness_bands.{pdf,png}` — 4.3, mean ±1 std bands from the
  25-seed study (added alongside the existing noise figure, not overwriting it).
- `fig_nonsmooth_reconstruction.{pdf,png}` — 4.4, tent truth vs best smoothed-log
  (bw=6, E2=0.053) vs optimal Tikhonov (E2=0.009).

**Relabeled (Task 2.1 / 4.1), regenerated from the frozen CSVs:**
- `fig_representation_convergence.{pdf,png}` — "density oracle" → "density, exact
  score"; suptitle "density oracle far below" → "exact-score density far below".
- `fig_noise_robustness.{pdf,png}` — "oracle-tuned Tikhonov" → "optimally-tuned Tikhonov".
- `fig_variable_coefficient_results.{pdf,png}` — "oracle particles" → "exact-score
  particles"; "FD Tikhonov (oracle λ)" → "FD Tikhonov (optimal λ)".
- `fig_vh_mixture_bandwidth_refinement.{pdf,png}` — "oracle mixture level" →
  "exact-score mixture level".

All seven written to both `inverse_heat_score_grw/figures/` and copied to
`paper_draft/figures/` (the directory `main.tex` includes). Drivers:
`make_figures.py` (relabels) and `make_new_figures.py` (new figures).

---

## 4. Reconciliation note (decisive)

Restricting the new 25-seed run to **seeds {0,1,2}** reproduces the frozen
3-seed means **bit-for-bit** for every method and noise level:

| method | η | 3-seed subset | frozen 3-seed |
|---|---|---|---|
| smoothed_log_bw4 | 0/.001/.005/.01 | 0.0120 / 0.0280 / 0.1101 / 0.1970 | 0.012 / 0.028 / 0.110 / 0.197 |
| smoothed_log_bw6 | 0/.001/.005/.01 | 0.0256 / 0.0256 / 0.0268 / 0.0299 | 0.026 / 0.026 / 0.027 / 0.030 |
| fd_grid_ratio_bw4 | 0/.001/.005/.01 | 0.0133 / 0.4551 / 0.4952 / 0.5041 | 0.013 / 0.455 / 0.495 / 0.504 |
| tikhonov_optimal | 0/.001/.005/.01 | 0.0010 / 0.0029 / 0.0134 / 0.0186 | 0.001 / 0.003 / 0.013 / 0.019 |

⇒ the configuration is **identical**; no drift. The 25-seed vs frozen-3-seed
differences are purely the effect of averaging more realizations:

- All **particle-method** means are within **≤12.7%** of the frozen 3-seed means
  (sl4: +9–13%; sl6: ≤6%; fd4: ≤10%) — inside the 15–20% reconciliation band.
- The **Tikhonov reference only** exceeds 20% at the two lowest noise levels
  (η=0.001 +30%, η=0.005 +23%), where absolute values are tiny (0.003→0.0039,
  0.013→0.016) and the 3-seed subset matches frozen exactly. This is the
  small-sample instability of the optimal-λ Tikhonov reference at low noise —
  precisely the weakness this experiment was built to remove. The 25-seed mean
  is the more reliable estimate. Not a config issue.

Also separately verified before any new run: Test B gradient-glob E2 = 0.1749
(≈0.175) and density-particle E2 = 0.0069 reproduce exactly (fig 4.2 plots these
same runs).

### Section 7 checklist
- [x] No `oracle` in any **paper** figure's visible labels/titles. The only
  remaining occurrences in `make_figures.py` are internal CSV identifiers
  (`method_type=="..._oracle"`, `score_method=="oracle"`, `df.method=="oracle"`),
  a function name, comments, and the title of `fig_representation_audit` — which
  is **not** in the paper's `\includegraphics` list (task said to ignore it).
- [x] 25-seed particle means match frozen 3-seed means within 15–20%; 3-seed
  subset is a bit-exact match (config verified unchanged). Tikhonov-reference
  deviation at low η explained above.
- [x] Every new run used `alpha=0.01, n_grid=400, dt=0.001, [0,1]`, Neumann, and
  the `h/dx` bandwidth convention.
- [x] Fig 4.2 reproduces the existing Test B numbers (0.175 / 0.0069) — it plots
  the same exact-score runs, not new ones.
- [x] No `.tex` modified by this task. (Aside: `paper_draft/main.tex` changed on
  disk during the session — 75067→79378 bytes, today's mtime — from the user's
  own IDE editing, **not** this task. Flagging for awareness only.)
- [x] No number in this report is estimated or hand-filled; all from actual runs.
