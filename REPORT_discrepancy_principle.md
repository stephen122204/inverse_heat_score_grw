# Discrepancy-principle experiment — realistic parameter choice

Branch: `experiment-discrepancy-principle`. Both Tikhonov's λ and the particle
bandwidth are chosen by the **discrepancy principle** (Morozov), using only the
noisy data and the known noise level — never `u0`. `u0` is read for **scoring
only** (E2). This converts the paper's "best-case vs best-case" into
realistic-vs-realistic, with oracle columns kept for contrast.

**Bring these numbers back for a consistency check before any table/text. No `.tex` edited.**

Frozen config: Test B, `alpha=0.01`, `[0,1]`, Neumann, `n_grid=400`, `dt=0.001`,
`epsilon=1e-8`, `N=10000`, smoothed-log. Noise `g+eta*max(g)*N(0,1)`,
`rng=default_rng(seed)`, seeds `0..24`, `eta∈{0.001,0.005,0.01}`. Tikhonov λ on
the frozen grid `[1e-8..1e-2]`; particle bandwidth factor on `{1,2,4,6,8,12,16}`.
Forward operator `H_T` = `forward_heat_solve_dct` (the operator the paper's
forward-consistency uses and `tikhonov_inverse` inverts).

---

## 3. How `delta` was computed (no `u0`)

Relative forward residual `r = ||H_T[u0_hat] - g_noisy||_2 / ||g_noisy||_2`,
discrete norm `||v||_2 = sqrt(dx·Σ v_i²)`.

Noise-level estimate (same for every seed at a given `eta`, from the noise model
and the **data-side** noiseless field `g_clean = u(·,T)`, **not** `u0`):

```
delta_nominal(eta) = eta * max(g_clean) * sqrt(dx * n_grid) / ||g_clean||_2
```
= the expected relative L2 norm of the added perturbation
(`E||noise||_2 = eta·max(g_clean)·sqrt(dx·Σ E[z_i²]) = eta·max(g_clean)·sqrt(dx·n_grid)`).
Values: `max(g_clean)=0.82509`, `||g_clean||_2=0.34205`, so
`delta_nominal = 0.0024 / 0.0121 / 0.0242` at `eta = 0.001 / 0.005 / 0.01`.
(`g_clean` is the forward field noise is added to — the data — not the
reconstruction target `u0`. DCT-forward(`u0`) vs analytic `g_clean` differ by
6.9e-4 rel-L2, so `H_T` is consistent.)

**Morozov target = `tau · delta_nominal`.** `tau=1` is the literal expected-noise
level; `tau=1.2` is the **conventional safety factor**. The safety factor is
needed here because the **Tikhonov forward residual is flat at the noise floor**
across `λ∈[1e-8,1e-3]` (it bottoms at ≈0.97·delta and only rises past `λ≈1e-3`).
With `tau=1` the target sits inside that flat band, so per-seed noise pushes
"smallest λ with r≥delta" to `λ=1e-8` (no regularization) for ~half the seeds →
blow-up. `tau=1.2` puts the target above the flat floor and restores the unique
crossing. **The same `delta` (and `tau`) tunes both methods at each `(seed,eta)`.**
Selection rules: Tikhonov — smallest λ with `r≥tau·delta`, log-interpolated to the
crossing. Particle — the smoothed-log forward residual is **U-shaped (non-monotone)
in bandwidth**, so per the spec the bandwidth is chosen as `argmin_f |r(f)-tau·delta|`
on the grid (flagged non-monotone; this held for every seed).

---

## 1. Headline table — E2 (mean ± std, 25 seeds)

`disc` = discrepancy-tuned (realistic, no truth). `oracle` = tuned against truth
(grid-restricted), kept for contrast.

| eta | particle (oracle bw) | particle (disc bw) | Tikhonov (oracle λ) | Tikhonov (disc λ) |
|---|---|---|---|---|
| **τ = 1.2 (standard safety factor — headline)** ||||
| 0.001 | 0.0251 ± 0.0016 | 0.0306 ± 0.0059 | 0.0039 ± 0.0009 | 0.0038 ± 0.0009 |
| 0.005 | 0.0273 ± 0.0009 | 0.1235 ± 0.0265 | 0.0160 ± 0.0024 | 0.0126 ± 0.0027 |
| 0.010 | 0.0319 ± 0.0023 | 0.1107 ± 0.1109 | 0.0209 ± 0.0025 | 0.0209 ± 0.0025 |
| **τ = 1.0 (literal; Tikhonov fragile — see note)** ||||
| 0.001 | 0.0251 ± 0.0016 | 0.0306 ± 0.0059 | 0.0039 ± 0.0009 | 0.1653 ± 0.3738 |
| 0.005 | 0.0273 ± 0.0009 | 0.1235 ± 0.0265 | 0.0160 ± 0.0024 | 0.8118 ± 1.8765 |
| 0.010 | 0.0319 ± 0.0023 | 0.2221 ± 0.0473 | 0.0209 ± 0.0025 | 1.6232 ± 3.7537 |

(Particle oracle/disc are identical across τ at eta≤0.005 because the particle
target stays below the U-curve's residual floor either way; at eta=0.01 the
larger τ shifts some seeds from bw=4 to bw=6, hence the different disc mean.)

**Reading (τ=1.2, the realistic standard):**
- **Tikhonov: discrepancy ≈ oracle** (0.004/0.013/0.021 vs 0.004/0.016/0.021).
  Its residual is monotone in λ, so the achievable rule finds essentially the
  optimal λ — at eta=0.005 the interpolated continuous λ even edges below the
  grid-restricted oracle (0.0126 < 0.0160). Realistic tuning is a non-issue for Tikhonov.
- **Particle: discrepancy is markedly worse than oracle**, and the gap grows with
  noise — 1.2× at eta=0.001, **4.5×** at eta=0.005 (0.124 vs 0.027), ~3.5× at
  eta=0.01 with large seed-to-seed spread. The discrepancy principle picks
  bandwidth ≈4 while the error-optimal bandwidth is ≈6: the particle forward
  residual bottoms out (U-shape) at a *smaller* bandwidth than the E2 optimum and
  never reaches the noise floor, so residual-matching systematically under-smooths.

**This is an honest limitation, not a failure of the method:** with truth-blind
parameter choice the particle reconstruction is usable but visibly noisier than
its oracle-tuned self, whereas Tikhonov is unaffected. It sharpens the paper's
existing point that automatic bandwidth selection is the open practical problem.

---

## 2. Chosen parameters (mean over 25 seeds)

| eta | particle bw: oracle / disc(τ1.0, τ1.2) | Tikhonov λ: oracle / disc(τ1.0, τ1.2) |
|---|---|---|
| 0.001 | 5.44 / 4.00, 4.00 | 1.0e-3 / 3.6e-5, 1.0e-3 |
| 0.005 | 6.00 / 4.00, 4.00 | 4.4e-3 / 1.4e-4, 3.0e-3 |
| 0.010 | 6.00 / 4.00, 5.28 | 9.1e-3 / 2.0e-4, 1.0e-2 |

τ=1.2 Tikhonov λ tracks the oracle (1e-3, 3e-3, 1e-2); τ=1.0 λ collapses toward
the grid floor (the fragility). Particle disc bw sits at ~4 vs oracle ~6.

---

## 4. Figure

`fig_discrepancy_comparison.{pdf,png}` (in `figures/` and `paper_draft/figures/`),
τ=1.2, eta=0.005:
- **(a)** representative seed 7 (disc-particle E2=0.122 ≈ the mean 0.124): true `u0`,
  discrepancy-tuned particle (E2=0.122, visibly under-smoothed/wiggly), and
  discrepancy-tuned Tikhonov (E2=0.012, overlays truth).
- **(b)** seed-mean E2 vs bandwidth U-curve: mean discrepancy bw = **4.0** (red,
  on the steep left branch) vs the error-minimizing bw = **6** (green, at the
  minimum). The visible gap is the diagnostic — the achievable rule does not find
  the particle optimum.

---

## 5/6. Reconciliation

- [x] **Oracle-λ Tikhonov 25-seed means = 0.0039 / 0.0160 / 0.0209** → 0.004 / 0.016
  / 0.021. Exact match to the anchor; config did not drift.
- [x] `u0` used **only** for scoring. In code it is read solely inside `E2(...)`
  (`run_discrepancy_principle.py` / `reselect_discrepancy.py`); `delta` and every
  parameter selection use only `g_clean`, `g_noisy`, the noise level, and the
  forward residual. Re-selection validated bit-identical to the first run
  (particle-oracle max|ΔE2|=0, Tikhonov-oracle max|ΔE2|=1e-16, particle-disc(τ1)
  max|ΔE2|=0).
- [x] Same `delta` (and τ) tunes both methods at each `(seed,eta)`; `delta` from
  the noise model, not the truth.
- [x] Every run: `alpha=0.01`, `n_grid=400`, `dt=0.001`, `[0,1]`, Neumann, frozen
  noise model, seeds `0..24`, `h/dx` bandwidth convention.
- [x] No `.tex` modified. No reported number estimated or hand-filled; all from runs.

**Caveats to weigh before this goes into text:**
1. The **safety factor τ=1.2** is a real assumption (standard Morozov practice).
   I report τ=1.0 too; the τ=1.0 Tikhonov blow-up is a genuine fragility of the
   discrepancy principle at the residual floor, not a property of Tikhonov.
2. The oracle columns are **grid-restricted** (λ on the 7-point grid; bw on the
   7-point grid) while discrepancy-λ is continuously interpolated — so at eta=0.005
   discrepancy-Tikhonov can edge below grid-oracle. The takeaway (Tikhonov disc ≈
   oracle) is unaffected.

Data/scripts: `outputs/discrepancy_principle_20260629_165325/` (full sweeps),
`outputs/discrepancy_principle_final_20260629_165900/` (final selection + figure
arrays); `scripts/run_discrepancy_principle.py`, `scripts/reselect_discrepancy.py`,
`make_discrepancy_figure.py`.
