# Frozen numbers — single source of truth

Every headline number in the paper, regenerated from committed code on the
frozen artifact set in `outputs/paper_run_20260627/`. All values read directly
from the experiment CSVs. Fixed throughout: `alpha=0.01`, `dt=0.001`,
domain `[0,1]`, Neumann BC, density reconstruction bandwidth factor = 4
(production value). Python 3.11, numpy 2.4.4 (pinned).

Determinism: the density-particle method has no RNG (quantile init, analytic/KDE
scores). Independent reruns are **bit-identical** — verified:
- variable-coefficient audit: max |Δ rel_L2| = 0.00e+00 (20 rows)
- representation grid×N sweep: max |Δ rel_L2| = 0.00e+00 (240 rows)
Only the noise study uses seeds `{0,1,2}`; its seed-averaged means are reproducible.

---

## 1. Representation theorem — the spine (Fig 1 + Table 1)

**Structural-ceiling evidence (Fig 1).** Gradient-glob ORACLE rel_L2 across the
entire grid×globs sweep (n_grid ∈ {100,200,400,800} × globs ∈ {20,80}):

| Test | mean rel_L2 | min–max | **coeff. of variation** |
|------|-------------|---------|--------------------------|
| B    | 0.1755 | 0.1748–0.1771 | **0.52 %** |
| H    | 0.2400 | 0.2390–0.2422 | **0.53 %** |
| Z    | 0.1549 | 0.1546–0.1552 | **0.14 %** |

CoV < 1 % everywhere ⇒ the ceiling is structural (representation mismatch),
not undersampling. Refining grid OR glob count does not move it.

**Density-particle ORACLE (bw=4), grid×N — invariant in N, improves with grid:**

| Test | n_grid=100 | 200 | 400 | 800 | (across N=1k…20k) |
|------|-----------|-----|-----|-----|-------------------|
| B    | 0.0968 | 0.0268 | 0.0069 | 0.0018–0.0048 | N-invariant |
| H    | 0.1579 | 0.0505 | 0.0160 | 0.0072 | N-invariant |
| Z    | 0.2055 | 0.0647 | 0.0175 | 0.0046–0.0052 | N-invariant |

Density oracle is floored by KDE step-zero reconstruction (not backward
dynamics), so it is ~flat in N and never approaches the gradient-glob floor.

**Table 1 (canonical n_grid=400, density N=5000, bw=4):**

| Test | gradient-glob | density oracle | ratio |
|------|---------------|----------------|-------|
| B    | 0.175 | 0.0069 | 25× |
| H    | 0.239 | 0.0160 | 15× |
| Z    | 0.155 | 0.0175 | 8.9× |

---

## 2. Bandwidth = regularization (U-curves)

Best estimated-score vs oracle (rel_L2):

| Test | best method | best bw | best rel_L2 | oracle | ratio |
|------|-------------|---------|-------------|--------|-------|
| B | smoothed_log | 4 | 0.0122 | 0.0069 | 1.8× |
| H | fd_grid_ratio | 4 | 0.0657 | 0.0160 | 4.1× |
| Z | smoothed_log | 2 | 0.0102 | 0.0175 | 0.59× |

`direct_kde` ≈ `fd_grid_ratio` at the same bandwidth (verified bw=4, ε=1e-8):
B 0.0257 vs 0.0231 (+11%), H 0.0650 vs 0.0657 (−1%), Z 0.0229 vs 0.0230 (−0.6%)
⇒ the score *formula* matters less than the smoothing *scale* (§5.4).

---

## 3. Particle-count convergence (Test B, bw=4, clean)

| N | smoothed_log | fd_grid_ratio |
|---|--------------|---------------|
| 5000  | 0.0122 | 0.0231 |
| 10000 | 0.0120 | 0.0133 |

smoothed_log plateaus (bandwidth-bias dominated); fd keeps improving (lower bias,
higher variance).

---

## 4. Noise robustness (Test B, mean over seeds {0,1,2})

| Method | η=0 | 0.001 | 0.005 | 0.01 |
|--------|-----|-------|-------|------|
| smoothed_log bw4 | 0.012 | 0.028 | 0.110 | 0.197 |
| smoothed_log bw6 | 0.026 | 0.026 | 0.027 | 0.030 |
| fd_grid_ratio bw4 | 0.013 | 0.455 | 0.495 | 0.504 |
| Tikhonov (oracle λ) | 0.001 | 0.003 | 0.013 | 0.019 |

Larger bandwidth trades clean-data bias for noise stability; fd (no pre-smooth)
is catastrophic under noise; Tikhonov is the (oracle-tuned, best-case) reference.

---

## 5. Variable-coefficient heat (rel_L2)

| Case | oracle | sl bw4 | sl bw6 | fd bw4 | Tikhonov |
|------|--------|--------|--------|--------|----------|
| VB β=0.5 | 0.0083 | 0.0127 | 0.0270 | 0.0568 | <1e-3 |
| VB β=0.9 | 0.0084 | 0.0145 | 0.0277 | 0.1469 | <1e-3 |
| VH β=0.5 | 0.0201 | 0.0794 | 0.0870 | 0.0917 | 0.0054 |
| VH β=0.9 | 0.0204 | 0.0877 | 0.0897 | 0.2508 | 0.0055 |

Velocity `v = a(x) ∂ₓ log u`, no `a'` term. Forward solver: Crank–Nicolson at
`dt = 1e-3` (= reverse dt; oracle reads snapshots at this cadence).
**Paper appendix correction:** the appendix states `Δt_fwd = 1e-4`; the code uses
`1e-3`. Update the appendix text to `1e-3` (the code value is coupled to the
oracle snapshot cadence and is the frozen setting).

---

## 6. VH-mixture bandwidth refinement (β=0.5, smoothed_log)

| bw | 2 | 3 | 4 | 5 | 6 |
|----|---|---|---|---|---|
| rel_L2 | 0.734 | 0.106 | 0.079 | **0.078** | 0.087 |

Oracle floor 0.0201. Narrow stable window; flat minimum at bw 4–5; bandwidth
tuning alone cannot close the mixture/oracle gap.

---

## Provenance (all under `outputs/paper_run_20260627/`)

```
representation_audit_20260627_094153/   -> Fig 1, Table 1  (grid×N sweep)
score_estimation_audit_20260627_084924/ -> bandwidth U-curves (+ direct_kde bw4)
validation_stage_20260627_082621/       -> N-convergence + noise
variable_coefficient_audit_20260627_082333/ -> variable-coefficient
vh_mixture_bandwidth_refinement_20260627_084940/ -> VH-mixture
figures/                                 -> frozen copy of all 10 figures
```
Regenerate everything: `python reproduce.py` (or `--figures-only`).
