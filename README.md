# Inverse Heat Score-Guided GRW

Research prototype for testing the score-guided backward Gradient Random Walk (GRW) method for the inverse heat equation.

## Background

The forward heat equation `u_t = alpha * u_xx` smooths fields over time. Naive backward GRW (flipping the sign of the Brownian increment) does **not** invert diffusion because `xi` and `-xi` have the same Gaussian distribution.

The correct reverse-time SDE requires a **score drift**:
```
s(x, t) = ∂_x log u(x, t) = u_x(x, t) / u(x, t)
```

### Methods tested

| Method | Update rule | Expected behavior |
|---|---|---|
| Naive backward | `X += -sqrt(2 α dt) ξ` | Fails — same distribution as forward |
| Oracle score deterministic | `X += α s(x,t) dt` | Should sharpen toward peak |
| Oracle score stochastic | `X += 2α s(x,t) dt + sqrt(2α dt) ξ` | Same trend, more noise |
| Estimated score deterministic (raw) | `X += α ŝ(x,t) dt` with `ŝ = u_x/u` from globs | May blow up if u ≈ 0 |
| Estimated score stochastic (raw) | `X += 2α ŝ(x,t) dt + sqrt(2α dt) ξ` | Same, with noise |

No regularization, clipping, smoothing, or KDE is applied in this version.

## Project structure

```
inverse_heat_score_grw/
  configs/gaussian_base.yaml   # experiment config
  src/invheat_grw/             # library modules
  scripts/run_all.py           # main entry point
  scripts/run_single.py        # run one method
  outputs/                     # timestamped result folders
```

## Installation

```bash
pip install -r requirements.txt
```

## Reproduce the paper (clone → run → numbers match)

```bash
git clone <repo-url> && cd inverse_heat_score_grw
python -m venv .venv && source .venv/bin/activate   # Python 3.11
pip install -r requirements.txt                     # pinned exact versions
python reproduce.py                                 # all CSVs -> outputs/, all figures -> figures/
```

`reproduce.py` runs the five paper experiments plus the grid×N
representation/convergence sweep (which backs **Fig 1**), then regenerates every
figure **from the resulting CSVs** via `make_figures.py` (no figure hardcodes any
result). Use `python reproduce.py --figures-only` to rebuild figures from
existing CSVs.

**Determinism.** The density-particle method is deterministic (quantile
initialisation, analytic/KDE scores, no RNG), so the experiment CSVs are
bit-for-bit reproducible; only the noise study uses fixed seeds `{0,1,2}` and is
reproducible in its seed-averaged means. A single source of truth for every
headline number lives in `FROZEN_NUMBERS.md`.

> Note: `np.trapz` was removed in NumPy 2.0; the code uses a `np.trapezoid`
> compatibility shim, so it runs on both NumPy 1.26.x and 2.x.

## Running a single GRW config (legacy driver)

```bash
cd inverse_heat_score_grw
python scripts/run_all.py --config configs/gaussian_base.yaml
```

## Outputs

Each run creates `outputs/YYYYMMDD_HHMMSS/` containing:
- `metrics_summary.csv` — L2 error, peak, width, TV per method
- `run_summary.txt` — human-readable report
- `config_used.yaml` — exact config used
- `*.png` — all diagnostic plots
- `*.npz` / `*.csv` — saved arrays

## Success / failure indicators

- **Step-0 reconstruction** should closely match `observed_final` (L2 < 0.01).
- **Naive backward** should *not* recover the true peak (L2 similar to baseline or worse).
- **Oracle deterministic** should improve L2 recovery vs. naive backward.
- **Raw estimated score** methods: improvement = promising; blow-up = justifies regularization.
