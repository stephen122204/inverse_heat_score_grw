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

## Running

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
