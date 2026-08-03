# Score-Guided Density Particles for Backward Heat Reconstruction

This repository contains computer code for reproducing the numerical results
described in the manuscript *Score-Guided Density Particles for Backward Heat
Reconstruction* by Stephen Abkin and Prabir Daripa.

**Paper:** link to be added (arXiv preprint forthcoming).

## Getting Started

```bash
git clone https://github.com/stephen122204/inverse_heat_score_grw.git
cd inverse_heat_score_grw
python -m venv .venv && source .venv/bin/activate   # tested with Python 3.11
pip install -r requirements.txt                     # pinned exact versions
```

`np.trapz` was removed in NumPy 2.0; the code uses a `np.trapezoid`
compatibility shim, so it runs on both NumPy 1.26.x and 2.x.

## Reproducing Numerical Results

Check every number reported in the paper against the checked-in study
outputs (runs in seconds, no simulation):

```bash
python reproduce.py verify
```

One command reruns every experiment behind the paper's tables and figures,
then regenerates the twelve shipped figures from the resulting CSVs (no
figure hardcodes any result):

```bash
python reproduce.py                  # all CSVs -> outputs/, all figures -> figures/
python reproduce.py --figures-only   # rebuild figures from existing CSVs
```

The studies, in the order they run: the grid-by-count representation and
convergence sweep, the time-step sweep, the score-estimation and bandwidth
audit, the validation stage, the 25-realization noise study, the
discrepancy-principle run with its tau = 1.2 reselection, the non-smooth
cases, the variable-coefficient audit, and the VH-mixture bandwidth
refinement. Figures are then rebuilt via `make_figures.py --only ...`,
`make_new_figures.py`, and `make_discrepancy_figure.py`.

**Determinism.** The density-particle method is deterministic (quantile
initialization, analytic/KDE scores, no RNG), so those experiment CSVs are
bit-for-bit reproducible. The studies that draw random observation noise use
fixed realizations: `0-24` for the reported noise and discrepancy tables, and
`0-2` for the superseded validation-stage reconciliation task; both are
reproducible in their realization means. `python reproduce.py verify` checks
every headline number against the archived outputs.

## Repository Layout

```
configs/                     experiment configs (YAML)
src/invheat_grw/             library modules (fields, methods, estimators, config)
scripts/run_*.py             the experiment studies listed above
scripts/verify_numbers.py    number checks behind reproduce.py verify
reproduce.py                 one-shot pipeline (studies + figures)
make_figures.py              core figures from archived CSVs
make_new_figures.py          noise-band, representation-failure, nonsmooth figures
make_discrepancy_figure.py   discrepancy figure
figures/                     the twelve shipped figures
outputs/                     archived study outputs behind the tables and figures
```

## Running Your Own Experiments

The sections above reproduce the paper. To run the solver drivers on your own
configuration, copy a YAML config from `configs/`, edit it, and run:

```bash
python scripts/run_all.py --config configs/gaussian_base.yaml
```

Each run creates a timestamped folder under `outputs/` containing
`metrics_summary.csv` (per-method error metrics), `run_summary.txt` (a
human-readable report), `config_used.yaml` (the exact configuration used),
and the diagnostic plots and saved arrays.
