# Score-Guided Density Particles for Backward Heat Reconstruction

This repository contains computer code for reproducing the numerical results
described in the manuscript *Score-Guided Density Particles for Backward Heat
Reconstruction* by Stephen Abkin and Prabir Daripa.

**Paper:** link to be added (arXiv preprint forthcoming).

> **Status (2026-09-01).** The repository has two layers. The archived
> v5.1 pipeline documented below reproduces the earlier draft's twelve
> figures and is kept verbatim for provenance; its free-space estimator
> is retained only for the labeled historical transition table. The
> current production layer is the frozen Phase 2C campaign: the
> canonical cell-centered Neumann estimator (`neumann_kde`, the default
> for all campaign work, no longer "opt-in"), the preregistered
> 3,203-row protocol (`PHASE2C_PROTOCOL.md`, FROZEN), its schema and
> result contracts (`campaign_schema.py`, `campaign_results.py`), the
> study drivers (`campaign_drivers.py`, `campaign_driver_transition.py`),
> and the orchestrator (`scripts/run_science_campaign.py`; `--run STUDY`
> executes behind the freeze gate with per-study hash manifests). The
> test suite (`python -m unittest discover -s tests`) covers both
> layers. The campaign itself is intentionally not yet run; see the
> parent repository's REFRAME_PLAN.md.

## Getting Started

```bash
git clone https://github.com/stephen122204/inverse_heat_score_grw.git
cd inverse_heat_score_grw
python -m venv .venv && source .venv/bin/activate   # tested with Python 3.11
pip install -r requirements.txt                     # pinned exact versions
```

`np.trapz` was removed in NumPy 2.0. The code uses a `np.trapezoid`
compatibility shim, so it runs on both NumPy 1.26.x and 2.x.

## Reproducing Numerical Results

Three commands cover everything:

```bash
python reproduce.py verify           # check every reported number against the archived outputs
python reproduce.py                  # rerun every experiment, then rebuild all twelve figures
python reproduce.py --figures-only   # rebuild from manifests/paper_v5_1.json
```

`verify` runs in seconds and needs no simulation. The full run executes the
studies below in order, writes a manifest containing the exact output paths and
file hashes, then regenerates the twelve shipped figures from that manifest.
No figure hardcodes a result or selects the lexicographically latest output.
To use a different archived run, pass `--manifest PATH` to `reproduce.py`.

| Study | Wall clock |
|---|---|
| grid-by-count representation and convergence sweep | ~26 s |
| time-step sweep | ~2 s |
| score-estimation and bandwidth audit | ~56 min |
| validation stage | ~4 min |
| 25-realization noise study | ~13 min |
| discrepancy-principle run and tau = 1.2 reselection | ~28 min |
| non-smooth cases | ~8 s |
| variable-coefficient audit | ~44 s |
| VH-mixture bandwidth refinement | ~16 s |
| figure rebuild (all twelve figures) | ~11 s |

Times were measured on an Apple-Silicon laptop. A full rerun totals about
1 h 45 min.

**Determinism.** The density-particle method is deterministic (quantile
initialization, analytic/KDE scores, no RNG), so those experiment CSVs are
bit-for-bit reproducible. The noise-driven studies use fixed realizations
(`0-24` for the noise and discrepancy tables, `0-2` for the superseded
validation-stage task) and are reproducible in their realization means.

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
provenance.py                run-manifest validation and creation
manifests/                   pinned study paths used by archived paper versions
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

### Canonical Neumann score estimator

The repaired analysis branch uses one mass-preserving Neumann heat kernel and
its analytic derivative. It is opt-in while the bandwidth studies are being
rerun and does not alter the archived paper outputs:

```python
run_density_particle_estimated_score_deterministic(
    observed, x, cfg, n_particles=5000,
    recon_method="neumann_kde",
    score_method="neumann_kde",
    bandwidth=0.02,       # physical units; never inferred from grid spacing
    epsilon=1e-8,
)
```

Run its acceptance tests with
`PYTHONPATH=src python -m unittest -v tests.test_neumann_kernels`.

## Acknowledgments

**Principal Investigator:** [Professor Prabir Daripa](https://artsci.tamu.edu/mathematics/contact/profiles/prabir-daripa.html) — Texas A&M University, Department of Mathematics

Other projects from the Daripa Research Group are available on the
[group's GitHub page](https://github.com/Daripa-Research-Group).
