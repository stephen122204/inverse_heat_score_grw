# Normalized-score transport for inverse heat reconstruction

Companion code for *Failure of Uniform Density Recovery in Normalized-Score Backward Heat Transport*, by Stephen Abkin and Prabir Daripa (unpublished manuscript, September 11, 2026).

The paper studies a nonlinear limitation of density recovery from exact heat observations. This repository contains the numerical illustration, its refinement checks, and the earlier inverse-heat comparisons.

## Quick start

Use Python 3.11 with the pinned dependencies. From a terminal:

```bash
git clone https://github.com/stephen122204/inverse_heat_score_grw.git
cd inverse_heat_score_grw
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python research/normalized_score/reproduce.py verify
python research/normalized_score/reproduce.py figures
```

The figure is written to `research/normalized_score/generated/`. To recompute all eight saved refinement runs in a separate output directory:

```bash
python research/normalized_score/reproduce.py replay --all --output /tmp/normalized-score-replay
```

Omit `--all` to replay only the selected finest calculation. Verification and decimal replay use only the Python standard library; figure generation uses NumPy and Matplotlib.

## Code and evidence

- [`research/normalized_score/`](research/normalized_score/): decimal integrator, fixed inputs, saved results, and reproduction commands.
- [`src/invheat_grw/`](src/invheat_grw/): the earlier bounded-interval solver.
- [`analysis/phase2c/`](analysis/phase2c/), [`manifests/phase2c_campaign.json`](manifests/phase2c_campaign.json), and [`PHASE2C_PROTOCOL.txt`](PHASE2C_PROTOCOL.txt): preserved comparisons, file hashes, and the frozen experimental protocol.

The obstruction concerns **uniform recovery over a source class with varying targets**. The illustration shows an extremely small generated oscillation; it does not establish practical failure or failure for a single fixed target. Numerical refinement checks are separate from the manuscript's analytic remainder bounds. Earlier comparisons favoring Tikhonov are retained.

Run the solver tests with:

```bash
PYTHONPATH=src python -m pytest -q tests
```

Citation information is in [`CITATION.cff`](CITATION.cff).
