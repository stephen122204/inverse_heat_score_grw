# Normalized-score transport for inverse heat reconstruction

Companion code for **Failure of Uniform Density Recovery in Normalized-Score Backward Heat Transport**, by Stephen Abkin and Prabir Daripa (unpublished manuscript, September 10, 2026).

The active analysis contribution is a continuum obstruction to uniform density recovery from exact heat observations on a fixed smooth positive source class. Its numerical illustration uses the actual nonlinear quotient and the returned smoothed field. Fixed-target failure and practical severity at ordinary amplitudes remain open. No Lean formalization is claimed.

## Current manuscript illustration

The self-contained entry point is [research/normalized_score](research/normalized_score/README.md):

```bash
git clone --branch codex/normalized-score-manuscript https://github.com/stephen122204/inverse_heat_score_grw.git
cd inverse_heat_score_grw
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python research/normalized_score/reproduce.py verify
python research/normalized_score/reproduce.py figures
python research/normalized_score/reproduce.py replay --all --output /tmp/normalized-score-replay
```

Verification and decimal replay use only Python's standard library. Plotting uses the pinned scientific dependencies. Eight fixed refinement records are preserved by hashes; replays compare every numerical field except wall time. The analytic PDE enclosure is separate from numerical refinement differences.

## Preserved solver and earlier evidence

The bounded-Neumann production solver and frozen Phase 2C protocol remain unchanged by the new illustration. Earlier noisy comparisons favor Tikhonov; the data, analysis and definitions remain available in [analysis/phase2c](analysis/phase2c), [the campaign manifest](manifests/phase2c_campaign.json), and [the protocol](PHASE2C_PROTOCOL.md). These older experiments are distinct from the new theorem illustration.

Run the existing solver tests with:

```bash
PYTHONPATH=src python -m pytest -q tests
```

The September 10 validation passed all 133 tests. All eight new illustration replays matched their saved outputs. The existing root-level reproduction scripts belong to earlier manuscript versions. Their original instructions are preserved in [the historical README](docs/README-v5.1-and-phase2c.md); do not use those scripts to reproduce the normalized-score illustration.

## Research record and citation

The manuscript source and the full investigation archive are maintained in the separate manuscript repository. This public repository supplies the runnable numerical companion and earlier benchmark evidence. Cite the software or unpublished manuscript using [CITATION.cff](CITATION.cff). No public manuscript identifier or journal acceptance is asserted.
