# Normalized-score backward heat transport

Code to reproduce the numerical illustration in *Failure of Uniform Density Recovery in Normalized-Score Backward Heat Transport*, by Stephen Abkin and Prabir Daripa.

## Setup

Use Python 3.11.

```bash
git clone --depth 1 https://github.com/stephen122204/inverse_heat_score_grw.git
cd inverse_heat_score_grw
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproduce the paper

Check the saved inputs and numerical bounds:

```bash
python reproduce.py verify
```

Recompute all eight refinement runs and compare their numerical fields with the saved results:

```bash
python reproduce.py replay --all --output outputs/paper
```

Results are written as JSON files in `outputs/paper/`. The selected paper calculation is `illustration_J4_dt0025_p6_d80.json`; it includes the source amplitude, exact-data amplitude, reconstructed second harmonic, and nonlinear correction. Omit `--all` to recompute only that calculation.

Generate the paper figure from the saved, verified results:

```bash
python reproduce.py figures --output outputs/figures
```

This writes `nonlinear_illustration.pdf` and `nonlinear_illustration.png`.

## Change the numerical resolution

```bash
python reproduce.py run --J 4 --P 6 --dt 0.05 --dps 80 --output outputs/custom
```

`J` sets the number of positive Fourier modes, `P` the denominator series order, `dt` the step in rescaled time, and `dps` the decimal precision. Results are saved in `outputs/custom/custom_run.json`. The target and physical parameters remain those of the paper's fixed example; this is not a solver for arbitrary input data.

Verification and numerical runs use only Python's standard library. The installed packages are needed for plotting. Numerical refinement does not certify time or spectral discretization error.
