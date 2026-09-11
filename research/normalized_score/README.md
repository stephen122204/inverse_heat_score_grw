# Normalized-score reconstruction obstruction

This directory reproduces the admissible numerical illustration for *Failure of Uniform Density Recovery in Normalized-Score Backward Heat Transport* by Stephen Abkin and Prabir Daripa. It is separate from the bounded-Neumann benchmark solver and does not change its production equation or frozen protocol.

## Mathematical object

The zero-mean perturbation of the density evolves by the full normalized Gaussian-score quotient. The worked example is on the unit Neumann interval with diffusivity 0.01, endpoint 1, frequency 32π, bandwidth 1/(64π), denominator regularization h² and amplitude parameter 10⁻⁸. The returned field receives the final Gaussian smoothing. The true-source and observed amplitudes are represented as separate decimal coefficients so that adding them to one does not destroy the input in floating point.

`integrate.py` is the unchanged research integrator whose hash is recorded in `manifest.json`. It integrates the complete rational Fourier-Galerkin flux in weighted variables, subtracting the known linear and quadratic solution terms algebraically. Intermediate convolution modes are retained before final projection. Its denominator-series tail estimate is distinct from spatial and timestep error.

## Reproduce

From the repository root, with the pinned requirements installed:

```bash
python research/normalized_score/reproduce.py verify
python research/normalized_score/reproduce.py figures
python research/normalized_score/reproduce.py replay --all --output /tmp/normalized-score-replay
```

Verification uses Python's standard library. Plotting additionally uses NumPy and Matplotlib from the root requirements. Decimal replay needs no external numerical library. `replay` without `--all` recomputes the selected finest calculation. All eight refinement runs are selected explicitly by the manifest and checked field by field, excluding elapsed wall time. The full replay took about four minutes in the original research environment; machine-dependent duration is not a performance claim.

Outputs never overwrite the archived `results/` inputs. Figure metadata omits creation and modification times. The figure shows perturbations at physical scales of 10⁻¹⁸ and 10⁻³⁵; it has no embedded title and remains distinguishable in grayscale.

## Evidence boundary

The analytic remainder gives the actual PDE second cosine coefficient an enclosure near [9.4550, 9.5751]×10⁻¹⁸. The numerical refinement record is not an interval certificate of time or spectral discretization. In particular, the much smaller difference between the nonlinear and quadratic Galerkin calculations is not claimed to be a certified PDE coefficient.

The theorem concerns failure of uniform recovery across a fixed source class with varying targets. Neither this example nor this program establishes failure for one fixed nonconstant target, practical severity at ordinary amplitudes, algorithmic superiority, or a Lean-verified proof. All unfavorable earlier comparisons with Tikhonov remain in `analysis/phase2c` and the frozen campaign record.

## Files

- `manifest.json`: fixed input list and hashes.
- `results/`: eight complete numerical refinement records and the original research summary.
- `integrate.py`: decimal nonlinear integrator, preserved byte for byte.
- `reproduce.py`: portable verification, figure generation, and isolated replay.
- `generated/`: rebuilt figure and verification output; reproducible from the inputs above.

The integrator uses the eighth-order coefficients of E. Fehlberg, NASA TR R-287 (1968), <https://ntrs.nasa.gov/citations/19680027281>. The manuscript gives the equation, variable transformation, complete analytic proof and numerical-enclosure interpretation. No randomness or fitted coefficient enters this illustration.
