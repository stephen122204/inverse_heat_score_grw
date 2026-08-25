"""
baselines.py — Classical spectral inversion baselines for the heat equation.

These are NOT particle methods.  They serve as accuracy ceilings / reference
solutions for comparing against the GRW-based inverse methods.

Manuscript labels (draft): spectral cutoff `eq:spectral_cutoff`, Tikhonov
`eq:tikhonov` (`sec:baselines`).

Two baselines are provided:

1. spectral_cutoff_inverse
   Inverts the heat equation in frequency space, keeping only modes whose
   forward multiplier A_n = exp(-alpha * k_n^2 * T) is not catastrophically
   small.  Unstable (amplified) high-frequency modes are zeroed out.

   Cutoff rule: keep mode n if exp(alpha * k_n^2 * T) <= 1 / noise_delta,
   i.e., k_n <= sqrt(log(1/noise_delta) / (alpha * T)).

   If k_cut is given directly, it overrides the noise_delta calculation.
   Default noise_delta = 1e-8.

2. tikhonov_inverse
   Tikhonov-regularized inversion: each mode n is recovered as
       u0_hat_n = (A_n / (A_n^2 + lambda)) * obs_hat_n
   instead of the unstable 1/A_n division.
   At low frequencies, A_n ≈ 1, so the correction is small.
   At high frequencies, A_n → 0, so the Tikhonov term dominates and
   damps the mode rather than amplifying it.

Spatial convention:
   Domain [x_min, x_max], Neumann boundary conditions.
   DCT-II with norm="ortho" is used throughout (same as forward_heat_solve_dct
   in metrics.py).  Mode n=0 is the constant (DC) mode, mode n has spatial
   frequency k_n = n*pi/(x_max - x_min).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.fft import dct, idct


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class SpectralCutoffResult:
    """Result of spectral cutoff inversion."""
    candidate: np.ndarray
    method_name: str = "spectral_cutoff_baseline"
    # Cutoff parameters
    k_cut: float = 0.0          # cutoff wavenumber
    n_modes_kept: int = 0       # number of Fourier modes retained
    n_total_modes: int = 0      # total number of modes (= len(observed))
    noise_delta: float = 1e-8   # noise_delta used to compute k_cut (if auto)
    max_inverse_multiplier: float = 0.0  # max exp(alpha * k_n^2 * T) over kept modes
    # Stability diagnostics
    completed: bool = True
    failure_msg: str = ""


@dataclass
class TikhonovResult:
    """Result of Tikhonov-regularized inversion."""
    candidate: np.ndarray
    method_name: str = "tikhonov_baseline"
    lam: float = 0.0            # Tikhonov regularization parameter
    # Stability diagnostics
    completed: bool = True
    failure_msg: str = ""


# ---------------------------------------------------------------------------
# Spectral cutoff inversion
# ---------------------------------------------------------------------------

def spectral_cutoff_inverse(
    observed: np.ndarray,
    x_grid: np.ndarray,
    alpha: float,
    T: float,
    k_cut: Optional[float] = None,
    noise_delta: Optional[float] = None,
) -> SpectralCutoffResult:
    """
    Invert the heat equation via spectral cutoff.

    Parameters
    ----------
    observed    : observed field u(., T) on x_grid.
    x_grid      : uniform spatial grid on [x_min, x_max].
    alpha       : thermal diffusivity.
    T           : forward diffusion time.
    k_cut       : cutoff wavenumber (rad/unit length).  If None, computed
                  from noise_delta.
    noise_delta : noise level estimate; used to compute k_cut automatically:
                  k_cut = sqrt(log(1/noise_delta) / (alpha * T)).
                  Default: 1e-8 (essentially noiseless, aggressive retention).

    Returns
    -------
    SpectralCutoffResult with candidate field and diagnostics.
    """
    N = len(observed)
    L = float(x_grid[-1] - x_grid[0])

    if noise_delta is None:
        noise_delta = 1e-8

    # Compute k_cut from noise_delta if not given
    if k_cut is None:
        # exp(alpha * k^2 * T) <= 1/noise_delta  =>  k <= sqrt(log(1/noise_delta)/(alpha*T))
        if alpha * T > 0:
            k_cut = math.sqrt(math.log(1.0 / noise_delta) / (alpha * T))
        else:
            k_cut = float("inf")

    # Forward DCT-II (same as metrics.forward_heat_solve_dct)
    c: np.ndarray = dct(observed, type=2, norm="ortho")  # type: ignore[assignment]

    # Wavenumber for mode n: k_n = n * pi / L
    n_modes = np.arange(N)
    k_n = n_modes * math.pi / L
    A_n = np.exp(-alpha * k_n ** 2 * T)  # forward multiplier (decays)

    # Determine which modes to keep
    keep = k_n <= k_cut

    # Invert: multiply by exp(+alpha * k_n^2 * T) for kept modes, zero for cut
    inv_multiplier = np.where(keep, np.exp(alpha * k_n ** 2 * T), 0.0)

    n_kept = int(np.sum(keep))
    max_inv = float(np.max(inv_multiplier[keep])) if n_kept > 0 else 0.0

    c_inv = c * inv_multiplier
    candidate: np.ndarray = idct(c_inv, type=2, norm="ortho")  # type: ignore[assignment]

    return SpectralCutoffResult(
        candidate=candidate,
        k_cut=k_cut,
        n_modes_kept=n_kept,
        n_total_modes=N,
        noise_delta=noise_delta,
        max_inverse_multiplier=max_inv,
        completed=True,
        failure_msg="",
    )


# ---------------------------------------------------------------------------
# Tikhonov-regularized inversion
# ---------------------------------------------------------------------------

def tikhonov_inverse(
    observed: np.ndarray,
    x_grid: np.ndarray,
    alpha: float,
    T: float,
    lam: float,
) -> TikhonovResult:
    """
    Tikhonov-regularized inversion of the heat equation.

    For each Fourier mode n with forward multiplier A_n = exp(-alpha*k_n^2*T):
        u0_hat_n = (A_n / (A_n^2 + lambda)) * obs_hat_n

    This reduces to pure inversion (1/A_n) when A_n >> sqrt(lambda),
    and damps to (A_n/lambda)*obs_hat_n when A_n << sqrt(lambda).

    Parameters
    ----------
    observed : observed field u(., T) on x_grid.
    x_grid   : uniform spatial grid.
    alpha    : thermal diffusivity.
    T        : forward diffusion time.
    lam      : Tikhonov regularization parameter (> 0).

    Returns
    -------
    TikhonovResult with candidate field and diagnostics.
    """
    if lam <= 0.0:
        raise ValueError(f"Tikhonov lambda must be positive, got {lam}")

    N = len(observed)
    L = float(x_grid[-1] - x_grid[0])

    c: np.ndarray = dct(observed, type=2, norm="ortho")  # type: ignore[assignment]

    n_modes = np.arange(N)
    k_n = n_modes * math.pi / L
    A_n = np.exp(-alpha * k_n ** 2 * T)

    # Tikhonov filter: A_n / (A_n^2 + lam)
    tik_filter = A_n / (A_n ** 2 + lam)

    c_inv = c * tik_filter
    candidate: np.ndarray = idct(c_inv, type=2, norm="ortho")  # type: ignore[assignment]

    return TikhonovResult(
        candidate=candidate,
        lam=lam,
        completed=True,
        failure_msg="",
    )
