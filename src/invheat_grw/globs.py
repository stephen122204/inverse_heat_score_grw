"""
globs.py — Gradient-glob (GRW) representation of a 1-D field.

A "gradient glob" encodes the cumulative-sum (staircase) representation of a
discrete field.  Given field values u[0..N-1] on grid x[0..N-1]:

  • u_left : the base value u[0].
  • For each neighboring pair (j, j+1): delta_j = u[j+1] - u[j].
    – delta_j is split into `gradient_globs_per_jump` equal-weight sub-globs,
      each initially located at the interval midpoint x_mid = (x[j] + x[j+1]) / 2.

Reconstruction:
  Sort globs by position.  For each query point x_q:
    u_recon(x_q) = u_left + sum(weight_i  for all glob_i with pos_i < x_q)

This is a piecewise-constant cumulative-sum approximation.
It becomes exact in the limit gradient_globs_per_jump → ∞.

The GRW inverse method moves these globs according to a reverse-time SDE and
then reconstructs the field at the end of backward integration.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from .config import Config


@dataclass
class GlobState:
    """
    Mutable state for a collection of gradient globs.

    Attributes
    ----------
    positions : (N_globs,) array   — current x-coordinates of globs.
    weights   : (N_globs,) array   — fixed glob weights (delta / n_per_jump).
    u_left    : float              — base field value (u at x_min).
    x_min     : float
    x_max     : float
    """
    positions: np.ndarray
    weights: np.ndarray
    u_left: float
    x_min: float
    x_max: float


def field_to_globs(u: np.ndarray, x: np.ndarray, cfg: Config) -> GlobState:
    """
    Convert a discrete field u (sampled on grid x) into gradient globs.

    Parameters
    ----------
    u   : field values at each grid point.
    x   : corresponding spatial coordinates.
    cfg : experiment config (provides gradient_globs_per_jump, boundary, etc.).

    Returns
    -------
    GlobState with positions and weights.
    """
    n_per = cfg.grw.gradient_globs_per_jump
    n_grid = len(x)

    all_pos = []
    all_wts = []

    for j in range(n_grid - 1):
        delta = u[j + 1] - u[j]                  # jump at this grid edge
        x_mid = 0.5 * (x[j] + x[j + 1])         # midpoint of interval
        sub_weight = delta / n_per                 # each sub-glob carries this weight
        for _ in range(n_per):
            all_pos.append(x_mid)
            all_wts.append(sub_weight)

    return GlobState(
        positions=np.array(all_pos, dtype=float),
        weights=np.array(all_wts, dtype=float),
        u_left=float(u[0]),
        x_min=float(x[0]),
        x_max=float(x[-1]),
    )


def reconstruct_field(state: GlobState, x: np.ndarray) -> np.ndarray:
    """
    Reconstruct the field at query points x from glob positions and weights.

    For each query point x_q:
        u_recon(x_q) = u_left + sum(weight_i  for all i with positions_i < x_q)

    Globs are sorted by position for efficient cumulative summation.
    """
    # Sort globs by current position
    order = np.argsort(state.positions)
    sorted_pos = state.positions[order]
    sorted_wts = state.weights[order]
    cum_wts = np.cumsum(sorted_wts)  # cumulative weight up to and including glob i

    u_recon = np.empty(len(x), dtype=float)
    for qi, xq in enumerate(x):
        # sum weights of all globs strictly to the left of xq
        idx = np.searchsorted(sorted_pos, xq, side="left")  # first index >= xq
        if idx == 0:
            u_recon[qi] = state.u_left
        else:
            u_recon[qi] = state.u_left + cum_wts[idx - 1]

    return u_recon


def apply_reflecting_boundary(positions: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    """
    Apply reflecting boundary conditions to glob positions.

    Globs that exit [x_min, x_max] are reflected back.
    One mirror pass per wall, then a clip fallback for displacements
    longer than the domain (never reached at the paper's step sizes).
    """
    pos = positions.copy()
    L = x_max - x_min

    # Reflect off left wall
    below = pos < x_min
    pos[below] = 2.0 * x_min - pos[below]

    # Reflect off right wall
    above = pos > x_max
    pos[above] = 2.0 * x_max - pos[above]

    # Clip any remaining out-of-bounds (handles extreme steps)
    pos = np.clip(pos, x_min, x_max)
    return pos
