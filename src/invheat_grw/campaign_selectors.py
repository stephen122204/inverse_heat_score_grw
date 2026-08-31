"""campaign_selectors.py — parameter-selection rules for Phase 2C.

Implements the selection contract of PHASE2C_PROTOCOL.md Sections 5 and 6:

- **Oracle continuous lambda**: a 65-point uniform scan in
  ``z = log10(lambda)`` on ``[-12, -1]`` including both endpoints, bounded
  scalar refinement inside the two neighboring grid intervals around the best
  grid point at ``xatol = 1e-6`` in ``z``, endpoint re-evaluation, and the
  best of all candidates retained.  The record carries the coarse-grid
  winner, refined value, objective value, evaluation count, and endpoint
  flag.  The label is a truth-dependent diagnostic, never data-driven.
- **Morozov selection** for the linear Tikhonov family: the residual is
  checked numerically for monotonicity at relative tolerance ``1e-10``; a
  violation is a failed selection, not a switch to another rule.  When the
  target is attainable the bracketed root of ``r(lambda) = target`` is
  labeled ``morozov``; an unattainable target selects the nearest endpoint
  with an endpoint flag and is never labeled Morozov.
- **Residual-matched bandwidth** for the particle family, whose residual
  need not be monotone: the grid point minimizing ``|r(h) - target|``, ties
  resolved toward the larger bandwidth, with the full curve, selected index,
  absolute mismatch, and endpoint flag recorded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import brentq, minimize_scalar

Z_BOUNDS = (-12.0, -1.0)
SCAN_POINTS = 65
Z_XATOL = 1e-6
MONOTONE_RTOL = 1e-10


@dataclass
class SelectionRecord:
    label: str
    z: float | None
    lam: float | None
    value: float | None
    evaluations: int
    endpoint: bool
    extra: dict = field(default_factory=dict)


def oracle_continuous(
    objective: Callable[[float], float],
    *,
    z_bounds: tuple[float, float] = Z_BOUNDS,
    scan_points: int = SCAN_POINTS,
    xatol: float = Z_XATOL,
) -> SelectionRecord:
    """Scan-then-refine truth-error minimization over log10(lambda)."""
    count = 0

    def f(z: float) -> float:
        nonlocal count
        count += 1
        return float(objective(float(z)))

    zs = np.linspace(z_bounds[0], z_bounds[1], scan_points)
    values = [f(z) for z in zs]
    best_index = int(np.argmin(values))
    coarse_z = float(zs[best_index])
    candidates = [(coarse_z, values[best_index])]

    lo = float(zs[max(best_index - 1, 0)])
    hi = float(zs[min(best_index + 1, scan_points - 1)])
    if hi > lo:
        refined = minimize_scalar(
            f, bounds=(lo, hi), method="bounded", options={"xatol": xatol})
        candidates.append((float(refined.x), float(refined.fun)))

    for z_end in z_bounds:
        candidates.append((float(z_end), f(z_end)))

    best_z, best_value = min(candidates, key=lambda pair: pair[1])
    endpoint = (abs(best_z - z_bounds[0]) <= xatol
                or abs(best_z - z_bounds[1]) <= xatol)
    return SelectionRecord(
        label="oracle_continuous",
        z=best_z,
        lam=10.0 ** best_z,
        value=best_value,
        evaluations=count,
        endpoint=endpoint,
        extra={"coarse_winner_z": coarse_z},
    )


def morozov_tikhonov(
    residual: Callable[[float], float],
    target: float,
    *,
    z_bounds: tuple[float, float] = Z_BOUNDS,
    scan_points: int = SCAN_POINTS,
    rtol: float = MONOTONE_RTOL,
) -> SelectionRecord:
    """Bracketed monotone-branch discrepancy selection for Tikhonov."""
    count = 0

    def r(z: float) -> float:
        nonlocal count
        count += 1
        return float(residual(float(z)))

    zs = np.linspace(z_bounds[0], z_bounds[1], scan_points)
    curve = np.array([r(z) for z in zs])

    scale = float(np.max(np.abs(curve)))
    drops = np.diff(curve) < -rtol * max(scale, 1.0)
    if bool(np.any(drops)):
        return SelectionRecord(
            label="failed_monotonicity", z=None, lam=None, value=None,
            evaluations=count, endpoint=False,
            extra={"curve_z": zs.tolist(), "curve_r": curve.tolist()})

    if target <= curve[0]:
        z_sel, endpoint_label = float(zs[0]), True
    elif target >= curve[-1]:
        z_sel, endpoint_label = float(zs[-1]), True
    else:
        signs = curve - target
        bracket = None
        for i in range(scan_points - 1):
            if signs[i] <= 0.0 <= signs[i + 1]:
                bracket = i
                break
        if bracket is None:
            raise RuntimeError("no sign change found for an attainable target")
        z_sel = float(brentq(lambda z: r(z) - target,
                             float(zs[bracket]), float(zs[bracket + 1]),
                             xtol=Z_XATOL))
        endpoint_label = False
    return SelectionRecord(
        label="endpoint" if endpoint_label else "morozov",
        z=z_sel,
        lam=10.0 ** z_sel,
        value=r(z_sel),
        evaluations=count,
        endpoint=endpoint_label,
        extra={"target": target, "curve_z": zs.tolist(),
               "curve_r": curve.tolist()})


def residual_matched_bandwidth(
    bandwidths: Sequence[float],
    residuals: Sequence[float],
    target: float,
) -> SelectionRecord:
    """Grid residual matching for the particle bandwidth (never Morozov)."""
    h = np.asarray(bandwidths, dtype=float)
    r = np.asarray(residuals, dtype=float)
    if h.shape != r.shape or h.ndim != 1 or h.size == 0:
        raise ValueError("bandwidths and residuals must be matching 1-D grids")
    if not np.all(np.diff(h) > 0):
        raise ValueError("bandwidths must be strictly increasing")
    mismatch = np.abs(r - target)
    best = float(np.min(mismatch))
    tied = np.flatnonzero(mismatch == best)
    index = int(tied[-1])          # ties go to the larger bandwidth
    return SelectionRecord(
        label="residual_matched",
        z=None,
        lam=None,
        value=float(r[index]),
        evaluations=int(h.size),
        endpoint=index in (0, h.size - 1),
        extra={
            "selected_h": float(h[index]),
            "selected_index": index,
            "absolute_mismatch": best,
            "target": target,
            "curve_h": h.tolist(),
            "curve_r": r.tolist(),
        })
