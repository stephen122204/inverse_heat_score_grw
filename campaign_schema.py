"""campaign_schema.py — machine-readable form of PHASE2C_PROTOCOL.md.

Every preregistered grid, case, default, gate, and study row of the science
campaign is encoded here as data.  The unit tests hold this module and the
protocol document together: changing either without the other fails the
suite.  scripts/run_science_campaign.py enumerates the preregistered rows
from this schema and refuses execution while the protocol status line reads
PROPOSED.

Implementation notes carried into the campaign build (protocol references):
- Section 4/5: campaign Tikhonov output is never projected; the variable
  helper's legacy output clipping must be bypassed for campaign arms.
- Section 8: the gradient-carrier score S_{h,eps}[U] (cosine-smoothed U) and
  the MC-limited Rusanov SSPRK3 reference solver are new components with
  their own pre-use tests.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent
PROTOCOL_FILE = REPO / "PHASE2C_PROTOCOL.md"

# ---------------------------------------------------------------------------
# Global parameter grids and defaults (protocol Sections 2-6)
# ---------------------------------------------------------------------------

BANDWIDTHS = (0.005, 0.007, 0.010, 0.014, 0.020, 0.028, 0.040)
HEADLINE_H = 0.014

EPS_REL_GRID = (1e-10, 1e-8, 1e-6, 1e-4)
HEADLINE_EPS_REL = 1e-8
EPS_FLOOR_FACTOR = 100.0          # epsilon_abs >= 100 * B0(K)

ETAS = (0.0, 0.001, 0.005, 0.01)
NONZERO_ETAS = (0.001, 0.005, 0.01)
SEEDS = tuple(range(25))
ARMS = ("R", "P")
TAU_HEADLINE = 1.2
TAU_SENSITIVITY = 1.0

DEFAULT_M = 400
DEFAULT_DT = 1e-3
DEFAULT_N = 4000

ADEQUACY_CASES = ("C1", "H")
ADEQUACY_N = (1000, 2000, 4000, 8000, 10000)
ADEQUACY_REL_GATE = 0.005
ADEQUACY_MASS_GATE = 1e-12

LAMBDA_LOG_BOUNDS = (-12.0, -1.0)
LAMBDA_SCAN_POINTS = 65
LAMBDA_XATOL = 1e-6

DECRIME_FINE = (1600, 2.5e-4)
DECRIME_FINER = (3200, 1.25e-4)
DECRIME_GATE = 1e-6

CLOSURE_REFINEMENTS = ((200, 2e-3), (400, 1e-3), (800, 5e-4))
CLOSURE_H_BRIDGE = (0.020, 0.014, 0.010, 0.007)
CLOSURE_REF_RESOLUTIONS = (DECRIME_FINE, DECRIME_FINER)
CLOSURE_REF_GATE = 1e-4
CLOSURE_LAST_REDUCTION = 1.5
SPLIT_INVARIANCE_TOL = 1e-13
CLOSURES = ("frozen_left", "mass")
REFERENCE_KINDS = ("regularized", "unregularized")

# ---------------------------------------------------------------------------
# Case registry (protocol Section 1 and Section 8)
# ---------------------------------------------------------------------------

CASES = {
    "C1": {"kind": "cosine", "role": "primary", "T": 1.0, "alpha": 0.01,
           "background": 1.0, "modes": ((3, 0.5),)},
    "C2": {"kind": "cosine", "role": "primary", "T": 1.0, "alpha": 0.01,
           "background": 1.0, "modes": ((1, 0.4), (2, 0.2))},
    "B": {"kind": "gaussian", "role": "secondary", "T": 0.15, "alpha": 0.01,
          "mu": 0.4, "sigma0": 0.08, "amplitude": 1.0},
    "H": {"kind": "mixture", "role": "secondary", "T": 0.15, "alpha": 0.01,
          "background": 0.05,
          "components": ((0.75, 0.35, 0.05), (0.45, 0.62, 0.08))},
    "Z": {"kind": "gaussian", "role": "secondary", "T": 0.05, "alpha": 0.01,
          "mu": 0.4, "sigma0": 0.05, "amplitude": 1.0},
    "VB05": {"kind": "variable", "role": "variable", "T": 0.15,
             "alpha0": 0.01, "beta": 0.5, "ic": "B"},
    "VB09": {"kind": "variable", "role": "variable", "T": 0.15,
             "alpha0": 0.01, "beta": 0.9, "ic": "B"},
    "VH05": {"kind": "variable", "role": "variable", "T": 0.15,
             "alpha0": 0.01, "beta": 0.5, "ic": "H"},
    "VH09": {"kind": "variable", "role": "variable", "T": 0.15,
             "alpha0": 0.01, "beta": 0.9, "ic": "H"},
    "G1": {"kind": "closure", "role": "closure", "T": 1.0, "alpha": 0.01,
           "background": 2.0, "modes": ((1, 1.0),)},
    "G2": {"kind": "closure", "role": "closure", "T": 1.0, "alpha": 0.01,
           "background": 1.0, "modes": ((1, 0.4), (2, 0.2))},
    "GB": {"kind": "closure", "role": "closure", "T": 0.15, "alpha": 0.01,
           "mu": 0.4, "sigma0": 0.08, "amplitude": 1.0,
           "approximate_neumann": True},
}

BANDWIDTH_CASES = ("C1", "C2", "B", "H", "Z", "VB05", "VB09", "VH05", "VH09")
EPSILON_CASES = ("C1", "Z")
NOISE_CASES = ("C1", "B")
CLOSURE_CASES = ("G1", "G2", "GB")
TRANSITION_CASE = "C1"


def frozen_closure_offset(a: float, alpha: float, T: float) -> float:
    """Signed frozen-left closure error for u0 = c + a cos(pi x):
    U_frozen - u0 = -a (1 - exp(-alpha pi^2 T))  (protocol Section 8)."""
    return -a * (1.0 - math.exp(-alpha * math.pi ** 2 * T))


# ---------------------------------------------------------------------------
# Preregistered row enumeration (protocol Section 1 study matrix)
# ---------------------------------------------------------------------------

def rows_bandwidth_clean() -> list[dict]:
    return [{"study": "bandwidth_clean", "case": c, "h": h,
             "eps_rel": HEADLINE_EPS_REL, "eta": 0.0}
            for c in BANDWIDTH_CASES for h in BANDWIDTHS]


def rows_epsilon_sensitivity() -> list[dict]:
    return [{"study": "epsilon_sensitivity", "case": c, "h": HEADLINE_H,
             "eps_rel": e, "eta": 0.0}
            for c in EPSILON_CASES for e in EPS_REL_GRID]


def rows_adequacy() -> list[dict]:
    return [{"study": "adequacy_N", "case": c, "h": HEADLINE_H,
             "eps_rel": HEADLINE_EPS_REL, "N": n}
            for c in ADEQUACY_CASES for n in ADEQUACY_N]


def rows_noise_paired() -> list[dict]:
    rows = []
    for c in NOISE_CASES:
        for h in BANDWIDTHS:
            # eta = 0: arms coincide (clean data are nonnegative).
            rows.append({"study": "noise_paired", "case": c, "h": h,
                         "eta": 0.0, "seed": None, "arm": "shared"})
        for eta in NONZERO_ETAS:
            for seed in SEEDS:
                for arm in ARMS:
                    for h in BANDWIDTHS:
                        rows.append({"study": "noise_paired", "case": c,
                                     "h": h, "eta": eta, "seed": seed,
                                     "arm": arm})
    return rows


def rows_lambda_oracle_clean() -> list[dict]:
    return [{"study": "lambda_oracle_clean", "case": c,
             "selection": "oracle_continuous"} for c in BANDWIDTH_CASES]


def rows_lambda_noise() -> list[dict]:
    return [{"study": "lambda_noise", "case": c, "eta": eta, "seed": seed,
             "arm": arm, "selection": sel}
            for c in NOISE_CASES for eta in NONZERO_ETAS for seed in SEEDS
            for arm in ARMS for sel in ("oracle_continuous", "residual")]


def rows_closure() -> list[dict]:
    rows = []
    for c in CLOSURE_CASES:
        for closure in CLOSURES:
            for m, dt in CLOSURE_REFINEMENTS:
                rows.append({"study": "closure", "block": "carrier",
                             "case": c, "closure": closure, "M": m, "dt": dt})
            rows.append({"study": "closure", "block": "split_invariance",
                         "case": c, "closure": closure})
            for kind in REFERENCE_KINDS:
                for m, dt in CLOSURE_REF_RESOLUTIONS:
                    rows.append({"study": "closure", "block": "reference",
                                 "case": c, "closure": closure, "kind": kind,
                                 "M": m, "dt": dt})
        for h in CLOSURE_H_BRIDGE:
            for m, dt in CLOSURE_REF_RESOLUTIONS:
                rows.append({"study": "closure", "block": "h_bridge",
                             "case": c, "h": h, "M": m, "dt": dt})
    return rows


def rows_transition() -> list[dict]:
    return [{"study": "transition_table", "case": TRANSITION_CASE, "era": era}
            for era in ("endpoint_free_space_legacy",
                        "cell_centered_free_space_legacy",
                        "cell_centered_neumann_canonical")]


STUDIES = {
    "bandwidth_clean": rows_bandwidth_clean,
    "epsilon_sensitivity": rows_epsilon_sensitivity,
    "adequacy_N": rows_adequacy,
    "noise_paired": rows_noise_paired,
    "lambda_oracle_clean": rows_lambda_oracle_clean,
    "lambda_noise": rows_lambda_noise,
    "closure": rows_closure,
    "transition_table": rows_transition,
}


def expected_counts() -> dict[str, int]:
    return {name: len(fn()) for name, fn in STUDIES.items()}


# ---------------------------------------------------------------------------
# Freeze gate (protocol status line)
# ---------------------------------------------------------------------------

class ProtocolNotFrozen(RuntimeError):
    """Raised when campaign execution is requested before the freeze."""


def protocol_status() -> str:
    text = PROTOCOL_FILE.read_text(encoding="utf-8")
    match = re.search(r"\*\*Status:\s*([A-Z]+)", text)
    if not match:
        raise RuntimeError(f"no status line found in {PROTOCOL_FILE}")
    return match.group(1)


def assert_frozen() -> None:
    status = protocol_status()
    if status != "FROZEN":
        raise ProtocolNotFrozen(
            f"PHASE2C_PROTOCOL.md status is {status}; campaign execution is "
            "authorized only after the status line reads FROZEN in its own "
            "commit"
        )
