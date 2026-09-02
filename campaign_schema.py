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

# Amendment 1 (2026-08-31): fixed-h studies anchor per case at the first
# preregistered bandwidth strictly above the linear stability window
# h_* = sqrt(2 alpha T / (e ln(1/delta))) at the production floor
# delta = HEADLINE_EPS_REL; for the T = 1 cases that window is ~0.020.
ANCHOR_H = {"C1": 0.028, "C2": 0.028}


def anchor_h(case: str) -> float:
    """Per-case anchor bandwidth for fixed-h studies (Amendment 1)."""
    return ANCHOR_H.get(case, HEADLINE_H)


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

# Amendment 2 (2026-08-31): truth resolution raised after the 4x/8x gate
# failed pre-campaign verification (Section 7 rule 5 prescribes exactly this
# committed amendment); the gradient-reference resolutions below are stated
# independently and are unchanged.
DECRIME_FINE = (3200, 1.25e-4)
DECRIME_FINER = (6400, 6.25e-5)
DECRIME_GATE = 1e-6

CLOSURE_BANDWIDTH = 0.014          # protocol Section 8 production score h
CLOSURE_EPS_REL = 1e-8
CLOSURE_REFINEMENTS = ((200, 2e-3), (400, 1e-3), (800, 5e-4))
CLOSURE_H_BRIDGE = (0.020, 0.014, 0.010, 0.007)
CLOSURE_REF_RESOLUTIONS = ((1600, 2.5e-4), (3200, 1.25e-4))
CLOSURE_REF_GATE = 1e-4
CLOSURE_LAST_REDUCTION = 1.5
SPLIT_INVARIANCE_TOL = 1e-13
DECOMP_RECONCILE_TOL = 1e-10
CLOSURES = ("frozen_left", "mass")
REFERENCE_KINDS = ("regularized", "unregularized")

# Amendment 3: initial-rate diagnostic (Theorem R1; ledger 65-69).
INITIAL_RATE_CASE = "G1"
INITIAL_RATE_CLOSURE = "mass"
INITIAL_RATE_TAU = 0.005                    # inside the certified [0, 0.0074]
INITIAL_RATE_M_CERT = 0.1812                # Corollary R1b second-derivative bound
INITIAL_RATE_GATE = 0.0074                  # (M/2) tau / c_rep, outward-rounded
INITIAL_RATE_PAIR_GATE = 1e-4               # |e_fine - e_finer| / (c_rep tau)
INITIAL_RATE_REFERENCES = ((6400, 6.25e-5), (12800, 3.125e-5))
INITIAL_RATE_CARRIERS = ((200, 1e-3), (400, 5e-4), (800, 2.5e-4))
INITIAL_RATE_Q_TAUS = (0.0125, 0.025, 0.05, 0.1, 0.2, 0.4)
INITIAL_RATE_Q_RESOLUTION = (3200, 1.25e-4)

# Amendment 4: crossover phenomenology block (evidence-grade; ledger 62-64).
CROSSOVER_MODE = 3                          # u0 = 1 + a cos(3 pi x)
CROSSOVER_ALPHA = 0.01
CROSSOVER_T = 1.0
CROSSOVER_EPS_REL = 1e-8
CROSSOVER_A = (0.35, 0.50, 0.55, 0.60)
CROSSOVER_KH = (0.23, 0.264, 0.29)
CROSSOVER_PARTICLE_POINTS = ((0.35, 0.264), (0.60, 0.264))
CROSSOVER_N = (4000, 10000)
CROSSOVER_CONTINUUM = (1024, 150, 5e-4)     # cells, modes, dt (clean at h >= 0.024)

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
# Document binding: needles generated from the schema constants, so changing
# a constant here without amending the protocol document (or vice versa)
# fails the binding test.
# ---------------------------------------------------------------------------

def _pow10(v: float) -> str:
    e = round(math.log10(v))
    if abs(v - 10.0 ** e) > 1e-12 * v:
        raise ValueError(f"{v!r} is not a power of ten")
    return f"1e{e}"


def _step(dt: float) -> str:
    return re.sub(r"e([+-])0(\d)", r"e\g<1>\g<2>", f"{dt:.0e}")


def document_binding_needles() -> tuple[str, ...]:
    return (
        ", ".join(f"{h:.3f}" for h in BANDWIDTHS),
        ", ".join(str(n) for n in ADEQUACY_N),
        ", ".join(_pow10(e) for e in EPS_REL_GRID),
        f"{LAMBDA_SCAN_POINTS}-point",
        f"N = {DEFAULT_N}",
        f"**{ADEQUACY_CASES[0]}**, the exact bounded-domain primary, "
        f"and **{ADEQUACY_CASES[1]}**",
        f"|E_4000-E_10000|/E_10000 <= {ADEQUACY_REL_GATE:g}",
        f"at most `{_pow10(ADEQUACY_MASS_GATE)}` relative to `M0`",
        ", ".join(f"({m}, {_step(dt)})" for m, dt in CLOSURE_REFINEMENTS),
        "{" + ", ".join(f"{h:.3f}" for h in CLOSURE_H_BRIDGE) + "}",
        f"`tau = {TAU_HEADLINE:.1f}`",
        f"`tau = {TAU_SENSITIVITY:.1f}`",
        "three preregistered noisy-lambda selections",
        f"at most `{_pow10(DECRIME_GATE)}`",
        f"at most `{_pow10(CLOSURE_REF_GATE)}`",
        f"at least `{CLOSURE_LAST_REDUCTION:g}`",
        f"`{_pow10(SPLIT_INVARIANCE_TOL)}` relative",
        f"`epsilon_abs >= {EPS_FLOOR_FACTOR:g} B0`",
        f"reconcile to `{_pow10(DECOMP_RECONCILE_TOL)}` relative",
        "Rusanov",
        "SSPRK3",
        "monotonized-central",
        f"between the `{DECRIME_FINE[0] // DEFAULT_M}x` and "
        f"`{DECRIME_FINER[0] // DEFAULT_M}x` projected data",
        f"each reference at `M = {CLOSURE_REF_RESOLUTIONS[0][0]}`",
        f"closure comparison uses `h = {CLOSURE_BANDWIDTH:.3f}`, "
        f"`epsilon_rel = {_pow10(CLOSURE_EPS_REL)}`",
        f"(`h = {anchor_h('C1'):.3f}` for C1, `h = {anchor_h('Z'):.3f}` for Z)",
        f"(`h = {anchor_h('C1'):.3f}` for C1, `h = {anchor_h('H'):.3f}` for H)",
        f"`h = {anchor_h('C1'):.3f}` for the `T = 1` cases C1 and C2",
        f"`tau = {INITIAL_RATE_TAU}`",
        f"c_rep = {INITIAL_RATE_GATE}`",
        f"c_rep tau) <= {_pow10(INITIAL_RATE_PAIR_GATE)}`",
        "reference pair `" + ", ".join(f"({m}, {dt:g})"
                                      for m, dt in INITIAL_RATE_REFERENCES) + "`",
        "carrier ladder `" + ", ".join(f"({m}, {dt:g})"
                                      for m, dt in INITIAL_RATE_CARRIERS) + "`",
        f"`M = {INITIAL_RATE_Q_RESOLUTION[0]}, dt = "
        f"{INITIAL_RATE_Q_RESOLUTION[1]:g}`",
        "`a in {" + ", ".join(f"{a:.2f}" for a in CROSSOVER_A) + "}`",
        "`kh in {" + ", ".join(str(k) for k in CROSSOVER_KH) + "}`",
        "`N in {" + ", ".join(str(n) for n in CROSSOVER_N) + "}`",
        f"`{CROSSOVER_CONTINUUM[0]}` collocation cells, `{CROSSOVER_CONTINUUM[1]}` "
        f"modes, `dt = {CROSSOVER_CONTINUUM[2]:g}`",
    ) + tuple(f"**Arm {arm}" for arm in ARMS)


# ---------------------------------------------------------------------------
# Preregistered row enumeration (protocol Section 1 study matrix)
# ---------------------------------------------------------------------------

def rows_bandwidth_clean() -> list[dict]:
    return [{"study": "bandwidth_clean", "case": c, "h": h,
             "eps_rel": HEADLINE_EPS_REL, "eta": 0.0}
            for c in BANDWIDTH_CASES for h in BANDWIDTHS]


def rows_epsilon_sensitivity() -> list[dict]:
    return [{"study": "epsilon_sensitivity", "case": c, "h": anchor_h(c),
             "eps_rel": e, "eta": 0.0}
            for c in EPSILON_CASES for e in EPS_REL_GRID]


def rows_adequacy() -> list[dict]:
    return [{"study": "adequacy_N", "case": c, "h": anchor_h(c),
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
    """Three preregistered selections per nonzero-noise block: the oracle
    continuous diagnostic and the residual selections at both tau values."""
    rows = []
    for c in NOISE_CASES:
        for eta in NONZERO_ETAS:
            for seed in SEEDS:
                for arm in ARMS:
                    rows.append({"study": "lambda_noise", "case": c,
                                 "eta": eta, "seed": seed, "arm": arm,
                                 "selection": "oracle_continuous",
                                 "tau": None})
                    for tau in (TAU_HEADLINE, TAU_SENSITIVITY):
                        rows.append({"study": "lambda_noise", "case": c,
                                     "eta": eta, "seed": seed, "arm": arm,
                                     "selection": "residual", "tau": tau})
    return rows


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
        # The h-bridge compares the closure-specific regularized and
        # unregularized references, so it carries the closure dimension.
        for closure in CLOSURES:
            for h in CLOSURE_H_BRIDGE:
                for m, dt in CLOSURE_REF_RESOLUTIONS:
                    rows.append({"study": "closure", "block": "h_bridge",
                                 "case": c, "closure": closure, "h": h,
                                 "M": m, "dt": dt})
    return rows


def rows_initial_rate() -> list[dict]:
    """Amendment 3: Block A theorem-backed continuum references, Block B
    evidence-grade carriers, Block C retained q-level table."""
    rows = []
    for m, dt in INITIAL_RATE_REFERENCES:
        rows.append({"study": "initial_rate", "block": "reference",
                     "case": INITIAL_RATE_CASE, "closure": INITIAL_RATE_CLOSURE,
                     "M": m, "dt": dt, "tau": INITIAL_RATE_TAU})
    for m, dt in INITIAL_RATE_CARRIERS:
        rows.append({"study": "initial_rate", "block": "carrier",
                     "case": INITIAL_RATE_CASE, "closure": INITIAL_RATE_CLOSURE,
                     "M": m, "dt": dt, "tau": INITIAL_RATE_TAU})
    m, dt = INITIAL_RATE_Q_RESOLUTION
    for tau in INITIAL_RATE_Q_TAUS:
        rows.append({"study": "initial_rate", "block": "q_level",
                     "case": INITIAL_RATE_CASE, "closure": INITIAL_RATE_CLOSURE,
                     "M": m, "dt": dt, "tau": tau})
    return rows


def rows_crossover() -> list[dict]:
    """Amendment 4: prespecified continuum points and particle tracking."""
    rows = []
    for a in CROSSOVER_A:
        for kh in CROSSOVER_KH:
            rows.append({"study": "crossover", "block": "continuum",
                         "a": a, "kh": kh, "N": None})
    for a, kh in CROSSOVER_PARTICLE_POINTS:
        for n in CROSSOVER_N:
            rows.append({"study": "crossover", "block": "particle",
                         "a": a, "kh": kh, "N": n})
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
    "initial_rate": rows_initial_rate,
    "crossover": rows_crossover,
}


def expected_counts() -> dict[str, int]:
    return {name: len(fn()) for name, fn in STUDIES.items()}


# ---------------------------------------------------------------------------
# Freeze gate (protocol status line)
# ---------------------------------------------------------------------------

class ProtocolNotFrozen(RuntimeError):
    """Raised when campaign execution is requested before the freeze."""


CHECKLIST_HEADER = "## Freeze checklist"

EXPECTED_CHECKLIST = (
    "Primary/secondary case roles and analytic formulas independently checked.",
    "Fixed discretization and `N = 4000` adequacy gate accepted.",
    "Physical bandwidth grid, endpoint censoring, and noiseless rule accepted.",
    "Score-floor diagnostics and fixed headline epsilon accepted.",
    "Input-only positivity arms and residual normalization accepted.",
    "Continuous-lambda scan/refinement and boundary rules accepted.",
    "Paired noise seeds, selection labels, and paired summaries accepted.",
    "De-crimed variable-data projection and convergence gate accepted.",
    "Closure definitions, wrong-limit reference, and exact field decomposition accepted.",
    "Canonical estimator/self-interaction and transition-table scope accepted.",
    "Row accounting, acceptance gates, provenance, and amendment rules accepted.",
    "Status changed from PROPOSED to FROZEN in a dedicated commit.",
)


def status_of(text: str) -> str:
    match = re.search(r"\*\*Status:\s*([A-Z]+)", text)
    if not match:
        raise RuntimeError("no protocol status line found")
    return match.group(1)


def checklist_items(text: str) -> list[tuple[bool, str]]:
    """(checked, label) for every checklist row, in document order."""
    parts = text.split(CHECKLIST_HEADER, 1)
    if len(parts) < 2:
        raise RuntimeError("no freeze checklist section found")
    items = []
    for line in parts[1].splitlines():
        match = re.match(r"- \[( |x)\] (.+)$", line.strip())
        if match:
            items.append((match.group(1) == "x", match.group(2).strip()))
    return items


def unchecked_checklist_items(text: str) -> list[str]:
    return [label for checked, label in checklist_items(text) if not checked]


def validate_freeze(text: str) -> None:
    """Raise ProtocolNotFrozen unless the document authorizes execution:
    FROZEN status, exactly the expected checklist labels in order, and every
    box checked.  A deleted, reworded, or malformed row is a refusal, not a
    bypass."""
    problems = []
    status = status_of(text)
    if status != "FROZEN":
        problems.append(f"status is {status}")
    items = checklist_items(text)
    labels = tuple(label for _, label in items)
    if labels != EXPECTED_CHECKLIST:
        problems.append(
            f"freeze checklist does not match the {len(EXPECTED_CHECKLIST)} "
            f"expected items (found {len(labels)})"
        )
    unchecked = [label for checked, label in items if not checked]
    if unchecked:
        problems.append(f"{len(unchecked)} freeze-checklist boxes are unchecked")
    if problems:
        raise ProtocolNotFrozen(
            "PHASE2C_PROTOCOL.md: " + "; ".join(problems) + ".  Campaign "
            "execution is authorized only when the status line reads FROZEN "
            "in its own commit and every expected checklist box is checked."
        )


def protocol_status() -> str:
    return status_of(PROTOCOL_FILE.read_text(encoding="utf-8"))


def assert_frozen() -> None:
    validate_freeze(PROTOCOL_FILE.read_text(encoding="utf-8"))
