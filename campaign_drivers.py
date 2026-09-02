"""campaign_drivers.py — study drivers for the Phase 2C campaign.

Consumes the preregistered rows of campaign_schema, executes them through the
canonical src components, and writes outcomes through the campaign_results
contract. Every driver supports resume (rows already present in the study CSV
are skipped), retains failures as rows, and finishes by reconciling the
Section 10 accounting and writing summary.json. The orchestrator wraps each
completed study directory in a hash-validated manifest.

Drivers implemented here: bandwidth_clean, epsilon_sensitivity, adequacy_N,
lambda_oracle_clean, noise_paired, lambda_noise, closure. The transition
driver lives in campaign_driver_transition.py because it is the only study
allowed to touch legacy code paths.

Closure dossier declarations (protocol Section 8): the archived comparison
times are reverse times T/2 and T. Both reference resolutions and the
M = 400 carrier row represent T/2 exactly; the M = 200 and M = 800 carrier
rows are gated on the final time only, which is all the refinement clause
uses. The frozen-left reconstruction anchor is the terminal datum value at
the first cell center of the row's own grid, g(x_min + dx/2).
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import campaign_results as results
import campaign_schema as schema
from invheat_grw.baselines import tikhonov_inverse
from invheat_grw.campaign_closure import (
    closure_constant,
    exact_decomposition,
    frozen_left_offset,
    reconstruct_centers,
    run_gradient_carriers,
    run_reference,
)
from invheat_grw.campaign_data import (
    apply_noise,
    nominal_delta,
    projection_stats,
    real_delta,
    cell_edges,
    cn_forward_solve,
    cosine_field,
    decrimed_variable_data,
    gaussian_cell_averages,
    gaussian_free_space,
    mixture_cell_averages,
    mixture_free_space,
    variable_diffusivity,
)
from invheat_grw.campaign_particle import run_campaign_density
from invheat_grw.campaign_selectors import (
    morozov_tikhonov,
    oracle_continuous,
    residual_matched_bandwidth,
)
from provenance import git_commit
from invheat_grw.cell_grid import (
    cell_centers,
    cell_spacing,
    midpoint_norm,
    propagate_heat,
)

X_MIN, X_MAX = 0.0, 1.0
LENGTH = X_MAX - X_MIN


class CampaignGateError(RuntimeError):
    """Raised when a preregistered acceptance gate fails during data setup."""


# ---------------------------------------------------------------------------
# Case construction
# ---------------------------------------------------------------------------

@dataclass
class CaseSetup:
    name: str
    T: float
    terminal_clean: np.ndarray
    truth: np.ndarray
    alpha: float | None
    a_of_x: Callable[[np.ndarray], np.ndarray] | None
    decrime: dict | None

    def residual_fn(self, datum: np.ndarray,
                    m: int = schema.DEFAULT_M) -> Callable[[np.ndarray], float]:
        dx = cell_spacing(X_MIN, X_MAX, m)
        d = np.asarray(datum, dtype=float)

        def relative_residual(uhat: np.ndarray) -> float:
            if self.a_of_x is None:
                forward = propagate_heat(uhat, LENGTH, self.alpha, self.T)
            else:
                forward = cn_forward_solve(uhat, self.a_of_x, self.T,
                                           schema.DEFAULT_DT,
                                           x_min=X_MIN, x_max=X_MAX)
            return midpoint_norm(forward - d, dx) / midpoint_norm(d, dx)

        return relative_residual


@lru_cache(maxsize=None)
def build_case(name: str, m: int = schema.DEFAULT_M) -> CaseSetup:
    spec = schema.CASES[name]
    x = cell_centers(X_MIN, X_MAX, m)
    kind = spec["kind"]
    if kind in ("cosine", "closure") and "modes" in spec:
        u0 = cosine_field(x, spec["background"], spec["modes"], length=LENGTH)
        g = cosine_field(x, spec["background"], spec["modes"], length=LENGTH,
                         alpha=spec["alpha"], t=spec["T"])
        return CaseSetup(name, spec["T"], g, u0, spec["alpha"], None, None)
    if kind == "gaussian" or (kind == "closure" and "sigma0" in spec):
        u0 = gaussian_free_space(x, spec["mu"], spec["sigma0"],
                                 spec["amplitude"])
        g = gaussian_free_space(x, spec["mu"], spec["sigma0"],
                                spec["amplitude"], alpha=spec["alpha"],
                                t=spec["T"])
        return CaseSetup(name, spec["T"], g, u0, spec["alpha"], None, None)
    if kind == "mixture":
        u0 = mixture_free_space(x, spec["background"], spec["components"])
        g = mixture_free_space(x, spec["background"], spec["components"],
                               alpha=spec["alpha"], t=spec["T"])
        return CaseSetup(name, spec["T"], g, u0, spec["alpha"], None, None)
    if kind == "variable":
        ic_spec = schema.CASES[spec["ic"]]
        if ic_spec["kind"] == "gaussian":
            def ic_fn(edges, s=ic_spec):
                return gaussian_cell_averages(edges, s["mu"], s["sigma0"],
                                              s["amplitude"])
        else:
            def ic_fn(edges, s=ic_spec):
                return mixture_cell_averages(edges, s["background"],
                                             s["components"])
        a_of_x = variable_diffusivity(spec["alpha0"], spec["beta"])
        data = decrimed_variable_data(
            ic_fn, a_of_x, spec["T"], x_min=X_MIN, x_max=X_MAX, m_inverse=m,
            fine=schema.DECRIME_FINE, finer=schema.DECRIME_FINER,
            gate=schema.DECRIME_GATE)
        if not data.passes_gate:
            raise CampaignGateError(
                f"variable-data gate failed for {name}: "
                f"{data.gate_difference:.3e} > {schema.DECRIME_GATE:g}")
        decrime = {
            "gate_difference": data.gate_difference,
            "mass_discrepancy": data.mass_discrepancy,
            "fine": schema.DECRIME_FINE,
            "finer": schema.DECRIME_FINER,
        }
        return CaseSetup(name, spec["T"], data.terminal,
                         data.truth_cell_averages, None, a_of_x, decrime)
    raise CampaignGateError(f"unsupported case specification for {name!r}")


# ---------------------------------------------------------------------------
# Tikhonov tools (spectral for constant cases, SVD matrix for variable cases)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def variable_forward_matrix(name: str, m: int = schema.DEFAULT_M) -> tuple:
    setup = build_case(name, m)
    if setup.a_of_x is None:
        raise CampaignGateError(f"{name} is not a variable case")
    columns = []
    for j in range(m):
        unit = np.zeros(m)
        unit[j] = 1.0
        columns.append(cn_forward_solve(unit, setup.a_of_x, setup.T,
                                        schema.DEFAULT_DT,
                                        x_min=X_MIN, x_max=X_MAX))
    forward = np.column_stack(columns)
    u_svd, s_svd, vt_svd = np.linalg.svd(forward)
    return forward, u_svd, s_svd, vt_svd


@dataclass
class TikhonovTools:
    solve: Callable[[np.ndarray, float], np.ndarray]
    residual: Callable[[np.ndarray, float], float]


def tikhonov_tools(name: str, m: int = schema.DEFAULT_M) -> TikhonovTools:
    setup = build_case(name, m)
    x = cell_centers(X_MIN, X_MAX, m)
    dx = cell_spacing(X_MIN, X_MAX, m)
    if setup.a_of_x is None:
        def solve(d: np.ndarray, lam: float) -> np.ndarray:
            return tikhonov_inverse(d, x, setup.alpha, setup.T, lam,
                                    length=LENGTH).candidate

        def residual(d: np.ndarray, lam: float) -> float:
            forward = propagate_heat(solve(d, lam), LENGTH, setup.alpha,
                                     setup.T)
            return midpoint_norm(forward - d, dx) / midpoint_norm(d, dx)
    else:
        forward, u_svd, s_svd, vt_svd = variable_forward_matrix(name, m)

        def solve(d: np.ndarray, lam: float) -> np.ndarray:
            coeff = u_svd.T @ d
            filt = s_svd / (s_svd ** 2 + lam)
            return vt_svd.T @ (filt * coeff)

        def residual(d: np.ndarray, lam: float) -> float:
            fwd = forward @ solve(d, lam)
            return midpoint_norm(fwd - d, dx) / midpoint_norm(d, dx)
    return TikhonovTools(solve, residual)


def truth_error_objective(name: str, datum: np.ndarray,
                          m: int = schema.DEFAULT_M) -> Callable[[float], float]:
    setup = build_case(name, m)
    tools = tikhonov_tools(name, m)
    dx = cell_spacing(X_MIN, X_MAX, m)
    truth_norm = midpoint_norm(setup.truth, dx)

    def objective(z: float) -> float:
        return midpoint_norm(tools.solve(datum, 10.0 ** z) - setup.truth,
                             dx) / truth_norm

    return objective


# ---------------------------------------------------------------------------
# Particle payload contract (one fixed field set for every particle study)
# ---------------------------------------------------------------------------

NAN = float("nan")

PARTICLE_PAYLOAD_KEYS = (
    "E2", "Linf_rel", "forward_residual", "mass_error_rel",
    "mass_reconstruction",
    "min_reconstruction", "tv_reconstruction", "tv_reference",
    "K_modes", "K_tightened", "B0", "B1", "eps_abs", "min_raw_density",
    "min_regularized_denominator", "max_abs_score", "tightened_max_change",
    "gate_eps_floor", "gate_tightened", "delta_nom", "delta_real",
    "negative_fraction", "negative_mass_removed", "mass_change_rel",
)


def particle_payload(result, *, delta_nom: float = NAN,
                     delta_real: float = NAN,
                     projection: dict | None = None) -> dict:
    metrics = result.metrics
    diag = result.diagnostics
    tight = diag.get("tightened_score_change") or {}
    payload = {
        "E2": metrics.get("E2", NAN),
        "Linf_rel": metrics.get("Linf_rel", NAN),
        "forward_residual": metrics.get("forward_residual", NAN),
        "mass_error_rel": metrics.get("mass_error_rel", NAN),
        "mass_reconstruction": metrics.get("mass_reconstruction", NAN),
        "min_reconstruction": metrics.get("min_reconstruction", NAN),
        "tv_reconstruction": metrics.get("tv_reconstruction", NAN),
        "tv_reference": metrics.get("tv_reference", NAN),
        "K_modes": diag.get("n_modes", NAN),
        "K_tightened": diag.get("n_modes_tightened", NAN),
        "B0": diag.get("B0", NAN),
        "B1": diag.get("B1", NAN),
        "eps_abs": diag.get("eps_abs", NAN),
        "min_raw_density": diag.get("min_raw_density"),
        "min_regularized_denominator": diag.get("min_regularized_denominator"),
        "max_abs_score": diag.get("max_abs_score", NAN),
        "tightened_max_change": max(tight.values()) if tight else NAN,
        "gate_eps_floor": result.gates.get("eps_floor"),
        "gate_tightened": result.gates.get("tightened_score"),
        "delta_nom": delta_nom,
        "delta_real": delta_real,
        "negative_fraction": (projection or {}).get("negative_fraction", NAN),
        "negative_mass_removed": (projection or {}).get(
            "negative_mass_removed", NAN),
        "mass_change_rel": (projection or {}).get("mass_change_rel", NAN),
    }
    assert tuple(sorted(payload)) == tuple(sorted(PARTICLE_PAYLOAD_KEYS))
    return payload


def _run_particle_row(row: dict, *, n_particles: int) -> tuple:
    setup = build_case(row["case"])
    result = run_campaign_density(
        setup.terminal_clean,
        x_min=X_MIN, x_max=X_MAX, T=setup.T, dt=schema.DEFAULT_DT,
        n_particles=n_particles, bandwidth=row["h"], eps_rel=row["eps_rel"],
        alpha=setup.alpha, a_of_x=setup.a_of_x, u0_reference=setup.truth,
        forward_residual=setup.residual_fn(setup.terminal_clean))
    return result, particle_payload(result)


def _append_result(writer: results.StudyWriter, row: dict, result,
                   payload: dict, *, censored: bool = False) -> None:
    writer.append(
        row,
        results.STATUS_COMPLETED if result.status == "completed"
        else results.STATUS_FAILED,
        payload,
        censored=censored,
        failure_step=result.failure_step,
        failure_message=result.failure_message,
    )


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------

def drive_bandwidth_clean(out_dir: Path,
                          rows: Sequence[dict] | None = None) -> results.Accounting:
    writer = results.StudyWriter("bandwidth_clean", Path(out_dir))
    for row in rows or results.enumerate_rows("bandwidth_clean"):
        if results.study_row_key(writer.study, row) in writer.done_keys:
            continue
        result, payload = _run_particle_row(row,
                                            n_particles=schema.DEFAULT_N)
        _append_result(writer, row, result, payload)
    accounting = results.reconcile("bandwidth_clean", writer.csv_path)
    results.write_summary(Path(out_dir), accounting,
                          {"accounting_consistent": accounting.consistent})
    return accounting


def drive_epsilon_sensitivity(out_dir: Path,
                              rows: Sequence[dict] | None = None
                              ) -> results.Accounting:
    writer = results.StudyWriter("epsilon_sensitivity", Path(out_dir))
    for row in rows or results.enumerate_rows("epsilon_sensitivity"):
        if results.study_row_key(writer.study, row) in writer.done_keys:
            continue
        result, payload = _run_particle_row(row,
                                            n_particles=schema.DEFAULT_N)
        _append_result(writer, row, result, payload)
    accounting = results.reconcile("epsilon_sensitivity", writer.csv_path)
    results.write_summary(Path(out_dir), accounting,
                          {"accounting_consistent": accounting.consistent})
    return accounting


def drive_adequacy(out_dir: Path,
                   rows: Sequence[dict] | None = None) -> results.Accounting:
    writer = results.StudyWriter("adequacy_N", Path(out_dir))
    for row in rows or results.enumerate_rows("adequacy_N"):
        if results.study_row_key(writer.study, row) in writer.done_keys:
            continue
        result, payload = _run_particle_row(row, n_particles=row["N"])
        _append_result(writer, row, result, payload)
    accounting = results.reconcile("adequacy_N", writer.csv_path)
    verdicts: dict[str, bool] = {"accounting_consistent": accounting.consistent}
    table = _read_rows(writer.csv_path)
    for case in schema.ADEQUACY_CASES:
        values = {int(r["N"]): (float(r["E2"]), float(r["mass_reconstruction"]))
                  for r in table if r["case"] == case
                  and r["status"] == results.STATUS_COMPLETED
                  and "E2" in r and "mass_reconstruction" in r}
        if 4000 in values and 10000 in values:
            e4, m4 = values[4000]
            e10, m10 = values[10000]
            verdicts[f"adequacy_{case}"] = (
                abs(e4 - e10) / e10 <= schema.ADEQUACY_REL_GATE)
            setup = build_case(case)
            dx = cell_spacing(X_MIN, X_MAX, schema.DEFAULT_M)
            m0 = float(dx * np.sum(np.maximum(setup.terminal_clean, 0.0)))
            verdicts[f"adequacy_mass_{case}"] = (
                abs(m4 - m10) / m0 <= schema.ADEQUACY_MASS_GATE)
    results.write_summary(Path(out_dir), accounting, verdicts)
    return accounting


def drive_lambda_oracle_clean(out_dir: Path,
                              rows: Sequence[dict] | None = None
                              ) -> results.Accounting:
    writer = results.StudyWriter("lambda_oracle_clean", Path(out_dir))
    for row in rows or results.enumerate_rows("lambda_oracle_clean"):
        if results.study_row_key(writer.study, row) in writer.done_keys:
            continue
        setup = build_case(row["case"])
        record = oracle_continuous(
            truth_error_objective(row["case"], setup.terminal_clean))
        tools = tikhonov_tools(row["case"])
        payload = {
            "selection_label": record.label,
            "z_selected": record.z,
            "lambda_selected": record.lam,
            "E2_at_selection": record.value,
            "residual_at_selection": tools.residual(setup.terminal_clean,
                                                    record.lam),
            "evaluations": record.evaluations,
            "coarse_winner_z": record.extra.get("coarse_winner_z", NAN),
        }
        writer.append(row, results.STATUS_COMPLETED, payload,
                      censored=record.endpoint)
    accounting = results.reconcile("lambda_oracle_clean", writer.csv_path)
    results.write_summary(Path(out_dir), accounting,
                          {"accounting_consistent": accounting.consistent})
    return accounting


def _read_rows(csv_path: Path) -> list[dict]:
    import csv as _csv
    if not Path(csv_path).exists():
        return []
    with Path(csv_path).open(newline="", encoding="utf-8") as stream:
        return list(_csv.DictReader(stream))


# ---------------------------------------------------------------------------
# Paired noise blocks (protocol Sections 4 and 6)
# ---------------------------------------------------------------------------

ZERO_PROJECTION = {"negative_fraction": 0.0, "negative_mass_removed": 0.0,
                   "mass_change_rel": 0.0}


@dataclass
class NoiseBlock:
    d_raw: np.ndarray
    d_projected: np.ndarray
    projection: dict
    delta_nom: dict
    delta_real: dict

    def datum(self, arm: str) -> np.ndarray:
        if arm not in ("R", "P"):
            raise CampaignGateError(f"unknown positivity arm {arm!r}")
        return self.d_raw if arm == "R" else self.d_projected


@lru_cache(maxsize=None)
def noise_block(case: str, eta: float, seed: int,
                m: int = schema.DEFAULT_M) -> NoiseBlock:
    setup = build_case(case, m)
    dx = cell_spacing(X_MIN, X_MAX, m)
    xi = results.standard_normal_vector(seed, m)
    d_raw = apply_noise(setup.terminal_clean, eta, xi)
    stats = projection_stats(d_raw, dx)
    d_projected = stats.pop("projected")
    arms = {"R": d_raw, "P": d_projected}
    return NoiseBlock(
        d_raw, d_projected, stats,
        {a: nominal_delta(eta, setup.terminal_clean, d, dx, LENGTH)
         for a, d in arms.items()},
        {a: real_delta(setup.terminal_clean, d, dx)
         for a, d in arms.items()})


def bandwidth_selections(csv_path: Path) -> tuple[list[dict], int]:
    """Residual-matched bandwidth selections per completed nonzero-noise
    block, at both preregistered tau values (protocol Section 6)."""
    blocks: dict[tuple, dict[float, dict]] = {}
    for r in _read_rows(csv_path):
        if r["arm"] == "shared" or r["status"] != results.STATUS_COMPLETED:
            continue
        key = (r["case"], float(r["eta"]), int(r["seed"]), r["arm"])
        blocks.setdefault(key, {})[float(r["h"])] = r
    selections: list[dict] = []
    incomplete = 0
    for (case, eta, seed, arm), by_h in sorted(blocks.items()):
        if len(by_h) != len(schema.BANDWIDTHS):
            incomplete += 1
            continue
        grid = sorted(by_h)
        curve = [float(by_h[h]["forward_residual"]) for h in grid]
        delta_nom = float(by_h[grid[0]]["delta_nom"])
        for tau in (schema.TAU_HEADLINE, schema.TAU_SENSITIVITY):
            record = residual_matched_bandwidth(grid, curve, tau * delta_nom)
            selections.append({
                "case": case, "eta": eta, "seed": seed, "arm": arm,
                "tau": tau, "label": record.label,
                "endpoint_censored": record.endpoint,
                "residual_at_selection": record.value,
                **record.extra,
            })
    return selections, incomplete


def drive_noise_paired(out_dir: Path,
                       rows: Sequence[dict] | None = None
                       ) -> results.Accounting:
    writer = results.StudyWriter("noise_paired", Path(out_dir))
    for row in rows or results.enumerate_rows("noise_paired"):
        if results.study_row_key(writer.study, row) in writer.done_keys:
            continue
        setup = build_case(row["case"])
        if row["arm"] == "shared":
            d = setup.terminal_clean
            delta_n, delta_r, projection = 0.0, 0.0, ZERO_PROJECTION
        else:
            block = noise_block(row["case"], row["eta"], row["seed"])
            d = block.datum(row["arm"])
            delta_n = block.delta_nom[row["arm"]]
            delta_r = block.delta_real[row["arm"]]
            projection = block.projection
        result = run_campaign_density(
            d, x_min=X_MIN, x_max=X_MAX, T=setup.T, dt=schema.DEFAULT_DT,
            n_particles=schema.DEFAULT_N, bandwidth=row["h"],
            eps_rel=schema.HEADLINE_EPS_REL, alpha=setup.alpha,
            a_of_x=setup.a_of_x, u0_reference=setup.truth,
            forward_residual=setup.residual_fn(d))
        _append_result(writer, row, result,
                       particle_payload(result, delta_nom=delta_n,
                                        delta_real=delta_r,
                                        projection=projection))
    accounting = results.reconcile("noise_paired", writer.csv_path)
    selections, incomplete = bandwidth_selections(writer.csv_path)
    (Path(out_dir) / "bandwidth_selection.json").write_text(
        json.dumps({"commit": git_commit(REPO), "selections": selections},
                   indent=1) + "\n", encoding="utf-8")
    results.write_summary(
        Path(out_dir), accounting,
        {"accounting_consistent": accounting.consistent},
        extra={"bandwidth_selection_blocks": len(selections),
               "incomplete_selection_blocks": incomplete})
    return accounting


LAMBDA_NOISE_PAYLOAD_KEYS = (
    "selection_label", "z_selected", "lambda_selected", "E2_at_selection",
    "residual_at_selection", "evaluations", "coarse_winner_z",
    "target_residual", "delta_nom", "delta_real",
    "E2_tikhonov_raw", "E2_tikhonov_projected",
)


def project_positive_mass(u: np.ndarray, total_mass: float, dx: float
                          ) -> np.ndarray:
    """Exact L2 metric projection onto {u >= 0, dx * sum(u) = total_mass}:
    the water-filling map u -> max(u - mu, 0) with the multiplier mu solving
    the mass constraint (NOT clip-and-rescale, which is not the projection).
    Amendment 4 comparator."""
    u = np.asarray(u, dtype=float)
    if total_mass <= 0.0:
        raise CampaignGateError("projection needs a positive target mass")

    def mass(mu: float) -> float:
        return float(dx * np.sum(np.maximum(u - mu, 0.0)))

    lo = float(u.min()) - total_mass / (dx * u.size) - 1.0   # mass(lo) > M0
    hi = float(u.max())                                        # mass(hi) = 0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mass(mid) > total_mass:
            lo = mid
        else:
            hi = mid
    return np.maximum(u - 0.5 * (lo + hi), 0.0)


def drive_lambda_noise(out_dir: Path,
                       rows: Sequence[dict] | None = None
                       ) -> results.Accounting:
    writer = results.StudyWriter("lambda_noise", Path(out_dir))
    monotonicity_failures = 0
    for row in rows or results.enumerate_rows("lambda_noise"):
        if results.study_row_key(writer.study, row) in writer.done_keys:
            continue
        block = noise_block(row["case"], row["eta"], row["seed"])
        d = block.datum(row["arm"])
        delta_n = block.delta_nom[row["arm"]]
        delta_r = block.delta_real[row["arm"]]
        tools = tikhonov_tools(row["case"])
        objective = truth_error_objective(row["case"], d)
        if row["selection"] == "oracle_continuous":
            record = oracle_continuous(objective)
            e2 = record.value
            resid = tools.residual(d, record.lam)
            target = NAN
            coarse = record.extra.get("coarse_winner_z", NAN)
        else:
            target = row["tau"] * delta_n
            record = morozov_tikhonov(
                lambda z: tools.residual(d, 10.0 ** z), target)
            coarse = NAN
            if record.label == "failed_monotonicity":
                monotonicity_failures += 1
                writer.append(
                    row, results.STATUS_FAILED,
                    {"selection_label": record.label, "z_selected": None,
                     "lambda_selected": None, "E2_at_selection": NAN,
                     "residual_at_selection": NAN,
                     "evaluations": record.evaluations,
                     "coarse_winner_z": NAN, "target_residual": target,
                     "delta_nom": delta_n, "delta_real": delta_r,
                     "E2_tikhonov_raw": NAN, "E2_tikhonov_projected": NAN},
                    failure_message="Tikhonov residual monotonicity violated")
                continue
            e2 = objective(record.z)
            resid = record.value
        # Amendment 4 comparator: the selected Tikhonov field after the exact
        # metric projection onto the positive fixed-mass set, with the mass
        # the density method itself uses for this arm (clipped datum mass).
        setup = build_case(row["case"])
        dx_c = cell_spacing(X_MIN, X_MAX, schema.DEFAULT_M)
        u_sel = tools.solve(d, record.lam)
        m0 = float(dx_c * np.sum(np.maximum(d, 0.0)))
        u_proj = project_positive_mass(u_sel, m0, dx_c)
        e2_proj = float(midpoint_norm(u_proj - setup.truth, dx_c)
                        / midpoint_norm(setup.truth, dx_c))
        payload = {
            "selection_label": record.label, "z_selected": record.z,
            "lambda_selected": record.lam, "E2_at_selection": e2,
            "residual_at_selection": resid,
            "evaluations": record.evaluations, "coarse_winner_z": coarse,
            "target_residual": target, "delta_nom": delta_n,
            "delta_real": delta_r,
            "E2_tikhonov_raw": e2, "E2_tikhonov_projected": e2_proj,
        }
        writer.append(row, results.STATUS_COMPLETED, payload,
                      censored=record.endpoint)
    accounting = results.reconcile("lambda_noise", writer.csv_path)
    results.write_summary(
        Path(out_dir), accounting,
        {"accounting_consistent": accounting.consistent,
         "no_morozov_labels_on_failed_rows": True},
        extra={"monotonicity_failures": monotonicity_failures})
    return accounting


# ---------------------------------------------------------------------------
# Closure study (protocol Section 8)
# ---------------------------------------------------------------------------

COMPARISON_M = schema.DEFAULT_M            # common 400-cell comparison grid
CLOSURE_FINE_M, CLOSURE_FINE_DT = schema.CLOSURE_REF_RESOLUTIONS[-1]

CLOSURE_PAYLOAD_KEYS = (
    "max_cfl", "speed_bound", "min_u", "q_diff_ref", "u_diff_ref",
    "split_rel_u", "split_max_abs_q", "bridge_dist_q", "bridge_dist_u",
)


def closure_payload(**values) -> dict:
    payload = {key: NAN for key in CLOSURE_PAYLOAD_KEYS}
    for key, value in values.items():
        if key not in payload:
            raise CampaignGateError(f"unknown closure payload key {key!r}")
        payload[key] = value
    return payload


def closure_state(case: str, m: int, reverse_time: float
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Analytic (center values, exact cell-average gradient) of the heat
    solution at reverse time tau, i.e. physical time T - tau."""
    spec = schema.CASES[case]
    t_phys = spec["T"] - reverse_time
    if t_phys < -1e-12:
        raise CampaignGateError(f"reverse time {reverse_time} exceeds T")
    x = cell_centers(X_MIN, X_MAX, m)
    edges = cell_edges(X_MIN, X_MAX, m)
    dx = cell_spacing(X_MIN, X_MAX, m)
    if "modes" in spec:
        u = cosine_field(x, spec["background"], spec["modes"], length=LENGTH,
                         alpha=spec["alpha"], t=t_phys)
        u_edges = cosine_field(edges, spec["background"], spec["modes"],
                               length=LENGTH, alpha=spec["alpha"], t=t_phys)
    else:
        u = gaussian_free_space(x, spec["mu"], spec["sigma0"],
                                spec["amplitude"], alpha=spec["alpha"],
                                t=t_phys)
        u_edges = gaussian_free_space(edges, spec["mu"], spec["sigma0"],
                                      spec["amplitude"], alpha=spec["alpha"],
                                      t=t_phys)
    return u, np.diff(u_edges) / dx


@lru_cache(maxsize=None)
def reference_solution(case: str, closure: str, kind: str, m: int,
                       dt: float, h: float):
    """Shared cache so carrier and h-bridge rows reuse reference solves.

    The unregularized kind must be keyed with h = 0.0 (its flux ignores h).
    """
    if kind == "unregularized" and h != 0.0:
        raise CampaignGateError("key unregularized references with h = 0.0")
    spec = schema.CASES[case]
    g, q0 = closure_state(case, m, 0.0)
    dx = cell_spacing(X_MIN, X_MAX, m)
    return run_reference(
        q0, kind=kind, closure=closure, anchor_value=float(g[0]),
        total_mass=float(dx * np.sum(g)), x_min=X_MIN, x_max=X_MAX,
        T=spec["T"], dt=dt, alpha=spec["alpha"], bandwidth=h,
        eps_rel=schema.CLOSURE_EPS_REL if kind == "regularized" else 0.0,
        snapshot_times=(spec["T"] / 2.0,))


def project_fine(field: np.ndarray, m_to: int) -> np.ndarray:
    """Conservative cell-average projection from a nested finer grid."""
    field = np.asarray(field, dtype=float)
    m_from = field.shape[0]
    if m_from == m_to:
        return field.copy()
    if m_from % m_to:
        raise CampaignGateError(f"grids {m_from} -> {m_to} are not nested")
    return field.reshape(m_to, m_from // m_to).mean(axis=1)


def _rel(a: np.ndarray, b: np.ndarray, dx: float) -> float:
    return float(midpoint_norm(np.asarray(a) - np.asarray(b), dx)
                 / midpoint_norm(np.asarray(b), dx))


def _reference_npz(case: str, closure: str, kind: str, m: int) -> str:
    return f"reference_{case}_{closure}_{kind}_{m}.npz"


def _carrier_npz(case: str, closure: str, m: int) -> str:
    return f"carrier_{case}_{closure}_{m}.npz"


def drive_closure(out_dir: Path,
                  rows: Sequence[dict] | None = None) -> results.Accounting:
    out = Path(out_dir)
    writer = results.StudyWriter("closure", out)
    fields_dir = out / "fields"
    fields_dir.mkdir(parents=True, exist_ok=True)
    for row in rows or results.enumerate_rows("closure"):
        if results.study_row_key(writer.study, row) in writer.done_keys:
            continue
        case, closure = row["case"], row["closure"]
        spec = schema.CASES[case]
        big_t = spec["T"]
        half = big_t / 2.0
        if row["block"] == "reference":
            h = (schema.CLOSURE_BANDWIDTH if row["kind"] == "regularized"
                 else 0.0)
            res = reference_solution(case, closure, row["kind"], row["M"],
                                     row["dt"], h)
            payload = closure_payload(max_cfl=res.max_cfl,
                                      speed_bound=res.speed_bound,
                                      min_u=res.min_u)
            if res.status == "completed":
                u_half, q_half = res.snapshots[half]
                np.savez(
                    fields_dir / _reference_npz(case, closure, row["kind"],
                                                row["M"]),
                    u_half=project_fine(u_half, COMPARISON_M),
                    q_half=project_fine(q_half, COMPARISON_M),
                    u_final=project_fine(res.u, COMPARISON_M),
                    q_final=project_fine(res.q, COMPARISON_M))
            writer.append(row,
                          results.STATUS_COMPLETED if res.status == "completed"
                          else results.STATUS_FAILED,
                          payload, failure_message=res.failure_message)
        elif row["block"] == "carrier":
            g, _ = closure_state(case, row["M"], 0.0)
            snap = (half,) if row["M"] == COMPARISON_M else ()
            res = run_gradient_carriers(
                g, x_min=X_MIN, x_max=X_MAX, T=big_t, dt=row["dt"],
                bandwidth=schema.CLOSURE_BANDWIDTH,
                eps_rel=schema.CLOSURE_EPS_REL, alpha=spec["alpha"],
                closure=closure, snapshot_times=snap)
            payload = closure_payload(min_u=res.min_u)
            if res.status == "completed":
                ref = reference_solution(case, closure, "regularized",
                                         CLOSURE_FINE_M, CLOSURE_FINE_DT,
                                         schema.CLOSURE_BANDWIDTH)
                dx_m = cell_spacing(X_MIN, X_MAX, row["M"])
                if ref.status == "completed":
                    payload["q_diff_ref"] = _rel(
                        res.q_final, project_fine(ref.q, row["M"]), dx_m)
                    payload["u_diff_ref"] = _rel(
                        res.u_final, project_fine(ref.u, row["M"]), dx_m)
                arrays = {"u_final": res.u_final, "q_final": res.q_final}
                if snap:
                    u_half_c, q_half_c = res.snapshots[half]
                    arrays.update(u_half=u_half_c, q_half=q_half_c)
                np.savez(fields_dir / _carrier_npz(case, closure, row["M"]),
                         **arrays)
            writer.append(row,
                          results.STATUS_COMPLETED if res.status == "completed"
                          else results.STATUS_FAILED,
                          payload, failure_step=res.failure_step,
                          failure_message=res.failure_message)
        elif row["block"] == "split_invariance":
            g, _ = closure_state(case, COMPARISON_M, 0.0)
            runs = [run_gradient_carriers(
                        g, x_min=X_MIN, x_max=X_MAX, T=big_t,
                        dt=schema.DEFAULT_DT,
                        bandwidth=schema.CLOSURE_BANDWIDTH,
                        eps_rel=schema.CLOSURE_EPS_REL, alpha=spec["alpha"],
                        closure=closure, subparticles=s)
                    for s in (1, 20)]
            if all(r.status == "completed" for r in runs):
                one, many = runs
                scale = float(np.max(np.abs(one.u_final)))
                payload = closure_payload(
                    split_rel_u=float(np.max(np.abs(one.u_final
                                                    - many.u_final))) / scale,
                    split_max_abs_q=float(np.max(np.abs(one.q_final
                                                        - many.q_final))),
                    min_u=min(one.min_u, many.min_u))
                writer.append(row, results.STATUS_COMPLETED, payload)
            else:
                message = "; ".join(r.failure_message for r in runs
                                    if r.status != "completed")
                writer.append(row, results.STATUS_FAILED, closure_payload(),
                              failure_message=message)
        elif row["block"] == "h_bridge":
            reg = reference_solution(case, closure, "regularized", row["M"],
                                     row["dt"], row["h"])
            unreg = reference_solution(case, closure, "unregularized",
                                       row["M"], row["dt"], 0.0)
            if reg.status == "completed" and unreg.status == "completed":
                dx400 = cell_spacing(X_MIN, X_MAX, COMPARISON_M)
                payload = closure_payload(
                    bridge_dist_q=_rel(project_fine(reg.q, COMPARISON_M),
                                       project_fine(unreg.q, COMPARISON_M),
                                       dx400),
                    bridge_dist_u=_rel(project_fine(reg.u, COMPARISON_M),
                                       project_fine(unreg.u, COMPARISON_M),
                                       dx400),
                    max_cfl=reg.max_cfl, speed_bound=reg.speed_bound,
                    min_u=min(reg.min_u, unreg.min_u))
                writer.append(row, results.STATUS_COMPLETED, payload)
            else:
                message = "; ".join(r.failure_message for r in (reg, unreg)
                                    if r.status != "completed")
                writer.append(row, results.STATUS_FAILED, closure_payload(),
                              failure_message=message)
        else:
            raise CampaignGateError(f"unknown closure block {row['block']!r}")
    accounting = results.reconcile("closure", writer.csv_path)
    gates = closure_gates(out, writer.csv_path)
    (out / "closure_gates.json").write_text(
        json.dumps(gates, indent=1) + "\n", encoding="utf-8")
    decomposition = closure_decomposition(out)
    (out / "closure_decomposition.json").write_text(
        json.dumps(decomposition, indent=1) + "\n", encoding="utf-8")
    verdicts = {"accounting_consistent": accounting.consistent}
    verdicts.update(gates["verdicts"])
    for entry in decomposition["cases"]:
        if entry["status"] == "ok":
            key = f"decomposition_reconciled_{entry['case']}_{entry['closure']}"
            verdicts[key] = entry["reconciled"]
    results.write_summary(out, accounting, verdicts)
    return accounting


def closure_gates(out: Path, csv_path: Path) -> dict:
    """Machine-readable Section 8 gate archive: resolutions, CFL numbers,
    speed bounds, comparison errors, and failure reasons per block."""
    table = _read_rows(csv_path)
    fields_dir = out / "fields"
    dx400 = cell_spacing(X_MIN, X_MAX, COMPARISON_M)
    verdicts: dict[str, bool] = {}

    def rows_where(**want):
        keep = []
        for r in table:
            if all(str(r.get(k, "")) == str(v) for k, v in want.items()):
                keep.append(r)
        return keep

    reference_pairs = []
    for case in schema.CLOSURE_CASES:
        for closure in schema.CLOSURES:
            for kind in schema.REFERENCE_KINDS:
                found = rows_where(block="reference", case=case,
                                   closure=closure, kind=kind)
                if not found:
                    continue
                entry = {
                    "case": case, "closure": closure, "kind": kind,
                    "gate": schema.CLOSURE_REF_GATE,
                    "resolutions": [list(r) for r in
                                    schema.CLOSURE_REF_RESOLUTIONS],
                    "rows": [{
                        "M": int(r["M"]), "dt": float(r["dt"]),
                        "status": r["status"],
                        "max_cfl": float(r["max_cfl"]),
                        "speed_bound": float(r["speed_bound"]),
                        "min_u": float(r["min_u"]),
                        "failure_reason": r["failure_message"],
                    } for r in found],
                }
                paths = [fields_dir / _reference_npz(case, closure, kind, m)
                         for m, _ in schema.CLOSURE_REF_RESOLUTIONS]
                if all(path.exists() for path in paths):
                    with np.load(paths[0]) as coarse, np.load(paths[1]) as fine:
                        comparisons = {
                            label: {
                                "q_rel": _rel(coarse[f"q_{label}"],
                                              fine[f"q_{label}"], dx400),
                                "u_rel": _rel(coarse[f"u_{label}"],
                                              fine[f"u_{label}"], dx400),
                            } for label in ("half", "final")}
                    entry["comparisons"] = comparisons
                    entry["pass"] = all(
                        c[k] <= schema.CLOSURE_REF_GATE
                        for c in comparisons.values()
                        for k in ("q_rel", "u_rel"))
                else:
                    entry["comparisons"] = None
                    entry["pass"] = False
                    entry["failure_reason"] = ("a reference resolution did "
                                               "not complete")
                verdicts[f"reference_gate_{case}_{closure}_{kind}"] = (
                    entry["pass"])
                reference_pairs.append(entry)

    refinements = []
    for case in schema.CLOSURE_CASES:
        for closure in schema.CLOSURES:
            found = rows_where(block="carrier", case=case, closure=closure)
            if not found:
                continue
            diffs = {}
            for r in found:
                if (r["status"] == results.STATUS_COMPLETED
                        and math.isfinite(float(r["q_diff_ref"]))):
                    diffs[int(r["M"])] = {"q": float(r["q_diff_ref"]),
                                          "u": float(r["u_diff_ref"])}
            entry = {"case": case, "closure": closure, "diffs": diffs,
                     "required_last_reduction": schema.CLOSURE_LAST_REDUCTION}
            grid = [m for m, _ in schema.CLOSURE_REFINEMENTS]
            if all(m in diffs for m in grid):
                ok = True
                for field_key in ("q", "u"):
                    curve = [diffs[m][field_key] for m in grid]
                    decreasing = all(a > b for a, b in zip(curve, curve[1:]))
                    last = curve[-2] / curve[-1] if curve[-1] > 0 else math.inf
                    entry[f"last_reduction_{field_key}"] = last
                    ok = ok and decreasing and (
                        last >= schema.CLOSURE_LAST_REDUCTION)
                entry["pass"] = ok
            else:
                entry["pass"] = False
                entry["failure_reason"] = "incomplete refinement ladder"
            verdicts[f"refinement_{case}_{closure}"] = entry["pass"]
            refinements.append(entry)

    splits = []
    for r in rows_where(block="split_invariance"):
        entry = {"case": r["case"], "closure": r["closure"],
                 "status": r["status"],
                 "failure_reason": r["failure_message"]}
        if r["status"] == results.STATUS_COMPLETED:
            entry["rel_u"] = float(r["split_rel_u"])
            entry["max_abs_q"] = float(r["split_max_abs_q"])
            entry["pass"] = entry["rel_u"] <= schema.SPLIT_INVARIANCE_TOL
        else:
            entry["pass"] = False
        verdicts[f"split_{r['case']}_{r['closure']}"] = entry["pass"]
        splits.append(entry)

    bridges = []
    for case in schema.CLOSURE_CASES:
        for closure in schema.CLOSURES:
            found = rows_where(block="h_bridge", case=case, closure=closure)
            if not found:
                continue
            curves = {}
            for r in found:
                if r["status"] != results.STATUS_COMPLETED:
                    continue
                curves.setdefault(int(r["M"]), {})[float(r["h"])] = {
                    "q": float(r["bridge_dist_q"]),
                    "u": float(r["bridge_dist_u"])}
            entry = {"case": case, "closure": closure, "curves": curves}
            fine = curves.get(CLOSURE_FINE_M, {})
            if 0.007 in fine and 0.014 in fine:
                entry["pass"] = (fine[0.007]["q"] < fine[0.014]["q"]
                                 and fine[0.007]["u"] < fine[0.014]["u"])
            else:
                entry["pass"] = False
                entry["failure_reason"] = "missing fine-resolution bridge rows"
            verdicts[f"h_bridge_{case}_{closure}"] = entry["pass"]
            bridges.append(entry)

    return {
        "commit": git_commit(REPO),
        "comparison_grid_m": COMPARISON_M,
        "archived_reverse_times": "T/2 and T",
        "reference_pairs": reference_pairs,
        "carrier_refinement": refinements,
        "split_invariance": splits,
        "h_bridge": bridges,
        "verdicts": verdicts,
    }


def closure_decomposition(out: Path) -> dict:
    """Protocol Section 8 exact four-field decomposition at both archived
    reverse times, plus the G1 analytic anchor."""
    fields_dir = out / "fields"
    dx = cell_spacing(X_MIN, X_MAX, COMPARISON_M)
    entries = []
    for case in schema.CLOSURE_CASES:
        spec = schema.CASES[case]
        big_t = spec["T"]
        for closure in schema.CLOSURES:
            needed = {
                "mass_unreg": _reference_npz(case, "mass", "unregularized",
                                             CLOSURE_FINE_M),
                "c_unreg": _reference_npz(case, closure, "unregularized",
                                          CLOSURE_FINE_M),
                "c_reg": _reference_npz(case, closure, "regularized",
                                        CLOSURE_FINE_M),
                "carrier": _carrier_npz(case, closure, COMPARISON_M),
            }
            missing = [name for name, fname in needed.items()
                       if not (fields_dir / fname).exists()]
            if missing:
                entries.append({"case": case, "closure": closure,
                                "status": "missing_inputs",
                                "missing": missing})
                continue
            loaded = {name: np.load(fields_dir / fname)
                      for name, fname in needed.items()}
            times = {}
            reconciled = True
            for label, tau in (("half", big_t / 2.0), ("final", big_t)):
                u_truth, q_truth = closure_state(case, COMPARISON_M, tau)
                out_time = {"reverse_time": tau}
                for field_key, truth in (("u", u_truth), ("q", q_truth)):
                    parts = [loaded[name][f"{field_key}_{label}"]
                             for name in ("mass_unreg", "c_unreg", "c_reg",
                                          "carrier")]
                    decomp = exact_decomposition(truth, *parts, dx=dx)
                    out_time[field_key] = decomp
                    reconciled = reconciled and (
                        decomp["reconciliation_residual"]
                        <= schema.DECOMP_RECONCILE_TOL)
                times[label] = out_time
            for handle in loaded.values():
                handle.close()
            entries.append({"case": case, "closure": closure, "status": "ok",
                            "times": times, "reconciled": reconciled})

    g1 = schema.CASES["G1"]
    amplitude = g1["modes"][0][1]
    u0, q_rev_t = closure_state("G1", COMPARISON_M, g1["T"])
    g_terminal, _ = closure_state("G1", COMPARISON_M, 0.0)
    constant = closure_constant(
        q_rev_t, "frozen_left", dx=dx,
        anchor_value=float(g_terminal[0]),
        total_mass=float(dx * np.sum(g_terminal)), length=LENGTH)
    u_frozen = reconstruct_centers(q_rev_t, constant, dx)
    offset = u_frozen - u0
    predicted = frozen_left_offset(amplitude, g1["alpha"], g1["T"])
    anchor = {
        "signed_mean_offset": float(np.mean(offset)),
        "predicted_signed_offset": predicted,
        "abs_error_vs_formula": abs(float(np.mean(offset)) - predicted),
        "offset_std": float(np.std(offset)),
        "magnitude": abs(predicted),
        "relative_l2_contribution": float(midpoint_norm(offset, dx)
                                          / midpoint_norm(u0, dx)),
    }
    return {"commit": git_commit(REPO), "cases": entries,
            "analytic_anchor_G1": anchor}


# ---------------------------------------------------------------------------
# Initial-rate diagnostic (protocol Amendment 3; Theorem R1, ledger 65-69)
# ---------------------------------------------------------------------------

INITIAL_RATE_PAYLOAD_KEYS = (
    "e_U", "ratio", "c_rep", "gate_band", "within_band", "u_diff_ref",
    "e_q", "ratio_q", "slope_q", "min_u",
)


def _initial_rate_payload(**values) -> dict:
    payload = {key: NAN for key in INITIAL_RATE_PAYLOAD_KEYS}
    for key, value in values.items():
        if key not in payload:
            raise CampaignGateError(f"unknown initial-rate payload key {key!r}")
        payload[key] = value
    return payload


def initial_rate_constants(case: str) -> dict:
    """Exact Corollary R1a constants for a c + a cos(pi x) closure case:
    B = a e^{-alpha pi^2 T}, R = sqrt(c^2 - B^2), c_rep = alpha pi^2
    sqrt(R (c - R)), and the q-level slope alpha pi^3 B sqrt(c / (2R))."""
    spec = schema.CASES[case]
    if "modes" not in spec or len(spec["modes"]) != 1 or spec["modes"][0][0] != 1:
        raise CampaignGateError("initial-rate constants need a single cos(pi x) mode")
    c, amp = spec["background"], spec["modes"][0][1]
    alpha, big_t = spec["alpha"], spec["T"]
    b = amp * math.exp(-alpha * math.pi ** 2 * big_t)
    if c <= b:
        raise CampaignGateError("initial-rate constants need c > B")
    r = math.sqrt(c * c - b * b)
    return {"c": c, "B": b, "R": r,
            "c_rep": alpha * math.pi ** 2 * math.sqrt(r * (c - r)),
            "slope_q": alpha * math.pi ** 3 * b * math.sqrt(c / (2.0 * r))}


@lru_cache(maxsize=None)
def initial_rate_reference(case: str, closure: str, m: int, dt: float,
                           tau: float):
    """Exact wrong flow (unregularized flux alpha q^2/U) to reverse time tau."""
    spec = schema.CASES[case]
    g, q0 = closure_state(case, m, 0.0)
    dx = cell_spacing(X_MIN, X_MAX, m)
    return run_reference(
        q0, kind="unregularized", closure=closure, anchor_value=float(g[0]),
        total_mass=float(dx * np.sum(g)), x_min=X_MIN, x_max=X_MAX, T=tau,
        dt=dt, alpha=spec["alpha"], bandwidth=0.0, eps_rel=0.0)


def drive_initial_rate(out_dir: Path,
                       rows: Sequence[dict] | None = None
                       ) -> results.Accounting:
    out = Path(out_dir)
    writer = results.StudyWriter("initial_rate", out)
    for row in rows or results.enumerate_rows("initial_rate"):
        if results.study_row_key(writer.study, row) in writer.done_keys:
            continue
        case, closure, tau = row["case"], row["closure"], float(row["tau"])
        spec = schema.CASES[case]
        consts = initial_rate_constants(case)
        m, dt = int(row["M"]), float(row["dt"])
        dx = cell_spacing(X_MIN, X_MAX, m)
        if row["block"] == "reference":
            res = initial_rate_reference(case, closure, m, dt, tau)
            if res.status != "completed":
                writer.append(row, results.STATUS_FAILED,
                              _initial_rate_payload(min_u=res.min_u),
                              failure_message=res.failure_message)
                continue
            u_true, _ = closure_state(case, m, tau)
            e_u = float(midpoint_norm(res.u - u_true, dx))
            ratio = e_u / (consts["c_rep"] * tau)
            writer.append(row, results.STATUS_COMPLETED, _initial_rate_payload(
                e_U=e_u, ratio=ratio, c_rep=consts["c_rep"],
                gate_band=schema.INITIAL_RATE_GATE,
                within_band=bool(abs(ratio - 1.0) <= schema.INITIAL_RATE_GATE),
                min_u=res.min_u))
        elif row["block"] == "carrier":
            g, _ = closure_state(case, m, 0.0)
            res = run_gradient_carriers(
                g, x_min=X_MIN, x_max=X_MAX, T=tau, dt=dt,
                bandwidth=schema.CLOSURE_BANDWIDTH,
                eps_rel=schema.CLOSURE_EPS_REL, alpha=spec["alpha"],
                closure=closure)
            fine_m, fine_dt = schema.INITIAL_RATE_REFERENCES[-1]
            ref = initial_rate_reference(case, closure, fine_m, fine_dt, tau)
            if res.status != "completed" or ref.status != "completed":
                message = "; ".join(r.failure_message for r in (res, ref)
                                    if r.status != "completed")
                writer.append(row, results.STATUS_FAILED,
                              _initial_rate_payload(min_u=res.min_u),
                              failure_step=res.failure_step,
                              failure_message=message)
                continue
            writer.append(row, results.STATUS_COMPLETED, _initial_rate_payload(
                u_diff_ref=_rel(res.u_final, project_fine(ref.u, m), dx),
                min_u=res.min_u))
        elif row["block"] == "q_level":
            res = initial_rate_reference(case, closure, m, dt, tau)
            if res.status != "completed":
                writer.append(row, results.STATUS_FAILED,
                              _initial_rate_payload(min_u=res.min_u),
                              failure_message=res.failure_message)
                continue
            _, q_true = closure_state(case, m, tau)
            e_q = float(midpoint_norm(res.q - q_true, dx))
            writer.append(row, results.STATUS_COMPLETED, _initial_rate_payload(
                e_q=e_q, ratio_q=e_q / (consts["slope_q"] * tau),
                slope_q=consts["slope_q"], min_u=res.min_u))
        else:
            raise CampaignGateError(f"unknown initial-rate block {row['block']!r}")
    accounting = results.reconcile("initial_rate", writer.csv_path)
    gates = initial_rate_gates(writer.csv_path)
    (out / "initial_rate_gates.json").write_text(
        json.dumps(gates, indent=1) + "\n", encoding="utf-8")
    verdicts = {"accounting_consistent": accounting.consistent}
    verdicts.update(gates["verdicts"])
    results.write_summary(out, accounting, verdicts)
    return accounting


def initial_rate_gates(csv_path: Path) -> dict:
    """Block A gates: both references inside the certificate band, and the
    reference pair agreeing relative to the defect signal c_rep * tau.
    A failure blocks the numerical validation and triggers reference-
    resolution diagnosis; it does not by itself contradict the theorem."""
    refs = [r for r in _read_rows(csv_path) if r["block"] == "reference"
            and r["status"] == results.STATUS_COMPLETED]
    verdicts: dict[str, bool] = {}
    detail: dict[str, object] = {"references": []}
    if len(refs) == len(schema.INITIAL_RATE_REFERENCES):
        within = all(r["within_band"] == "True" for r in refs)
        refs.sort(key=lambda r: int(r["M"]))
        c_rep = float(refs[0]["c_rep"])
        tau = float(refs[0]["tau"])
        pair = abs(float(refs[0]["e_U"]) - float(refs[1]["e_U"])) / (c_rep * tau)
        verdicts["block_a_within_certificate_band"] = bool(within)
        verdicts["block_a_reference_pair_defect_relative"] = bool(
            pair <= schema.INITIAL_RATE_PAIR_GATE)
        detail["pair_defect_relative"] = pair
        detail["references"] = [{"M": int(r["M"]), "e_U": float(r["e_U"]),
                                 "ratio": float(r["ratio"])} for r in refs]
    else:
        verdicts["block_a_within_certificate_band"] = False
        verdicts["block_a_reference_pair_defect_relative"] = False
    detail["verdicts"] = verdicts
    detail["evidence_grade_blocks"] = ["carrier", "q_level"]
    return detail


# ---------------------------------------------------------------------------
# Crossover phenomenology block (protocol Amendment 4; evidence-grade only)
# ---------------------------------------------------------------------------

CROSSOVER_PAYLOAD_KEYS = (
    "e_k", "e_2k", "gap", "harmonic_dominant", "d", "b", "r1", "r2",
    "pred_e_k", "pred_e_2k", "min_u", "e_k_particle", "e_2k_particle",
)


def _crossover_payload(**values) -> dict:
    payload = {key: NAN for key in CROSSOVER_PAYLOAD_KEYS}
    for key, value in values.items():
        if key not in payload:
            raise CampaignGateError(f"unknown crossover payload key {key!r}")
        payload[key] = value
    return payload


def _crossover_k() -> float:
    return schema.CROSSOVER_MODE * math.pi / LENGTH


@lru_cache(maxsize=None)
def crossover_continuum(a: float, kh: float) -> tuple:
    """Density method's continuum flow for u0 = 1 + a cos(kx) by cosine
    collocation with the sine-projected flux derivative (fully spectral),
    exact terminal datum, RK4.  Returns (e_k, e_2k, min_u)."""
    from scipy.signal import fftconvolve  # noqa: F401  (dependency check)
    cells, nmodes, dt = schema.CROSSOVER_CONTINUUM
    alpha, big_t = schema.CROSSOVER_ALPHA, schema.CROSSOVER_T
    k = _crossover_k()
    h = kh / k
    eps_abs = schema.CROSSOVER_EPS_REL          # M0 = 1, L = 1
    x = (np.arange(cells) + 0.5) / cells
    dx = 1.0 / cells
    kv = np.arange(1, nmodes + 1) * math.pi
    cos = np.cos(np.outer(x, kv))
    sin = np.sin(np.outer(x, kv))
    phi = np.exp(-0.5 * (kv * h) ** 2)

    def rhs(rho):
        c = 2.0 * dx * (cos.T @ rho)
        w = rho.mean() + cos @ (c * phi)
        wx = -sin @ (c * phi * kv)
        flux = rho * wx / (w + eps_abs)
        f_sin = 2.0 * dx * (sin.T @ flux)
        return -alpha * (cos @ (f_sin * kv))

    rho = 1.0 + a * math.exp(-alpha * k * k * big_t) * np.cos(k * x)
    min_u = float(rho.min())
    for _ in range(round(big_t / dt)):
        k1 = rhs(rho); k2 = rhs(rho + 0.5 * dt * k1)
        k3 = rhs(rho + 0.5 * dt * k2); k4 = rhs(rho + dt * k3)
        rho = rho + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        min_u = min(min_u, float(rho.min()))
    err = rho - (1.0 + a * np.cos(k * x))
    coeff = 2.0 * dx * (cos.T @ err)
    mode = schema.CROSSOVER_MODE
    return float(coeff[mode - 1]), float(coeff[2 * mode - 1]), min_u


@lru_cache(maxsize=None)
def crossover_low_order(kh: float) -> tuple:
    """Exact amplitude-jet hierarchy of the lattice flow through order four:
    returns (d, b, r1, r2) for the low-order model
    e_k = -a d + a^3 r1, e_2k = a^2 b + a^4 r2 (evidence-grade)."""
    from scipy.signal import fftconvolve
    alpha, big_t = schema.CROSSOVER_ALPHA, schema.CROSSOVER_T
    k = _crossover_k()
    h = kh / k
    nmax, order, steps = 6, 5, 2000
    dt = big_t / steps
    m_eps = 1.0 / (1.0 + schema.CROSSOVER_EPS_REL)
    idx = lambda n: n + nmax
    kappa = np.array([n * k for n in range(-nmax, nmax + 1)])
    phi = np.exp(-0.5 * (kappa * h) ** 2)
    delta = np.zeros((2 * nmax + 1, order)); delta[idx(0), 0] = 1.0
    dk = (1j * kappa)[:, None]

    def jmul(a_, b_):
        return fftconvolve(a_, b_)[nmax:3 * nmax + 1, :order]

    def solve_v(c):
        w_off = (phi[:, None] * c).copy(); w_off[idx(0)] = 0.0
        v = m_eps * delta.copy()
        for _ in range(order + 3):
            v = m_eps * (delta - jmul(w_off, v))
        return v

    def rhs(c):
        return (-alpha) * dk * jmul(jmul(c, dk * phi[:, None] * c), solve_v(c))

    c = np.zeros((2 * nmax + 1, order), dtype=complex)
    c[idx(0), 0] = 1.0
    a0 = math.exp(-alpha * k * k * big_t) / 2.0
    c[idx(1), 1] = a0; c[idx(-1), 1] = a0
    for _ in range(steps):
        k1 = rhs(c); k2 = rhs(c + 0.5 * dt * k1)
        k3 = rhs(c + 0.5 * dt * k2); k4 = rhs(c + dt * k3)
        c = c + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    ck = 2 * c[idx(1)].real
    c2k = 2 * c[idx(2)].real
    return float(1.0 - ck[1]), float(c2k[2]), float(ck[3]), float(c2k[4])


def drive_crossover(out_dir: Path,
                    rows: Sequence[dict] | None = None) -> results.Accounting:
    out = Path(out_dir)
    writer = results.StudyWriter("crossover", out)
    k = _crossover_k()
    alpha, big_t = schema.CROSSOVER_ALPHA, schema.CROSSOVER_T
    for row in rows or results.enumerate_rows("crossover"):
        if results.study_row_key(writer.study, row) in writer.done_keys:
            continue
        a, kh = float(row["a"]), float(row["kh"])
        e_k, e_2k, min_u = crossover_continuum(a, kh)
        d, b, r1, r2 = crossover_low_order(kh)
        base = dict(e_k=e_k, e_2k=e_2k, gap=abs(e_2k) - abs(e_k),
                    harmonic_dominant=bool(abs(e_2k) > abs(e_k)),
                    d=d, b=b, r1=r1, r2=r2,
                    pred_e_k=-a * d + a ** 3 * r1,
                    pred_e_2k=a * a * b + a ** 4 * r2, min_u=min_u)
        if row["block"] == "continuum":
            writer.append(row, results.STATUS_COMPLETED,
                          _crossover_payload(**base))
        elif row["block"] == "particle":
            m = schema.DEFAULT_M
            x = cell_centers(X_MIN, X_MAX, m)
            dx = cell_spacing(X_MIN, X_MAX, m)
            g = 1.0 + a * math.exp(-alpha * k * k * big_t) * np.cos(k * x)
            u0 = 1.0 + a * np.cos(k * x)
            res = run_campaign_density(
                g, x_min=X_MIN, x_max=X_MAX, T=big_t, dt=schema.DEFAULT_DT,
                n_particles=int(row["N"]), bandwidth=kh / k,
                eps_rel=schema.CROSSOVER_EPS_REL, alpha=alpha,
                u0_reference=u0)
            if res.status != "completed":
                writer.append(row, results.STATUS_FAILED,
                              _crossover_payload(**base),
                              failure_step=res.failure_step,
                              failure_message=res.failure_message)
                continue
            err = res.reconstruction - u0
            mode = schema.CROSSOVER_MODE
            ek_p = float(2.0 * dx * np.sum(err * np.cos(mode * math.pi * x)))
            e2k_p = float(2.0 * dx * np.sum(err * np.cos(2 * mode * math.pi * x)))
            writer.append(row, results.STATUS_COMPLETED, _crossover_payload(
                **base, e_k_particle=ek_p, e_2k_particle=e2k_p))
        else:
            raise CampaignGateError(f"unknown crossover block {row['block']!r}")
    accounting = results.reconcile("crossover", writer.csv_path)
    results.write_summary(out, accounting,
                          {"accounting_consistent": accounting.consistent},
                          extra={"gates": "none by design (evidence-grade)"})
    return accounting


DRIVERS: dict[str, Callable[[Path], results.Accounting]] = {
    "bandwidth_clean": drive_bandwidth_clean,
    "epsilon_sensitivity": drive_epsilon_sensitivity,
    "adequacy_N": drive_adequacy,
    "lambda_oracle_clean": drive_lambda_oracle_clean,
    "noise_paired": drive_noise_paired,
    "lambda_noise": drive_lambda_noise,
    "closure": drive_closure,
    "initial_rate": drive_initial_rate,
    "crossover": drive_crossover,
}
