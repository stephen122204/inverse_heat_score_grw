"""campaign_drivers.py — study drivers for the Phase 2C campaign.

Consumes the preregistered rows of campaign_schema, executes them through the
canonical src components, and writes outcomes through the campaign_results
contract. Every driver supports resume (rows already present in the study CSV
are skipped), retains failures as rows, and finishes by reconciling the
Section 10 accounting and writing summary.json. The orchestrator wraps each
completed study directory in a hash-validated manifest.

Drivers implemented here: bandwidth_clean, epsilon_sensitivity, adequacy_N,
lambda_oracle_clean. The noise, lambda_noise, closure, and transition drivers
land in the next build increment.
"""

from __future__ import annotations

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
from invheat_grw.campaign_data import (
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
from invheat_grw.campaign_selectors import morozov_tikhonov, oracle_continuous
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
        if results.row_key(row) in writer.done_keys:
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
        if results.row_key(row) in writer.done_keys:
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
        if results.row_key(row) in writer.done_keys:
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
        if results.row_key(row) in writer.done_keys:
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


DRIVERS: dict[str, Callable[[Path], results.Accounting]] = {
    "bandwidth_clean": drive_bandwidth_clean,
    "epsilon_sensitivity": drive_epsilon_sensitivity,
    "adequacy_N": drive_adequacy,
    "lambda_oracle_clean": drive_lambda_oracle_clean,
}
