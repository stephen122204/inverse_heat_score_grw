"""run_verification_dossier.py — operator-verification dossier for the
cell-centered numerical core.

Every reference value is analytic on the bounded Neumann domain (cosine
manufactured solutions), so each table verifies the implemented operators
against exact mathematics rather than against another simulation:

  1. Neumann-kernel verification: mass, wall derivative, exact modal damping.
  2. Static score verification: Neumann KDE score of a quantile particle set
     against the analytic score of a known positive density.
  3. Constant-coefficient modal commutation: forward multiplier, Tikhonov
     filter, and spectral cutoff acting exactly on discrete cosine modes
     through the public APIs.
  4. Variable operator: structure (symmetry, nullspace, conservation,
     dissipation, adjoint) and forced manufactured-solution convergence with
     separate spatial and temporal tables.
  5. Exact-score density-particle refinement at fixed physical bandwidth:
     particle count, time step, and grid varied independently, with the
     particle-count study compared against the analytically predicted
     kernel-bias floor.

Outputs: one CSV per table, a machine-readable summary.json, a referee-facing
dossier.md, and a hash-validated run manifest through provenance.  Exits
nonzero if any verdict fails.

Usage:
    python scripts/run_verification_dossier.py
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd

from invheat_grw.baselines import spectral_cutoff_inverse, tikhonov_inverse
from invheat_grw.cell_grid import (
    cell_centers,
    cell_edge_quantile_positions,
    cell_spacing,
    midpoint_mass,
    midpoint_norm,
    wave_numbers,
)
from invheat_grw.config import exact_step_count, load_config
from invheat_grw.fields import GRID_CONVENTION, make_grid
from invheat_grw.globs import apply_reflecting_boundary
from invheat_grw.metrics import forward_heat_solve_dct
from invheat_grw.neumann_kernels import (
    neumann_kde_density_derivative,
    neumann_kde_score,
)
from provenance import git_commit, write_manifest

_spec = importlib.util.spec_from_file_location(
    "vc_audit", str(REPO / "scripts" / "run_variable_coefficient_audit.py"))
vc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vc)  # type: ignore[union-attr]

X_MIN, X_MAX = 0.0, 1.0
LENGTH = X_MAX - X_MIN
ALPHA = 0.01

# Manufactured cosine Neumann solution for the particle studies:
#   u(x, t) = 1 + A_MODE exp(-alpha k^2 t) cos(k (x - x_min)),  k = n pi / L.
MODE_N = 3
A_MODE = 0.5
K_MODE = MODE_N * np.pi / LENGTH
T_PART = 1.0


def manufactured_u(x: np.ndarray, t: float) -> np.ndarray:
    return 1.0 + A_MODE * np.exp(-ALPHA * K_MODE ** 2 * t) * np.cos(K_MODE * (x - X_MIN))


def manufactured_score(x: np.ndarray, t: float) -> np.ndarray:
    amp = A_MODE * np.exp(-ALPHA * K_MODE ** 2 * t)
    return (-amp * K_MODE * np.sin(K_MODE * (x - X_MIN))
            / (1.0 + amp * np.cos(K_MODE * (x - X_MIN))))


def run_exact_score_particles(m_grid: int, n_particles: int, dt: float,
                              h: float, T: float = T_PART):
    """Backward exact-score transport + Neumann KDE reconstruction.

    Returns (relative L2 error vs u0, reconstruction mass error, recon, grid).
    """
    x = cell_centers(X_MIN, X_MAX, m_grid)
    dx = cell_spacing(X_MIN, X_MAX, m_grid)
    u_T = manufactured_u(x, T)
    positions, total_mass = cell_edge_quantile_positions(u_T, X_MIN, X_MAX, n_particles)
    for k in range(exact_step_count(T, dt)):
        t_phys = T - k * dt
        positions = apply_reflecting_boundary(
            positions + ALPHA * manufactured_score(positions, t_phys) * dt,
            X_MIN, X_MAX)
    weights = np.full(n_particles, total_mass / n_particles)
    recon, _, _ = neumann_kde_density_derivative(
        x, positions, weights, X_MIN, X_MAX, h)
    u0 = manufactured_u(x, 0.0)
    rel = midpoint_norm(recon - u0, dx) / midpoint_norm(u0, dx)
    mass_rel = midpoint_mass(recon, dx) / total_mass - 1.0
    return rel, mass_rel, recon, x


def predicted_kernel_bias(h: float) -> float:
    """Relative L2 bias of the Neumann kernel on u0: the single cosine mode
    is damped by exactly exp(-(k h)^2 / 2)."""
    damping = float(np.exp(-0.5 * (K_MODE * h) ** 2))
    return (A_MODE * (1.0 - damping) * np.sqrt(0.5)
            / np.sqrt(1.0 + 0.5 * A_MODE ** 2))


def orders_of(errors: list[float]) -> list[float]:
    return [float(np.log2(errors[i] / errors[i + 1]))
            for i in range(len(errors) - 1)]


def fmt_table(df: pd.DataFrame) -> str:
    return df.to_string(index=False,
                        float_format=lambda v: f"{v:.4e}")


def main() -> None:
    t_start = time.perf_counter()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = REPO / "outputs" / f"verification_dossier_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    verdicts: dict[str, bool] = {}
    tables: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # 1. Neumann-kernel verification
    # ------------------------------------------------------------------
    print("[1/5] Neumann-kernel verification", flush=True)
    rows = []
    m = 256
    x = cell_centers(X_MIN, X_MAX, m)
    dx = cell_spacing(X_MIN, X_MAX, m)
    # Fine midpoint quadrature grid: every kernel mode below 2 * m_fine sums
    # to exactly zero, so the midpoint integral equals the mass to roundoff.
    m_fine = 20_000
    x_fine = cell_centers(X_MIN, X_MAX, m_fine)
    dx_fine = cell_spacing(X_MIN, X_MAX, m_fine)
    walls = np.array([X_MIN, X_MAX])
    for h in (0.01, 0.02, 0.04):
        for n in (4, 12):
            k = float(wave_numbers(m, LENGTH)[n])
            u = 1.0 + 0.3 * np.cos(k * (x - X_MIN))
            dens_fine, _, _ = neumann_kde_density_derivative(
                x_fine, x, u * dx, X_MIN, X_MAX, h)
            mass_err = abs(midpoint_mass(dens_fine, dx_fine) - midpoint_mass(u, dx))
            _, deriv_walls, _ = neumann_kde_density_derivative(
                walls, x, u * dx, X_MIN, X_MAX, h)
            wall = float(np.max(np.abs(deriv_walls)))
            dens_c, _, _ = neumann_kde_density_derivative(
                x, x, u * dx, X_MIN, X_MAX, h)
            amp = float(2.0 / m * np.sum(dens_c * np.cos(k * (x - X_MIN))))
            damping_err = abs(amp - 0.3 * float(np.exp(-0.5 * (k * h) ** 2)))
            rows.append({"h": h, "mode": n, "mass_error": mass_err,
                         "wall_derivative": wall, "damping_error": damping_err})
    tables["kernel_verification"] = pd.DataFrame(rows)
    kv = tables["kernel_verification"]
    verdicts["kernel_mass<=1e-11"] = bool(kv.mass_error.max() <= 1e-11)
    verdicts["kernel_wall<=1e-9"] = bool(kv.wall_derivative.max() <= 1e-9)
    verdicts["kernel_damping<=1e-11"] = bool(kv.damping_error.max() <= 1e-11)

    # ------------------------------------------------------------------
    # 2. Static score verification (epsilon = 0; density bounded below)
    # ------------------------------------------------------------------
    print("[2/5] static score verification", flush=True)
    rows = []
    m = 400
    x = cell_centers(X_MIN, X_MAX, m)
    dx = cell_spacing(X_MIN, X_MAX, m)
    u0 = manufactured_u(x, 0.0)
    s_exact = manufactured_score(x, 0.0)
    for h in (0.02, 0.01):
        for n_particles in (1000, 4000, 16000):
            pos, total = cell_edge_quantile_positions(u0, X_MIN, X_MAX, n_particles)
            w = np.full(n_particles, total / n_particles)
            _, _, s_est, _ = neumann_kde_score(x, pos, w, X_MIN, X_MAX, h,
                                               epsilon=0.0)
            err = s_est - s_exact
            rows.append({"h": h, "n_particles": n_particles,
                         "score_L2": midpoint_norm(err, dx),
                         "score_Linf": float(np.max(np.abs(err)))})
    tables["score_static"] = pd.DataFrame(rows)
    sc = tables["score_static"]
    verdicts["score_finite"] = bool(
        np.isfinite(sc[["score_L2", "score_Linf"]].to_numpy()).all())
    linf_h002 = float(sc[(sc.h == 0.02) & (sc.n_particles == 16000)].score_Linf.iloc[0])
    linf_h001 = float(sc[(sc.h == 0.01) & (sc.n_particles == 16000)].score_Linf.iloc[0])
    verdicts["score_bias_shrinks_with_h"] = bool(linf_h001 < linf_h002)

    # ------------------------------------------------------------------
    # 3. Constant-coefficient modal commutation (public APIs)
    # ------------------------------------------------------------------
    print("[3/5] constant-coefficient modal commutation", flush=True)
    cfg = copy.deepcopy(load_config(str(REPO / "configs" / "gaussian_base.yaml")))
    cfg.domain.n_grid, cfg.heat.alpha, cfg.heat.T = 400, ALPHA, 0.15
    x = make_grid(cfg)
    ks = wave_numbers(400, LENGTH)
    lam = 1e-4
    rows = []
    for n in (5, 25, 60):
        k = float(ks[n])
        mode = np.cos(k * (x - X_MIN))
        A = float(np.exp(-ALPHA * k ** 2 * cfg.heat.T))
        fwd_res = float(np.max(np.abs(forward_heat_solve_dct(mode, x, cfg) - A * mode)))
        tik = tikhonov_inverse(mode, x, ALPHA, cfg.heat.T, lam, length=LENGTH).candidate
        tik_res = float(np.max(np.abs(tik - (A / (A ** 2 + lam)) * mode)))
        cut = spectral_cutoff_inverse(mode, x, ALPHA, cfg.heat.T,
                                      k_cut=k + 1.0, length=LENGTH).candidate
        # Relative form A*cut - mode keeps the check meaningful when 1/A is huge.
        cut_res = float(np.max(np.abs(A * cut - mode)))
        rows.append({"mode": n, "forward_residual": fwd_res,
                     "tikhonov_residual": tik_res,
                     "cutoff_residual_relative": cut_res})
    tables["commutation_constant"] = pd.DataFrame(rows)
    cm = tables["commutation_constant"]
    verdicts["commutation<=1e-9"] = bool(
        cm[["forward_residual", "tikhonov_residual",
            "cutoff_residual_relative"]].to_numpy().max() <= 1e-9)

    # ------------------------------------------------------------------
    # 4a. Variable-operator structure
    # ------------------------------------------------------------------
    print("[4/5] variable operator: structure and manufactured convergence",
          flush=True)
    rows = []
    for m_op in (50, 100, 200, 400):
        x_op = cell_centers(X_MIN, X_MAX, m_op)
        L_op = vc.build_varcoeff_operator(x_op, ALPHA, 0.9).toarray()
        ones = np.ones(m_op)
        rng = np.random.default_rng(7)
        z, y = rng.standard_normal(m_op), rng.standard_normal(m_op)
        rows.append({
            "M": m_op,
            "symmetry": float(np.max(np.abs(L_op - L_op.T))),
            "nullspace": float(np.max(np.abs(L_op @ ones))),
            "column_sums": float(np.max(np.abs(ones @ L_op))),
            "max_eigenvalue": float(np.linalg.eigvalsh(0.5 * (L_op + L_op.T)).max()),
            "adjoint_residual": abs(float(z @ (L_op @ y)) - float((L_op @ z) @ y)),
        })
    tables["operator_structure"] = pd.DataFrame(rows)
    st = tables["operator_structure"]
    verdicts["op_symmetry==0"] = bool(st.symmetry.max() == 0.0)
    verdicts["op_nullspace<=1e-8"] = bool(st.nullspace.max() <= 1e-8)
    verdicts["op_conservation<=1e-8"] = bool(st.column_sums.max() <= 1e-8)
    verdicts["op_dissipative<=1e-8"] = bool(st.max_eigenvalue.max() <= 1e-8)
    verdicts["op_adjoint<=1e-7"] = bool(st.adjoint_residual.max() <= 1e-7)

    # ------------------------------------------------------------------
    # 4b. Variable-operator manufactured convergence (beta = 0.9)
    # ------------------------------------------------------------------
    BETA = 0.9
    T_MMS = 0.1

    def mms_exact(xx, t):
        return np.exp(-t) * (1.0 + 0.5 * np.cos(np.pi * xx))

    def mms_forcing(xx, t):
        v = 1.0 + 0.5 * np.cos(np.pi * xx)
        vp = -0.5 * np.pi * np.sin(np.pi * xx)
        vpp = -0.5 * np.pi ** 2 * np.cos(np.pi * xx)
        a = ALPHA * (1.0 + BETA * np.sin(2.0 * np.pi * xx))
        ap = 2.0 * np.pi * ALPHA * BETA * np.cos(2.0 * np.pi * xx)
        return np.exp(-t) * (-v - (ap * vp + a * vpp))

    def mms_solve(m_s, dt_s):
        x_s = cell_centers(X_MIN, X_MAX, m_s)
        u_T, _ = vc.solve_varcoeff_forward(
            mms_exact(x_s, 0.0), x_s, ALPHA, BETA, dt_s,
            exact_step_count(T_MMS, dt_s), forcing=mms_forcing)
        return u_T, x_s, cell_spacing(X_MIN, X_MAX, m_s)

    rows, errs = [], []
    dt_space = 1e-4
    for m_s in (25, 50, 100, 200):
        u_T, x_s, dxs = mms_solve(m_s, dt_space)
        e = midpoint_norm(u_T - mms_exact(x_s, T_MMS), dxs) \
            / midpoint_norm(mms_exact(x_s, T_MMS), dxs)
        errs.append(e)
        rows.append({"M": m_s, "dt": dt_space, "rel_error": e,
                     "order": np.nan if len(errs) < 2 else orders_of(errs[-2:])[0]})
    tables["mms_spatial"] = pd.DataFrame(rows)
    verdicts["mms_spatial_order>=1.9"] = bool(min(orders_of(errs)) >= 1.9)

    m_time = 400
    u_ref, x_t, dxt = mms_solve(m_time, 1.5625e-4)
    rows, errs_self = [], []
    for dt_s in (5e-3, 2.5e-3, 1.25e-3, 6.25e-4):
        u_T, _, _ = mms_solve(m_time, dt_s)
        e_exact = midpoint_norm(u_T - mms_exact(x_t, T_MMS), dxt) \
            / midpoint_norm(mms_exact(x_t, T_MMS), dxt)
        e_self = midpoint_norm(u_T - u_ref, dxt) / midpoint_norm(u_ref, dxt)
        errs_self.append(e_self)
        rows.append({"M": m_time, "dt": dt_s, "rel_error_vs_exact": e_exact,
                     "rel_error_vs_dt_ref": e_self,
                     "order_self": np.nan if len(errs_self) < 2
                     else orders_of(errs_self[-2:])[0]})
    tables["mms_temporal"] = pd.DataFrame(rows)
    verdicts["mms_temporal_order>=1.9"] = bool(min(orders_of(errs_self)) >= 1.9)

    # ------------------------------------------------------------------
    # 5. Exact-score particle refinement at fixed physical bandwidth
    # ------------------------------------------------------------------
    print("[5/5] exact-score particle refinement studies", flush=True)
    rows = []
    for h in (0.02, 0.01):
        floor = predicted_kernel_bias(h)
        for n_particles in (500, 1000, 2000, 4000, 8000):
            rel, mass_rel, _, _ = run_exact_score_particles(400, n_particles, 2e-3, h)
            rows.append({"h": h, "n_particles": n_particles, "rel_error": rel,
                         "mass_error": mass_rel, "predicted_bias_floor": floor,
                         "error_over_floor": rel / floor})
    tables["particle_N_refinement"] = pd.DataFrame(rows)
    nh = tables["particle_N_refinement"]
    final_ratio = float(nh[(nh.h == 0.02) & (nh.n_particles == 8000)]
                        .error_over_floor.iloc[0])
    verdicts["particle_N_floor_ratio_in_[0.7,1.5]"] = bool(0.7 <= final_ratio <= 1.5)
    verdicts["particle_mass<=1e-12"] = bool(np.max(np.abs(nh.mass_error)) <= 1e-12)

    h_dt, n_dt, m_dt = 0.02, 4000, 400
    _, _, recon_ref, x_p = run_exact_score_particles(m_dt, n_dt, 2.5e-4, h_dt)
    dxp = cell_spacing(X_MIN, X_MAX, m_dt)
    u0_norm = midpoint_norm(manufactured_u(x_p, 0.0), dxp)
    rows, errs_self = [], []
    for dt_p in (8e-3, 4e-3, 2e-3, 1e-3):
        rel, _, recon, _ = run_exact_score_particles(m_dt, n_dt, dt_p, h_dt)
        e_self = midpoint_norm(recon - recon_ref, dxp) / u0_norm
        errs_self.append(e_self)
        rows.append({"h": h_dt, "n_particles": n_dt, "M": m_dt, "dt": dt_p,
                     "rel_error_vs_truth": rel, "rel_error_vs_dt_ref": e_self,
                     "order_self": np.nan if len(errs_self) < 2
                     else orders_of(errs_self[-2:])[0]})
    tables["particle_dt_refinement"] = pd.DataFrame(rows)
    verdicts["particle_dt_order_in_[0.8,1.4]"] = bool(
        all(0.8 <= o <= 1.4 for o in orders_of(errs_self)))

    rows = []
    for m_g in (100, 200, 400, 800):
        rel, mass_rel, _, _ = run_exact_score_particles(m_g, 4000, 2e-3, 0.02)
        rows.append({"h": 0.02, "n_particles": 4000, "dt": 2e-3, "M": m_g,
                     "rel_error": rel, "mass_error": mass_rel})
    tables["particle_M_refinement"] = pd.DataFrame(rows)
    spread = tables["particle_M_refinement"].rel_error
    verdicts["particle_M_grid_independent(spread<=5%)"] = bool(
        (spread.max() - spread.min()) / spread.min() <= 0.05)

    # ------------------------------------------------------------------
    # Write outputs, dossier, and manifest
    # ------------------------------------------------------------------
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit(REPO),
        "grid_convention": GRID_CONVENTION,
        "manufactured_solutions": {
            "particle": (f"u = 1 + {A_MODE} exp(-alpha k^2 t) cos(k x), "
                         f"k = {MODE_N} pi / L, alpha = {ALPHA}, T = {T_PART}"),
            "variable": "u = exp(-t)(1 + 0.5 cos(pi x)), beta = 0.9, forced CN",
        },
        "verdicts": verdicts,
        "tables": {k: json.loads(v.to_json(orient="records"))
                   for k, v in tables.items()},
        "runtime_seconds": time.perf_counter() - t_start,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                      encoding="utf-8")

    passed = sum(verdicts.values())
    section_text = {
        "kernel_verification": "Neumann kernel: unit mass through the constant "
            "mode, termwise zero wall derivative (evaluated exactly at the "
            "walls), and modal damping equal to exp(-(kh)^2/2) at machine "
            "precision.",
        "score_static": "Neumann KDE score of quantile particles against the "
            "analytic score of the manufactured density (epsilon = 0; the "
            "density is bounded below by 0.5, so no floor is involved).  The "
            "error is bandwidth bias; halving h shrinks it.",
        "commutation_constant": "The forward multiplier, Tikhonov filter, and "
            "spectral cutoff act exactly on discrete cosine modes through the "
            "public APIs.  The cutoff column is the relative residual "
            "|A * output - mode| because 1/A is astronomically large at high "
            "modes.",
        "operator_structure": "The conservative finite-volume operator is "
            "exactly symmetric, annihilates constants, conserves mass, is "
            "negative semidefinite, and satisfies the adjoint identity.",
        "mms_spatial": "Forced manufactured solution, spatial refinement at "
            "fixed dt = 1e-4: second-order convergence against the exact "
            "solution.",
        "mms_temporal": "Forced manufactured solution, temporal refinement at "
            "fixed M = 400: second-order self-convergence against the "
            "dt-reference.  The vs-exact column floors at the spatial error, "
            "as it must.",
        "particle_N_refinement": "Exact-score density particles at fixed "
            "physical bandwidth: the particle-count error converges to the "
            "analytically predicted kernel-bias floor and the reconstruction "
            "mass is exact by construction.",
        "particle_dt_refinement": "First-order (explicit Euler) "
            "self-convergence in the time step at fixed h, N, M.  The "
            "vs-truth column floors at the kernel bias.",
        "particle_M_refinement": "Grid refinement at fixed physical bandwidth "
            "leaves the error unchanged: the endpoint-era h = c * dx coupling "
            "that confounded the archived grid study is gone.",
    }
    lines = [
        "# Operator-verification dossier",
        "",
        f"Code commit `{summary['code_commit'][:12]}`, grid convention "
        f"`{GRID_CONVENTION}`, generated {summary['created_utc']}.",
        "",
        f"**Verdicts: {passed}/{len(verdicts)} PASS.**  Every reference value "
        "is analytic on the bounded Neumann domain.  No table compares one "
        "simulation against another, except the two self-convergence columns "
        "that isolate temporal order from spatial and kernel floors.",
        "",
    ]
    for name, df in tables.items():
        lines += [f"## {name}", "", section_text[name], "",
                  "```", fmt_table(df), "```", ""]
    lines += ["## Verdicts", ""]
    lines += [f"- {'PASS' if v else 'FAIL'}  {k}" for k, v in verdicts.items()]
    lines.append("")
    (out / "dossier.md").write_text("\n".join(lines), encoding="utf-8")

    manifest_path = write_manifest(
        REPO / "outputs" / f"run_manifest_verification_{ts}.json",
        {"verification_dossier": out},
        [{"study": "verification_dossier",
          "argv": [sys.executable] + sys.argv,
          "elapsed_seconds": summary["runtime_seconds"]}],
        REPO,
        run_id=f"verification_dossier_{ts}",
        extra={"grid_convention": GRID_CONVENTION},
    )

    print(f"\nVerdicts: {passed}/{len(verdicts)} PASS")
    for k, v in verdicts.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"Wrote: {out}")
    print(f"[provenance] {manifest_path}")
    if passed != len(verdicts):
        sys.exit(1)


if __name__ == "__main__":
    main()
