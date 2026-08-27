"""figure_data.py — frozen field arrays behind the four illustration figures.

The naive-reversal, density-loop, variable-field, and representation-failure
figures plot reconstructions rather than archived CSV columns.  If the figure
builders recomputed those reconstructions at build time, any later change to
the numerical methods would silently leak into figures that must keep showing
the archived paper's results.  The arrays are therefore computed once, stored
as a manifest-pinned study with hashes, and loaded by the builders with no
recomputation fallback.

Entry points:
    scripts/generate_figure_data.py   create the dataset for a manifest
    scripts/verify_figure_freeze.py   recompute all four figures' data and
                                      compare with an archived dataset
"""

from __future__ import annotations

import copy
import importlib.util
import json
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from invheat_grw.config import load_config, Config
from invheat_grw.fields import make_grid, true_u0, observed_final
from invheat_grw.methods import (
    run_naive_backward,
    run_oracle_score_deterministic,
    run_density_particle_oracle_score_deterministic,
    run_density_particle_estimated_score_deterministic,
)

DATA_FILE = "figure_data.npz"
META_FILE = "figure_data_meta.json"
STUDY_KEY = "figure_data"
FIGURES = (
    "fig_naive",
    "fig_density_loop",
    "fig_variable_field",
    "fig_representation_failure",
)

GAUSS_CFG = REPO / "configs" / "gaussian_base.yaml"


def _patch(cfg: Config, **ov) -> Config:
    cfg = copy.deepcopy(cfg)
    for k, v in ov.items():
        obj = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], v)
    return cfg


def _test_b_config() -> Config:
    return _patch(load_config(str(GAUSS_CFG)),
                  **{"heat.T": 0.15, "initial_condition.sigma0": 0.08,
                     "domain.n_grid": 400})


def _import_varcoeff():
    spec = importlib.util.spec_from_file_location(
        "vc_audit", str(REPO / "scripts" / "run_variable_coefficient_audit.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# The four frozen computations.  Each reproduces, operation for operation,
# what the corresponding figure builder computed inline before the freeze.
# ---------------------------------------------------------------------------

def compute_fig_naive() -> dict[str, np.ndarray]:
    """Test B naive backward walk vs exact-score density particles."""
    cfg = _test_b_config()
    x = make_grid(cfg)
    u0 = true_u0(x, cfg)
    uT = observed_final(x, cfg)
    rng = np.random.default_rng(0)
    naive = run_naive_backward(uT, x, cfg, rng).candidate
    score = run_density_particle_oracle_score_deterministic(
        uT, x, cfg, n_particles=5000, recon_method="kde",
        bandwidth_factor=4.0).candidate
    return {"x": x, "u0": u0, "uT": uT, "naive": naive, "score": score,
            "rng_seed": np.array(0)}


def compute_fig_density_loop() -> dict[str, np.ndarray]:
    """Test B smoothed-log reconstruction snapshots through reverse time."""
    cfg = _test_b_config()
    x = make_grid(cfg)
    u0 = true_u0(x, cfg)
    uT = observed_final(x, cfg)
    res = run_density_particle_estimated_score_deterministic(
        uT, x, cfg, n_particles=5000, recon_method="kde", bandwidth_factor=4.0,
        epsilon=1e-8, score_method="smoothed_log", smooth_sigma_factor=1.0,
        save_snapshots=True)
    steps = np.array(sorted(res.step_snapshots), dtype=np.int64)
    snapshots = np.stack([res.step_snapshots[int(k)] for k in steps])
    return {"x": x, "u0": u0, "uT": uT, "snapshot_steps": steps,
            "snapshots": snapshots, "dt": np.array(cfg.heat.dt)}


def compute_fig_variable_field() -> dict[str, np.ndarray]:
    """Variable-diffusivity field, forward solution, and reconstruction."""
    vc = _import_varcoeff()
    cfg = _patch(load_config(str(GAUSS_CFG)),
                 **{"domain.n_grid": 400, "heat.T": 0.15, "heat.dt": 0.001})
    alpha0, beta = cfg.heat.alpha, 0.9
    dt, n_steps = cfg.heat.dt, cfg.n_steps
    x = make_grid(cfg)
    u0 = true_u0(x, cfg)
    uT, snaps = vc.solve_varcoeff_forward(u0, x, alpha0, beta, dt, n_steps)
    r = vc.run_varcoeff_estimated(
        uT, x, snaps, alpha0, beta, dt, n_steps,
        score_method="smoothed_log", bandwidth_factor=4.0, n_particles=10000,
        x_min=float(cfg.domain.x_min), x_max=float(cfg.domain.x_max))
    a = vc.a_of_x(x, alpha0, beta)
    return {"x": x, "u0": u0, "uT": uT,
            "candidate": np.asarray(r["candidate"], dtype=float), "a": a,
            "beta": np.array(beta)}


def compute_fig_representation_failure() -> dict[str, np.ndarray]:
    """Test B exact-score density particles vs exact-score gradient globs."""
    cfg = _test_b_config()
    x = make_grid(cfg)
    u0 = true_u0(x, cfg)
    g = observed_final(x, cfg)
    dx = float(x[1] - x[0])
    u0n = float(np.sqrt(dx * np.sum(u0 ** 2)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dp = run_density_particle_oracle_score_deterministic(
            g, x, cfg, 5000, recon_method="kde", bandwidth_factor=4.0).candidate
        cfg_gg = _patch(cfg, **{"grw.gradient_globs_per_jump": 80})
        gg = run_oracle_score_deterministic(
            g, x, cfg_gg, np.random.default_rng(42)).candidate

    def e2(c: np.ndarray) -> float:
        return float(np.sqrt(dx * np.sum((c - u0) ** 2))) / u0n

    return {"x": x, "u0": u0, "g": g, "density": dp, "glob": gg,
            "e2_density": np.array(e2(dp)), "e2_glob": np.array(e2(gg)),
            "rng_seed": np.array(42)}


COMPUTERS = {
    "fig_naive": compute_fig_naive,
    "fig_density_loop": compute_fig_density_loop,
    "fig_variable_field": compute_fig_variable_field,
    "fig_representation_failure": compute_fig_representation_failure,
}

PARAMETERS = {
    "fig_naive": {"test": "B", "T": 0.15, "sigma0": 0.08, "n_grid": 400,
                  "n_particles": 5000, "recon_method": "kde",
                  "bandwidth_factor": 4.0, "naive_rng_seed": 0},
    "fig_density_loop": {"test": "B", "T": 0.15, "sigma0": 0.08, "n_grid": 400,
                         "n_particles": 5000, "recon_method": "kde",
                         "bandwidth_factor": 4.0, "epsilon": 1e-8,
                         "score_method": "smoothed_log",
                         "smooth_sigma_factor": 1.0},
    "fig_variable_field": {"ic": "gaussian_base", "T": 0.15, "dt": 0.001,
                           "n_grid": 400, "beta": 0.9,
                           "score_method": "smoothed_log",
                           "bandwidth_factor": 4.0, "n_particles": 10000},
    "fig_representation_failure": {"test": "B", "T": 0.15, "sigma0": 0.08,
                                   "n_grid": 400, "density_n_particles": 5000,
                                   "density_bandwidth_factor": 4.0,
                                   "gradient_globs_per_jump": 80,
                                   "glob_rng_seed": 42},
}


def compute_all() -> dict[str, dict[str, np.ndarray]]:
    return {name: fn() for name, fn in COMPUTERS.items()}


# ---------------------------------------------------------------------------
# Save / load / compare
# ---------------------------------------------------------------------------

def save_dataset(data: dict[str, dict[str, np.ndarray]], out_dir: Path,
                 command: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    flat = {f"{fig}/{name}": np.asarray(arr)
            for fig, entries in data.items() for name, arr in entries.items()}
    np.savez(out_dir / DATA_FILE, **flat)

    from provenance import git_commit
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "code_commit": git_commit(REPO),
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "figures": {fig: sorted(entries) for fig, entries in data.items()},
        "parameters": PARAMETERS,
    }
    (out_dir / META_FILE).write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_dataset(study_path: Path) -> dict[str, dict[str, np.ndarray]]:
    """Load a frozen dataset back into {figure: {name: array}} form."""
    nested: dict[str, dict[str, np.ndarray]] = {}
    with np.load(study_path / DATA_FILE) as archive:
        for key in archive.files:
            fig, _, name = key.partition("/")
            nested.setdefault(fig, {})[name] = archive[key]
    missing = [fig for fig in FIGURES if fig not in nested]
    if missing:
        raise ValueError(f"Frozen figure data is missing: {', '.join(missing)}")
    return nested


def compare_datasets(live: dict[str, dict[str, np.ndarray]],
                     frozen: dict[str, dict[str, np.ndarray]],
                     tol: float = 0.0) -> list[str]:
    """Compare datasets; tol=0 demands bitwise equality, tol>0 a max-abs bound
    relative to max(1, max|frozen|).  Returns a list of mismatch messages."""
    problems: list[str] = []
    for fig in FIGURES:
        live_keys = set(live.get(fig, {}))
        frozen_keys = set(frozen.get(fig, {}))
        for name in sorted(frozen_keys - live_keys):
            problems.append(f"{fig}/{name}: missing from live recomputation")
        for name in sorted(live_keys - frozen_keys):
            problems.append(f"{fig}/{name}: missing from frozen dataset")
        for name in sorted(live_keys & frozen_keys):
            a = np.asarray(live[fig][name])
            b = np.asarray(frozen[fig][name])
            if a.shape != b.shape:
                problems.append(f"{fig}/{name}: shape {a.shape} != {b.shape}")
                continue
            if tol <= 0.0:
                if not np.array_equal(a, b):
                    worst = float(np.max(np.abs(a - b))) if a.size else 0.0
                    problems.append(f"{fig}/{name}: not bitwise equal "
                                    f"(max abs diff {worst:.3e})")
            else:
                scale = max(1.0, float(np.max(np.abs(b))) if b.size else 1.0)
                worst = float(np.max(np.abs(a - b))) if a.size else 0.0
                if worst > tol * scale:
                    problems.append(f"{fig}/{name}: max abs diff {worst:.3e} "
                                    f"> {tol:g} * {scale:g}")
    return problems
