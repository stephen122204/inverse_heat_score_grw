"""
config.py — Load and validate experiment configuration from YAML.

Uses dataclasses for structured, type-annotated config objects.
All numerical parameters are kept as plain Python floats/ints so
downstream code can modify them programmatically without YAML coupling.
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class GaussianComponentConfig:
    """One component of a Gaussian mixture initial condition."""
    amplitude: float
    mu: float
    sigma0: float


@dataclass
class DomainConfig:
    x_min: float
    x_max: float
    n_grid: int


@dataclass
class HeatConfig:
    alpha: float   # diffusivity
    T: float       # total forward time (observation time)
    dt: float      # backward integration time step


@dataclass
class InitialConditionConfig:
    type: str       # "gaussian" or "gaussian_mixture"
    # Single-Gaussian fields (used when type == "gaussian")
    mu: float = 0.5
    sigma0: float = 0.08
    amplitude: float = 1.0
    # Mixture fields (used when type == "gaussian_mixture")
    background: float = 0.0
    components: List[GaussianComponentConfig] = field(default_factory=list)


@dataclass
class GRWConfig:
    gradient_globs_per_jump: int
    rng_seed: int
    boundary: str   # "reflecting" only for now


@dataclass
class ExperimentsConfig:
    run_naive_backward: bool
    run_oracle_score_deterministic: bool
    run_oracle_score_stochastic: bool
    run_estimated_score_deterministic_raw: bool
    run_estimated_score_stochastic_raw: bool
    # New regularized methods (default False so old YAML configs keep working)
    run_estimated_score_deterministic_regularized: bool = False
    run_estimated_score_stochastic_regularized: bool = False


@dataclass
class SafetyConfig:
    score_abs_fail_threshold: float
    value_abs_fail_threshold: float


# ---------------------------------------------------------------------------
# Regularization configuration (Phase 5+)
# ---------------------------------------------------------------------------

@dataclass
class EpsilonFloorConfig:
    """Epsilon-floor regularization: denom = u + epsilon."""
    enabled: bool = False
    value: float = 1.0e-6
    scale_by_peak: bool = False   # if True, epsilon = value * max(u)


@dataclass
class ScoreClippingConfig:
    """Hard clip on estimated score magnitude: |s| clipped to max_abs_score."""
    enabled: bool = False
    max_abs_score: float = 100.0


@dataclass
class SmoothingConfig:
    """Field smoothing before score estimation (not yet fully implemented)."""
    enabled: bool = False
    method: str = "gaussian"          # "gaussian" only for now
    sigma_grid_points: float = 2.0    # smoothing sigma in grid-point units


@dataclass
class RegularizationConfig:
    """
    Top-level regularization config.
    If enabled=False, all regularization is skipped (methods behave like raw).
    Validation is performed on first use; see validate_regularization().
    """
    enabled: bool = False
    epsilon_floor: EpsilonFloorConfig = field(default_factory=EpsilonFloorConfig)
    score_clipping: ScoreClippingConfig = field(default_factory=ScoreClippingConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)


def validate_regularization(reg: RegularizationConfig) -> None:
    """Raise ValueError for invalid regularization settings."""
    if not reg.enabled:
        return
    if reg.epsilon_floor.enabled:
        if reg.epsilon_floor.value < 0.0:
            raise ValueError(f"epsilon_floor.value must be nonnegative, got {reg.epsilon_floor.value}")
    if reg.score_clipping.enabled:
        if reg.score_clipping.max_abs_score <= 0.0:
            raise ValueError(f"score_clipping.max_abs_score must be positive, got {reg.score_clipping.max_abs_score}")
    if not reg.epsilon_floor.enabled and not reg.score_clipping.enabled and not reg.smoothing.enabled:
        import warnings
        warnings.warn("regularization.enabled=True but no specific regularizer is enabled.", stacklevel=3)


def exact_step_count(T: float, dt: float, rel_tol: float = 1e-9) -> int:
    """Step count n with n * dt = T to within rel_tol * max(T, dt).

    Every integration in this project must land exactly on the requested
    final time.  A horizon that is not an integer multiple of the step is an
    explicit configuration error, never a silent early or late stop.
    """
    if not (T > 0.0 and dt > 0.0):
        raise ValueError(f"T and dt must be positive, got T={T}, dt={dt}")
    n = round(T / dt)
    if n < 1 or abs(n * dt - T) > rel_tol * max(T, dt):
        raise ValueError(
            f"time horizon T={T} is not an integer multiple of dt={dt} "
            f"(nearest n={n} gives n*dt={n * dt!r}); adjust T or dt"
        )
    return n


@dataclass
class Config:
    domain: DomainConfig
    heat: HeatConfig
    initial_condition: InitialConditionConfig
    grw: GRWConfig
    experiments: ExperimentsConfig
    safety: SafetyConfig
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)

    @property
    def n_steps(self) -> int:
        """Number of backward integration steps (n * dt = T exactly)."""
        return exact_step_count(self.heat.T, self.heat.dt)


def load_config(path: str | Path) -> Config:
    """Load a YAML config file and return a validated Config dataclass."""
    path = Path(path)
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh)

    d = raw["domain"]
    h = raw["heat"]
    ic = raw["initial_condition"]
    g = raw["grw"]
    e = raw["experiments"]
    s = raw["safety"]

    ic_type = str(ic["type"])
    if ic_type == "gaussian":
        ic_cfg = InitialConditionConfig(
            type="gaussian",
            mu=float(ic["mu"]),
            sigma0=float(ic["sigma0"]),
            amplitude=float(ic["amplitude"]),
        )
    elif ic_type == "gaussian_mixture":
        components = [
            GaussianComponentConfig(
                amplitude=float(c["amplitude"]),
                mu=float(c["mu"]),
                sigma0=float(c["sigma0"]),
            )
            for c in ic["components"]
        ]
        ic_cfg = InitialConditionConfig(
            type="gaussian_mixture",
            background=float(ic.get("background", 0.0)),
            components=components,
        )
    else:
        raise ValueError(f"Unknown initial_condition type: {ic_type!r}")

    return Config(
        domain=DomainConfig(
            x_min=float(d["x_min"]),
            x_max=float(d["x_max"]),
            n_grid=int(d["n_grid"]),
        ),
        heat=HeatConfig(
            alpha=float(h["alpha"]),
            T=float(h["T"]),
            dt=float(h["dt"]),
        ),
        initial_condition=ic_cfg,
        grw=GRWConfig(
            gradient_globs_per_jump=int(g["gradient_globs_per_jump"]),
            rng_seed=int(g["rng_seed"]),
            boundary=str(g["boundary"]),
        ),
        experiments=ExperimentsConfig(
            run_naive_backward=bool(e["run_naive_backward"]),
            run_oracle_score_deterministic=bool(e["run_oracle_score_deterministic"]),
            run_oracle_score_stochastic=bool(e["run_oracle_score_stochastic"]),
            run_estimated_score_deterministic_raw=bool(e["run_estimated_score_deterministic_raw"]),
            run_estimated_score_stochastic_raw=bool(e["run_estimated_score_stochastic_raw"]),
            run_estimated_score_deterministic_regularized=bool(
                e.get("run_estimated_score_deterministic_regularized", False)),
            run_estimated_score_stochastic_regularized=bool(
                e.get("run_estimated_score_stochastic_regularized", False)),
        ),
        safety=SafetyConfig(
            score_abs_fail_threshold=float(s["score_abs_fail_threshold"]),
            value_abs_fail_threshold=float(s["value_abs_fail_threshold"]),
        ),
        regularization=_parse_regularization(raw.get("regularization", {})),
    )


def _parse_regularization(reg_raw: dict) -> RegularizationConfig:
    """Parse the optional regularization YAML block."""
    if not reg_raw:
        return RegularizationConfig()
    ef = reg_raw.get("epsilon_floor", {})
    sc = reg_raw.get("score_clipping", {})
    sm = reg_raw.get("smoothing", {})
    cfg = RegularizationConfig(
        enabled=bool(reg_raw.get("enabled", False)),
        epsilon_floor=EpsilonFloorConfig(
            enabled=bool(ef.get("enabled", False)),
            value=float(ef.get("value", 1.0e-6)),
            scale_by_peak=bool(ef.get("scale_by_peak", False)),
        ),
        score_clipping=ScoreClippingConfig(
            enabled=bool(sc.get("enabled", False)),
            max_abs_score=float(sc.get("max_abs_score", 100.0)),
        ),
        smoothing=SmoothingConfig(
            enabled=bool(sm.get("enabled", False)),
            method=str(sm.get("method", "gaussian")),
            sigma_grid_points=float(sm.get("sigma_grid_points", 2.0)),
        ),
    )
    validate_regularization(cfg)
    return cfg
