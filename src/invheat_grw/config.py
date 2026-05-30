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
from typing import Dict


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
    type: str       # currently only "gaussian" supported
    mu: float
    sigma0: float
    amplitude: float


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


@dataclass
class SafetyConfig:
    score_abs_fail_threshold: float
    value_abs_fail_threshold: float


@dataclass
class Config:
    domain: DomainConfig
    heat: HeatConfig
    initial_condition: InitialConditionConfig
    grw: GRWConfig
    experiments: ExperimentsConfig
    safety: SafetyConfig

    @property
    def n_steps(self) -> int:
        """Number of backward integration steps."""
        return round(self.heat.T / self.heat.dt)


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
        initial_condition=InitialConditionConfig(
            type=str(ic["type"]),
            mu=float(ic["mu"]),
            sigma0=float(ic["sigma0"]),
            amplitude=float(ic["amplitude"]),
        ),
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
        ),
        safety=SafetyConfig(
            score_abs_fail_threshold=float(s["score_abs_fail_threshold"]),
            value_abs_fail_threshold=float(s["value_abs_fail_threshold"]),
        ),
    )
