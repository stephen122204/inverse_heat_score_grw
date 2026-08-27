"""
make_new_figures.py — representation-failure, noise-band, and nonsmooth figures.

Manuscript labels (draft): `fig:representation_failure`, `fig:noise`,
`fig:nonsmooth`.

All data come from real runs selected by an explicit manifest:
  fig_representation_failure_visual : Test B exact-score density-particle
        (N=5000, bw4, E2~0.0069) vs gradient-glob (E2~0.175), reproduced inline
        — the SAME runs behind Table 1.
  fig_noise_robustness_bands        : reads the manifest-selected noise study
        arrays; mean line + shaded +/-1 std band for smoothed-log bw4, bw6, and
        optimally tuned Tikhonov.
  fig_nonsmooth_reconstruction      : reads the manifest-selected nonsmooth arrays;
        tent true u0, best smoothed-log reconstruction, Tikhonov.

Nothing here hardcodes a numeric result; legends print E2 read from the data.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import copy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from invheat_grw.config import load_config, Config
from invheat_grw.fields import make_grid, true_u0, observed_final
from invheat_grw.methods import (
    run_oracle_score_deterministic,
    run_density_particle_oracle_score_deterministic,
)
from provenance import DEFAULT_MANIFEST, load_manifest, study_dir

OUT = REPO / "figures"
OUT.mkdir(exist_ok=True)

import figstyle
figstyle.apply_paper_style()


def patch(cfg: Config, **ov) -> Config:
    cfg = copy.deepcopy(cfg)
    for k, v in ov.items():
        obj = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], v)
    return cfg


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=220)
    plt.close(fig)
    print(f"  wrote figures/{stem}.pdf")


_MANIFEST = None


def selected_study(key: str) -> Path:
    global _MANIFEST
    if _MANIFEST is None:
        _, _MANIFEST = load_manifest(DEFAULT_MANIFEST)
    return study_dir(_MANIFEST, key, REPO)


# ---------------------------------------------------------------------------
# Gradient representation failure, shown visually (Test B)
# ---------------------------------------------------------------------------
def fig_representation_failure_visual():
    base = load_config(str(REPO / "configs" / "gaussian_base.yaml"))
    cfg = patch(base, **{"heat.T": 0.15, "initial_condition.sigma0": 0.08,
                         "domain.n_grid": 400})
    x = make_grid(cfg)
    u0 = true_u0(x, cfg)
    g = observed_final(x, cfg)
    dx = float(x[1] - x[0])
    u0n = float(np.sqrt(dx * np.sum(u0 ** 2)))

    def e2(c):
        return float(np.sqrt(dx * np.sum((c - u0) ** 2))) / u0n

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        # density-particle carrying u (the representation that matches)
        dp = run_density_particle_oracle_score_deterministic(
            g, x, cfg, 5000, recon_method="kde", bandwidth_factor=4.0).candidate
        # gradient-glob carrying q = u_x (the representation that fails)
        cfg_gg = patch(cfg, **{"grw.gradient_globs_per_jump": 80})
        gg = run_oracle_score_deterministic(
            g, x, cfg_gg, np.random.default_rng(42)).candidate

    e2_dp, e2_gg = e2(dp), e2(gg)
    print(f"  density E2={e2_dp:.4f}  gradient-glob E2={e2_gg:.4f}")

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(x, u0, "-", color=figstyle.TRUTH, lw=2.4, label=r"true $u_0$", zorder=3)
    ax.plot(x, dp, "--", color=figstyle.EXACT, lw=2.0, zorder=6,
            label=rf"density particles, carrying $u$ (exact score), $E_2={e2_dp:.4f}$")
    ax.plot(x, gg, "-", color=figstyle.GLOB, lw=1.9,
            label=rf"gradient globs, carrying $q=u_x$ (exact score), $E_2={e2_gg:.3f}$")
    ax.plot(x, g, "--", color=figstyle.OBS, lw=1.2, alpha=0.9,
            label=r"observed $g=u(\cdot,T)$")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u$")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    save(fig, "fig_representation_failure_visual")


# ---------------------------------------------------------------------------
# Noise study with error bands (mean +/- 1 std)
# ---------------------------------------------------------------------------
def fig_noise_robustness_bands():
    d = selected_study("noise_study")
    agg = pd.read_csv(d / "noise_study_summary.csv")
    series = [
        ("smoothed_log_bw4", r"smoothed-log, $h/\Delta x = 4$", figstyle.METHOD, "o", "-"),
        ("smoothed_log_bw6", r"smoothed-log, $h/\Delta x = 6$", figstyle.METHOD_ALT, "s", "-"),
        ("tikhonov_optimal", "optimally tuned Tikhonov", figstyle.TIKH, "^", "--"),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for key, lbl, col, mk, ls in series:
        sub = agg[agg.method == key].sort_values("eta")
        eta = sub.eta.to_numpy()
        mean = sub["mean"].to_numpy()
        std = sub["std"].to_numpy()
        ax.plot(eta, mean, marker=mk, ls=ls, color=col, label=lbl, lw=1.9)
        ax.fill_between(eta, np.maximum(mean - std, 1e-6), mean + std,
                        color=col, alpha=0.18, linewidth=0)
    ax.set_yscale("log")
    ax.set_xlabel(r"relative observation noise $\eta$")
    ax.set_ylabel(r"relative $L^2$ error $E_2$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save(fig, "fig_noise_robustness_bands")


# ---------------------------------------------------------------------------
# Non-smooth (tent) reconstruction vs truth
# ---------------------------------------------------------------------------
def fig_nonsmooth_reconstruction():
    d = selected_study("nonsmooth_case")
    arr = np.load(d / "nonsmooth_arrays.npz")
    met = pd.read_csv(d / "nonsmooth_metrics.csv")
    case = "tent"
    x = arr[f"{case}__x"]
    u0 = arr[f"{case}__u0"]

    def e2_of(method):
        r = met[(met.case == case) & (met.method == method)]
        return float(r.E2.iloc[0]) if not r.empty else float("nan")

    # best smoothed-log bw for tent
    best_bw = int(met[(met.case == case) & (met.method == "E_fwd_best_smoothed_log")].bw.iloc[0])
    sl_key = f"{case}__smoothed_log_bw{best_bw}"
    sl = arr[sl_key]
    tik = arr[f"{case}__tikhonov_optimal"]
    e2_sl = e2_of(f"smoothed_log_bw{best_bw}")
    e2_tik = e2_of("tikhonov_optimal")

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(x, u0, "-", color=figstyle.TRUTH, lw=2.4, label=r"true $u_0$ (tent)", zorder=5)
    ax.plot(x, sl, "-", color=figstyle.METHOD, lw=1.9,
            label=rf"smoothed-log, $h/\Delta x = {best_bw}$, $E_2={e2_sl:.3f}$")
    ax.plot(x, tik, "--", color=figstyle.TIKH, lw=1.7,
            label=rf"optimally tuned Tikhonov, $E_2={e2_tik:.3f}$")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u$")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "fig_nonsmooth_reconstruction")


def main():
    global _MANIFEST
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST),
                    help="explicit run manifest; defaults to the archived paper-v5.1 manifest")
    args = ap.parse_args()
    manifest_path, _MANIFEST = load_manifest(args.manifest)
    print(f"[provenance] {manifest_path}")
    print("[fig] representation_failure_visual")
    fig_representation_failure_visual()
    print("[fig] noise_robustness_bands")
    fig_noise_robustness_bands()
    print("[fig] nonsmooth_reconstruction")
    fig_nonsmooth_reconstruction()
    print("done.")


if __name__ == "__main__":
    main()
