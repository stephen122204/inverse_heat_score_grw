"""
make_figures.py — Generate all paper figures FROM REAL EXPERIMENT OUTPUT.

Manuscript labels (draft) per --only key: naive `fig:naive`, convergence
`fig:representation_convergence`, loop `fig:density_loop`, bandwidth
`fig:bandwidth_sweep`, particle_count `fig:particle_count`, variable_field
`fig:variable_field`, variable_results `fig:variable_results`, vh_mixture
`fig:vh_mixture`.

This script reads the CSV files written by the experiment scripts in
``outputs/`` and regenerates every quantitative figure so that the figures
match the numbers in the paper tables exactly.  Field-illustration figures
(naive reversal, density-particle loop, variable-coefficient field) load
their reconstruction arrays from the manifest's frozen ``figure_data`` study
(see figure_data.py); they are never recomputed here, so archived figures
cannot silently mix with later method changes.

Nothing in this file hardcodes experiment results.  The only literals are
plot styling.

Usage
-----
    python make_figures.py
    python make_figures.py --manifest manifests/paper_v5_1.json

Required experiment outputs (run these first):
    scripts/run_representation_audit.py          -> representation_audit_*
    scripts/run_validation_stage.py              -> validation_stage_*
    scripts/run_score_estimation_audit.py        -> score_estimation_audit_*
    scripts/run_variable_coefficient_audit.py    -> variable_coefficient_audit_*
    scripts/run_vh_mixture_bandwidth_refinement.py -> vh_mixture_bandwidth_refinement_*
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

import figure_data as figdata
from provenance import DEFAULT_MANIFEST, load_manifest, study_dir

OUT = REPO / "figures"
OUT.mkdir(exist_ok=True)

import figstyle
figstyle.apply_paper_style()

# Unified color key (one method = one color in every figure) lives in figstyle.py.
C_TRUTH = figstyle.TRUTH
C_OBS = figstyle.OBS
C_SL = figstyle.METHOD
C_SL6 = figstyle.METHOD_ALT
C_FD = figstyle.GRID_RATIO
C_TIK = figstyle.TIKH
C_EXACT = figstyle.EXACT
C_GLOB = figstyle.GLOB
C_NAIVE = figstyle.NAIVE
TEST_MARKER = figstyle.TEST_MARKER
TEST_LS = figstyle.TEST_LS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_MANIFEST = None


def selected_study(key: str) -> Path:
    global _MANIFEST
    if _MANIFEST is None:
        _, _MANIFEST = load_manifest(DEFAULT_MANIFEST)
    return study_dir(_MANIFEST, key, REPO)


def read_csv(d: Path | None, name: str) -> pd.DataFrame | None:
    if d is None:
        return None
    p = d / name
    return pd.read_csv(p) if p.exists() else None


_FIGURE_DATA = None


def frozen_figure(fig_key: str) -> dict[str, np.ndarray]:
    """Frozen reconstruction arrays for one illustration figure (no fallback)."""
    global _FIGURE_DATA
    if _FIGURE_DATA is None:
        _FIGURE_DATA = figdata.load_dataset(selected_study(figdata.STUDY_KEY))
    return _FIGURE_DATA[fig_key]


def save(fig, stem: str):
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=220)
    plt.close(fig)
    print(f"  wrote figures/{stem}.pdf")


# ---------------------------------------------------------------------------
# Figure: naive reversal vs score-guided (REAL reconstructions, Test B)
# ---------------------------------------------------------------------------
def fig_naive():
    data = frozen_figure("fig_naive")
    x, u0, uT = data["x"], data["u0"], data["uT"]
    naive, score = data["naive"], data["score"]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(x, u0, color=C_TRUTH, label=r"true $u_0$", lw=2)
    ax.plot(x, uT, "--", color=C_OBS, label=r"observed $u_T$", lw=1.8)
    ax.plot(x, naive, ":", color=C_NAIVE, label="naive backward random walk", lw=2)
    ax.plot(x, score, "-.", color=C_EXACT, label="score-guided density particles", lw=1.8)
    ax.set_xlabel("$x$"); ax.set_ylabel("$u$")
    ax.grid(True, alpha=0.25); ax.legend()
    fig.tight_layout(); save(fig, "fig_naive_reversal_score_guided")


# ---------------------------------------------------------------------------
# Figure: grid x N convergence — the gradient-glob error is set by the
# representation.
# Left panel:  rel_L2 vs n_grid (density at largest N) — gradient-glob flat,
#              density exact-score drops far below.
# Right panel: rel_L2 vs N at n_grid=400 — gradient-glob level (N-independent),
#              density exact-score flat at its KDE reconstruction floor.
#              Refining neither grid nor N moves the gradient-glob error.
# ---------------------------------------------------------------------------
def fig_convergence():
    d = selected_study("representation_audit")
    df = read_csv(d, "representation_audit_metrics.csv")
    if df is None:
        print("  [skip] representation_audit metrics not found"); return
    if df.n_grid.nunique() < 2:
        print("  [skip] representation CSV is single-grid; run the grid x N sweep first")
        return
    tests = ["B", "H", "Z"]
    Nmax = int(df.n_particles[df.n_particles > 0].max())

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    # ---- Panel A: vs n_grid (density at N = Nmax, bw=4) ----
    for t in tests:
        gg = (df[(df.test == t) & (df.method_type == "gradient_glob_oracle")]
              .groupby("n_grid").relative_l2.mean().sort_index())
        dp = (df[(df.test == t) & (df.method_type == "density_particle_oracle")
                 & (df.bandwidth_factor == 4.0) & (df.n_particles == Nmax)]
              .groupby("n_grid").relative_l2.mean().sort_index())
        axA.plot(gg.index, gg.values, ls=TEST_LS[t], marker=TEST_MARKER[t],
                 color=C_GLOB, alpha=0.9, label=f"{t}: gradient-glob")
        axA.plot(dp.index, dp.values, ls=TEST_LS[t], marker=TEST_MARKER[t],
                 color=C_EXACT, label=f"{t}: density, exact score")
    axA.set_yscale("log"); axA.set_xscale("log", base=2)
    axA.set_xticks([100, 200, 400, 800]); axA.set_xticklabels([100, 200, 400, 800])
    axA.set_xlabel(r"$n_\mathrm{grid}$ (at $N=%d$)" % Nmax)
    axA.set_ylabel(r"relative $L^2$ error")
    axA.set_title("(a)", loc="left")
    axA.grid(True, which="both", alpha=0.25); axA.legend(fontsize=8, ncol=1)

    # ---- Panel B: vs N at n_grid=400 ----
    for t in tests:
        dp = (df[(df.test == t) & (df.method_type == "density_particle_oracle")
                 & (df.bandwidth_factor == 4.0) & (df.n_grid == 400)
                 & (df.n_particles > 0)]
              .groupby("n_particles").relative_l2.mean().sort_index())
        axB.plot(dp.index, dp.values, ls=TEST_LS[t], marker=TEST_MARKER[t],
                 color=C_EXACT, label=f"{t}: density, exact score")
        gg = df[(df.test == t) & (df.method_type == "gradient_glob_oracle")
                & (df.n_grid == 400)].relative_l2.mean()
        axB.axhline(gg, ls=TEST_LS[t], color=C_GLOB, alpha=0.9, lw=1.2,
                    label=f"{t}: gradient-glob reference")
        axB.annotate(t, xy=(0.985, gg), xycoords=("axes fraction", "data"),
                     ha="right", va="bottom", fontsize=7, color=C_GLOB)
    axB.set_yscale("log"); axB.set_xscale("log")
    axB.set_xlabel(r"$N$ density particles (at $n_\mathrm{grid}=400$)")
    axB.set_ylabel(r"relative $L^2$ error")
    axB.set_title("(b)", loc="left")
    axB.grid(True, which="both", alpha=0.25); axB.legend(fontsize=8)

    fig.tight_layout(); save(fig, "fig_representation_convergence")


# ---------------------------------------------------------------------------
# Figure: density-particle reconstruction loop (REAL snapshots, Test B)
# ---------------------------------------------------------------------------
def fig_density_loop():
    data = frozen_figure("fig_density_loop")
    x, u0, uT = data["x"], data["u0"], data["uT"]
    dt = float(data["dt"])
    keys = [int(k) for k in data["snapshot_steps"]]
    snaps = {k: data["snapshots"][i] for i, k in enumerate(keys)}
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.plot(x, u0, color=C_TRUTH, lw=2, label=r"true $u_0$")
    ax.plot(x, uT, "--", color=C_OBS, lw=1.2, label=r"observed $u_T$")
    mid_key = min(keys, key=lambda k: abs(k - keys[-1] // 2))
    pick = [keys[0], mid_key, keys[-1]]
    shades = figstyle.LOOP_SHADES
    for k, col, ls in zip(pick, shades, [":", "--", "-"]):
        tau = k * dt
        ax.plot(x, snaps[k], ls=ls, lw=1.7, color=col, label=rf"$\tau={tau:.3f}$")
    ax.set_xlabel("$x$"); ax.set_ylabel("$u$")
    ax.grid(True, alpha=0.25); ax.legend()
    fig.tight_layout(); save(fig, "fig_density_particle_loop")


# ---------------------------------------------------------------------------
# Figure: bandwidth sweep U-curves (REAL data, B/H/Z, production smoothed_log)
# ---------------------------------------------------------------------------
def fig_bandwidth_sweep():
    d = selected_study("score_estimation_audit")
    df = read_csv(d, "score_estimation_audit_metrics.csv")
    if df is None:
        print("  [skip] score_estimation_audit metrics not found"); return
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    labels = {"B": "B: Gaussian, $T=0.15$", "H": "H: mixture, $T=0.15$",
              "Z": "Z: near-zero, $T=0.05$"}
    method = "smoothed_log"
    for t in ["B", "H", "Z"]:
        sub = df[(df.test == t) & (df.score_method == method)].sort_values("bandwidth_factor")
        if sub.empty:
            continue
        ax.plot(sub.bandwidth_factor, sub.relative_l2, color=C_SL,
                ls=TEST_LS[t], marker=TEST_MARKER[t], label=labels[t])
        orc = df[(df.test == t) & (df.score_method == "oracle")]
        if not orc.empty:
            floor = float(orc.relative_l2.min())
            ax.axhline(floor, ls="--", lw=0.9, alpha=0.7, color=C_EXACT)
            ax.annotate(t, xy=(0.985, floor), xycoords=("axes fraction", "data"),
                        ha="right", va="bottom", fontsize=7, color=C_EXACT)
    ax.plot([], [], ls="--", lw=0.9, color=C_EXACT, label="exact-score error (per test)")
    ax.set_yscale("log"); ax.set_xlabel(r"bandwidth factor $h/\Delta x$")
    ax.set_ylabel(r"relative $L^2$ error")
    ax.grid(True, which="both", alpha=0.25); ax.legend()
    fig.tight_layout(); save(fig, "fig_bandwidth_sweep")


# ---------------------------------------------------------------------------
# Figure: particle-count refinement (REAL data from validation N-convergence)
# ---------------------------------------------------------------------------
def fig_particle_count():
    d = selected_study("validation_stage")
    df = read_csv(d, "validation_metrics.csv")
    if df is None:
        print("  [skip] validation metrics not found"); return
    nc = df[df.task.str.contains("conv", case=False, na=False)]
    if nc.empty:
        print("  [skip] no n-convergence rows"); return
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for label, key, col in [(r"smoothed-log, $h/\Delta x = 4$", "smoothed_log", C_SL),
                            (r"grid-ratio, $h/\Delta x = 4$", "fd_grid_ratio", C_FD)]:
        sub = nc[(nc.score_method == key) & (nc.bandwidth_factor == 4.0)].sort_values("n_particles")
        if sub.empty:
            continue
        ax.plot(sub.n_particles, sub.rel_L2, marker="o", color=col, label=label)
    ax.set_xscale("log"); ax.set_xlabel("number of density particles $N$")
    ax.set_ylabel(r"relative $L^2$ error")
    ax.grid(True, which="both", alpha=0.25); ax.legend()
    fig.tight_layout(); save(fig, "fig_particle_count_refinement")


# ---------------------------------------------------------------------------
# Variable-coefficient figures (REAL data)
# ---------------------------------------------------------------------------
def fig_variable_field():
    data = frozen_figure("fig_variable_field")
    x, u0, uT = data["x"], data["u0"], data["uT"]
    candidate, a = data["candidate"], data["a"]

    fig, ax1 = plt.subplots(figsize=(6.6, 4.0))
    ax1.plot(x, u0, color=C_TRUTH, lw=2, label=r"true $u_0$")
    ax1.plot(x, uT, "--", color=C_OBS, label=r"observed $u_T$")
    ax1.plot(x, candidate, "-.", color=C_SL, label=r"particle reconstruction ($h/\Delta x = 4$)")
    ax1.set_xlabel("$x$"); ax1.set_ylabel("$u$"); ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx(); ax2.plot(x, a, ":", lw=1.5, color="0.35", label="$a(x)$"); ax2.set_ylabel("$a(x)$")
    ax2.grid(False)
    l1, lab1 = ax1.get_legend_handles_labels(); l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="upper right")
    fig.tight_layout(); save(fig, "fig_variable_coefficient_field")


def fig_variable_results():
    d = selected_study("variable_coefficient_audit")
    df = read_csv(d, "variable_coeff_metrics.csv")
    if df is None:
        print("  [skip] variable_coeff metrics not found"); return
    cases = ["VB_beta05", "VB_beta09", "VH_beta05", "VH_beta09"]
    disp = ["Gaussian\n" + r"$\beta=0.5$", "Gaussian\n" + r"$\beta=0.9$",
            "Mixture\n" + r"$\beta=0.5$", "Mixture\n" + r"$\beta=0.9$"]

    def get(case, method):
        s = df[(df.case == case) & (df.method == method)]
        return float(s.relative_l2.iloc[0]) if not s.empty else np.nan

    oracle = [get(c, "variable_oracle_deterministic") for c in cases]
    est = [get(c, "variable_estimated_smoothed_log_bw4") for c in cases]
    tikh = [max(get(c, "varcoeff_tikhonov_best"), 1e-4) for c in cases]
    xp = np.arange(len(cases)); w = 0.25
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.bar(xp - w, oracle, w, color=C_EXACT, label="exact-score particles")
    ax.bar(xp, est, w, color=C_SL, label=r"smoothed-log, $h/\Delta x = 4$")
    ax.bar(xp + w, tikh, w, color=C_TIK, label="Tikhonov (optimal $\\lambda$)")
    ax.set_yscale("log"); ax.set_xticks(xp); ax.set_xticklabels(disp)
    ax.set_ylabel(r"relative $L^2$ error (log scale)")
    ax.grid(True, axis="y", which="both", alpha=0.25); ax.grid(False, axis="x")
    ax.legend()
    fig.tight_layout(); save(fig, "fig_variable_coefficient_results")


def fig_vh_mixture():
    d = selected_study("vh_mixture_bandwidth")
    df = read_csv(d, "vh_mixture_bandwidth_metrics.csv")
    if df is None:
        print("  [skip] vh_mixture metrics not found"); return
    sl = df[(df.case == "VH_beta05") & (df.method == "smoothed_log")].sort_values("bandwidth_factor")
    orc = df[(df.case == "VH_beta05") & (df.method == "oracle")]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(sl.bandwidth_factor, sl.relative_l2, marker="o", color=C_SL,
            label="smoothed-log (estimated score)")
    if not orc.empty:
        ax.axhline(float(orc.relative_l2.iloc[0]), ls="--", lw=1, color=C_EXACT,
                   label=f"exact-score level = {float(orc.relative_l2.iloc[0]):.3f}")
    ax.set_yscale("log")
    ax.set_xlabel(r"bandwidth factor $h/\Delta x$"); ax.set_ylabel(r"relative $L^2$ error")
    ax.grid(True, which="both", alpha=0.25); ax.legend()
    fig.tight_layout(); save(fig, "fig_vh_mixture_bandwidth_refinement")


def main():
    global _MANIFEST
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of figure keys to build")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST),
                    help="explicit run manifest; defaults to the archived paper-v5.1 manifest")
    args = ap.parse_args()
    manifest_path, _MANIFEST = load_manifest(args.manifest)
    print(f"[provenance] {manifest_path}")
    figs = {
        "naive": fig_naive,
        "convergence": fig_convergence,
        "loop": fig_density_loop,
        "bandwidth": fig_bandwidth_sweep,
        "particle_count": fig_particle_count,
        "variable_field": fig_variable_field,
        "variable_results": fig_variable_results,
        "vh_mixture": fig_vh_mixture,
    }
    keys = args.only or list(figs)
    for k in keys:
        print(f"[fig] {k}")
        try:
            figs[k]()
        except Exception as e:  # keep going; report which figure failed
            print(f"  [ERROR] {k}: {e}")
    print("done.")


if __name__ == "__main__":
    main()
