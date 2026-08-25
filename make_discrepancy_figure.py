"""
make_discrepancy_figure.py — the discrepancy-principle comparison figure
(manuscript label, draft: `fig:discrepancy`).

Reads the latest outputs/discrepancy_principle_* and builds a two-panel figure:
 (a) realistic reconstruction overlay at eta=0.005 for the representative
     realization: true u0, discrepancy-tuned particle, discrepancy-tuned
     Tikhonov (E2 in legend).
 (b) realization-mean true E2 vs particle bandwidth factor at eta=0.005 (the
     noisy U-curve), with the mean discrepancy-chosen bandwidth and the
     bandwidth that minimizes the mean true error marked. The gap between them
     is the diagnostic.

No editorializing title; the caption carries the message. Writes to figures/
(and syncs to a sibling paper_draft/figures/ when that directory exists).
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent
PAPER_FIGS = REPO.parent / "paper_draft" / "figures"

import figstyle
figstyle.apply_paper_style()


def latest(pattern):
    dirs = [Path(h) for h in glob.glob(str(REPO / "outputs" / pattern)) if Path(h).is_dir()]
    return max(dirs, key=lambda p: p.name) if dirs else None


def main():
    d = latest("discrepancy_principle_*")
    if d is None:
        print("no discrepancy_principle_* dir"); return
    a = np.load(d / "discrepancy_arrays.npz")
    x, u0 = a["x"], a["u0"]
    eta = float(a["eta_fig"])
    rep_seed = int(a["rep_seed"])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # ---- panel (a): reconstruction overlay, discrepancy-tuned parameters ----
    axA.plot(x, u0, "-", color=figstyle.TRUTH, lw=2.4, label=r"true $u_0$", zorder=5)
    axA.plot(x, a["rep_part_disc"], "-", color=figstyle.METHOD, lw=1.8,
             label=rf"particle, discrepancy bandwidth, $E_2={float(a['rep_E2_part']):.3f}$")
    axA.plot(x, a["rep_tik_disc"], "--", color=figstyle.TIKH, lw=1.7,
             label=rf"Tikhonov, discrepancy $\lambda$, $E_2={float(a['rep_E2_tik']):.3f}$")
    axA.set_xlabel("$x$"); axA.set_ylabel("$u$")
    axA.set_title("(a)", loc="left")
    axA.grid(True, alpha=0.25)
    axA.legend(loc="upper right", fontsize=8)

    # ---- panel (b): realization-mean E2 vs bandwidth, chosen vs optimal marked ----
    bw = a["bw_factors"]; e2 = a["bw_E2_mean"]
    f_disc = float(a["f_disc_mean"]); f_min = float(a["f_min_mean"])
    axB.plot(bw, e2, "o-", color=figstyle.METHOD, lw=1.8,
             label="mean $E_2$ over realizations (particle)")
    # interpolate curve value at the mean discrepancy bw for the marker
    e2_at_disc = float(np.interp(f_disc, bw, e2))
    e2_at_min = float(np.interp(f_min, bw, e2))
    axB.axvline(f_min, color=figstyle.EXACT, ls="--", lw=1.4,
                label=rf"error-minimizing bandwidth factor $= {f_min:.0f}$")
    axB.axvline(f_disc, color=figstyle.TRUTH, ls=":", lw=1.6,
                label=rf"mean selected bandwidth factor $= {f_disc:.1f}$")
    axB.plot([f_min], [e2_at_min], "s", color=figstyle.EXACT, ms=8, zorder=6)
    axB.plot([f_disc], [e2_at_disc], "o", color=figstyle.TRUTH, ms=8, zorder=6)
    axB.set_yscale("log")
    axB.set_xlabel(r"bandwidth factor $h/\Delta x$")
    axB.set_ylabel(r"mean relative $L^2$ error $E_2$")
    axB.set_title("(b)", loc="left")
    axB.grid(True, which="both", alpha=0.25)
    axB.legend(fontsize=8)

    fig.tight_layout()
    targets = [REPO / "figures"]
    targets[0].mkdir(exist_ok=True)
    if PAPER_FIGS.is_dir():
        targets.append(PAPER_FIGS)
    for base in targets:
        fig.savefig(base / "fig_discrepancy_comparison.pdf")
        fig.savefig(base / "fig_discrepancy_comparison.png", dpi=220)
    plt.close(fig)
    print(f"wrote fig_discrepancy_comparison (representative realization {rep_seed}, "
          f"eta={eta}, disc bw {f_disc:.2f} vs min-error bw {f_min:.0f})")


if __name__ == "__main__":
    main()
