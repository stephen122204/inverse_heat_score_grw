"""
make_discrepancy_figure.py — fig_discrepancy_comparison (Task §4).

Reads the latest outputs/discrepancy_principle_* and builds a two-panel figure:
 (a) realistic reconstruction overlay at eta=0.005 for the representative seed:
     true u0, discrepancy-tuned particle, discrepancy-tuned Tikhonov (E2 in legend).
 (b) seed-mean true E2 vs particle bandwidth factor at eta=0.005 (the noisy
     U-curve), with the mean discrepancy-chosen bandwidth and the bandwidth that
     minimizes the mean true error marked. The gap between them is the diagnostic.

No editorializing title; the caption (written separately) carries the message.
Writes to figures/ and paper_draft/figures/.
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
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                     "legend.fontsize": 8, "figure.dpi": 160})


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
    axA.plot(x, u0, "k-", lw=2.2, label=r"true $u_0$", zorder=5)
    axA.plot(x, a["rep_part_disc"], "-", color="tab:blue", lw=1.8,
             label=rf"particle, discrepancy bw, $E_2={float(a['rep_E2_part']):.3f}$")
    axA.plot(x, a["rep_tik_disc"], "--", color="tab:green", lw=1.6,
             label=rf"Tikhonov, discrepancy $\lambda$, $E_2={float(a['rep_E2_tik']):.3f}$")
    axA.set_xlabel("$x$"); axA.set_ylabel("$u$")
    axA.set_title("(a)", loc="left")
    axA.grid(True, alpha=0.25)
    axA.legend(frameon=False, loc="upper right")

    # ---- panel (b): seed-mean E2 vs bandwidth, with chosen vs optimal marked ----
    bw = a["bw_factors"]; e2 = a["bw_E2_mean"]
    f_disc = float(a["f_disc_mean"]); f_min = float(a["f_min_mean"])
    axB.plot(bw, e2, "o-", color="tab:blue", lw=1.8,
             label="mean $E_2$ over realizations (particle)")
    # interpolate curve value at the mean discrepancy bw for the marker
    e2_at_disc = float(np.interp(f_disc, bw, e2))
    e2_at_min = float(np.interp(f_min, bw, e2))
    axB.axvline(f_min, color="tab:red", ls="--", lw=1.4,
                label=rf"error-minimizing bw = {f_min:.0f}")
    axB.axvline(f_disc, color="black", ls=":", lw=1.6,
                label=rf"mean discrepancy bw = {f_disc:.1f}")
    axB.plot([f_min], [e2_at_min], "s", color="tab:red", ms=8, zorder=6)
    axB.plot([f_disc], [e2_at_disc], "o", color="black", ms=8, zorder=6)
    axB.set_yscale("log")
    axB.set_xlabel(r"bandwidth factor $h/\Delta x$")
    axB.set_ylabel(r"mean relative $L^2$ error $E_2$")
    axB.set_title("(b)", loc="left")
    axB.grid(True, which="both", alpha=0.25)
    axB.legend(frameon=False)

    fig.tight_layout()
    for base in (REPO / "figures", PAPER_FIGS):
        base.mkdir(exist_ok=True)
        fig.savefig(base / "fig_discrepancy_comparison.pdf")
        fig.savefig(base / "fig_discrepancy_comparison.png", dpi=220)
    plt.close(fig)
    print(f"wrote fig_discrepancy_comparison (rep seed {rep_seed}, eta={eta}, "
          f"disc bw {f_disc:.2f} vs min-error bw {f_min:.0f})")


if __name__ == "__main__":
    main()
