"""Verify, render, or replay the fixed normalized-score illustration.

The manifest names every input. Replays never overwrite the archived results.
Only the analytic remainder encloses the continuum coefficient; differences
between numerical runs are not certified time or spectral error bounds.
"""
from __future__ import annotations

import argparse
from decimal import Decimal as D, getcontext
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SELECTED = "illustration_J4_dt0025_p6_d80.json"


def read_inputs():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Archived input changed: {name}")
    actual = hashlib.sha256((ROOT / "integrate.py").read_bytes()).hexdigest()
    if actual != manifest["integrator_sha256"]:
        raise ValueError("Integrator differs from the recorded implementation")
    return {Path(name).name: json.loads((ROOT / name).read_text())
            for name in manifest["files"]}


def verify():
    getcontext().prec = 90
    rows = read_inputs()
    selected = rows[SELECTED]
    summary = rows["illustration_summary.json"]
    s = D(selected["s"])
    h = D(selected["h"])
    m = 1 / (1 + h * h)
    L = D(selected["L"])
    p, q = (-D(1) / 8).exp(), (-D(1) / 2).exp()
    d1 = m * (2 * q - p)
    quadratic = q * m * p * (1 - m * p) / (2 * d1) * (1 - (-2 * d1 * L).exp()) * s * s
    remainder = 60000 * s**3
    lo, hi = map(D, summary["analytic_PDE_cos2_interval"])
    phi = [(-D(j*j)/8).exp() for j in range(5)]
    first = s*(-d1*L).exp()/2
    second = quadratic/(2*phi[2])
    residual = [D(v) for v in selected["residual_coefficients"]]
    full = [2*phi[j]*(residual[j-1] + (first if j == 1 else second if j == 2 else 0)) for j in range(1,5)]
    difference = [2*phi[j]*residual[j-1] for j in range(1,5)]
    checks = {
        "source_class": D(selected["source_graph_norm"]) < 2 and D(selected["source_minimum"]) > D(".5"),
        "amplitude_envelope": s <= D(1) / 16000000,
        "finite_gaussian_hypotheses": m >= D(".9") and L >= 10 and d1 >= D(".1") and 2*m*q >= 1,
        "all_mode_gap": m * (2*q - 3*(-D(9)/8).exp()) > D(".1"),
        "nonlinear_C0": F(8, 3)*10 + F(8, 9)*4 <= 32,
        "bootstrap": F(1, 4096) <= min(F(1, 8), F(1, 10)/(4*32), F(1, 2*1842)),
        "remainder_constant": (F(16)*2*1800 + F(128, 3)*2)/2 + 12/F(4) + 144/F(64) <= 30000,
        "strict_positive_lower_bound": F(1, 200) - F(30000, 16000000) >= F(1, 400),
        "analytic_enclosure": abs(lo - quadratic + remainder) < D("1e-90") and abs(hi - quadratic - remainder) < D("1e-90"),
        "numerical_value_inside_enclosure": lo < D(selected["returned_cos2_full"]) < hi,
        "denominator_tail": D(selected["denominator_propagated_l1_tail_bound"]) < D("9e-67"),
        "selected_summary_matches_input": summary["selected_source"] == selected,
        "plotted_full_coefficients": all(abs(a-D(b)) < D("1e-89") for a,b in zip(full,summary["returned_cosine_coefficients"])),
        "plotted_difference_coefficients": all(abs(a-D(b)) < D("1e-89") for a,b in zip(difference,summary["full_minus_quadratic_cosine_coefficients"])),
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    return dict(checks=checks, all_passed=True, number_of_refinement_runs=len(rows)-1,
                analytic_PDE_cos2_interval=[str(lo), str(hi)],
                interpretation="Arithmetic and archive checks, not a formal proof or an interval certificate for Galerkin integration.")


def render(output: Path):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    verify()
    rows = read_inputs()
    summary = rows["illustration_summary.json"]
    selected = rows[SELECTED]
    getcontext().prec = 90
    coefs = [D(v) for v in summary["returned_cosine_coefficients"]]
    correction = [D(v) for v in summary["full_minus_quadratic_cosine_coefficients"]]
    phase = np.linspace(0, 2*np.pi, 801)
    target = float(D(selected["source_amplitude"])/D("1e-18"))*np.cos(phase)
    full = sum(float(v/D("1e-18"))*np.cos(j*phase) for j,v in enumerate(coefs, 1))
    quadratic = sum(float((v-c)/D("1e-18"))*np.cos(j*phase) for j,(v,c) in enumerate(zip(coefs,correction),1))
    difference = sum(float(v/D("1e-35"))*np.cos(j*phase) for j,v in enumerate(correction,1))
    plt.rcParams.update({"font.size":9, "axes.spines.top":False, "axes.spines.right":False,
                         "pdf.fonttype":42, "savefig.dpi":180})
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0), layout="constrained")
    axes[0].plot(phase,target,color="#253b57",lw=1.5,label="Target")
    axes[0].plot(phase,full,color="#ae3d32",lw=1.7,label="Nonlinear output")
    axes[0].plot(phase,quadratic,color="#111111",lw=1.1,ls="--",label="Quadratic approximation")
    axes[0].set_ylabel(r"Density minus 1 ($10^{-18}$)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3,
               fontsize=7.5, frameon=False)
    axes[1].plot(phase,difference,color="#16736b",lw=1.5)
    axes[1].axhline(0,color="#888888",lw=1,ls=":")
    axes[1].set_ylabel(r"Nonlinear difference ($10^{-35}$)")
    for label, ax in zip(["(a)","(b)"],axes):
        ax.text(.01,1.02,label,transform=ax.transAxes)
        ax.set_xticks([0,np.pi,2*np.pi],["0",r"$\pi$",r"$2\pi$"])
        ax.set_xlabel(r"Phase $kx$")
        ax.grid(axis="y",alpha=.2,lw=.5)
    output.mkdir(parents=True,exist_ok=True)
    fig.savefig(output/"nonlinear_illustration.pdf",metadata={"CreationDate":None,"ModDate":None})
    fig.savefig(output/"nonlinear_illustration.png")
    plt.close(fig)
    (output/"verification.json").write_text(json.dumps(verify(),indent=2)+"\n")


def replay(output: Path, all_runs: bool):
    from integrate import run

    inputs = read_inputs()
    names = sorted(name for name in inputs if name.startswith("illustration_J")) if all_runs else [SELECTED]
    output.mkdir(parents=True,exist_ok=True)
    comparisons = []
    for name in names:
        old = inputs[name]
        current = run(old["J"],old["P"],old["dt_requested"],old["dps"])
        (output/name).write_text(json.dumps(current,indent=2)+"\n")
        differing = [key for key in old if key != "seconds" and old[key] != current[key]]
        comparisons.append(dict(run=name,identical_except_wall_time=not differing,differing_fields=differing))
        print(json.dumps(comparisons[-1]),flush=True)
    (output/"replay_verification.json").write_text(json.dumps(comparisons,indent=2)+"\n")
    if any(row["differing_fields"] for row in comparisons):
        raise AssertionError("Replay differs from recorded numerical fields")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation",choices=["verify","figures","replay","run"])
    parser.add_argument("--output",type=Path,default=ROOT/"generated")
    parser.add_argument("--all",action="store_true",help="Replay every recorded refinement instead of the selected run")
    parser.add_argument("--J", type=int, default=4, help="Retained positive Fourier modes for a custom run (at least 2)")
    parser.add_argument("--P", type=int, default=6, help="Denominator series order for a custom run (nonnegative)")
    parser.add_argument("--dt", default="0.05", help="Requested step in rescaled time for a custom run")
    parser.add_argument("--dps", type=int, default=80, help="Decimal precision for a custom run (at least 20)")
    args=parser.parse_args()
    if args.operation == "run":
        try:
            step = D(args.dt)
        except ArithmeticError:
            parser.error("--dt must be a positive finite decimal")
        if not step.is_finite() or step <= 0:
            parser.error("--dt must be a positive finite decimal")
        if args.J < 2 or args.P < 0 or args.dps < 20:
            parser.error("Custom runs require J >= 2, P >= 0, and dps >= 20")
    destination = args.output.resolve()
    archive = (ROOT/"results").resolve()
    if destination == ROOT or destination == archive or archive in destination.parents:
        parser.error("Output must be separate from the source and archived results directories")
    if args.operation == "verify": print(json.dumps(verify(),indent=2))
    elif args.operation == "figures": render(args.output)
    elif args.operation == "replay": replay(args.output,args.all)
    else:
        from integrate import run
        read_inputs()
        path = destination / "custom_run.json"
        if path.exists():
            parser.error("custom_run.json already exists; choose a new output directory")
        result = run(args.J, args.P, args.dt, args.dps)
        destination.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({key: result[key] for key in (
            "J", "P", "dt_requested", "dps", "returned_cos2_full",
            "returned_cos2_nonlinear_correction")}, indent=2))
        print(f"Saved {path}")


if __name__ == "__main__": main()
