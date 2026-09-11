#!/usr/bin/env python3
"""Analyze the pinned Phase 2C campaign against the frozen claim set.

The only input is the pinned campaign manifest. Every file it lists is
hash-verified before it is read, and nothing here recomputes a study: the
script summarizes, tabulates, and plots what the campaign archived.

Outputs (all regenerable, none hand-edited):

  analysis/phase2c/tables/*.csv and *.tex   manuscript tables (booktabs)
  analysis/phase2c/figures/*.pdf            manuscript figures (figstyle)
  analysis/phase2c/analysis_summary.json    every number the manuscript cites

Usage
-----
    python scripts/analyze_campaign.py
    python scripts/analyze_campaign.py --manifest manifests/phase2c_campaign.json
    python scripts/analyze_campaign.py --paper-figures ../paper_phase2c/figures
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import campaign_schema as schema  # noqa: E402
import figstyle  # noqa: E402
from provenance import sha256_file  # noqa: E402

BANDWIDTHS = list(schema.BANDWIDTHS) if hasattr(schema, "BANDWIDTHS") else [0.005, 0.007, 0.010, 0.014, 0.020, 0.028, 0.040]
EXACT_CASES = ["C1", "C2"]
SECONDARY_CASES = ["B", "H", "Z"]
VARIABLE_CASES = ["VB05", "VB09", "VH05", "VH09"]
CASE_MARKER = {"C1": "o", "C2": "s", "B": "o", "H": "s", "Z": "^",
               "VB05": "D", "VB09": "d", "VH05": "v", "VH09": "<"}
CASE_LS = {"C1": "-", "C2": "--", "B": "-", "H": "--", "Z": ":",
           "VB05": "-.", "VB09": (0, (5, 1, 1, 1)), "VH05": (0, (3, 1)), "VH09": (0, (1, 1))}
ETA_MARKER = {0.0: "o", 0.001: "s", 0.005: "^", 0.01: "D"}
ETA_LS = {0.0: "-", 0.001: "--", 0.005: "-.", 0.01: ":"}
VARIANT_NAME = {"P": "projected input", "R": "raw input"}
# The archived CSV columns keep the protocol's internal names; the analysis
# uses the manuscript vocabulary (noise realization, input variant).
COLUMN_ALIASES = {"seed": "realization", "arm": "variant"}
CLOSURE_NAME = {"frozen_left": "frozen-left", "mass": "mass"}
COMPONENTS = ["wrong_transport", "closure", "score_regularization", "particle_discretization"]
COMPONENT_NAME = {"wrong_transport": "wrong transport", "closure": "closure offset",
                  "score_regularization": "score regularization",
                  "particle_discretization": "particle discretization"}
# Table headers use the symbols of the decomposition identity in the manuscript,
# e_1 wrong transport, e_2 closure offset, e_3 score regularization, e_4 particle discretization.
COMPONENT_SYMBOL = {"wrong_transport": "$\\|e_1\\|$", "closure": "$\\|e_2\\|$",
                    "score_regularization": "$\\|e_3\\|$", "particle_discretization": "$\\|e_4\\|$"}
COMPONENT_INDEX = {"wrong_transport": 1, "closure": 2, "score_regularization": 3, "particle_discretization": 4}
# A decomposition row whose reference pair failed its refinement requirement is
# printed with an asterisk on the case label and excluded from the admissible-row summaries.
INADMISSIBLE_MARK = "$^{*}$"


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for study, entry in manifest["studies"].items():
        directory = REPO / entry["path"]
        for name, digest in entry["sha256"].items():
            actual = sha256_file(directory / name)
            if actual != digest:
                raise SystemExit(f"hash mismatch for {study}/{name}: manifest {digest} != file {actual}")
    return manifest


def study_dir(manifest: dict, study: str) -> Path:
    return REPO / manifest["studies"][study]["path"]


def read_rows(manifest: dict, study: str) -> pd.DataFrame:
    frame = pd.read_csv(study_dir(manifest, study) / f"{study}_rows.csv")
    return frame


def read_json(manifest: dict, study: str, name: str) -> dict:
    return json.loads((study_dir(manifest, study) / name).read_text(encoding="utf-8"))


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def paired_summary(diff: np.ndarray) -> dict:
    """Mean, median, two-sided 95% paired confidence interval, and the
    fraction of realizations with a negative difference."""
    from scipy import stats

    diff = np.asarray(diff, dtype=float)
    n = diff.size
    mean = float(diff.mean())
    sd = float(diff.std(ddof=1)) if n > 1 else float("nan")
    half = float(stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)) if n > 1 else float("nan")
    return {
        "n": int(n),
        "mean": mean,
        "median": float(np.median(diff)),
        "ci_low": mean - half,
        "ci_high": mean + half,
        "fraction_negative": float(np.mean(diff < 0)),
    }


def window_scale(alpha: float, T: float, delta: float) -> float:
    return math.sqrt(2.0 * alpha * T / (math.e * math.log(1.0 / delta)))


def fmt(x, digits=3, sci_below=1e-3):
    if x is None or (isinstance(x, float) and (math.isnan(x))):
        return "--"
    if isinstance(x, (bool, np.bool_)):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    ax = abs(float(x))
    if ax != 0.0 and ax < sci_below:
        mant, exp = f"{x:.{digits - 1}e}".split("e")
        return f"${mant}\\times10^{{{int(exp)}}}$"
    return f"{x:.{digits}f}" if ax < 1e4 else f"{x:.{digits}e}"


def write_table(out_dir: Path, name: str, frame: pd.DataFrame, formats: dict | None = None,
                caption_note: str = "") -> None:
    """Write a CSV and a booktabs LaTeX body for one table."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / f"{name}.csv", index=False)
    formats = formats or {}
    cols = list(frame.columns)
    lines = ["\\begin{tabular}{" + "l" + "r" * (len(cols) - 1) + "}", "\\toprule",
             " & ".join(str(c) for c in cols) + " \\\\", "\\midrule"]
    for _, row in frame.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            f = formats.get(c)
            if isinstance(v, str):
                cells.append(v)
            elif f is None:
                cells.append(fmt(v))
            else:
                cells.append(f(v))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if caption_note:
        lines.insert(0, f"% {caption_note}")
    (out_dir / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------
# Study analyses
# ----------------------------------------------------------------------
def analyze_bandwidth_clean(manifest: dict, tables: Path) -> dict:
    rows = read_rows(manifest, "bandwidth_clean")
    hs = sorted(rows.h.unique())
    pivot = rows.pivot(index="case", columns="h", values="E2")
    order = [c for c in EXACT_CASES + SECONDARY_CASES + VARIABLE_CASES if c in pivot.index]
    pivot = pivot.loc[order]
    summary = {"bandwidths": hs, "cases": {}}
    table = []
    for case in order:
        curve = pivot.loc[case]
        idx = int(np.argmin(curve.values))
        h_or = float(hs[idx])
        endpoint = idx in (0, len(hs) - 1)
        row_min = rows[(rows.case == case) & (rows.h == h_or)].iloc[0]
        summary["cases"][case] = {
            "E2_by_h": {str(h): float(curve[h]) for h in hs},
            "h_or": h_or, "h_or_endpoint": bool(endpoint),
            "E2_min": float(curve.min()),
            "forward_residual_at_min": float(row_min.forward_residual),
            "mass_error_rel_at_min": float(row_min.mass_error_rel),
            "min_reconstruction_at_min": float(row_min.min_reconstruction),
            "u_shaped": bool(idx not in (0, len(hs) - 1)
                             and all(np.diff(curve.values[: idx + 1]) <= 0)
                             and all(np.diff(curve.values[idx:]) >= 0)),
            "all_completed": bool((rows[rows.case == case].status == "completed").all()),
        }
        table.append({"case": case, **{f"{h:g}": float(curve[h]) for h in hs},
                      "$h_{\\mathrm{or}}$": f"{h_or:g}" + ("$^{\\dagger}$" if endpoint else "")})
    frame = pd.DataFrame(table)
    write_table(tables, "bandwidth_clean", frame,
                formats={f"{h:g}": (lambda v: fmt(v, 4)) for h in hs},
                caption_note="Clean relative L2 error E2 over the bandwidth set; dagger marks an endpoint minimizer")
    # variable-vs-constant degradation at matched initial condition
    deg = {}
    for v, base in (("VB05", "B"), ("VB09", "B"), ("VH05", "H"), ("VH09", "H")):
        deg[v] = summary["cases"][v]["E2_min"] / summary["cases"][base]["E2_min"]
    summary["variable_over_constant_E2min_ratio"] = deg
    # the exact-mode kernel factors quoted in the manuscript
    summary["C1_signal_mode_kernel_factor"] = {str(h): math.exp(-0.5 * (3 * math.pi * h) ** 2) for h in hs}
    summary["window_scale_C1"] = {"1e-8": window_scale(0.01, 1.0, 1e-8), "1e-2": window_scale(0.01, 1.0, 1e-2)}
    return summary


def analyze_lambda_oracle_clean(manifest: dict, tables: Path, bw: dict) -> dict:
    rows = read_rows(manifest, "lambda_oracle_clean")
    out = {}
    table = []
    for _, r in rows.iterrows():
        out[r.case] = {"lambda_or": float(r.lambda_selected), "E2": float(r.E2_at_selection),
                       "endpoint": bool(r.censored), "residual": float(r.residual_at_selection)}
        table.append({"case": r.case,
                      "particle $E_2$ at $h_{\\mathrm{or}}$": bw["cases"][r.case]["E2_min"],
                      "$h_{\\mathrm{or}}$": f"{bw['cases'][r.case]['h_or']:g}",
                      "Tikhonov $E_2$ at $\\lambda_{\\mathrm{or}}$": float(r.E2_at_selection),
                      "$\\log_{10}\\lambda_{\\mathrm{or}}$": f"{math.log10(r.lambda_selected):.2f}" + ("$^{\\dagger}$" if r.censored else "")})
    order = [c for c in EXACT_CASES + SECONDARY_CASES + VARIABLE_CASES if c in out]
    frame = pd.DataFrame(table).set_index("case").loc[order].reset_index()
    write_table(tables, "clean_comparison", frame,
                formats={"particle $E_2$ at $h_{\\mathrm{or}}$": lambda v: fmt(v, 4),
                         "Tikhonov $E_2$ at $\\lambda_{\\mathrm{or}}$": lambda v: fmt(v, 3, sci_below=1e-2)},
                caption_note="Clean data: particle method at its oracle bandwidth versus Tikhonov at its oracle regularization parameter; dagger marks an endpoint of the search interval")
    return out


def analyze_epsilon(manifest: dict, tables: Path) -> dict:
    rows = read_rows(manifest, "epsilon_sensitivity")
    out = {}
    table = []
    for case in ["C1", "Z"]:
        sub = rows[rows.case == case].sort_values("eps_rel")
        ref = float(sub[sub.eps_rel == 1e-8].E2.iloc[0])
        out[case] = {"h": float(sub.h.iloc[0]),
                     "E2_by_eps": {f"{e:g}": float(v) for e, v in zip(sub.eps_rel, sub.E2)},
                     "max_rel_change_vs_production": float(max(abs(v / ref - 1) for v in sub.E2)),
                     "min_raw_density": float(sub.min_raw_density.min()),
                     "all_completed": bool((sub.status == "completed").all())}
        table.append({"case": case, "$h$": f"{sub.h.iloc[0]:g}",
                      **{f"$\\epsilon_{{\\mathrm{{rel}}}}=10^{{{round(math.log10(e))}}}$": float(v) for e, v in zip(sub.eps_rel, sub.E2)},
                      "min reconstructed density": float(sub.min_reconstruction.min())})
    frame = pd.DataFrame(table)
    write_table(tables, "epsilon_sensitivity", frame,
                formats={c: (lambda v: fmt(v, 5)) for c in frame.columns if c.startswith("$\\epsilon")},
                caption_note="Positivity-regularization sensitivity at the per-case anchor bandwidth")
    return out


def analyze_adequacy(manifest: dict, tables: Path) -> dict:
    rows = read_rows(manifest, "adequacy_N")
    out = {}
    table = []
    for case in ["C1", "H"]:
        sub = rows[rows.case == case].sort_values("N")
        e4000 = float(sub[sub.N == 4000].E2.iloc[0]); e10000 = float(sub[sub.N == 10000].E2.iloc[0])
        out[case] = {"h": float(sub.h.iloc[0]), "E2_by_N": {str(int(n)): float(v) for n, v in zip(sub.N, sub.E2)},
                     "rel_change_4000_to_10000": abs(e10000 / e4000 - 1), "gate": 0.005,
                     "pass": bool(abs(e10000 / e4000 - 1) <= 0.005)}
        table.append({"case": case, "$h$": f"{sub.h.iloc[0]:g}",
                      **{f"$N={int(n)}$": float(v) for n, v in zip(sub.N, sub.E2)},
                      "relative change 4000 to 10000": abs(e10000 / e4000 - 1)})
    frame = pd.DataFrame(table)
    write_table(tables, "adequacy_N", frame,
                formats={c: (lambda v: fmt(v, 5)) for c in frame.columns if c.startswith("$N")},
                caption_note="Particle-count adequacy at the per-case anchor bandwidth")
    return out


def analyze_transition(manifest: dict, tables: Path) -> dict:
    rows = read_rows(manifest, "transition_table")
    era_name = {"endpoint_free_space_legacy": "endpoint grid, free-space kernel",
                "cell_centered_free_space_legacy": "cell-centered grid, free-space kernel",
                "cell_centered_neumann_canonical": "cell-centered grid, Neumann kernel"}
    table, out = [], {}
    for _, r in rows.iterrows():
        out[r.era] = {"E2": float(r.E2), "h_nominal": float(r.h_nominal), "h_effective": float(r.h_effective),
                      "smoothing_operations": int(r.smoothing_operations), "n_grid": int(r.n_grid),
                      "n_particles": int(r.n_particles)}
        table.append({"construction": era_name[r.era], "nominal $h$": f"{r.h_nominal:.4f}",
                      "effective $h$": f"{r.h_effective:.4f}", "smoothings": int(r.smoothing_operations),
                      "$E_2$": float(r.E2)})
    write_table(tables, "transition_table", pd.DataFrame(table), formats={"$E_2$": lambda v: fmt(v, 4)},
                caption_note="Historical transition on C1")
    return out


def analyze_noise(manifest: dict, tables: Path, figures: Path) -> dict:
    npr = read_rows(manifest, "noise_paired")
    sel = read_json(manifest, "noise_paired", "bandwidth_selection.json")["selections"]
    lam = read_rows(manifest, "lambda_noise")
    hs = sorted(npr.h.unique())
    out = {"bandwidths": hs, "blocks": {}, "curves": {}, "selection_gap": {}, "residual_target": {}}
    sel_df = pd.DataFrame(sel).rename(columns=COLUMN_ALIASES)
    npr = npr.rename(columns=COLUMN_ALIASES)
    lam = lam.rename(columns=COLUMN_ALIASES)
    noisy = npr[npr.eta > 0].copy()
    noisy["realization"] = noisy["realization"].astype(int)
    lam["realization"] = pd.to_numeric(lam["realization"], errors="coerce")
    summary_rows = []
    for case in ["C1", "B"]:
        for eta in sorted(noisy.eta.unique()):
            for variant in ["P", "R"]:
                blk = noisy[(noisy.case == case) & (noisy.eta == eta) & (noisy.variant == variant)]
                realizations = sorted(blk.realization.unique())
                # particle: truth-selected and residual-matched errors per realization
                e_or, h_or, e_rm12, h_rm12, cens12, e_rm10, h_rm10, cens10 = [], [], [], [], [], [], [], []
                for s in realizations:
                    curve = blk[blk.realization == s].set_index("h").E2.reindex(hs)
                    idx = int(np.nanargmin(curve.values)); e_or.append(float(curve.values[idx])); h_or.append(float(hs[idx]))
                    for tau, e_l, h_l, c_l in ((1.2, e_rm12, h_rm12, cens12), (1.0, e_rm10, h_rm10, cens10)):
                        srow = sel_df[(sel_df.case == case) & (sel_df.eta == eta) & (sel_df.variant == variant)
                                      & (sel_df.realization == s) & (sel_df.tau == tau)].iloc[0]
                        h_l.append(float(srow.selected_h)); c_l.append(bool(srow.endpoint_censored))
                        e_l.append(float(curve[srow.selected_h]))
                # Tikhonov per realization: oracle, Morozov 1.2, residual 1.0; raw and projected outputs
                tik = lam[(lam.case == case) & (lam.eta == eta) & (lam.variant == variant)]
                def tik_series(selection, tau=None, col="E2_tikhonov_raw"):
                    sub = tik[tik.selection == selection] if tau is None else tik[(tik.selection == selection) & (tik.tau.astype(str) == str(tau))]
                    sub = sub.set_index("realization").reindex(realizations)
                    return sub[col].to_numpy(dtype=float), sub["censored"].to_numpy()
                t_or, _ = tik_series("oracle_continuous")
                t_or_p, _ = tik_series("oracle_continuous", col="E2_tikhonov_projected")
                t_m12, c_m12 = tik_series("residual", 1.2)
                t_m12_p, _ = tik_series("residual", 1.2, col="E2_tikhonov_projected")
                t_m10, c_m10 = tik_series("residual", 1.0)
                t_m10_p, _ = tik_series("residual", 1.0, col="E2_tikhonov_projected")
                e_or, e_rm12, e_rm10 = map(np.array, (e_or, e_rm12, e_rm10))
                block = {
                    "n_realizations": len(realizations),
                    "particle_oracle": {"mean": float(e_or.mean()), "std": float(e_or.std(ddof=1)),
                                        "h_or_counts": {str(h): int(sum(1 for x in h_or if x == h)) for h in hs},
                                        "endpoint_count": int(sum(1 for x in h_or if x in (hs[0], hs[-1])))},
                    "particle_residual_matched_1.2": {"mean": float(e_rm12.mean()), "std": float(e_rm12.std(ddof=1)),
                                                      "h_counts": {str(h): int(sum(1 for x in h_rm12 if x == h)) for h in hs},
                                                      "endpoint_count": int(sum(cens12))},
                    "particle_residual_matched_1.0": {"mean": float(e_rm10.mean()), "std": float(e_rm10.std(ddof=1)),
                                                      "h_counts": {str(h): int(sum(1 for x in h_rm10 if x == h)) for h in hs},
                                                      "endpoint_count": int(sum(cens10))},
                    "tikhonov_oracle": {"mean": float(np.mean(t_or)), "std": float(np.std(t_or, ddof=1)),
                                        "projected_mean": float(np.mean(t_or_p))},
                    "tikhonov_morozov_1.2": {"mean": float(np.mean(t_m12)), "std": float(np.std(t_m12, ddof=1)),
                                             "projected_mean": float(np.mean(t_m12_p)), "endpoint_count": int(np.sum(c_m12)),
                                             "median": float(np.median(t_m12))},
                    "tikhonov_residual_1.0": {"mean": float(np.mean(t_m10)), "std": float(np.std(t_m10, ddof=1)),
                                              "projected_mean": float(np.mean(t_m10_p)), "endpoint_count": int(np.sum(c_m10)),
                                              "median": float(np.median(t_m10))},
                    "paired_oracle_vs_oracle": paired_summary(e_or - t_or),
                    "paired_deployable_1.2": paired_summary(e_rm12 - t_m12),
                    "paired_deployable_1.2_projected_tikhonov": paired_summary(e_rm12 - t_m12_p),
                    "ratio_oracle_means": float(e_or.mean() / np.mean(t_or)),
                    "ratio_deployable_means": float(e_rm12.mean() / np.mean(t_m12)),
                    "selection_gap_steps": {str(k): int(v) for k, v in
                                            pd.Series([hs.index(a) - hs.index(b) for a, b in zip(h_rm12, h_or)]).value_counts().sort_index().items()},
                    "selection_gap_error_ratio": float(e_rm12.mean() / e_or.mean()),
                }
                # residual target attainability
                tgt = [float(sel_df[(sel_df.case == case) & (sel_df.eta == eta) & (sel_df.variant == variant) & (sel_df.realization == s) & (sel_df.tau == 1.2)].iloc[0].target) for s in realizations]
                rmin = [float(min(sel_df[(sel_df.case == case) & (sel_df.eta == eta) & (sel_df.variant == variant) & (sel_df.realization == s) & (sel_df.tau == 1.2)].iloc[0].curve_r)) for s in realizations]
                block["residual_target"] = {"target_mean": float(np.mean(tgt)), "min_residual_mean": float(np.mean(rmin)),
                                            "fraction_attainable": float(np.mean(np.array(rmin) <= np.array(tgt)))}
                # mean curves for the figure
                cm = blk.groupby("h").E2.agg(["mean", "std"]).reindex(hs)
                out["curves"][f"{case}|{eta:g}|{variant}"] = {"mean": cm["mean"].tolist(), "std": cm["std"].tolist()}
                out["blocks"][f"{case}|{eta:g}|{variant}"] = block
                summary_rows.append({
                    "case": case, "$\\eta$": f"{eta:g}", "input": VARIANT_NAME[variant],
                    "$E_2(h_{\\mathrm{or}})$": block["particle_oracle"]["mean"],
                    "$E_2(h_{\\mathrm{rm}})$": block["particle_residual_matched_1.2"]["mean"],
                    "$E_2(\\lambda_{\\mathrm{or}})$": block["tikhonov_oracle"]["mean"],
                    "$E_2(\\lambda_{\\mathrm{M}})$": block["tikhonov_morozov_1.2"]["mean"],
                    "$E_2(\\lambda_{\\mathrm{M}})$, projected": block["tikhonov_morozov_1.2"]["projected_mean"],
                    "$\\Delta$": block["paired_deployable_1.2"]["mean"],
                    "95\\% CI": f"[{block['paired_deployable_1.2']['ci_low']:.4f}, {block['paired_deployable_1.2']['ci_high']:.4f}]",
                    "favoring particle": block["paired_deployable_1.2"]["fraction_negative"],
                })
    frame = pd.DataFrame(summary_rows)
    error_cols = ["case", "$\\eta$", "input", "$E_2(h_{\\mathrm{or}})$", "$E_2(h_{\\mathrm{rm}})$",
                  "$E_2(\\lambda_{\\mathrm{or}})$", "$E_2(\\lambda_{\\mathrm{M}})$", "$E_2(\\lambda_{\\mathrm{M}})$, projected"]
    write_table(tables, "noise_paired", frame[error_cols],
                formats={c: (lambda v: fmt(v, 4)) for c in error_cols if c not in ("case", "$\\eta$", "input")},
                caption_note="Paired noise comparison, means over 25 realizations, both input variants; errors only")
    stat_cols = ["case", "$\\eta$", "input", "$\\Delta$", "95\\% CI", "favoring particle"]
    write_table(tables, "noise_paired_stats", frame[stat_cols],
                formats={"$\\Delta$": lambda v: fmt(v, 4), "favoring particle": lambda v: f"{v:.2f}"},
                caption_note="Paired statistics: particle at the residual-matched bandwidth minus Tikhonov at the Morozov choice (tau = 1.2)")
    body = []
    for key, block in out["blocks"].items():
        case, eta, variant = key.split("|")
        if variant != "P":
            continue
        po = block["particle_oracle"]; pr = block["particle_residual_matched_1.2"]
        cens = f" ({po['endpoint_count']}/{block['n_realizations']})" if po["endpoint_count"] else ""
        body.append({"case": case, "$\\eta$": eta,
                     "$E_2(h_{\\mathrm{or}})$": f"{po['mean']:.4f}" + ("$^{\\dagger}$" if po["endpoint_count"] else ""),
                     "$E_2(h_{\\mathrm{rm}})$": f"{pr['mean']:.4f}",
                     "$E_2(\\lambda_{\\mathrm{or}})$": f"{block['tikhonov_oracle']['mean']:.4f}",
                     "$E_2(\\lambda_{\\mathrm{M}})$": f"{block['tikhonov_morozov_1.2']['mean']:.4f}",
                     "$\\Delta$ [95\\% CI]": f"{block['paired_deployable_1.2']['mean']:.4f} [{block['paired_deployable_1.2']['ci_low']:.4f}, {block['paired_deployable_1.2']['ci_high']:.4f}]",
                     "endpoint": f"{po['endpoint_count']}/{block['n_realizations']}"})
    write_table(tables, "noise_paired_body", pd.DataFrame(body),
                caption_note="Body table: common projected input, means over 25 realizations; dagger marks an oracle bandwidth at the top of the candidate set, with the count of such realizations in the last column")
    # clean curves for the figure
    clean = npr[npr.eta == 0]
    for case in ["C1", "B"]:
        cc = clean[clean.case == case].set_index("h").E2.reindex(hs)
        out["curves"][f"{case}|0|shared"] = {"mean": cc.tolist(), "std": [0.0] * len(hs)}
    # sensitivity table at tau = 1.0
    sens = []
    for key, block in out["blocks"].items():
        case, eta, variant = key.split("|")
        sens.append({"case": case, "$\\eta$": eta, "input": VARIANT_NAME[variant],
                     "$E_2(h_{\\mathrm{rm}})$, $\\tau=1$": block["particle_residual_matched_1.0"]["mean"],
                     "endpoints": block["particle_residual_matched_1.0"]["endpoint_count"],
                     "median $E_2(\\lambda)$, $\\tau=1$": block["tikhonov_residual_1.0"]["median"],
                     "endpoints ": block["tikhonov_residual_1.0"]["endpoint_count"]})
    write_table(tables, "noise_sensitivity_tau1", pd.DataFrame(sens),
                formats={"$E_2(h_{\\mathrm{rm}})$, $\\tau=1$": lambda v: fmt(v, 4),
                         "median $E_2(\\lambda)$, $\\tau=1$": lambda v: fmt(v, 4)},
                caption_note="Residual-rule sensitivity at tau = 1.0 (labeled sensitivity, not the headline)")
    figure_noise_window(out, hs, figures)
    return out


def figure_bandwidth_clean(bw: dict, figures: Path) -> None:
    from scripts.plot_revision_figures import figure_bandwidth_clean as render
    render(bw, figures)


def figure_noise_window(noise: dict, hs: list, figures: Path) -> None:
    from scripts.plot_revision_figures import figure_noise_window as render
    render(noise, hs, figures)


def analyze_closure(manifest: dict, tables: Path, figures: Path) -> dict:
    rows = read_rows(manifest, "closure")
    dec = read_json(manifest, "closure", "closure_decomposition.json")
    gates = read_json(manifest, "closure", "closure_gates.json")
    out = {"decomposition": {}, "analytic_anchor_G1": dec["analytic_anchor_G1"], "reference_pairs": {},
           "carrier_refinement": {}, "h_bridge": {}, "split_invariance": {}, "verdicts": gates["verdicts"]}
    failed_pairs = {(rp["case"], rp["closure"]) for rp in gates["reference_pairs"] if not rp["pass"]}
    out["inadmissible_rows"] = sorted(f"{c}|{cl}" for c, cl in failed_pairs)

    def case_label(c: dict) -> str:
        return c["case"] + (INADMISSIBLE_MARK if (c["case"], c["closure"]) in failed_pairs else "")

    table = []
    for c in dec["cases"]:
        key = f"{c['case']}|{c['closure']}"
        entry = {}
        for tname, t in c["times"].items():
            entry[tname] = {"reverse_time": t["reverse_time"]}
            for level in ("u", "q"):
                comp = t[level]["component_norms"]
                entry[tname][level] = {"total": t[level]["total_norm"], **{k: comp[k] for k in COMPONENTS},
                                       "inner_products": t[level]["inner_products"],
                                       "reconciliation_residual": t[level]["reconciliation_residual"],
                                       "dominant": max(COMPONENTS, key=lambda k: comp[k]),
                                       "wrong_transport_share": comp["wrong_transport"] / t[level]["total_norm"]}
        out["decomposition"][key] = {"status": c["status"], "reconciled": c["reconciled"],
                                     "admissible": (c["case"], c["closure"]) not in failed_pairs, **entry}
        t = c["times"]["final"]["u"]
        table.append({"case": case_label(c), "closure": CLOSURE_NAME[c["closure"]], "total": t["total_norm"],
                      **{COMPONENT_SYMBOL[k]: t["component_norms"][k] for k in COMPONENTS}})
    frame = pd.DataFrame(table)
    out["max_reconciliation_residual_u"] = max(c["times"]["final"]["u"]["reconciliation_residual"] for c in dec["cases"])
    write_table(tables, "closure_decomposition_u", frame,
                formats={c: (lambda v: fmt(v, 4, sci_below=1e-3)) for c in frame.columns if c not in ("case", "closure")},
                caption_note="Field-level decomposition at reverse time T, absolute L2 norms on the M = 400 grid; e_1 wrong transport, e_2 closure offset, e_3 score regularization, e_4 particle discretization; asterisk marks a row whose reference pair failed its refinement requirement")
    tq = []
    for c in dec["cases"]:
        t = c["times"]["final"]["q"]
        tq.append({"case": case_label(c), "closure": CLOSURE_NAME[c["closure"]], "total": t["total_norm"],
                   **{COMPONENT_SYMBOL[k]: t["component_norms"][k] for k in COMPONENTS}})
    fq = pd.DataFrame(tq)
    write_table(tables, "closure_decomposition_q", fq,
                formats={c: (lambda v: fmt(v, 4, sci_below=1e-3)) for c in fq.columns if c not in ("case", "closure")},
                caption_note="Gradient-level decomposition at reverse time T; same symbols as the field-level table")
    # inner products table (u, final)
    ip = []
    for c in dec["cases"]:
        ips = c["times"]["final"]["u"]["inner_products"]
        row = {"case": case_label(c), "closure": CLOSURE_NAME[c["closure"]]}
        for k, v in ips.items():
            i, j = (COMPONENT_INDEX[part] for part in k.split("|"))
            row[f"$\\langle e_{i}, e_{j}\\rangle$"] = v
        ip.append(row)
    fip = pd.DataFrame(ip)
    write_table(tables, "closure_inner_products_u", fip,
                formats={c: (lambda v: fmt(v, 2, sci_below=1.0)) for c in fip.columns if c not in ("case", "closure")},
                caption_note="Pairwise inner products of the four field-level components at reverse time T; same symbols as the decomposition table")
    for rp in gates["reference_pairs"]:
        out["reference_pairs"][f"{rp['case']}|{rp['closure']}|{rp['kind']}"] = {
            "pass": rp["pass"], "gate": rp["gate"], "comparisons": rp["comparisons"],
            "statuses": [r["status"] for r in rp["rows"]], "max_cfl": max((r["max_cfl"] for r in rp["rows"] if r["status"] == "completed"), default=None),
            "failure_reasons": [r["failure_reason"] for r in rp["rows"] if r["status"] != "completed"]}
    for cr in gates["carrier_refinement"]:
        out["carrier_refinement"][f"{cr['case']}|{cr['closure']}"] = {
            "diffs": cr["diffs"], "last_reduction_u": cr["last_reduction_u"], "last_reduction_q": cr["last_reduction_q"],
            "required": cr["required_last_reduction"], "pass": cr["pass"],
            "u_order_estimate": math.log2(cr["diffs"]["400"]["u"] / cr["diffs"]["800"]["u"])}
    for hb in gates["h_bridge"]:
        fine = hb["curves"][max(hb["curves"], key=int)]
        hsb = sorted((float(h) for h in fine), reverse=True)
        u = [fine[f"{h:g}"]["u"] for h in hsb]; q = [fine[f"{h:g}"]["q"] for h in hsb]
        slope_u = float(np.polyfit(np.log(hsb), np.log(u), 1)[0])
        out["h_bridge"][f"{hb['case']}|{hb['closure']}"] = {"h": hsb, "u": u, "q": q, "pass": hb["pass"], "u_slope_in_h": slope_u}
    for si in gates["split_invariance"]:
        out["split_invariance"][f"{si['case']}|{si['closure']}"] = {"rel_u": si["rel_u"], "max_abs_q": si["max_abs_q"], "pass": si["pass"]}
    # refinement table
    rt = []
    for key, cr in out["carrier_refinement"].items():
        case, closure = key.split("|")
        rt.append({"case": case + (INADMISSIBLE_MARK if (case, closure) in failed_pairs else ""), "closure": CLOSURE_NAME[closure],
                   "$u$: $M=200$": cr["diffs"]["200"]["u"], "$M=400$": cr["diffs"]["400"]["u"], "$M=800$": cr["diffs"]["800"]["u"],
                   "$q$: $M=200$": cr["diffs"]["200"]["q"], "$M=400$ ": cr["diffs"]["400"]["q"], "$M=800$ ": cr["diffs"]["800"]["q"]})
    frt = pd.DataFrame(rt)
    write_table(tables, "closure_refinement", frt,
                formats={c: (lambda v: fmt(v, 3, sci_below=2e-2)) for c in frt.columns if c not in ("case", "closure")},
                caption_note="Particle-discretization distance to the regularized reference under coupled (M, dt) refinement")
    hbt = []
    for key, hb in out["h_bridge"].items():
        case, closure = key.split("|")
        hbt.append({"case": case + (INADMISSIBLE_MARK if (case, closure) in failed_pairs else ""), "closure": CLOSURE_NAME[closure],
                    **{f"$h={h:g}$": u for h, u in zip(hb["h"], hb["u"])}, "slope in $h$": hb["u_slope_in_h"]})
    fhb = pd.DataFrame(hbt)
    write_table(tables, "closure_h_bridge", fhb,
                formats={c: (lambda v: fmt(v, 3, sci_below=2e-2)) for c in fhb.columns if c.startswith("$h")} | {"slope in $h$": lambda v: f"{v:.2f}"},
                caption_note="Distance between the regularized and wrong-limit references (u level, M = 3200) as h decreases")
    figure_closure(out, figures)
    return out


def figure_closure(cl: dict, figures: Path) -> None:
    from scripts.plot_revision_figures import figure_closure as render
    render(cl, figures)


def analyze_initial_rate(manifest: dict, tables: Path, figures: Path) -> dict:
    rows = read_rows(manifest, "initial_rate")
    gates = read_json(manifest, "initial_rate", "initial_rate_gates.json")
    ref = rows[rows.block == "reference"]
    car = rows[rows.block == "carrier"]
    q = rows[rows.block == "q_level"].sort_values("tau")
    ratio_minus_one = (q.ratio_q.to_numpy() - 1.0)
    taus = q.tau.to_numpy(dtype=float)
    coef = float(np.sum(ratio_minus_one * taus) / np.sum(taus * taus))  # least squares through the origin
    out = {
        "c_rep": float(ref.c_rep.iloc[0]), "tau": float(ref.tau.iloc[0]),
        "gate_band": float(ref.gate_band.iloc[0]),
        "references": [{"M": int(r.M), "dt": float(r["dt"]), "e_U": float(r.e_U), "ratio": float(r.ratio), "within_band": bool(r.within_band)} for _, r in ref.iterrows()],
        "pair_defect_relative": gates["pair_defect_relative"], "verdicts": gates["verdicts"],
        "carrier": [{"M": int(r.M), "dt": float(r["dt"]), "u_diff_ref": float(r.u_diff_ref)} for _, r in car.iterrows()],
        "carrier_relative_spread": float(car.u_diff_ref.max() / car.u_diff_ref.min() - 1.0),
        "q_level": {"slope_q": float(q.slope_q.iloc[0]), "taus": taus.tolist(), "e_q": q.e_q.tolist(),
                    "ratio_q": q.ratio_q.tolist(), "linear_coefficient_of_ratio_minus_one": coef,
                    "max_abs_residual_of_linear_fit": float(np.max(np.abs(ratio_minus_one - coef * taus)))},
    }
    t1 = pd.DataFrame([{"reference grid $M$": int(r.M), "$\\Delta t$": f"${r['dt'] / 10 ** math.floor(math.log10(r['dt'])):.4g}\\times10^{{{math.floor(math.log10(r['dt']))}}}$",
                        "$\\|U_{\\mathrm{wrong}} - u\\|_2$ at $\\tau = 0.005$": float(r.e_U),
                        "ratio to $c_{\\mathrm{rep}}\\tau$": float(r.ratio), "within band": bool(r.within_band)} for _, r in ref.iterrows()])
    write_table(tables, "initial_rate_reference", t1,
                formats={"$\\|U_{\\mathrm{wrong}} - u\\|_2$ at $\\tau = 0.005$": lambda v: fmt(v, 4, sci_below=1e-2),
                         "ratio to $c_{\\mathrm{rep}}\\tau$": lambda v: f"{v:.6f}"},
                caption_note="Theorem-backed continuum check of the first-order representation defect")
    t2 = pd.DataFrame([{"$\\tau$": f"{r.tau:g}", "$\\|q_{\\mathrm{wrong}} - u_x\\|_2$": float(r.e_q), "ratio to first-order slope": float(r.ratio_q)} for _, r in q.iterrows()])
    write_table(tables, "initial_rate_q_level", t2, formats={"$\\|q_{\\mathrm{wrong}} - u_x\\|_2$": lambda v: fmt(v, 4, sci_below=1e-2), "ratio to first-order slope": lambda v: f"{v:.4f}"},
                caption_note="Gradient-level defect growth on the mass-closed wrong flow, M = 3200")
    figure_initial_rate(out, figures)
    return out


def figure_initial_rate(ir: dict, figures: Path) -> None:
    from scripts.plot_revision_figures import figure_initial_rate as render
    render(ir, figures)


def analyze_crossover(manifest: dict, tables: Path, figures: Path) -> dict:
    rows = read_rows(manifest, "crossover")
    cont = rows[rows.block == "continuum"].copy()
    part = rows[rows.block == "particle"].copy()
    # The continuum rows record the latent density rho; the particle rows record the
    # returned field, which is the kernel density of the final positions, i.e. K_h rho.
    # The like-for-like continuum values apply the same final smoothing to the
    # latent coefficients: mode k carries phi_k = exp(-k^2 h^2 / 2).
    k_mode = schema.CROSSOVER_MODE * math.pi
    for frame_ in (cont, part):
        h_ = frame_.kh / k_mode
        frame_["phi_k"] = np.exp(-0.5 * (k_mode * h_) ** 2)
        frame_["phi_2k"] = np.exp(-0.5 * (2.0 * k_mode * h_) ** 2)
        frame_["e_k_out"] = frame_.phi_k * (frame_.a + frame_.e_k) - frame_.a
        frame_["e_2k_out"] = frame_.phi_2k * frame_.e_2k
    cont["ratio"] = cont.e_2k / cont.e_k.abs()
    cont["ratio_out"] = cont.e_2k_out / cont.e_k_out.abs()
    cont["pred_ratio"] = cont.pred_e_2k / cont.pred_e_k.abs()
    cont["model_rel_err_2k"] = (cont.pred_e_2k / cont.e_2k - 1).abs()
    cont["model_rel_err_k"] = (cont.pred_e_k / cont.e_k - 1).abs()
    out = {"continuum": [], "particle": [], "phase_boundary": {}, "low_order_coefficients": {}}
    for _, r in cont.sort_values(["kh", "a"]).iterrows():
        out["continuum"].append({"a": float(r.a), "kh": float(r.kh), "e_k": float(r.e_k), "e_2k": float(r.e_2k),
                                 "pred_e_k": float(r.pred_e_k), "pred_e_2k": float(r.pred_e_2k), "ratio": float(r.ratio),
                                 "e_k_out": float(r.e_k_out), "e_2k_out": float(r.e_2k_out), "ratio_out": float(r.ratio_out),
                                 "pred_ratio": float(r.pred_ratio), "harmonic_dominant": bool(r.harmonic_dominant),
                                 "model_rel_err_2k": float(r.model_rel_err_2k), "model_rel_err_k": float(r.model_rel_err_k),
                                 "min_u": float(r.min_u)})
    for kh in sorted(cont.kh.unique()):
        sub = cont[cont.kh == kh].sort_values("a")
        dom = sub[sub.harmonic_dominant]
        out["phase_boundary"][f"{kh:g}"] = {"first_dominant_a": float(dom.a.min()) if len(dom) else None,
                                           "dominant_as": [float(a) for a in dom.a]}
    for _, r in part.iterrows():
        out["particle"].append({"a": float(r.a), "kh": float(r.kh), "N": int(r.N), "e_k_particle": float(r.e_k_particle),
                                "e_2k_particle": float(r.e_2k_particle), "e_k_continuum": float(r.e_k), "e_2k_continuum": float(r.e_2k),
                                "e_k_continuum_out": float(r.e_k_out), "e_2k_continuum_out": float(r.e_2k_out),
                                "rel_diff_k_out": float(abs(r.e_k_particle - r.e_k_out) / abs(r.e_k_out)),
                                "rel_diff_2k_out": float(abs(r.e_2k_particle - r.e_2k_out) / abs(r.e_2k_out)),
                                "ratio_particle": float(r.e_2k_particle / abs(r.e_k_particle)),
                                "attenuation_2k": float(r.e_2k_particle / r.e_2k), "signal_ratio": float(r.e_k_particle / r.e_k)})
    out["max_ratio_out"] = float(cont.ratio_out.max())
    for kh in sorted(cont.kh.unique()):
        r = cont[cont.kh == kh].iloc[0]
        out["low_order_coefficients"][f"{kh:g}"] = {"d": float(r.d), "b": float(r.b), "r1": float(r.r1), "r2": float(r.r2)}
    out["model_max_rel_err_2k"] = float(cont.model_rel_err_2k.max()); out["model_max_rel_err_k"] = float(cont.model_rel_err_k.max())
    t = pd.DataFrame([{"$a$": f"{r['a']:g}", "$kh$": f"{r['kh']:g}", "$e_k$": r["e_k"], "$e_{2k}$": r["e_2k"],
                       "$e_{2k}/|e_k|$": r["ratio"], "expansion": r["pred_ratio"], "dominant": r["harmonic_dominant"],
                       "$e^{\\mathrm{out}}_{2k}/|e^{\\mathrm{out}}_k|$": r["ratio_out"]}
                      for r in out["continuum"]])
    write_table(tables, "crossover_continuum", t, formats={"$e_k$": lambda v: fmt(v, 4, sci_below=1e-3), "$e_{2k}$": lambda v: fmt(v, 4, sci_below=1e-3),
                                                           "$e_{2k}/|e_k|$": lambda v: f"{v:.3f}", "expansion": lambda v: f"{v:.3f}",
                                                           "$e^{\\mathrm{out}}_{2k}/|e^{\\mathrm{out}}_k|$": lambda v: f"{v:.3f}"},
                caption_note="Continuum harmonic transition over the (a, kh) grid; latent-density errors, the expansion's ratio, and the ratio after the output smoothing")
    tp = pd.DataFrame([{"$a$": f"{r['a']:g}", "$N$": r["N"], "particle $e_k$": r["e_k_particle"], "$K_h\\rho$: $e_k$": r["e_k_continuum_out"],
                        "rel. diff.": r["rel_diff_k_out"],
                        "particle $e_{2k}$": r["e_2k_particle"], "$K_h\\rho$: $e_{2k}$": r["e_2k_continuum_out"], "rel. diff. ": r["rel_diff_2k_out"]}
                       for r in out["particle"]])
    write_table(tables, "crossover_particle", tp, formats={c: (lambda v: fmt(v, 4, sci_below=1e-3)) for c in tp.columns if "e_" in c}
                | {"rel. diff.": lambda v: f"{100 * v:.2f}\\%", "rel. diff. ": lambda v: f"{100 * v:.2f}\\%"},
                caption_note="Particle method at kh = 0.264 against the continuum latent density after the same final kernel smoothing (K_h rho)")
    figure_crossover(out, figures)
    return out


def figure_crossover(co: dict, figures: Path) -> None:
    from scripts.plot_revision_figures import figure_crossover as render
    render(co, figures)


def accounting_table(manifest: dict, tables: Path) -> dict:
    tot = manifest["accounting_totals"]
    rows = []
    for study in schema.STUDIES:
        s = json.loads((study_dir(manifest, study) / "summary.json").read_text())["accounting"]
        rows.append({"study": study.replace("_", " "), "expected": s["expected_rows"], "attempted": s["attempted_rows"],
                     "completed": s["completed_rows"], "failed": s["failed_rows"], "endpoint-selected": s["censored_rows"]})
    rows.append({"study": "total", "expected": tot["expected_rows"], "attempted": tot["attempted_rows"], "completed": tot["completed_rows"],
                 "failed": tot["failed_rows"], "endpoint-selected": tot["censored_rows"]})
    write_table(tables, "accounting", pd.DataFrame(rows), caption_note="Run accounting for the pinned campaign")
    return tot


# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="manifests/phase2c_campaign.json")
    ap.add_argument("--out", default="analysis/phase2c")
    ap.add_argument("--paper-figures", default=None, help="optional directory to copy the figures into")
    args = ap.parse_args()
    manifest = load_manifest(REPO / args.manifest)
    out = REPO / args.out
    tables, figures = out / "tables", out / "figures"
    tables.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True)

    summary = {"manifest": args.manifest, "code_commit": manifest["code_commit"], "run_id": manifest["run_id"]}
    summary["accounting"] = accounting_table(manifest, tables)
    summary["bandwidth_clean"] = analyze_bandwidth_clean(manifest, tables)
    figure_bandwidth_clean(summary["bandwidth_clean"], figures)
    summary["lambda_oracle_clean"] = analyze_lambda_oracle_clean(manifest, tables, summary["bandwidth_clean"])
    summary["epsilon"] = analyze_epsilon(manifest, tables)
    summary["adequacy"] = analyze_adequacy(manifest, tables)
    summary["transition"] = analyze_transition(manifest, tables)
    summary["noise"] = analyze_noise(manifest, tables, figures)
    summary["closure"] = analyze_closure(manifest, tables, figures)
    summary["initial_rate"] = analyze_initial_rate(manifest, tables, figures)
    summary["crossover"] = analyze_crossover(manifest, tables, figures)
    from scripts.plot_revision_figures import figure_evidence
    figure_evidence(manifest, summary, figures)
    (out / "analysis_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")
    if args.paper_figures:
        dest = Path(args.paper_figures)
        if not dest.is_absolute():
            dest = REPO / dest
        dest.mkdir(parents=True, exist_ok=True)
        for f in sorted(figures.glob("*.pdf")):
            shutil.copy2(f, dest / f.name)
    shown = out.relative_to(REPO) if out.is_relative_to(REPO) else out
    print(f"analysis written to {shown}: {len(list(tables.glob('*.tex')))} tables, {len(list(figures.glob('*.pdf')))} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
