"""
plotting.py — All diagnostic plots for the inverse heat score GRW experiment.

Each function saves one PNG to the specified output directory.
Functions return the figure path for logging.

Plots generated:
  1. field_comparison_all.png    — all candidates vs true_u0 and observed_final
  2. method_panels.png           — per-method panel: true, observed, candidate, error
  3. step_snapshots_oracle_det.png
  4. step_snapshots_estimated_det_raw.png
  5. metrics_bar_l2.png          — L2 error bar chart
  6. peak_width_tv_by_method.png — peak / FWHM / TV comparison
  7. score_diagnostics.png       — max|score|, mean, std vs step per method
  8. forward_consistency_by_method.png
  9. naive_vs_oracle.png         — direct comparison
 10. residuals.png               — true_u0 - candidate per method
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for script use
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional

from .methods import MethodResult
from .metrics import MethodMetrics

# Consistent color map for methods
METHOD_COLORS = {
    "naive_backward":                       "#e41a1c",
    "oracle_score_deterministic":           "#377eb8",
    "oracle_score_stochastic":              "#4daf4a",
    "estimated_score_deterministic_raw":    "#ff7f00",
    "estimated_score_stochastic_raw":       "#984ea3",
    "true_u0":                              "black",
    "observed_final":                       "gray",
}

METHOD_LABELS = {
    "naive_backward":                       "Naive backward",
    "oracle_score_deterministic":           "Oracle det. (prob-flow)",
    "oracle_score_stochastic":              "Oracle stoch. (SDE)",
    "estimated_score_deterministic_raw":    "Est. det. raw",
    "estimated_score_stochastic_raw":       "Est. stoch. raw",
    "true_u0":                              "True u₀",
    "observed_final":                       "Observed u(T)",
}


def _method_label(name: str) -> str:
    return METHOD_LABELS.get(name, name)


def _method_color(name: str) -> str:
    return METHOD_COLORS.get(name, "black")


def _save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 1. field_comparison_all
# ---------------------------------------------------------------------------

def plot_field_comparison_all(
    x: np.ndarray,
    true_u0: np.ndarray,
    observed_final: np.ndarray,
    results: Dict[str, MethodResult],
    out_dir: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, true_u0, color=_method_color("true_u0"), lw=2.5,
            label=_method_label("true_u0"), zorder=10)
    ax.plot(x, observed_final, color=_method_color("observed_final"), lw=1.5,
            ls="--", label=_method_label("observed_final"))
    for name, res in results.items():
        ls = "-" if res.completed else ":"
        label = _method_label(name)
        if not res.completed:
            label += f" [FAILED @ step {res.failure_step}]"
        ax.plot(x, res.candidate, color=_method_color(name), lw=1.5,
                ls=ls, label=label)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.set_title("Field comparison: all methods vs true u₀")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    return _save(fig, out_dir / "field_comparison_all.png")


# ---------------------------------------------------------------------------
# 2. method_panels
# ---------------------------------------------------------------------------

def plot_method_panels(
    x: np.ndarray,
    true_u0: np.ndarray,
    observed_final: np.ndarray,
    results: Dict[str, MethodResult],
    out_dir: Path,
) -> Path:
    names = list(results.keys())
    ncols = 2
    nrows = (len(names) + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows), squeeze=False)
    axes_flat = [ax for row in axes for ax in row]

    for i, name in enumerate(names):
        ax = axes_flat[i]
        res = results[name]
        ax.plot(x, true_u0, color="black", lw=1.5, label="True u₀")
        ax.plot(x, observed_final, color="gray", lw=1, ls="--", label="Observed u(T)")
        ax.plot(x, res.candidate, color=_method_color(name), lw=2,
                label="Candidate")

        # Pointwise error
        ax2 = ax.twinx()
        err = res.candidate - true_u0
        ax2.fill_between(x, err, alpha=0.2, color="red", label="Error")
        ax2.axhline(0, color="red", lw=0.5, ls="--")
        ax2.set_ylabel("Error", color="red", fontsize=8)

        title = _method_label(name)
        if not res.completed:
            title += f"\n[FAILED step {res.failure_step}]"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for j in range(len(names), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout()
    return _save(fig, out_dir / "method_panels.png")


# ---------------------------------------------------------------------------
# 3/4. step_snapshots
# ---------------------------------------------------------------------------

def _plot_step_snapshots(
    x: np.ndarray,
    true_u0: np.ndarray,
    observed_final: np.ndarray,
    result: MethodResult,
    out_path: Path,
) -> Path:
    snaps = result.step_snapshots
    if not snaps:
        return out_path  # nothing to plot

    steps = sorted(snaps.keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, true_u0, "k-", lw=2, label="True u₀", zorder=10)
    ax.plot(x, observed_final, "k--", lw=1, alpha=0.5, label="Observed u(T)")

    cmap = plt.cm.plasma
    for i, step in enumerate(steps):
        color = cmap(i / max(len(steps) - 1, 1))
        ax.plot(x, snaps[step], color=color, lw=1.2, label=f"Step {step}")

    ax.set_xlabel("x")
    ax.set_ylabel("u")
    title = _method_label(result.method_name)
    if not result.completed:
        title += f" [FAILED @ step {result.failure_step}]"
    ax.set_title(f"Step snapshots — {title}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return _save(fig, out_path)


def plot_step_snapshots_oracle_det(
    x, true_u0, observed_final, results: Dict[str, MethodResult], out_dir: Path
) -> Path:
    res = results.get("oracle_score_deterministic")
    if res is None:
        return out_dir / "step_snapshots_oracle_det.png"
    return _plot_step_snapshots(x, true_u0, observed_final, res,
                                out_dir / "step_snapshots_oracle_det.png")


def plot_step_snapshots_estimated_det_raw(
    x, true_u0, observed_final, results: Dict[str, MethodResult], out_dir: Path
) -> Path:
    res = results.get("estimated_score_deterministic_raw")
    if res is None:
        return out_dir / "step_snapshots_estimated_det_raw.png"
    return _plot_step_snapshots(x, true_u0, observed_final, res,
                                out_dir / "step_snapshots_estimated_det_raw.png")


# ---------------------------------------------------------------------------
# 5. metrics_bar_l2
# ---------------------------------------------------------------------------

def plot_metrics_bar_l2(
    metrics_list: List[MethodMetrics],
    out_dir: Path,
) -> Path:
    names = [m.method_name for m in metrics_list]
    l2s = [m.l2_error for m in metrics_list]
    colors = [_method_color(n) for n in names]
    labels = [_method_label(n) for n in names]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(range(len(names)), l2s, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("L2_h(candidate − true_u₀)")
    ax.set_title("L2 Recovery Error by Method")

    for bar, val, m in zip(bars, l2s, metrics_list):
        note = "" if m.completed else f"\nFAILED@{m.failure_step}"
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.001,
                f"{val:.4f}{note}", ha="center", va="bottom", fontsize=7)

    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, out_dir / "metrics_bar_l2.png")


# ---------------------------------------------------------------------------
# 6. peak_width_tv_by_method
# ---------------------------------------------------------------------------

def plot_peak_width_tv(
    metrics_list: List[MethodMetrics],
    true_peak: float,
    true_fwhm: float,
    true_tv: float,
    out_dir: Path,
) -> Path:
    names = [m.method_name for m in metrics_list]
    labels = [_method_label(n) for n in names]
    colors = [_method_color(n) for n in names]
    x_idx = np.arange(len(names))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Peak
    axes[0].bar(x_idx, [m.peak_value for m in metrics_list], color=colors)
    axes[0].axhline(true_peak, color="black", ls="--", lw=1.5, label=f"True peak={true_peak:.3f}")
    axes[0].set_title("Peak value")
    axes[0].legend(fontsize=8)

    # FWHM
    fwhms = [m.peak_width_fwhm for m in metrics_list]
    axes[1].bar(x_idx, fwhms, color=colors)
    axes[1].axhline(true_fwhm, color="black", ls="--", lw=1.5, label=f"True FWHM={true_fwhm:.4f}")
    axes[1].set_title("Peak width (FWHM)")
    axes[1].legend(fontsize=8)

    # TV
    axes[2].bar(x_idx, [m.total_variation for m in metrics_list], color=colors)
    axes[2].axhline(true_tv, color="black", ls="--", lw=1.5, label=f"True TV={true_tv:.4f}")
    axes[2].set_title("Total variation")
    axes[2].legend(fontsize=8)

    for ax in axes:
        ax.set_xticks(x_idx)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Peak / Width / TV by method", fontsize=11)
    fig.tight_layout()
    return _save(fig, out_dir / "peak_width_tv_by_method.png")


# ---------------------------------------------------------------------------
# 7. score_diagnostics
# ---------------------------------------------------------------------------

def plot_score_diagnostics(
    results: Dict[str, MethodResult],
    out_dir: Path,
) -> Path:
    # Only methods that have score history
    score_methods = {
        name: res for name, res in results.items()
        if len(res.score_max_abs) > 0
    }
    if not score_methods:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No score diagnostics available", ha="center")
        return _save(fig, out_dir / "score_diagnostics.png")

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    for name, res in score_methods.items():
        color = _method_color(name)
        label = _method_label(name)
        steps = np.arange(len(res.score_max_abs))

        axes[0].plot(steps, res.score_max_abs, color=color, label=label)
        axes[1].plot(steps, res.score_mean, color=color, label=label)
        axes[2].plot(steps, res.score_std, color=color, label=label)

        if res.failure_step is not None and res.failure_step <= len(res.score_max_abs):
            for ax in axes:
                ax.axvline(res.failure_step, color=color, ls=":", lw=1.5, alpha=0.7)

    axes[0].set_ylabel("max |score|")
    axes[0].set_title("Score diagnostics vs backward step")
    axes[1].set_ylabel("mean score")
    axes[2].set_ylabel("std score")
    axes[2].set_xlabel("backward step k")

    for ax in axes:
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("symlog", linthresh=1.0)

    fig.tight_layout()
    return _save(fig, out_dir / "score_diagnostics.png")


# ---------------------------------------------------------------------------
# 8. forward_consistency_by_method
# ---------------------------------------------------------------------------

def plot_forward_consistency(
    metrics_list: List[MethodMetrics],
    out_dir: Path,
) -> Path:
    names = [m.method_name for m in metrics_list]
    labels = [_method_label(n) for n in names]
    colors = [_method_color(n) for n in names]
    vals = [m.forward_consistency_l2 for m in metrics_list]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(range(len(names)), vals, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("L2_h(forward(candidate) − observed_final)")
    ax.set_title("Forward Consistency by Method")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.0001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, out_dir / "forward_consistency_by_method.png")


# ---------------------------------------------------------------------------
# 9. naive_vs_oracle
# ---------------------------------------------------------------------------

def plot_naive_vs_oracle(
    x: np.ndarray,
    true_u0: np.ndarray,
    observed_final: np.ndarray,
    results: Dict[str, MethodResult],
    out_dir: Path,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax_i, (method_name, title) in enumerate([
        ("naive_backward", "Naive Backward"),
        ("oracle_score_deterministic", "Oracle Score Deterministic"),
    ]):
        ax = axes[ax_i]
        ax.plot(x, true_u0, "k-", lw=2, label="True u₀")
        ax.plot(x, observed_final, "k--", lw=1, alpha=0.5, label="Observed u(T)")
        res = results.get(method_name)
        if res is not None:
            ls = "-" if res.completed else ":"
            label = "Candidate"
            if not res.completed:
                label += f" [FAILED@{res.failure_step}]"
            ax.plot(x, res.candidate, color=_method_color(method_name),
                    lw=2, ls=ls, label=label)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Naive backward vs Oracle score deterministic", fontsize=11)
    fig.tight_layout()
    return _save(fig, out_dir / "naive_vs_oracle.png")


# ---------------------------------------------------------------------------
# 10. residuals
# ---------------------------------------------------------------------------

def plot_residuals(
    x: np.ndarray,
    true_u0: np.ndarray,
    results: Dict[str, MethodResult],
    out_dir: Path,
) -> Path:
    names = list(results.keys())
    ncols = 2
    nrows = (len(names) + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.5 * nrows), squeeze=False)
    axes_flat = [ax for row in axes for ax in row]

    for i, name in enumerate(names):
        ax = axes_flat[i]
        residual = true_u0 - results[name].candidate
        ax.plot(x, residual, color=_method_color(name), lw=1.5)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.fill_between(x, residual, alpha=0.2, color=_method_color(name))
        title = _method_label(name)
        if not results[name].completed:
            title += f" [FAILED@{results[name].failure_step}]"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x")
        ax.set_ylabel("true_u₀ − candidate")
        ax.grid(True, alpha=0.3)

    for j in range(len(names), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Residuals: true_u₀ − candidate", fontsize=11)
    fig.tight_layout()
    return _save(fig, out_dir / "residuals.png")
