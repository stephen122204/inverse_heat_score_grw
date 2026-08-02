"""Shared matplotlib style for the inverse-heat manuscript figures.

Mirrors the figure style of the companion GRW papers: Computer Modern serif
text, one method = one color in every figure across the whole paper, framed
legends, and a short factual title on each figure. Test cases are
distinguished by markers and linestyles, never by color.
"""

import matplotlib.pyplot as plt

# One method = one color in every figure across the whole paper.
TRUTH = "#111111"       # true initial condition u0
OBS = "#777777"         # observed terminal field g = u(., T)
METHOD = "#2166ac"      # density-particle method, smoothed-log score, bandwidth factor 4
METHOD_ALT = "#74add1"  # same method at bandwidth factor 6
EXACT = "#d6604d"       # exact-score diagnostic runs and reference levels
GLOB = "#762a83"        # gradient-glob method
TIKH = "#1b7837"        # Tikhonov reference
GRID_RATIO = "#e08214"  # finite-difference grid-ratio estimator
NAIVE = "#8c510a"       # naive backward random walk

# Test cases carry markers and linestyles, not colors.
TEST_MARKER = {"B": "o", "H": "s", "Z": "^"}
TEST_LS = {"B": "-", "H": "--", "Z": ":"}

# Reverse-time snapshot shades for the reconstruction-loop figure (light to dark).
LOOP_SHADES = ["#92c5de", "#4393c3", "#2166ac"]


def apply_paper_style():
    """Apply the shared publication style (no LaTeX installation required)."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "lines.linewidth": 1.9,
        "lines.markersize": 5.5,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })
