"""verify_figure_freeze.py — recompute the four illustration-figure datasets
with the current code and compare them against the manifest-pinned archive.

This is the live integration gate for supposedly behavior-preserving changes:
run it after each such commit.  The default comparison demands bitwise
equality, which holds on the pinned local environment; --tol enables a
platform-tolerant comparison (max abs difference relative to
max(1, max|frozen|)) for use across numerical libraries.

Usage:
    python scripts/verify_figure_freeze.py
    python scripts/verify_figure_freeze.py --tol 1e-13
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from figure_data import FIGURES, STUDY_KEY, compare_datasets, compute_all, load_dataset
from provenance import DEFAULT_MANIFEST, load_manifest, study_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST),
                    help="manifest naming the frozen figure-data study")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="0 (default) demands bitwise equality; a positive "
                         "value allows that relative max-abs tolerance")
    args = ap.parse_args()

    manifest_path, manifest = load_manifest(args.manifest)
    frozen_dir = study_dir(manifest, STUDY_KEY, REPO, verify_hashes=True)
    print(f"[provenance] {manifest_path}")
    print(f"[frozen]     {frozen_dir} (hashes verified)")

    frozen = load_dataset(frozen_dir)
    print(f"Recomputing {len(FIGURES)} figure datasets with the current code ...",
          flush=True)
    live = compute_all()

    problems = compare_datasets(live, frozen, tol=args.tol)
    mode = "bitwise" if args.tol <= 0.0 else f"tol={args.tol:g}"
    for fig in FIGURES:
        fig_problems = [p for p in problems if p.startswith(f"{fig}/")]
        verdict = "MATCH" if not fig_problems else "MISMATCH"
        print(f"  {fig:28s} {verdict} ({mode})")
        for p in fig_problems:
            print(f"    {p}")
    if problems:
        print(f"\nFREEZE VERIFY: FAIL ({len(problems)} mismatches)")
        return 1
    print("\nFREEZE VERIFY: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
