"""sync_paper_figures.py — deliberately update the manuscript's figure copies.

Copies from the code repository's figures/ into the sibling
paper_draft/figures/, updating only files that already exist there, so stale
or exploratory figures never enter the manuscript tree.  This is the ONLY
path by which figure generation reaches the paper; no figure builder writes
outside the code repository as a side effect.

Usage:
    python scripts/sync_paper_figures.py            # report what would change
    python scripts/sync_paper_figures.py --apply    # perform the copies
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOURCE = REPO / "figures"
TARGET = REPO.parent / "paper_draft" / "figures"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="copy the files; without this flag only report")
    args = ap.parse_args()

    if not TARGET.is_dir():
        print(f"No manuscript figure directory at {TARGET}; nothing to sync.")
        return 1

    changed, same, missing = [], [], []
    for dest in sorted(TARGET.iterdir()):
        if not dest.is_file() or dest.suffix not in (".pdf", ".png"):
            continue
        src = SOURCE / dest.name
        if not src.is_file():
            missing.append(dest.name)
        elif filecmp.cmp(src, dest, shallow=False):
            same.append(dest.name)
        else:
            changed.append(dest.name)

    for name in changed:
        print(f"  update: {name}")
        if args.apply:
            shutil.copy2(SOURCE / name, TARGET / name)
    for name in missing:
        print(f"  no source in figures/: {name} (left untouched)")
    print(f"{len(changed)} to update, {len(same)} identical, "
          f"{len(missing)} without source"
          + ("" if args.apply else "  (dry run; use --apply)"))
    return 0


if __name__ == "__main__":
    main()
