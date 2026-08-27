"""generate_figure_data.py — create the frozen field arrays behind the four
illustration figures (naive reversal, density loop, variable field, and
representation failure).

Writes outputs/paper_figure_data_TIMESTAMP/ containing figure_data.npz and
figure_data_meta.json.  reproduce.py runs this as a study step so every full
reproduction manifests its own figure data; the archived paper manifest pins
the dataset generated from the archived methods.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from figure_data import FIGURES, compute_all, save_dataset


def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPO / "outputs" / f"paper_figure_data_{ts}"
    print(f"Computing frozen data for {len(FIGURES)} figures ...", flush=True)
    data = compute_all()
    save_dataset(data, out, command=" ".join([sys.executable] + sys.argv))
    for fig, entries in data.items():
        arrays = ", ".join(sorted(entries))
        print(f"  {fig}: {arrays}")
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
