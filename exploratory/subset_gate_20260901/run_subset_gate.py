"""Small end-to-end subset through the production drivers (EXPLORATORY).

Coverage chosen to hit every machinery family on real physics before the
full campaign launch:

  - closure: every preregistered G1 and G2 row (references at both
    resolutions with the per-time gate, carrier refinement ladder, split
    invariance, h-bridge, decomposition, analytic anchor);
  - bandwidth_clean: the mixture case H and the variable case VB05, the
    latter exercising the de-crimed 8x/16x data gate and the banded
    Crank-Nicolson forward residual through the production path;
  - lambda_oracle_clean: VB05, exercising the SVD Tikhonov branch.

Exploratory output only; the campaign runs fresh from the orchestrator.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import campaign_drivers as drivers
import campaign_results as results
from provenance import git_commit


def main() -> int:
    out_root = HERE / "outputs"
    out_root.mkdir(exist_ok=True)
    report: dict = {"exploratory": True, "commit": git_commit(REPO),
                    "studies": {}}

    plan = [
        ("closure", drivers.drive_closure,
         lambda r: r["case"] in ("G1", "G2")),
        ("bandwidth_clean", drivers.drive_bandwidth_clean,
         lambda r: r["case"] in ("H", "VB05")),
        ("lambda_oracle_clean", drivers.drive_lambda_oracle_clean,
         lambda r: r["case"] == "VB05"),
    ]
    for study, driver, keep in plan:
        rows = [r for r in results.enumerate_rows(study) if keep(r)]
        t0 = time.perf_counter()
        accounting = driver(out_root / study, rows=rows)
        elapsed = time.perf_counter() - t0
        report["studies"][study] = {
            "rows": len(rows),
            "completed": accounting.completed,
            "failed": accounting.failed,
            "censored": accounting.censored,
            "seconds": round(elapsed, 1),
        }
        print(f"{study}: {accounting.completed}/{len(rows)} completed, "
              f"{accounting.failed} failed, {elapsed:.0f}s", flush=True)

    (out_root / "subset_report.json").write_text(
        json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"report: {out_root / 'subset_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
