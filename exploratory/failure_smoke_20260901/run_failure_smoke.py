"""Injected-failure smoke for the campaign drivers (EXPLORATORY).

Injects one failure per driver family through mock patches of the
underlying runners and checks the production plumbing around it:

  1. the failed row is retained with its failure message (and step where
     the runner reports one), never dropped or repaired;
  2. the driver continues past the failure and completes the other rows;
  3. resume treats the failed row as attempted and does not re-run it;
  4. noise_paired excludes the broken block from bandwidth selection and
     reports it as incomplete instead of selecting from a partial curve.

Exploratory output only: results land under this directory with their own
manifest and never feed the manuscript or the campaign summaries.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import campaign_driver_transition as transition
import campaign_drivers as drivers
import campaign_results as results
from invheat_grw.campaign_selectors import SelectionRecord
from provenance import git_commit

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    CHECKS.append((name, passed, detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" +
          (f" ({detail})" if detail else ""))


def read_rows(csv_path: Path) -> list[dict]:
    import csv
    with csv_path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def fail_one_particle_row(real_runner, poison_h: float):
    def wrapper(datum, **kwargs):
        result = real_runner(datum, **kwargs)
        if kwargs.get("bandwidth") == poison_h:
            result.status = "failed"
            result.failure_step = 7
            result.failure_message = "injected failure for the smoke run"
        return result
    return wrapper


def smoke_particle_study(out_root: Path) -> None:
    print("bandwidth_clean: inject a failure at one bandwidth")
    rows = [r for r in results.enumerate_rows("bandwidth_clean")
            if r["case"] == "Z"]
    poison = rows[2]["h"]
    out = out_root / "bandwidth_clean"
    real = drivers.run_campaign_density
    with mock.patch.object(drivers, "run_campaign_density",
                           fail_one_particle_row(real, poison)):
        accounting = drivers.drive_bandwidth_clean(out, rows=rows)
    table = read_rows(out / "bandwidth_clean_rows.csv")
    failed = [r for r in table if r["status"] == results.STATUS_FAILED]
    check("failed row retained", len(failed) == 1 and accounting.failed == 1)
    check("failure step and message recorded",
          failed[0]["failure_step"] == "7"
          and "injected" in failed[0]["failure_message"])
    check("driver continued past the failure",
          accounting.completed == len(rows) - 1)
    accounting_again = drivers.drive_bandwidth_clean(out, rows=rows)
    check("resume does not re-run the failed row",
          accounting_again.failed == 1
          and len(read_rows(out / "bandwidth_clean_rows.csv")) == len(rows))


def smoke_noise_selection(out_root: Path) -> None:
    print("noise_paired: broken block must not select a bandwidth")
    rows = [r for r in results.enumerate_rows("noise_paired")
            if r["case"] == "B" and r["eta"] == 0.001 and r["seed"] == 1]
    poison = rows[3]["h"]
    out = out_root / "noise_paired"
    real = drivers.run_campaign_density
    with mock.patch.object(drivers, "run_campaign_density",
                           fail_one_particle_row(real, poison)):
        accounting = drivers.drive_noise_paired(out, rows=rows)
    check("noise failure retained", accounting.failed == 2)  # both arms
    selection = json.loads((out / "bandwidth_selection.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    check("no selection from a partial curve",
          len(selection["selections"]) == 0)
    check("incomplete blocks reported",
          summary["incomplete_selection_blocks"] == 2)


def smoke_lambda_monotonicity(out_root: Path) -> None:
    print("lambda_noise: inject a monotonicity violation")
    rows = [r for r in results.enumerate_rows("lambda_noise")
            if r["case"] == "B" and r["eta"] == 0.005 and r["seed"] == 2
            and r["arm"] == "P"]
    out = out_root / "lambda_noise"
    broken = SelectionRecord(label="failed_monotonicity", z=None, lam=None,
                             value=None, evaluations=65, endpoint=False)
    with mock.patch.object(drivers, "morozov_tikhonov",
                           lambda residual, target: broken):
        accounting = drivers.drive_lambda_noise(out, rows=rows)
    table = read_rows(out / "lambda_noise_rows.csv")
    failed = [r for r in table if r["status"] == results.STATUS_FAILED]
    check("monotonicity violations are failed rows, not Morozov",
          accounting.failed == 2
          and all(r["selection_label"] == "failed_monotonicity"
                  for r in failed))
    check("oracle row unaffected", accounting.completed == 1)
    summary = json.loads((out / "summary.json").read_text())
    check("failure count surfaced", summary["monotonicity_failures"] == 2)


def smoke_closure_positivity(out_root: Path) -> None:
    print("closure: inject a carrier positivity loss")
    rows = [r for r in results.enumerate_rows("closure")
            if r["case"] == "GB" and r["closure"] == "mass"
            and r["block"] == "carrier"]
    out = out_root / "closure"
    real = drivers.run_gradient_carriers

    def wrapper(g_grid, **kwargs):
        result = real(g_grid, **kwargs)
        if len(g_grid) == 200:
            result.status = "failed"
            result.failure_step = 3
            result.failure_message = "loss of positivity of U (injected)"
            result.u_final = None
            result.q_final = None
        return result

    with mock.patch.object(drivers, "run_gradient_carriers", wrapper):
        accounting = drivers.drive_closure(out, rows=rows)
    table = read_rows(out / "closure_rows.csv")
    failed = [r for r in table if r["status"] == results.STATUS_FAILED]
    check("positivity loss is a failed row",
          accounting.failed == 1 and len(failed) == 1
          and "positivity" in failed[0]["failure_message"])
    gates = json.loads((out / "closure_gates.json").read_text())
    entry = gates["carrier_refinement"][0]
    check("refinement verdict fails on the incomplete ladder",
          entry["pass"] is False)


def smoke_transition(out_root: Path) -> None:
    print("transition_table: inject a legacy-era failure")
    out = out_root / "transition_table"
    rows = [r for r in results.enumerate_rows("transition_table")
            if r["era"] != "endpoint_free_space_legacy"]
    broken = {"completed": False,
              "failure_message": "injected legacy failure"}
    with mock.patch.dict(transition.ERA_RUNNERS,
                         {"cell_centered_free_space_legacy": lambda: broken}):
        accounting = transition.drive_transition_table(out, rows=rows)
    table = read_rows(out / "transition_table_rows.csv")
    failed = [r for r in table if r["status"] == results.STATUS_FAILED]
    check("legacy failure retained",
          accounting.failed == 1 and len(failed) == 1)
    check("canonical era unaffected", accounting.completed == 1)


def main() -> int:
    out_root = HERE / "outputs"
    out_root.mkdir(exist_ok=True)
    smoke_particle_study(out_root)
    smoke_noise_selection(out_root)
    smoke_lambda_monotonicity(out_root)
    smoke_closure_positivity(out_root)
    smoke_transition(out_root)
    passed = sum(1 for _, ok, _ in CHECKS if ok)
    print(f"\n{passed}/{len(CHECKS)} checks passed")
    import hashlib
    files = sorted(f for f in out_root.rglob("*") if f.is_file()
                   and f.name != "smoke_record.json")
    record = {
        "exploratory": True,
        "script": "run_failure_smoke.py",
        "commit": git_commit(REPO),
        "checks": [{"name": n, "passed": ok, "detail": d}
                   for n, ok, d in CHECKS],
        "files": {str(f.relative_to(out_root)):
                  hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
    }
    (out_root / "smoke_record.json").write_text(
        json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(f"record: {out_root / 'smoke_record.json'}")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
