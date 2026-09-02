"""Pre-use verification tests for the Stage 6 campaign drivers.

Each test runs a small preregistered subset through the real production path
(build_case -> run_campaign_density / oracle_continuous -> StudyWriter), so
the accounting is intentionally inconsistent (missing rows) while attempted,
completed, resume, and payload behavior are checked exactly.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import campaign_drivers as drivers  # noqa: E402
import campaign_results as results  # noqa: E402
import campaign_schema as schema  # noqa: E402
from invheat_grw.campaign_selectors import Z_BOUNDS  # noqa: E402


def study_rows(study: str, keep) -> list[dict]:
    return [r for r in results.enumerate_rows(study) if keep(r)]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


class TestCaseConstruction(unittest.TestCase):
    def test_truth_forwards_to_the_clean_terminal_datum(self):
        for name in ("C1", "H", "B"):
            setup = drivers.build_case(name)
            residual = setup.residual_fn(setup.terminal_clean)
            self.assertLess(residual(setup.truth), 5e-4, name)

    def test_oracle_objective_beats_a_detuned_lambda(self):
        setup = drivers.build_case("C1")
        objective = drivers.truth_error_objective("C1", setup.terminal_clean)
        self.assertLess(objective(-8.0), objective(-2.0))


class TestParticleDrivers(unittest.TestCase):
    def test_epsilon_driver_runs_resumes_and_keeps_the_payload_contract(self):
        rows = study_rows("epsilon_sensitivity",
                          lambda r: r["case"] == "C1")[:2]
        self.assertEqual(len(rows), 2)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            accounting = drivers.drive_epsilon_sensitivity(out, rows=rows)
            self.assertEqual(accounting.attempted, 2)
            self.assertEqual(accounting.completed, 2)
            self.assertEqual(accounting.failed, 0)
            table = read_csv(out / "epsilon_sensitivity_rows.csv")
            for key in drivers.PARTICLE_PAYLOAD_KEYS:
                self.assertIn(key, table[0])
            for r in table:
                self.assertLess(float(r["E2"]), 0.5)
                self.assertTrue(math.isfinite(float(r["mass_reconstruction"])))
                self.assertEqual(r["status"], results.STATUS_COMPLETED)
            again = drivers.drive_epsilon_sensitivity(out, rows=rows)
            self.assertEqual(again.attempted, 2)
            self.assertEqual(len(read_csv(out / "epsilon_sensitivity_rows.csv")), 2)

    def test_adequacy_driver_writes_the_case_verdicts(self):
        rows = study_rows("adequacy_N",
                          lambda r: r["case"] == "C1"
                          and r["N"] in (4000, 10000))
        self.assertEqual(len(rows), 2)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            accounting = drivers.drive_adequacy(out, rows=rows)
            self.assertEqual(accounting.completed, 2)
            summary = json.loads((out / "summary.json").read_text())
            self.assertIn("adequacy_C1", summary["verdicts"])
            self.assertIn("adequacy_mass_C1", summary["verdicts"])
            self.assertTrue(summary["verdicts"]["adequacy_mass_C1"])


class TestNoiseDrivers(unittest.TestCase):
    def test_noise_paired_block_runs_both_arms_and_selects_bandwidths(self):
        rows = study_rows("noise_paired",
                          lambda r: r["case"] == "B" and r["eta"] == 0.005
                          and r["seed"] == 0)
        self.assertEqual(len(rows), 2 * len(schema.BANDWIDTHS))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            accounting = drivers.drive_noise_paired(out, rows=rows)
            self.assertEqual(accounting.completed, len(rows))
            table = read_csv(out / "noise_paired_rows.csv")
            raw = [r for r in table if r["arm"] == "R"]
            self.assertGreater(float(raw[0]["negative_fraction"]), 0.0)
            for r in table:
                self.assertGreater(float(r["delta_nom"]), 0.0)
                self.assertTrue(math.isfinite(float(r["delta_real"])))
            selection = json.loads(
                (out / "bandwidth_selection.json").read_text())
            self.assertEqual(len(selection["selections"]), 4)
            for s in selection["selections"]:
                self.assertEqual(s["label"], "residual_matched")
                self.assertIn(s["selected_h"], schema.BANDWIDTHS)
                self.assertEqual(len(s["curve_r"]), len(schema.BANDWIDTHS))
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["incomplete_selection_blocks"], 0)

    def test_lambda_noise_writes_all_three_selections_for_a_block(self):
        rows = study_rows("lambda_noise",
                          lambda r: r["case"] == "B" and r["eta"] == 0.01
                          and r["seed"] == 3 and r["arm"] == "R")
        self.assertEqual(len(rows), 3)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            accounting = drivers.drive_lambda_noise(out, rows=rows)
            self.assertEqual(accounting.completed, 3)
            table = {r["selection"] + r["tau"]: r
                     for r in read_csv(out / "lambda_noise_rows.csv")}
            oracle = table["oracle_continuousnone"]
            self.assertEqual(oracle["selection_label"], "oracle_continuous")
            self.assertLess(float(oracle["E2_at_selection"]), 1.0)
            for tau in (schema.TAU_HEADLINE, schema.TAU_SENSITIVITY):
                r = table[f"residual{tau}"]
                self.assertIn(r["selection_label"], ("morozov", "endpoint"))
                target = float(r["target_residual"])
                self.assertAlmostEqual(target,
                                       tau * float(r["delta_nom"]), places=12)
                if r["selection_label"] == "morozov":
                    self.assertAlmostEqual(
                        float(r["residual_at_selection"]), target,
                        delta=1e-6 * target)
            for r in table.values():
                for key in ("E2_tikhonov_raw", "E2_tikhonov_projected"):
                    self.assertTrue(math.isfinite(float(r[key])), key)
                self.assertEqual(r["E2_tikhonov_raw"], r["E2_at_selection"])

    def test_projection_is_nonnegative_and_mass_exact(self):
        dx = 1.0 / 400
        u = np.linspace(-0.5, 1.5, 400)
        m0 = 0.8
        p = drivers.project_positive_mass(u, m0, dx)
        self.assertGreaterEqual(float(p.min()), 0.0)
        self.assertAlmostEqual(float(dx * p.sum()), m0, places=12)
        # already-feasible fields are fixed points
        v = np.full(400, m0)
        self.assertLess(float(np.max(np.abs(drivers.project_positive_mass(v, m0, dx) - v))), 1e-12)


class TestClosureDriver(unittest.TestCase):
    def test_g1_frozen_left_subset_produces_gates_and_anchor(self):
        keep = {("reference", "regularized", 1600), ("reference",
                                                     "regularized", 3200)}

        def selector(r):
            if r["case"] != "G1" or r["closure"] != "frozen_left":
                return False
            if r["block"] == "reference":
                return (r["block"], r["kind"], r["M"]) in keep
            if r["block"] == "carrier":
                return r["M"] == 400
            return r["block"] == "split_invariance"

        rows = study_rows("closure", selector)
        self.assertEqual(len(rows), 4)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            accounting = drivers.drive_closure(out, rows=rows)
            self.assertEqual(accounting.completed, 4)
            gates = json.loads((out / "closure_gates.json").read_text())
            pair = gates["reference_pairs"][0]
            self.assertEqual((pair["case"], pair["kind"]),
                             ("G1", "regularized"))
            for label in ("half", "final"):
                for key in ("q_rel", "u_rel"):
                    self.assertTrue(
                        math.isfinite(pair["comparisons"][label][key]))
            self.assertLessEqual(pair["comparisons"]["final"]["q_rel"],
                                 schema.CLOSURE_REF_GATE)
            split = gates["split_invariance"][0]
            self.assertTrue(split["pass"], split)
            carrier_row = [r for r in read_csv(out / "closure_rows.csv")
                           if r["block"] == "carrier"][0]
            self.assertTrue(math.isfinite(float(carrier_row["q_diff_ref"])))
            decomposition = json.loads(
                (out / "closure_decomposition.json").read_text())
            anchor = decomposition["analytic_anchor_G1"]
            self.assertLess(anchor["abs_error_vs_formula"], 1e-3)
            self.assertGreater(anchor["magnitude"], 0.09)
            statuses = {(e["case"], e["closure"]): e["status"]
                        for e in decomposition["cases"]}
            self.assertEqual(statuses[("G1", "frozen_left")],
                             "missing_inputs")
            # Heterogeneous closure rows must resume, not duplicate.
            again = drivers.drive_closure(out, rows=rows)
            self.assertEqual(again.attempted, 4)
            self.assertEqual(len(read_csv(out / "closure_rows.csv")), 4)

    def test_closure_row_keys_are_unique_under_the_study_convention(self):
        rows = results.enumerate_rows("closure")
        keys = {results.study_row_key("closure", r) for r in rows}
        self.assertEqual(len(keys), len(rows))


class TestSelectorDriver(unittest.TestCase):
    def test_lambda_oracle_driver_records_the_clean_endpoint_selection(self):
        rows = study_rows("lambda_oracle_clean",
                          lambda r: r["case"] == "C1")
        self.assertEqual(len(rows), 1)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            accounting = drivers.drive_lambda_oracle_clean(out, rows=rows)
            self.assertEqual(accounting.completed, 1)
            # Clean data: truth error decreases monotonically as lambda -> 0,
            # so the oracle sits at the small-lambda endpoint and the row is
            # recorded as censored rather than silently clipped.
            self.assertEqual(accounting.censored, 1)
            row = read_csv(out / "lambda_oracle_clean_rows.csv")[0]
            self.assertEqual(row["selection_label"], "oracle_continuous")
            self.assertLess(float(row["z_selected"]), Z_BOUNDS[0] + 0.5)
            self.assertLess(float(row["E2_at_selection"]), 0.05)
            self.assertGreater(int(row["evaluations"]), 60)


class TestRegistry(unittest.TestCase):
    def test_registry_matches_the_implemented_studies(self):
        self.assertEqual(
            sorted(drivers.DRIVERS),
            ["adequacy_N", "bandwidth_clean", "closure", "crossover",
             "epsilon_sensitivity", "initial_rate", "lambda_noise",
             "lambda_oracle_clean", "noise_paired"])
        for name in drivers.DRIVERS:
            self.assertIn(name, schema.STUDIES)


if __name__ == "__main__":
    unittest.main()


class TestInitialRateDriver(unittest.TestCase):
    def test_block_a_references_land_in_the_certificate_band(self):
        rows = study_rows("initial_rate",
                          lambda r: r["block"] == "reference"
                          or (r["block"] == "carrier" and r["M"] == 200)
                          or (r["block"] == "q_level" and r["tau"] == 0.0125))
        self.assertEqual(len(rows), 4)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            accounting = drivers.drive_initial_rate(out, rows=rows)
            self.assertEqual(accounting.completed, 4)
            table = read_csv(out / "initial_rate_rows.csv")
            refs = [r for r in table if r["block"] == "reference"]
            for r in refs:
                self.assertEqual(r["within_band"], "True")
                self.assertLess(abs(float(r["ratio"]) - 1.0), 0.0074)
            gates = json.loads((out / "initial_rate_gates.json").read_text())
            self.assertTrue(gates["verdicts"]["block_a_within_certificate_band"])
            self.assertTrue(
                gates["verdicts"]["block_a_reference_pair_defect_relative"])
            carrier = [r for r in table if r["block"] == "carrier"][0]
            self.assertTrue(math.isfinite(float(carrier["u_diff_ref"])))
            q = [r for r in table if r["block"] == "q_level"][0]
            self.assertLess(abs(float(q["ratio_q"]) - 1.0), 0.05)
            again = drivers.drive_initial_rate(out, rows=rows)
            self.assertEqual(again.attempted, 4)

    def test_constants_match_the_closed_forms(self):
        c = drivers.initial_rate_constants("G1")
        self.assertAlmostEqual(c["c_rep"], 0.061389608316868, places=12)
        self.assertAlmostEqual(c["slope_q"], 0.2103823758, places=9)


class TestCrossoverDriver(unittest.TestCase):
    def test_continuum_and_particle_rows_complete_with_finite_payloads(self):
        rows = study_rows("crossover",
                          lambda r: r["a"] == 0.35 and r["kh"] == 0.264
                          and r["N"] in (None, 4000))
        self.assertEqual(len(rows), 2)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            accounting = drivers.drive_crossover(out, rows=rows)
            self.assertEqual(accounting.completed, 2)
            table = read_csv(out / "crossover_rows.csv")
            for r in table:
                for key in ("e_k", "e_2k", "d", "b", "r1", "r2"):
                    self.assertTrue(math.isfinite(float(r[key])), key)
            particle = [r for r in table if r["block"] == "particle"][0]
            self.assertTrue(math.isfinite(float(particle["e_2k_particle"])))

    def test_low_order_model_reproduces_the_archived_anchor(self):
        # Locks the jet hierarchy at kh = 0.264 against normalization or
        # factor-of-two regressions (values reproduced independently before
        # locking; ledger 71).
        d, b, r1, r2 = drivers.crossover_low_order(0.264)
        self.assertAlmostEqual(d, 0.0299630452, places=9)
        self.assertAlmostEqual(b, 0.0594158705, places=9)
        self.assertAlmostEqual(r1, -0.0035205899, places=9)
        self.assertAlmostEqual(r2, 0.0144296719, places=9)

