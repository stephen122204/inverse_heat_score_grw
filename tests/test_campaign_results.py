"""Contract tests for campaign_results.py (Phase 2C result contract)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import campaign_results as results  # noqa: E402
import campaign_schema as schema  # noqa: E402


class TestEnumeration(unittest.TestCase):
    def test_total_row_count_matches_preregistration(self):
        total = sum(schema.expected_counts().values())
        self.assertEqual(total, 3230)

    def test_row_keys_unique_within_each_study(self):
        for study in schema.STUDIES:
            keys = results.expected_keys(study)
            self.assertEqual(len(keys), len(set(keys)), study)

    def test_row_keys_unique_across_studies(self):
        all_keys = []
        for study in schema.STUDIES:
            all_keys.extend(results.expected_keys(study))
        self.assertEqual(len(all_keys), len(set(all_keys)))

    def test_identity_fields_cover_every_row(self):
        for study in schema.STUDIES:
            fields = results.identity_fields(study)
            for row in results.enumerate_rows(study):
                self.assertTrue(set(row) <= set(fields), study)


class TestNoiseVectors(unittest.TestCase):
    def test_deterministic_and_seed_dependent(self):
        a = results.standard_normal_vector(3, 400)
        b = results.standard_normal_vector(3, 400)
        c = results.standard_normal_vector(4, 400)
        self.assertEqual(a.shape, (400,))
        np.testing.assert_array_equal(a, b)
        self.assertGreater(float(np.max(np.abs(a - c))), 0.0)


class TestStudyWriter(unittest.TestCase):
    def _rows(self):
        return results.enumerate_rows("epsilon_sensitivity")

    def test_round_trip_accounting_resume_and_guards(self):
        rows = self._rows()
        with TemporaryDirectory() as tmp:
            w = results.StudyWriter("epsilon_sensitivity", Path(tmp))
            w.append(rows[0], results.STATUS_COMPLETED, {"E2": 0.01, "K": 183})
            w.append(rows[1], results.STATUS_COMPLETED, {"E2": 0.02, "K": 183},
                     censored=True)
            w.append(rows[2], results.STATUS_FAILED, {"E2": float("nan"), "K": 0},
                     failure_step=17, failure_message="nonfinite score")

            with self.assertRaises(results.ResultContractError):
                w.append(rows[0], results.STATUS_COMPLETED,
                         {"E2": 0.0, "K": 1})  # duplicate
            with self.assertRaises(results.ResultContractError):
                w.append(rows[3], results.STATUS_COMPLETED,
                         {"E2": 0.0})  # payload schema mismatch
            with self.assertRaises(results.ResultContractError):
                w.append(rows[3], results.STATUS_FAILED,
                         {"E2": 0.0, "K": 1})  # failed without message
            with self.assertRaises(results.ResultContractError):
                w.append({"study": "epsilon_sensitivity", "case": "B",
                          "h": 0.014, "eps_rel": 1e-8, "eta": 0.0},
                         results.STATUS_COMPLETED,
                         {"E2": 0.0, "K": 1})  # not preregistered

            acct = results.reconcile("epsilon_sensitivity", w.csv_path)
            self.assertEqual(acct.attempted, 3)
            self.assertEqual(acct.completed, 2)
            self.assertEqual(acct.failed, 1)
            self.assertEqual(acct.censored, 1)
            self.assertEqual(len(acct.missing), len(rows) - 3)
            self.assertEqual(acct.unexpected, ())
            self.assertEqual(acct.duplicates, ())
            self.assertFalse(acct.consistent)  # not all rows attempted yet

            # Resume: a fresh writer on the same directory sees the work.
            w2 = results.StudyWriter("epsilon_sensitivity", Path(tmp))
            done = w2.done_keys
            for row in rows[:3]:
                self.assertIn(
                    results.study_row_key("epsilon_sensitivity", row), done)
            self.assertNotIn(
                results.study_row_key("epsilon_sensitivity", rows[3]), done)
            with self.assertRaises(results.ResultContractError):
                w2.append(rows[0], results.STATUS_COMPLETED,
                          {"E2": 0.0, "K": 1})  # duplicate across resume

            # Completing the study makes the accounting consistent.
            for row in rows[3:]:
                w2.append(row, results.STATUS_COMPLETED, {"E2": 0.1, "K": 42})
            acct = results.reconcile("epsilon_sensitivity", w2.csv_path)
            self.assertTrue(acct.consistent)
            self.assertEqual(acct.attempted, acct.expected)

    def test_summary_payload(self):
        rows = self._rows()
        with TemporaryDirectory() as tmp:
            w = results.StudyWriter("epsilon_sensitivity", Path(tmp))
            w.append(rows[0], results.STATUS_COMPLETED, {"E2": 0.01})
            acct = results.reconcile("epsilon_sensitivity", w.csv_path)
            path = results.write_summary(Path(tmp), acct, {"gate_x": True})
            self.assertTrue(path.is_file())
            text = path.read_text(encoding="utf-8")
            self.assertIn('"expected_rows"', text)
            self.assertIn('"gate_x": true', text)


if __name__ == "__main__":
    unittest.main()
