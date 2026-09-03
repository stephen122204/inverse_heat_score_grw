"""The campaign analysis reproduces the pinned verdicts from the manifest alone."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "manifests" / "phase2c_campaign.json"


@unittest.skipUnless(MANIFEST.exists(), "pinned campaign manifest not present")
class TestAnalyzeCampaign(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        out = Path(cls.tmp.name) / "analysis"
        subprocess.run([sys.executable, "scripts/analyze_campaign.py", "--out", str(out)],
                       cwd=REPO, check=True, capture_output=True)
        cls.out = out
        cls.summary = json.loads((out / "analysis_summary.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_accounting_matches_the_pinned_totals(self):
        manifest = json.loads(MANIFEST.read_text())
        self.assertEqual(self.summary["accounting"], manifest["accounting_totals"])
        self.assertEqual(self.summary["code_commit"], manifest["code_commit"])

    def test_every_table_and_figure_is_written(self):
        tables = sorted(p.name for p in (self.out / "tables").glob("*.tex"))
        for name in ("accounting.tex", "bandwidth_clean.tex", "noise_paired.tex",
                     "closure_decomposition_u.tex", "initial_rate_reference.tex", "crossover_continuum.tex"):
            self.assertIn(name, tables)
        figures = sorted(p.name for p in (self.out / "figures").glob("*.pdf"))
        self.assertEqual(figures, ["bandwidth_clean.pdf", "closure_decomposition.pdf", "crossover.pdf",
                                   "initial_rate.pdf", "noise_window.pdf"])

    def test_theorem_gate_and_selection_verdicts_are_reported_not_recomputed(self):
        ir = self.summary["initial_rate"]
        self.assertTrue(ir["verdicts"]["block_a_within_certificate_band"])
        self.assertTrue(all(r["within_band"] for r in ir["references"]))
        closure = self.summary["closure"]
        self.assertTrue(all(v["reconciled"] for v in closure["decomposition"].values()))
        self.assertFalse(closure["verdicts"]["refinement_G1_mass"])
        noise = self.summary["noise"]
        block = noise["blocks"]["C1|0.001|P"]
        self.assertEqual(block["n_realizations"], 25)
        self.assertEqual(block["particle_oracle"]["endpoint_count"], 25)

    def test_paired_statistics_are_internally_consistent(self):
        for block in self.summary["noise"]["blocks"].values():
            d = block["paired_deployable_1.2"]
            self.assertLessEqual(d["ci_low"], d["mean"])
            self.assertLessEqual(d["mean"], d["ci_high"])
            self.assertGreaterEqual(d["fraction_negative"], 0.0)
            self.assertLessEqual(d["fraction_negative"], 1.0)


if __name__ == "__main__":
    unittest.main()
