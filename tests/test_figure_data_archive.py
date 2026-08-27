"""Fast checks on the frozen illustration-figure dataset.

These validate the archived artifact itself — manifest entry, hashes,
metadata, array inventory, and internal consistency — without recomputing
any simulation.  The live recomputation gate is scripts/verify_figure_freeze.py.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from figure_data import DATA_FILE, FIGURES, META_FILE, STUDY_KEY, load_dataset
from provenance import DEFAULT_MANIFEST, load_manifest, study_dir

EXPECTED_KEYS = {
    "fig_naive": {"x", "u0", "uT", "naive", "score", "rng_seed"},
    "fig_density_loop": {"x", "u0", "uT", "snapshot_steps", "snapshots", "dt"},
    "fig_variable_field": {"x", "u0", "uT", "candidate", "a", "beta"},
    "fig_representation_failure": {"x", "u0", "g", "density", "glob",
                                   "e2_density", "e2_glob", "rng_seed"},
}


class FigureDataArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, manifest = load_manifest(DEFAULT_MANIFEST)
        cls.study = study_dir(manifest, STUDY_KEY, REPO, verify_hashes=True)
        cls.data = load_dataset(cls.study)

    def test_manifest_pins_both_files_with_hashes(self):
        self.assertTrue((self.study / DATA_FILE).is_file())
        self.assertTrue((self.study / META_FILE).is_file())

    def test_every_figure_has_its_expected_arrays(self):
        for fig in FIGURES:
            self.assertEqual(set(self.data[fig]), EXPECTED_KEYS[fig], fig)

    def test_arrays_are_finite_and_shape_consistent(self):
        for fig, entries in self.data.items():
            n = entries["x"].shape[0]
            for name, arr in entries.items():
                arr = np.asarray(arr, dtype=float)
                self.assertTrue(np.all(np.isfinite(arr)), f"{fig}/{name}")
                if arr.ndim == 1 and name not in ("snapshot_steps",):
                    self.assertEqual(arr.shape[0], n, f"{fig}/{name}")
        loop = self.data["fig_density_loop"]
        self.assertEqual(loop["snapshots"].shape,
                         (loop["snapshot_steps"].shape[0], loop["x"].shape[0]))

    def test_metadata_records_provenance(self):
        meta = json.loads((self.study / META_FILE).read_text(encoding="utf-8"))
        for field in ("created_utc", "command", "code_commit",
                      "environment", "figures", "parameters"):
            self.assertIn(field, meta)
        self.assertEqual(set(meta["figures"]), set(FIGURES))
        for fig in FIGURES:
            self.assertEqual(set(meta["figures"][fig]), EXPECTED_KEYS[fig], fig)

    def test_frozen_error_scalars_match_their_arrays(self):
        rf = self.data["fig_representation_failure"]
        x, u0 = rf["x"], rf["u0"]
        dx = float(x[1] - x[0])
        u0n = float(np.sqrt(dx * np.sum(u0 ** 2)))

        def e2(c):
            return float(np.sqrt(dx * np.sum((c - u0) ** 2))) / u0n

        self.assertAlmostEqual(float(rf["e2_density"]), e2(rf["density"]), places=15)
        self.assertAlmostEqual(float(rf["e2_glob"]), e2(rf["glob"]), places=15)


if __name__ == "__main__":
    unittest.main()
