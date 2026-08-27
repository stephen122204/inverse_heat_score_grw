from __future__ import annotations

import copy
import unittest
from pathlib import Path

import numpy as np

from invheat_grw.config import load_config
from invheat_grw.fields import make_grid
from invheat_grw.methods import MethodResult
from invheat_grw.metrics import compute_metrics


REPO = Path(__file__).resolve().parent.parent


class MetricsSchemaTests(unittest.TestCase):
    def test_forward_consistency_primary_field_is_relative(self):
        cfg = copy.deepcopy(load_config(str(REPO / "configs" / "gaussian_base.yaml")))
        cfg.domain.n_grid = 101
        cfg.heat.T = cfg.heat.dt
        x = make_grid(cfg)
        observed = np.ones_like(x)
        result = MethodResult(
            method_name="zero_candidate",
            completed=True,
            failure_step=None,
            failure_msg="",
            candidate=np.zeros_like(x),
        )
        metrics = compute_metrics(result, observed, observed, x, cfg)
        expected_absolute = float(np.sqrt((x[1] - x[0]) * observed.size))
        self.assertAlmostEqual(metrics.forward_consistency_l2, 1.0, places=14)
        self.assertAlmostEqual(
            metrics.forward_consistency_l2_absolute, expected_absolute, places=14
        )


if __name__ == "__main__":
    unittest.main()

