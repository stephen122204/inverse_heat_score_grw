"""
run_single.py — Run a single backward GRW method.

Usage:
    python scripts/run_single.py --config configs/gaussian_base.yaml \
        --method oracle_score_deterministic

Available methods:
    naive_backward
    oracle_score_deterministic
    oracle_score_stochastic
    estimated_score_deterministic_raw
    estimated_score_stochastic_raw
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from invheat_grw.config import load_config
from invheat_grw.fields import make_grid, true_u0 as compute_true_u0, observed_final as compute_observed_final
from invheat_grw.methods import (
    run_naive_backward,
    run_oracle_score_deterministic,
    run_oracle_score_stochastic,
    run_estimated_score_deterministic_raw,
    run_estimated_score_stochastic_raw,
)
from invheat_grw.metrics import compute_metrics
from invheat_grw.io_utils import make_output_dir, save_config, save_arrays

METHODS = {
    "naive_backward": run_naive_backward,
    "oracle_score_deterministic": run_oracle_score_deterministic,
    "oracle_score_stochastic": run_oracle_score_stochastic,
    "estimated_score_deterministic_raw": run_estimated_score_deterministic_raw,
    "estimated_score_stochastic_raw": run_estimated_score_stochastic_raw,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single inverse heat score GRW method.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--method", type=str, required=True, choices=list(METHODS.keys()))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    cfg = load_config(config_path)

    project_root = Path(__file__).resolve().parent.parent
    out_dir = make_output_dir(project_root / "outputs")
    print(f"Output directory: {out_dir}")

    rng = np.random.default_rng(cfg.grw.rng_seed)
    x = make_grid(cfg)
    u0 = compute_true_u0(x, cfg)
    u_obs = compute_observed_final(x, cfg)

    fn = METHODS[args.method]
    print(f"Running: {args.method} ...")
    res = fn(u_obs, x, cfg, rng)
    status = "OK" if res.completed else f"FAILED at step {res.failure_step}: {res.failure_msg}"
    print(f"Status: {status}")

    m = compute_metrics(res, u0, u_obs, x, cfg)
    print(f"L2 error:          {m.l2_error:.6f}")
    print(f"Peak value:        {m.peak_value:.4f}  (true: {float(np.max(u0)):.4f})")
    print(f"Forward cons. L2:  {m.forward_consistency_l2:.6f}")

    save_config(cfg, config_path, out_dir)
    save_arrays(x, u0, u_obs, {args.method: res}, out_dir)
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
