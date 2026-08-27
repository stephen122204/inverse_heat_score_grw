"""run_science_campaign.py — preregistered-campaign planner and gatekeeper.

While PHASE2C_PROTOCOL.md is PROPOSED, this script can only plan: it
enumerates every preregistered row from campaign_schema and reports the
expected-row accounting that the campaign summaries must later reproduce.
Execution is refused until the protocol status line reads FROZEN, and the
study implementations land only after that freeze.

Usage:
    python scripts/run_science_campaign.py             # print the plan
    python scripts/run_science_campaign.py --out F     # also write JSON
    python scripts/run_science_campaign.py --run STUDY # refused until FROZEN
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import campaign_schema as schema
from provenance import git_commit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None,
                    help="write the full row plan to this JSON file")
    ap.add_argument("--run", default=None, metavar="STUDY",
                    help="execute one preregistered study (requires FROZEN)")
    args = ap.parse_args()

    status = schema.protocol_status()
    counts = schema.expected_counts()
    total = sum(counts.values())

    print(f"protocol status: {status}")
    print(f"code commit:     {git_commit(REPO)[:12]}")
    print("preregistered rows per study:")
    for name, count in counts.items():
        print(f"  {name:22s} {count:5d}")
    print(f"  {'TOTAL':22s} {total:5d}")

    if args.out:
        payload = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_status": status,
            "code_commit": git_commit(REPO),
            "expected_counts": counts,
            "total_rows": total,
            "rows": {name: fn() for name, fn in schema.STUDIES.items()},
        }
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n",
                                  encoding="utf-8")
        print(f"wrote plan: {args.out}")

    if args.run is not None:
        if args.run not in schema.STUDIES:
            print(f"unknown study {args.run!r}; choose from "
                  f"{sorted(schema.STUDIES)}")
            return 2
        try:
            schema.assert_frozen()
        except schema.ProtocolNotFrozen as refusal:
            print(f"refused: {refusal}", file=sys.stderr)
            return 3
        print("not implemented: study execution lands with the frozen "
              "campaign build", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
