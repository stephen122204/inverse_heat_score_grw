"""campaign_results.py — result contract for the Phase 2C campaign.

One canonical row identity, one CSV convention, resume support, deterministic
noise vectors, and the expected/attempted/completed/failed/censored accounting
that PHASE2C_PROTOCOL.md Section 10 requires.  Study drivers append outcome
rows to per-study CSVs; this module owns identity, column ordering, accounting
arithmetic, and the summary payloads that feed the pinned campaign manifest
(written through provenance.write_manifest at orchestration time).

Contract rules enforced here:
- every preregistered row (campaign_schema.STUDIES) has one stable identity
  string; a study CSV may contain each identity at most once;
- an outcome is either completed or failed; a failed row keeps its identity,
  failure step, and message, and stays in every denominator;
- endpoint selections are completed rows carrying censored = True;
- the payload field set is fixed per study at first write, so a partial rerun
  cannot silently change the schema of an existing CSV;
- accounting reconciles the CSV against the preregistered enumeration and
  reports missing, unexpected, and duplicate identities explicitly.
"""

from __future__ import annotations

import csv
from functools import lru_cache
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import campaign_schema as schema

STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUSES = (STATUS_COMPLETED, STATUS_FAILED)

RESERVED_FIELDS = ("status", "censored", "failure_step", "failure_message")

# Fixed salt for the campaign noise stream; the standard-normal vector of a
# block is a function of (seed, m) alone, so it is shared by every method,
# input variant, and parameter candidate inside a (case, eta, seed) block,
# exactly as the paired design requires.
NOISE_SALT = 0x2C0DE


class ResultContractError(RuntimeError):
    """Raised when an outcome row violates the campaign result contract."""


def _format_value(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def row_key(row: Mapping[str, Any]) -> str:
    """Canonical stable identity string for a preregistered row."""
    return "|".join(f"{k}={_format_value(row[k])}" for k in sorted(row))


def enumerate_rows(study: str) -> list[dict]:
    if study not in schema.STUDIES:
        raise ResultContractError(f"unknown study {study!r}")
    return schema.STUDIES[study]()


def study_row_key(study: str, row: Mapping[str, Any]) -> str:
    """Identity string over the study's full identity-field union, so
    heterogeneous rows (the closure blocks) key consistently with the CSV."""
    fields = identity_fields(study)
    return "|".join(f"{k}={_format_value(row.get(k))}" for k in sorted(fields))


def expected_keys(study: str) -> list[str]:
    return [study_row_key(study, r) for r in enumerate_rows(study)]


def identity_fields(study: str) -> list[str]:
    """Union of identity fields over the study's enumeration, in first-seen
    order, so every CSV in one study shares one identity header."""
    return list(_identity_fields_cached(study))


@lru_cache(maxsize=None)
def _identity_fields_cached(study: str) -> tuple[str, ...]:
    fields: list[str] = []
    for row in enumerate_rows(study):
        for key in row:
            if key not in fields:
                fields.append(key)
    return tuple(fields)


def standard_normal_vector(seed: int, m: int) -> np.ndarray:
    """Deterministic standard-normal vector for a paired noise block."""
    rng = np.random.default_rng(np.random.SeedSequence((NOISE_SALT, int(seed))))
    return rng.standard_normal(int(m))


@dataclass(frozen=True)
class Accounting:
    study: str
    expected: int
    attempted: int
    completed: int
    failed: int
    censored: int
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    duplicates: tuple[str, ...]

    @property
    def consistent(self) -> bool:
        return (
            self.attempted == self.expected
            and self.completed + self.failed == self.attempted
            and not self.missing
            and not self.unexpected
            and not self.duplicates
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "study": self.study,
            "expected_rows": self.expected,
            "attempted_rows": self.attempted,
            "completed_rows": self.completed,
            "failed_rows": self.failed,
            "censored_rows": self.censored,
            "missing_rows": list(self.missing),
            "unexpected_rows": list(self.unexpected),
            "duplicate_rows": list(self.duplicates),
            "consistent": self.consistent,
        }


class StudyWriter:
    """Append-only CSV writer for one study with identity and payload checks."""

    def __init__(self, study: str, directory: Path):
        self.study = study
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.directory / f"{study}_rows.csv"
        self._identity = identity_fields(study)
        self._expected = set(expected_keys(study))
        self._payload_fields: list[str] | None = None
        self._written: set[str] = set()
        if self.csv_path.exists():
            self._load_existing()

    def _load_existing(self) -> None:
        with self.csv_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            header = reader.fieldnames or []
            prefix = self._identity + list(RESERVED_FIELDS)
            if header[: len(prefix)] != prefix:
                raise ResultContractError(
                    f"{self.csv_path} header does not match the {self.study} "
                    "identity/reserved contract"
                )
            self._payload_fields = header[len(prefix):]
            for record in reader:
                identity = {k: record[k] for k in self._identity}
                key = "|".join(
                    f"{k}={identity[k]}" for k in sorted(identity)
                )
                self._written.add(key)

    @property
    def done_keys(self) -> set[str]:
        return set(self._written)

    def _serialized_identity(self, row: Mapping[str, Any]) -> dict[str, str]:
        return {k: _format_value(row.get(k)) for k in self._identity}

    def append(
        self,
        row: Mapping[str, Any],
        status: str,
        payload: Mapping[str, Any],
        *,
        censored: bool = False,
        failure_step: int | None = None,
        failure_message: str = "",
    ) -> None:
        if status not in STATUSES:
            raise ResultContractError(f"invalid status {status!r}")
        key = study_row_key(self.study, row)
        if key not in self._expected:
            raise ResultContractError(
                f"row is not in the preregistered {self.study} enumeration: {key}"
            )
        serialized = self._serialized_identity(row)
        stored_key = "|".join(f"{k}={serialized[k]}" for k in sorted(serialized))
        if stored_key in self._written:
            raise ResultContractError(f"duplicate outcome for row: {key}")
        if status == STATUS_FAILED and failure_message == "":
            raise ResultContractError("a failed row requires a failure message")
        if status == STATUS_COMPLETED and failure_step is not None:
            raise ResultContractError("a completed row cannot carry a failure step")

        payload_fields = sorted(payload)
        if self._payload_fields is None:
            self._payload_fields = payload_fields
            header = self._identity + list(RESERVED_FIELDS) + payload_fields
            with self.csv_path.open("w", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerow(header)
        elif payload_fields != self._payload_fields:
            raise ResultContractError(
                f"payload fields {payload_fields} do not match the study's "
                f"established fields {self._payload_fields}"
            )

        record = dict(serialized)
        record["status"] = status
        record["censored"] = str(bool(censored))
        record["failure_step"] = "" if failure_step is None else str(failure_step)
        record["failure_message"] = failure_message
        for name in self._payload_fields:
            record[name] = _format_value(payload[name])
        with self.csv_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=self._identity + list(RESERVED_FIELDS) + self._payload_fields,
            )
            writer.writerow(record)
        self._written.add(stored_key)


def reconcile(study: str, csv_path: Path) -> Accounting:
    """Compute the Section 10 accounting for one study CSV."""
    expected_list = expected_keys(study)
    expected = set(expected_list)
    ident = identity_fields(study)
    attempted = 0
    completed = 0
    failed = 0
    censored = 0
    seen: list[str] = []
    if Path(csv_path).exists():
        with Path(csv_path).open(newline="", encoding="utf-8") as stream:
            for record in csv.DictReader(stream):
                identity = {k: record[k] for k in ident if k in record}
                key = "|".join(f"{k}={identity[k]}" for k in sorted(identity))
                seen.append(key)
                attempted += 1
                if record.get("status") == STATUS_COMPLETED:
                    completed += 1
                elif record.get("status") == STATUS_FAILED:
                    failed += 1
                if record.get("censored") == "True":
                    censored += 1
    seen_set = set(seen)
    duplicates = tuple(sorted({k for k in seen if seen.count(k) > 1}))
    missing = tuple(sorted(expected - seen_set))
    unexpected = tuple(sorted(seen_set - expected))
    return Accounting(
        study=study,
        expected=len(expected_list),
        attempted=attempted,
        completed=completed,
        failed=failed,
        censored=censored,
        missing=missing,
        unexpected=unexpected,
        duplicates=duplicates,
    )


def write_summary(
    directory: Path,
    accounting: Accounting,
    verdicts: Mapping[str, bool],
    extra: Mapping[str, Any] | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "accounting": accounting.as_dict(),
        "verdicts": dict(verdicts),
    }
    if extra:
        payload.update(dict(extra))
    path = Path(directory) / "summary.json"
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path
