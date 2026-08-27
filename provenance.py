"""Explicit run-manifest support for figures and manuscript verification.

The paper pipeline previously selected the lexicographically latest matching
directory under ``outputs/``.  That allowed a partial rerun to supply figures
while ``verify_numbers.py`` continued to check a different archived run.  A
manifest replaces that implicit selection with named, validated study paths.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parent
DEFAULT_MANIFEST = REPO / "manifests" / "paper_v5_1.json"
SCHEMA_VERSION = 1


class ManifestError(RuntimeError):
    """Raised when a run manifest is missing, unsafe, or incomplete."""


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(repo: Path = REPO) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_manifest(path: str | Path | None = None) -> tuple[Path, dict[str, Any]]:
    manifest_path = Path(path) if path is not None else DEFAULT_MANIFEST
    if not manifest_path.is_absolute():
        manifest_path = REPO / manifest_path
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise ManifestError(f"Run manifest not found: {manifest_path}")
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"Cannot read run manifest {manifest_path}: {exc}") from exc
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError(
            f"Unsupported manifest schema {data.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    if not isinstance(data.get("studies"), dict):
        raise ManifestError("Manifest must contain a 'studies' object")
    return manifest_path, data


def study_dir(
    manifest: Mapping[str, Any],
    key: str,
    repo: Path = REPO,
    *,
    verify_hashes: bool = False,
) -> Path:
    """Resolve and validate one named study directory from a manifest."""
    entry = manifest.get("studies", {}).get(key)
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
        raise ManifestError(f"Manifest has no valid study entry for {key!r}")
    root = repo.resolve()
    outputs = (root / "outputs").resolve()
    resolved = (root / entry["path"]).resolve()
    if not _inside(resolved, outputs):
        raise ManifestError(f"Study {key!r} escapes outputs/: {resolved}")
    if not resolved.is_dir():
        raise ManifestError(f"Study directory for {key!r} does not exist: {resolved}")

    required = entry.get("required_files", [])
    if not isinstance(required, list) or not all(isinstance(v, str) for v in required):
        raise ManifestError(f"Study {key!r} has invalid required_files")
    missing = [name for name in required if not (resolved / name).is_file()]
    if missing:
        raise ManifestError(
            f"Study {key!r} is incomplete; missing: {', '.join(sorted(missing))}"
        )

    if verify_hashes:
        hashes = entry.get("sha256", {})
        if not isinstance(hashes, dict):
            raise ManifestError(f"Study {key!r} has invalid sha256 mapping")
        unhashed = sorted(set(required) - set(hashes))
        if unhashed:
            raise ManifestError(
                f"Study {key!r} has required files without hashes: {', '.join(unhashed)}"
            )
        for name, expected in hashes.items():
            candidate = resolved / name
            if not candidate.is_file():
                raise ManifestError(f"Hashed file is missing for {key!r}: {name}")
            actual = sha256_file(candidate)
            if actual != expected:
                raise ManifestError(
                    f"Hash mismatch for {key!r}/{name}: {actual} != {expected}"
                )
    return resolved


def validate_manifest(
    manifest: Mapping[str, Any],
    keys: Sequence[str],
    repo: Path = REPO,
    *,
    verify_hashes: bool = False,
) -> dict[str, Path]:
    return {
        key: study_dir(manifest, key, repo, verify_hashes=verify_hashes)
        for key in keys
    }


def write_manifest(
    path: str | Path,
    studies: Mapping[str, Path],
    commands: Sequence[Mapping[str, Any]],
    repo: Path = REPO,
    *,
    run_id: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write an immutable manifest for directories created by one run."""
    root = repo.resolve()
    output_path = Path(path)
    if not output_path.is_absolute():
        output_path = root / output_path
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    entries: dict[str, Any] = {}
    for key, directory in studies.items():
        resolved = directory.resolve()
        if not _inside(resolved, (root / "outputs").resolve()):
            raise ManifestError(f"Cannot manifest directory outside outputs/: {resolved}")
        files = sorted(p for p in resolved.iterdir() if p.is_file())
        entries[key] = {
            "path": str(resolved.relative_to(root)),
            "required_files": [p.name for p in files],
            "sha256": {p.name: sha256_file(p) for p in files},
        }

    created = datetime.now(timezone.utc).isoformat()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "created_utc": created,
        "code_commit": git_commit(root),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "commands": list(commands),
        "studies": entries,
    }
    if extra:
        for key, value in extra.items():
            if key in payload:
                raise ManifestError(f"extra manifest field {key!r} collides with a reserved field")
            payload[key] = value
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path
