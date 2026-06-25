from __future__ import annotations

import re
from pathlib import Path

from kagglebot.submission_artifacts import find_submission_manifest, resolve_manifest_references

_INTERMEDIATE_SUBMISSION_RE = re.compile(r"^submission(?:_[A-Za-z0-9_.-]+)?_fold(?P<fold>\d+)\.csv$", re.IGNORECASE)


def find_submission_file(output_dir: Path) -> Path | None:
    manifest_path = find_submission_manifest(output_dir)
    if manifest_path is not None:
        _, submission_path, staging_dir, members = resolve_manifest_references(manifest_path)
        if submission_path is not None and submission_path.exists() and submission_path.is_file():
            return submission_path
        if staging_dir is not None or members:
            return manifest_path
    candidate = find_output_file(output_dir, "submission.csv")
    if candidate:
        return candidate
    candidate = find_intermediate_submission_file(output_dir)
    if candidate:
        return candidate
    return _find_submission_by_extension(output_dir)


def find_output_file(output_dir: Path, filename: str) -> Path | None:
    """Find the newest matching artifact within an output tree.

    Local kernels can be executed repeatedly for the same run/iteration while
    iterating on fixes. In that scenario, stale artifacts may exist alongside
    fresh ones (or nested under additional run directories). Prefer the most
    recently modified match to avoid accidentally reusing stale outputs.
    """

    candidates: list[Path] = []
    direct = output_dir / filename
    if direct.exists():
        candidates.append(direct)
    try:
        candidates.extend(path for path in output_dir.rglob(filename) if path.exists())
    except OSError:
        # Best-effort discovery; callers handle missing artifacts.
        pass
    files = [path for path in candidates if path.is_file()]
    if not files:
        return None
    # Deterministic tie-breaker: path string.
    return max(files, key=lambda path: (path.stat().st_mtime, str(path)))


def find_intermediate_submission_file(output_dir: Path) -> Path | None:
    """Find the newest fold-level submission emitted by an interrupted run."""

    candidates: list[tuple[int, Path]] = []
    try:
        paths = output_dir.rglob("submission*_fold*.csv")
    except OSError:
        return None
    for path in paths:
        if not path.is_file():
            continue
        match = _INTERMEDIATE_SUBMISSION_RE.match(path.name)
        if match is None:
            continue
        try:
            fold = int(match.group("fold"))
        except ValueError:
            continue
        candidates.append((fold, path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[1].stat().st_mtime, item[0], str(item[1])))[1]


def _find_submission_by_extension(output_dir: Path) -> Path | None:
    suffixes = {".csv", ".tsv", ".txt", ".parquet", ".json", ".jsonl", ".zip"}
    compound_names = {"submission.tar.gz", "submission.tgz"}
    candidates: list[Path] = []
    for name in sorted(compound_names):
        candidate = output_dir / name
        if candidate.is_file():
            candidates.append(candidate)
    for suffix in sorted(suffixes):
        candidate = output_dir / f"submission{suffix}"
        if candidate.is_file():
            candidates.append(candidate)
    for path in output_dir.rglob("submission.*"):
        if not path.is_file():
            continue
        if path.name.lower() in compound_names:
            candidates.append(path)
            continue
        if path.suffix.lower() not in suffixes:
            continue
        candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime, str(path)))
