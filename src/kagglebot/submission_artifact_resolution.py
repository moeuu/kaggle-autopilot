from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from kagglebot.exceptions import SubmissionValidationError
from kagglebot.hashing import sha256_file_or_none
from kagglebot.json_utils import load_json_object
from kagglebot.submission_sample_discovery import tabular_suffix


class SubmissionArtifactResolutionError(ValueError):
    pass


@dataclass(frozen=True)
class SubmissionArtifactCandidate:
    path: Path
    provenance: str
    precedence: int
    expected_sha256: str | None = None


@dataclass(frozen=True)
class ValidatedSubmissionArtifact:
    source_path: Path
    prepared_path: Path
    provenance: str


_STDOUT_FILENAMES = ("local_kernel_stdout.log", "local_kernel_stdout_oom_retry.log")
_KNOWN_METADATA_FILENAMES = {
    "metrics.json",
    "predictions_manifest.json",
    "run_manifest.json",
}


def is_known_submission_metadata(path: Path) -> bool:
    """Return whether *path* is metadata that must never be submitted directly."""

    name = path.name.lower()
    if name in _KNOWN_METADATA_FILENAMES:
        return True
    if path.suffix.lower() != ".json":
        return False
    normalized = name.replace("-", "_")
    return "manifest" in normalized or "candidate_contract" in normalized


def find_authoritative_submission_path(artifact_root: Path) -> Path | None:
    """Resolve a kernel-reported submission path without guessing by mtime."""

    candidates = _authoritative_candidates(artifact_root)
    for precedence in sorted({candidate.precedence for candidate in candidates}):
        usable = _usable_candidates(candidate for candidate in candidates if candidate.precedence == precedence)
        unique = _unique_candidates_by_path(usable)
        if len(unique) == 1:
            return unique[0].path
        if len(unique) > 1:
            return None
    return None


def resolve_valid_submission_artifact(
    *,
    iteration_dir: Path,
    validate_and_prepare: Callable[[Path], Path],
    explicit_submission_path: Path | None = None,
) -> ValidatedSubmissionArtifact:
    """Resolve one semantically valid current-iteration submission artifact.

    An explicit operator path is validated as-is and never replaced. Automatic
    recovery prefers kernel-reported paths, then the conventional
    ``submission.csv`` name, and finally a bounded scan of immediate CSV files.
    """

    if explicit_submission_path is not None:
        prepared = validate_and_prepare(explicit_submission_path)
        return ValidatedSubmissionArtifact(
            source_path=explicit_submission_path,
            prepared_path=prepared,
            provenance="operator_explicit",
        )

    authoritative = _authoritative_candidates(iteration_dir)
    for precedence in sorted({candidate.precedence for candidate in authoritative}):
        validated = _validate_candidates(
            (candidate for candidate in authoritative if candidate.precedence == precedence),
            validate_and_prepare=validate_and_prepare,
        )
        if len(validated) == 1:
            return validated[0]
        if len(validated) > 1:
            sources = ", ".join(str(candidate.source_path) for candidate in validated)
            raise SubmissionArtifactResolutionError(
                "authoritative submission reports are ambiguous at the same precedence: " + sources
            )

    conventional = [
        SubmissionArtifactCandidate(
            path=path,
            provenance="conventional_submission_filename",
            precedence=3,
        )
        for path in (iteration_dir / "output" / "submission.csv", iteration_dir / "submission.csv")
    ]
    validated_conventional = _validate_candidates(conventional, validate_and_prepare=validate_and_prepare)
    if len(validated_conventional) == 1:
        return validated_conventional[0]
    if len(validated_conventional) > 1:
        sources = ", ".join(str(candidate.source_path) for candidate in validated_conventional)
        raise SubmissionArtifactResolutionError(
            "multiple valid conventional submission.csv artifacts are ambiguous: " + sources
        )

    reported_paths = {candidate.path.resolve() for candidate in authoritative if candidate.path.exists()}
    conventional_paths = {candidate.path.resolve() for candidate in conventional if candidate.path.exists()}
    csv_candidates: list[SubmissionArtifactCandidate] = []
    for directory in (iteration_dir / "output", iteration_dir):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.csv")):
            resolved = path.resolve()
            if resolved in reported_paths or resolved in conventional_paths:
                continue
            csv_candidates.append(
                SubmissionArtifactCandidate(
                    path=path,
                    provenance="bounded_csv_search",
                    precedence=4,
                )
            )
    validated_csvs = _validate_candidates(csv_candidates, validate_and_prepare=validate_and_prepare)
    if len(validated_csvs) == 1:
        return validated_csvs[0]
    if len(validated_csvs) > 1:
        sources = ", ".join(str(candidate.source_path) for candidate in validated_csvs)
        raise SubmissionArtifactResolutionError(
            "multiple valid uncorroborated CSV submission artifacts are ambiguous: " + sources
        )
    raise SubmissionArtifactResolutionError(
        f"no semantically valid submission artifact was found in current iteration: {iteration_dir}"
    )


def atomic_copy_submission_autofix(
    *,
    source_path: Path,
    iteration_dir: Path,
    validate_and_prepare: Callable[[Path], Path],
) -> Path:
    """Copy a recovered artifact to a new run-owned path and revalidate it."""

    suffix = tabular_suffix(source_path) or source_path.suffix or ".csv"
    destination = iteration_dir / f"submission_autofix{suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            with source_path.open("rb") as source:
                shutil.copyfileobj(source, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    prepared = validate_and_prepare(destination)
    if not prepared.exists() or not prepared.is_file():
        raise SubmissionArtifactResolutionError(
            f"validation did not produce an existing repaired submission file: {prepared}"
        )
    return prepared


def find_current_iteration_dir(*, artifact_paths: Iterable[Path], fallback_dirs: Iterable[Path]) -> Path | None:
    fallbacks = [path for path in fallback_dirs if path.exists() and path.is_dir()]
    fallback_resolved = {path.resolve(): path for path in fallbacks}
    for artifact_path in artifact_paths:
        for parent in (artifact_path, *artifact_path.parents):
            if not parent.name.startswith("iter-"):
                continue
            resolved = parent.resolve()
            if not fallback_resolved or resolved in fallback_resolved:
                return fallback_resolved.get(resolved, parent)
    return fallbacks[0] if fallbacks else None


def _authoritative_candidates(artifact_root: Path) -> list[SubmissionArtifactCandidate]:
    scope_root = _artifact_scope_root(artifact_root)
    candidates: list[SubmissionArtifactCandidate] = []
    for log_path in _stdout_log_paths(artifact_root, scope_root):
        payload = _load_last_submission_stdout_payload(log_path)
        if payload is None:
            continue
        candidate = _candidate_from_payload(
            payload,
            document_path=log_path,
            scope_root=scope_root,
            provenance=f"stdout:{log_path}",
            precedence=0,
        )
        if candidate is not None:
            candidates.append(candidate)

    for filename, precedence in (("metrics.json", 1), ("run_manifest.json", 2)):
        for report_path in _report_paths(artifact_root, filename):
            payload = load_json_object(report_path)
            if payload is None:
                continue
            candidate = _candidate_from_payload(
                payload,
                document_path=report_path,
                scope_root=scope_root,
                provenance=f"{filename}:{report_path}",
                precedence=precedence,
            )
            if candidate is not None:
                candidates.append(candidate)
    return _unique_candidates_by_provenance(candidates)


def _artifact_scope_root(artifact_root: Path) -> Path:
    if artifact_root.name == "output" and artifact_root.parent.name.startswith("iter-"):
        return artifact_root.parent
    return artifact_root


def _stdout_log_paths(artifact_root: Path, scope_root: Path) -> list[Path]:
    paths: list[Path] = []
    for directory in (artifact_root / "logs", scope_root / "logs", artifact_root):
        for filename in _STDOUT_FILENAMES:
            path = directory / filename
            if path.is_file() and path not in paths:
                paths.append(path)
    return paths


def _report_paths(artifact_root: Path, filename: str) -> list[Path]:
    paths: list[Path] = []
    for path in (artifact_root / filename, artifact_root / "output" / filename):
        if path.is_file() and path not in paths:
            paths.append(path)
    return paths


def _load_last_submission_stdout_payload(path: Path) -> dict[str, object] | None:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(size - 1024 * 1024, 0))
            text = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    for line in reversed(text.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and (payload.get("submission") or payload.get("submission_path")):
            return payload
    return None


def _candidate_from_payload(
    payload: dict[str, object],
    *,
    document_path: Path,
    scope_root: Path,
    provenance: str,
    precedence: int,
) -> SubmissionArtifactCandidate | None:
    submission = payload.get("submission")
    expected_sha256: str | None = None
    raw_path: object = submission
    if isinstance(submission, dict):
        raw_path = submission.get("path") or submission.get("submission_path")
        expected_sha256 = str(submission.get("sha256") or "").strip().lower() or None
    if not isinstance(raw_path, str) or not raw_path.strip():
        raw_path = payload.get("submission_path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    path = _resolve_reported_path(raw_path, document_path=document_path, scope_root=scope_root)
    if path is None or is_known_submission_metadata(path):
        return None
    return SubmissionArtifactCandidate(
        path=path,
        provenance=provenance,
        precedence=precedence,
        expected_sha256=expected_sha256,
    )


def _resolve_reported_path(raw_path: str, *, document_path: Path, scope_root: Path) -> Path | None:
    requested = Path(raw_path.strip()).expanduser()
    candidates = (
        [requested]
        if requested.is_absolute()
        else [
            document_path.parent / requested,
            scope_root / requested,
            scope_root / "output" / requested,
        ]
    )
    scope_resolved = scope_root.resolve()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            resolved.relative_to(scope_resolved)
        except (OSError, ValueError):
            continue
        if resolved.exists() and resolved.is_file():
            return resolved
    return None


def _usable_candidates(candidates: Iterable[SubmissionArtifactCandidate]) -> list[SubmissionArtifactCandidate]:
    usable: list[SubmissionArtifactCandidate] = []
    for candidate in candidates:
        if not candidate.path.exists() or not candidate.path.is_file():
            continue
        if is_known_submission_metadata(candidate.path):
            continue
        if candidate.expected_sha256:
            actual_sha256 = sha256_file_or_none(candidate.path)
            if actual_sha256 != candidate.expected_sha256:
                continue
        usable.append(candidate)
    return usable


def _validate_candidates(
    candidates: Iterable[SubmissionArtifactCandidate],
    *,
    validate_and_prepare: Callable[[Path], Path],
) -> list[ValidatedSubmissionArtifact]:
    validated: list[ValidatedSubmissionArtifact] = []
    for candidate in _usable_candidates(candidates):
        try:
            prepared = validate_and_prepare(candidate.path)
        except SubmissionValidationError:
            continue
        if not prepared.exists() or not prepared.is_file():
            continue
        validated.append(
            ValidatedSubmissionArtifact(
                source_path=candidate.path,
                prepared_path=prepared,
                provenance=candidate.provenance,
            )
        )
    unique: dict[tuple[Path, Path], ValidatedSubmissionArtifact] = {}
    for candidate in validated:
        key = (candidate.source_path.resolve(), candidate.prepared_path.resolve())
        unique.setdefault(key, candidate)
    return list(unique.values())


def _unique_candidates_by_path(
    candidates: Iterable[SubmissionArtifactCandidate],
) -> list[SubmissionArtifactCandidate]:
    unique: dict[Path, SubmissionArtifactCandidate] = {}
    for candidate in candidates:
        unique.setdefault(candidate.path.resolve(), candidate)
    return list(unique.values())


def _unique_candidates_by_provenance(
    candidates: Iterable[SubmissionArtifactCandidate],
) -> list[SubmissionArtifactCandidate]:
    unique: dict[tuple[Path, int], SubmissionArtifactCandidate] = {}
    for candidate in candidates:
        unique.setdefault((candidate.path.resolve(), candidate.precedence), candidate)
    return list(unique.values())
