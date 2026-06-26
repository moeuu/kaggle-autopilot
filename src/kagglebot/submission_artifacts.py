from __future__ import annotations

from pathlib import Path

from kagglebot.artifact_io import copy_artifact_if_needed
from kagglebot.json_utils import load_json_object

SUBMISSION_MANIFEST_FILENAME = "submission_manifest.json"
ARTIFACT_CLASS_TABULAR = "tabular"
ARTIFACT_CLASS_SINGLE_FILE = "single_file"
ARTIFACT_CLASS_BUNDLE = "bundle"
ARTIFACT_CLASS_MULTI_FILE_ZIP = "multi_file_zip"
ARTIFACT_CLASS_NOTEBOOK_OUTPUT = "notebook_output"
ARTIFACT_CLASS_WRITEUP = "writeup"
ARTIFACT_CLASS_UNKNOWN = "unknown"

_KNOWN_ARTIFACT_CLASSES = {
    ARTIFACT_CLASS_TABULAR,
    ARTIFACT_CLASS_SINGLE_FILE,
    ARTIFACT_CLASS_BUNDLE,
    ARTIFACT_CLASS_MULTI_FILE_ZIP,
    ARTIFACT_CLASS_NOTEBOOK_OUTPUT,
    ARTIFACT_CLASS_WRITEUP,
    ARTIFACT_CLASS_UNKNOWN,
}


def normalize_artifact_class(value: object, *, default: str = ARTIFACT_CLASS_UNKNOWN) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in _KNOWN_ARTIFACT_CLASSES:
        return normalized
    return default


def load_submission_manifest(path: Path) -> dict[str, object] | None:
    if not path.exists() or not path.is_file():
        return None
    return load_json_object(path)


def find_submission_manifest(root: Path) -> Path | None:
    candidates = [
        root / SUBMISSION_MANIFEST_FILENAME,
        root / "output" / SUBMISSION_MANIFEST_FILENAME,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    try:
        matches = sorted(path for path in root.rglob(SUBMISSION_MANIFEST_FILENAME) if path.is_file())
    except OSError:
        return None
    return matches[0] if matches else None


def resolve_manifest_references(
    manifest_path: Path,
) -> tuple[str, Path | None, Path | None, list[Path]]:
    payload = load_submission_manifest(manifest_path) or {}
    base_dir = manifest_path.parent
    artifact_class = normalize_artifact_class(payload.get("artifact_class"))
    submission_path = _resolve_manifest_path(base_dir, payload.get("submission_path"))
    staging_dir = _resolve_manifest_path(base_dir, payload.get("staging_dir"))
    if staging_dir is not None and not staging_dir.is_dir():
        staging_dir = None
    members = _resolve_manifest_members(base_dir, payload.get("members"))
    return artifact_class, submission_path, staging_dir, members


def store_submission_artifact(*, source: Path, destination_dir: Path, run_id: str) -> Path:
    destination = destination_dir / f"{run_id}_submission{source.suffix}"
    return copy_artifact_if_needed(source=source, destination=destination)


def _resolve_manifest_path(base_dir: Path, value: object) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = base_dir / path
    return path


def _resolve_manifest_members(base_dir: Path, value: object) -> list[Path]:
    if not isinstance(value, list):
        return []
    members: list[Path] = []
    for item in value:
        if isinstance(item, str):
            path = Path(item)
        elif isinstance(item, dict) and isinstance(item.get("path"), str):
            path = Path(str(item["path"]))
        else:
            continue
        if not path.is_absolute():
            path = base_dir / path
        members.append(path)
    return members
