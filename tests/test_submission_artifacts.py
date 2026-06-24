from __future__ import annotations

import json
from pathlib import Path

from kagglebot.submission_artifacts import (
    ARTIFACT_CLASS_BUNDLE,
    ARTIFACT_CLASS_UNKNOWN,
    load_submission_manifest,
    normalize_artifact_class,
    resolve_manifest_references,
)


def test_load_submission_manifest_returns_object_payload(tmp_path: Path) -> None:
    path = tmp_path / "submission_manifest.json"
    path.write_text(json.dumps({"artifact_class": "bundle"}), encoding="utf-8")

    assert load_submission_manifest(path) == {"artifact_class": "bundle"}


def test_load_submission_manifest_ignores_missing_invalid_or_non_object_payload(tmp_path: Path) -> None:
    assert load_submission_manifest(tmp_path / "missing.json") is None

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    assert load_submission_manifest(invalid) is None

    array_payload = tmp_path / "array.json"
    array_payload.write_text("[]", encoding="utf-8")
    assert load_submission_manifest(array_payload) is None


def test_resolve_manifest_references_resolves_relative_paths(tmp_path: Path) -> None:
    manifest = tmp_path / "submission_manifest.json"
    staging = tmp_path / "bundle"
    staging.mkdir()
    manifest.write_text(
        json.dumps(
            {
                "artifact_class": "bundle",
                "submission_path": "submission.csv",
                "staging_dir": "bundle",
                "members": ["bundle/a.csv", {"path": "bundle/b.csv"}],
            }
        ),
        encoding="utf-8",
    )

    artifact_class, submission_path, staging_dir, members = resolve_manifest_references(manifest)

    assert artifact_class == ARTIFACT_CLASS_BUNDLE
    assert submission_path == tmp_path / "submission.csv"
    assert staging_dir == staging
    assert members == [tmp_path / "bundle" / "a.csv", tmp_path / "bundle" / "b.csv"]


def test_normalize_artifact_class_defaults_unknown_values() -> None:
    assert normalize_artifact_class("multi file zip") == "multi_file_zip"
    assert normalize_artifact_class("unsupported") == ARTIFACT_CLASS_UNKNOWN
