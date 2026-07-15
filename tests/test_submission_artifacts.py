from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest

from kagglebot.submission_artifacts import (
    ARTIFACT_CLASS_BUNDLE,
    ARTIFACT_CLASS_MULTI_FILE_ZIP,
    ARTIFACT_CLASS_UNKNOWN,
    find_submission_manifest,
    load_submission_manifest,
    normalize_artifact_class,
    resolve_manifest_reference_details,
    resolve_manifest_references,
    store_submission_artifact,
    submission_specific_manifest_path,
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


def test_find_submission_manifest_prefers_newest_recursive_match(tmp_path: Path) -> None:
    old_manifest = tmp_path / "old" / "submission_manifest.json"
    new_manifest = tmp_path / "new" / "submission_manifest.json"
    old_manifest.parent.mkdir()
    new_manifest.parent.mkdir()
    old_manifest.write_text(json.dumps({"artifact_class": "bundle"}), encoding="utf-8")
    new_manifest.write_text(json.dumps({"artifact_class": "multi_file_zip"}), encoding="utf-8")
    old_time = 1_000_000
    new_time = 2_000_000
    os.utime(old_manifest, (old_time, old_time))
    os.utime(new_manifest, (new_time, new_time))

    assert find_submission_manifest(tmp_path) == new_manifest


def test_find_submission_manifest_skips_invalid_direct_manifest(tmp_path: Path) -> None:
    output_manifest = tmp_path / "output" / "submission_manifest.json"
    nested_manifest = tmp_path / "nested" / "submission_manifest.json"
    output_manifest.parent.mkdir()
    nested_manifest.parent.mkdir()
    output_manifest.write_text("{", encoding="utf-8")
    nested_manifest.write_text(json.dumps({"artifact_class": "bundle"}), encoding="utf-8")

    assert find_submission_manifest(tmp_path) == nested_manifest


def test_find_submission_manifest_ignores_generated_runtime_site(tmp_path: Path) -> None:
    expected = tmp_path / "results" / "submission_manifest.json"
    decoy = tmp_path / ".dependency_runtime_site" / "submission_manifest.json"
    expected.parent.mkdir()
    decoy.parent.mkdir()
    expected.write_text(json.dumps({"artifact_class": "bundle"}), encoding="utf-8")
    decoy.write_text(json.dumps({"artifact_class": "multi_file_zip"}), encoding="utf-8")
    os.utime(expected, (1000, 1000))
    os.utime(decoy, (2000, 2000))

    assert find_submission_manifest(tmp_path) == expected


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


def test_resolve_manifest_references_accepts_common_alias_keys(tmp_path: Path) -> None:
    manifest = tmp_path / "submission_manifest.json"
    staging = tmp_path / "bundle"
    staging.mkdir()
    manifest.write_text(
        json.dumps(
            {
                "artifact_class": "bundle",
                "artifact_path": "bundle.zip",
                "bundle_dir": "bundle",
                "files": ["bundle/a.csv", {"file": "bundle/b.csv"}, {"source": "bundle/c.csv"}],
            }
        ),
        encoding="utf-8",
    )

    artifact_class, submission_path, staging_dir, members = resolve_manifest_references(manifest)

    assert artifact_class == ARTIFACT_CLASS_BUNDLE
    assert submission_path == tmp_path / "bundle.zip"
    assert staging_dir == staging
    assert members == [
        tmp_path / "bundle" / "a.csv",
        tmp_path / "bundle" / "b.csv",
        tmp_path / "bundle" / "c.csv",
    ]


@pytest.mark.parametrize("key", ["output_file", "submission_file", "result_path", "prediction_file"])
def test_resolve_manifest_references_accepts_output_file_aliases(tmp_path: Path, key: str) -> None:
    manifest = tmp_path / "submission_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact_class": "tabular",
                key: "predictions.jsonl",
            }
        ),
        encoding="utf-8",
    )

    artifact_class, submission_path, staging_dir, members = resolve_manifest_references(manifest)

    assert artifact_class == "tabular"
    assert submission_path == tmp_path / "predictions.jsonl"
    assert staging_dir is None
    assert members == []


def test_resolve_manifest_references_accepts_nested_path_object_values(tmp_path: Path) -> None:
    manifest = tmp_path / "submission_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact_class": "single_file",
                "submission_path": {"sourcePath": "predictions.zarr"},
            }
        ),
        encoding="utf-8",
    )

    artifact_class, submission_path, staging_dir, members = resolve_manifest_references(manifest)

    assert artifact_class == "single_file"
    assert submission_path == tmp_path / "predictions.zarr"
    assert staging_dir is None
    assert members == []


def test_resolve_manifest_references_accepts_submission_payload_path_value(tmp_path: Path) -> None:
    manifest = tmp_path / "submission_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "submission": {
                    "artifactClass": "singleFile",
                    "path": "answers.nii.gz",
                },
            }
        ),
        encoding="utf-8",
    )

    artifact_class, submission_path, staging_dir, members = resolve_manifest_references(manifest)

    assert artifact_class == "single_file"
    assert submission_path == tmp_path / "answers.nii.gz"
    assert staging_dir is None
    assert members == []


def test_resolve_manifest_reference_details_preserves_requested_output_path(tmp_path: Path) -> None:
    manifest = tmp_path / "submission_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact_class": "tabular",
                "submission_path": "submission.csv",
                "requested_output_path": "answers.nii.gz",
            }
        ),
        encoding="utf-8",
    )

    details = resolve_manifest_reference_details(manifest)

    assert details.artifact_class == "tabular"
    assert details.submission_path == tmp_path / "submission.csv"
    assert details.requested_output_path == tmp_path / "answers.nii.gz"
    assert details.staging_dir is None
    assert details.members == []


def test_resolve_manifest_reference_details_accepts_requested_output_aliases(tmp_path: Path) -> None:
    manifest = tmp_path / "submission_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact_class": "tabular",
                "submission_path": "submission.csv",
                "expectedOutputFile": {"path": "predictions.zarr"},
            }
        ),
        encoding="utf-8",
    )

    details = resolve_manifest_reference_details(manifest)

    assert details.submission_path == tmp_path / "submission.csv"
    assert details.requested_output_path == tmp_path / "predictions.zarr"


def test_resolve_manifest_references_skips_metadata_only_path_objects(tmp_path: Path) -> None:
    manifest = tmp_path / "submission_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "submission": {"artifactClass": "singleFile"},
                "artifactPath": "answers.nii.gz",
            }
        ),
        encoding="utf-8",
    )

    artifact_class, submission_path, staging_dir, members = resolve_manifest_references(manifest)

    assert artifact_class == "single_file"
    assert submission_path == tmp_path / "answers.nii.gz"
    assert staging_dir is None
    assert members == []


def test_resolve_manifest_references_accepts_folder_path_alias_for_staging(tmp_path: Path) -> None:
    manifest = tmp_path / "submission_manifest.json"
    staging = tmp_path / "bundle"
    staging.mkdir()
    manifest.write_text(
        json.dumps(
            {
                "artifactClass": "bundle",
                "folderPath": {"path": "bundle"},
            }
        ),
        encoding="utf-8",
    )

    artifact_class, submission_path, staging_dir, members = resolve_manifest_references(manifest)

    assert artifact_class == ARTIFACT_CLASS_BUNDLE
    assert submission_path is None
    assert staging_dir == staging
    assert members == []


def test_resolve_manifest_references_rejects_staging_path_traversal(tmp_path: Path) -> None:
    manifest = tmp_path / "output" / "submission_manifest.json"
    manifest.parent.mkdir()
    outside = tmp_path / "outside_bundle"
    outside.mkdir()
    manifest.write_text(
        json.dumps(
            {
                "artifactClass": "bundle",
                "stagingDir": "../outside_bundle",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsafe path traversal in manifest staging path"):
        resolve_manifest_reference_details(manifest)


def test_resolve_manifest_references_rejects_absolute_staging_path(tmp_path: Path) -> None:
    manifest = tmp_path / "submission_manifest.json"
    outside = tmp_path / "outside_bundle"
    outside.mkdir()
    manifest.write_text(
        json.dumps(
            {
                "artifactClass": "bundle",
                "stagingDir": str(outside),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsafe absolute manifest staging path"):
        resolve_manifest_reference_details(manifest)


@pytest.mark.parametrize("suffix", [".tar.gz", ".tgz", ".tar.xz", ".tar.zst", ".7z", ".rar"])
def test_resolve_manifest_references_infers_archive_artifact_class(tmp_path: Path, suffix: str) -> None:
    manifest = tmp_path / "submission_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact_path": f"submission{suffix}",
            }
        ),
        encoding="utf-8",
    )

    artifact_class, submission_path, staging_dir, members = resolve_manifest_references(manifest)

    assert artifact_class == ARTIFACT_CLASS_MULTI_FILE_ZIP
    assert submission_path == tmp_path / f"submission{suffix}"
    assert staging_dir is None
    assert members == []


def test_resolve_manifest_references_accepts_camel_case_alias_keys(tmp_path: Path) -> None:
    manifest = tmp_path / "submission_manifest.json"
    staging = tmp_path / "bundle"
    staging.mkdir()
    manifest.write_text(
        json.dumps(
            {
                "artifactClass": "bundle",
                "artifactPath": "bundle.zip",
                "stagingDir": "bundle",
                "filePaths": ["bundle/a.csv", {"artifactPath": "bundle/b.csv"}],
            }
        ),
        encoding="utf-8",
    )

    artifact_class, submission_path, staging_dir, members = resolve_manifest_references(manifest)

    assert artifact_class == ARTIFACT_CLASS_BUNDLE
    assert submission_path == tmp_path / "bundle.zip"
    assert staging_dir == staging
    assert members == [tmp_path / "bundle" / "a.csv", tmp_path / "bundle" / "b.csv"]


def test_resolve_manifest_references_accepts_camel_case_artifact_class_values(tmp_path: Path) -> None:
    manifest = tmp_path / "submission_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifactClass": "multiFileZip",
                "artifactPath": "bundle.zip",
            }
        ),
        encoding="utf-8",
    )

    artifact_class, submission_path, staging_dir, members = resolve_manifest_references(manifest)

    assert artifact_class == "multi_file_zip"
    assert submission_path == tmp_path / "bundle.zip"
    assert staging_dir is None
    assert members == []


def test_resolve_manifest_references_accepts_nested_submission_and_bundle_payloads(tmp_path: Path) -> None:
    manifest = tmp_path / "submission_manifest.json"
    staging = tmp_path / "bundle"
    staging.mkdir()
    manifest.write_text(
        json.dumps(
            {
                "artifactClass": "multiFileZip",
                "submission": {"artifactPath": "bundle.zip"},
                "bundle": {
                    "stagingDir": "bundle",
                    "filePaths": ["bundle/a.csv", {"source": "bundle/b.csv"}],
                },
            }
        ),
        encoding="utf-8",
    )

    artifact_class, submission_path, staging_dir, members = resolve_manifest_references(manifest)

    assert artifact_class == "multi_file_zip"
    assert submission_path == tmp_path / "bundle.zip"
    assert staging_dir == staging
    assert members == [tmp_path / "bundle" / "a.csv", tmp_path / "bundle" / "b.csv"]


def test_resolve_manifest_references_infers_bundle_class_from_staging_and_members(tmp_path: Path) -> None:
    manifest = tmp_path / "submission_manifest.json"
    staging = tmp_path / "bundle"
    staging.mkdir()
    manifest.write_text(
        json.dumps(
            {
                "stagingDir": "bundle",
                "filePaths": ["bundle/a.csv"],
            }
        ),
        encoding="utf-8",
    )

    artifact_class, submission_path, staging_dir, members = resolve_manifest_references(manifest)

    assert artifact_class == ARTIFACT_CLASS_BUNDLE
    assert submission_path is None
    assert staging_dir == staging
    assert members == [tmp_path / "bundle" / "a.csv"]


def test_resolve_manifest_references_accepts_member_source_path_aliases(tmp_path: Path) -> None:
    manifest = tmp_path / "submission_manifest.json"
    staging = tmp_path / "bundle"
    staging.mkdir()
    manifest.write_text(
        json.dumps(
            {
                "bundleDir": "bundle",
                "entries": [
                    {"sourcePath": "bundle/a.csv"},
                    {"localPath": "bundle/b.csv"},
                    {"relativePath": "bundle/c.csv"},
                ],
            }
        ),
        encoding="utf-8",
    )

    artifact_class, submission_path, staging_dir, members = resolve_manifest_references(manifest)

    assert artifact_class == ARTIFACT_CLASS_BUNDLE
    assert submission_path is None
    assert staging_dir == staging
    assert members == [
        tmp_path / "bundle" / "a.csv",
        tmp_path / "bundle" / "b.csv",
        tmp_path / "bundle" / "c.csv",
    ]


def test_resolve_manifest_references_accepts_dict_member_values(tmp_path: Path) -> None:
    manifest = tmp_path / "submission_manifest.json"
    staging = tmp_path / "bundle"
    staging.mkdir()
    manifest.write_text(
        json.dumps(
            {
                "bundleDir": "bundle",
                "files": {
                    "a.tif": "bundle/a.tif",
                    "b.tif": {"sourcePath": "bundle/b.tif"},
                },
            }
        ),
        encoding="utf-8",
    )

    artifact_class, submission_path, staging_dir, members = resolve_manifest_references(manifest)

    assert artifact_class == ARTIFACT_CLASS_BUNDLE
    assert submission_path is None
    assert staging_dir == staging
    assert members == [tmp_path / "bundle" / "a.tif", tmp_path / "bundle" / "b.tif"]


def test_resolve_manifest_references_accepts_source_to_archive_member_mapping(tmp_path: Path) -> None:
    manifest = tmp_path / "submission_manifest.json"
    staging = tmp_path / "bundle"
    staging.mkdir()
    (staging / "a.tif").write_bytes(b"a")
    manifest.write_text(
        json.dumps(
            {
                "bundleDir": "bundle",
                "files": {
                    "bundle/a.tif": "masks/a.tif",
                },
            }
        ),
        encoding="utf-8",
    )

    details = resolve_manifest_reference_details(manifest)

    assert details.members == [tmp_path / "bundle" / "a.tif"]
    assert [member.archive_path for member in details.member_specs] == ["masks/a.tif"]


def test_resolve_manifest_references_accepts_single_member_object(tmp_path: Path) -> None:
    manifest = tmp_path / "submission_manifest.json"
    staging = tmp_path / "bundle"
    staging.mkdir()
    (staging / "a.tif").write_bytes(b"a")
    manifest.write_text(
        json.dumps(
            {
                "bundleDir": "bundle",
                "files": {
                    "sourcePath": "bundle/a.tif",
                    "targetPath": "masks/a.tif",
                },
            }
        ),
        encoding="utf-8",
    )

    details = resolve_manifest_reference_details(manifest)

    assert details.artifact_class == ARTIFACT_CLASS_BUNDLE
    assert details.staging_dir == staging
    assert details.members == [tmp_path / "bundle" / "a.tif"]
    assert [member.archive_path for member in details.member_specs] == ["masks/a.tif"]


def test_resolve_manifest_references_accepts_member_path_object_values(tmp_path: Path) -> None:
    manifest = tmp_path / "submission_manifest.json"
    staging = tmp_path / "bundle"
    staging.mkdir()
    (staging / "a.tif").write_bytes(b"a")
    manifest.write_text(
        json.dumps(
            {
                "bundleDir": "bundle",
                "files": [
                    {
                        "sourcePath": {"path": "bundle/a.tif"},
                        "targetPath": {"path": "masks/a.tif"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    details = resolve_manifest_reference_details(manifest)

    assert details.members == [tmp_path / "bundle" / "a.tif"]
    assert [member.archive_path for member in details.member_specs] == ["masks/a.tif"]


def test_resolve_manifest_references_expands_member_globs(tmp_path: Path) -> None:
    manifest = tmp_path / "submission_manifest.json"
    staging = tmp_path / "bundle"
    staging.mkdir()
    (staging / "a.tif").write_bytes(b"a")
    (staging / "b.tif").write_bytes(b"b")
    (staging / "notes.txt").write_text("ignore\n", encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "bundleDir": "bundle",
                "files": ["bundle/*.tif"],
            }
        ),
        encoding="utf-8",
    )

    artifact_class, submission_path, staging_dir, members = resolve_manifest_references(manifest)

    assert artifact_class == ARTIFACT_CLASS_BUNDLE
    assert submission_path is None
    assert staging_dir == staging
    assert members == [tmp_path / "bundle" / "a.tif", tmp_path / "bundle" / "b.tif"]


def test_resolve_manifest_references_accepts_single_string_member_glob(tmp_path: Path) -> None:
    manifest = tmp_path / "submission_manifest.json"
    staging = tmp_path / "bundle"
    staging.mkdir()
    (staging / "a.tif").write_bytes(b"a")
    (staging / "b.tif").write_bytes(b"b")
    (staging / "notes.txt").write_text("ignore\n", encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "bundleDir": "bundle",
                "files": "bundle/*.tif",
            }
        ),
        encoding="utf-8",
    )

    artifact_class, submission_path, staging_dir, members = resolve_manifest_references(manifest)

    assert artifact_class == ARTIFACT_CLASS_BUNDLE
    assert submission_path is None
    assert staging_dir == staging
    assert members == [tmp_path / "bundle" / "a.tif", tmp_path / "bundle" / "b.tif"]


def test_resolve_manifest_references_expands_member_directories(tmp_path: Path) -> None:
    manifest = tmp_path / "submission_manifest.json"
    staging = tmp_path / "bundle"
    nested = staging / "nested"
    nested.mkdir(parents=True)
    empty = nested / "empty_group"
    empty.mkdir()
    (staging / "a.tif").write_bytes(b"a")
    (nested / "b.tif").write_bytes(b"b")
    manifest.write_text(
        json.dumps(
            {
                "bundleDir": "bundle",
                "files": ["bundle"],
            }
        ),
        encoding="utf-8",
    )

    artifact_class, submission_path, staging_dir, members = resolve_manifest_references(manifest)

    assert artifact_class == ARTIFACT_CLASS_BUNDLE
    assert submission_path is None
    assert staging_dir == staging
    assert members == [
        tmp_path / "bundle" / "a.tif",
        tmp_path / "bundle" / "nested",
        tmp_path / "bundle" / "nested" / "b.tif",
        tmp_path / "bundle" / "nested" / "empty_group",
    ]


def test_resolve_manifest_references_deduplicates_expanded_members(tmp_path: Path) -> None:
    manifest = tmp_path / "submission_manifest.json"
    staging = tmp_path / "bundle"
    staging.mkdir()
    (staging / "a.tif").write_bytes(b"a")
    (staging / "b.tif").write_bytes(b"b")
    manifest.write_text(
        json.dumps(
            {
                "bundleDir": "bundle",
                "files": ["bundle/*.tif", "bundle/a.tif"],
            }
        ),
        encoding="utf-8",
    )

    _, _, _, members = resolve_manifest_references(manifest)

    assert members == [tmp_path / "bundle" / "a.tif", tmp_path / "bundle" / "b.tif"]


def test_resolve_manifest_reference_details_preserves_archive_paths(tmp_path: Path) -> None:
    manifest = tmp_path / "submission_manifest.json"
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "a.tif").write_bytes(b"a")
    (bundle / "b.tif").write_bytes(b"b")
    manifest.write_text(
        json.dumps(
            {
                "bundleDir": "bundle",
                "files": {
                    "masks/a.tif": "bundle/a.tif",
                    "nested/b.tif": {"sourcePath": "bundle/b.tif"},
                },
            }
        ),
        encoding="utf-8",
    )

    details = resolve_manifest_reference_details(manifest)

    assert details.members == [tmp_path / "bundle" / "a.tif", tmp_path / "bundle" / "b.tif"]
    assert [member.archive_path for member in details.member_specs] == ["masks/a.tif", "nested/b.tif"]


def test_resolve_manifest_reference_details_keeps_same_source_with_distinct_archive_paths(tmp_path: Path) -> None:
    manifest = tmp_path / "submission_manifest.json"
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "a.tif").write_bytes(b"a")
    manifest.write_text(
        json.dumps(
            {
                "bundleDir": "bundle",
                "files": [
                    {"sourcePath": "bundle/a.tif", "targetPath": "masks/a.tif"},
                    {"sourcePath": "bundle/a.tif", "targetPath": "backup/a.tif"},
                ],
            }
        ),
        encoding="utf-8",
    )

    details = resolve_manifest_reference_details(manifest)

    assert details.members == [tmp_path / "bundle" / "a.tif", tmp_path / "bundle" / "a.tif"]
    assert [member.archive_path for member in details.member_specs] == ["masks/a.tif", "backup/a.tif"]


def test_resolve_manifest_reference_details_applies_archive_directory_to_globs(tmp_path: Path) -> None:
    manifest = tmp_path / "submission_manifest.json"
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "a.tif").write_bytes(b"a")
    (bundle / "b.tif").write_bytes(b"b")
    manifest.write_text(
        json.dumps(
            {
                "bundleDir": "bundle",
                "files": [
                    {
                        "sourcePath": "bundle/*.tif",
                        "targetPath": "masks",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    details = resolve_manifest_reference_details(manifest)

    assert details.members == [tmp_path / "bundle" / "a.tif", tmp_path / "bundle" / "b.tif"]
    assert [member.archive_path for member in details.member_specs] == ["masks/a.tif", "masks/b.tif"]


def test_resolve_manifest_reference_details_preserves_recursive_glob_layout_in_archive_dir(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "submission_manifest.json"
    bundle = tmp_path / "bundle"
    left = bundle / "fold1" / "mask.tif"
    right = bundle / "fold2" / "mask.tif"
    left.parent.mkdir(parents=True)
    right.parent.mkdir(parents=True)
    left.write_bytes(b"left")
    right.write_bytes(b"right")
    manifest.write_text(
        json.dumps(
            {
                "bundleDir": "bundle",
                "files": [
                    {
                        "sourcePath": "bundle/**/*.tif",
                        "targetPath": "masks/",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    details = resolve_manifest_reference_details(manifest)

    assert details.members == [
        tmp_path / "bundle" / "fold1" / "mask.tif",
        tmp_path / "bundle" / "fold2" / "mask.tif",
    ]
    assert [member.archive_path for member in details.member_specs] == [
        "masks/fold1/mask.tif",
        "masks/fold2/mask.tif",
    ]


def test_resolve_manifest_reference_details_rejects_archive_path_traversal(tmp_path: Path) -> None:
    manifest = tmp_path / "submission_manifest.json"
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "a.tif").write_bytes(b"a")
    manifest.write_text(
        json.dumps(
            {
                "bundleDir": "bundle",
                "files": [
                    {"sourcePath": "bundle/a.tif", "targetPath": "../evil.tif"},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsafe path traversal"):
        resolve_manifest_reference_details(manifest)


def test_resolve_manifest_reference_details_rejects_source_path_traversal(tmp_path: Path) -> None:
    manifest = tmp_path / "submission_manifest.json"
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    secret = tmp_path / "secret.tif"
    secret.write_bytes(b"secret")
    manifest.write_text(
        json.dumps(
            {
                "bundleDir": "bundle",
                "files": [
                    {"sourcePath": "../secret.tif", "targetPath": "secret.tif"},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsafe path traversal in manifest source path"):
        resolve_manifest_reference_details(manifest)


def test_resolve_manifest_reference_details_rejects_absolute_archive_path(tmp_path: Path) -> None:
    manifest = tmp_path / "submission_manifest.json"
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "a.tif").write_bytes(b"a")
    manifest.write_text(
        json.dumps(
            {
                "bundleDir": "bundle",
                "files": [
                    {"sourcePath": "bundle/a.tif", "targetPath": "/tmp/evil.tif"},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsafe manifest archive path"):
        resolve_manifest_reference_details(manifest)


def test_resolve_manifest_reference_details_rejects_absolute_source_path(tmp_path: Path) -> None:
    manifest = tmp_path / "submission_manifest.json"
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    source = tmp_path / "outside.tif"
    source.write_bytes(b"outside")
    manifest.write_text(
        json.dumps(
            {
                "bundleDir": "bundle",
                "files": [
                    {"sourcePath": str(source), "targetPath": "outside.tif"},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsafe absolute manifest source path"):
        resolve_manifest_reference_details(manifest)


def test_normalize_artifact_class_defaults_unknown_values() -> None:
    assert normalize_artifact_class("multi file zip") == "multi_file_zip"
    assert normalize_artifact_class("singleFile") == "single_file"
    assert normalize_artifact_class("notebookOutput") == "notebook_output"
    assert normalize_artifact_class("unsupported") == ARTIFACT_CLASS_UNKNOWN


def test_store_submission_artifact_copies_with_run_id_prefix(tmp_path: Path) -> None:
    source = tmp_path / "output" / "submission.csv"
    source.parent.mkdir(exist_ok=True)
    source.write_text("id,target\n1,0.1\n", encoding="utf-8")

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    assert stored == tmp_path / "submissions" / "run-123_submission.csv"
    assert stored.read_text(encoding="utf-8") == "id,target\n1,0.1\n"


def test_store_submission_artifact_preserves_non_csv_suffix(tmp_path: Path) -> None:
    source = tmp_path / "output" / "submission.jsonl"
    source.parent.mkdir(exist_ok=True)
    source.write_text('{"id": 1, "target": 0.1}\n', encoding="utf-8")

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    assert stored == tmp_path / "submissions" / "run-123_submission.jsonl"
    assert stored.read_text(encoding="utf-8") == '{"id": 1, "target": 0.1}\n'


@pytest.mark.parametrize("suffix", [".db", ".sqlite", ".sqlite3"])
def test_store_submission_artifact_preserves_sqlite_suffix(tmp_path: Path, suffix: str) -> None:
    source = tmp_path / "output" / f"submission{suffix}"
    source.parent.mkdir(exist_ok=True)
    with sqlite3.connect(source) as conn:
        conn.execute("CREATE TABLE predictions (id INTEGER, target REAL)")
        conn.execute("INSERT INTO predictions VALUES (?, ?)", (1, 0.5))

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    assert stored == tmp_path / "submissions" / f"run-123_submission{suffix}"
    with sqlite3.connect(stored) as conn:
        row = conn.execute("SELECT id, target FROM predictions").fetchone()
    assert row == (1, 0.5)


def test_store_submission_artifact_preserves_compound_suffix(tmp_path: Path) -> None:
    source = tmp_path / "output" / "submission.tar.gz"
    source.parent.mkdir(exist_ok=True)
    source.write_bytes(b"archive")

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    assert stored == tmp_path / "submissions" / "run-123_submission.tar.gz"
    assert stored.read_bytes() == b"archive"


def test_store_submission_artifact_preserves_shapefile_sidecars_with_run_id_prefix(tmp_path: Path) -> None:
    source = tmp_path / "output" / "submission.shp"
    source.parent.mkdir(exist_ok=True)
    source.write_bytes(b"shape")
    (source.parent / "submission.dbf").write_bytes(b"attributes")
    (source.parent / "submission.shx").write_bytes(b"index")
    (source.parent / "submission.prj").write_text("EPSG:4326\n", encoding="utf-8")
    (source.parent / "submission.qix").write_bytes(b"qix")
    (source.parent / "submission.shp.aux.xml").write_text("<PAMDataset />\n", encoding="utf-8")

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    assert stored == tmp_path / "submissions" / "run-123_submission.shp"
    assert stored.read_bytes() == b"shape"
    assert (tmp_path / "submissions" / "run-123_submission.dbf").read_bytes() == b"attributes"
    assert (tmp_path / "submissions" / "run-123_submission.shx").read_bytes() == b"index"
    assert (tmp_path / "submissions" / "run-123_submission.prj").read_text(encoding="utf-8") == "EPSG:4326\n"
    assert (tmp_path / "submissions" / "run-123_submission.qix").read_bytes() == b"qix"
    assert (tmp_path / "submissions" / "run-123_submission.shp.aux.xml").read_text(encoding="utf-8") == (
        "<PAMDataset />\n"
    )


def test_store_submission_artifact_preserves_mapinfo_sidecars_with_run_id_prefix(tmp_path: Path) -> None:
    source = tmp_path / "output" / "submission.tab"
    source.parent.mkdir(exist_ok=True)
    source.write_text("!table\n!version 300\n", encoding="utf-8")
    (source.parent / "submission.dat").write_bytes(b"data")
    (source.parent / "submission.id").write_bytes(b"ids")
    (source.parent / "submission.map").write_bytes(b"map")
    (source.parent / "submission.ind").write_bytes(b"index")

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    assert stored == tmp_path / "submissions" / "run-123_submission.tab"
    assert stored.read_text(encoding="utf-8") == "!table\n!version 300\n"
    assert (tmp_path / "submissions" / "run-123_submission.dat").read_bytes() == b"data"
    assert (tmp_path / "submissions" / "run-123_submission.id").read_bytes() == b"ids"
    assert (tmp_path / "submissions" / "run-123_submission.map").read_bytes() == b"map"
    assert (tmp_path / "submissions" / "run-123_submission.ind").read_bytes() == b"index"


def test_store_submission_artifact_preserves_mapinfo_interchange_sidecar_with_run_id_prefix(tmp_path: Path) -> None:
    source = tmp_path / "output" / "submission.mif"
    source.parent.mkdir(exist_ok=True)
    source.write_text("Version 300\nColumns 1\n  Name Char(20)\nData\n", encoding="utf-8")
    (source.parent / "submission.mid").write_text('"parcel-a"\n', encoding="utf-8")

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    assert stored == tmp_path / "submissions" / "run-123_submission.mif"
    assert stored.read_text(encoding="utf-8").startswith("Version 300")
    assert (tmp_path / "submissions" / "run-123_submission.mid").read_text(encoding="utf-8") == '"parcel-a"\n'


def test_store_submission_artifact_preserves_georeferenced_raster_sidecars_with_run_id_prefix(
    tmp_path: Path,
) -> None:
    source = tmp_path / "output" / "submission.tif"
    source.parent.mkdir(exist_ok=True)
    source.write_bytes(b"raster")
    (source.parent / "submission.tfw").write_text("1\n0\n0\n-1\n100\n200\n", encoding="ascii")
    (source.parent / "submission.tif.aux.xml").write_text("<PAMDataset />\n", encoding="utf-8")

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    assert stored == tmp_path / "submissions" / "run-123_submission.tif"
    assert stored.read_bytes() == b"raster"
    assert (tmp_path / "submissions" / "run-123_submission.tfw").read_text(encoding="ascii") == (
        "1\n0\n0\n-1\n100\n200\n"
    )
    assert (tmp_path / "submissions" / "run-123_submission.tif.aux.xml").read_text(encoding="utf-8") == (
        "<PAMDataset />\n"
    )


def test_store_submission_artifact_preserves_vrt_sidecars(tmp_path: Path) -> None:
    source = tmp_path / "output" / "submission.vrt"
    (source.parent / "rasters").mkdir(parents=True)
    source.write_text(
        """
        <VRTDataset rasterXSize="2" rasterYSize="2">
          <VRTRasterBand dataType="Byte" band="1">
            <SimpleSource><SourceFilename relativeToVRT="1">rasters/source.tif</SourceFilename></SimpleSource>
          </VRTRasterBand>
        </VRTDataset>
        """,
        encoding="utf-8",
    )
    (source.parent / "rasters" / "source.tif").write_bytes(b"raster")

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    assert stored == tmp_path / "submissions" / "run-123_submission.vrt"
    assert "rasters/source.tif" in stored.read_text(encoding="utf-8")
    assert (tmp_path / "submissions" / "rasters" / "source.tif").read_bytes() == b"raster"


def test_store_submission_artifact_preserves_metaimage_sidecars(tmp_path: Path) -> None:
    source = tmp_path / "output" / "submission.mhd"
    (source.parent / "raw").mkdir(parents=True)
    source.write_text("ObjectType = Image\nElementDataFile = raw/volume.raw\n", encoding="utf-8")
    (source.parent / "raw" / "volume.raw").write_bytes(b"voxels")

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    assert stored == tmp_path / "submissions" / "run-123_submission.mhd"
    assert stored.read_text(encoding="utf-8") == "ObjectType = Image\nElementDataFile = raw/volume.raw\n"
    assert (tmp_path / "submissions" / "raw" / "volume.raw").read_bytes() == b"voxels"


def test_store_submission_artifact_preserves_detached_nrrd_sidecars(tmp_path: Path) -> None:
    source = tmp_path / "output" / "submission.nhdr"
    (source.parent / "raw").mkdir(parents=True)
    source.write_text("NRRD0005\nsizes: 4 5 6\ndata file: raw/volume.raw\n", encoding="utf-8")
    (source.parent / "raw" / "volume.raw").write_bytes(b"voxels")

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    assert stored == tmp_path / "submissions" / "run-123_submission.nhdr"
    assert stored.read_text(encoding="utf-8") == "NRRD0005\nsizes: 4 5 6\ndata file: raw/volume.raw\n"
    assert (tmp_path / "submissions" / "raw" / "volume.raw").read_bytes() == b"voxels"


def test_store_submission_artifact_preserves_analyze_pair_sidecars_with_run_id_prefix(tmp_path: Path) -> None:
    source = tmp_path / "output" / "submission.hdr"
    source.parent.mkdir(exist_ok=True)
    source.write_bytes(b"header")
    (source.parent / "submission.img").write_bytes(b"volume")

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    assert stored == tmp_path / "submissions" / "run-123_submission.hdr"
    assert stored.read_bytes() == b"header"
    assert (tmp_path / "submissions" / "run-123_submission.img").read_bytes() == b"volume"


def test_store_submission_artifact_preserves_kml_sidecars(tmp_path: Path) -> None:
    source = tmp_path / "output" / "submission.kml"
    (source.parent / "icons").mkdir(parents=True)
    source.write_text(
        """
        <kml xmlns="http://www.opengis.net/kml/2.2">
          <Document><Icon><href>icons/pin.png</href></Icon></Document>
        </kml>
        """,
        encoding="utf-8",
    )
    (source.parent / "icons" / "pin.png").write_bytes(b"pin")

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    assert stored == tmp_path / "submissions" / "run-123_submission.kml"
    assert "<href>icons/pin.png</href>" in stored.read_text(encoding="utf-8")
    assert (tmp_path / "submissions" / "icons" / "pin.png").read_bytes() == b"pin"


def test_store_submission_artifact_preserves_envi_header_sidecars_with_run_id_prefix(tmp_path: Path) -> None:
    source = tmp_path / "output" / "submission.hdr"
    source.parent.mkdir(exist_ok=True)
    source.write_text("ENVI\nsamples = 2\nlines = 2\nbands = 1\n", encoding="utf-8")
    (source.parent / "submission.dat").write_bytes(b"raster")

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    assert stored == tmp_path / "submissions" / "run-123_submission.hdr"
    assert stored.read_text(encoding="utf-8").startswith("ENVI")
    assert (tmp_path / "submissions" / "run-123_submission.dat").read_bytes() == b"raster"


def test_store_submission_artifact_preserves_nested_kml_layout(tmp_path: Path) -> None:
    source = tmp_path / "output" / "layers" / "submission.kml"
    source.parent.mkdir(parents=True)
    (source.parent.parent / "icons").mkdir()
    source.write_text(
        """
        <kml xmlns="http://www.opengis.net/kml/2.2">
          <Document><Icon><href>../icons/pin.png</href></Icon></Document>
        </kml>
        """,
        encoding="utf-8",
    )
    (source.parent.parent / "icons" / "pin.png").write_bytes(b"pin")

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    assert stored == tmp_path / "submissions" / "run-123_submission_bundle" / "layers" / "submission.kml"
    assert (tmp_path / "submissions" / "run-123_submission_bundle" / "icons" / "pin.png").read_bytes() == b"pin"


def test_store_submission_artifact_preserves_obj_sidecars(tmp_path: Path) -> None:
    source = tmp_path / "output" / "submission.obj"
    (source.parent / "materials" / "textures").mkdir(parents=True)
    source.write_text("mtllib materials/model.mtl\nv 0 0 0\n", encoding="utf-8")
    (source.parent / "materials" / "model.mtl").write_text(
        "newmtl surface\nmap_Kd textures/diffuse.png\n",
        encoding="utf-8",
    )
    (source.parent / "materials" / "textures" / "diffuse.png").write_bytes(b"texture")

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    assert stored == tmp_path / "submissions" / "run-123_submission.obj"
    assert stored.read_text(encoding="utf-8") == "mtllib materials/model.mtl\nv 0 0 0\n"
    assert (tmp_path / "submissions" / "materials" / "model.mtl").read_text(encoding="utf-8") == (
        "newmtl surface\nmap_Kd textures/diffuse.png\n"
    )
    assert (tmp_path / "submissions" / "materials" / "textures" / "diffuse.png").read_bytes() == b"texture"


def test_store_submission_artifact_preserves_ply_sidecars(tmp_path: Path) -> None:
    source = tmp_path / "output" / "submission.ply"
    (source.parent / "textures").mkdir(parents=True)
    source.write_text(
        "ply\nformat ascii 1.0\nobj_info TextureFile textures/diffuse.png\nelement vertex 0\nend_header\n",
        encoding="ascii",
    )
    (source.parent / "textures" / "diffuse.png").write_bytes(b"texture")

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    assert stored == tmp_path / "submissions" / "run-123_submission.ply"
    assert "TextureFile textures/diffuse.png" in stored.read_text(encoding="ascii")
    assert (tmp_path / "submissions" / "textures" / "diffuse.png").read_bytes() == b"texture"


def test_store_submission_artifact_preserves_dae_sidecars(tmp_path: Path) -> None:
    source = tmp_path / "output" / "submission.dae"
    (source.parent / "textures").mkdir(parents=True)
    source.write_text(
        """
        <COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema">
          <library_images>
            <image id="diffuse"><init_from>textures/diffuse.png</init_from></image>
          </library_images>
        </COLLADA>
        """,
        encoding="utf-8",
    )
    (source.parent / "textures" / "diffuse.png").write_bytes(b"texture")

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    assert stored == tmp_path / "submissions" / "run-123_submission.dae"
    assert "<init_from>textures/diffuse.png</init_from>" in stored.read_text(encoding="utf-8")
    assert (tmp_path / "submissions" / "textures" / "diffuse.png").read_bytes() == b"texture"


def test_store_submission_artifact_preserves_nested_dae_layout(tmp_path: Path) -> None:
    source = tmp_path / "output" / "meshes" / "submission.dae"
    source.parent.mkdir(parents=True)
    (source.parent.parent / "textures").mkdir()
    source.write_text(
        """
        <COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema">
          <library_images>
            <image id="diffuse"><init_from>../textures/diffuse.png</init_from></image>
          </library_images>
        </COLLADA>
        """,
        encoding="utf-8",
    )
    (source.parent.parent / "textures" / "diffuse.png").write_bytes(b"texture")

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    assert stored == tmp_path / "submissions" / "run-123_submission_bundle" / "meshes" / "submission.dae"
    assert (tmp_path / "submissions" / "run-123_submission_bundle" / "textures" / "diffuse.png").read_bytes() == (
        b"texture"
    )


def test_store_submission_artifact_preserves_x3d_sidecars(tmp_path: Path) -> None:
    source = tmp_path / "output" / "submission.x3d"
    (source.parent / "textures").mkdir(parents=True)
    source.write_text(
        """
        <X3D>
          <Scene><ImageTexture url='"textures/diffuse.png"'/></Scene>
        </X3D>
        """,
        encoding="utf-8",
    )
    (source.parent / "textures" / "diffuse.png").write_bytes(b"texture")

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    assert stored == tmp_path / "submissions" / "run-123_submission.x3d"
    assert "textures/diffuse.png" in stored.read_text(encoding="utf-8")
    assert (tmp_path / "submissions" / "textures" / "diffuse.png").read_bytes() == b"texture"


def test_store_submission_artifact_preserves_nested_x3d_layout(tmp_path: Path) -> None:
    source = tmp_path / "output" / "scenes" / "submission.x3d"
    source.parent.mkdir(parents=True)
    (source.parent.parent / "textures").mkdir()
    source.write_text(
        """
        <X3D>
          <Scene><ImageTexture url='"../textures/diffuse.png"'/></Scene>
        </X3D>
        """,
        encoding="utf-8",
    )
    (source.parent.parent / "textures" / "diffuse.png").write_bytes(b"texture")

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    assert stored == tmp_path / "submissions" / "run-123_submission_bundle" / "scenes" / "submission.x3d"
    assert (tmp_path / "submissions" / "run-123_submission_bundle" / "textures" / "diffuse.png").read_bytes() == (
        b"texture"
    )


def test_store_submission_artifact_preserves_gltf_sidecars(tmp_path: Path) -> None:
    source = tmp_path / "output" / "submission.gltf"
    (source.parent / "buffers").mkdir(parents=True)
    (source.parent / "textures").mkdir()
    source.write_text(
        json.dumps(
            {
                "asset": {"version": "2.0"},
                "buffers": [{"uri": "buffers/scene.bin"}],
                "images": [{"uri": "textures/diffuse.png"}],
            }
        ),
        encoding="utf-8",
    )
    (source.parent / "buffers" / "scene.bin").write_bytes(b"buffer")
    (source.parent / "textures" / "diffuse.png").write_bytes(b"texture")

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    assert stored == tmp_path / "submissions" / "run-123_submission.gltf"
    assert json.loads(stored.read_text(encoding="utf-8"))["asset"] == {"version": "2.0"}
    assert (tmp_path / "submissions" / "buffers" / "scene.bin").read_bytes() == b"buffer"
    assert (tmp_path / "submissions" / "textures" / "diffuse.png").read_bytes() == b"texture"


def test_store_submission_artifact_preserves_nested_gltf_layout(tmp_path: Path) -> None:
    source = tmp_path / "output" / "scenes" / "submission.gltf"
    source.parent.mkdir(parents=True)
    (source.parent.parent / "textures").mkdir()
    source.write_text(
        json.dumps({"asset": {"version": "2.0"}, "images": [{"uri": "../textures/diffuse.png"}]}),
        encoding="utf-8",
    )
    (source.parent.parent / "textures" / "diffuse.png").write_bytes(b"texture")

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    assert stored == tmp_path / "submissions" / "run-123_submission_bundle" / "scenes" / "submission.gltf"
    assert (tmp_path / "submissions" / "run-123_submission_bundle" / "textures" / "diffuse.png").read_bytes() == (
        b"texture"
    )


def test_store_submission_artifact_preserves_las_sidecars_with_run_id_prefix(tmp_path: Path) -> None:
    source = tmp_path / "output" / "submission.las"
    source.parent.mkdir(exist_ok=True)
    source.write_bytes(b"las")
    (source.parent / "submission.prj").write_text("EPSG:4326\n", encoding="utf-8")
    (source.parent / "submission.lax").write_bytes(b"index")
    (source.parent / "submission.las.aux.xml").write_text("<PAMDataset />\n", encoding="utf-8")

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    assert stored == tmp_path / "submissions" / "run-123_submission.las"
    assert stored.read_bytes() == b"las"
    assert (tmp_path / "submissions" / "run-123_submission.prj").read_text(encoding="utf-8") == "EPSG:4326\n"
    assert (tmp_path / "submissions" / "run-123_submission.lax").read_bytes() == b"index"
    assert (tmp_path / "submissions" / "run-123_submission.las.aux.xml").read_text(encoding="utf-8") == (
        "<PAMDataset />\n"
    )


def test_store_submission_artifact_preserves_model_index_shards(tmp_path: Path) -> None:
    source = tmp_path / "output" / "model.safetensors.index.json"
    source.parent.mkdir(exist_ok=True)
    source.write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer.weight": "model-00001-of-00002.safetensors",
                    "layer.bias": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    (source.parent / "model-00001-of-00002.safetensors").write_bytes(b"shard-1")
    (source.parent / "model-00002-of-00002.safetensors").write_bytes(b"shard-2")

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    assert stored == tmp_path / "submissions" / "run-123_submission.safetensors.index.json"
    assert (tmp_path / "submissions" / "model-00001-of-00002.safetensors").read_bytes() == b"shard-1"
    assert (tmp_path / "submissions" / "model-00002-of-00002.safetensors").read_bytes() == b"shard-2"


def test_store_submission_artifact_preserves_tensorflow_checkpoint_sidecars(tmp_path: Path) -> None:
    source = tmp_path / "output" / "model.ckpt.index"
    source.parent.mkdir(exist_ok=True)
    source.write_bytes(b"index")
    (source.parent / "model.ckpt.data-00000-of-00001").write_bytes(b"weights")
    (source.parent / "model.ckpt.meta").write_bytes(b"graph")
    (source.parent / "checkpoint").write_text('model_checkpoint_path: "model.ckpt"\n', encoding="utf-8")

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    assert stored == tmp_path / "submissions" / "run-123_submission.ckpt.index"
    assert (tmp_path / "submissions" / "model.ckpt.data-00000-of-00001").read_bytes() == b"weights"
    assert (tmp_path / "submissions" / "model.ckpt.meta").read_bytes() == b"graph"
    assert (tmp_path / "submissions" / "checkpoint").read_text(encoding="utf-8") == (
        'model_checkpoint_path: "model.ckpt"\n'
    )


def test_store_submission_artifact_preserves_model_artifact_sidecars(tmp_path: Path) -> None:
    source = tmp_path / "output" / "adapter_model.safetensors"
    source.parent.mkdir(exist_ok=True)
    source.write_bytes(b"weights")
    (source.parent / "adapter_config.json").write_text('{"peft_type": "LORA"}\n', encoding="utf-8")
    (source.parent / "tokenizer_config.json").write_text('{"model_max_length": 512}\n', encoding="utf-8")

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    assert stored == tmp_path / "submissions" / "run-123_submission.safetensors"
    assert stored.read_bytes() == b"weights"
    assert (tmp_path / "submissions" / "adapter_config.json").read_text(encoding="utf-8") == '{"peft_type": "LORA"}\n'
    assert (tmp_path / "submissions" / "tokenizer_config.json").read_text(encoding="utf-8") == (
        '{"model_max_length": 512}\n'
    )


def test_submission_specific_manifest_path_strips_compound_suffix(tmp_path: Path) -> None:
    submission_path = tmp_path / "submissions" / "run-123_submission.csv.gz"

    manifest_path = submission_specific_manifest_path(submission_path)

    assert manifest_path == tmp_path / "submissions" / "run-123_submission_manifest.json"


def test_store_submission_artifact_preserves_matching_manifest_with_rewritten_submission_path(tmp_path: Path) -> None:
    source = tmp_path / "output" / "submission.csv"
    source.parent.mkdir(exist_ok=True)
    source.write_text("id,target\n1,0.1\n", encoding="utf-8")
    (source.parent / "submission_manifest.json").write_text(
        json.dumps(
            {
                "artifact_class": "tabular",
                "submission_path": "submission.csv",
                "requested_output_path": "answers.nii.gz",
            }
        ),
        encoding="utf-8",
    )

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    manifest_path = submission_specific_manifest_path(stored)
    assert manifest_path == tmp_path / "submissions" / "run-123_submission_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["submission_path"] == "run-123_submission.csv"
    assert manifest["requested_output_path"] == "answers.nii.gz"


def test_store_submission_artifact_preserves_sqlite_manifest_with_rewritten_submission_path(tmp_path: Path) -> None:
    source = tmp_path / "output" / "submission.sqlite"
    source.parent.mkdir(exist_ok=True)
    with sqlite3.connect(source) as conn:
        conn.execute("CREATE TABLE predictions (id INTEGER, target REAL)")
        conn.execute("INSERT INTO predictions VALUES (?, ?)", (1, 0.5))
    (source.parent / "submission_manifest.json").write_text(
        json.dumps(
            {
                "artifact_class": "single_file",
                "submission_path": "submission.sqlite",
            }
        ),
        encoding="utf-8",
    )

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    manifest_path = submission_specific_manifest_path(stored)
    assert manifest_path == tmp_path / "submissions" / "run-123_submission_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["artifact_class"] == "single_file"
    assert manifest["submission_path"] == "run-123_submission.sqlite"


def test_store_submission_artifact_preserves_manifest_primary_artifact_name(tmp_path: Path) -> None:
    source = tmp_path / "output" / "submission_manifest.json"
    bundle = source.parent / "bundle"
    bundle.mkdir(parents=True)
    (bundle / "mask.tif").write_bytes(b"mask")
    source.parent.mkdir(exist_ok=True)
    source.write_text(
        json.dumps(
            {
                "artifact_class": "bundle",
                "staging_dir": "bundle",
            }
        ),
        encoding="utf-8",
    )

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    assert stored == tmp_path / "submissions" / "run-123_submission_manifest.json"
    manifest = json.loads(stored.read_text(encoding="utf-8"))
    assert manifest["artifact_class"] == "bundle"
    assert manifest["staging_dir"] == "run-123_submission_bundle"
    assert (tmp_path / "submissions" / "run-123_submission_bundle" / "mask.tif").read_bytes() == b"mask"


def test_store_submission_manifest_submission_path_preserves_shapefile_sidecars(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "submission.shp").write_bytes(b"shape")
    (output_dir / "submission.dbf").write_bytes(b"attributes")
    (output_dir / "submission.prj").write_text("EPSG:4326\n", encoding="utf-8")
    (output_dir / "submission.qix").write_bytes(b"qix")
    source = output_dir / "submission_manifest.json"
    source.write_text(
        json.dumps(
            {
                "artifact_class": "single_file",
                "submission_path": "submission.shp",
            }
        ),
        encoding="utf-8",
    )

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    manifest = json.loads(stored.read_text(encoding="utf-8"))
    assert manifest["submission_path"] == "run-123_submission.shp"
    assert (tmp_path / "submissions" / "run-123_submission.shp").read_bytes() == b"shape"
    assert (tmp_path / "submissions" / "run-123_submission.dbf").read_bytes() == b"attributes"
    assert (tmp_path / "submissions" / "run-123_submission.prj").read_text(encoding="utf-8") == "EPSG:4326\n"
    assert (tmp_path / "submissions" / "run-123_submission.qix").read_bytes() == b"qix"


def test_store_submission_manifest_submission_path_preserves_georeferenced_raster_sidecars(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "submission.tif").write_bytes(b"raster")
    (output_dir / "submission.tfw").write_text("1\n0\n0\n-1\n100\n200\n", encoding="ascii")
    (output_dir / "submission.tif.aux.xml").write_text("<PAMDataset />\n", encoding="utf-8")
    source = output_dir / "submission_manifest.json"
    source.write_text(
        json.dumps(
            {
                "artifact_class": "single_file",
                "submission_path": "submission.tif",
            }
        ),
        encoding="utf-8",
    )

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    manifest = json.loads(stored.read_text(encoding="utf-8"))
    assert manifest["submission_path"] == "run-123_submission.tif"
    assert (tmp_path / "submissions" / "run-123_submission.tif").read_bytes() == b"raster"
    assert (tmp_path / "submissions" / "run-123_submission.tfw").read_text(encoding="ascii") == (
        "1\n0\n0\n-1\n100\n200\n"
    )
    assert (tmp_path / "submissions" / "run-123_submission.tif.aux.xml").read_text(encoding="utf-8") == (
        "<PAMDataset />\n"
    )


def test_store_submission_manifest_submission_path_preserves_vrt_sidecars(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    (output_dir / "rasters").mkdir(parents=True)
    (output_dir / "submission.vrt").write_text(
        """
        <VRTDataset rasterXSize="2" rasterYSize="2">
          <VRTRasterBand dataType="Byte" band="1">
            <SimpleSource><SourceFilename relativeToVRT="1">rasters/source.tif</SourceFilename></SimpleSource>
          </VRTRasterBand>
        </VRTDataset>
        """,
        encoding="utf-8",
    )
    (output_dir / "rasters" / "source.tif").write_bytes(b"raster")
    source = output_dir / "submission_manifest.json"
    source.write_text(
        json.dumps(
            {
                "artifact_class": "single_file",
                "submission_path": "submission.vrt",
            }
        ),
        encoding="utf-8",
    )

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    manifest = json.loads(stored.read_text(encoding="utf-8"))
    assert manifest["submission_path"] == "run-123_submission.vrt"
    assert (tmp_path / "submissions" / "run-123_submission.vrt").exists()
    assert (tmp_path / "submissions" / "rasters" / "source.tif").read_bytes() == b"raster"


def test_store_submission_manifest_submission_path_preserves_las_sidecars(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "submission.las").write_bytes(b"las")
    (output_dir / "submission.prj").write_text("EPSG:4326\n", encoding="utf-8")
    (output_dir / "submission.lax").write_bytes(b"index")
    source = output_dir / "submission_manifest.json"
    source.write_text(
        json.dumps(
            {
                "artifact_class": "single_file",
                "submission_path": "submission.las",
            }
        ),
        encoding="utf-8",
    )

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    manifest = json.loads(stored.read_text(encoding="utf-8"))
    assert manifest["submission_path"] == "run-123_submission.las"
    assert (tmp_path / "submissions" / "run-123_submission.las").read_bytes() == b"las"
    assert (tmp_path / "submissions" / "run-123_submission.prj").read_text(encoding="utf-8") == "EPSG:4326\n"
    assert (tmp_path / "submissions" / "run-123_submission.lax").read_bytes() == b"index"


def test_store_submission_manifest_submission_path_preserves_model_index_shards(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    index = output_dir / "model.safetensors.index.json"
    index.write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer.weight": "model-00001-of-00002.safetensors",
                    "layer.bias": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "model-00001-of-00002.safetensors").write_bytes(b"shard-1")
    (output_dir / "model-00002-of-00002.safetensors").write_bytes(b"shard-2")
    source = output_dir / "submission_manifest.json"
    source.write_text(
        json.dumps(
            {
                "artifact_class": "single_file",
                "submission_path": "model.safetensors.index.json",
            }
        ),
        encoding="utf-8",
    )

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    manifest = json.loads(stored.read_text(encoding="utf-8"))
    assert manifest["submission_path"] == "run-123_submission.safetensors.index.json"
    assert (tmp_path / "submissions" / "run-123_submission.safetensors.index.json").exists()
    assert (tmp_path / "submissions" / "model-00001-of-00002.safetensors").read_bytes() == b"shard-1"
    assert (tmp_path / "submissions" / "model-00002-of-00002.safetensors").read_bytes() == b"shard-2"


def test_store_submission_artifact_preserves_manifest_primary_members(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    mask = output_dir / "mask.tif"
    mask.write_bytes(b"mask")
    source = output_dir / "submission_manifest.json"
    source.write_text(
        json.dumps(
            {
                "artifact_class": "multi_file_zip",
                "members": [{"source_path": "mask.tif", "archive_path": "masks/mask.tif"}],
            }
        ),
        encoding="utf-8",
    )

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    manifest = json.loads(stored.read_text(encoding="utf-8"))
    assert manifest["staging_dir"] == "run-123_submission_bundle"
    assert manifest["members"] == [
        {
            "source_path": "run-123_submission_bundle/mask.tif",
            "archive_path": "masks/mask.tif",
        }
    ]
    assert (tmp_path / "submissions" / "run-123_submission_bundle" / "mask.tif").read_bytes() == b"mask"


def test_store_submission_manifest_member_preserves_model_index_shards(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    index = output_dir / "model.safetensors.index.json"
    index.write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer.weight": "model-00001-of-00002.safetensors",
                    "layer.bias": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "model-00001-of-00002.safetensors").write_bytes(b"shard-1")
    (output_dir / "model-00002-of-00002.safetensors").write_bytes(b"shard-2")
    source = output_dir / "submission_manifest.json"
    source.write_text(
        json.dumps(
            {
                "artifact_class": "multi_file_zip",
                "members": [
                    {
                        "source_path": "model.safetensors.index.json",
                        "archive_path": "models/model.safetensors.index.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    manifest = json.loads(stored.read_text(encoding="utf-8"))
    assert manifest["staging_dir"] == "run-123_submission_bundle"
    assert manifest["members"] == [
        {
            "source_path": "run-123_submission_bundle/model.safetensors.index.json",
            "archive_path": "models/model.safetensors.index.json",
        },
        {
            "source_path": "run-123_submission_bundle/model-00001-of-00002.safetensors",
            "archive_path": "models/model-00001-of-00002.safetensors",
        },
        {
            "source_path": "run-123_submission_bundle/model-00002-of-00002.safetensors",
            "archive_path": "models/model-00002-of-00002.safetensors",
        },
    ]
    assert (
        tmp_path / "submissions" / "run-123_submission_bundle" / "model-00001-of-00002.safetensors"
    ).read_bytes() == b"shard-1"


def test_store_submission_manifest_member_preserves_tensorflow_checkpoint_sidecars(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    index = output_dir / "model.ckpt.index"
    index.write_bytes(b"index")
    (output_dir / "model.ckpt.data-00000-of-00001").write_bytes(b"weights")
    (output_dir / "checkpoint").write_text('model_checkpoint_path: "model.ckpt"\n', encoding="utf-8")
    source = output_dir / "submission_manifest.json"
    source.write_text(
        json.dumps(
            {
                "artifact_class": "multi_file_zip",
                "members": [
                    {
                        "source_path": "model.ckpt.index",
                        "archive_path": "models/model.ckpt.index",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    manifest = json.loads(stored.read_text(encoding="utf-8"))
    assert manifest["members"] == [
        {
            "source_path": "run-123_submission_bundle/model.ckpt.index",
            "archive_path": "models/model.ckpt.index",
        },
        {
            "source_path": "run-123_submission_bundle/model.ckpt.data-00000-of-00001",
            "archive_path": "models/model.ckpt.data-00000-of-00001",
        },
        {
            "source_path": "run-123_submission_bundle/checkpoint",
            "archive_path": "models/checkpoint",
        },
    ]
    assert (tmp_path / "submissions" / "run-123_submission_bundle" / "model.ckpt.data-00000-of-00001").read_bytes() == (
        b"weights"
    )


def test_store_submission_manifest_member_preserves_metaimage_sidecars(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    (output_dir / "raw").mkdir(parents=True)
    (output_dir / "submission.mhd").write_text(
        "ObjectType = Image\nElementDataFile = raw/volume.raw\n",
        encoding="utf-8",
    )
    (output_dir / "raw" / "volume.raw").write_bytes(b"voxels")
    source = output_dir / "submission_manifest.json"
    source.write_text(
        json.dumps(
            {
                "artifact_class": "multi_file_zip",
                "members": [
                    {
                        "source_path": "submission.mhd",
                        "archive_path": "medical/submission.mhd",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    manifest = json.loads(stored.read_text(encoding="utf-8"))
    assert manifest["staging_dir"] == "run-123_submission_bundle"
    assert manifest["members"] == [
        {
            "source_path": "run-123_submission_bundle/submission.mhd",
            "archive_path": "medical/submission.mhd",
        },
        {
            "source_path": "run-123_submission_bundle/raw/volume.raw",
            "archive_path": "medical/raw/volume.raw",
        },
    ]
    assert (tmp_path / "submissions" / "run-123_submission_bundle" / "raw" / "volume.raw").read_bytes() == b"voxels"


def test_store_submission_manifest_member_preserves_detached_nrrd_sidecars(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    (output_dir / "raw").mkdir(parents=True)
    (output_dir / "submission.nhdr").write_text(
        "NRRD0005\nsizes: 4 5 6\ndata file: raw/volume.raw\n",
        encoding="utf-8",
    )
    (output_dir / "raw" / "volume.raw").write_bytes(b"voxels")
    source = output_dir / "submission_manifest.json"
    source.write_text(
        json.dumps(
            {
                "artifact_class": "multi_file_zip",
                "members": [
                    {
                        "source_path": "submission.nhdr",
                        "archive_path": "medical/submission.nhdr",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    manifest = json.loads(stored.read_text(encoding="utf-8"))
    assert manifest["staging_dir"] == "run-123_submission_bundle"
    assert manifest["members"] == [
        {
            "source_path": "run-123_submission_bundle/submission.nhdr",
            "archive_path": "medical/submission.nhdr",
        },
        {
            "source_path": "run-123_submission_bundle/raw/volume.raw",
            "archive_path": "medical/raw/volume.raw",
        },
    ]
    assert (tmp_path / "submissions" / "run-123_submission_bundle" / "raw" / "volume.raw").read_bytes() == b"voxels"


def test_store_submission_manifest_member_preserves_analyze_pair_sidecars(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "submission.hdr").write_bytes(b"header")
    (output_dir / "submission.img").write_bytes(b"volume")
    source = output_dir / "submission_manifest.json"
    source.write_text(
        json.dumps(
            {
                "artifact_class": "multi_file_zip",
                "members": [
                    {
                        "source_path": "submission.hdr",
                        "archive_path": "medical/submission.hdr",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    manifest = json.loads(stored.read_text(encoding="utf-8"))
    assert manifest["staging_dir"] == "run-123_submission_bundle"
    assert manifest["members"] == [
        {
            "source_path": "run-123_submission_bundle/submission.hdr",
            "archive_path": "medical/submission.hdr",
        },
        {
            "source_path": "run-123_submission_bundle/submission.img",
            "archive_path": "medical/submission.img",
        },
    ]
    assert (tmp_path / "submissions" / "run-123_submission_bundle" / "submission.img").read_bytes() == b"volume"


def test_store_submission_manifest_member_preserves_obj_sidecars(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    (output_dir / "materials" / "textures").mkdir(parents=True)
    (output_dir / "submission.obj").write_text("mtllib materials/model.mtl\nv 0 0 0\n", encoding="utf-8")
    (output_dir / "materials" / "model.mtl").write_text(
        "newmtl surface\nmap_Kd textures/diffuse.png\n",
        encoding="utf-8",
    )
    (output_dir / "materials" / "textures" / "diffuse.png").write_bytes(b"texture")
    source = output_dir / "submission_manifest.json"
    source.write_text(
        json.dumps(
            {
                "artifact_class": "multi_file_zip",
                "members": [
                    {
                        "source_path": "submission.obj",
                        "archive_path": "mesh/submission.obj",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    manifest = json.loads(stored.read_text(encoding="utf-8"))
    assert manifest["staging_dir"] == "run-123_submission_bundle"
    assert manifest["members"] == [
        {
            "source_path": "run-123_submission_bundle/submission.obj",
            "archive_path": "mesh/submission.obj",
        },
        {
            "source_path": "run-123_submission_bundle/materials/model.mtl",
            "archive_path": "mesh/materials/model.mtl",
        },
        {
            "source_path": "run-123_submission_bundle/materials/textures/diffuse.png",
            "archive_path": "mesh/materials/textures/diffuse.png",
        },
    ]
    assert (
        tmp_path / "submissions" / "run-123_submission_bundle" / "materials" / "textures" / "diffuse.png"
    ).read_bytes() == b"texture"


def test_store_submission_manifest_member_preserves_kml_sidecars(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    (output_dir / "icons").mkdir(parents=True)
    (output_dir / "submission.kml").write_text(
        """
        <kml xmlns="http://www.opengis.net/kml/2.2">
          <Document><Icon><href>icons/pin.png</href></Icon></Document>
        </kml>
        """,
        encoding="utf-8",
    )
    (output_dir / "icons" / "pin.png").write_bytes(b"pin")
    source = output_dir / "submission_manifest.json"
    source.write_text(
        json.dumps(
            {
                "artifact_class": "multi_file_zip",
                "members": [
                    {
                        "source_path": "submission.kml",
                        "archive_path": "geo/submission.kml",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    manifest = json.loads(stored.read_text(encoding="utf-8"))
    assert manifest["staging_dir"] == "run-123_submission_bundle"
    assert manifest["members"] == [
        {
            "source_path": "run-123_submission_bundle/submission.kml",
            "archive_path": "geo/submission.kml",
        },
        {
            "source_path": "run-123_submission_bundle/icons/pin.png",
            "archive_path": "geo/icons/pin.png",
        },
    ]
    assert (tmp_path / "submissions" / "run-123_submission_bundle" / "icons" / "pin.png").read_bytes() == b"pin"


def test_store_submission_manifest_member_preserves_georeferenced_raster_sidecars(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "submission.tif").write_bytes(b"raster")
    (output_dir / "submission.tfw").write_text("1\n0\n0\n-1\n100\n200\n", encoding="ascii")
    (output_dir / "submission.tif.aux.xml").write_text("<PAMDataset />\n", encoding="utf-8")
    source = output_dir / "submission_manifest.json"
    source.write_text(
        json.dumps(
            {
                "artifact_class": "multi_file_zip",
                "members": [{"source_path": "submission.tif", "archive_path": "raster/submission.tif"}],
            }
        ),
        encoding="utf-8",
    )

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    manifest = json.loads(stored.read_text(encoding="utf-8"))
    assert manifest["staging_dir"] == "run-123_submission_bundle"
    assert manifest["members"] == [
        {"source_path": "run-123_submission_bundle/submission.tif", "archive_path": "raster/submission.tif"},
        {"source_path": "run-123_submission_bundle/submission.tfw", "archive_path": "raster/submission.tfw"},
        {
            "source_path": "run-123_submission_bundle/submission.tif.aux.xml",
            "archive_path": "raster/submission.tif.aux.xml",
        },
    ]
    assert (tmp_path / "submissions" / "run-123_submission_bundle" / "submission.tfw").read_text(
        encoding="ascii"
    ) == "1\n0\n0\n-1\n100\n200\n"
    assert (tmp_path / "submissions" / "run-123_submission_bundle" / "submission.tif.aux.xml").read_text(
        encoding="utf-8"
    ) == "<PAMDataset />\n"


def test_store_submission_manifest_member_preserves_vrt_sidecars(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    (output_dir / "rasters").mkdir(parents=True)
    (output_dir / "submission.vrt").write_text(
        """
        <VRTDataset rasterXSize="2" rasterYSize="2">
          <VRTRasterBand dataType="Byte" band="1">
            <SimpleSource><SourceFilename relativeToVRT="1">rasters/source.tif</SourceFilename></SimpleSource>
          </VRTRasterBand>
        </VRTDataset>
        """,
        encoding="utf-8",
    )
    (output_dir / "rasters" / "source.tif").write_bytes(b"raster")
    source = output_dir / "submission_manifest.json"
    source.write_text(
        json.dumps(
            {
                "artifact_class": "multi_file_zip",
                "members": [{"source_path": "submission.vrt", "archive_path": "vrt/submission.vrt"}],
            }
        ),
        encoding="utf-8",
    )

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    manifest = json.loads(stored.read_text(encoding="utf-8"))
    assert manifest["staging_dir"] == "run-123_submission_bundle"
    assert manifest["members"] == [
        {"source_path": "run-123_submission_bundle/submission.vrt", "archive_path": "vrt/submission.vrt"},
        {
            "source_path": "run-123_submission_bundle/rasters/source.tif",
            "archive_path": "vrt/rasters/source.tif",
        },
    ]
    assert (tmp_path / "submissions" / "run-123_submission_bundle" / "rasters" / "source.tif").read_bytes() == (
        b"raster"
    )


def test_store_submission_manifest_member_preserves_envi_header_sidecars(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "submission.hdr").write_text(
        "ENVI\nsamples = 2\nlines = 2\nbands = 1\n",
        encoding="utf-8",
    )
    (output_dir / "submission.dat").write_bytes(b"raster")
    source = output_dir / "submission_manifest.json"
    source.write_text(
        json.dumps(
            {
                "artifact_class": "multi_file_zip",
                "members": [{"source_path": "submission.hdr", "archive_path": "raster/submission.hdr"}],
            }
        ),
        encoding="utf-8",
    )

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    manifest = json.loads(stored.read_text(encoding="utf-8"))
    assert manifest["staging_dir"] == "run-123_submission_bundle"
    assert manifest["members"] == [
        {"source_path": "run-123_submission_bundle/submission.hdr", "archive_path": "raster/submission.hdr"},
        {"source_path": "run-123_submission_bundle/submission.dat", "archive_path": "raster/submission.dat"},
    ]
    assert (tmp_path / "submissions" / "run-123_submission_bundle" / "submission.dat").read_bytes() == b"raster"


def test_store_submission_manifest_member_preserves_dae_sidecars(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    (output_dir / "textures").mkdir(parents=True)
    (output_dir / "submission.dae").write_text(
        """
        <COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema">
          <library_images>
            <image id="diffuse"><init_from>textures/diffuse.png</init_from></image>
          </library_images>
        </COLLADA>
        """,
        encoding="utf-8",
    )
    (output_dir / "textures" / "diffuse.png").write_bytes(b"texture")
    source = output_dir / "submission_manifest.json"
    source.write_text(
        json.dumps(
            {
                "artifact_class": "multi_file_zip",
                "members": [
                    {
                        "source_path": "submission.dae",
                        "archive_path": "mesh/submission.dae",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    manifest = json.loads(stored.read_text(encoding="utf-8"))
    assert manifest["staging_dir"] == "run-123_submission_bundle"
    assert manifest["members"] == [
        {
            "source_path": "run-123_submission_bundle/submission.dae",
            "archive_path": "mesh/submission.dae",
        },
        {
            "source_path": "run-123_submission_bundle/textures/diffuse.png",
            "archive_path": "mesh/textures/diffuse.png",
        },
    ]
    assert (
        tmp_path / "submissions" / "run-123_submission_bundle" / "textures" / "diffuse.png"
    ).read_bytes() == b"texture"


def test_store_submission_manifest_member_preserves_las_sidecars(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "submission.las").write_bytes(b"las")
    (output_dir / "submission.prj").write_text("EPSG:4326\n", encoding="utf-8")
    (output_dir / "submission.lax").write_bytes(b"index")
    source = output_dir / "submission_manifest.json"
    source.write_text(
        json.dumps(
            {
                "artifact_class": "multi_file_zip",
                "members": [{"source_path": "submission.las", "archive_path": "pointcloud/submission.las"}],
            }
        ),
        encoding="utf-8",
    )

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    manifest = json.loads(stored.read_text(encoding="utf-8"))
    assert manifest["staging_dir"] == "run-123_submission_bundle"
    assert manifest["members"] == [
        {"source_path": "run-123_submission_bundle/submission.las", "archive_path": "pointcloud/submission.las"},
        {"source_path": "run-123_submission_bundle/submission.prj", "archive_path": "pointcloud/submission.prj"},
        {"source_path": "run-123_submission_bundle/submission.lax", "archive_path": "pointcloud/submission.lax"},
    ]
    assert (tmp_path / "submissions" / "run-123_submission_bundle" / "submission.lax").read_bytes() == b"index"


def test_store_submission_manifest_member_preserves_ply_sidecars(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    (output_dir / "textures").mkdir(parents=True)
    (output_dir / "submission.ply").write_text(
        "ply\nformat ascii 1.0\ncomment TextureFile textures/diffuse.png\nelement vertex 0\nend_header\n",
        encoding="ascii",
    )
    (output_dir / "textures" / "diffuse.png").write_bytes(b"texture")
    source = output_dir / "submission_manifest.json"
    source.write_text(
        json.dumps(
            {
                "artifact_class": "multi_file_zip",
                "members": [
                    {
                        "source_path": "submission.ply",
                        "archive_path": "mesh/submission.ply",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    manifest = json.loads(stored.read_text(encoding="utf-8"))
    assert manifest["staging_dir"] == "run-123_submission_bundle"
    assert manifest["members"] == [
        {
            "source_path": "run-123_submission_bundle/submission.ply",
            "archive_path": "mesh/submission.ply",
        },
        {
            "source_path": "run-123_submission_bundle/textures/diffuse.png",
            "archive_path": "mesh/textures/diffuse.png",
        },
    ]
    assert (
        tmp_path / "submissions" / "run-123_submission_bundle" / "textures" / "diffuse.png"
    ).read_bytes() == b"texture"


def test_store_submission_manifest_member_preserves_x3d_sidecars(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    (output_dir / "textures").mkdir(parents=True)
    (output_dir / "submission.x3d").write_text(
        """
        <X3D>
          <Scene><ImageTexture url='"textures/diffuse.png"'/></Scene>
        </X3D>
        """,
        encoding="utf-8",
    )
    (output_dir / "textures" / "diffuse.png").write_bytes(b"texture")
    source = output_dir / "submission_manifest.json"
    source.write_text(
        json.dumps(
            {
                "artifact_class": "multi_file_zip",
                "members": [
                    {
                        "source_path": "submission.x3d",
                        "archive_path": "scene/submission.x3d",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    manifest = json.loads(stored.read_text(encoding="utf-8"))
    assert manifest["staging_dir"] == "run-123_submission_bundle"
    assert manifest["members"] == [
        {
            "source_path": "run-123_submission_bundle/submission.x3d",
            "archive_path": "scene/submission.x3d",
        },
        {
            "source_path": "run-123_submission_bundle/textures/diffuse.png",
            "archive_path": "scene/textures/diffuse.png",
        },
    ]
    assert (
        tmp_path / "submissions" / "run-123_submission_bundle" / "textures" / "diffuse.png"
    ).read_bytes() == b"texture"


def test_store_submission_manifest_member_preserves_gltf_sidecars(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    (output_dir / "buffers").mkdir(parents=True)
    (output_dir / "textures").mkdir()
    (output_dir / "submission.gltf").write_text(
        json.dumps(
            {
                "asset": {"version": "2.0"},
                "buffers": [{"uri": "buffers/scene.bin"}],
                "images": [{"uri": "textures/diffuse.png"}],
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "buffers" / "scene.bin").write_bytes(b"buffer")
    (output_dir / "textures" / "diffuse.png").write_bytes(b"texture")
    source = output_dir / "submission_manifest.json"
    source.write_text(
        json.dumps(
            {
                "artifact_class": "multi_file_zip",
                "members": [
                    {
                        "source_path": "submission.gltf",
                        "archive_path": "scene/submission.gltf",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    manifest = json.loads(stored.read_text(encoding="utf-8"))
    assert manifest["staging_dir"] == "run-123_submission_bundle"
    assert manifest["members"] == [
        {
            "source_path": "run-123_submission_bundle/submission.gltf",
            "archive_path": "scene/submission.gltf",
        },
        {
            "source_path": "run-123_submission_bundle/buffers/scene.bin",
            "archive_path": "scene/buffers/scene.bin",
        },
        {
            "source_path": "run-123_submission_bundle/textures/diffuse.png",
            "archive_path": "scene/textures/diffuse.png",
        },
    ]
    assert (tmp_path / "submissions" / "run-123_submission_bundle" / "buffers" / "scene.bin").read_bytes() == b"buffer"
    assert (
        tmp_path / "submissions" / "run-123_submission_bundle" / "textures" / "diffuse.png"
    ).read_bytes() == b"texture"


def test_store_submission_manifest_member_preserves_gltf_parent_relative_sidecars(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    (output_dir / "scenes").mkdir(parents=True)
    (output_dir / "textures").mkdir()
    (output_dir / "scenes" / "submission.gltf").write_text(
        json.dumps(
            {
                "asset": {"version": "2.0"},
                "images": [{"uri": "../textures/diffuse.png"}],
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "textures" / "diffuse.png").write_bytes(b"texture")
    source = output_dir / "submission_manifest.json"
    source.write_text(
        json.dumps(
            {
                "artifact_class": "multi_file_zip",
                "members": [
                    {
                        "source_path": "scenes/submission.gltf",
                        "archive_path": "scene/submission.gltf",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    manifest = json.loads(stored.read_text(encoding="utf-8"))
    assert manifest["members"] == [
        {
            "source_path": "run-123_submission_bundle/scenes/submission.gltf",
            "archive_path": "scene/submission.gltf",
        },
        {
            "source_path": "run-123_submission_bundle/textures/diffuse.png",
            "archive_path": "textures/diffuse.png",
        },
    ]
    assert (tmp_path / "submissions" / "run-123_submission_bundle" / "textures" / "diffuse.png").read_bytes() == (
        b"texture"
    )


def test_store_submission_manifest_member_preserves_model_artifact_sidecars(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "adapter_model.safetensors").write_bytes(b"weights")
    (output_dir / "adapter_config.json").write_text('{"peft_type": "LORA"}\n', encoding="utf-8")
    (output_dir / "tokenizer_config.json").write_text('{"model_max_length": 512}\n', encoding="utf-8")
    source = output_dir / "submission_manifest.json"
    source.write_text(
        json.dumps(
            {
                "artifact_class": "multi_file_zip",
                "members": [
                    {
                        "source_path": "adapter_model.safetensors",
                        "archive_path": "models/adapter_model.safetensors",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    manifest = json.loads(stored.read_text(encoding="utf-8"))
    assert manifest["staging_dir"] == "run-123_submission_bundle"
    assert manifest["members"] == [
        {
            "source_path": "run-123_submission_bundle/adapter_model.safetensors",
            "archive_path": "models/adapter_model.safetensors",
        },
        {
            "source_path": "run-123_submission_bundle/adapter_config.json",
            "archive_path": "models/adapter_config.json",
        },
        {
            "source_path": "run-123_submission_bundle/tokenizer_config.json",
            "archive_path": "models/tokenizer_config.json",
        },
    ]
    assert (tmp_path / "submissions" / "run-123_submission_bundle" / "adapter_config.json").read_text(
        encoding="utf-8"
    ) == '{"peft_type": "LORA"}\n'


def test_store_submission_manifest_member_preserves_shapefile_sidecars(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "submission.shp").write_bytes(b"shape")
    (output_dir / "submission.dbf").write_bytes(b"attributes")
    (output_dir / "submission.prj").write_text("EPSG:4326\n", encoding="utf-8")
    (output_dir / "submission.qix").write_bytes(b"qix")
    source = output_dir / "submission_manifest.json"
    source.write_text(
        json.dumps(
            {
                "artifact_class": "multi_file_zip",
                "members": [{"source_path": "submission.shp", "archive_path": "geo/submission.shp"}],
            }
        ),
        encoding="utf-8",
    )

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    manifest = json.loads(stored.read_text(encoding="utf-8"))
    assert manifest["staging_dir"] == "run-123_submission_bundle"
    assert manifest["members"] == [
        {"source_path": "run-123_submission_bundle/submission.shp", "archive_path": "geo/submission.shp"},
        {"source_path": "run-123_submission_bundle/submission.dbf", "archive_path": "geo/submission.dbf"},
        {"source_path": "run-123_submission_bundle/submission.prj", "archive_path": "geo/submission.prj"},
        {"source_path": "run-123_submission_bundle/submission.qix", "archive_path": "geo/submission.qix"},
    ]
    assert (tmp_path / "submissions" / "run-123_submission_bundle" / "submission.dbf").read_bytes() == b"attributes"
    assert (tmp_path / "submissions" / "run-123_submission_bundle" / "submission.qix").read_bytes() == b"qix"


def test_store_submission_manifest_member_preserves_mapinfo_sidecars(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "submission.tab").write_text("!table\n!version 300\n", encoding="utf-8")
    (output_dir / "submission.dat").write_bytes(b"data")
    (output_dir / "submission.id").write_bytes(b"ids")
    (output_dir / "submission.map").write_bytes(b"map")
    source = output_dir / "submission_manifest.json"
    source.write_text(
        json.dumps(
            {
                "artifact_class": "multi_file_zip",
                "members": [{"source_path": "submission.tab", "archive_path": "geo/submission.tab"}],
            }
        ),
        encoding="utf-8",
    )

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    manifest = json.loads(stored.read_text(encoding="utf-8"))
    assert manifest["staging_dir"] == "run-123_submission_bundle"
    assert manifest["members"] == [
        {"source_path": "run-123_submission_bundle/submission.tab", "archive_path": "geo/submission.tab"},
        {"source_path": "run-123_submission_bundle/submission.dat", "archive_path": "geo/submission.dat"},
        {"source_path": "run-123_submission_bundle/submission.id", "archive_path": "geo/submission.id"},
        {"source_path": "run-123_submission_bundle/submission.map", "archive_path": "geo/submission.map"},
    ]
    assert (tmp_path / "submissions" / "run-123_submission_bundle" / "submission.map").read_bytes() == b"map"


def test_store_submission_manifest_member_preserves_mapinfo_interchange_sidecar(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "submission.mif").write_text(
        "Version 300\nColumns 1\n  Name Char(20)\nData\n",
        encoding="utf-8",
    )
    (output_dir / "submission.mid").write_text('"parcel-a"\n', encoding="utf-8")
    source = output_dir / "submission_manifest.json"
    source.write_text(
        json.dumps(
            {
                "artifact_class": "multi_file_zip",
                "members": [{"source_path": "submission.mif", "archive_path": "geo/submission.mif"}],
            }
        ),
        encoding="utf-8",
    )

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    manifest = json.loads(stored.read_text(encoding="utf-8"))
    assert manifest["staging_dir"] == "run-123_submission_bundle"
    assert manifest["members"] == [
        {"source_path": "run-123_submission_bundle/submission.mif", "archive_path": "geo/submission.mif"},
        {"source_path": "run-123_submission_bundle/submission.mid", "archive_path": "geo/submission.mid"},
    ]
    assert (tmp_path / "submissions" / "run-123_submission_bundle" / "submission.mid").read_text(
        encoding="utf-8"
    ) == '"parcel-a"\n'


def test_store_submission_artifact_preserves_manifest_primary_member_layout(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    left = output_dir / "fold1" / "mask.tif"
    right = output_dir / "fold2" / "mask.tif"
    left.parent.mkdir(parents=True)
    right.parent.mkdir(parents=True)
    left.write_bytes(b"left")
    right.write_bytes(b"right")
    source = output_dir / "submission_manifest.json"
    source.write_text(
        json.dumps(
            {
                "artifact_class": "multi_file_zip",
                "members": ["fold1/mask.tif", "fold2/mask.tif"],
            }
        ),
        encoding="utf-8",
    )

    stored = store_submission_artifact(
        source=source,
        destination_dir=tmp_path / "submissions",
        run_id="run-123",
    )

    manifest = json.loads(stored.read_text(encoding="utf-8"))
    assert manifest["staging_dir"] == "run-123_submission_bundle"
    assert manifest["members"] == [
        "run-123_submission_bundle/fold1/mask.tif",
        "run-123_submission_bundle/fold2/mask.tif",
    ]
    assert (tmp_path / "submissions" / "run-123_submission_bundle" / "fold1" / "mask.tif").read_bytes() == b"left"
    assert (tmp_path / "submissions" / "run-123_submission_bundle" / "fold2" / "mask.tif").read_bytes() == b"right"


def test_store_submission_artifact_noops_when_destination_is_source(tmp_path: Path) -> None:
    source = tmp_path / "submissions" / "run-123_submission.csv"
    source.parent.mkdir()
    source.write_text("id,target\n1,0.1\n", encoding="utf-8")

    stored = store_submission_artifact(
        source=source,
        destination_dir=source.parent,
        run_id="run-123",
    )

    assert stored == source
    assert source.read_text(encoding="utf-8") == "id,target\n1,0.1\n"
