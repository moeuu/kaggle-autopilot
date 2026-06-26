from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from kagglebot.kernel_outputs import (
    copy_artifact_if_needed,
    find_intermediate_submission_file,
    find_newest_existing_path,
    find_output_file,
    find_submission_file,
    pick_latest_artifact,
    resolve_local_kernel_artifact_file,
    resolve_local_kernel_artifacts,
)


def test_find_submission_file(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    nested = output_dir / "nested"
    nested.mkdir()
    submission = nested / "submission.csv"
    submission.write_text("id,target\n1,0.1\n", encoding="utf-8")
    assert find_submission_file(output_dir) == submission


def test_find_submission_file_supports_zip_submission(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    submission = output_dir / "submission.zip"
    submission.write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    assert find_submission_file(output_dir) == submission


@pytest.mark.parametrize("name", ["submission.tar.gz", "submission.tgz"])
@pytest.mark.parametrize("nested", [False, True])
def test_find_submission_file_supports_compound_code_submission_archives(
    tmp_path: Path,
    name: str,
    nested: bool,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    artifact_dir = output_dir / "nested" if nested else output_dir
    artifact_dir.mkdir(exist_ok=True)
    submission = artifact_dir / name
    submission.write_bytes(b"\x1f\x8b")
    assert find_submission_file(output_dir) == submission


def test_find_submission_file_supports_submission_manifest(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    bundle_dir = output_dir / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "mask.tif").write_bytes(b"mask")
    manifest = output_dir / "submission_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact_class": "multi_file_zip",
                "staging_dir": "bundle",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    assert find_submission_file(output_dir) == manifest


def test_find_output_file_picks_newest_match(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    direct = output_dir / "metrics.json"
    direct.write_text('{"metric":"rmse"}\n', encoding="utf-8")
    nested = output_dir / "nested"
    nested.mkdir()
    newest = nested / "metrics.json"
    newest.write_text('{"metric":"rmse","offline_value":0.1}\n', encoding="utf-8")

    os.utime(direct, (1000, 1000))
    os.utime(newest, (2000, 2000))

    assert find_output_file(output_dir, "metrics.json") == newest


def test_find_output_file_prefers_newest_under_run_tree(tmp_path: Path) -> None:
    root = tmp_path / "kernel-run"
    (root / "outputs").mkdir(parents=True, exist_ok=True)
    (root / "runs" / "run_2").mkdir(parents=True, exist_ok=True)
    older = root / "outputs" / "metrics.json"
    newer = root / "runs" / "run_2" / "metrics.json"
    older.write_text('{"metric":"rmse"}\n', encoding="utf-8")
    newer.write_text('{"metric":"rmse","offline_value":0.1}\n', encoding="utf-8")

    os.utime(older, (1000, 1000))
    os.utime(newer, (2000, 2000))

    assert find_output_file(root, "metrics.json") == newer


def test_find_submission_file_uses_newest_fold_intermediate_when_final_missing(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    fold1 = output_dir / "submission_qwen_fold1.csv"
    fold2 = output_dir / "nested" / "submission_qwen_fold2.csv"
    fold2.parent.mkdir()
    fold1.write_text("id,target\n1,0.1\n", encoding="utf-8")
    fold2.write_text("id,target\n1,0.2\n", encoding="utf-8")

    os.utime(fold1, (1000, 1000))
    os.utime(fold2, (2000, 2000))

    assert find_intermediate_submission_file(output_dir) == fold2
    assert find_submission_file(output_dir) == fold2


def test_find_submission_file_prefers_final_submission_over_fold_intermediate(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    final = output_dir / "submission.csv"
    fold = output_dir / "submission_qwen_fold1.csv"
    final.write_text("id,target\n1,0.3\n", encoding="utf-8")
    fold.write_text("id,target\n1,0.1\n", encoding="utf-8")

    os.utime(final, (1000, 1000))
    os.utime(fold, (2000, 2000))

    assert find_submission_file(output_dir) == final


def test_pick_latest_artifact_filters_stale_files(tmp_path: Path) -> None:
    stale = tmp_path / "stale.csv"
    fresh = tmp_path / "fresh.csv"
    stale.write_text("old", encoding="utf-8")
    fresh.write_text("new", encoding="utf-8")
    os.utime(stale, (1000, 1000))
    os.utime(fresh, (2000, 2000))

    assert pick_latest_artifact([stale, fresh], min_mtime=1500) == fresh
    assert pick_latest_artifact([stale], min_mtime=1500) is None


def test_find_newest_existing_path_uses_size_and_path_tiebreakers(tmp_path: Path) -> None:
    smaller = tmp_path / "a.json"
    larger = tmp_path / "b.json"
    smaller.write_text("{}", encoding="utf-8")
    larger.write_text('{"value": 1}', encoding="utf-8")
    os.utime(smaller, (2000, 2000))
    os.utime(larger, (2000, 2000))

    assert find_newest_existing_path([smaller, larger]) == larger


def test_resolve_local_kernel_artifacts_finds_fresh_nested_outputs(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "demo" / "kernels" / "run-1" / "local-iter-1"
    output_dir = tmp_path / "demo" / "runs" / "run-1" / "iter-1" / "output"
    nested_outputs = kernel_dir.parent / "outputs"
    nested_outputs.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    submission = nested_outputs / "submission.csv"
    metrics = nested_outputs / "metrics.json"
    submission.write_text("id,target\n1,0.2\n", encoding="utf-8")
    metrics.write_text('{"metric":"rmse"}\n', encoding="utf-8")
    os.utime(submission, (2000, 2000))
    os.utime(metrics, (2000, 2000))

    resolved_submission, resolved_metrics = resolve_local_kernel_artifacts(
        kernel_dir=kernel_dir,
        output_dir=output_dir,
        started_at=1500,
    )

    assert resolved_submission == submission
    assert resolved_metrics == metrics


def test_resolve_local_kernel_artifact_file_and_copy(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "demo" / "kernels" / "run-1" / "local-iter-1"
    output_dir = tmp_path / "demo" / "runs" / "run-1" / "iter-1" / "output"
    artifact_dir = kernel_dir / "outputs"
    artifact_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    source = artifact_dir / "cv_results.json"
    source.write_text("{}", encoding="utf-8")
    os.utime(source, (2000, 2000))

    resolved = resolve_local_kernel_artifact_file(
        kernel_dir=kernel_dir,
        output_dir=output_dir,
        started_at=1500,
        filename="cv_results.json",
    )
    destination = output_dir / "cv_results.json"

    assert resolved == source
    assert copy_artifact_if_needed(source=source, destination=destination) == destination
    assert destination.read_text(encoding="utf-8") == "{}"
