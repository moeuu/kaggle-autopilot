from __future__ import annotations

from pathlib import Path

import pytest

import kagglebot.autopilot as autopilot
from kagglebot.autopilot import _iteration_submission_path


def test_iteration_submission_path_uses_tabular_sample_suffix(tmp_path: Path) -> None:
    assert (
        _iteration_submission_path(
            iter_dir=tmp_path / "iter-1",
            sample_submission_path=tmp_path / "sample_submission.jsonl",
        ).name
        == "submission.jsonl"
    )


def test_iteration_submission_path_falls_back_to_csv_for_unknown_sample_suffix(tmp_path: Path) -> None:
    assert (
        _iteration_submission_path(
            iter_dir=tmp_path / "iter-1",
            sample_submission_path=tmp_path / "sample_submission.zip",
        ).name
        == "submission.csv"
    )


def test_iteration_submission_path_uses_submission_format_archive_suffix(tmp_path: Path) -> None:
    format_path = tmp_path / "submission_format.md"
    format_path.write_text(
        "## Submission Format\nSubmit a submission.tar.xz archive containing model weights and inference code.\n",
        encoding="utf-8",
    )

    assert (
        _iteration_submission_path(
            iter_dir=tmp_path / "iter-1",
            sample_submission_path=tmp_path / "sample_submission.csv",
            submission_format_path=format_path,
        ).name
        == "submission.tar.xz"
    )


def test_iteration_submission_path_uses_submission_format_single_file_suffix(tmp_path: Path) -> None:
    format_path = tmp_path / "submission_format.md"
    format_path.write_text(
        "## Submission Format\nSubmit a single NIfTI file named `submission.nii.gz`.\n",
        encoding="utf-8",
    )

    assert (
        _iteration_submission_path(
            iter_dir=tmp_path / "iter-1",
            sample_submission_path=tmp_path / "sample_submission.csv",
            submission_format_path=format_path,
        ).name
        == "submission.nii.gz"
    )


def test_iteration_submission_path_preserves_explicit_model_artifact_filename(tmp_path: Path) -> None:
    format_path = tmp_path / "submission_format.md"
    format_path.write_text(
        "## Submission Format\nUpload `model.safetensors.index.json` as the final output.\n",
        encoding="utf-8",
    )

    assert (
        _iteration_submission_path(
            iter_dir=tmp_path / "iter-1",
            sample_submission_path=tmp_path / "sample_submission.csv",
            submission_format_path=format_path,
        ).name
        == "model.safetensors.index.json"
    )


def test_iteration_submission_path_ignores_template_filename_and_uses_suffix(tmp_path: Path) -> None:
    format_path = tmp_path / "submission_format.md"
    format_path.write_text(
        "## Submission Format\n"
        "Use `sample_submission.safetensors.index.json` as a template, then submit the final file.\n",
        encoding="utf-8",
    )

    assert (
        _iteration_submission_path(
            iter_dir=tmp_path / "iter-1",
            sample_submission_path=tmp_path / "sample_submission.csv",
            submission_format_path=format_path,
        ).name
        == "submission.safetensors.index.json"
    )


def test_iteration_submission_path_normalizes_submission_format_suffix_without_dot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    format_path = tmp_path / "submission_format.md"
    format_path.write_text("## Submission Format\nSubmit a tar.zst archive.\n", encoding="utf-8")

    class Hint:
        expected_suffixes = ["tar.zst"]

    monkeypatch.setattr(autopilot, "load_submission_format_hint", lambda _path: Hint())

    assert (
        _iteration_submission_path(
            iter_dir=tmp_path / "iter-1",
            sample_submission_path=tmp_path / "sample_submission.csv",
            submission_format_path=format_path,
        ).name
        == "submission.tar.zst"
    )
