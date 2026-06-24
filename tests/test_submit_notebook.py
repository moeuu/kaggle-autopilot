from __future__ import annotations

from pathlib import Path

from kagglebot.submit_notebook import (
    build_kaggle_submit_kernel_kwargs,
    build_notebook_submit_reference,
    normalize_notebook_submit_artifact_mode,
)


def test_normalize_notebook_submit_artifact_mode_defaults_to_wrapper() -> None:
    assert normalize_notebook_submit_artifact_mode(None) == "wrapper"
    assert normalize_notebook_submit_artifact_mode("") == "wrapper"
    assert normalize_notebook_submit_artifact_mode(" Inference ") == "inference"


def test_build_notebook_submit_reference_prefers_copied_artifact_path() -> None:
    reference = build_notebook_submit_reference(
        kernel_id="user/demo",
        submission_artifact_path=Path("/tmp/copied/submission-fixed.csv"),
        kernel_submission_path=Path("/kaggle/working/submission.csv"),
        version_label="7",
    )

    assert reference.kernel_ref == "user/demo"
    assert reference.submission_ref == "kernel:user/demo"
    assert reference.output_file == "submission-fixed.csv"
    assert reference.version == "7"


def test_build_notebook_submit_reference_uses_kernel_path_and_default_version() -> None:
    reference = build_notebook_submit_reference(
        kernel_id="user/demo",
        submission_artifact_path=None,
        kernel_submission_path=Path("/kaggle/working/submission.csv"),
        version_label=None,
    )

    assert reference.output_file == "submission.csv"
    assert reference.version == "1"


def test_build_kaggle_submit_kernel_kwargs_uses_reference_fields() -> None:
    reference = build_notebook_submit_reference(
        kernel_id="user/demo",
        submission_artifact_path=Path("/tmp/submission.csv"),
        kernel_submission_path=None,
        version_label="3",
    )

    assert build_kaggle_submit_kernel_kwargs(
        slug="demo-competition",
        reference=reference,
        message="submit message",
        dry_run=True,
    ) == {
        "slug": "demo-competition",
        "kernel": "user/demo",
        "message": "submit message",
        "output_file": "submission.csv",
        "version": "3",
        "dry_run": True,
    }
