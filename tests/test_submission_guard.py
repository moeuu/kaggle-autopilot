from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from kagglebot.exceptions import SubmissionCliError, SubmissionValidationError
from kagglebot.exec_utils import CommandResult
from kagglebot.submission.guard import (
    classify_submit_error,
    compute_error_fingerprint,
    normalize_error_text,
    run_kaggle_submit,
)
from kagglebot.submission.validate import validate_submission


def _write_sample_and_submission(tmp_path: Path) -> tuple[Path, Path]:
    sample = tmp_path / "sample_submission.csv"
    submission = tmp_path / "submission.csv"
    pd.DataFrame({"id": [1, 2, 3], "target": [0.0, 0.0, 0.0]}).to_csv(sample, index=False)
    pd.DataFrame({"id": [1, 2, 3], "target": [0.1, 0.2, 0.3]}).to_csv(submission, index=False)
    return sample, submission


def test_validate_submission_columns_mismatch(tmp_path: Path) -> None:
    sample, submission = _write_sample_and_submission(tmp_path)
    pd.DataFrame({"id": [1, 2, 3], "score": [0.1, 0.2, 0.3]}).to_csv(submission, index=False)
    with pytest.raises(SubmissionValidationError, match="columns do not match"):
        validate_submission(submission, sample)


def test_validate_submission_row_count_mismatch(tmp_path: Path) -> None:
    sample, submission = _write_sample_and_submission(tmp_path)
    pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]}).to_csv(submission, index=False)
    with pytest.raises(SubmissionValidationError, match="row count does not match"):
        validate_submission(submission, sample)


def test_validate_submission_id_nan(tmp_path: Path) -> None:
    sample, submission = _write_sample_and_submission(tmp_path)
    pd.DataFrame({"id": [1, None, 3], "target": [0.1, 0.2, 0.3]}).to_csv(submission, index=False)
    with pytest.raises(SubmissionValidationError, match="id column 'id' contains NaN"):
        validate_submission(submission, sample)


def test_validate_submission_id_duplicate(tmp_path: Path) -> None:
    sample, submission = _write_sample_and_submission(tmp_path)
    pd.DataFrame({"id": [1, 1, 3], "target": [0.1, 0.2, 0.3]}).to_csv(submission, index=False)
    with pytest.raises(SubmissionValidationError, match="contains duplicates"):
        validate_submission(submission, sample)


def test_validate_submission_pred_nan_or_non_numeric(tmp_path: Path) -> None:
    sample, submission = _write_sample_and_submission(tmp_path)
    pd.DataFrame({"id": [1, 2, 3], "target": [0.1, "abc", 0.3]}).to_csv(submission, index=False)
    with pytest.raises(SubmissionValidationError, match="NaN or non-numeric"):
        validate_submission(submission, sample)


def test_validate_submission_pred_inf(tmp_path: Path) -> None:
    sample, submission = _write_sample_and_submission(tmp_path)
    pd.DataFrame({"id": [1, 2, 3], "target": [0.1, float("inf"), 0.3]}).to_csv(submission, index=False)
    with pytest.raises(SubmissionValidationError, match="contains inf"):
        validate_submission(submission, sample)


def test_classify_submit_error_permanent() -> None:
    classified = classify_submit_error("", "You must accept the rules before submitting", 1)
    assert classified.kind == "permanent"
    assert classified.reason == "rules_not_accepted"


def test_classify_submit_error_transient() -> None:
    classified = classify_submit_error("", "ConnectionError: temporarily unavailable (503)", 1)
    assert classified.kind == "transient"


def test_normalize_and_fingerprint_are_stable() -> None:
    a = "Error at /home/user/repo/artifacts/demo/runs/20260101T000000Z-abcd1234: timeout 2026-02-15T12:00:00Z"
    b = "Error at /home/other/repo/artifacts/demo/runs/20260101T000000Z-efef2222: timeout 2026-02-16T12:00:00Z"
    na = normalize_error_text(a)
    normalize_error_text(b)
    assert "<PATH>" in na or "<ARTIFACT_PATH>" in na
    assert compute_error_fingerprint(a, "") == compute_error_fingerprint(b, "")


def test_run_kaggle_submit_raises_submission_cli_error(monkeypatch) -> None:
    def fake_run_command(*args, **kwargs):  # noqa: ARG001
        return CommandResult(
            args=["kaggle"],
            returncode=2,
            stdout="out",
            stderr="err",
            duration_sec=0.1,
        )

    monkeypatch.setattr("kagglebot.submission.guard.run_command", fake_run_command)
    with pytest.raises(SubmissionCliError) as exc:
        run_kaggle_submit(slug="demo", submission_file=Path("submission.csv"), message="m")
    assert exc.value.exit_code == 2
    assert exc.value.stdout == "out"
    assert exc.value.stderr == "err"
