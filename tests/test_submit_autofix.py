from __future__ import annotations

from pathlib import Path

from kagglebot.exceptions import SubmissionValidationError
from kagglebot.submit_autofix import (
    prepare_submit_file_autofix,
    prepare_submit_file_autofix_for_run,
    submit_file_fix_required_for_attempt,
)


def _file_fix_attempt() -> dict[str, object]:
    return {
        "reason": "submission_poll_status_error",
        "error_kind": "validation",
        "stderr_tail": "Kaggle reported: submission file row count mismatch",
    }


def test_submit_file_fix_required_for_attempt_detects_file_issue() -> None:
    assert submit_file_fix_required_for_attempt(_file_fix_attempt())
    assert not submit_file_fix_required_for_attempt(
        {
            "reason": "submission_poll_status_error",
            "error_kind": "validation",
            "stderr_tail": "Kaggle submission status: error",
        }
    )


def test_prepare_submit_file_autofix_returns_empty_when_not_required(tmp_path: Path) -> None:
    result = prepare_submit_file_autofix(
        latest_submit_attempt={"reason": "bad_request", "error_kind": "permanent", "stderr_tail": "generic"},
        resolve_source=lambda: tmp_path / "submission.csv",
        validate_and_prepare=lambda path: path,
        save_repaired_path=lambda path: None,
    )

    assert result.path is None
    assert result.summary == ""
    assert result.file_fix_required is False


def test_prepare_submit_file_autofix_records_repaired_path(tmp_path: Path) -> None:
    source = tmp_path / "submission.csv"
    fixed = tmp_path / "submission-fixed.csv"
    source.write_text("id,target\n1,bad\n", encoding="utf-8")
    fixed.write_text("id,target\n1,0.2\n", encoding="utf-8")
    saved: list[Path] = []

    result = prepare_submit_file_autofix(
        latest_submit_attempt=_file_fix_attempt(),
        resolve_source=lambda: source,
        validate_and_prepare=lambda path: fixed,
        save_repaired_path=saved.append,
    )

    assert result.path == fixed
    assert "fixed_submission_path" in result.summary
    assert saved == [fixed]
    assert result.file_fix_required is True


def test_prepare_submit_file_autofix_reports_missing_source() -> None:
    result = prepare_submit_file_autofix(
        latest_submit_attempt=_file_fix_attempt(),
        resolve_source=lambda: None,
        validate_and_prepare=lambda path: path,
        save_repaired_path=lambda path: None,
    )

    assert result.path is None
    assert "could not locate" in result.summary
    assert result.file_fix_required is True


def test_prepare_submit_file_autofix_reports_validation_failure(tmp_path: Path) -> None:
    source = tmp_path / "submission.csv"
    source.write_text("id,target\n1,bad\n", encoding="utf-8")

    def fail_validation(path: Path) -> Path:
        raise SubmissionValidationError("prediction column contains NaN")

    result = prepare_submit_file_autofix(
        latest_submit_attempt=_file_fix_attempt(),
        resolve_source=lambda: source,
        validate_and_prepare=fail_validation,
        save_repaired_path=lambda path: None,
    )

    assert result.path is None
    assert "prediction column contains NaN" in result.summary
    assert result.file_fix_required is True


def test_prepare_submit_file_autofix_for_run_resolves_iteration_fallback(tmp_path: Path) -> None:
    iter_dir = tmp_path / "iter-1"
    source = iter_dir / "submission.csv"
    fixed = iter_dir / "submission-fixed.csv"
    source.parent.mkdir(parents=True)
    source.write_text("id,target\n1,bad\n", encoding="utf-8")
    fixed.write_text("id,target\n1,0.2\n", encoding="utf-8")
    saved: list[Path] = []

    result = prepare_submit_file_autofix_for_run(
        latest_submit_attempt=_file_fix_attempt(),
        run_state={},
        failure_context={},
        fallback_iteration_dirs=lambda: [iter_dir],
        resolve_iteration_submission_artifact=lambda path: path / "submission.csv",
        validate_and_prepare=lambda path: fixed if path == source else path,
        save_repaired_path=saved.append,
    )

    assert result.path == fixed
    assert "fixed_submission_path" in result.summary
    assert saved == [fixed]
