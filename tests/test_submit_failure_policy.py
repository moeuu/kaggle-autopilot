from __future__ import annotations

from kagglebot.submit_failure_policy import (
    SUBMIT_FAILURE_REPAIR_TARGET_MANUAL,
    SUBMIT_FAILURE_REPAIR_TARGET_PLATFORM,
    SUBMIT_FAILURE_REPAIR_TARGET_SUBMISSION_ARTIFACT,
    SUBMIT_FAILURE_REPAIR_TARGET_SUBMIT_MODE,
    classify_submit_failure_repair,
    normalize_loaded_submit_failure_context,
    should_retry_ambiguous_notebook_submit_error,
    should_use_notebook_submit_fallback,
    submit_error_requires_file_fix,
)


def test_classify_submit_failure_repair_treats_submission_limit_as_manual() -> None:
    decision = classify_submit_failure_repair(
        reason="bad_request",
        error_kind="permanent",
        detail="You have reached the maximum number of submissions for this competition.",
    )

    assert decision.repair_target == SUBMIT_FAILURE_REPAIR_TARGET_MANUAL
    assert decision.repairable is False
    assert "submission limit" in decision.manual_next_step.lower()


def test_classify_submit_failure_repair_treats_daily_allowance_as_manual() -> None:
    decision = classify_submit_failure_repair(
        reason="submission_limit",
        error_kind="permanent",
        detail="Submission not allowed: Your team has used its daily Submission allowance (10) today.",
    )

    assert decision.repair_target == SUBMIT_FAILURE_REPAIR_TARGET_MANUAL
    assert decision.repairable is False
    assert "submission limit" in decision.manual_next_step.lower()


def test_classify_submit_failure_repair_detects_scoring_file_issue() -> None:
    decision = classify_submit_failure_repair(
        reason="submission_poll_status_complete_no_score",
        error_kind="validation",
        detail="Kaggle scoring error inferred: invalid submission file with row count mismatch.",
    )

    assert decision.repair_target == SUBMIT_FAILURE_REPAIR_TARGET_SUBMISSION_ARTIFACT
    assert decision.repairable is True
    assert decision.manual_next_step == ""


def test_classify_submit_failure_repair_routes_notebook_mode_errors() -> None:
    decision = classify_submit_failure_repair(
        reason="notebook_only_submission_required",
        error_kind="permanent",
        detail="Only accepts submissions from notebooks.",
    )

    assert decision.repair_target == SUBMIT_FAILURE_REPAIR_TARGET_SUBMIT_MODE
    assert decision.repairable is True


def test_classify_submit_failure_repair_routes_notebook_submit_argument_errors() -> None:
    decision = classify_submit_failure_repair(
        reason="notebook_submit_argument_missing",
        error_kind="permanent",
        detail="Code competition submissions require both the output file name and the version label.",
    )

    assert decision.repair_target == SUBMIT_FAILURE_REPAIR_TARGET_SUBMIT_MODE
    assert decision.repairable is True


def test_classify_submit_failure_repair_routes_polling_without_file_hint_to_platform() -> None:
    decision = classify_submit_failure_repair(
        reason="submission_poll_status_error",
        error_kind="validation",
        detail="Kaggle submission status: error",
    )

    assert decision.repair_target == SUBMIT_FAILURE_REPAIR_TARGET_PLATFORM
    assert decision.repairable is True


def test_submit_error_requires_file_fix_for_local_validation() -> None:
    assert submit_error_requires_file_fix(
        reason="local_submission_validation_failed",
        error_kind="validation",
        detail="",
    )


def test_notebook_submit_fallback_requires_clear_hint() -> None:
    assert should_use_notebook_submit_fallback(
        reason="notebook_submit_argument_missing",
        stdout="",
        stderr="",
    )
    assert should_use_notebook_submit_fallback(
        reason="bad_request",
        stdout="",
        stderr="Code competition submissions require both the output file name and the version label.",
    )
    assert not should_use_notebook_submit_fallback(reason="bad_request", stdout="", stderr="generic 400")


def test_ambiguous_notebook_retry_requires_clear_hint() -> None:
    assert should_retry_ambiguous_notebook_submit_error(
        reason="ambiguous_notebook_bad_request",
        stdout="",
        stderr="kernel must be specified as <owner>/<notebook>",
    )
    assert not should_retry_ambiguous_notebook_submit_error(
        reason="ambiguous_notebook_bad_request",
        stdout="",
        stderr="generic bad request",
    )


def test_normalize_loaded_submit_failure_context_backfills_manual_blocker() -> None:
    payload = {
        "reason": "ambiguous_notebook_bad_request",
        "repair_target": "submit_mode_or_kernel",
        "repairable": True,
        "stderr_tail": "400 Client Error: Bad Request",
    }

    normalized = normalize_loaded_submit_failure_context(payload)

    assert normalized["repair_target"] == SUBMIT_FAILURE_REPAIR_TARGET_MANUAL
    assert normalized["repairable"] is False
    assert "submit-notebook 400" in str(normalized["manual_next_step"])
