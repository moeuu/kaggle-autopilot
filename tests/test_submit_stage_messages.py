from __future__ import annotations

from kagglebot.submit_stage_messages import (
    extract_submission_row_message,
    format_competition_faithfulness_detail,
    format_iteration_submit_status_message,
)


def test_format_iteration_submit_status_message_handles_disabled_allowed_and_blocked() -> None:
    assert (
        format_iteration_submit_status_message(
            iteration=1,
            max_iterations=3,
            submit_enabled=False,
            submit_allowed_by_gate=False,
            submit_phase_state="disabled",
            quality_reasons=[],
        )
        is None
    )
    assert (
        format_iteration_submit_status_message(
            iteration=1,
            max_iterations=3,
            submit_enabled=True,
            submit_allowed_by_gate=True,
            submit_phase_state="ready",
            quality_reasons=[],
        )
        == "[cyan]submit[/cyan]: iter 1/3 attempting submission now."
    )

    blocked = format_iteration_submit_status_message(
        iteration=2,
        max_iterations=3,
        submit_enabled=True,
        submit_allowed_by_gate=False,
        submit_phase_state="blocked_quality_guard",
        quality_reasons=["collapsed_predictions", "weak_cv"],
        competition_faithfulness={
            "expected_metric": "logloss",
            "actual_metric": "accuracy",
            "expected_split_strategy": "group_kfold",
            "actual_split_strategy": "kfold",
            "dataset_mode": "sample",
        },
    )

    assert blocked == (
        "[cyan]submit[/cyan]: iter 2/3 not attempted yet "
        "(state=blocked_quality_guard reasons=collapsed_predictions,weak_cv "
        "metric=accuracy/logloss split=kfold/group_kfold dataset_mode=sample)."
    )


def test_format_competition_faithfulness_detail_uses_unknown_for_missing_sides() -> None:
    assert (
        format_competition_faithfulness_detail(
            {
                "expected_metric": "rmse",
                "actual_split_strategy": "kfold",
            }
        )
        == " metric=unknown/rmse split=kfold/unknown"
    )


def test_extract_submission_row_message_prefers_kaggle_error_fields() -> None:
    assert (
        extract_submission_row_message(
            {
                "message": "generic",
                "failureReason": " row count mismatch ",
            }
        )
        == "row count mismatch"
    )
    assert extract_submission_row_message({"message": " fallback "}) == "fallback"
    assert extract_submission_row_message({"comments": ""}) == ""
