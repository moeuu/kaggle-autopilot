from __future__ import annotations

import json
from pathlib import Path

from kagglebot.submit_failure_context import (
    format_submit_autofix_context,
    load_submit_failure_context,
    mark_submit_failure_context_resolved,
    path_from_submit_reference,
    save_submit_failure_context,
    submit_failure_context_path,
)


def test_load_submit_failure_context_normalizes_stale_manual_blocker(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    submit_failure_context_path(run_dir).write_text(
        json.dumps(
            {
                "active": True,
                "reason": "ambiguous_notebook_bad_request",
                "repair_target": "submit_mode_or_kernel",
                "repairable": True,
                "stderr_tail": "400 Client Error: Bad Request",
            }
        ),
        encoding="utf-8",
    )

    payload = load_submit_failure_context(run_dir)

    assert payload["repair_target"] == "manual_intervention"
    assert payload["repairable"] is False
    assert "submit-notebook 400" in str(payload["manual_next_step"])


def test_save_and_mark_submit_failure_context_resolved(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    save_submit_failure_context(
        run_dir,
        {
            "active": True,
            "reason": "local_submission_validation_failed",
            "repair_target": "submission_artifact",
        },
    )

    mark_submit_failure_context_resolved(run_dir=run_dir, resolution="submitted", submission_ref="kernel:user/demo")
    payload = json.loads(submit_failure_context_path(run_dir).read_text(encoding="utf-8"))

    assert payload["active"] is False
    assert payload["resolution"] == "submitted"
    assert payload["resolved_submission_ref"] == "kernel:user/demo"
    assert payload["resolved_at"]


def test_path_from_submit_reference_ignores_kernel_references() -> None:
    assert path_from_submit_reference("kernel:user/demo") is None
    assert path_from_submit_reference("") is None
    assert path_from_submit_reference("/tmp/submission.csv") == Path("/tmp/submission.csv")


def test_format_submit_autofix_context_includes_failure_state_and_latest_attempt() -> None:
    text = format_submit_autofix_context(
        failure_context={
            "active": True,
            "repair_target": "submission_artifact",
            "repairable": True,
            "reason": "submission_poll_status_error",
            "summary": "Kaggle reported row count mismatch",
            "latest_submit_attempt": {
                "ts": "2026-06-25T00:00:00+00:00",
                "ok": False,
                "reason": "submission_poll_status_error",
                "sub_path": "/tmp/bad.csv",
            },
            "run_state_excerpt": {
                "submit_attempted": True,
                "submit_ok": False,
                "last_reason": "submission_poll_status_error",
            },
        },
        run_state={
            "submit_attempted": True,
            "submit_ok": False,
            "last_reason": "submission_poll_status_error",
            "submit_autofix_submission_path": "/tmp/fixed.csv",
        },
        latest_submit_attempt={
            "ts": "2026-06-25T00:00:01+00:00",
            "ok": False,
            "reason": "submission_poll_status_error",
            "stderr_tail": "Submission validation failed: prediction column contains NaN",
        },
    )

    assert "submit_failure_context:" in text
    assert "repair_target: submission_artifact" in text
    assert "failure_context_latest_submit_attempt:" in text
    assert "failure_context_run_state:" in text
    assert "run_state:" in text
    assert "latest_submit_attempt:" in text
    assert "stderr_tail: Submission validation failed:" in text
