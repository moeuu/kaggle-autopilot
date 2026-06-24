from __future__ import annotations

from kagglebot.submit_attempts import (
    build_submit_attempt_payload,
    build_submit_knowledge_payload,
    build_submit_result_payload,
    build_submit_run_state_update,
)


def test_build_submit_attempt_payload_sets_required_fields_and_tails() -> None:
    payload = build_submit_attempt_payload(
        run_id="run-1",
        submission_ref="submission.csv",
        submission_sha256="sha",
        exit_code=1,
        ok=False,
        fingerprint="fp",
        error_kind="transient",
        action_taken="retry",
        reason="network_or_timeout",
        stdout="abcdef",
        stderr="uvwxyz",
        stdout_tail_chars=3,
        stderr_tail_chars=4,
    )

    assert payload == {
        "run_id": "run-1",
        "sub_path": "submission.csv",
        "sub_sha256": "sha",
        "exit_code": 1,
        "ok": False,
        "fingerprint": "fp",
        "error_kind": "transient",
        "action_taken": "retry",
        "reason": "network_or_timeout",
        "stdout_tail": "def",
        "stderr_tail": "wxyz",
    }


def test_build_submit_attempt_payload_includes_optional_fields() -> None:
    payload = build_submit_attempt_payload(
        run_id="run-1",
        submission_ref="submission.csv",
        submission_sha256=None,
        exit_code=None,
        ok=False,
        fingerprint="fp",
        error_kind="none",
        action_taken="skip",
        reason="duplicate_submission_sha_seen",
        stdout="",
        stderr="",
        stdout_tail_chars=100,
        stderr_tail_chars=100,
        code_fingerprint="code-fp",
        extra={"duplicate_sources": ["run_attempts"]},
    )

    assert payload["code_fingerprint"] == "code-fp"
    assert payload["duplicate_sources"] == ["run_attempts"]
    assert payload["sub_sha256"] is None


def test_build_submit_run_state_update_sets_common_last_submit_fields() -> None:
    update = build_submit_run_state_update(
        prior_state={"submit_attempts_count": 4},
        fingerprint="fp",
        code_fingerprint="code-fp",
        error_kind="none",
        action_taken="submit",
        reason="submitted",
        submission_ref="submission.csv",
        submit_ok=True,
    )

    assert update == {
        "submit_attempted": True,
        "submit_ok": True,
        "last_submit_fingerprint": "fp",
        "last_fingerprint": "fp",
        "last_submit_code_fingerprint": "code-fp",
        "last_error_kind": "none",
        "last_action": "submit",
        "last_reason": "submitted",
        "last_submission_path": "submission.csv",
        "submit_attempts_count": 5,
    }


def test_build_submit_run_state_update_can_preserve_skip_without_submit_ok() -> None:
    update = build_submit_run_state_update(
        prior_state={},
        fingerprint="fp",
        code_fingerprint="code-fp",
        error_kind="none",
        action_taken="skip",
        reason="duplicate_submission_sha_seen",
        submission_ref="submission.csv",
        submission_sha256="sha",
    )

    assert "submit_ok" not in update
    assert update["last_submission_sha256"] == "sha"
    assert update["submit_attempts_count"] == 1


def test_build_submit_knowledge_payload_formats_error_and_fix_summary() -> None:
    payload = build_submit_knowledge_payload(
        iteration=None,
        error_kind="transient",
        reason="network_or_timeout",
        action_taken="retry",
        fingerprint="fp",
        details="  first line\nsecond line  ",
        normalize_detail=lambda text, max_chars: " ".join(str(text).split())[:max_chars],
    )

    assert payload.iteration == 1
    assert payload.error_message == "submit_error kind=transient reason=network_or_timeout fingerprint=fp"
    assert payload.fix_summary == "submit_action=retry; detail=first line second line"


def test_build_submit_knowledge_payload_preserves_explicit_iteration() -> None:
    payload = build_submit_knowledge_payload(
        iteration=3,
        error_kind="validation",
        reason="local_submission_validation_failed",
        action_taken="abort",
        fingerprint="fp",
        details="bad submission",
        normalize_detail=lambda text, max_chars: str(text)[:max_chars],
    )

    assert payload.iteration == 3


def test_build_submit_result_payload_for_success_includes_outcome() -> None:
    payload = build_submit_result_payload(
        message="submit message",
        submission_ref="submission.csv",
        submitted_at_iso="2026-06-25T00:00:00+00:00",
        iteration=2,
        outcome={"status": "complete", "score": 0.42},
    )

    assert payload == {
        "message": "submit message",
        "submission_path": "submission.csv",
        "submitted_at": "2026-06-25T00:00:00+00:00",
        "iteration": 2,
        "outcome": {"status": "complete", "score": 0.42},
    }


def test_build_submit_result_payload_for_duplicate_skip_omits_outcome() -> None:
    payload = build_submit_result_payload(
        message="submit message",
        submission_ref="submission.csv",
        submitted_at_iso="2026-06-25T00:00:00+00:00",
        iteration=2,
        skipped=True,
        reason="duplicate_submission_sha_seen",
        duplicate_sources=["run_attempts"],
    )

    assert payload == {
        "message": "submit message",
        "submission_path": "submission.csv",
        "submitted_at": "2026-06-25T00:00:00+00:00",
        "iteration": 2,
        "skipped": True,
        "reason": "duplicate_submission_sha_seen",
        "duplicate_sources": ["run_attempts"],
    }
