from __future__ import annotations

from kagglebot.submit_attempts import build_submit_attempt_payload, build_submit_run_state_update


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
