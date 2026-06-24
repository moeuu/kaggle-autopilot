from __future__ import annotations

from kagglebot.submit_attempts import build_submit_attempt_payload


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
