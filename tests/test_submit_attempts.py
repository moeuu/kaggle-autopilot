from __future__ import annotations

import json

from kagglebot.submit_attempts import (
    append_submit_attempt,
    build_submit_abort_record_payloads,
    build_submit_attempt_payload,
    build_submit_knowledge_payload,
    build_submit_result_payload,
    build_submit_retry_attempt_payload,
    build_submit_run_state_update,
    build_submit_skip_attempt_payload,
    build_submit_skip_record_payloads,
    build_submit_success_record_payloads,
    decide_submit_outcome_recording,
    submit_attempt_sha_seen,
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


def test_append_submit_attempt_writes_jsonl_record(tmp_path) -> None:
    append_submit_attempt(
        run_dir=tmp_path,
        payload={"run_id": "run-1", "sub_sha256": "sha", "ok": False},
        now_iso="2026-06-25T00:00:00+00:00",
    )

    attempts_path = tmp_path / "submit_attempts.jsonl"
    rows = [json.loads(line) for line in attempts_path.read_text(encoding="utf-8").splitlines()]

    assert rows == [{"ts": "2026-06-25T00:00:00+00:00", "run_id": "run-1", "sub_sha256": "sha", "ok": False}]


def test_submit_attempt_sha_seen_ignores_invalid_rows(tmp_path) -> None:
    (tmp_path / "submit_attempts.jsonl").write_text(
        "\n".join(
            [
                "",
                "not-json",
                json.dumps(["not", "a", "dict"]),
                json.dumps({"sub_sha256": "other"}),
                json.dumps({"sub_sha256": "sha"}),
            ]
        ),
        encoding="utf-8",
    )

    assert submit_attempt_sha_seen(run_dir=tmp_path, submission_sha="sha")
    assert not submit_attempt_sha_seen(run_dir=tmp_path, submission_sha="missing")
    assert not submit_attempt_sha_seen(run_dir=tmp_path, submission_sha="")


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


def test_build_submit_success_record_payloads_combines_attempt_and_run_state() -> None:
    payloads = build_submit_success_record_payloads(
        run_id="run-1",
        submission_ref="submission.csv",
        submission_sha256="sha",
        exit_code=0,
        fingerprint="fp",
        code_fingerprint="code-fp",
        stdout="abcdef",
        stderr="uvwxyz",
        prior_state={"submit_attempts_count": 2},
        stdout_tail_chars=3,
        stderr_tail_chars=4,
    )

    assert payloads.attempt_payload == {
        "run_id": "run-1",
        "sub_path": "submission.csv",
        "sub_sha256": "sha",
        "exit_code": 0,
        "ok": True,
        "fingerprint": "fp",
        "error_kind": "none",
        "action_taken": "submit",
        "reason": "submitted",
        "stdout_tail": "def",
        "stderr_tail": "wxyz",
    }
    assert payloads.run_state_update == {
        "submit_attempted": True,
        "submit_ok": True,
        "last_submit_fingerprint": "fp",
        "last_fingerprint": "fp",
        "last_submit_code_fingerprint": "code-fp",
        "last_error_kind": "none",
        "last_action": "submit",
        "last_reason": "submitted",
        "last_submission_path": "submission.csv",
        "submit_attempts_count": 3,
    }


def test_build_submit_abort_record_payloads_combines_attempt_and_run_state() -> None:
    payloads = build_submit_abort_record_payloads(
        run_id="run-1",
        submission_ref="submission.csv",
        submission_sha256="sha",
        exit_code=2,
        fingerprint="fp",
        code_fingerprint="code-fp",
        error_kind="validation",
        reason="local_submission_validation_failed",
        stdout="abcdef",
        stderr="uvwxyz",
        prior_state={"submit_attempts_count": 5},
        prior_submit_ok=False,
        stdout_tail_chars=3,
        stderr_tail_chars=4,
    )

    assert payloads.attempt_payload == {
        "run_id": "run-1",
        "sub_path": "submission.csv",
        "sub_sha256": "sha",
        "exit_code": 2,
        "ok": False,
        "fingerprint": "fp",
        "error_kind": "validation",
        "action_taken": "abort",
        "reason": "local_submission_validation_failed",
        "stdout_tail": "def",
        "stderr_tail": "wxyz",
        "code_fingerprint": "code-fp",
    }
    assert payloads.run_state_update == {
        "submit_attempted": True,
        "submit_ok": False,
        "last_submit_fingerprint": "fp",
        "last_fingerprint": "fp",
        "last_submit_code_fingerprint": "code-fp",
        "last_error_kind": "validation",
        "last_action": "abort",
        "last_reason": "local_submission_validation_failed",
        "last_submission_path": "submission.csv",
        "submit_attempts_count": 6,
    }


def test_build_submit_skip_attempt_payload_sets_skip_contract() -> None:
    payload = build_submit_skip_attempt_payload(
        run_id="run-1",
        submission_ref="submission.csv",
        submission_sha256="sha",
        fingerprint="fp",
        error_kind="unknown",
        reason="same_submission_path",
        stdout_tail_chars=3,
        stderr_tail_chars=4,
    )

    assert payload == {
        "run_id": "run-1",
        "sub_path": "submission.csv",
        "sub_sha256": "sha",
        "exit_code": None,
        "ok": False,
        "fingerprint": "fp",
        "error_kind": "unknown",
        "action_taken": "skip",
        "reason": "same_submission_path",
        "stdout_tail": "",
        "stderr_tail": "",
    }


def test_build_submit_skip_record_payloads_combines_attempt_and_run_state() -> None:
    payloads = build_submit_skip_record_payloads(
        run_id="run-1",
        submission_ref="submission.csv",
        submission_sha256="sha",
        fingerprint="fp",
        code_fingerprint="code-fp",
        error_kind="none",
        reason="duplicate_submission_sha_seen",
        prior_state={"submit_attempts_count": 7},
        stdout_tail_chars=3,
        stderr_tail_chars=4,
        duplicate_sources=["run_attempts"],
    )

    assert payloads.attempt_payload == {
        "run_id": "run-1",
        "sub_path": "submission.csv",
        "sub_sha256": "sha",
        "exit_code": None,
        "ok": False,
        "fingerprint": "fp",
        "error_kind": "none",
        "action_taken": "skip",
        "reason": "duplicate_submission_sha_seen",
        "stdout_tail": "",
        "stderr_tail": "",
        "duplicate_sources": ["run_attempts"],
    }
    assert payloads.run_state_update == {
        "submit_attempted": True,
        "last_submit_fingerprint": "fp",
        "last_fingerprint": "fp",
        "last_submit_code_fingerprint": "code-fp",
        "last_error_kind": "none",
        "last_action": "skip",
        "last_reason": "duplicate_submission_sha_seen",
        "last_submission_path": "submission.csv",
        "submit_attempts_count": 8,
        "last_submission_sha256": "sha",
    }


def test_build_submit_retry_attempt_payload_sets_retry_contract() -> None:
    payload = build_submit_retry_attempt_payload(
        run_id="run-1",
        submission_ref="submission.csv",
        submission_sha256="sha",
        exit_code=1,
        fingerprint="fp",
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


def test_decide_submit_outcome_recording_formats_scored_result() -> None:
    decision = decide_submit_outcome_recording(
        outcome={"status": "complete", "score": "0.42"},
        submission_artifact_exists=True,
    )

    assert decision.message == "[cyan]submission result[/cyan]: status=complete score=0.420000"
    assert decision.ledger_outcome == {"status": "complete", "score": "0.42"}


def test_decide_submit_outcome_recording_skips_ledger_without_artifact() -> None:
    decision = decide_submit_outcome_recording(
        outcome={"status": "complete", "score": 0.42},
        submission_artifact_exists=False,
    )

    assert "score=0.420000" in decision.message
    assert decision.ledger_outcome is None


def test_decide_submit_outcome_recording_handles_scoreless_or_missing_outcome() -> None:
    scoreless = decide_submit_outcome_recording(
        outcome={"status": "complete", "score": None},
        submission_artifact_exists=True,
    )
    missing = decide_submit_outcome_recording(outcome=None, submission_artifact_exists=True)

    assert scoreless.message == "[yellow]submission result[/yellow]: score not available yet; knowledge update skipped"
    assert scoreless.ledger_outcome == {"status": "complete", "score": None}
    assert missing.message == "[yellow]submission result[/yellow]: score not available yet; knowledge update skipped"
    assert missing.ledger_outcome is None
