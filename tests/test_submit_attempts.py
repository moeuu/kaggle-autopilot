from __future__ import annotations

import json
from pathlib import Path

from kagglebot.submit_attempts import (
    SubmitAttemptRecorder,
    SubmitAttemptStatePayloads,
    append_submit_attempt,
    build_same_submission_path_skip_attempt_payload,
    build_seen_submit_fingerprint_set,
    build_submit_abort_record_payloads,
    build_submit_attempt_payload,
    build_submit_knowledge_payload,
    build_submit_result_payload,
    build_submit_retry_attempt_payload,
    build_submit_run_state_update,
    build_submit_skip_attempt_payload,
    build_submit_skip_record_payloads,
    build_submit_success_record_payloads,
    count_successful_submit_attempts,
    decide_submit_outcome_recording,
    format_submit_retry_knowledge_details,
    has_submit_attempt_records,
    has_successful_submit_attempt,
    load_latest_submit_attempt,
    load_submit_attempt_rows,
    load_submit_fingerprints,
    record_submit_reason_knowledge,
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


def test_submit_attempt_recorder_appends_attempt_and_saves_state(tmp_path) -> None:
    saved_updates: list[dict[str, object]] = []

    recorder = SubmitAttemptRecorder(
        run_dir=tmp_path,
        save_run_state=saved_updates.append,
    )
    recorder.record_payloads(
        SubmitAttemptStatePayloads(
            attempt_payload={"run_id": "run-1", "ok": True},
            run_state_update={"submit_attempted": True, "submit_ok": True},
        )
    )

    rows = load_submit_attempt_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["run_id"] == "run-1"
    assert rows[0]["ok"] is True
    assert "ts" in rows[0]
    assert saved_updates == [{"submit_attempted": True, "submit_ok": True}]


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


def test_load_submit_attempt_rows_filters_invalid_jsonl_rows(tmp_path) -> None:
    (tmp_path / "submit_attempts.jsonl").write_text(
        "\n".join(
            [
                "",
                "not-json",
                json.dumps(["not", "a", "dict"]),
                json.dumps({"ok": False, "fingerprint": "fp-1"}),
                json.dumps({"ok": True, "action_taken": "submit", "fingerprint": "fp-2"}),
                json.dumps({"ok": True, "action_taken": "skip", "fingerprint": "fp-3"}),
            ]
        ),
        encoding="utf-8",
    )

    rows = load_submit_attempt_rows(tmp_path)

    assert rows == [
        {"ok": False, "fingerprint": "fp-1"},
        {"ok": True, "action_taken": "submit", "fingerprint": "fp-2"},
        {"ok": True, "action_taken": "skip", "fingerprint": "fp-3"},
    ]
    assert has_submit_attempt_records(tmp_path)
    assert has_successful_submit_attempt(tmp_path)
    assert count_successful_submit_attempts(tmp_path) == 1
    assert load_submit_fingerprints(tmp_path) == ["fp-1", "fp-2", "fp-3"]
    assert load_latest_submit_attempt(tmp_path) == {"ok": True, "action_taken": "skip", "fingerprint": "fp-3"}


def test_submit_attempt_readers_return_empty_defaults_for_missing_file(tmp_path) -> None:
    assert load_submit_attempt_rows(tmp_path) == []
    assert not has_submit_attempt_records(tmp_path)
    assert not has_successful_submit_attempt(tmp_path)
    assert count_successful_submit_attempts(tmp_path) == 0
    assert load_submit_fingerprints(tmp_path) == []
    assert load_latest_submit_attempt(tmp_path) == {}


def test_build_seen_submit_fingerprint_set_merges_attempts_and_run_state() -> None:
    assert build_seen_submit_fingerprint_set(
        attempt_fingerprints=[" fp-1 ", "", "fp-2", "fp-1"],
        run_state={"last_fingerprint": "legacy-fp"},
    ) == {"fp-1", "fp-2", "legacy-fp"}

    assert build_seen_submit_fingerprint_set(
        attempt_fingerprints=[],
        run_state={"last_submit_fingerprint": "submit-fp", "last_fingerprint": "legacy-fp"},
    ) == {"submit-fp"}


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


def test_build_same_submission_path_skip_attempt_payload_sets_unknown_error_kind() -> None:
    payload = build_same_submission_path_skip_attempt_payload(
        run_id="run-1",
        submission_ref="submission.csv",
        submission_sha256="sha",
        fingerprint="fp",
        reason="same_submission_path_reused_in_run",
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
        "reason": "same_submission_path_reused_in_run",
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


def test_format_submit_retry_knowledge_details_formats_attempt_and_wait() -> None:
    assert format_submit_retry_knowledge_details(attempt=2, wait_seconds=3.456) == "attempt=2; wait=3.5s"


def test_record_submit_reason_knowledge_records_payload() -> None:
    calls: list[dict[str, object]] = []
    knowledge_paths = object()

    recorded = record_submit_reason_knowledge(
        knowledge_paths=knowledge_paths,
        slug="demo",
        run_id="run-1",
        problem_types=["tabular"],
        submission_path=Path("runs/run-1/iter-3/submission.csv"),
        error_kind="transient",
        reason="network_or_timeout",
        action_taken="retry",
        fingerprint="fp",
        details="retry later",
        infer_iteration=lambda path: 3 if "iter-3" in path.as_posix() else None,
        normalize_detail=lambda text, max_chars: str(text)[:max_chars],
        record_error_fix_insight=lambda **kwargs: calls.append(kwargs),
    )

    assert recorded is True
    assert calls == [
        {
            "knowledge_paths": knowledge_paths,
            "slug": "demo",
            "run_id": "run-1",
            "iteration": 3,
            "problem_types": ["tabular"],
            "error_message": "submit_error kind=transient reason=network_or_timeout fingerprint=fp",
            "fix_summary": "submit_action=retry; detail=retry later",
            "resolved": False,
            "outcome_bucket": "unknown",
            "submission_score": None,
        }
    ]


def test_record_submit_reason_knowledge_suppresses_record_errors() -> None:
    def fail_record(**_kwargs: object) -> None:
        raise RuntimeError("db unavailable")

    recorded = record_submit_reason_knowledge(
        knowledge_paths=object(),
        slug="demo",
        run_id="run-1",
        problem_types=[],
        submission_path=Path("submission.csv"),
        error_kind="validation",
        reason="bad_submission",
        action_taken="abort",
        fingerprint="fp",
        details="bad",
        infer_iteration=lambda _path: None,
        normalize_detail=lambda text, max_chars: str(text)[:max_chars],
        record_error_fix_insight=fail_record,
    )

    assert recorded is False


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
