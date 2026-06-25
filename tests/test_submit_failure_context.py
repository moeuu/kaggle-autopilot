from __future__ import annotations

import json
from pathlib import Path

from kagglebot.submit_attempts import SubmitAttemptRecorder, load_latest_submit_attempt
from kagglebot.submit_failure_context import (
    apply_stale_submit_autofix_decision,
    build_submit_failure_context_payload,
    build_submit_failure_context_payload_from_error,
    build_submit_failure_improvement_context,
    decide_stale_submit_autofix_artifact,
    decide_submit_abort_autofixability,
    decide_submit_autofix_input_submission,
    format_submit_autofix_context,
    format_submit_file_repair_contract_prompt,
    format_submit_file_repair_contract_retry_feedback,
    load_submit_failure_context,
    mark_submit_failure_context_duplicate_skipped,
    mark_submit_failure_context_resolved,
    mark_submit_failure_context_submitted,
    path_from_submit_reference,
    persist_submit_abort_failure,
    resolve_submit_abort_artifact_path,
    resolve_submit_autofix_submission_artifact,
    save_submit_failure_context,
    should_defer_submit_abort_to_next_iteration,
    should_force_resubmit_after_submit_abort,
    submit_failure_context_path,
    submit_file_fix_contract_satisfied,
)
from kagglebot.submit_failure_policy import SubmitFailureRepairDecision


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


def test_mark_submit_failure_context_submitted_uses_submitted_resolution(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    save_submit_failure_context(
        run_dir,
        {
            "active": True,
            "reason": "submission_polling_error",
            "repair_target": "platform_or_transient",
        },
    )

    mark_submit_failure_context_submitted(run_dir=run_dir, submission_ref="submission.csv")
    payload = json.loads(submit_failure_context_path(run_dir).read_text(encoding="utf-8"))

    assert payload["active"] is False
    assert payload["resolution"] == "submitted"
    assert payload["resolved_submission_ref"] == "submission.csv"
    assert payload["resolved_at"]


def test_mark_submit_failure_context_duplicate_skipped_uses_duplicate_resolution(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    save_submit_failure_context(
        run_dir,
        {
            "active": True,
            "reason": "submission_polling_error",
            "repair_target": "platform_or_transient",
        },
    )

    mark_submit_failure_context_duplicate_skipped(run_dir=run_dir, submission_ref="submission.csv")
    payload = json.loads(submit_failure_context_path(run_dir).read_text(encoding="utf-8"))

    assert payload["active"] is False
    assert payload["resolution"] == "duplicate_submission_sha_seen"
    assert payload["resolved_submission_ref"] == "submission.csv"
    assert payload["resolved_at"]


def test_build_submit_failure_context_payload_centralizes_record_shape(tmp_path: Path) -> None:
    artifact = tmp_path / "submission.csv"
    payload = build_submit_failure_context_payload(
        now_iso="2026-06-25T00:00:00+00:00",
        submission_ref="kernel:user/demo",
        artifact_path=artifact,
        artifact_sha256="sha",
        artifact_mode="inference",
        code_fingerprint="code-fp",
        fingerprint="error-fp",
        error_kind="validation",
        reason="submission_poll_status_error",
        message="Kaggle scoring failed.",
        stdout_tail="abcdef",
        stderr_tail="row count mismatch",
        exit_code=1,
        repair_decision=SubmitFailureRepairDecision(
            repair_target="submission_artifact",
            repairable=True,
            manual_next_step="",
        ),
        latest_submit_attempt={"reason": "previous"},
        run_state={
            "submit_attempted": True,
            "submit_ok": False,
            "last_reason": "previous",
            "last_error_kind": "validation",
            "last_submission_path": "/tmp/old.csv",
            "submit_autofix_submission_path": "/tmp/fixed.csv",
        },
        stdout_tail_chars=3,
        stderr_tail_chars=8,
    )

    assert payload["ts"] == "2026-06-25T00:00:00+00:00"
    assert payload["active"] is True
    assert payload["submit_mode"] == "notebook"
    assert payload["artifact_mode"] == "inference"
    assert payload["submission_artifact_path"] == str(artifact)
    assert payload["submission_artifact_sha256"] == "sha"
    assert payload["stdout_tail"] == "def"
    assert payload["stderr_tail"] == "mismatch"
    assert payload["repair_target"] == "submission_artifact"
    assert payload["repairable"] is True
    assert payload["latest_submit_attempt"] == {"reason": "previous"}
    assert payload["run_state_excerpt"]["submit_autofix_submission_path"] == "/tmp/fixed.csv"
    assert "Kaggle scoring failed" in str(payload["summary"])


def test_build_submit_failure_context_payload_from_error_classifies_repair_target(tmp_path: Path) -> None:
    artifact = tmp_path / "submission.csv"
    payload = build_submit_failure_context_payload_from_error(
        now_iso="2026-06-25T00:00:00+00:00",
        submission_ref=str(artifact),
        artifact_path=artifact,
        artifact_sha256="sha",
        artifact_mode="wrapper",
        code_fingerprint="code-fp",
        fingerprint="error-fp",
        error_kind="validation",
        reason="submission_poll_status_error",
        message="Kaggle scoring failed.",
        stdout_tail="",
        stderr_tail="row count mismatch",
        exit_code=1,
        latest_submit_attempt={"reason": "previous"},
        run_state={"submit_attempted": True, "submit_ok": False},
        stdout_tail_chars=100,
        stderr_tail_chars=100,
    )

    assert payload["submit_mode"] == "file"
    assert payload["repair_target"] == "submission_artifact"
    assert payload["repairable"] is True
    assert payload["submission_artifact_sha256"] == "sha"
    assert "row count mismatch" in str(payload["summary"])


def test_path_from_submit_reference_ignores_kernel_references() -> None:
    assert path_from_submit_reference("kernel:user/demo") is None
    assert path_from_submit_reference("") is None
    assert path_from_submit_reference("/tmp/submission.csv") == Path("/tmp/submission.csv")


def test_resolve_submit_abort_artifact_path_prefers_explicit_artifact(tmp_path: Path) -> None:
    explicit = tmp_path / "copied" / "submission.csv"

    assert (
        resolve_submit_abort_artifact_path(
            submission_ref=tmp_path / "original.csv",
            submission_artifact_path=explicit,
        )
        == explicit
    )


def test_resolve_submit_abort_artifact_path_uses_path_submission_ref(tmp_path: Path) -> None:
    submission_path = tmp_path / "submission.csv"

    assert (
        resolve_submit_abort_artifact_path(
            submission_ref=submission_path,
            submission_artifact_path=None,
        )
        == submission_path
    )
    assert (
        resolve_submit_abort_artifact_path(
            submission_ref="kernel:user/demo",
            submission_artifact_path=None,
        )
        is None
    )


def test_persist_submit_abort_failure_records_attempt_and_context(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_state: dict[str, object] = {"submit_attempts_count": 2, "submit_ok": True}
    recorder = SubmitAttemptRecorder(
        run_dir=run_dir,
        save_run_state=lambda updates: run_state.update(updates),
    )

    persist_submit_abort_failure(
        run_dir=run_dir,
        run_id="run-1",
        submission_ref=str(tmp_path / "submission.csv"),
        submission_sha256="sha",
        artifact_path=tmp_path / "submission.csv",
        artifact_mode="wrapper",
        code_fingerprint="code-fp",
        fingerprint="error-fp",
        error_kind="validation",
        reason="submission_poll_status_error",
        message="Kaggle scoring failed.",
        stdout_tail="stdout detail",
        stderr_tail="row count mismatch",
        exit_code=1,
        prior_state=dict(run_state),
        prior_submit_ok=True,
        submit_attempt_recorder=recorder,
        load_latest_submit_attempt=load_latest_submit_attempt,
        load_run_state=lambda _run_dir: dict(run_state),
        stdout_tail_chars=100,
        stderr_tail_chars=100,
        now_iso="2026-06-25T00:00:00+00:00",
    )

    latest = load_latest_submit_attempt(run_dir)
    context = load_submit_failure_context(run_dir)

    assert latest["ok"] is False
    assert latest["action_taken"] == "abort"
    assert latest["sub_sha256"] == "sha"
    assert run_state["submit_ok"] is True
    assert run_state["submit_attempts_count"] == 3
    assert context["ts"] == "2026-06-25T00:00:00+00:00"
    assert context["latest_submit_attempt"]["fingerprint"] == "error-fp"
    assert context["run_state_excerpt"]["submit_attempted"] is True
    assert context["repair_target"] == "submission_artifact"


def test_decide_stale_submit_autofix_artifact_returns_context_updates(tmp_path: Path) -> None:
    original = tmp_path / "iter-1" / "submission.csv"
    repaired = tmp_path / "iter-1" / "output" / "submission-fixed.csv"
    new_submission = tmp_path / "iter-2" / "submission.csv"

    decision = decide_stale_submit_autofix_artifact(
        run_state={"submit_autofix_submission_path": str(repaired)},
        failure_context={"submission_artifact_path": str(original)},
        submission_path=new_submission,
        now_iso="2026-06-25T00:00:00+00:00",
    )

    assert decision is not None
    assert decision.clear_repaired_path is True
    assert decision.failure_context_updates["superseded_by_submission_path"] == str(new_submission)


def test_apply_stale_submit_autofix_decision_updates_state_and_context(tmp_path: Path) -> None:
    original = tmp_path / "iter-1" / "submission.csv"
    repaired = tmp_path / "iter-1" / "output" / "submission-fixed.csv"
    new_submission = tmp_path / "iter-2" / "submission.csv"
    failure_context = {"submission_artifact_path": str(original), "active": True}
    decision = decide_stale_submit_autofix_artifact(
        run_state={"submit_autofix_submission_path": str(repaired)},
        failure_context=failure_context,
        submission_path=new_submission,
        now_iso="2026-06-25T00:00:00+00:00",
    )
    state_updates: list[dict[str, object]] = []
    saved_contexts: list[dict[str, object]] = []

    updated = apply_stale_submit_autofix_decision(
        decision=decision,
        failure_context=failure_context,
        save_run_state=state_updates.append,
        save_failure_context=saved_contexts.append,
    )

    assert state_updates == [{"submit_autofix_submission_path": ""}]
    assert saved_contexts == [updated]
    assert updated["active"] is True
    assert updated["stale_repaired_artifact_cleared_at"] == "2026-06-25T00:00:00+00:00"
    assert updated["superseded_by_submission_path"] == str(new_submission)
    assert "superseded_by_submission_path" not in failure_context


def test_apply_stale_submit_autofix_decision_noops_without_decision() -> None:
    failure_context = {"active": True}
    state_updates: list[dict[str, object]] = []
    saved_contexts: list[dict[str, object]] = []

    updated = apply_stale_submit_autofix_decision(
        decision=None,
        failure_context=failure_context,
        save_run_state=state_updates.append,
        save_failure_context=saved_contexts.append,
    )

    assert updated == failure_context
    assert state_updates == []
    assert saved_contexts == []


def test_decide_stale_submit_autofix_artifact_keeps_same_failed_artifact(tmp_path: Path) -> None:
    original = tmp_path / "iter-1" / "submission.csv"
    repaired = tmp_path / "iter-1" / "output" / "submission-fixed.csv"

    assert (
        decide_stale_submit_autofix_artifact(
            run_state={"submit_autofix_submission_path": str(repaired)},
            failure_context={"submission_artifact_path": str(original)},
            submission_path=original,
            now_iso="2026-06-25T00:00:00+00:00",
        )
        is None
    )


def test_decide_submit_autofix_input_submission_prefers_matching_repaired_artifact(tmp_path: Path) -> None:
    original = tmp_path / "iter-1" / "submission.csv"
    repaired = tmp_path / "iter-1" / "output" / "submission-fixed.csv"
    repaired.parent.mkdir(parents=True)
    repaired.write_text("id,target\n1,0.2\n", encoding="utf-8")

    decision = decide_submit_autofix_input_submission(
        run_state={"submit_autofix_submission_path": str(repaired)},
        latest_submit_attempt={},
        failure_context={"repair_target": "submission_artifact", "submission_artifact_path": str(original)},
        submission_path=original,
    )

    assert decision.input_submission_path == repaired
    assert str(repaired) in decision.message


def test_decide_submit_autofix_input_submission_honors_legacy_file_fix_attempt(tmp_path: Path) -> None:
    original = tmp_path / "iter-1" / "submission.csv"
    repaired = tmp_path / "iter-1" / "output" / "submission-fixed.csv"
    repaired.parent.mkdir(parents=True)
    repaired.write_text("id,target\n1,0.2\n", encoding="utf-8")

    decision = decide_submit_autofix_input_submission(
        run_state={"submit_autofix_submission_path": str(repaired), "last_error_kind": "validation"},
        latest_submit_attempt={"stderr_tail": "Submission validation failed: row count mismatch"},
        failure_context={},
        submission_path=original,
    )

    assert decision.input_submission_path == repaired


def test_decide_submit_autofix_input_submission_ignores_repair_for_different_failed_artifact(tmp_path: Path) -> None:
    original = tmp_path / "iter-1" / "submission.csv"
    other_failed = tmp_path / "iter-0" / "submission.csv"
    repaired = tmp_path / "iter-0" / "output" / "submission-fixed.csv"
    repaired.parent.mkdir(parents=True)
    repaired.write_text("id,target\n1,0.2\n", encoding="utf-8")

    decision = decide_submit_autofix_input_submission(
        run_state={"submit_autofix_submission_path": str(repaired)},
        latest_submit_attempt={},
        failure_context={"repair_target": "submission_artifact", "submission_artifact_path": str(other_failed)},
        submission_path=original,
    )

    assert decision.input_submission_path == original
    assert decision.message == ""


def test_decide_submit_abort_autofixability_allows_repairable_failure_context() -> None:
    decision = decide_submit_abort_autofixability(
        failure_context={"repairable": True, "reason": "local_submission_validation_failed"},
        run_state={},
    )

    assert decision.autofixable is True
    assert decision.message == ""


def test_decide_submit_abort_autofixability_rejects_manual_failure_context() -> None:
    decision = decide_submit_abort_autofixability(
        failure_context={
            "repairable": False,
            "repair_target": "manual_intervention",
            "reason": "rules_not_accepted",
            "manual_next_step": "Accept the competition rules.",
        },
        run_state={},
    )

    assert decision.autofixable is False
    assert "manual intervention" in decision.message
    assert "Accept the competition rules." in decision.message


def test_decide_submit_abort_autofixability_rejects_ambiguous_notebook_bad_request() -> None:
    decision = decide_submit_abort_autofixability(
        failure_context={"repairable": True, "reason": "ambiguous_notebook_bad_request"},
        run_state={},
    )

    assert decision.autofixable is False
    assert "ambiguous notebook 400" in decision.message


def test_decide_submit_abort_autofixability_uses_legacy_run_state() -> None:
    validation = decide_submit_abort_autofixability(
        failure_context={},
        run_state={"last_error_kind": "validation", "last_reason": "local_submission_validation_failed"},
    )
    repeated = decide_submit_abort_autofixability(
        failure_context={},
        run_state={"last_error_kind": "permanent", "last_reason": "same_error_fingerprint_recurred"},
    )
    permanent = decide_submit_abort_autofixability(
        failure_context={},
        run_state={"last_error_kind": "permanent", "last_reason": "rules_not_accepted"},
    )

    assert validation.autofixable is True
    assert repeated.autofixable is True
    assert permanent.autofixable is False
    assert "not safely auto-fixable" in permanent.message


def test_should_force_resubmit_after_submit_abort_only_for_polling_or_scoring_failures() -> None:
    assert should_force_resubmit_after_submit_abort({"last_reason": "submission_polling_error"})
    assert should_force_resubmit_after_submit_abort({"last_reason": "submission_poll_status_error"})
    assert should_force_resubmit_after_submit_abort({"last_reason": "submission_poll_status_complete_no_score"})
    assert not should_force_resubmit_after_submit_abort({"last_reason": "rules_not_accepted"})
    assert not should_force_resubmit_after_submit_abort({})


def test_should_defer_submit_abort_to_next_iteration_requires_kaggle_gpu_repairable_nonfinal() -> None:
    repairable_context = {"active": True, "repairable": True}

    assert should_defer_submit_abort_to_next_iteration(
        compute="kaggle_gpu",
        failure_context=repairable_context,
        iteration=1,
        max_iterations=3,
    )
    assert not should_defer_submit_abort_to_next_iteration(
        compute="local_gpu",
        failure_context=repairable_context,
        iteration=1,
        max_iterations=3,
    )
    assert not should_defer_submit_abort_to_next_iteration(
        compute="kaggle_gpu",
        failure_context=repairable_context,
        iteration=3,
        max_iterations=3,
    )
    assert not should_defer_submit_abort_to_next_iteration(
        compute="kaggle_gpu",
        failure_context={"active": True, "repairable": False},
        iteration=1,
        max_iterations=3,
    )


def test_format_submit_autofix_context_includes_latest_attempt() -> None:
    context = format_submit_autofix_context(
        failure_context={
            "ts": "2026-02-15T00:00:01+00:00",
            "active": True,
            "repair_target": "submission_artifact",
            "repairable": True,
            "reason": "local_submission_validation_failed",
            "error_kind": "validation",
            "fingerprint": "abc123",
            "submission_ref": "/tmp/submission.csv",
            "submission_artifact_path": "/tmp/submission.csv",
            "summary": "Local submission validation failed.",
            "latest_submit_attempt": {
                "ts": "2026-02-15T00:00:00+00:00",
                "ok": False,
                "exit_code": 6,
                "error_kind": "validation",
                "reason": "local_submission_validation_failed",
                "action_taken": "abort",
                "fingerprint": "abc123",
                "sub_path": "/tmp/submission.csv",
            },
            "run_state_excerpt": {
                "submit_attempted": True,
                "submit_ok": False,
                "last_reason": "local_submission_validation_failed",
                "last_error_kind": "validation",
                "last_submission_path": "/tmp/submission.csv",
            },
        },
        run_state={
            "submit_attempted": True,
            "submit_ok": False,
            "last_error_kind": "validation",
            "last_reason": "local_submission_validation_failed",
            "last_action": "abort",
            "last_submit_fingerprint": "abc123",
            "last_submission_path": "/tmp/submission.csv",
        },
        latest_submit_attempt={
            "ts": "2026-02-15T00:00:00+00:00",
            "ok": False,
            "exit_code": 6,
            "error_kind": "validation",
            "reason": "local_submission_validation_failed",
            "action_taken": "abort",
            "fingerprint": "abc123",
            "sub_path": "/tmp/submission.csv",
            "stdout_tail": "",
            "stderr_tail": "Submission validation failed: prediction column contains NaN",
        },
    )

    assert "submit_failure_context:" in context
    assert "repair_target: submission_artifact" in context
    assert "failure_context_latest_submit_attempt:" in context
    assert "run_state:" in context
    assert "latest_submit_attempt:" in context
    assert "last_reason: local_submission_validation_failed" in context
    assert "error_kind: validation" in context
    assert "stderr_tail: Submission validation failed:" in context


def test_build_submit_failure_improvement_context_for_file_issue() -> None:
    notes, reason = build_submit_failure_improvement_context(
        failure_context={
            "active": True,
            "repairable": True,
            "repair_target": "submission_artifact",
            "reason": "submission_poll_status_error",
            "error_kind": "validation",
            "submission_artifact_path": "/tmp/submission.csv",
            "summary": "Kaggle reported a submission row-count mismatch.",
        },
        latest_submit_attempt={
            "stderr_tail": "Evaluation Exception: Submission must have 132133 rows",
            "reason": "submission_poll_status_error",
            "error_kind": "validation",
        },
    )

    assert reason is not None
    assert "repair submit format" in reason.lower()
    assert "failed_submission_artifact=/tmp/submission.csv" in notes
    assert any("Submission must have 132133 rows" in note for note in notes)
    assert any("authoritative `kernel.py`" in note for note in notes)


def test_build_submit_failure_improvement_context_ignores_manual_or_non_file_issue() -> None:
    manual_notes, manual_reason = build_submit_failure_improvement_context(
        failure_context={"active": True, "repairable": False, "reason": "rules_not_accepted"},
        latest_submit_attempt={},
    )
    non_file_notes, non_file_reason = build_submit_failure_improvement_context(
        failure_context={
            "active": True,
            "repairable": True,
            "reason": "kaggle_credentials_missing",
            "error_kind": "permanent",
            "summary": "Credentials are missing.",
        },
        latest_submit_attempt={},
    )

    assert manual_notes == []
    assert manual_reason is None
    assert non_file_notes == []
    assert non_file_reason is None


def test_resolve_submit_autofix_submission_artifact_prefers_repaired_path(tmp_path: Path) -> None:
    repaired = tmp_path / "iter-1" / "output" / "submission-fixed.csv"
    failed = tmp_path / "iter-1" / "submission.csv"
    repaired.parent.mkdir(parents=True)
    failed.parent.mkdir(parents=True, exist_ok=True)
    repaired.write_text("id,target\n1,0.2\n", encoding="utf-8")
    failed.write_text("id,target\n1,0.1\n", encoding="utf-8")

    resolved = resolve_submit_autofix_submission_artifact(
        run_state={"submit_autofix_submission_path": str(repaired), "last_submission_path": str(failed)},
        latest_submit_attempt={},
        failure_context={"submission_artifact_path": str(failed)},
        fallback_iteration_dirs=[],
        resolve_iteration_submission_artifact=lambda path: None,
    )

    assert resolved == repaired


def test_resolve_submit_autofix_submission_artifact_uses_iteration_fallback(tmp_path: Path) -> None:
    iter_dir = tmp_path / "iter-1"
    fallback = iter_dir / "submission.csv"
    fallback.parent.mkdir(parents=True)
    fallback.write_text("id,target\n1,0.1\n", encoding="utf-8")

    resolved = resolve_submit_autofix_submission_artifact(
        run_state={},
        latest_submit_attempt={},
        failure_context={},
        fallback_iteration_dirs=[iter_dir],
        resolve_iteration_submission_artifact=lambda path: path / "submission.csv",
    )

    assert resolved == fallback


def test_submit_file_fix_contract_satisfied_requires_changed_artifact(tmp_path: Path) -> None:
    baseline = tmp_path / "submission.csv"
    fixed = tmp_path / "submission-fixed.csv"
    baseline.write_text("id,target\n1,0.1\n", encoding="utf-8")
    fixed.write_text("id,target\n1,0.2\n", encoding="utf-8")
    hashes = {baseline: "old", fixed: "new"}

    assert submit_file_fix_contract_satisfied(
        run_state={"submit_autofix_submission_path": str(fixed)},
        baseline_path=baseline,
        baseline_sha256="old",
        sha256_or_none=lambda path: hashes.get(path),
    )
    assert not submit_file_fix_contract_satisfied(
        run_state={"submit_autofix_submission_path": str(baseline)},
        baseline_path=baseline,
        baseline_sha256="old",
        sha256_or_none=lambda path: hashes.get(path),
    )


def test_submit_file_repair_contract_text_and_retry_feedback(tmp_path: Path) -> None:
    prompt = format_submit_file_repair_contract_prompt()
    assert "## Submission File Repair Contract" in prompt
    assert "submit_autofix_submission_path" in prompt
    assert "authoritative source" in prompt

    baseline = tmp_path / "submission.csv"
    feedback = format_submit_file_repair_contract_retry_feedback(
        baseline_path=baseline,
        baseline_sha256="abc123",
    )
    assert "Submission file repair contract not satisfied." in feedback
    assert f"baseline_submission_path={baseline}" in feedback
    assert "baseline_submission_sha256=abc123" in feedback
    assert "submit_autofix_submission_path" in feedback


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
