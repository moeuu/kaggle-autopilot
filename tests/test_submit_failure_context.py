from __future__ import annotations

import json
from pathlib import Path

from kagglebot.submit_failure_context import (
    build_submit_failure_context_payload,
    decide_stale_submit_autofix_artifact,
    decide_submit_autofix_input_submission,
    format_submit_autofix_context,
    load_submit_failure_context,
    mark_submit_failure_context_resolved,
    path_from_submit_reference,
    resolve_submit_autofix_submission_artifact,
    save_submit_failure_context,
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


def test_path_from_submit_reference_ignores_kernel_references() -> None:
    assert path_from_submit_reference("kernel:user/demo") is None
    assert path_from_submit_reference("") is None
    assert path_from_submit_reference("/tmp/submission.csv") == Path("/tmp/submission.csv")


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
