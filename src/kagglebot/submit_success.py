from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from kagglebot import submit_attempts as _submit_attempts
from kagglebot import submit_failure_context as _submit_failure_context
from kagglebot import submit_stage_duplicate as _submit_stage_duplicate
from kagglebot.history import SubmissionLedger


@dataclass(frozen=True)
class SubmitStageSuccessRecord:
    exit_code: int | None
    fingerprint: str
    stdout: str
    stderr: str


def build_submit_stage_success_record(
    *,
    submission_result: object,
    compute_error_fingerprint: Callable[[str, str], str],
) -> SubmitStageSuccessRecord:
    stdout = str(getattr(submission_result, "stdout", "") or "")
    stderr = str(getattr(submission_result, "stderr", "") or "")
    return SubmitStageSuccessRecord(
        exit_code=getattr(submission_result, "exit_code", getattr(submission_result, "returncode", None)),
        fingerprint=compute_error_fingerprint(stdout, stderr),
        stdout=stdout,
        stderr=stderr,
    )


def record_successful_submit_stage_result(
    *,
    run_id: str,
    message: str,
    submitted_at: datetime,
    submission_ref: str,
    submission_result: object,
    submission_path: Path,
    submission_artifact_path: Path | None,
    outcome: object,
    code_fingerprint: str,
    prior_state: dict[str, object],
    compute_error_fingerprint: Callable[[str, str], str],
    compute_submission_sha256: Callable[[Path | None], str | None],
    record_submit_attempt_payloads: Callable[[object], object],
    record_outcome: Callable[[Path, dict[str, object]], object],
    mark_failure_context_submitted: Callable[[str], object],
    stdout_tail_chars: int,
    stderr_tail_chars: int,
    on_message: Callable[[str], object],
) -> dict[str, object]:
    submit_success_record = build_submit_stage_success_record(
        submission_result=submission_result,
        compute_error_fingerprint=compute_error_fingerprint,
    )
    submit_success_payloads = _submit_attempts.build_submit_success_record_payloads(
        run_id=run_id,
        submission_ref=submission_ref,
        submission_sha256=compute_submission_sha256(submission_artifact_path),
        exit_code=submit_success_record.exit_code,
        fingerprint=submit_success_record.fingerprint,
        code_fingerprint=code_fingerprint,
        stdout=submit_success_record.stdout,
        stderr=submit_success_record.stderr,
        prior_state=prior_state,
        stdout_tail_chars=stdout_tail_chars,
        stderr_tail_chars=stderr_tail_chars,
    )
    record_submit_attempt_payloads(submit_success_payloads)
    on_message("[green]submission recorded[/green]")

    outcome_recording = _submit_attempts.decide_submit_outcome_recording(
        outcome=outcome,
        submission_artifact_exists=bool(submission_artifact_path is not None and submission_artifact_path.exists()),
    )
    on_message(outcome_recording.message)
    _submit_attempts.record_submit_outcome_if_available(
        decision=outcome_recording,
        submission_path=submission_artifact_path,
        record_outcome=record_outcome,
    )
    mark_failure_context_submitted(submission_ref)
    return _submit_attempts.build_successful_submit_result_payload(
        message=message,
        submission_ref=submission_ref,
        submitted_at=submitted_at,
        submission_path=submission_path,
        outcome=outcome,
        infer_iteration=_submit_stage_duplicate.infer_iteration_from_submission_path,
    )


def record_successful_submit_for_run(
    *,
    run_dir: Path,
    submission_ledger_path: Path,
    slug: str,
    run_id: str,
    message: str,
    submitted_at: datetime,
    submission_ref: str,
    submission_result: object,
    submission_path: Path,
    submission_artifact_path: Path | None,
    outcome: object,
    code_fingerprint: str,
    load_run_state: Callable[[Path], dict[str, object]],
    compute_error_fingerprint: Callable[[str, str], str],
    compute_submission_sha256: Callable[[Path | None], str | None],
    record_submit_attempt_payloads: Callable[[object], object],
    stdout_tail_chars: int,
    stderr_tail_chars: int,
    on_message: Callable[[str], object],
) -> dict[str, object]:
    def record_outcome(path: Path, ledger_outcome: dict[str, object]) -> None:
        SubmissionLedger(submission_ledger_path).record_outcome(
            slug=slug,
            message=message,
            submission_path=path,
            run_id=run_id,
            outcome=ledger_outcome,
        )

    def mark_failure_context_submitted(submitted_ref: str) -> None:
        _submit_failure_context.mark_submit_failure_context_submitted(
            run_dir=run_dir,
            submission_ref=submitted_ref,
        )

    return record_successful_submit_stage_result(
        run_id=run_id,
        message=message,
        submitted_at=submitted_at,
        submission_ref=submission_ref,
        submission_result=submission_result,
        submission_path=submission_path,
        submission_artifact_path=submission_artifact_path,
        outcome=outcome,
        code_fingerprint=code_fingerprint,
        prior_state=load_run_state(run_dir),
        compute_error_fingerprint=compute_error_fingerprint,
        compute_submission_sha256=compute_submission_sha256,
        record_submit_attempt_payloads=record_submit_attempt_payloads,
        record_outcome=record_outcome,
        mark_failure_context_submitted=mark_failure_context_submitted,
        stdout_tail_chars=stdout_tail_chars,
        stderr_tail_chars=stderr_tail_chars,
        on_message=on_message,
    )
