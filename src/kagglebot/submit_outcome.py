from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rich import print

from kagglebot import submit_abort_specs as _submit_abort_specs
from kagglebot import submit_outcome_decisions as _submit_outcome_decisions
from kagglebot import submit_success as _submit_success
from kagglebot.submission.outcome_service import SubmissionOutcomePollingError, SubmissionOutcomeService
from kagglebot.submit_abort_specs import SubmitAbortSpec
from kagglebot.submit_cli_error_resolution import SubmitStageRuntimeState


@dataclass(frozen=True)
class SubmitOutcomeResolution:
    outcome: object
    abort_spec: SubmitAbortSpec | None = None


def wait_for_submission_outcome(
    *,
    slug: str,
    message: str,
    submitted_at: datetime,
    fetch_submission_rows: Callable[[str], list[dict[str, str]]],
    max_attempts: int | None,
    poll_interval_sec: float,
    max_fetch_errors: int,
) -> dict[str, object] | None:
    print(f"[cyan]submission polling[/cyan]: waiting for result (interval={poll_interval_sec:.0f}s)")
    service = SubmissionOutcomeService(
        fetch_rows=fetch_submission_rows,
        max_attempts=max_attempts,
        poll_interval_sec=poll_interval_sec,
        max_fetch_errors=max_fetch_errors,
    )
    return service.wait_for_outcome(
        slug=slug,
        message=message,
        submitted_at=submitted_at,
    )


def resolve_submission_outcome_after_submit(
    *,
    slug: str,
    message: str,
    submitted_at: datetime,
    deliverable_mode: str,
    fetch_submission_rows: Callable[[str], list[dict[str, str]]],
    max_attempts: int | None,
    poll_interval_sec: float,
    max_fetch_errors: int,
    normalize_detail: Callable[[str], str],
    compute_error_fingerprint: Callable[[str, str], str],
) -> SubmitOutcomeResolution:
    try:
        outcome = wait_for_submission_outcome(
            slug=slug,
            message=message,
            submitted_at=submitted_at,
            fetch_submission_rows=fetch_submission_rows,
            max_attempts=max_attempts,
            poll_interval_sec=poll_interval_sec,
            max_fetch_errors=max_fetch_errors,
        )
    except SubmissionOutcomePollingError as exc:
        return SubmitOutcomeResolution(
            outcome=None,
            abort_spec=_submit_abort_specs.build_submission_polling_error_abort_spec(
                error=exc,
                detail=exc.detail,
                normalize_detail=normalize_detail,
                compute_error_fingerprint=compute_error_fingerprint,
            ),
        )

    outcome_post_poll = _submit_outcome_decisions.evaluate_submission_outcome_after_poll(
        slug=slug,
        message=message,
        submitted_at=submitted_at,
        outcome=outcome,
        deliverable_mode=deliverable_mode,
        fetch_submission_rows=fetch_submission_rows,
        normalize_detail=normalize_detail,
    )
    if outcome_post_poll.abort_decision.should_abort:
        return SubmitOutcomeResolution(
            outcome=outcome_post_poll.outcome,
            abort_spec=_submit_abort_specs.build_submission_outcome_abort_spec(
                decision=outcome_post_poll.abort_decision,
                compute_error_fingerprint=compute_error_fingerprint,
            ),
        )
    return SubmitOutcomeResolution(outcome=outcome_post_poll.outcome)


def finalize_submit_outcome_for_run_or_abort(
    *,
    run_dir: Path,
    submission_ledger_path: Path,
    slug: str,
    run_id: str,
    message: str,
    submitted_at: datetime,
    submission_ref: str,
    submission_result: object,
    source_submission_path: Path,
    submission_artifact_path: Path | None,
    submit_stage_state: SubmitStageRuntimeState,
    code_fingerprint: str,
    deliverable_mode: str,
    fetch_submission_rows: Callable[[str], list[dict[str, str]]],
    max_attempts: int | None,
    poll_interval_sec: float,
    max_fetch_errors: int,
    normalize_detail: Callable[[str], str],
    compute_error_fingerprint: Callable[[str, str], str],
    compute_submission_sha256: Callable[[Path | None], str | None],
    load_run_state: Callable[[Path], dict[str, object]],
    record_submit_attempt_payloads: Callable[[object], object],
    submit_aborter: object,
    submit_attempt_recorder: object,
    stdout_tail_chars: int,
    stderr_tail_chars: int,
    on_message: Callable[[str], object],
) -> dict[str, object]:
    outcome_resolution = resolve_submission_outcome_after_submit(
        slug=slug,
        message=message,
        submitted_at=submitted_at,
        deliverable_mode=deliverable_mode,
        fetch_submission_rows=fetch_submission_rows,
        max_attempts=max_attempts,
        poll_interval_sec=poll_interval_sec,
        max_fetch_errors=max_fetch_errors,
        normalize_detail=normalize_detail,
        compute_error_fingerprint=compute_error_fingerprint,
    )
    if outcome_resolution.abort_spec is not None:
        return submit_aborter.abort(
            submission_ref=submission_ref,
            submission_artifact_path=submission_artifact_path,
            artifact_mode=submit_stage_state.submission_artifact_mode,
            code_fingerprint=code_fingerprint,
            **_submit_abort_specs.build_submit_abort_spec_kwargs(outcome_resolution.abort_spec),
            submit_attempt_recorder=submit_attempt_recorder,
        )
    return _submit_success.record_successful_submit_for_run(
        run_dir=run_dir,
        submission_ledger_path=submission_ledger_path,
        slug=slug,
        run_id=run_id,
        message=message,
        submitted_at=submitted_at,
        submission_ref=submission_ref,
        submission_result=submission_result,
        submission_path=source_submission_path,
        submission_artifact_path=submission_artifact_path,
        outcome=outcome_resolution.outcome,
        code_fingerprint=code_fingerprint,
        load_run_state=load_run_state,
        compute_error_fingerprint=compute_error_fingerprint,
        compute_submission_sha256=compute_submission_sha256,
        record_submit_attempt_payloads=record_submit_attempt_payloads,
        stdout_tail_chars=stdout_tail_chars,
        stderr_tail_chars=stderr_tail_chars,
        on_message=on_message,
    )
