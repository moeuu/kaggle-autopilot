from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from kagglebot import submit_abort as _submit_abort
from kagglebot import submit_attempts as _submit_attempts
from kagglebot import submit_failure_context as _submit_failure_context
from kagglebot import submit_message as _submit_message
from kagglebot.submission_service import SubmissionConfig, SubmissionService


@dataclass(frozen=True)
class SubmitRunContext:
    submit_attempt_recorder: object
    autofix_attempt_context: _submit_failure_context.SubmitAutofixAttemptContext
    submit_code_fingerprint: str
    allow_force: bool
    input_submission_path: Path
    run_state: dict[str, object]
    latest_submit_attempt: dict[str, object]
    submit_aborter: _submit_abort.SubmitRunAborter
    submit_retry_recorder: _submit_abort.SubmitRunRetryRecorder


@dataclass(frozen=True)
class SubmitRuntimeContext:
    message: str
    submission_service: SubmissionService
    submitted_at: datetime


def build_submit_run_context(
    *,
    run_dir: Path,
    run_id: str,
    slug: str,
    submission_path: Path,
    src_root: Path,
    kernel_source_dir: Path,
    knowledge_paths: object,
    problem_types: list[str],
    force_submit: bool,
    force_resubmit: bool,
    save_run_state_for_run: Callable[[Path, dict[str, object]], object],
    load_run_state: Callable[[Path], dict[str, object]],
    compute_submit_code_fingerprint: Callable[..., str],
    compute_submission_sha256: Callable[[Path | None], str | None],
    stdout_tail_chars: int,
    stderr_tail_chars: int,
    now_iso: Callable[[], str],
    normalize_detail: Callable[..., str],
    record_error_fix_insight: Callable[..., object],
    on_message: Callable[[str], object],
    build_error: Callable[[str], BaseException],
) -> SubmitRunContext:
    submit_attempt_recorder = _submit_attempts.build_submit_attempt_recorder_for_run(
        run_dir=run_dir,
        save_run_state_for_run=save_run_state_for_run,
    )
    autofix_attempt_context = _submit_failure_context.resolve_submit_autofix_context_for_run(
        run_dir=run_dir,
        submission_path=submission_path,
        load_run_state=load_run_state,
        save_run_state_for_run=save_run_state_for_run,
        now_iso=now_iso(),
    )
    if autofix_attempt_context.message:
        on_message(autofix_attempt_context.message)
    submit_code_fingerprint = compute_submit_code_fingerprint(
        src_root=src_root,
        kernel_source_dir=kernel_source_dir,
        sha256_or_none=compute_submission_sha256,
    )
    submit_aborter = _submit_abort.build_submit_run_aborter_for_run(
        run_dir=run_dir,
        run_id=run_id,
        slug=slug,
        knowledge_paths=knowledge_paths,
        problem_types=problem_types,
        save_run_state_for_run=save_run_state_for_run,
        load_run_state=load_run_state,
        compute_submission_sha256=compute_submission_sha256,
        stdout_tail_chars=stdout_tail_chars,
        stderr_tail_chars=stderr_tail_chars,
        now_iso=now_iso,
        normalize_detail=normalize_detail,
        record_error_fix_insight=record_error_fix_insight,
        on_message=on_message,
        build_error=build_error,
    )
    submit_retry_recorder = _submit_abort.SubmitRunRetryRecorder(
        submit_attempt_recorder=submit_attempt_recorder,
        run_id=run_id,
        slug=slug,
        problem_types=problem_types,
        knowledge_paths=knowledge_paths,
        compute_submission_sha256=compute_submission_sha256,
        stdout_tail_chars=stdout_tail_chars,
        stderr_tail_chars=stderr_tail_chars,
        normalize_detail=normalize_detail,
        record_error_fix_insight=record_error_fix_insight,
    )
    return SubmitRunContext(
        submit_attempt_recorder=submit_attempt_recorder,
        autofix_attempt_context=autofix_attempt_context,
        submit_code_fingerprint=submit_code_fingerprint,
        allow_force=force_submit or force_resubmit,
        input_submission_path=autofix_attempt_context.input_submission_path,
        run_state=autofix_attempt_context.run_state,
        latest_submit_attempt=autofix_attempt_context.latest_submit_attempt,
        submit_aborter=submit_aborter,
        submit_retry_recorder=submit_retry_recorder,
    )


def build_submit_runtime_context(
    *,
    slug: str,
    context_dir: Path,
    run_id: str,
    best_score: float | None,
    explicit_message: str | None,
    submission_path: Path,
    campaign_mode: str,
    target_direction: str,
    data_dir: Path,
    sample_submission_path: Path,
    submission_ledger_path: Path,
    dry_run: bool,
    force_submit: bool,
    now: Callable[[], datetime],
    on_message: Callable[[str], object],
) -> SubmitRuntimeContext:
    message = _submit_message.resolve_submission_message(
        context_dir=context_dir,
        run_id=run_id,
        best_score=best_score,
        explicit_message=explicit_message,
        submission_path=submission_path,
        campaign_mode=campaign_mode,
        target_direction=target_direction,
    )
    submission_service = SubmissionService(
        SubmissionConfig(
            slug=slug,
            data_dir=data_dir,
            sample_submission_path=sample_submission_path,
            submission_ledger_path=submission_ledger_path,
            dry_run=dry_run,
            force_submit=force_submit,
            bypass_rate_limit=False,
        )
    )
    on_message(f"[cyan]submit[/cyan]: {slug}")
    return SubmitRuntimeContext(
        message=message,
        submission_service=submission_service,
        submitted_at=now(),
    )
