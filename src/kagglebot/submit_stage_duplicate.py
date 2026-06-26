from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from kagglebot import submit_attempts as _submit_attempts
from kagglebot import submit_failure_context as _submit_failure_context
from kagglebot.history import SubmissionLedger


def infer_iteration_from_submission_path(path: Path | None) -> int | None:
    if path is None:
        return None
    try:
        name = path.parent.name
        if not name.startswith("iter-"):
            return None
        return int(name.split("-", 1)[1])
    except Exception:  # noqa: BLE001
        return None


def apply_same_submission_path_decision(
    *,
    decision: object,
    run_id: str,
    submission_path: Path,
    compute_submission_sha256: Callable[[Path | None], str | None],
    record_submit_attempt: Callable[[dict[str, object]], object],
    stdout_tail_chars: int,
    stderr_tail_chars: int,
    on_message: Callable[[str], object],
) -> bool:
    action = str(getattr(decision, "action", "") or "").strip().lower()
    message = str(getattr(decision, "message", "") or "").strip()
    if action == "retry":
        if message:
            on_message(message)
        return False
    if action != "skip":
        return False

    if message:
        on_message(message)
    record_submit_attempt(
        _submit_attempts.build_same_submission_path_skip_attempt_payload(
            run_id=run_id,
            submission_ref=str(submission_path),
            submission_sha256=compute_submission_sha256(submission_path),
            fingerprint=str(getattr(decision, "fingerprint", "") or ""),
            reason=str(getattr(decision, "reason", "") or ""),
            stdout_tail_chars=stdout_tail_chars,
            stderr_tail_chars=stderr_tail_chars,
        )
    )
    return True


def resolve_same_submission_path_for_submit(
    *,
    run_state: dict[str, object],
    latest_submit_attempt: dict[str, object],
    prepared_submission_path: Path,
    current_submission_sha: str,
    submit_code_fingerprint: str,
    allow_force: bool,
    notebook_submit_required: bool,
    decide_same_submission_path_action: Callable[..., object],
    run_id: str,
    compute_submission_sha256: Callable[[Path | None], str | None],
    record_submit_attempt: Callable[[dict[str, object]], object],
    stdout_tail_chars: int,
    stderr_tail_chars: int,
    on_message: Callable[[str], object],
) -> bool:
    if notebook_submit_required:
        return False
    decision = decide_same_submission_path_action(
        run_state=run_state,
        latest_submit_attempt=latest_submit_attempt,
        prepared_submission_path=prepared_submission_path,
        current_submission_sha=current_submission_sha,
        submit_code_fingerprint=submit_code_fingerprint,
        allow_force=allow_force,
        notebook_submit_required=notebook_submit_required,
    )
    return apply_same_submission_path_decision(
        decision=decision,
        run_id=run_id,
        submission_path=prepared_submission_path,
        compute_submission_sha256=compute_submission_sha256,
        record_submit_attempt=record_submit_attempt,
        stdout_tail_chars=stdout_tail_chars,
        stderr_tail_chars=stderr_tail_chars,
        on_message=on_message,
    )


def resolve_same_submission_path_for_run(
    *,
    run_id: str,
    run_state: dict[str, object],
    latest_submit_attempt: dict[str, object],
    prepared_submission_path: Path,
    current_submission_sha: str,
    submit_code_fingerprint: str,
    allow_force: bool,
    notebook_submit_required: bool,
    decide_same_submission_path_action: Callable[..., object],
    compute_submission_sha256: Callable[[Path | None], str | None],
    submit_attempt_recorder: object,
    stdout_tail_chars: int,
    stderr_tail_chars: int,
    on_message: Callable[[str], object],
) -> bool:
    return resolve_same_submission_path_for_submit(
        run_state=run_state,
        latest_submit_attempt=latest_submit_attempt,
        prepared_submission_path=prepared_submission_path,
        current_submission_sha=current_submission_sha,
        submit_code_fingerprint=submit_code_fingerprint,
        allow_force=allow_force,
        notebook_submit_required=notebook_submit_required,
        decide_same_submission_path_action=decide_same_submission_path_action,
        run_id=run_id,
        compute_submission_sha256=compute_submission_sha256,
        record_submit_attempt=submit_attempt_recorder.append,
        stdout_tail_chars=stdout_tail_chars,
        stderr_tail_chars=stderr_tail_chars,
        on_message=on_message,
    )


def apply_duplicate_submission_decision(
    *,
    decision: object,
    run_id: str,
    message: str,
    submitted_at: datetime,
    submission_path: Path,
    prepared_submission_path: Path,
    prepared_submission_sha: str,
    code_fingerprint: str,
    prior_state: dict[str, object],
    record_submit_attempt_payloads: Callable[[object], object],
    mark_duplicate_skipped: Callable[[str, str], object],
    stdout_tail_chars: int,
    stderr_tail_chars: int,
    on_message: Callable[[str], object],
) -> dict[str, object] | None:
    action = str(getattr(decision, "action", "") or "").strip().lower()
    if action != "skip":
        return None

    decision_message = str(getattr(decision, "message", "") or "").strip()
    if decision_message:
        on_message(decision_message)
    reason = str(getattr(decision, "reason", "") or "")
    duplicate_sources = list(getattr(decision, "duplicate_sources", []) or [])
    submission_ref = str(prepared_submission_path)
    skip_payloads = _submit_attempts.build_duplicate_submit_skip_record_payloads(
        run_id=run_id,
        submission_ref=submission_ref,
        submission_sha256=prepared_submission_sha,
        fingerprint=str(getattr(decision, "fingerprint", "") or ""),
        code_fingerprint=code_fingerprint,
        reason=reason,
        prior_state=prior_state,
        stdout_tail_chars=stdout_tail_chars,
        stderr_tail_chars=stderr_tail_chars,
        duplicate_sources=duplicate_sources,
    )
    record_submit_attempt_payloads(skip_payloads)
    mark_duplicate_skipped(submission_ref, reason)
    return _submit_attempts.build_duplicate_submit_skip_result_payload(
        message=message,
        submission_ref=submission_ref,
        submitted_at=submitted_at,
        submission_path=submission_path,
        reason=reason,
        duplicate_sources=duplicate_sources,
        infer_iteration=infer_iteration_from_submission_path,
    )


def resolve_duplicate_submission_for_submit(
    *,
    slug: str,
    run_id: str,
    message: str,
    submitted_at: datetime,
    submission_path: Path,
    prepared_submission_path: Path,
    prepared_submission_sha: str,
    code_fingerprint: str,
    allow_force: bool,
    prior_state: dict[str, object],
    collect_duplicate_submission_sources: Callable[..., list[str]],
    decide_duplicate_submission_action: Callable[..., object],
    submission_attempt_sha_seen: Callable[[str], bool],
    submission_ledger_duplicate: Callable[[], bool],
    compute_error_fingerprint: Callable[[str, str], str],
    record_submit_attempt_payloads: Callable[[object], object],
    mark_duplicate_skipped: Callable[[str, str], object],
    stdout_tail_chars: int,
    stderr_tail_chars: int,
    on_message: Callable[[str], object],
) -> dict[str, object] | None:
    duplicate_sources = collect_duplicate_submission_sources(
        prepared_submission_sha=prepared_submission_sha,
        allow_force=allow_force,
        submission_attempt_sha_seen=submission_attempt_sha_seen,
        submission_ledger_duplicate=submission_ledger_duplicate,
    )
    duplicate_decision = decide_duplicate_submission_action(
        slug=slug,
        prepared_submission_sha=prepared_submission_sha,
        duplicate_sources=duplicate_sources,
        allow_force=allow_force,
        compute_fingerprint=compute_error_fingerprint,
    )
    return apply_duplicate_submission_decision(
        decision=duplicate_decision,
        run_id=run_id,
        message=message,
        submitted_at=submitted_at,
        submission_path=submission_path,
        prepared_submission_path=prepared_submission_path,
        prepared_submission_sha=prepared_submission_sha,
        code_fingerprint=code_fingerprint,
        prior_state=prior_state,
        record_submit_attempt_payloads=record_submit_attempt_payloads,
        mark_duplicate_skipped=mark_duplicate_skipped,
        stdout_tail_chars=stdout_tail_chars,
        stderr_tail_chars=stderr_tail_chars,
        on_message=on_message,
    )


def resolve_duplicate_submission_for_run(
    *,
    run_dir: Path,
    submission_ledger_path: Path,
    slug: str,
    run_id: str,
    message: str,
    submitted_at: datetime,
    submission_path: Path,
    prepared_submission_path: Path,
    prepared_submission_sha: str,
    code_fingerprint: str,
    allow_force: bool,
    load_run_state: Callable[[Path], dict[str, object]],
    collect_duplicate_submission_sources: Callable[..., list[str]],
    decide_duplicate_submission_action: Callable[..., object],
    compute_error_fingerprint: Callable[[str, str], str],
    record_submit_attempt_payloads: Callable[[object], object],
    stdout_tail_chars: int,
    stderr_tail_chars: int,
    on_message: Callable[[str], object],
) -> dict[str, object] | None:
    return resolve_duplicate_submission_for_submit(
        slug=slug,
        run_id=run_id,
        message=message,
        submitted_at=submitted_at,
        submission_path=submission_path,
        prepared_submission_path=prepared_submission_path,
        prepared_submission_sha=prepared_submission_sha,
        code_fingerprint=code_fingerprint,
        allow_force=allow_force,
        prior_state=load_run_state(run_dir),
        collect_duplicate_submission_sources=collect_duplicate_submission_sources,
        decide_duplicate_submission_action=decide_duplicate_submission_action,
        submission_attempt_sha_seen=lambda submission_sha: _submit_attempts.submit_attempt_sha_seen(
            run_dir=run_dir,
            submission_sha=submission_sha,
        ),
        submission_ledger_duplicate=lambda: SubmissionLedger(submission_ledger_path).is_duplicate(
            slug=slug,
            message=message,
            submission_path=prepared_submission_path,
        ),
        compute_error_fingerprint=compute_error_fingerprint,
        record_submit_attempt_payloads=record_submit_attempt_payloads,
        mark_duplicate_skipped=lambda submission_ref, reason: (
            _submit_failure_context.mark_submit_failure_context_duplicate_skipped(
                run_dir=run_dir,
                submission_ref=submission_ref,
                reason=reason,
            )
        ),
        stdout_tail_chars=stdout_tail_chars,
        stderr_tail_chars=stderr_tail_chars,
        on_message=on_message,
    )
