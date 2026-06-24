from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class SubmitKnowledgePayload:
    iteration: int
    error_message: str
    fix_summary: str


@dataclass(frozen=True)
class SubmitOutcomeRecordingDecision:
    message: str
    ledger_outcome: dict[str, object] | None


@dataclass(frozen=True)
class SubmitSuccessRecordPayloads:
    attempt_payload: dict[str, object]
    run_state_update: dict[str, object]


def build_submit_attempt_payload(
    *,
    run_id: str,
    submission_ref: str,
    submission_sha256: str | None,
    exit_code: int | None,
    ok: bool,
    fingerprint: str,
    error_kind: str,
    action_taken: str,
    reason: str,
    stdout: str,
    stderr: str,
    stdout_tail_chars: int,
    stderr_tail_chars: int,
    code_fingerprint: str | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "run_id": run_id,
        "sub_path": submission_ref,
        "sub_sha256": submission_sha256,
        "exit_code": exit_code,
        "ok": ok,
        "fingerprint": fingerprint,
        "error_kind": error_kind,
        "action_taken": action_taken,
        "reason": reason,
        "stdout_tail": stdout[-stdout_tail_chars:],
        "stderr_tail": stderr[-stderr_tail_chars:],
    }
    if code_fingerprint is not None:
        payload["code_fingerprint"] = code_fingerprint
    if extra:
        payload.update(extra)
    return payload


def build_submit_run_state_update(
    *,
    prior_state: dict[str, object],
    fingerprint: str,
    code_fingerprint: str,
    error_kind: str,
    action_taken: str,
    reason: str,
    submission_ref: str,
    submit_ok: bool | None = None,
    submission_sha256: str | None = None,
) -> dict[str, object]:
    update: dict[str, object] = {
        "submit_attempted": True,
        "last_submit_fingerprint": fingerprint,
        "last_fingerprint": fingerprint,
        "last_submit_code_fingerprint": code_fingerprint,
        "last_error_kind": error_kind,
        "last_action": action_taken,
        "last_reason": reason,
        "last_submission_path": submission_ref,
        "submit_attempts_count": int(prior_state.get("submit_attempts_count", 0)) + 1,
    }
    if submit_ok is not None:
        update["submit_ok"] = submit_ok
    if submission_sha256 is not None:
        update["last_submission_sha256"] = submission_sha256
    return update


def build_submit_success_record_payloads(
    *,
    run_id: str,
    submission_ref: str,
    submission_sha256: str | None,
    exit_code: int | None,
    fingerprint: str,
    code_fingerprint: str,
    stdout: str,
    stderr: str,
    prior_state: dict[str, object],
    stdout_tail_chars: int,
    stderr_tail_chars: int,
) -> SubmitSuccessRecordPayloads:
    return SubmitSuccessRecordPayloads(
        attempt_payload=build_submit_attempt_payload(
            run_id=run_id,
            submission_ref=submission_ref,
            submission_sha256=submission_sha256,
            exit_code=exit_code,
            ok=True,
            fingerprint=fingerprint,
            error_kind="none",
            action_taken="submit",
            reason="submitted",
            stdout=stdout,
            stderr=stderr,
            stdout_tail_chars=stdout_tail_chars,
            stderr_tail_chars=stderr_tail_chars,
        ),
        run_state_update=build_submit_run_state_update(
            prior_state=prior_state,
            fingerprint=fingerprint,
            code_fingerprint=code_fingerprint,
            error_kind="none",
            action_taken="submit",
            reason="submitted",
            submission_ref=submission_ref,
            submit_ok=True,
        ),
    )


def build_submit_knowledge_payload(
    *,
    iteration: int | None,
    error_kind: str,
    reason: str,
    action_taken: str,
    fingerprint: str,
    details: str,
    normalize_detail: Callable[..., str],
) -> SubmitKnowledgePayload:
    normalized_detail = normalize_detail(details, max_chars=1200)
    return SubmitKnowledgePayload(
        iteration=iteration or 1,
        error_message=f"submit_error kind={error_kind} reason={reason} fingerprint={fingerprint}",
        fix_summary=f"submit_action={action_taken}; detail={normalized_detail}",
    )


def build_submit_result_payload(
    *,
    message: str,
    submission_ref: str,
    submitted_at_iso: str,
    iteration: int | None,
    outcome: object | None = None,
    skipped: bool = False,
    reason: str | None = None,
    duplicate_sources: list[str] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "message": message,
        "submission_path": submission_ref,
        "submitted_at": submitted_at_iso,
        "iteration": iteration,
    }
    if skipped:
        payload["skipped"] = True
    if reason:
        payload["reason"] = reason
    if duplicate_sources is not None:
        payload["duplicate_sources"] = list(duplicate_sources)
    if not skipped:
        payload["outcome"] = outcome
    return payload


def decide_submit_outcome_recording(
    *,
    outcome: object,
    submission_artifact_exists: bool,
) -> SubmitOutcomeRecordingDecision:
    if isinstance(outcome, dict) and outcome.get("score") is not None:
        message = (
            "[cyan]submission result[/cyan]: "
            f"status={outcome.get('status') or 'unknown'} score={float(outcome['score']):.6f}"
        )
    else:
        message = "[yellow]submission result[/yellow]: score not available yet; knowledge update skipped"
    ledger_outcome = dict(outcome) if isinstance(outcome, dict) and submission_artifact_exists else None
    return SubmitOutcomeRecordingDecision(message=message, ledger_outcome=ledger_outcome)
