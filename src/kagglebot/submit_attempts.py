from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


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
class SubmitAttemptStatePayloads:
    attempt_payload: dict[str, object]
    run_state_update: dict[str, object]


SubmitSuccessRecordPayloads = SubmitAttemptStatePayloads
SubmitAbortRecordPayloads = SubmitAttemptStatePayloads
SubmitSkipRecordPayloads = SubmitAttemptStatePayloads


@dataclass(frozen=True)
class SubmitAttemptRecorder:
    run_dir: Path
    save_run_state: Callable[[dict[str, object]], None]

    def append(self, payload: dict[str, object]) -> None:
        append_submit_attempt(run_dir=self.run_dir, payload=payload)

    def record_state(self, *, attempt_payload: dict[str, object], run_state_update: dict[str, object]) -> None:
        self.append(attempt_payload)
        self.save_run_state(run_state_update)

    def record_payloads(self, payloads: SubmitAttemptStatePayloads) -> None:
        self.record_state(
            attempt_payload=payloads.attempt_payload,
            run_state_update=payloads.run_state_update,
        )


def append_submit_attempt(*, run_dir: Path, payload: dict[str, object], now_iso: str | None = None) -> None:
    record = {
        "ts": now_iso or datetime.now(UTC).isoformat(),
        **payload,
    }
    attempts_path = run_dir / "submit_attempts.jsonl"
    attempts_path.parent.mkdir(parents=True, exist_ok=True)
    with attempts_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def load_submit_attempt_rows(run_dir: Path) -> list[dict[str, object]]:
    attempts_path = run_dir / "submit_attempts.jsonl"
    if not attempts_path.exists():
        return []
    try:
        lines = attempts_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows: list[dict[str, object]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def submit_attempt_sha_seen(*, run_dir: Path, submission_sha: str) -> bool:
    normalized_sha = str(submission_sha or "").strip()
    if not normalized_sha:
        return False
    for row in load_submit_attempt_rows(run_dir):
        if str(row.get("sub_sha256") or "").strip() == normalized_sha:
            return True
    return False


def has_submit_attempt_records(run_dir: Path) -> bool:
    return bool(load_submit_attempt_rows(run_dir))


def has_successful_submit_attempt(run_dir: Path) -> bool:
    return any(bool(row.get("ok")) for row in load_submit_attempt_rows(run_dir))


def count_successful_submit_attempts(run_dir: Path) -> int:
    count = 0
    for row in load_submit_attempt_rows(run_dir):
        if not bool(row.get("ok")):
            continue
        action_taken = str(row.get("action_taken") or "").strip().lower()
        if action_taken and action_taken != "submit":
            continue
        count += 1
    return count


def load_submit_fingerprints(run_dir: Path) -> list[str]:
    fingerprints: list[str] = []
    for row in load_submit_attempt_rows(run_dir):
        fingerprint = str(row.get("fingerprint") or "").strip()
        if not fingerprint:
            continue
        fingerprints.append(fingerprint)
    return fingerprints


def build_seen_submit_fingerprint_set(
    *,
    attempt_fingerprints: list[str],
    run_state: dict[str, object],
) -> set[str]:
    fingerprints = {str(fingerprint).strip() for fingerprint in attempt_fingerprints if str(fingerprint).strip()}
    state_fingerprint = str(run_state.get("last_submit_fingerprint") or run_state.get("last_fingerprint") or "").strip()
    if state_fingerprint:
        fingerprints.add(state_fingerprint)
    return fingerprints


def load_latest_submit_attempt(run_dir: Path) -> dict[str, object]:
    rows = load_submit_attempt_rows(run_dir)
    if not rows:
        return {}
    return rows[-1]


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
) -> SubmitAttemptStatePayloads:
    return SubmitAttemptStatePayloads(
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


def build_submit_abort_record_payloads(
    *,
    run_id: str,
    submission_ref: str,
    submission_sha256: str | None,
    exit_code: int | None,
    fingerprint: str,
    code_fingerprint: str,
    error_kind: str,
    reason: str,
    stdout: str,
    stderr: str,
    prior_state: dict[str, object],
    prior_submit_ok: bool,
    stdout_tail_chars: int,
    stderr_tail_chars: int,
) -> SubmitAttemptStatePayloads:
    return SubmitAttemptStatePayloads(
        attempt_payload=build_submit_attempt_payload(
            run_id=run_id,
            submission_ref=submission_ref,
            submission_sha256=submission_sha256,
            exit_code=exit_code,
            ok=False,
            fingerprint=fingerprint,
            code_fingerprint=code_fingerprint,
            error_kind=error_kind,
            action_taken="abort",
            reason=reason,
            stdout=stdout,
            stderr=stderr,
            stdout_tail_chars=stdout_tail_chars,
            stderr_tail_chars=stderr_tail_chars,
        ),
        run_state_update=build_submit_run_state_update(
            prior_state=prior_state,
            fingerprint=fingerprint,
            code_fingerprint=code_fingerprint,
            error_kind=error_kind,
            action_taken="abort",
            reason=reason,
            submission_ref=submission_ref,
            submit_ok=prior_submit_ok,
        ),
    )


def build_submit_skip_attempt_payload(
    *,
    run_id: str,
    submission_ref: str,
    submission_sha256: str | None,
    fingerprint: str,
    error_kind: str,
    reason: str,
    stdout_tail_chars: int,
    stderr_tail_chars: int,
    duplicate_sources: list[str] | None = None,
) -> dict[str, object]:
    extra = {"duplicate_sources": list(duplicate_sources)} if duplicate_sources is not None else None
    return build_submit_attempt_payload(
        run_id=run_id,
        submission_ref=submission_ref,
        submission_sha256=submission_sha256,
        exit_code=None,
        ok=False,
        fingerprint=fingerprint,
        error_kind=error_kind,
        action_taken="skip",
        reason=reason,
        stdout="",
        stderr="",
        stdout_tail_chars=stdout_tail_chars,
        stderr_tail_chars=stderr_tail_chars,
        extra=extra,
    )


def build_same_submission_path_skip_attempt_payload(
    *,
    run_id: str,
    submission_ref: str,
    submission_sha256: str | None,
    fingerprint: str,
    reason: str,
    stdout_tail_chars: int,
    stderr_tail_chars: int,
) -> dict[str, object]:
    return build_submit_skip_attempt_payload(
        run_id=run_id,
        submission_ref=submission_ref,
        submission_sha256=submission_sha256,
        fingerprint=fingerprint,
        error_kind="unknown",
        reason=reason,
        stdout_tail_chars=stdout_tail_chars,
        stderr_tail_chars=stderr_tail_chars,
    )


def build_submit_skip_record_payloads(
    *,
    run_id: str,
    submission_ref: str,
    submission_sha256: str | None,
    fingerprint: str,
    code_fingerprint: str,
    error_kind: str,
    reason: str,
    prior_state: dict[str, object],
    stdout_tail_chars: int,
    stderr_tail_chars: int,
    duplicate_sources: list[str] | None = None,
) -> SubmitAttemptStatePayloads:
    return SubmitAttemptStatePayloads(
        attempt_payload=build_submit_skip_attempt_payload(
            run_id=run_id,
            submission_ref=submission_ref,
            submission_sha256=submission_sha256,
            fingerprint=fingerprint,
            error_kind=error_kind,
            reason=reason,
            stdout_tail_chars=stdout_tail_chars,
            stderr_tail_chars=stderr_tail_chars,
            duplicate_sources=duplicate_sources,
        ),
        run_state_update=build_submit_run_state_update(
            prior_state=prior_state,
            fingerprint=fingerprint,
            code_fingerprint=code_fingerprint,
            error_kind=error_kind,
            action_taken="skip",
            reason=reason,
            submission_ref=submission_ref,
            submission_sha256=submission_sha256,
        ),
    )


def build_submit_retry_attempt_payload(
    *,
    run_id: str,
    submission_ref: str,
    submission_sha256: str | None,
    exit_code: int | None,
    fingerprint: str,
    reason: str,
    stdout: str,
    stderr: str,
    stdout_tail_chars: int,
    stderr_tail_chars: int,
) -> dict[str, object]:
    return build_submit_attempt_payload(
        run_id=run_id,
        submission_ref=submission_ref,
        submission_sha256=submission_sha256,
        exit_code=exit_code,
        ok=False,
        fingerprint=fingerprint,
        error_kind="transient",
        action_taken="retry",
        reason=reason,
        stdout=stdout,
        stderr=stderr,
        stdout_tail_chars=stdout_tail_chars,
        stderr_tail_chars=stderr_tail_chars,
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


def format_submit_retry_knowledge_details(*, attempt: int, wait_seconds: float) -> str:
    return f"attempt={int(attempt)}; wait={float(wait_seconds):.1f}s"


def record_submit_reason_knowledge(
    *,
    knowledge_paths: object,
    slug: str,
    run_id: str,
    problem_types: list[str],
    submission_path: Path,
    error_kind: str,
    reason: str,
    action_taken: str,
    fingerprint: str,
    details: str,
    infer_iteration: Callable[[Path], int | None],
    normalize_detail: Callable[..., str],
    record_error_fix_insight: Callable[..., object],
) -> bool:
    payload = build_submit_knowledge_payload(
        iteration=infer_iteration(submission_path),
        error_kind=error_kind,
        reason=reason,
        action_taken=action_taken,
        fingerprint=fingerprint,
        details=details,
        normalize_detail=normalize_detail,
    )
    try:
        record_error_fix_insight(
            knowledge_paths=knowledge_paths,
            slug=slug,
            run_id=run_id,
            iteration=payload.iteration,
            problem_types=problem_types,
            error_message=payload.error_message,
            fix_summary=payload.fix_summary,
            resolved=False,
            outcome_bucket="unknown",
            submission_score=None,
        )
    except Exception:  # noqa: BLE001
        return False
    return True


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


def build_successful_submit_result_payload(
    *,
    message: str,
    submission_ref: str,
    submitted_at: datetime,
    submission_path: Path,
    outcome: object | None,
    infer_iteration: Callable[[Path], int | None],
) -> dict[str, object]:
    return build_submit_result_payload(
        message=message,
        submission_ref=submission_ref,
        submitted_at_iso=submitted_at.isoformat(),
        iteration=infer_iteration(submission_path),
        outcome=outcome,
    )


def build_duplicate_submit_skip_result_payload(
    *,
    message: str,
    submission_ref: str,
    submitted_at: datetime,
    submission_path: Path,
    reason: str,
    duplicate_sources: list[str],
    infer_iteration: Callable[[Path], int | None],
) -> dict[str, object]:
    return build_submit_result_payload(
        message=message,
        submission_ref=submission_ref,
        submitted_at_iso=submitted_at.isoformat(),
        iteration=infer_iteration(submission_path),
        skipped=True,
        reason=reason,
        duplicate_sources=duplicate_sources,
    )


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


def record_submit_outcome_if_available(
    *,
    decision: SubmitOutcomeRecordingDecision,
    submission_path: Path | None,
    record_outcome: Callable[[Path, dict[str, object]], object],
) -> bool:
    if decision.ledger_outcome is None or submission_path is None:
        return False
    record_outcome(submission_path, decision.ledger_outcome)
    return True
