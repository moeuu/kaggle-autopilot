from __future__ import annotations


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
