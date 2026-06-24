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
