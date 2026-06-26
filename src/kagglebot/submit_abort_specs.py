from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class SubmitAbortSpec:
    fingerprint: str
    error_kind: str
    reason: str
    message: str
    stdout_tail: str
    stderr_tail: str
    exit_code: int | None


def build_submit_abort_spec_kwargs(spec: SubmitAbortSpec) -> dict[str, object]:
    return {
        "fingerprint": spec.fingerprint,
        "error_kind": spec.error_kind,
        "reason": spec.reason,
        "message": spec.message,
        "stdout_tail": spec.stdout_tail,
        "stderr_tail": spec.stderr_tail,
        "exit_code": spec.exit_code,
    }


def build_kaggle_credentials_missing_abort_spec(
    *,
    stdout: str,
    stderr: str,
    output: str,
    exit_code: int | None,
    compute_error_fingerprint: Callable[[str, str], str],
) -> SubmitAbortSpec:
    stderr_tail = stderr or output
    return SubmitAbortSpec(
        fingerprint=compute_error_fingerprint(stdout, stderr_tail),
        error_kind="permanent",
        reason="kaggle_credentials_missing",
        message="Kaggle credentials not configured. Set ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY.",
        stdout_tail=stdout,
        stderr_tail=stderr_tail,
        exit_code=exit_code,
    )


def build_rules_not_accepted_abort_spec(
    *,
    exit_code: int | None,
    compute_error_fingerprint: Callable[[str, str], str],
) -> SubmitAbortSpec:
    return SubmitAbortSpec(
        fingerprint=compute_error_fingerprint("", "rules_not_accepted"),
        error_kind="permanent",
        reason="rules_not_accepted",
        message="Competition rules are not accepted; aborting submit stage for this run.",
        stdout_tail="",
        stderr_tail="rules_not_accepted",
        exit_code=exit_code,
    )


def build_local_submission_guardrail_abort_spec(
    *,
    error: object,
    exit_code: int | None,
    compute_error_fingerprint: Callable[[str, str], str],
) -> SubmitAbortSpec:
    stderr_tail = str(error)
    return SubmitAbortSpec(
        fingerprint=compute_error_fingerprint("", stderr_tail),
        error_kind="permanent",
        reason="local_submission_guardrail",
        message=f"Local submission guardrail blocked submit: {stderr_tail}",
        stdout_tail="",
        stderr_tail=stderr_tail,
        exit_code=exit_code,
    )


def resolve_local_submission_guardrail_abort_spec(
    *,
    error: object,
    compute_error_fingerprint: Callable[[str, str], str],
    default_exit_code: int = 1,
) -> SubmitAbortSpec:
    return build_local_submission_guardrail_abort_spec(
        error=error,
        exit_code=getattr(error, "exit_code", default_exit_code),
        compute_error_fingerprint=compute_error_fingerprint,
    )


def resolve_kaggle_cli_submit_abort_spec(
    *,
    error: BaseException,
    is_missing_credentials_error: Callable[[BaseException], bool],
    compute_error_fingerprint: Callable[[str, str], str],
) -> SubmitAbortSpec | None:
    if not is_missing_credentials_error(error):
        return None
    return build_kaggle_credentials_missing_abort_spec(
        stdout=str(getattr(error, "stdout", "") or ""),
        stderr=str(getattr(error, "stderr", "") or ""),
        output=str(getattr(error, "output", "") or ""),
        exit_code=getattr(error, "exit_code", None),
        compute_error_fingerprint=compute_error_fingerprint,
    )


def build_local_submission_validation_abort_spec(
    *,
    error: object,
    exit_code: int | None,
    compute_error_fingerprint: Callable[[str, str], str],
) -> SubmitAbortSpec:
    stderr_tail = str(error)
    return SubmitAbortSpec(
        fingerprint=compute_error_fingerprint("", stderr_tail),
        error_kind="validation",
        reason="local_submission_validation_failed",
        message="Local submission validation failed; Kaggle CLI submit is skipped.",
        stdout_tail="",
        stderr_tail=stderr_tail,
        exit_code=exit_code,
    )


def build_submission_polling_error_abort_spec(
    *,
    error: object,
    detail: object,
    normalize_detail: Callable[[str], str],
    compute_error_fingerprint: Callable[[str, str], str],
) -> SubmitAbortSpec:
    normalized_detail = normalize_detail(str(detail or error))
    stderr_tail = normalized_detail or str(error)
    return SubmitAbortSpec(
        fingerprint=compute_error_fingerprint("", stderr_tail),
        error_kind="transient",
        reason="submission_polling_error",
        message="Submission outcome polling failed; aborting submit stage for this run.",
        stdout_tail="",
        stderr_tail=stderr_tail,
        exit_code=None,
    )


def build_submission_outcome_abort_spec(
    *,
    decision: object,
    compute_error_fingerprint: Callable[[str, str], str],
) -> SubmitAbortSpec:
    detail = str(getattr(decision, "detail", "") or "")
    return SubmitAbortSpec(
        fingerprint=compute_error_fingerprint("", detail),
        error_kind=str(getattr(decision, "error_kind", "") or ""),
        reason=str(getattr(decision, "reason", "") or ""),
        message=str(getattr(decision, "message", "") or ""),
        stdout_tail="",
        stderr_tail=detail,
        exit_code=None,
    )


def build_submit_stage_error_action_abort_spec(
    *,
    action: object,
    fingerprint: str,
    stdout: str,
    stderr: str,
    exit_code: int | None,
) -> SubmitAbortSpec:
    return SubmitAbortSpec(
        fingerprint=fingerprint,
        error_kind=str(getattr(action, "error_kind", "") or ""),
        reason=str(getattr(action, "reason", "") or ""),
        message=str(getattr(action, "abort_message", "") or ""),
        stdout_tail=stdout,
        stderr_tail=stderr,
        exit_code=exit_code,
    )
