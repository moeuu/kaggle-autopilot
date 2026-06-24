from __future__ import annotations

from dataclasses import dataclass

SUBMIT_FAILURE_REPAIR_TARGET_SUBMISSION_ARTIFACT = "submission_artifact"
SUBMIT_FAILURE_REPAIR_TARGET_SUBMIT_MODE = "submit_mode_or_kernel"
SUBMIT_FAILURE_REPAIR_TARGET_PLATFORM = "platform_or_polling"
SUBMIT_FAILURE_REPAIR_TARGET_MANUAL = "manual_intervention"
SUBMIT_FAILURE_REPAIR_TARGET_UNKNOWN = "unknown"

SUBMIT_FILE_ERROR_MARKERS = (
    "submission file",
    "submission.csv",
    "columns mismatch",
    "row count mismatch",
    "must have ",
    "id column missing",
    "missing a header row",
    "header does not resemble",
    "id values appear to require",
    "submission payload mismatch",
    "submission must have",
    "prediction column contains nan",
    "file format mismatch",
    "file must be named",
)

NOTEBOOK_FALLBACK_HINTS = (
    "only accepts submissions from notebooks",
    "must be made through notebooks",
    "code competition submissions require both the output file name and the version label",
    "output file name and version label",
    "output file name and the version label",
    "kernel must be specified as <owner>/<notebook>",
    "kernel must be specified",
)


@dataclass(frozen=True)
class SubmitFailureRepairDecision:
    repair_target: str
    repairable: bool
    manual_next_step: str = ""


def should_use_notebook_submit_fallback(*, reason: str, stdout: str, stderr: str) -> bool:
    """Return True only when submit errors clearly indicate notebook-only submission."""
    normalized_reason = str(reason or "").strip().lower()
    if normalized_reason == "notebook_only_submission_required":
        return True
    if normalized_reason not in {"bad_request", "unclassified_submit_error", "unknown"}:
        return False
    return _detail_has_notebook_fallback_hint(stdout=stdout, stderr=stderr)


def should_retry_ambiguous_notebook_submit_error(*, reason: str, stdout: str, stderr: str) -> bool:
    if str(reason or "").strip().lower() != "ambiguous_notebook_bad_request":
        return False
    return _detail_has_notebook_fallback_hint(stdout=stdout, stderr=stderr)


def normalize_loaded_submit_failure_context(payload: dict[str, object]) -> dict[str, object]:
    """Backfill manual blocker classification for contexts written by older runs."""
    reason = str(payload.get("reason") or "").strip().lower()
    detail = "\n".join(
        str(payload.get(key) or "") for key in ("stdout_tail", "stderr_tail", "summary", "message") if payload.get(key)
    ).lower()
    decision = _manual_blocker_decision(reason=reason, detail=detail)
    if decision is not None:
        payload["repair_target"] = decision.repair_target
        payload["repairable"] = decision.repairable
        payload["manual_next_step"] = decision.manual_next_step
    return payload


def submit_failure_manual_next_step(*, reason: str, detail: str) -> str:
    normalized_reason = str(reason or "").strip().lower()
    lowered_detail = str(detail or "").strip().lower()
    if normalized_reason == "kaggle_credentials_missing":
        return "Configure ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY before retrying submit."
    if normalized_reason == "rules_not_accepted":
        return "Accept the competition rules in the Kaggle browser UI, then retry submit."
    if normalized_reason == "local_submission_guardrail":
        if "duplicate" in lowered_detail:
            return "Change the submission predictions before retrying, or use --force-submit only when intentional."
        if "cooldown" in lowered_detail or "rate limit" in lowered_detail:
            return "Wait for the submission cooldown/rate-limit window to expire before retrying."
        return "Resolve the local submission guardrail before retrying submit."
    return "Resolve the manual submit blocker, then retry submit."


def submit_error_targets_submit_mode(*, reason: str, detail: str) -> bool:
    normalized_reason = str(reason or "").strip().lower()
    lowered_detail = str(detail or "").strip().lower()
    if normalized_reason == "notebook_only_submission_required":
        return True
    mode_markers = (
        "notebook",
        "kernel",
        "internet access",
        "enable_internet",
        "competition does not allow internet",
        "submission not allowed",
        "output file name and the version label",
    )
    return any(marker in lowered_detail for marker in mode_markers)


def classify_submit_failure_repair(
    *,
    reason: object,
    error_kind: object,
    detail: str,
) -> SubmitFailureRepairDecision:
    normalized_reason = str(reason or "").strip().lower()
    normalized_kind = str(error_kind or "").strip().lower()
    normalized_detail = str(detail or "").strip()
    manual_decision = _manual_blocker_decision(reason=normalized_reason, detail=normalized_detail)
    if manual_decision is not None:
        return manual_decision
    if normalized_reason in {"kaggle_credentials_missing", "rules_not_accepted", "local_submission_guardrail"}:
        return SubmitFailureRepairDecision(
            repair_target=SUBMIT_FAILURE_REPAIR_TARGET_MANUAL,
            repairable=False,
            manual_next_step=submit_failure_manual_next_step(reason=normalized_reason, detail=normalized_detail),
        )
    if submit_error_requires_file_fix(
        reason=normalized_reason,
        error_kind=normalized_kind,
        detail=normalized_detail,
    ):
        return SubmitFailureRepairDecision(SUBMIT_FAILURE_REPAIR_TARGET_SUBMISSION_ARTIFACT, True)
    if submit_error_targets_submit_mode(reason=normalized_reason, detail=normalized_detail):
        return SubmitFailureRepairDecision(SUBMIT_FAILURE_REPAIR_TARGET_SUBMIT_MODE, True)
    if normalized_reason in {
        "submission_polling_error",
        "submission_polling_timeout",
        "submission_polling_invalid_payload",
    } or normalized_reason.startswith("submission_poll_status_"):
        return SubmitFailureRepairDecision(SUBMIT_FAILURE_REPAIR_TARGET_PLATFORM, True)
    if normalized_kind in {"transient", "unknown"}:
        return SubmitFailureRepairDecision(SUBMIT_FAILURE_REPAIR_TARGET_UNKNOWN, True)
    return SubmitFailureRepairDecision(SUBMIT_FAILURE_REPAIR_TARGET_UNKNOWN, True)


def submit_error_text_indicates_file_issue(text: str) -> bool:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return False
    return any(marker in lowered for marker in SUBMIT_FILE_ERROR_MARKERS)


def submit_error_requires_file_fix(*, reason: object, error_kind: object, detail: str) -> bool:
    normalized_reason = str(reason or "").strip().lower()
    normalized_kind = str(error_kind or "").strip().lower()
    if normalized_reason == "local_submission_validation_failed":
        return True
    if normalized_reason.startswith("submission_poll_status_") and submit_error_text_indicates_file_issue(detail):
        return True
    if normalized_kind == "validation" and submit_error_text_indicates_file_issue(detail):
        return True
    return False


def _detail_has_notebook_fallback_hint(*, stdout: str, stderr: str) -> bool:
    detail = f"{stdout}\n{stderr}".lower()
    return any(hint in detail for hint in NOTEBOOK_FALLBACK_HINTS)


def _manual_blocker_decision(*, reason: str, detail: str) -> SubmitFailureRepairDecision | None:
    normalized_reason = str(reason or "").strip().lower()
    lowered_detail = str(detail or "").strip().lower()
    if normalized_reason == "ambiguous_notebook_bad_request":
        return SubmitFailureRepairDecision(
            repair_target=SUBMIT_FAILURE_REPAIR_TARGET_MANUAL,
            repairable=False,
            manual_next_step=(
                "Kaggle returned a generic submit-notebook 400 without a clear notebook-only signal; "
                "do not auto-repair or rerun the kernel for this submit error."
            ),
        )
    if normalized_reason == "submission_limit" or any(
        marker in lowered_detail
        for marker in (
            "submission limit",
            "maximum number of submissions",
            "max submissions",
        )
    ):
        return SubmitFailureRepairDecision(
            repair_target=SUBMIT_FAILURE_REPAIR_TARGET_MANUAL,
            repairable=False,
            manual_next_step="Wait for the Kaggle submission limit window to expire before retrying submit.",
        )
    return None
