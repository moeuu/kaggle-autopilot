from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from kagglebot import submit_stage_messages as _submit_stage_messages
from kagglebot.submission.outcome_service import SubmissionOutcomeService


@dataclass(frozen=True)
class SubmitOutcomeAbortDecision:
    should_abort: bool
    error_kind: str = ""
    reason: str = ""
    message: str = ""
    detail: str = ""


@dataclass(frozen=True)
class SubmissionOutcomePostPollDecision:
    outcome: object
    abort_decision: SubmitOutcomeAbortDecision


FAILED_SUBMISSION_OUTCOME_STATUSES = {"error", "failed", "cancelled", "canceled"}
SCORELESS_COMPLETE_SUBMISSION_OUTCOME_STATUSES = {"complete", "completed"}


def normalize_submission_outcome_status(value: object) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return "unknown"
    if "." in raw:
        prefix, _, suffix = raw.rpartition(".")
        if suffix and "status" in prefix:
            return suffix.strip()
    return raw


def decide_submission_outcome_abort(
    *,
    outcome_status: str,
    outcome_score: object,
    deliverable_mode: str,
    raw_detail: str,
) -> SubmitOutcomeAbortDecision:
    if outcome_status in FAILED_SUBMISSION_OUTCOME_STATUSES:
        return SubmitOutcomeAbortDecision(
            should_abort=True,
            error_kind="validation",
            reason=f"submission_poll_status_{outcome_status}",
            message=(
                f"Submission finished with error status '{outcome_status}' during polling; "
                "aborting submit stage for this run."
            ),
            detail=raw_detail or outcome_status,
        )

    if (
        outcome_status in SCORELESS_COMPLETE_SUBMISSION_OUTCOME_STATUSES
        and outcome_score is None
        and str(deliverable_mode or "").strip().lower() == "leaderboard"
    ):
        detail = raw_detail
        if not detail:
            detail = (
                "Kaggle submission completed without a public/private score. "
                "For leaderboard submissions this usually indicates a scoring error, "
                "such as an invalid notebook-generated submission file."
            )
        elif "submission file" not in detail.lower() and "scoring error" not in detail.lower():
            detail = (
                detail + "\nKaggle scoring error inferred: leaderboard submission file completed "
                "without a public/private score."
            )
        return SubmitOutcomeAbortDecision(
            should_abort=True,
            error_kind="validation",
            reason=f"submission_poll_status_{outcome_status}_no_score",
            message=(
                f"Submission finished with status '{outcome_status}' but no score; "
                "treating as scoring failure for this leaderboard run."
            ),
            detail=detail,
        )

    return SubmitOutcomeAbortDecision(should_abort=False)


def build_submission_outcome_error_detail(
    *,
    slug: str,
    message: str,
    submitted_at: datetime,
    outcome: dict[str, object],
    fetch_submission_rows: Callable[[str], list[dict[str, str]]],
    normalize_detail: Callable[[str], str],
) -> str:
    row: dict[str, object] | None = None
    raw_payload = outcome.get("raw")
    if isinstance(raw_payload, dict):
        row = dict(raw_payload)
    try:
        rows = fetch_submission_rows(slug)
        selector = SubmissionOutcomeService(fetch_rows=lambda current_slug: rows)
        matched = selector._select_submission_row(rows=rows, message=message, submitted_at=submitted_at)  # noqa: SLF001
        if isinstance(matched, dict):
            row = dict(matched)
    except Exception:  # noqa: BLE001
        pass

    details: list[str] = []
    if row:
        row_message = _submit_stage_messages.extract_submission_row_message(row)
        if row_message:
            details.append(f"Kaggle reported: {row_message}")
        details.append(f"Kaggle submission row: {json.dumps(row, ensure_ascii=True)}")
    elif raw_payload is not None:
        details.append(f"Kaggle submission raw payload: {json.dumps(raw_payload, ensure_ascii=True)}")
    else:
        details.append(f"Kaggle submission status: {outcome.get('status') or 'unknown'}")
    return normalize_detail("\n".join(details))


def evaluate_submission_outcome_after_poll(
    *,
    slug: str,
    message: str,
    submitted_at: datetime,
    outcome: object,
    deliverable_mode: str,
    fetch_submission_rows: Callable[[str], list[dict[str, str]]],
    normalize_detail: Callable[[str], str],
) -> SubmissionOutcomePostPollDecision:
    if not isinstance(outcome, dict):
        return SubmissionOutcomePostPollDecision(
            outcome=outcome,
            abort_decision=SubmitOutcomeAbortDecision(should_abort=False),
        )

    normalized_outcome = dict(outcome)
    outcome_status = normalize_submission_outcome_status(normalized_outcome.get("status"))
    normalized_outcome["status"] = outcome_status
    raw_detail = ""
    if outcome_status in FAILED_SUBMISSION_OUTCOME_STATUSES or (
        outcome_status in SCORELESS_COMPLETE_SUBMISSION_OUTCOME_STATUSES
        and normalized_outcome.get("score") is None
        and str(deliverable_mode or "").strip().lower() == "leaderboard"
    ):
        raw_detail = build_submission_outcome_error_detail(
            slug=slug,
            message=message,
            submitted_at=submitted_at,
            outcome=normalized_outcome,
            fetch_submission_rows=fetch_submission_rows,
            normalize_detail=normalize_detail,
        )
    return SubmissionOutcomePostPollDecision(
        outcome=normalized_outcome,
        abort_decision=decide_submission_outcome_abort(
            outcome_status=outcome_status,
            outcome_score=normalized_outcome.get("score"),
            deliverable_mode=deliverable_mode,
            raw_detail=raw_detail,
        ),
    )
