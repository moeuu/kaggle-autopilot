from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

from kagglebot.exceptions import DuplicateSubmissionError, SubmissionRateLimitError
from kagglebot.history import SubmissionLedger
from kagglebot.submission.validate import validate_submission as _validate_submission


def validate_submission(sample_path: str, submission_path: str, *, data_dir: str | Path | None = None) -> None:
    """Backward-compatible wrapper using the canonical strict validator.

    This module historically exposed validate_submission(sample, submission).
    The canonical implementation uses validate_submission(submission, sample),
    so keep the old argument order here and delegate to avoid divergent checks.
    """
    _validate_submission(submission_path, sample_path, data_dir=data_dir)


def ensure_not_duplicate_submission(
    ledger: SubmissionLedger,
    *,
    slug: str,
    message: str,
    submission_path: str,
) -> None:
    if ledger.is_duplicate(slug=slug, message=message, submission_path=Path(submission_path)):
        raise DuplicateSubmissionError("Duplicate submission detected (hash already recorded).")


def ensure_submission_rate_limit(
    ledger: SubmissionLedger,
    *,
    max_submissions_per_day: int = 5,
    min_hours_between: float = 5.0 / 60.0,
) -> None:
    max_submissions_per_day = _rate_limit_int_env(
        "KAGGLEBOT_SUBMISSION_MAX_PER_DAY",
        default=max_submissions_per_day,
        min_value=1,
    )
    min_hours_between = _rate_limit_float_env(
        "KAGGLEBOT_SUBMISSION_MIN_HOURS_BETWEEN",
        default=min_hours_between,
        min_value=0.0,
    )
    now = datetime.now(UTC)
    last_ts = ledger.last_submission_time()
    recent = ledger.recent_submission_count(hours=24)

    if recent >= max_submissions_per_day:
        raise SubmissionRateLimitError("Submission rate limit exceeded (max per day).")
    if last_ts is not None:
        elapsed = (now - last_ts).total_seconds() / 3600.0
        if elapsed < min_hours_between:
            raise SubmissionRateLimitError("Submission rate limit exceeded (cooldown).")


def _rate_limit_int_env(name: str, *, default: int, min_value: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(min_value, value)


def _rate_limit_float_env(name: str, *, default: float, min_value: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(min_value, value)
