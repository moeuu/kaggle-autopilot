from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from kagglebot.exceptions import DuplicateSubmissionError, SubmissionRateLimitError
from kagglebot.history import SubmissionLedger


def validate_submission(sample_path: str, submission_path: str) -> None:
    sample = pd.read_csv(sample_path)
    sub = pd.read_csv(submission_path)

    # 1) Columns must match (including order).
    if list(sample.columns) != list(sub.columns):
        raise ValueError(
            "Submission columns do not match sample_submission.csv.\n"
            f"Expected: {list(sample.columns)}\n"
            f"Got:      {list(sub.columns)}"
        )

    # 2) Row count must match.
    if len(sample) != len(sub):
        raise ValueError(
            "Submission row count does not match sample_submission.csv.\n"
            f"Expected rows: {len(sample)}\n"
            f"Got rows:      {len(sub)}"
        )

    # 3) Basic id column checks.
    id_col = sample.columns[0]
    if sub[id_col].isna().any():
        raise ValueError(f"Submission contains missing values in id column '{id_col}'.")
    if sub[id_col].duplicated().any():
        raise ValueError(f"Submission contains duplicate values in id column '{id_col}'.")

    sample_ids = sample[id_col]
    sub_ids = sub[id_col]
    if sample_ids.isna().any():
        raise ValueError(f"Sample submission contains missing values in id column '{id_col}'.")
    if sample_ids.duplicated().any():
        raise ValueError(f"Sample submission contains duplicate values in id column '{id_col}'.")

    if set(sub_ids) != set(sample_ids):
        missing = sorted(set(sample_ids) - set(sub_ids))
        extra = sorted(set(sub_ids) - set(sample_ids))
        missing_preview = missing[:5]
        extra_preview = extra[:5]
        raise ValueError(
            "Submission id values do not match sample_submission.csv.\n"
            f"Missing ids (first 5): {missing_preview}\n"
            f"Extra ids (first 5):   {extra_preview}"
        )

    # 4) All-NaN target columns are not allowed (guard against bad output).
    for c in sample.columns[1:]:
        if sub[c].isna().all():
            raise ValueError(f"All values are NaN for target column '{c}'.")


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
    min_hours_between: float = 1.0,
) -> None:
    now = datetime.now(UTC)
    last_ts = ledger.last_submission_time()
    recent = ledger.recent_submission_count(hours=24)

    if recent >= max_submissions_per_day:
        raise SubmissionRateLimitError("Submission rate limit exceeded (max per day).")
    if last_ts is not None:
        elapsed = (now - last_ts).total_seconds() / 3600.0
        if elapsed < min_hours_between:
            raise SubmissionRateLimitError("Submission rate limit exceeded (cooldown).")
