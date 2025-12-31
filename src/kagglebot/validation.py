from __future__ import annotations

import pandas as pd

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

    # 3) Basic id column null check.
    id_col = sample.columns[0]
    if sub[id_col].isna().any():
        raise ValueError(f"Submission contains missing values in id column '{id_col}'.")

    # 4) All-NaN target columns are not allowed (guard against bad output).
    for c in sample.columns[1:]:
        if sub[c].isna().all():
            raise ValueError(f"All values are NaN for target column '{c}'.")


def ensure_not_duplicate_submission(ledger: SubmissionLedger, submission_path: str) -> None:
    if ledger.is_duplicate(submission_path):
        raise ValueError("Duplicate submission detected (hash already recorded).")
