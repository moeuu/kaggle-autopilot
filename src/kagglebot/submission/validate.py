from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kagglebot.exceptions import SubmissionValidationError


def validate_submission(submission_path: Path, sample_path: Path) -> None:
    """Strict local validation for Kaggle submissions."""
    if not sample_path.exists():
        raise SubmissionValidationError(f"Sample submission file not found: {sample_path}")
    if not submission_path.exists():
        raise SubmissionValidationError(f"Submission file not found: {submission_path}")

    sample = pd.read_csv(sample_path)
    submission = pd.read_csv(submission_path)

    expected_columns = list(sample.columns)
    actual_columns = list(submission.columns)
    if expected_columns != actual_columns:
        raise SubmissionValidationError(
            "Submission columns do not match sample_submission.csv.\n"
            f"Expected: {expected_columns}\n"
            f"Got:      {actual_columns}"
        )
    if len(sample) != len(submission):
        raise SubmissionValidationError(
            "Submission row count does not match sample_submission.csv.\n"
            f"Expected rows: {len(sample)}\n"
            f"Got rows:      {len(submission)}"
        )
    if not expected_columns:
        raise SubmissionValidationError("Sample submission has no columns.")

    id_col = expected_columns[0]
    sample_ids = sample[id_col]
    sub_ids = submission[id_col]
    if sample_ids.isna().any():
        raise SubmissionValidationError(f"Sample submission id column '{id_col}' contains NaN.")
    if sub_ids.isna().any():
        raise SubmissionValidationError(f"Submission id column '{id_col}' contains NaN.")
    if sample_ids.duplicated().any():
        raise SubmissionValidationError(f"Sample submission id column '{id_col}' contains duplicates.")
    if sub_ids.duplicated().any():
        raise SubmissionValidationError(f"Submission id column '{id_col}' contains duplicates.")
    if not sub_ids.reset_index(drop=True).equals(sample_ids.reset_index(drop=True)):
        raise SubmissionValidationError(
            f"Submission id column '{id_col}' does not match sample_submission.csv order/values."
        )

    pred_cols = [col for col in expected_columns if col != id_col]
    if not pred_cols:
        raise SubmissionValidationError("No prediction columns found in submission.")
    for col in pred_cols:
        numeric = pd.to_numeric(submission[col], errors="coerce")
        if numeric.isna().any():
            raise SubmissionValidationError(f"Prediction column '{col}' contains NaN or non-numeric values.")
        values = numeric.to_numpy(dtype=float, copy=False)
        if np.isinf(values).any():
            raise SubmissionValidationError(f"Prediction column '{col}' contains inf/-inf values.")
